"""Extract Co-DINO expert activation, routing and communication visual data."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.data import DATASETS, PoseidonDataset
from model.expert_diagnostics import ExpertDiagnosticsCollector, representative_moe_layers
from train.evaluate import autocast_context, load_forecaster
from train.runtime import prepare_device


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-path", "--data_path")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--split", choices=("test", "val"), default="test")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--first-target-time", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--amp", choices=("auto", "none", "bf16", "fp16"), default="auto")
    p.add_argument("--output-dir")
    p.add_argument("--no-heatmap", action="store_true")
    return p


def cross_dataset_specialization(layers):
    datasets = list(layers)
    modules = sorted({name for values in layers.values() for name in values})
    result = {}
    for name in modules:
        available = [dataset for dataset in datasets if name in layers[dataset]]
        if len(available) < 2:
            continue
        routed = [[row for row in layers[dataset][name]["experts"]
                   if row["kind"] == "routed"] for dataset in available]
        experts = []
        for index in range(len(routed[0])):
            values = torch.tensor([rows[index]["mean_router_weight"] for rows in routed],
                                  dtype=torch.float64)
            distribution = values / values.sum().clamp_min(1e-24)
            entropy = -(distribution * distribution.clamp_min(1e-24).log()).sum().item()
            preferred = int(values.argmax().item())
            experts.append({
                "expert": index,
                "mean_router_weight_by_dataset": dict(zip(available, values.tolist())),
                "preferred_dataset": available[preferred],
                "dataset_selectivity_one_minus_normalized_entropy":
                    1.0 - entropy / math.log(len(available)),
            })
        result[name] = experts
    return result


def write_csvs(output, layers, specialization):
    module_names = {name for modules in layers.values() for name in modules}
    role_by_module = {module: role for role, module in
                      representative_moe_layers(module_names).items()}
    fields = ["dataset", "module", "layer_role", "kind", "expert", "tokens",
              "activation_count", "activation_fraction", "mean_router_weight",
              "selected_weight_fraction", "mean_selected_weight", "mean_hidden_l2",
              "mean_output_l2", "mean_signed_contribution_per_token",
              "mean_signed_contribution_when_selected",
              "mean_absolute_contribution_when_selected"]
    with (output / "expert_activation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset, modules in layers.items():
            for name, result in modules.items():
                for expert in result["experts"]:
                    values = {"dataset": dataset, "module": name,
                              "layer_role": role_by_module.get(name, ""),
                              "tokens": result["tokens"], **expert}
                    writer.writerow({field: values.get(field) for field in fields})

    edge_fields = ["dataset", "module", "receiver_expert", "sender_expert",
                   "signed_weight", "absolute_row_share"]
    with (output / "expert_communication_edges.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=edge_fields)
        writer.writeheader()
        for dataset, modules in layers.items():
            for name, result in modules.items():
                matrix = result["communication_matrix"]
                for receiver, row in enumerate(matrix["signed"]):
                    for sender, value in enumerate(row):
                        writer.writerow({
                            "dataset": dataset, "module": name,
                            "receiver_expert": receiver, "sender_expert": sender,
                            "signed_weight": value,
                            "absolute_row_share":
                                matrix["absolute_row_normalized"][receiver][sender],
                        })

    with (output / "expert_specialization.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        fields = ["module", "expert", "preferred_dataset",
                  "dataset_selectivity_one_minus_normalized_entropy",
                  "mean_router_weight_by_dataset_json"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, experts in specialization.items():
            for expert in experts:
                writer.writerow({
                    "module": name, "expert": expert["expert"],
                    "preferred_dataset": expert["preferred_dataset"],
                    "dataset_selectivity_one_minus_normalized_entropy":
                        expert["dataset_selectivity_one_minus_normalized_entropy"],
                    "mean_router_weight_by_dataset_json": json.dumps(
                        expert["mean_router_weight_by_dataset"]),
                })


@torch.inference_mode()
def main(argv=None):
    args = parser().parse_args(argv)
    if args.samples < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("samples/batch-size must be positive and workers non-negative")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    model, metadata, kind = load_forecaster(checkpoint)
    cfg = metadata["config"]
    data_path = Path(args.data_path or os.environ.get("POSEIDON_ROOT") or
                     metadata["data_path"]).resolve()
    device, amp = prepare_device(args)
    model.to(device).eval()
    collector = ExpertDiagnosticsCollector(model)
    try:
        for name in args.datasets:
            dataset = PoseidonDataset(
                data_path, name, split=args.split,
                num_trajectories=cfg["num_trajectories"],
                validation_trajectories=cfg["validation_trajectories"],
                test_trajectories=cfg["test_trajectories"],
                max_num_time_steps=cfg["max_num_time_steps"])
            if not 1 <= args.first_target_time <= dataset.steps:
                raise ValueError(f"{name}: first-target-time must be within 1..{dataset.steps}")
            samples = min(args.samples, dataset.count)
            indices = range(args.first_target_time - 1, samples * dataset.steps, dataset.steps)
            loader = DataLoader(
                Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=device.type == "cuda")
            collector.select_dataset(name)
            for batch in loader:
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                times = batch["time"].to(device, non_blocking=True)
                with autocast_context(device, amp):
                    model(pixel_values, times)
            print(json.dumps({"dataset": name, "samples": samples}), flush=True)
        layer_results = collector.result()
    finally:
        collector.close()

    representative = representative_moe_layers(
        {name for modules in layer_results.values() for name in modules})
    specialization = cross_dataset_specialization(layer_results)
    output = (Path(args.output_dir) if args.output_dir else
              checkpoint.parent / "expert_analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(checkpoint), "checkpoint_kind": kind,
        "architecture": metadata["architecture"], "data_path": str(data_path),
        "analysis_config": vars(args), "representative_layers": representative,
        "layers": layer_results, "cross_dataset_specialization": specialization,
    }
    (output / "expert_diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    write_csvs(output, layer_results, specialization)
    files = ["expert_diagnostics.json", "expert_activation.csv",
             "expert_specialization.csv", "expert_communication_edges.csv"]
    if not args.no_heatmap:
        from train.plot_expert_activation_heatmap import generate_outputs
        generated = generate_outputs(
            output / "expert_activation.csv", output, args.datasets,
            metadata["architecture"]["num_routed_experts"])
        files.extend(path.name for path in generated)
    print(json.dumps({"event": "expert_analysis_complete", "output_dir": str(output),
                      "files": files}), flush=True)


if __name__ == "__main__":
    main()
