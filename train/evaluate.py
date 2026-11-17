"""Evaluate a trained Co-DINO encoder-decoder on held-out trajectories."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.architecture_config import architecture_from_dict
from model.checkpoint import load_checkpoint
from model.data import CHANNEL_CONTRACT, DATASETS, PoseidonDataset
from model.evaluation import DualSpaceErrors
from model.evaluation_summary import compact_evaluation_report
from model.model import CoDINOForForecasting
from train.evaluation_artifacts import RolloutFieldSaver
from train.plot_rollout_errors import generate_rollout_artifacts
from train.runtime import prepare_device


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Full-model pretraining or finetuning model_final.pt, best.pt or last.pt")
    p.add_argument("--data-path", "--data_path", default=None)
    p.add_argument("--datasets", "--dataset", nargs="+", choices=DATASETS,
                   help="Held-out Poseidon datasets; default: all six")
    p.add_argument("--split", choices=("test", "val"), default="test")
    p.add_argument("--samples", default="200", help="Trajectories per dataset (default: 200), or all")
    p.add_argument("--first-target-time", "--first_target_time", type=int, default=1,
                   help="First predicted frame, with input from preceding frame (default: t1)")
    p.add_argument("--rollout-steps", "--rollout_steps", type=int, default=20,
                   help="Autoregressive predictions from first target (default: 20)")
    p.add_argument("--dt1-only", "--dt1_only", action="store_true",
                   help="Only evaluate the first t0-to-t1 prediction by default")
    p.add_argument("--evaluation-resolution", "--evaluation_resolution",
                   choices=("64", "96", "128", "native"), default="128",
                   help="Inference/metric grid (default: 128); native preserves source resolution")
    p.add_argument("--metric-space", choices=("normalized", "physical"), default="normalized")
    p.add_argument("--batch-size", "--batch_size", type=int, default=4)
    p.add_argument("--num-workers", "--num_workers", type=int, default=1)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--amp", choices=("auto", "none", "bf16", "fp16"), default="auto")
    p.add_argument("--output", help="JSON report path; default: checkpoint/evaluation/<protocol>.json")
    p.add_argument("--save-detailed-metrics", "--save_detailed_metrics", action="store_true",
                   help="Also save full rollout per-step metrics to <output_stem>_detailed.json")
    p.add_argument("--save-ground-truth-and-forecast", "--save_ground_truth_and_forecast",
                   nargs="?", const="all", metavar="COUNT", default=None,
                   help="Save exact tensors and GT/forecast PNG contact sheets; flag alone saves all, or pass a trajectory count")
    p.add_argument("--field-output-dir", "--field_output_dir",
                   help="Default: <evaluation_output_stem>_fields")
    p.add_argument("--generate", choices=("graph",),
                   help="Generate rollout error artifacts from the detailed report")
    p.add_argument("--generate-graph", "--generate_graph", action="store_true",
                   help="Alias for --generate graph")
    p.add_argument("--graph-output-dir", "--graph_output_dir",
                   help="Default: <evaluation_output_stem>_rollout_graphs")
    p.add_argument("--graph-metric", "--graph_metric",
                   choices=("relative_l1_percent", "relative_l2_percent", "mae", "rmse"),
                   default="relative_l1_percent")
    p.add_argument("--graph-color", "--graph_color",
                   help="Override every dataset bar color; default: dataset-aware palette")
    return p


def load_forecaster(path):
    payload = load_checkpoint(path)
    if "encoder" in payload and "model" not in payload:
        raise ValueError("Encoder-only checkpoint cannot be evaluated for forecasting")
    if payload.get("channel_contract") != CHANNEL_CONTRACT:
        raise ValueError("Checkpoint channel/normalization contract does not match")
    if "identity" in payload:
        identity = payload["identity"]
        architecture = identity["architecture"]
        metadata = identity
        kind = "training_checkpoint"
        if (identity.get("stage") not in ("forecast_pretrain", "finetune")
                or identity.get("task") != "forecasting"):
            raise ValueError("Expected a complete pretraining or finetuning forecasting checkpoint")
        adjacent = Path(path).parent / "run_config.json"
        if adjacent.exists():
            saved = json.loads(adjacent.read_text(encoding="utf-8"))
            if saved.get("channel_contract") != CHANNEL_CONTRACT or saved.get("architecture") != architecture:
                raise ValueError("Checkpoint run_config.json has a different channel contract or architecture")
    else:
        architecture = payload.get("architecture")
        metadata = payload.get("metadata", {})
        kind = "model_final"
        if (metadata.get("stage") not in ("forecast_pretrain", "finetune")
                or metadata.get("task") != "forecasting"):
            raise ValueError("Expected a complete pretraining or finetuning forecasting checkpoint")
        if metadata.get("architecture") != architecture:
            raise ValueError("Checkpoint architecture metadata does not match")
    config = architecture_from_dict(architecture)
    model = CoDINOForForecasting(config)
    model.load_state_dict(payload["model"], strict=True)
    return model, metadata, kind


def autocast_context(device, amp):
    if amp == "none":
        return nullcontext()
    return torch.autocast(device_type=device.type,
                          dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


def resize_state(state, resolution):
    if resolution == "native" or state.shape[-2:] == (int(resolution), int(resolution)):
        return state
    return F.interpolate(state, size=(int(resolution), int(resolution)),
                         mode="bilinear", align_corners=False, antialias=True)


@torch.inference_mode()
def evaluate_one_step(model, dataset, sample_count, batch_size, num_workers, device, amp,
                      metric_space, first_target_time=1, resolution="native"):
    # One initial-condition target per trajectory in the aligned dt1 protocol.
    selected = Subset(dataset, range(first_target_time - 1,
                                     sample_count * dataset.steps, dataset.steps))
    loader = DataLoader(selected, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    errors = DualSpaceErrors(dataset.name, metric_space,
                             getattr(dataset, "normalization_contract", None))
    for batch in loader:
        x = resize_state(batch["pixel_values"], resolution).to(device, non_blocking=True)
        target = resize_state(batch["labels"], resolution).to(device, non_blocking=True)
        mask = batch["pixel_mask"].to(device, non_blocking=True)
        time = batch["time"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            prediction = model(x, time)
        errors.update(prediction, target, mask)
    return errors.result()


@torch.inference_mode()
def evaluate_rollout(model, dataset, sample_count, batch_size, steps, device, amp, metric_space,
                     first_target_time=1, resolution="native", field_saver=None):
    if first_target_time < 1 or steps < 1 or first_target_time + steps - 1 > dataset.steps:
        raise ValueError(f"rollout must fit target frames 1..{dataset.steps} for {dataset.name}")
    normalization = getattr(dataset, "normalization_contract", None)
    overall = DualSpaceErrors(dataset.name, metric_space, normalization)
    per_step = [DualSpaceErrors(dataset.name, metric_space, normalization) for _ in range(steps)]
    mask_values = [True, False, False, True] if dataset.name.startswith("NS-") else [False] * 4
    for first in range(0, sample_count, batch_size):
        trajectories = range(dataset.start + first, dataset.start + min(first + batch_size, sample_count))
        state = resize_state(torch.stack([dataset._state(t, first_target_time - 1)
                                          for t in trajectories]), resolution).to(device)
        mask = torch.tensor(mask_values, dtype=torch.bool, device=device).expand(len(state), -1)
        time = torch.full((len(state),), 1.0 / CHANNEL_CONTRACT["time_scale"], device=device)
        keep = (min(len(state), max(0, field_saver.save_count - first))
                if field_saver is not None else 0)
        saved_targets, saved_predictions = [], []
        for step in range(steps):
            target = resize_state(torch.stack([dataset._state(t, first_target_time + step)
                                               for t in trajectories]), resolution).to(device)
            with autocast_context(device, amp):
                prediction = model(state, time)
            overall.update(prediction, target, mask)
            per_step[step].update(prediction, target, mask)
            if keep:
                saved_targets.append(target[:keep].detach().float().cpu())
                saved_predictions.append(prediction[:keep].detach().float().cpu())
            state = prediction.detach().float()
            if dataset.name.startswith("NS-"):
                # NS rho/p are fixed physical placeholders in every training input.
                normalization = getattr(dataset, "normalization_contract", CHANNEL_CONTRACT)
                state[:, 0] = (1.0 - normalization["mean"][0]) / normalization["std"][0]
                state[:, 3] = (0.0 - normalization["mean"][3]) / normalization["std"][3]
        if keep:
            targets = torch.stack(saved_targets, dim=1)
            predictions = torch.stack(saved_predictions, dim=1)
            for offset in range(keep):
                field_saver.save(first + offset, dataset.start + first + offset,
                                 targets[offset], predictions[offset])
    result = {"steps": steps, "initial_time": first_target_time - 1,
            "first_target_time": first_target_time, "teacher_forcing": False,
            "overall": overall.result(),
            "per_step": [{"target_time": first_target_time + step, **metric.result()}
                         for step, metric in enumerate(per_step)]}
    if field_saver is not None:
        result["saved_fields"] = field_saver.finalize()
    return result


def resolve_save_count(value, sample_count):
    if value is None:
        return 0
    if value == "all":
        return sample_count
    if not str(value).isdecimal() or int(value) < 1:
        raise ValueError("--save-ground-truth-and-forecast accepts a positive count or no value for all")
    return min(int(value), sample_count)


def main(argv=None):
    args = parser().parse_args(argv)
    generate_graph = args.generate_graph or args.generate == "graph"
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if args.samples != "all" and (not args.samples.isdecimal() or int(args.samples) < 1):
        raise ValueError("samples must be a positive integer or all")
    if args.dt1_only and (generate_graph or args.save_ground_truth_and_forecast is not None):
        raise ValueError("Rollout graphs and saved GT/forecast fields require autoregressive rollout; remove --dt1-only")
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    model, metadata, kind = load_forecaster(checkpoint)
    cfg = metadata["config"]
    names = args.datasets or list(DATASETS)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate evaluation datasets")
    data_path = Path(args.data_path or os.environ.get("POSEIDON_ROOT") or metadata["data_path"]).resolve()
    device, amp = prepare_device(args)
    model.to(device).eval()
    protocol_tag = (f"t{args.first_target_time}_dt1" if args.dt1_only else
                    f"t{args.first_target_time}_t{args.first_target_time + args.rollout_steps - 1}")
    default_name = (f"eval_{'_'.join(names)}_{args.split}_n{args.samples}_"
                    f"{protocol_tag}_{args.evaluation_resolution}_{args.metric_space}.json")
    output = Path(args.output) if args.output else checkpoint.parent / "evaluation" / default_name
    report = {"model": "Co-DINO", "checkpoint": str(checkpoint), "checkpoint_kind": kind,
              "training_stage": metadata["stage"],
              "training_policy": metadata.get("policy"), "architecture": metadata["architecture"],
              "data_path": str(data_path), "split": args.split, "datasets": {},
              "evaluation_config": {"datasets": names, "samples": args.samples,
                  "batch_size": args.batch_size, "num_workers": args.num_workers,
                  "rollout_steps": args.rollout_steps, "dt1_only": args.dt1_only,
                  "first_target_time": args.first_target_time,
                  "evaluation_resolution": args.evaluation_resolution,
                  "metric_space": args.metric_space, "device": str(device), "amp": amp,
                  "save_ground_truth_and_forecast": args.save_ground_truth_and_forecast,
                  "generate_graph": generate_graph,
                  "graph_metric": args.graph_metric, "graph_color": args.graph_color,
                  "ns_masked_channels": [0, 3]},
              "comparison_protocol": {"time_indexing": "zero_based",
                  "first_target_time": args.first_target_time,
                  "context_frames": 1, "rollout_steps": 1 if args.dt1_only else args.rollout_steps,
                  "input_and_target_resize": "antialiased_bilinear_no_resize_back"
                    if args.evaluation_resolution != "native" else "none",
                  "metric_spaces": ["native_normalized", "physical"],
                  "relative_l1_units": "percent in joint/per_qoi and relative_l1_percent; ratio in relative_l1"}}
    for name in names:
        dataset = PoseidonDataset(data_path, name, split=args.split, task="forecasting",
            num_trajectories=cfg["num_trajectories"],
            validation_trajectories=cfg["validation_trajectories"],
            test_trajectories=cfg["test_trajectories"],
            max_num_time_steps=cfg["max_num_time_steps"])
        sample_count = dataset.count if args.samples == "all" else min(int(args.samples), dataset.count)
        steps = 1 if args.dt1_only else args.rollout_steps
        if args.first_target_time < 1 or steps < 1 or args.first_target_time + steps - 1 > dataset.steps:
            raise ValueError(f"{name}: target interval must fit frames 1..{dataset.steps}")
        entry = {"manifest": dataset.manifest, "evaluated_trajectories": sample_count}
        if args.dt1_only:
            entry["one_step"] = {"target_time": args.first_target_time,
                **evaluate_one_step(model, dataset, sample_count,
                    args.batch_size, args.num_workers, device, amp, args.metric_space,
                    args.first_target_time, args.evaluation_resolution)}
        else:
            field_saver = None
            save_count = resolve_save_count(args.save_ground_truth_and_forecast, sample_count)
            if save_count:
                field_root = (Path(args.field_output_dir) if args.field_output_dir else
                              output.with_name(output.stem + "_fields"))
                masked_channels = [0, 3] if name.startswith("NS-") else []
                field_saver = RolloutFieldSaver(
                    field_root, name, save_count, args.first_target_time,
                    args.evaluation_resolution, masked_channels)
            entry["autoregressive"] = evaluate_rollout(model, dataset, sample_count,
                args.batch_size, steps, device, amp, args.metric_space,
                args.first_target_time, args.evaluation_resolution, field_saver)
            # The first rollout prediction is the aligned one-step metric.
            entry["one_step"] = entry["autoregressive"]["per_step"][0]
        report["datasets"][name] = entry
        print(json.dumps({"dataset": name, "trajectories": sample_count,
                          "eval_channels": entry["one_step"]["eval_channels"],
                          "dt1_relative_l1_percent": entry["one_step"]["relative_l1_percent"],
                          "rollout_relative_l1_percent": entry.get("autoregressive", {})
                            .get("overall", {}).get("relative_l1_percent")}), flush=True)
    compact = compact_evaluation_report(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    detailed_output = None
    if args.save_detailed_metrics or generate_graph:
        detailed_output = output.with_name(output.stem + "_detailed" + output.suffix)
        detailed_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    graph_manifest = None
    if generate_graph:
        graph_manifest = generate_rollout_artifacts(
            detailed_output, args.graph_output_dir, names, args.metric_space,
            args.graph_metric, args.graph_color, "Co-DINO")
    print(json.dumps({"event": "evaluation_complete", "output": str(output.resolve()),
                      "detailed_output": str(detailed_output.resolve()) if detailed_output else None,
                      "graph_manifest": graph_manifest}),
          flush=True)
    return compact


if __name__ == "__main__":
    main()
