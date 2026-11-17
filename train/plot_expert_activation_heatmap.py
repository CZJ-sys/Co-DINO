"""Aggregate routed-expert diagnostics and draw dataset-by-expert heatmaps."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.expert_diagnostics import representative_moe_layers


DATASETS = ("NS-Gauss", "NS-Sines", "CE-RP", "CE-CRP", "CE-Gauss", "CE-KH")
LAYER_ROLES = ("encoder_first", "encoder_last", "decoder_first", "decoder_last")
ROLE_LABELS = {
    "encoder_first": "Encoder first layer",
    "encoder_last": "Encoder last layer",
    "decoder_first": "Decoder first layer",
    "decoder_last": "Decoder last layer",
}
POSITIVE_METRICS = {
    "activation_fraction": "Routed-expert activation (%)",
    "mean_router_weight": "Mean router probability (%)",
    "selected_weight_fraction": "Selected gate-weight share (%)",
}
SIGNED_METRIC = "mean_signed_contribution_per_token"
HEATMAP_FONT_SIZE = 9
HEATMAP_TITLE_SIZE = 10
HEATMAP_CELL_SIZE = 8
DEFAULT_ACTIVATION_VMAX = 0.5


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True,
                   help="expert_activation.csv produced by train/analyze_experts.py")
    p.add_argument("--output-dir", help="Default: directory containing input CSV")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--metric", choices=tuple(POSITIVE_METRICS),
                   default="activation_fraction")
    p.add_argument("--vmax", type=float, default=None,
                   help="Shared positive heatmap maximum as a raw fraction; activation default: 0.5")
    p.add_argument("--signed-vmax", type=float, default=None,
                   help="Symmetric signed-contribution limit as a raw fraction")
    p.add_argument("--output-prefix", default="expert_activation_layer_mean_heatmap")
    return p


def _read_vectors(input_csv, datasets, num_experts, metric):
    if num_experts < 1:
        raise ValueError("num-experts must be positive")
    requested = list(datasets)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("datasets must be non-empty and unique")
    values = {}
    with Path(input_csv).resolve(strict=True).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"dataset", "module", "kind", "expert", metric}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV fields: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            if row["dataset"] not in requested or row["kind"] != "routed":
                continue
            try:
                expert = int(row["expert"])
                value = float(row[metric])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid expert or {metric} at CSV line {line}") from error
            if expert not in range(num_experts):
                raise ValueError(
                    f"CSV line {line}: routed expert {expert} is outside 0..{num_experts - 1}")
            if not math.isfinite(value):
                raise ValueError(f"CSV line {line}: {metric} must be finite")
            if metric in POSITIVE_METRICS and not 0.0 <= value <= 1.0:
                raise ValueError(f"CSV line {line}: {metric} must be within [0, 1]")
            key = (row["dataset"], row["module"], expert)
            if key in values:
                raise ValueError(f"Duplicate dataset/module/expert row at CSV line {line}: {key}")
            values[key] = value
    return values, requested


def _modules_and_vectors(values, dataset, num_experts, require_unit_sum):
    modules = sorted({module for name, module, _ in values if name == dataset})
    if not modules:
        raise ValueError(f"No routed-expert rows found for dataset {dataset}")
    expected = set(range(num_experts))
    vectors = {}
    for module in modules:
        present = {expert for name, layer, expert in values
                   if name == dataset and layer == module}
        if present != expected:
            raise ValueError(
                f"{dataset}/{module}: missing routed experts {sorted(expected - present)}")
        vector = [values[(dataset, module, expert)] for expert in range(num_experts)]
        if require_unit_sum and abs(sum(vector) - 1.0) > 1e-4:
            raise ValueError(
                f"{dataset}/{module}: metric sums to {sum(vector):.6g}, expected 1")
        vectors[module] = vector
    return modules, vectors


def aggregate_layer_means(input_csv, datasets=DATASETS, num_experts=8,
                          metric="activation_fraction"):
    """Return equal-layer means for every dataset/expert from the long-form CSV."""
    if metric not in (*POSITIVE_METRICS, SIGNED_METRIC):
        raise ValueError(f"Unsupported metric: {metric}")
    values, requested = _read_vectors(input_csv, datasets, num_experts, metric)
    result, reference_modules = {}, None
    for dataset in requested:
        modules, vectors = _modules_and_vectors(
            values, dataset, num_experts, metric in POSITIVE_METRICS)
        if reference_modules is None:
            reference_modules = modules
        elif modules != reference_modules:
            raise ValueError(
                f"Layer mismatch for {dataset}: expected {reference_modules}, found {modules}")
        result[dataset] = {
            "num_layers": len(modules),
            "values": [sum(vectors[module][expert] for module in modules) / len(modules)
                       for expert in range(num_experts)],
        }
    return result


def aggregate_representative_layers(input_csv, datasets=DATASETS, num_experts=8,
                                    metric="activation_fraction"):
    """Return the four execution-order boundary layers for every dataset."""
    if metric not in (*POSITIVE_METRICS, SIGNED_METRIC):
        raise ValueError(f"Unsupported metric: {metric}")
    values, requested = _read_vectors(input_csv, datasets, num_experts, metric)
    per_dataset, reference_roles = {}, None
    for dataset in requested:
        modules, vectors = _modules_and_vectors(
            values, dataset, num_experts, metric in POSITIVE_METRICS)
        roles = representative_moe_layers(modules)
        if reference_roles is None:
            reference_roles = roles
        elif roles != reference_roles:
            raise ValueError(
                f"Representative-layer mismatch for {dataset}: "
                f"expected {reference_roles}, found {roles}")
        per_dataset[dataset] = vectors
    return {
        role: {
            dataset: {"module": module, "num_layers": 1,
                      "values": per_dataset[dataset][module]}
            for dataset in requested
        }
        for role, module in reference_roles.items()
    }


def write_summary_csv(summary, path):
    path = Path(path)
    fields = ["dataset", "num_layers"] + [f"expert_{index}" for index in
                                             range(len(next(iter(summary.values()))["values"]))]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset, row in summary.items():
            writer.writerow({"dataset": dataset, "num_layers": row["num_layers"],
                             **{f"expert_{index}": value
                                for index, value in enumerate(row["values"])}})


def write_representative_csv(summaries, path, value_name):
    num_experts = len(next(iter(next(iter(summaries.values())).values()))["values"])
    fields = ["layer_role", "module", "dataset", "metric"] + [
        f"expert_{index}" for index in range(num_experts)]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for role in LAYER_ROLES:
            for dataset, row in summaries[role].items():
                writer.writerow({"layer_role": role, "module": row["module"],
                    "dataset": dataset, "metric": value_name,
                    **{f"expert_{index}": value
                       for index, value in enumerate(row["values"])}})


def _plot(summary, output_dir, prefix, title, colorbar_label, signed=False,
          vmax=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        from matplotlib.ticker import PercentFormatter
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "Heatmap generation requires matplotlib; install requirements.txt") from error
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.size": 9,
        "axes.linewidth": 0.8,
    })
    datasets = list(summary)
    matrix = np.asarray([summary[name]["values"] for name in datasets], dtype=float)
    if signed:
        observed = float(np.abs(matrix).max())
        limit = observed if vmax is None else float(vmax)
        if limit <= 0:
            limit = 1e-12
        if limit + 1e-15 < observed:
            raise ValueError(
                f"signed-vmax must cover the observed absolute maximum {observed:.6g}")
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        cmap, image_kwargs = "RdBu_r", {"norm": norm}
    else:
        observed = float(matrix.max())
        limit = observed if vmax is None else float(vmax)
        if limit <= 0 or limit + 1e-15 < observed:
            raise ValueError(f"vmax must cover the observed maximum {observed:.6g}")
        cmap, image_kwargs = "Blues", {"vmin": 0.0, "vmax": limit}
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    image = ax.imshow(matrix, cmap=cmap, aspect="auto", **image_kwargs)
    ax.set_xticks(np.arange(matrix.shape[1]),
                  labels=[f"E{index}" for index in range(matrix.shape[1])])
    ax.set_yticks(np.arange(matrix.shape[0]), labels=datasets)
    ax.set_xlabel("Routed expert")
    ax.set_ylabel("Dataset")
    ax.set_title(title, fontsize=HEATMAP_TITLE_SIZE, fontweight="semibold", pad=8)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", which="major", length=0,
                   labelsize=HEATMAP_FONT_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            label = f"{100.0 * value:+.1f}" if signed else f"{100.0 * value:.1f}"
            ax.text(column, row, label, ha="center", va="center",
                    fontsize=HEATMAP_CELL_SIZE,
                    color="white" if luminance < 0.52 else "#17233C")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    colorbar.set_label(colorbar_label, rotation=270, rotation_mode="anchor",
                       labelpad=13, fontsize=HEATMAP_FONT_SIZE)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    colorbar.ax.tick_params(labelsize=HEATMAP_FONT_SIZE)
    colorbar.outline.set_linewidth(0.6)
    output_dir = Path(output_dir)
    svg = output_dir / f"{prefix}.svg"
    pdf = output_dir / f"{prefix}.pdf"
    png = output_dir / f"{prefix}.png"
    tiff = output_dir / f"{prefix}.tiff"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(tiff, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return [svg, pdf, png, tiff]


def plot_heatmap(summary, output_dir, prefix, metric="activation_fraction", vmax=None):
    if vmax is None and metric == "activation_fraction":
        vmax = DEFAULT_ACTIVATION_VMAX
    return _plot(summary, output_dir, prefix,
                 "Mean routed-expert activation across all layers",
                 POSITIVE_METRICS[metric], signed=False, vmax=vmax)


def generate_outputs(input_csv, output_dir=None, datasets=DATASETS, num_experts=8,
                     metric="activation_fraction", vmax=None,
                     prefix="expert_activation_layer_mean_heatmap", signed_vmax=None):
    input_path = Path(input_csv).resolve(strict=True)
    output = Path(output_dir).resolve() if output_dir else input_path.parent
    output.mkdir(parents=True, exist_ok=True)
    activation_all = aggregate_layer_means(input_path, datasets, num_experts, metric)
    activation_selected = aggregate_representative_layers(
        input_path, datasets, num_experts, metric)
    signed_all = aggregate_layer_means(input_path, datasets, num_experts, SIGNED_METRIC)
    signed_selected = aggregate_representative_layers(
        input_path, datasets, num_experts, SIGNED_METRIC)

    activation_csv = output / "expert_activation_layer_mean.csv"
    activation_selected_csv = output / "expert_activation_selected_layers.csv"
    signed_csv = output / "expert_signed_contribution_layer_mean.csv"
    signed_selected_csv = output / "expert_signed_contribution_selected_layers.csv"
    write_summary_csv(activation_all, activation_csv)
    write_representative_csv(activation_selected, activation_selected_csv, metric)
    write_summary_csv(signed_all, signed_csv)
    write_representative_csv(signed_selected, signed_selected_csv, SIGNED_METRIC)
    files = [activation_csv, activation_selected_csv, signed_csv, signed_selected_csv]

    positive_observed = max(
        [value for row in activation_all.values() for value in row["values"]] +
        [value for summary in activation_selected.values()
         for row in summary.values() for value in row["values"]])
    positive_limit = (DEFAULT_ACTIVATION_VMAX
                      if vmax is None and metric == "activation_fraction"
                      else positive_observed if vmax is None else vmax)
    signed_observed = max(
        [abs(value) for row in signed_all.values() for value in row["values"]] +
        [abs(value) for summary in signed_selected.values()
         for row in summary.values() for value in row["values"]])
    signed_limit = max(signed_observed if signed_vmax is None else signed_vmax, 1e-12)
    files.extend(plot_heatmap(activation_all, output, prefix, metric, positive_limit))
    files.extend(_plot(
        signed_all, output, "expert_signed_contribution_layer_mean_heatmap",
        "Mean signed expert contribution across all layers",
        "Signed contribution to routed output (%)", signed=True, vmax=signed_limit))
    for role in LAYER_ROLES:
        files.extend(_plot(
            activation_selected[role], output, f"expert_activation_{role}_heatmap",
            f"Routed-expert activation: {ROLE_LABELS[role]}",
            POSITIVE_METRICS[metric], signed=False, vmax=positive_limit))
        files.extend(_plot(
            signed_selected[role], output,
            f"expert_signed_contribution_{role}_heatmap",
            f"Signed expert contribution: {ROLE_LABELS[role]}",
            "Signed contribution to routed output (%)", signed=True,
            vmax=signed_limit))
    metadata = {
        "source_csv": str(input_path), "activation_metric": metric,
        "signed_metric": SIGNED_METRIC,
        "signed_metric_definition": (
            "dot(w_i*y_i_after_communication, routed_output) / "
            "squared_l2_norm(routed_output), averaged over tokens; inactive experts contribute zero"),
        "gate_is_nonnegative": True,
        "negative_values_mean": (
            "the post-communication weighted expert output opposes the final routed-output direction"),
        "aggregation": "arithmetic mean across layers; boundary layers follow forward execution order",
        "datasets": list(datasets), "num_routed_experts": num_experts,
        "representative_layers": {
            role: next(iter(activation_selected[role].values()))["module"]
            for role in LAYER_ROLES},
        "activation_color_scale": {"vmin": 0.0, "vmax": float(positive_limit)},
        "signed_color_scale": {"vmin": -float(signed_limit), "vcenter": 0.0,
                               "vmax": float(signed_limit)},
        "font_sizes_points": {"base": HEATMAP_FONT_SIZE,
                              "title": HEATMAP_TITLE_SIZE,
                              "cell": HEATMAP_CELL_SIZE},
        "cell_display": "percentage; CSV files store raw values",
    }
    metadata_path = output / "expert_heatmap_metadata.json"
    legacy_metadata_path = output / "expert_activation_layer_mean_metadata.json"
    metadata_text = json.dumps(metadata, indent=2) + "\n"
    metadata_path.write_text(metadata_text, encoding="utf-8")
    legacy_metadata_path.write_text(metadata_text, encoding="utf-8")
    files.extend((metadata_path, legacy_metadata_path))
    return files


def main(argv=None):
    args = parser().parse_args(argv)
    files = generate_outputs(args.input_csv, args.output_dir, args.datasets,
                             args.num_experts, args.metric, args.vmax,
                             args.output_prefix, args.signed_vmax)
    print(json.dumps({"event": "expert_heatmaps_complete",
                      "files": [str(path) for path in files]}), flush=True)
    return files


if __name__ == "__main__":
    main()
