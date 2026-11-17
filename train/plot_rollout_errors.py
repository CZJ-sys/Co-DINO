"""Export per-step rollout errors and sparse-step charts from detailed reports."""

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_BAR_COLOR = "#F2ECFF"
DATASET_BAR_COLORS = {
    "CE-RP": "#F2ECFF",
    "CE-KH": "#FFDE87",
    "CE-CRP": "#B4C7E7",
}
ROLLOUT_Y_LIMITS = (0.0, 110.0)
DISPLAY_ROLLOUT_STEPS = (1, 10, 20)
BAR_LABEL_FONT_SIZE = 8
BAR_LABEL_DECIMALS = 2
BAR_LABEL_OFFSET = 2.0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="Co-DINO detailed evaluation JSON")
    p.add_argument("--output-dir", type=Path,
                   help="Default: <input_stem_without_detailed>_rollout_graphs")
    p.add_argument("--datasets", nargs="+", help="Subset of datasets; default: all in report")
    p.add_argument("--metric-space", choices=("normalized", "physical"), default="normalized")
    p.add_argument("--graph-metric",
                   choices=("relative_l1_percent", "relative_l2_percent", "mae", "rmse"),
                   default="relative_l1_percent")
    p.add_argument("--color",
                   help="Override every dataset bar color; default: dataset-aware palette")
    p.add_argument("--model-label",
                   help="Plot title label; default: infer from the detailed report")
    return p


def _selected_space(step, metric_space):
    key = "native_normalized" if metric_space == "normalized" else "physical"
    spaces = step.get("metric_spaces")
    if not isinstance(spaces, dict) or key not in spaces:
        raise ValueError(f"Detailed rollout step is missing metric_spaces.{key}")
    return spaces[key]


def _metric_value(values, key):
    value = values.get(key)
    return None if value is None else float(value)


def _base_record(name, index, target_time, samples, eval_channels, values):
    joint = values.get("joint", {}) if isinstance(values.get("joint"), dict) else {}
    relative_l1 = joint.get("mean_relative_l1_percent",
                            values.get("mean_relative_l1_percent",
                                       values.get("relative_l1_percent")))
    relative_l2 = values.get("mean_relative_l2_percent")
    if relative_l2 is None and values.get("relative_l2") is not None:
        relative_l2 = 100.0 * float(values["relative_l2"])
    return {
        "dataset": name,
        "rollout_step": int(index),
        "target_time": int(target_time),
        "samples": int(samples),
        "eval_channels": ",".join(str(value) for value in eval_channels),
        "mean_relative_l1_percent": (None if relative_l1 is None else float(relative_l1)),
        "mean_relative_l2_percent": (None if relative_l2 is None else float(relative_l2)),
        "mae": _metric_value(values, "mae"),
        "rmse": _metric_value(values, "rmse"),
    }


def _records_from_codino(name, entry, metric_space):
    rollout = entry.get("autoregressive")
    if not isinstance(rollout, dict) or not isinstance(rollout.get("per_step"), list):
        return None
    records = []
    for index, step in enumerate(rollout["per_step"], start=1):
        values = _selected_space(step, metric_space)
        records.append(_base_record(
            name, index, step["target_time"], values["samples"],
            values["eval_channels"], values))
    return records


def extract_rollout_records(report, datasets=None, metric_space="normalized"):
    """Return one flat record per dataset and target step."""
    entries = report.get("datasets")
    if not isinstance(entries, dict):
        raise ValueError("Input is not a supported detailed evaluation report")
    names = list(entries) if datasets is None else list(datasets)
    missing = [name for name in names if name not in entries]
    if missing:
        raise ValueError(f"Datasets not present in report: {missing}")
    records = []
    for name in names:
        entry = entries[name]
        dataset_records = _records_from_codino(name, entry, metric_space)
        if not dataset_records:
            raise ValueError(f"{name} has no autoregressive.per_step metrics")
        records.extend(dataset_records)
    return records


def _infer_model_label(report, requested=None):
    if requested:
        return requested
    return str(report.get("model", "")).strip() or "Co-DINO"


def _dataset_bar_color(dataset, override=None):
    if override:
        return override
    return DATASET_BAR_COLORS.get(dataset, DEFAULT_BAR_COLOR)


def _select_display_rows(rows, required_steps=DISPLAY_ROLLOUT_STEPS):
    by_step = {int(row["rollout_step"]): row for row in rows}
    selected = [by_step[step] for step in required_steps if step in by_step]
    if not selected:
        dataset = rows[0]["dataset"] if rows else "dataset"
        raise ValueError(f"{dataset}: none of rollout steps {list(required_steps)} are present")
    return selected


def _metric_column(graph_metric):
    return ("mean_" + graph_metric
            if graph_metric in ("relative_l1_percent", "relative_l2_percent")
            else graph_metric)


def _metric_label(graph_metric, metric_space):
    space = "Normalized" if metric_space == "normalized" else "Physical"
    labels = {
        "relative_l1_percent": f"{space} mean Relative L1 (%)",
        "relative_l2_percent": f"{space} mean Relative L2 (%)",
        "mae": f"{space} MAE",
        "rmse": f"{space} RMSE",
    }
    return labels[graph_metric]


def _configure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Rollout graph generation requires matplotlib") from error
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })
    return plt


def _draw_dataset(ax, rows, dataset, graph_metric, metric_space, color, model_label):
    column = _metric_column(graph_metric)
    x = list(range(1, len(rows) + 1))
    values = [row[column] for row in rows]
    if any(value is None for value in values):
        raise ValueError(
            f"{dataset}: {graph_metric} is unavailable in this model's detailed report")
    bars = ax.bar(x, values, color=color, edgecolor="#8B78B8", linewidth=0.65)
    ax.set_title(f"{model_label} - {dataset}")
    ax.set_xlabel("Autoregressive rollout step")
    ax.set_ylabel(_metric_label(graph_metric, metric_space))
    ax.set_ylim(*ROLLOUT_Y_LIMITS)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {row['rollout_step']}" for row in rows])
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
    ax.set_axisbelow(True)
    label_ceiling = ROLLOUT_Y_LIMITS[1] - 4.0
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                min(float(value) + BAR_LABEL_OFFSET, label_ceiling),
                f"{float(value):.{BAR_LABEL_DECIMALS}f}",
                ha="center", va="bottom", fontsize=BAR_LABEL_FONT_SIZE,
                color="#2E2145", clip_on=True)


def _save_figure_bundle(fig, base):
    """Export editable vectors and a high-resolution raster preview."""
    base = str(base)
    svg = base + ".svg"
    pdf = base + ".pdf"
    png = base + ".png"
    tiff = base + ".tiff"
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(tiff, dpi=600, bbox_inches="tight", facecolor="white")
    return [png, pdf, svg, tiff]


def generate_rollout_artifacts(input_path, output_dir=None, datasets=None,
                               metric_space="normalized", graph_metric="relative_l1_percent",
                               color=None, model_label=None):
    """Write a per-step CSV, one chart per dataset and a multi-dataset overview."""
    input_path = Path(input_path).resolve(strict=True)
    report = json.loads(input_path.read_text(encoding="utf-8"))
    model_label = _infer_model_label(report, model_label)
    records = extract_rollout_records(report, datasets, metric_space)
    stem = input_path.stem.removesuffix("_detailed")
    output = Path(output_dir) if output_dir else input_path.with_name(stem + "_rollout_graphs")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "rollout_step", "target_time", "samples", "eval_channels",
                  "mean_relative_l1_percent", "mean_relative_l2_percent", "mae", "rmse"]
    csv_path = output / "rollout_per_step_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    names = list(dict.fromkeys(row["dataset"] for row in records))
    grouped = {name: [row for row in records if row["dataset"] == name] for name in names}
    display_grouped = {name: _select_display_rows(rows) for name, rows in grouped.items()}
    dataset_colors = {name: _dataset_bar_color(name, color) for name in names}
    plt = _configure_matplotlib()
    figures = []
    for name, rows in display_grouped.items():
        fig, ax = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
        _draw_dataset(ax, rows, name, graph_metric, metric_space, dataset_colors[name],
                      model_label)
        figures.extend(_save_figure_bundle(
            fig, output / f"{name}_rollout_{graph_metric}"))
        plt.close(fig)
    if len(names) > 1:
        columns = min(3, len(names))
        rows_count = math.ceil(len(names) / columns)
        fig, axes = plt.subplots(rows_count, columns,
                                 figsize=(7.2, 2.55 * rows_count),
                                 constrained_layout=True, squeeze=False)
        for ax, name in zip(axes.flat, names):
            _draw_dataset(ax, display_grouped[name], name, graph_metric, metric_space,
                          dataset_colors[name], model_label)
        for ax in list(axes.flat)[len(names):]:
            ax.set_visible(False)
        figures.extend(_save_figure_bundle(
            fig, output / f"all_datasets_rollout_{graph_metric}"))
        plt.close(fig)
    manifest = {
        "input": str(input_path),
        "metric_space": metric_space,
        "graph_metric": graph_metric,
        "requested_rollout_steps": list(DISPLAY_ROLLOUT_STEPS),
        "displayed_rollout_steps": {
            name: [row["rollout_step"] for row in rows]
            for name, rows in display_grouped.items()
        },
        "y_axis_limits": list(ROLLOUT_Y_LIMITS),
        "bar_value_labels": {"enabled": True,
                             "font_size_points": BAR_LABEL_FONT_SIZE,
                             "decimal_places": BAR_LABEL_DECIMALS},
        "bar_color": color or DEFAULT_BAR_COLOR,
        "color_mode": "override" if color else "dataset_palette",
        "dataset_bar_colors": dataset_colors,
        "model_label": model_label,
        "datasets": names,
        "csv": str(csv_path),
        "figures": figures,
    }
    manifest_path = output / "rollout_graph_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    args = parser().parse_args(argv)
    result = generate_rollout_artifacts(
        args.input, args.output_dir, args.datasets, args.metric_space,
        args.graph_metric, args.color, args.model_label)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
