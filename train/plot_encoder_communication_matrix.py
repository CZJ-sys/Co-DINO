"""Extract and plot first/last Encoder expert-communication matrices."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.blocks.sparse_moe import communication_moe_modules
from model.architecture_config import architecture_from_dict
from model.expert_diagnostics import representative_moe_layers
from model.model import CoDINOForForecasting


ROLES = ("encoder_first", "encoder_last")
ROLE_LABELS = {
    "encoder_first": "Encoder first layer",
    "encoder_last": "Encoder last layer",
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Full-model best.pt, last.pt or model_final.pt")
    p.add_argument("--output-dir",
                   help="Default: <checkpoint-dir>/expert_analysis/communication_matrices")
    p.add_argument("--vmax", type=float,
                   help="Shared absolute color limit in tanh-coefficient units")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--no-annotate", action="store_true",
                   help="Do not print coefficient values inside heatmap cells")
    return p


def load_forecaster_for_communication(path):
    """Load a trusted Co-DINO checkpoint without importing the data pipeline."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch releases before weights_only was added.
        payload = torch.load(path, map_location="cpu")
    if "encoder" in payload and "model" not in payload:
        raise ValueError("Encoder-only checkpoint has no complete forecasting model")
    if "identity" in payload:
        metadata = payload["identity"]
        architecture = metadata.get("architecture")
        kind = "training_checkpoint"
    else:
        architecture = payload.get("architecture")
        metadata = payload.get("metadata", {})
        kind = "model_final"
    if (metadata.get("stage") not in ("forecast_pretrain", "finetune")
            or metadata.get("task") != "forecasting"):
        raise ValueError(
            "Expected a complete forecasting pretraining or finetuning checkpoint")
    if kind == "model_final" and metadata.get("architecture") != architecture:
        raise ValueError("Checkpoint architecture metadata does not match")
    config = architecture_from_dict(architecture)
    model = CoDINOForForecasting(config)
    model.load_state_dict(payload["model"], strict=True)
    return model, metadata, kind


def extract_encoder_communication_matrices(model):
    """Return learned and forward-effective matrices for Encoder boundaries."""
    modules = communication_moe_modules(model)
    if not modules:
        raise ValueError("Checkpoint has no sparse MoE modules")
    role_names = representative_moe_layers(modules.keys())
    result = {}
    for role in ROLES:
        name = role_names[role]
        module = modules[name]
        raw = module.communication.mix.detach().float().cpu()
        bounded = torch.tanh(raw)
        effective = bounded.clone()
        effective.fill_diagonal_(0.0)
        divisor = max(1, module.num_active_experts - 1)
        result[role] = {
            "module": name,
            "num_routed_experts": module.num_routed_experts,
            "num_active_experts": module.num_active_experts,
            "raw_parameter": raw.numpy(),
            "bounded_parameter": bounded.numpy(),
            "effective_matrix": effective.numpy(),
            "per_selected_message_coefficient": (effective / divisor).numpy(),
        }
    return result


def _write_matrix_csv(path, matrix):
    experts = matrix.shape[0]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["receiver\\sender", *(f"E{i}" for i in range(experts))])
        for receiver, row in enumerate(matrix):
            writer.writerow([f"E{receiver}", *(float(value) for value in row)])


def _write_edge_csv(path, values):
    fields = ["receiver_expert", "sender_expert", "raw_parameter",
              "tanh_parameter", "effective_coefficient",
              "per_selected_message_coefficient", "diagonal_masked"]
    raw = values["raw_parameter"]
    bounded = values["bounded_parameter"]
    effective = values["effective_matrix"]
    scaled = values["per_selected_message_coefficient"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for receiver in range(raw.shape[0]):
            for sender in range(raw.shape[1]):
                writer.writerow({
                    "receiver_expert": receiver,
                    "sender_expert": sender,
                    "raw_parameter": float(raw[receiver, sender]),
                    "tanh_parameter": float(bounded[receiver, sender]),
                    "effective_coefficient": float(effective[receiver, sender]),
                    "per_selected_message_coefficient": float(scaled[receiver, sender]),
                    "diagonal_masked": receiver == sender,
                })


def _draw_cells(ax, image, matrix, annotate):
    experts = matrix.shape[0]
    ax.set_xticks(np.arange(experts), labels=[f"E{i}" for i in range(experts)])
    ax.set_yticks(np.arange(experts), labels=[f"E{i}" for i in range(experts)])
    ax.set_xlabel("Sender expert")
    ax.set_ylabel("Receiver expert")
    ax.set_xticks(np.arange(-0.5, experts, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, experts, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if not annotate:
        return
    annotation_size = max(5.0, 7.5 - 0.2 * experts)
    for receiver in range(experts):
        for sender in range(experts):
            value = matrix[receiver, sender]
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            ax.text(sender, receiver, f"{value:+.3f}", ha="center", va="center",
                    fontsize=annotation_size,
                    color="white" if luminance < 0.50 else "#17233C")


def _save_figure(fig, output, prefix, dpi):
    svg = output / f"{prefix}.svg"
    pdf = output / f"{prefix}.pdf"
    tiff = output / f"{prefix}.tiff"
    png = output / f"{prefix}.png"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(tiff, dpi=dpi, bbox_inches="tight")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    return [svg, pdf, tiff, png]


def plot_communication_matrices(values, output, vmax=None, dpi=600, annotate=True):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
    })
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    matrices = [values[role]["effective_matrix"] for role in ROLES]
    observed = max(float(np.max(np.abs(matrix))) for matrix in matrices)
    limit = observed if vmax is None else float(vmax)
    if not np.isfinite(limit) or limit <= 0:
        limit = 1e-12
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = "RdBu_r"  # negative=blue, zero=white, positive=red
    files = []

    for role, matrix in zip(ROLES, matrices):
        fig, ax = plt.subplots(figsize=(3.5, 3.35), constrained_layout=True)
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
        _draw_cells(ax, image, matrix, annotate)
        ax.set_title(f"Expert communication: {ROLE_LABELS[role]}",
                     fontsize=8, fontweight="semibold", pad=6)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Effective communication coefficient",
                           rotation=270, rotation_mode="anchor", labelpad=13)
        colorbar.outline.set_linewidth(0.6)
        files.extend(_save_figure(
            fig, output, f"expert_communication_{role}_heatmap", dpi))
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    image = None
    for ax, role, matrix in zip(axes, ROLES, matrices):
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
        _draw_cells(ax, image, matrix, annotate)
        ax.set_title(ROLE_LABELS[role], fontsize=8, fontweight="semibold", pad=6)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("Effective communication coefficient",
                       rotation=270, rotation_mode="anchor", labelpad=13)
    colorbar.outline.set_linewidth(0.6)
    files.extend(_save_figure(
        fig, output, "expert_communication_encoder_first_last_heatmap", dpi))
    plt.close(fig)
    return files, limit


def generate_outputs(checkpoint, output_dir=None, vmax=None, dpi=600, annotate=True):
    checkpoint = Path(checkpoint).resolve(strict=True)
    model, metadata, kind = load_forecaster_for_communication(checkpoint)
    values = extract_encoder_communication_matrices(model)
    output = (Path(output_dir).resolve() if output_dir else
              checkpoint.parent / "expert_analysis" / "communication_matrices")
    output.mkdir(parents=True, exist_ok=True)
    files = []
    npz_values = {}
    for role in ROLES:
        matrix_path = output / f"expert_communication_{role}_matrix.csv"
        edge_path = output / f"expert_communication_{role}_edges.csv"
        npy_path = output / f"expert_communication_{role}_matrix.npy"
        _write_matrix_csv(matrix_path, values[role]["effective_matrix"])
        _write_edge_csv(edge_path, values[role])
        np.save(npy_path, values[role]["effective_matrix"])
        files.extend((matrix_path, edge_path, npy_path))
        for name in ("raw_parameter", "bounded_parameter", "effective_matrix",
                     "per_selected_message_coefficient"):
            npz_values[f"{role}_{name}"] = values[role][name]
    archive = output / "expert_communication_encoder_first_last.npz"
    np.savez(archive, **npz_values)
    files.append(archive)
    figure_files, color_limit = plot_communication_matrices(
        values, output, vmax=vmax, dpi=dpi, annotate=annotate)
    files.extend(figure_files)
    metadata_path = output / "expert_communication_encoder_metadata.json"
    metadata_report = {
        "checkpoint": str(checkpoint),
        "checkpoint_kind": kind,
        "architecture": metadata["architecture"],
        "roles": {
            role: {
                "module": values[role]["module"],
                "num_routed_experts": values[role]["num_routed_experts"],
                "num_active_experts": values[role]["num_active_experts"],
            }
            for role in ROLES
        },
        "matrix_definition": {
            "row": "receiver routed expert i",
            "column": "sender routed expert j",
            "raw_parameter": "learned M_ij from communication.mix",
            "bounded_parameter": "tanh(M_ij)",
            "effective_matrix": "(1-delta_ij)*tanh(M_ij); plotted matrix",
            "per_selected_message_coefficient": (
                "effective_matrix/(top_k-1), the coefficient applied to each "
                "selected sender hidden state before residual addition"),
            "selection": "only the token-specific Top-k by Top-k submatrix executes",
        },
        "color_scale": {"cmap": "RdBu_r", "vmin": -color_limit,
                        "vcenter": 0.0, "vmax": color_limit,
                        "shared_between_layers": True},
    }
    metadata_path.write_text(
        json.dumps(metadata_report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    files.append(metadata_path)
    return files


def main(argv=None):
    args = parser().parse_args(argv)
    if args.vmax is not None and args.vmax <= 0:
        raise ValueError("--vmax must be positive")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    files = generate_outputs(
        args.checkpoint, args.output_dir, args.vmax, args.dpi,
        annotate=not args.no_annotate)
    print(json.dumps({"event": "encoder_communication_matrices_complete",
                      "files": [str(path) for path in files]}), flush=True)
    return files


if __name__ == "__main__":
    main()
