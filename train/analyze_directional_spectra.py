"""Analyze direction-conditioned rFFT energy in raw Poseidon state samples."""

import argparse
import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np


DATASETS = ("NS-Gauss", "NS-Sines", "CE-RP", "CE-CRP", "CE-Gauss", "CE-KH")
ANGLE_LABELS = ("0°", "45°", "90°", "135°")
FIGURE_FONT_SIZE = 10
FIGURE_TITLE_SIZE = 11
FIGURE_SUPTITLE_SIZE = 12
RAW_DIRECTIONAL_COLOR_MAX = 1.0


def _angle_labels(angle_offset_deg):
    return tuple(f"{(angle_offset_deg + index * 45.0) % 180:g}°" for index in range(4))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", required=True)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--trajectories-per-dataset", type=int, default=10)
    p.add_argument("--samples-per-trajectory", type=int, default=2)
    p.add_argument("--num-trajectories", type=int, default=200)
    p.add_argument("--validation-trajectories", type=int, default=120)
    p.add_argument("--test-trajectories", type=int, default=240)
    p.add_argument("--max-num-time-steps", type=int, default=20)
    p.add_argument("--angular-width-deg", type=float, default=22.5)
    p.add_argument("--angle-offset-deg", type=float, default=0.0)
    p.add_argument("--magnitude-percentile", type=float, default=99.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output-dir", default="analysis/directional_spectra")
    p.add_argument("--skip-sample-figures", action="store_true")
    return p


def angular_responses(height, width, angular_width_deg=22.5, angle_offset_deg=0.0):
    """Match Co-DINO's initialized four smooth angular responses on an rFFT grid."""
    if height < 1 or width < 1:
        raise ValueError("Spatial dimensions must be positive")
    if not 0.0 < angular_width_deg < 90.0:
        raise ValueError("angular-width-deg must lie between 0 and 90")
    if not 0.0 <= angle_offset_deg < 45.0:
        raise ValueError("angle-offset-deg must lie in [0, 45)")
    ky = np.fft.fftfreq(height)[:, None]
    kx = np.fft.rfftfreq(width)[None, :]
    angle = np.arctan2(ky, kx)
    centers = (np.deg2rad(angle_offset_deg) + np.arange(4) * np.pi / 4) % np.pi
    difference = angle[None] - centers[:, None, None]
    difference = 0.5 * np.arctan2(np.sin(2 * difference), np.cos(2 * difference))
    width_radians = np.deg2rad(angular_width_deg)
    logits = -0.5 * np.square(difference / width_radians)
    logits -= logits.max(axis=0, keepdims=True)
    responses = np.exp(logits)
    responses /= responses.sum(axis=0, keepdims=True)
    responses[:, 0, 0] = 0.25
    return responses, centers


def directional_spectrum(fields, angular_width_deg=22.5, angle_offset_deg=0.0):
    """Return channel-balanced rFFT magnitude and smooth-sector energy fractions."""
    values = np.asarray(fields, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("Expected channel-height-width sample")
    height, width = values.shape[-2:]
    centered = values - values.mean(axis=(-2, -1), keepdims=True)
    spectrum = np.fft.rfft2(centered, axes=(-2, -1), norm="forward")
    power = np.square(np.abs(spectrum))
    # Interior rFFT columns represent both positive and negative kx frequencies.
    conjugate_weight = np.full(width // 2 + 1, 2.0)
    conjugate_weight[0] = 1.0
    if width % 2 == 0:
        conjugate_weight[-1] = 1.0
    weighted_power = power * conjugate_weight[None, None, :]
    weighted_power[:, 0, 0] = 0.0
    totals = weighted_power.sum(axis=(-2, -1))
    valid = totals > np.finfo(np.float64).eps
    if not np.any(valid):
        raise ValueError("Every channel is spatially constant after demeaning")
    normalized_energy = weighted_power[valid] / totals[valid, None, None]
    energy_density = normalized_energy.mean(axis=0)

    magnitude_norm = np.sqrt(np.square(np.abs(spectrum[valid])).sum(axis=(-2, -1)))
    normalized_magnitude = np.abs(spectrum[valid]) / magnitude_norm[:, None, None]
    magnitude = normalized_magnitude.mean(axis=0)
    magnitude[0, 0] = 0.0

    responses, centers = angular_responses(
        height, width, angular_width_deg, angle_offset_deg)
    fractions = (responses * energy_density[None]).sum(axis=(-2, -1))
    fractions /= fractions.sum()
    return magnitude, fractions, responses, centers, int(valid.sum())


def _split_bounds(total, split, train_count, validation_count, test_count):
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("Trajectory split counts must be positive")
    if train_count + validation_count + test_count > total:
        raise ValueError("Requested train/val/test trajectory counts exceed the file")
    return {
        "train": (0, train_count),
        "val": (total - validation_count - test_count, validation_count),
        "test": (total - test_count, test_count),
    }[split]


def _read_state(dataset, trajectory, frame, name):
    channels = 2 if name.startswith("NS-") else 4
    state = np.asarray(dataset[trajectory, frame, :channels], dtype=np.float64)
    if state.ndim == 2:
        side = math.isqrt(state.shape[-1])
        if side * side != state.shape[-1]:
            raise ValueError(f"{name}: flattened fields are not square")
        state = state.reshape(channels, side, side)
    if state.ndim != 3:
        raise ValueError(f"{name}: expected C,H,W or C,HW state; got {state.shape}")
    return state


def _configure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as error:
        raise RuntimeError(
            "Directional-spectrum figures require matplotlib; install requirements.txt") from error
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 10,
        "axes.linewidth": 0.8,
    })
    cmap = LinearSegmentedColormap.from_list(
        "black_red", ("#000000", "#350000", "#8A0000", "#E31A1C", "#FF4A3D"))
    cmap.set_bad("#000000")
    cmap.set_under("#000000")
    cmap.set_over("#FF4A3D")
    return plt, cmap


def _ray_endpoint(angle_degrees):
    radians = math.radians(angle_degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    scale_x = math.inf if abs(cosine) < 1e-12 else 0.5 / abs(cosine)
    scale_y = math.inf if abs(sine) < 1e-12 else 0.5 / abs(sine)
    scale = min(scale_x, scale_y)
    return scale * cosine, scale * sine


def plot_sample(plt, cmap, magnitude, fractions, dataset, trajectory, frame,
                output_path, angular_width_deg, angle_offset_deg, percentile, dpi):
    shifted = np.fft.fftshift(magnitude, axes=0)
    positive = shifted[shifted > 0]
    if positive.size == 0:
        raise ValueError("Spectrum has no positive magnitude")
    upper = np.percentile(positive, percentile)
    lower = max(np.percentile(positive, 5.0), upper * 1e-4)
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5),
                             gridspec_kw={"width_ratios": (1.0, 0.9)},
                             constrained_layout=True)
    heatmap = axes[0].imshow(
        shifted, origin="lower", extent=(0.0, 0.5, -0.5, 0.5), aspect="equal",
        cmap=cmap, norm=LogNorm(vmin=lower, vmax=upper), interpolation="nearest")
    boundaries = (angle_offset_deg + 22.5 + np.arange(4) * 45.0) % 180.0
    display_boundaries = [value - 180.0 if value > 90.0 else value for value in boundaries]
    for angle in display_boundaries:
        x_end, y_end = _ray_endpoint(angle)
        if x_end >= -1e-12:
            axes[0].plot((0.0, x_end), (0.0, y_end), color="white",
                         linestyle="--", linewidth=0.8, alpha=0.9)
    axes[0].scatter((0.0,), (0.0,), s=10, facecolor="white", edgecolor="none", zorder=5)
    axes[0].set_xlim(0.0, 0.5)
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].set_xlabel("kx (cycles/pixel)")
    axes[0].set_ylabel("ky (cycles/pixel)")
    axes[0].set_title("Centered rFFT half-plane magnitude", fontsize=FIGURE_TITLE_SIZE)
    colorbar = fig.colorbar(heatmap, ax=axes[0], fraction=0.048, pad=0.04)
    colorbar.set_label("Channel-normalized |rFFT| (log scale)", rotation=270,
                       rotation_mode="anchor", labelpad=12)

    bars = axes[1].bar(np.arange(4), 100.0 * fractions, width=0.68,
                       color="#C51B1D", edgecolor="#6E0000", linewidth=0.6)
    axes[1].set_xticks(np.arange(4), _angle_labels(angle_offset_deg))
    axes[1].set_xlabel("Angular response center")
    axes[1].set_ylabel("Directional energy (%)")
    axes[1].set_ylim(0.0, max(35.0, 100.0 * fractions.max() * 1.25))
    axes[1].set_title(f"Smooth sectors (width {angular_width_deg:g}°)",
                      fontsize=FIGURE_TITLE_SIZE)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    axes[1].set_axisbelow(True)
    for bar, value in zip(bars, fractions):
        axes[1].text(bar.get_x() + bar.get_width() / 2, 100.0 * value + 0.8,
                     f"{100.0 * value:.1f}", ha="center", va="bottom",
                     fontsize=FIGURE_FONT_SIZE)
    fig.suptitle(f"{dataset} | trajectory {trajectory} | frame {frame}",
                 fontsize=FIGURE_SUPTITLE_SIZE, fontweight="semibold")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_summary(plt, records, output_dir, datasets, angle_offset_deg):
    matrix = np.asarray([[np.mean([row["fractions"][angle] for row in records
                                  if row["dataset"] == dataset])
                          for angle in range(4)] for dataset in datasets])
    spread = np.asarray([[np.std([row["fractions"][angle] for row in records
                                 if row["dataset"] == dataset], ddof=1)
                          for angle in range(4)] for dataset in datasets])
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Reds", vmin=0.0, vmax=RAW_DIRECTIONAL_COLOR_MAX,
                      aspect="auto")
    heatmap_label_size = FIGURE_FONT_SIZE
    ax.set_xticks(np.arange(4), _angle_labels(angle_offset_deg))
    ax.set_yticks(np.arange(len(datasets)), datasets)
    ax.tick_params(axis="both", which="major", labelsize=heatmap_label_size)
    ax.set_xlabel("Angular response center")
    ax.set_ylabel("Dataset")
    ax.set_title("Mean directional energy over sampled raw states",
                 fontsize=FIGURE_TITLE_SIZE,
                 fontweight="semibold")
    for row in range(len(datasets)):
        for column in range(4):
            red, green, blue, _ = image.cmap(image.norm(matrix[row, column]))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            ax.text(column, row,
                    f"{100.0 * matrix[row, column]:.1f}\n±{100.0 * spread[row, column]:.1f}",
                    ha="center", va="center", fontsize=heatmap_label_size,
                    color="white" if luminance < 0.52 else "#3A0000")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035)
    colorbar.set_label("Mean directional energy (%)", rotation=270,
                       rotation_mode="anchor", labelpad=14,
                       fontsize=FIGURE_FONT_SIZE)
    from matplotlib.ticker import PercentFormatter
    colorbar.set_ticks(np.linspace(0.0, RAW_DIRECTIONAL_COLOR_MAX, 5))
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    colorbar.ax.tick_params(labelsize=FIGURE_FONT_SIZE)
    svg = output_dir / "directional_energy_summary.svg"
    pdf = output_dir / "directional_energy_summary.pdf"
    png = output_dir / "directional_energy_summary.png"
    tiff = output_dir / "directional_energy_summary.tiff"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(tiff, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return matrix, spread, (svg, pdf, png, tiff)


def main(argv=None):
    args = parser().parse_args(argv)
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError("Duplicate datasets are not allowed")
    if args.trajectories_per_dataset < 1 or args.samples_per_trajectory < 1:
        raise ValueError("Trajectory and sample counts must be positive")
    if args.trajectories_per_dataset * args.samples_per_trajectory < 2:
        raise ValueError("At least two samples per dataset are required for sample SD")
    if args.seed < 0 or args.max_num_time_steps < 1 or args.dpi < 72:
        raise ValueError("seed/max-num-time-steps/dpi are invalid")
    if not 50.0 <= args.magnitude_percentile < 100.0:
        raise ValueError("magnitude-percentile must lie in [50, 100)")

    data_root = Path(args.data_path).resolve(strict=True)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt, cmap = _configure_matplotlib()
    records, manifests = [], {}
    for dataset_index, name in enumerate(args.datasets):
        path = data_root / f"{name}.nc"
        with h5py.File(path, "r") as reader:
            key = "velocity" if name.startswith("NS-") else "data"
            source = reader[key]
            total, frames = source.shape[:2]
            start, count = _split_bounds(
                total, args.split, args.num_trajectories,
                args.validation_trajectories, args.test_trajectories)
            if args.trajectories_per_dataset > count:
                raise ValueError(f"{name}: trajectories-per-dataset exceeds split size {count}")
            usable_frames = min(frames, args.max_num_time_steps + 1)
            if args.samples_per_trajectory > usable_frames:
                raise ValueError(f"{name}: samples-per-trajectory exceeds {usable_frames} frames")
            sampler = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence((args.seed, dataset_index))))
            relative_trajectories = sorted(sampler.choice(
                count, size=args.trajectories_per_dataset, replace=False).tolist())
            sampled_frames = np.rint(np.linspace(
                0, usable_frames - 1, args.samples_per_trajectory)).astype(int).tolist()
            if len(set(sampled_frames)) != len(sampled_frames):
                raise ValueError(f"{name}: frame selection contains duplicates")
            dataset_output = output / name
            dataset_output.mkdir(parents=True, exist_ok=True)
            for relative in relative_trajectories:
                trajectory = start + relative
                for frame in sampled_frames:
                    state = _read_state(source, trajectory, frame, name)
                    magnitude, fractions, _, centers, valid_channels = directional_spectrum(
                        state, args.angular_width_deg, args.angle_offset_deg)
                    figure_path = dataset_output / f"trajectory_{trajectory:04d}_frame_{frame:03d}.png"
                    if not args.skip_sample_figures:
                        plot_sample(plt, cmap, magnitude, fractions, name, trajectory, frame,
                                    figure_path, args.angular_width_deg, args.angle_offset_deg,
                                    args.magnitude_percentile, args.dpi)
                    records.append({"dataset": name, "split": args.split,
                                    "trajectory_relative": relative,
                                    "trajectory_file": trajectory, "frame": frame,
                                    "valid_channels": valid_channels,
                                    "fractions": fractions.tolist(),
                                    "figure": str(figure_path) if not args.skip_sample_figures else None})
            manifests[name] = {"file": str(path), "key": key, "shape": list(source.shape),
                               "split_start": start, "split_count": count,
                               "trajectory_relative_indices": relative_trajectories,
                               "trajectory_file_indices": [start + value
                                                           for value in relative_trajectories],
                               "frames": sampled_frames,
                               "samples": len(relative_trajectories) * len(sampled_frames)}
        print(json.dumps({"dataset": name, "samples": manifests[name]["samples"]}), flush=True)

    sample_csv = output / "directional_energy_samples.csv"
    fields = ["dataset", "split", "trajectory_relative", "trajectory_file", "frame",
              "valid_channels", "angle_0", "angle_45", "angle_90", "angle_135", "figure"]
    with sample_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in fields[:6]} | {
                f"angle_{label}": row["fractions"][index]
                for index, label in enumerate((0, 45, 90, 135))} | {"figure": row["figure"]})

    matrix, spread, figures = plot_summary(
        plt, records, output, args.datasets, args.angle_offset_deg)
    summary_csv = output / "directional_energy_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("dataset", "samples",
                         "angle_0_mean", "angle_0_sd", "angle_45_mean", "angle_45_sd",
                         "angle_90_mean", "angle_90_sd", "angle_135_mean", "angle_135_sd"))
        for dataset, means, standard_deviations in zip(args.datasets, matrix, spread):
            interleaved = [value for pair in zip(means, standard_deviations) for value in pair]
            writer.writerow((dataset, sum(row["dataset"] == dataset for row in records),
                             *interleaved))

    metadata = {
        "data_path": str(data_root), "datasets": args.datasets, "split": args.split,
        "selection": {"seed": args.seed,
                      "trajectories_per_dataset": args.trajectories_per_dataset,
                      "samples_per_trajectory": args.samples_per_trajectory,
                      "max_num_time_steps": args.max_num_time_steps},
        "angular_partition": {
            "centers_degrees": [float(np.rad2deg(value)) for value in centers],
            "initial_width_degrees": args.angular_width_deg,
            "offset_degrees": args.angle_offset_deg,
            "definition": "Co-DINO normalized Gaussian axis responses modulo 180 degrees"},
        "spectrum": {
            "input": "native raw channels from each HDF5 state",
            "preprocessing": "spatial mean removed independently from every channel",
            "fft": "numpy rfft2 with forward normalization; ky shifted only for display",
            "energy": "squared magnitude with rFFT conjugate weights; each nonconstant channel normalized before equal-channel averaging",
            "magnitude_display": "channel-normalized magnitude with per-sample logarithmic color normalization",
            "dc": "excluded because DC has no direction"},
        "figure_style": {
            "base_font_size_points": FIGURE_FONT_SIZE,
            "title_font_size_points": FIGURE_TITLE_SIZE,
            "summary_color_scale_fraction": [0.0, RAW_DIRECTIONAL_COLOR_MAX],
            "summary_color_scale_percent": [0.0, 100.0],
        },
        "dataset_manifests": manifests,
        "outputs": [str(sample_csv), str(summary_csv), *(str(path) for path in figures)],
    }
    metadata_path = output / "directional_spectrum_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "directional_spectrum_analysis_complete",
                      "output_dir": str(output), "samples": len(records)}), flush=True)
    return metadata


if __name__ == "__main__":
    main()
