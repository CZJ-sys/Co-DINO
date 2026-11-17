"""Optional exact field dumps and visual contact sheets for Co-DINO rollouts."""

import json
from pathlib import Path

import torch

CHANNEL_NAMES = ("rho", "u", "v", "p")


class RolloutFieldSaver:
    """Save normalized ground truth and forecast fields per trajectory."""

    def __init__(self, output_dir, dataset_name, save_count, first_target_time,
                 resolution, masked_channels):
        self.output_dir = Path(output_dir).resolve() / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.save_count = int(save_count)
        self.first_target_time = int(first_target_time)
        self.resolution = resolution
        self.masked_channels = tuple(masked_channels)
        self.valid_channels = tuple(i for i in range(4) if i not in self.masked_channels)
        self.trajectories = []

    @staticmethod
    def _plt():
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise RuntimeError("Saving rollout images requires matplotlib") from error
        return plt

    def _contact_sheet(self, values, comparison, path, title):
        plt = self._plt()
        steps = values.shape[0]
        channels = self.valid_channels
        fig, axes = plt.subplots(len(channels), steps,
                                 figsize=(max(6.0, 1.05 * steps), 1.75 * len(channels)),
                                 squeeze=False, constrained_layout=True)
        for row, channel in enumerate(channels):
            common = torch.cat((values[:, channel].flatten(), comparison[:, channel].flatten()))
            vmin, vmax = float(common.min()), float(common.max())
            if vmax <= vmin:
                vmax = vmin + 1e-6
            images = []
            for step in range(steps):
                image = axes[row, step].imshow(values[step, channel].numpy(), cmap="viridis",
                                               vmin=vmin, vmax=vmax, origin="lower")
                images.append(image)
                axes[row, step].set_xticks([])
                axes[row, step].set_yticks([])
                if row == 0:
                    axes[row, step].set_title(f"t{self.first_target_time + step}", fontsize=8)
                if step == 0:
                    axes[row, step].set_ylabel(CHANNEL_NAMES[channel])
            fig.colorbar(images[-1], ax=axes[row, :].tolist(), fraction=0.012, pad=0.008)
        fig.suptitle(title)
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)

    def save(self, local_index, source_index, ground_truth, forecast):
        if local_index >= self.save_count:
            return
        ground_truth = ground_truth.detach().float().cpu().contiguous()
        forecast = forecast.detach().float().cpu().contiguous()
        if ground_truth.shape != forecast.shape or ground_truth.ndim != 4:
            raise ValueError("Saved rollout fields must have matching [steps,4,H,W] shapes")
        prefix = f"trajectory_{local_index:04d}"
        tensor_path = self.output_dir / f"{prefix}_fields.pt"
        torch.save({
            "dataset": self.dataset_name,
            "local_trajectory_index": local_index,
            "source_trajectory_index": source_index,
            "target_times": list(range(self.first_target_time,
                                       self.first_target_time + ground_truth.shape[0])),
            "space": "native_normalized",
            "masked_channels": list(self.masked_channels),
            "ground_truth": ground_truth,
            "forecast": forecast,
        }, tensor_path)
        gt_path = self.output_dir / f"{prefix}_ground_truth.png"
        forecast_path = self.output_dir / f"{prefix}_forecast.png"
        self._contact_sheet(ground_truth, forecast, gt_path,
                            f"{self.dataset_name} trajectory {local_index} - ground truth")
        self._contact_sheet(forecast, ground_truth, forecast_path,
                            f"{self.dataset_name} trajectory {local_index} - forecast")
        self.trajectories.append({
            "local_trajectory_index": local_index,
            "source_trajectory_index": source_index,
            "tensor_file": str(tensor_path),
            "ground_truth_image": str(gt_path),
            "forecast_image": str(forecast_path),
        })

    def finalize(self):
        manifest = {
            "dataset": self.dataset_name,
            "saved_trajectories": len(self.trajectories),
            "requested_trajectories": self.save_count,
            "first_target_time": self.first_target_time,
            "resolution": self.resolution,
            "space": "native_normalized",
            "masked_channels": list(self.masked_channels),
            "valid_channels": list(self.valid_channels),
            "trajectories": self.trajectories,
        }
        path = self.output_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return {"manifest": str(path), **{key: manifest[key] for key in (
            "saved_trajectories", "space", "masked_channels", "valid_channels")}}
