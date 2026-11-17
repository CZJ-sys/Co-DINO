"""Masked forecasting metrics for Co-DINO's four-channel Poseidon contract."""

import math

import torch

from .data import CHANNEL_CONTRACT


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


class ErrorAccumulator:
    """Aggregate errors over valid pixels; NS rho/p never enter any metric."""

    def __init__(self, metric_space="normalized", dataset_name=None,
                 normalization_contract=None):
        if metric_space not in ("normalized", "physical"):
            raise ValueError("metric_space must be normalized or physical")
        self.metric_space = metric_space
        self.dataset_name = dataset_name
        contract = normalization_contract or {}
        self.physical_mean = tuple(contract.get("mean", CHANNEL_CONTRACT["mean"]))
        self.physical_std = tuple(contract.get("std", CHANNEL_CONTRACT["std"]))
        self.pressure_offset = float(contract.get(
            "pressure_offset", CHANNEL_CONTRACT["ce_pressure_offsets"].get(dataset_name, 0.0)))
        if len(self.physical_mean) != 4 or len(self.physical_std) != 4:
            raise ValueError("normalization_contract mean/std must each contain four values")
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.target_abs_sum = 0.0
        self.target_sq_sum = 0.0
        self.relative_l1_sum = 0.0
        self.relative_l2_sum = 0.0
        self.relative_l1_values = []
        self.channel_relative_l1_values = [[] for _ in range(4)]
        self.channel_abs_sum = [0.0] * 4
        self.channel_count = [0] * 4
        self.elements = 0
        self.samples = 0

    def update(self, prediction, target, ignore_mask):
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 4:
            raise ValueError("Prediction and target must have matching B,4,H,W shapes")
        if ignore_mask.shape != prediction.shape[:2] or ignore_mask.dtype != torch.bool:
            raise ValueError("ignore_mask must be a boolean B,4 tensor")
        ignored = ignore_mask[:, :, None, None]
        if (not torch.isfinite(prediction.masked_fill(ignored, 0)).all()
                or not torch.isfinite(target.masked_fill(ignored, 0)).all()):
            raise FloatingPointError("Evaluation received a non-finite value in a supervised channel")
        prediction, target = prediction.float(), target.float()
        if self.metric_space == "physical":
            std = prediction.new_tensor(self.physical_std)[None, :, None, None]
            mean = prediction.new_tensor(self.physical_mean)[None, :, None, None]
            prediction, target = prediction * std + mean, target * std + mean
            if self.pressure_offset:
                prediction = prediction.clone()
                target = target.clone()
                prediction[:, 3] += self.pressure_offset
                target[:, 3] += self.pressure_offset
        valid = ~ignored
        # Zero the err *before* computing joint, relative and per-channel metrics.
        err = (prediction - target).masked_fill(~valid, 0)
        valid_target = target.masked_fill(~valid, 0)
        abs_err = err.abs()
        sq_err = err.square()
        self.abs_sum += abs_err.sum().item()
        self.sq_sum += sq_err.sum().item()
        self.target_abs_sum += valid_target.abs().sum().item()
        self.target_sq_sum += valid_target.square().sum().item()
        relative_l1 = (abs_err.flatten(1).sum(1) /
                       valid_target.abs().flatten(1).sum(1).clamp_min(1e-10))
        self.relative_l1_sum += relative_l1.sum().item()
        self.relative_l1_values.extend(relative_l1.detach().cpu().tolist())
        self.relative_l2_sum += (sq_err.flatten(1).sum(1).sqrt() /
                                 valid_target.square().flatten(1).sum(1).sqrt().clamp_min(1e-12)).sum().item()
        spatial = prediction.shape[-2] * prediction.shape[-1]
        for channel in range(4):
            self.channel_abs_sum[channel] += abs_err[:, channel].sum().item()
            self.channel_count[channel] += int(valid[:, channel].sum().item()) * spatial
            if valid[:, channel].any():
                channel_relative_l1 = (abs_err[:, channel].flatten(1).sum(1) /
                    valid_target[:, channel].abs().flatten(1).sum(1).clamp_min(1e-10))
                self.channel_relative_l1_values[channel].extend(
                    channel_relative_l1[valid[:, channel, 0, 0]].detach().cpu().tolist())
        self.elements += int(valid.sum().item()) * spatial
        self.samples += prediction.shape[0]

    def result(self):
        if not self.elements or not self.samples:
            raise ValueError("No valid evaluation elements")
        names = CHANNEL_CONTRACT["channels"]
        channels = [i for i, count in enumerate(self.channel_count) if count]
        mean_qoi_percent = [100 * sum(self.channel_relative_l1_values[i]) /
                            len(self.channel_relative_l1_values[i]) for i in channels]
        median_qoi_percent = [100 * _median(self.channel_relative_l1_values[i]) for i in channels]
        joint_mean_percent = 100 * self.relative_l1_sum / self.samples
        return {
            "metric_space": self.metric_space,
            "valid_channels": channels,
            "eval_channels": channels,
            "masked_channels": [i for i, count in enumerate(self.channel_count) if not count],
            "samples": self.samples,
            "valid_elements": self.elements,
            "mae": self.abs_sum / self.elements,
            "rmse": math.sqrt(self.sq_sum / self.elements),
            "relative_l1": self.relative_l1_sum / self.samples,
            "relative_l1_percent": joint_mean_percent,
            "relative_l2": self.relative_l2_sum / self.samples,
            "joint": {"mean_relative_l1_percent": joint_mean_percent,
                      "median_relative_l1_percent": 100 * _median(self.relative_l1_values)},
            "per_qoi": {"mean_relative_l1_percent": mean_qoi_percent,
                        "median_relative_l1_percent": median_qoi_percent,
                        "macro_mean_relative_l1_percent": sum(mean_qoi_percent) / len(channels),
                        "macro_median_relative_l1_percent": sum(median_qoi_percent) / len(channels)},
            "per_channel_mae": {name: (self.channel_abs_sum[i] / self.channel_count[i]
                                       if self.channel_count[i] else None)
                                for i, name in enumerate(names)},
        }


class DualSpaceErrors:
    """Emit normalized and physical summaries from one model pass."""

    def __init__(self, dataset_name, selected_space="normalized",
                 normalization_contract=None):
        self.selected_space = selected_space
        self.normalized = ErrorAccumulator(
            "normalized", dataset_name, normalization_contract)
        self.physical = ErrorAccumulator(
            "physical", dataset_name, normalization_contract)

    def update(self, prediction, target, ignore_mask):
        self.normalized.update(prediction, target, ignore_mask)
        self.physical.update(prediction, target, ignore_mask)

    def result(self):
        normalized = self.normalized.result()
        physical = self.physical.result()
        selected = normalized if self.selected_space == "normalized" else physical
        return {**selected, "metric_spaces": {"native_normalized": normalized,
                                               "physical": physical}}
