"""Forecasting objectives for masked four-channel PDE states."""

import torch


def forecasting_relative_l1(prediction, target, ignore_mask, eps=1e-10, reduction="mean"):
    """Per-sample joint-channel Relative-L1 over valid spatial elements.

    NS density/pressure channels are removed before both the numerator and the
    denominator are formed.  This matches the field-norm convention used by
    Co-DINO evaluation rather than averaging unstable pointwise relative errors.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Prediction and target must have matching B,C,H,W shapes")
    if ignore_mask.shape != prediction.shape[:2] or ignore_mask.dtype != torch.bool:
        raise ValueError("ignore_mask must be a boolean B,C tensor")
    if reduction not in ("mean", "none"):
        raise ValueError("reduction must be mean or none")
    valid = ~ignore_mask[:, :, None, None]
    if not valid.any():
        raise ValueError("No valid supervised elements")
    difference = (prediction-target).abs().masked_fill(~valid, 0)
    valid_target = target.abs().masked_fill(~valid, 0)
    ratios = (difference.flatten(1).sum(1) /
              valid_target.flatten(1).sum(1).clamp_min(eps))
    return ratios.mean() if reduction == "mean" else ratios
