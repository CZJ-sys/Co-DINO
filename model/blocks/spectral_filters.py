"""Co-DINO frequency grids and differentiable filters.

Grid construction requires the original spatial width; rFFT output width alone
cannot distinguish odd and even spatial grids. No sibling repository imports.
"""

import math

import torch


def radial_frequency_grid(height, spatial_width, *, device=None, dtype=torch.float32):
    """Return [H, W//2+1] radii, normalized by the diagonal Nyquist radius."""
    if height <= 0 or spatial_width <= 0:
        raise ValueError("Spatial dimensions must be positive")
    fy = torch.fft.fftfreq(height, device=device, dtype=dtype).abs() / 0.5
    fx = torch.fft.rfftfreq(spatial_width, device=device, dtype=dtype) / 0.5
    return torch.sqrt(fy[:, None].square() + fx[None, :].square()) / math.sqrt(2.0)


def gaussian_low_pass_response(radius, cutoff):
    cutoff = torch.as_tensor(cutoff, device=radius.device, dtype=radius.dtype).clamp_min(1e-6)
    return torch.exp(-0.5 * (radius / cutoff).square())


def complementary_gaussian_responses(radius, cutoff):
    low = gaussian_low_pass_response(radius, cutoff)
    return low, 1.0 - low
