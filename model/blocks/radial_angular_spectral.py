"""Co-DINO's learnable radial-angular Fourier operator.

Four undirected axes at 0, 45, 90 and 135 degrees and four ordered radial
bands form sixteen smooth sectors. Sector widths and radial centers are
learned, and each sector owns an independent complex channel mixer.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .spectral_filters import radial_frequency_grid


def _inverse_softplus(value):
    return math.log(math.expm1(value))


class RadialAngularSpectral(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        self.channels = channels
        self.fft_norm = config.fft_norm
        self.num_bands = config.spectral_radial_bands
        self.num_angles = config.spectral_num_angles
        scale = config.spectral_mixer_init_scale
        mixer_shape = (self.num_bands, self.num_angles, channels, channels)
        self.weight_real = nn.Parameter(scale * torch.randn(*mixer_shape))
        self.weight_imag = nn.Parameter(scale * torch.randn(*mixer_shape))
        # Equal positive spacings initially give centers at 0, 1/(B-1), ..., 1.
        self.radial_spacing_raw = nn.Parameter(torch.zeros(self.num_bands - 1))
        initial_radial_width = 0.9 / (self.num_bands - 1)
        self.radial_width_raw = nn.Parameter(torch.full(
            (self.num_bands,), _inverse_softplus(initial_radial_width - 1e-3)))
        initial_angle_width = math.radians(config.spectral_angular_width_deg)
        self.angular_width_raw = nn.Parameter(torch.full(
            (self.num_angles,), _inverse_softplus(initial_angle_width - 1e-3)))
        self.register_buffer("angle_centers", torch.tensor(
            config.spectral_angle_centers_deg, dtype=torch.float32) * math.pi / 180.0)

    def frequency_masks(self, height, width, *, device=None, dtype=torch.float32):
        """Return normalized [B,D,H,W//2+1] responses on the rFFT grid."""
        radius = radial_frequency_grid(height, width, device=device, dtype=dtype)
        spacings = F.softplus(self.radial_spacing_raw.to(dtype)) + 1e-4
        centers = torch.cat((spacings.new_zeros(1), spacings.cumsum(0) / spacings.sum()))
        radial_width = F.softplus(self.radial_width_raw.to(dtype)) + 1e-3
        radial_logits = -0.5 * ((radius[None] - centers[:, None, None])
                                / radial_width[:, None, None]).square()
        radial = radial_logits.softmax(dim=0)

        # atan2 must use signed ky; abs(fftfreq) is valid for radius only.
        ky = torch.fft.fftfreq(height, device=device, dtype=dtype)[:, None]
        kx = torch.fft.rfftfreq(width, device=device, dtype=dtype)[None, :]
        angle = torch.atan2(ky, kx)
        difference = angle[None] - self.angle_centers.to(dtype)[:, None, None]
        # Axis orientations have period pi. Wrap to [-pi/2, pi/2].
        difference = 0.5 * torch.atan2(torch.sin(2 * difference),
                                        torch.cos(2 * difference))
        angular_width = F.softplus(self.angular_width_raw.to(dtype)) + 1e-3
        angular_logits = -0.5 * (difference / angular_width[:, None, None]).square()
        angular = angular_logits.softmax(dim=0)
        # DC has no orientation; assigning four equal responses avoids an arbitrary bias.
        angular = torch.where(radius[None] == 0, angular.new_full((), 1 / self.num_angles), angular)
        return radial[:, None] * angular[None]

    @staticmethod
    def _real_boundary_projection(spectrum, height, width):
        """Enforce Hermitian constraints on rFFT's self-conjugate columns."""
        projected = spectrum.clone()
        reflected_rows = (-torch.arange(height, device=spectrum.device)) % height
        columns = (0, width // 2) if width % 2 == 0 else (0,)
        for column in columns:
            boundary = spectrum[..., column]
            reflected = boundary.index_select(-1, reflected_rows).conj()
            projected[..., column] = (boundary + reflected) * 0.5
        return projected

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"Expected BCHW with C={self.channels}")
        if not x.is_floating_point():
            raise TypeError("Fourier layer expects real floating-point input")
        height, width = x.shape[-2:]
        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft2(x.to(work_dtype), norm=self.fft_norm)
            masks = self.frequency_masks(height, width, device=x.device, dtype=work_dtype)
            weights = torch.complex(self.weight_real.to(work_dtype), self.weight_imag.to(work_dtype))
            mixed = torch.einsum("bdoc,nchw->nbdohw", weights, spectrum)
            correction = (mixed * masks[None, :, :, None]).sum(dim=(1, 2))
            correction = self._real_boundary_projection(correction, height, width)
            result = torch.fft.irfft2(correction, s=(height, width), norm=self.fft_norm)
        return result.to(x.dtype)
