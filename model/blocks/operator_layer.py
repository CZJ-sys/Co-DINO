"""One Co-DINO layer: directional Fourier, local and communication branches."""

from torch import nn

from .local_branch import LocalDepthwiseBranch
from .radial_angular_spectral import RadialAngularSpectral
from .sparse_moe import SparseCommunicationMoE


class CoDINOOperatorLayer(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        self.channels = channels
        self.norm_input = nn.LayerNorm(channels)
        self.norm_update = nn.LayerNorm(channels)
        self.norm_ffn = nn.LayerNorm(channels)
        self.spectral = RadialAngularSpectral(config, channels)
        self.local = LocalDepthwiseBranch(channels)
        self.moe = SparseCommunicationMoE(config, channels)
        hidden_dim = channels * config.mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, channels))

    def forward(self, x):
        if x.ndim != 4 or x.shape[-1] != self.channels:
            raise ValueError(f"Expected BHWC with C={self.channels}")
        normalized = self.norm_input(x)
        channel_first = normalized.permute(0, 3, 1, 2).contiguous()
        field_update = self.spectral(channel_first) + self.local(channel_first)
        update = field_update.permute(0, 2, 3, 1) + self.moe(normalized)
        x = x + self.norm_update(update)
        return x + self.ffn(self.norm_ffn(x))
