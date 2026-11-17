"""Local multiscale Decoder with direct skips and time conditioning."""

import torch
from torch import nn

from .blocks import CoDINOOperatorLayer
from .patch_ops import PatchUnmerging, PatchRecovery


class CoDINODecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(*[CoDINOOperatorLayer(config, c) for _ in range(depth)])
            for c, depth in zip(config.stage_dims, config.depths)
        ])
        self.skip_projection = nn.ModuleList([
            nn.Sequential(nn.Linear(2*c, c), nn.LayerNorm(c)) for c in config.stage_dims
        ])
        self.time_embedding = nn.ModuleList([nn.Linear(1, c) for c in config.stage_dims])
        self.unmerging = nn.ModuleList([PatchUnmerging(c) for c in config.stage_dims[:-1]])
        self.recovery = PatchRecovery(config)

    def forward(self, encoded, time):
        x = encoded.last_hidden_state
        time = time.reshape(x.shape[0], 1).to(x.dtype)
        for i in range(len(self.stages)-1, -1, -1):
            x = self.skip_projection[i](torch.cat([x, encoded.stage_features[i]], dim=-1))
            x = x + self.time_embedding[i](time)[:, None, None, :]
            x = self.stages[i](x)
            if i:
                x = self.unmerging[i-1](x, encoded.spatial_shapes[i-1])
        return self.recovery(x, encoded.original_shape)
