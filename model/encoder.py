"""The spatial Encoder used identically by pretraining and forecasting."""

from dataclasses import dataclass
from torch import nn

from .blocks import CoDINOOperatorLayer
from .patch_ops import PatchEmbedding, PatchMerging


@dataclass
class EncoderOutput:
    stage_features: tuple
    original_shape: tuple

    @property
    def last_hidden_state(self):
        return self.stage_features[-1]

    @property
    def spatial_shapes(self):
        return tuple(feature.shape[1:3] for feature in self.stage_features)


class CoDINOEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = PatchEmbedding(config)
        self.stages = nn.ModuleList([
            nn.Sequential(*[CoDINOOperatorLayer(config, c) for _ in range(depth)])
            for c, depth in zip(config.stage_dims, config.depths)
        ])
        self.merging = nn.ModuleList([PatchMerging(c) for c in config.stage_dims[:-1]])

    def forward(self, x):
        original_shape = tuple(x.shape[-2:])
        x = self.embedding(x)
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
            if i < len(self.merging):
                x = self.merging[i](x)
        return EncoderOutput(tuple(features), original_shape)
