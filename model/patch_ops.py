"""Co-DINO patch operations, including odd and non-square grid padding."""

import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.projection = nn.Conv2d(config.input_channels, config.embed_dim,
                                    config.patch_size, stride=config.patch_size)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        p = self.patch_size
        x = F.pad(x, (0, (-w) % p, 0, (-h) % p))
        return self.norm(self.projection(x).permute(0, 2, 3, 1))


class PatchMerging(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(4 * channels)
        self.projection = nn.Linear(4 * channels, 2 * channels, bias=False)

    def forward(self, x):
        _, h, w, _ = x.shape
        x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x = torch.cat([x[:, 0::2, 0::2], x[:, 0::2, 1::2],
                       x[:, 1::2, 0::2], x[:, 1::2, 1::2]], dim=-1)
        return self.projection(self.norm(x))


class PatchUnmerging(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.channels = output_channels
        self.norm = nn.LayerNorm(2 * output_channels)
        self.projection = nn.Linear(2 * output_channels, 4 * output_channels)

    def forward(self, x, target_shape):
        b, h, w, _ = x.shape
        x = self.projection(self.norm(x)).reshape(b, h, w, 2, 2, self.channels)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, 2*h, 2*w, self.channels)
        return x[:, :target_shape[0], :target_shape[1]]


class PatchRecovery(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size, self.channels = config.patch_size, config.output_channels
        self.projection = nn.Linear(config.embed_dim, self.patch_size**2 * self.channels)

    def forward(self, x, original_shape):
        b, h, w, _ = x.shape
        p, c = self.patch_size, self.channels
        x = self.projection(x).reshape(b, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, h*p, w*p)
        return x[:, :, :original_shape[0], :original_shape[1]]
