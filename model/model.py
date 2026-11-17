"""Full Co-DINO encoder-decoder forecasting model."""

from torch import nn

from .encoder import CoDINOEncoder
from .decoder import CoDINODecoder


class CoDINOForForecasting(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = CoDINOEncoder(config)
        self.decoder = CoDINODecoder(config)

    def forward(self, x, time):
        return self.decoder(self.encoder(x), time)

