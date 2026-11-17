"""Independent pointwise expert maps for shared and routed expert banks."""

from torch import nn


class PointwiseExpert(nn.Module):
    def __init__(self, channels, hidden_dim):
        super().__init__()
        self.input_projection = nn.Linear(channels, hidden_dim)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(hidden_dim, channels)

    def encode(self, x):
        return self.activation(self.input_projection(x))

    def decode(self, hidden):
        return self.output_projection(hidden)

    def forward(self, x):
        return self.decode(self.encode(x))
