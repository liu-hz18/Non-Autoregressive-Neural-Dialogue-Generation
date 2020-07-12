import torch
from torch import nn
from .feedforward import FeedForward


class HighwayBlock(nn.Module):
    def __init__(self, layer, d_model, dropout=0., rezero=False):
        super(HighwayBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout)
        self.rezero = rezero
        if rezero:
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.layernorm = nn.LayerNorm(d_model)
            self.gate = FeedForward(d_model, 1)

    def forward(self, x, *param, **kwargs):
        if self.rezero:
            return x + self.dropout(self.layer(*param, **kwargs)) * self.alpha  # back to residual
        else:
            g = torch.sigmoid(self.gate(x))
            return self.layernorm(x * g + self.dropout(self.layer(*param, **kwargs)) * (1 - g))
