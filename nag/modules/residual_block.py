import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, layer, d_model, dropout=0.1, rezero=False, postnorm=True):
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout)
        self.postnorm = postnorm
        self.rezero = rezero
        if rezero:
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, *param, **kwargs):
        if self.rezero:
            return x + self.dropout(self.layer(*param, **kwargs)) * self.alpha
        elif self.postnorm:
            return self.layernorm(x + self.dropout(self.layer(*param, **kwargs)))
        else:
            param = param[1:]
            return x + self.dropout(self.layer(self.layernorm(x), *(param[1:]), **kwargs))
