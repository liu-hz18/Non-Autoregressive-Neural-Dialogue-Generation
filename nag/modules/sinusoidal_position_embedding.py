import math
import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128, device=None, residual=True, requires_grad=False):
        super(PositionalEncoding, self).__init__()
        self.multier = -math.log(10000.0)
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.residual = residual
        max_len = max_len + 1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (self.multier / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.position_encoding = nn.Embedding.from_pretrained(
            pe.to(self.device), freeze=not requires_grad)
        # self.position_encoding.weight = nn.Parameter(pe.to(self.device), requires_grad=requires_grad)

    def forward(self, x, input_lens=None):
        '''
        :param x: B x L x E
        return: B x L x E
        '''
        max_len = x.shape[1]
        if input_lens is not None:
            input_pos = torch.LongTensor(
                [list(range(1, lenx + 1)) + [0] * (max_len - lenx) for lenx in input_lens.cpu().numpy()]).to(self.device)
        else:
            B = x.shape[0]
            input_pos = torch.LongTensor(
                [list(range(1, max_len + 1)) for _ in range(B)]).to(self.device)
        if self.residual:
            out = self.position_encoding(input_pos).squeeze(2) + x
        else:
            out = self.position_encoding(input_pos).squeeze(2)
        return self.dropout(out)