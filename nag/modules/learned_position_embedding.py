import torch
from torch import nn


class LearnedPositionalEmbedding(nn.Module):
    """docstring for LearnedPositionalEmbedding"""
    def __init__(self, d_model, dropout=0.1, max_len=128, residual=True):
        super(LearnedPositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.residual = residual
        max_len = max_len + 1
        self.position_encoding = nn.Embedding(max_len, d_model)

    def forward(self, x, input_lens=None):
        max_len = x.shape[1]
        if input_lens is not None:
            input_pos = torch.LongTensor(
                [list(range(1, lenx + 1)) + [0] * (max_len - lenx)\
                 for lenx in input_lens.cpu().numpy()]).to(x.device)
        else:
            B = x.shape[0]
            input_pos = torch.LongTensor(
                [list(range(1, max_len + 1)) for _ in range(B)]).to(x.device)
        if self.residual:
            out = self.position_encoding(input_pos).squeeze(2) + x
        else:
            out = self.position_encoding(input_pos).squeeze(2)
        return self.dropout(out)
