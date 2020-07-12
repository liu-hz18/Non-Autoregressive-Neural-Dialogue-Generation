import math
import torch
from torch import nn
from torch.nn import functional as F

from .relative_position import RelativePosition
from .operators import GumbelSoftmax


class MultiHeadAttention(nn.Module):
    '''MultiHeadAttention with relative-position-encoding'''
    def __init__(self, d_q, d_k, d_v, d_key, d_value, nhead=8, dropout=0.1,
                 activation=F.relu, relative_clip=0, gumbels=False, bias=True,
                 device=None, use_wo=True):
        super(MultiHeadAttention, self).__init__()
        self.nhead = nhead
        self.head_dim_key = d_key // nhead
        self.head_dim_value = d_value // nhead
        self.d_key = d_key
        self.d_value = d_value
        self.relative_clip = relative_clip
        assert self.head_dim_key * nhead == d_key, "d_key must be divisible by nhead"
        assert self.head_dim_value * nhead == d_value, "d_value must be divisible by nhead"
        self.activation = activation
        self.use_wo = use_wo
        if relative_clip > 0:
            self.relative_position_k = RelativePosition(
                self.head_dim_key, relative_clip, device)
            self.relative_position_v = RelativePosition(
                self.head_dim_value, relative_clip, device)
        self.w_q = nn.Linear(d_q, d_key, bias)
        self.w_k = nn.Linear(d_k, d_key, bias)
        self.w_v = nn.Linear(d_v, d_value, bias)
        if gumbels:
            self.soft_max = GumbelSoftmax(dim=-1)
        else:
            self.soft_max = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if use_wo:
            self.w_o = nn.Linear(d_value, d_key, bias)
        self.reset_parameters()

    def reset_parameters(self):
        inv_sqrt2 = 1 / math.sqrt(2)
        nn.init.xavier_uniform_(self.w_q.weight, gain=inv_sqrt2)
        nn.init.xavier_uniform_(self.w_k.weight, gain=inv_sqrt2)
        nn.init.xavier_uniform_(self.w_v.weight, gain=inv_sqrt2)
        nn.init.xavier_uniform_(self.w_o.weight)
        if self.w_o.bias is not None:
            nn.init.constant_(self.w_o.bias, 0.)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        return x.reshape(batch_size, -1, self.nhead, in_feature//self.nhead)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size*self.nhead, -1, in_feature//self.nhead)

    def _reshape_from_batches(self, x):
        batch_mul_nhead, seq_len, in_feature_div_nhead = x.size()
        return x.reshape(batch_mul_nhead//self.nhead, self.nhead, -1, in_feature_div_nhead)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_mul_nhead//self.nhead, -1, in_feature_div_nhead*self.nhead)

    def _scaled_dot_product_attn(self, q, k, v, mask=None):
        scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(self.d_key)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.soft_max(scores)
        attn = self.dropout(attn)
        return attn.matmul(v)

    def _relative_attn(self, q, k, v, mask=None):
        _, length_q, _ = q.size()
        _, length_k, _ = k.size()  # (B x L x Da)
        _, length_v, _ = v.size()  # (B x L x Da)
        r_k = self.relative_position_k(length_q, length_k)  # (L x L x Da)
        r_v = self.relative_position_v(length_q, length_v)  # (L x L x Da)
        relative = q.unsqueeze(2).matmul(r_k.transpose(1, 2)).squeeze(2)
        dot = q.matmul(k.transpose(-2, -1))
        scores = (relative + dot) / math.sqrt(self.d_key)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.soft_max(scores)  # (nhead*bz) x src_length x tgt_length
        attn = self.dropout(attn)
        attn = attn.matmul(v) + attn.unsqueeze(3).mul(r_v).sum(dim=2)
        return attn

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        '''
        :q: B x L_q x E (tgt)
        :k: B x L_k x E (src)
        :v: B x L_v x E (src)
            assert L_k == L_v
        :attn_mask: L_q x L_k, Tensor(bool), (tgt x src)
        :key_padding_mask: B x L_k, Tensor(bool)
            value `0` is masked !!!
        :output: B x L x E
        '''
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        if self.activation is not None:
            q, k, v = self.activation(q), self.activation(k), self.activation(v)
        q, k, v = self._reshape_to_batches(q), self._reshape_to_batches(k), self._reshape_to_batches(v)
        # create mask
        mask = None
        if attn_mask is not None:
            mask = attn_mask = attn_mask.repeat(q.shape[0], 1, 1)  # BxN x L_q x L_k
        if key_padding_mask is not None:
            mask = key_padding_mask = key_padding_mask.unsqueeze(1).repeat(self.nhead, q.shape[1], 1)
        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask & key_padding_mask
        if self.relative_clip > 0:
            y = self._relative_attn(q, k, v, mask)
        else:
            y = self._scaled_dot_product_attn(q, k, v, mask)
        y = self._reshape_from_batches(y)
        if self.use_wo:
            y = self.w_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y
