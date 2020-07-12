
from torch import nn
from torch.nn import functional as F

from .residual_block import ResidualBlock
from .multihead_attention import MultiHeadAttention
from .highway import HighwayBlock
from .feedforward import FeedForward
from ..utils import generate_key_padding_mask


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_src, nhead, dropout=0.1, activation="relu", postnorm=True,
                 dim_feedforward=2048, relative_clip=4, use_wo=True, rezero=False,
                 gumbels=False, device=None, use_vocab_attn=False, highway=False):
        super(TransformerEncoderLayer, self).__init__()
        self.use_vocab_attn = use_vocab_attn
        self.d_src = d_src
        '''
        if self.use_vocab_attn:
            self.vocab_attn_layer = ResidualBlock(
                VocabularyAttention(d_src, gumbels=gumbels, dropout=dropout),
                d_src, dropout=0.0, rezero=rezero, postnorm=postnorm)
        '''

        self.self_attn = ResidualBlock(
            MultiHeadAttention(d_src, d_src, d_src, d_src, d_src, nhead,
                               dropout=dropout, bias=True, gumbels=gumbels,
                               relative_clip=relative_clip, device=device,
                               use_wo=use_wo),
            d_src, dropout, rezero=rezero, postnorm=postnorm)

        if highway:
            self.feedforward = HighwayBlock(
                FeedForward(d_src, dim_feedforward, dropout=dropout),
                d_src, dropout, rezero=rezero)
        else:
            self.feedforward = ResidualBlock(
                FeedForward(d_src, dim_feedforward, dropout=dropout),
                d_src, dropout, rezero=rezero, postnorm=postnorm)

    def forward(self, src, embedding=None, src_mask=None, src_key_padding_mask=None):
        '''
        :attn_mask: L_q x L_k, Tensor(bool)
        :key_padding_mask: B x L_k, Tensor(bool)
            value `0` is masked !!!
        '''
        '''
        if self.use_vocab_attn and embedding is not None:
            src = self.vocab_attn_layer(src, src, embedding)  # B x l_src x d_src
        '''
        self_attn_out = self.self_attn(
            src, src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        out = self.feedforward(self_attn_out, self_attn_out)
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, nlayers=6, dropout=0.):
        super(TransformerEncoder, self).__init__()
        self.nlayers = nlayers
        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = nn.ModuleList([
            encoder_layer for i in range(nlayers)])
        self.layernorm = nn.LayerNorm(encoder_layer.d_src)

    def forward(self, src, embedding=None, src_mask=None, src_lengths=None,
                src_key_padding_mask=None):
        if src_lengths is not None and src_key_padding_mask is None:
            src_key_padding_mask = generate_key_padding_mask(src.shape[1], src_lengths)
        x = self.dropout(src)
        xs = []
        for layer in self.transformer_encoder:
            x = layer(x, embedding=embedding, src_mask=src_mask,
                      src_key_padding_mask=src_key_padding_mask)
            xs.append(x)
        x = self.layernorm(x)
        return xs, x
