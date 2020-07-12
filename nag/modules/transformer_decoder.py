from torch import nn
from torch.nn import functional as F

from .residual_block import ResidualBlock
from .multihead_attention import MultiHeadAttention
from .highway import HighwayBlock
from .feedforward import FeedForward
from .vocabulary_attention import VocabularyAttention
from .sinusoidal_position_embedding import PositionalEncoding
from ..utils import generate_key_padding_mask


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_tar, d_src, nhead, gumbels=False, rezero=False,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 max_sent_length=64, relative_clip=4, device=None, use_wo=True,
                 use_pos_attn=False, use_vocab_attn=False, highway=False,
                 postnorm=True, position_encoding_layer=None):
        super(TransformerDecoderLayer, self).__init__()
        self.use_pos_attn = use_pos_attn
        self.use_vocab_attn = use_vocab_attn
        self.d_tar = d_tar
        if use_vocab_attn:
            self.vocab_attn_layer = ResidualBlock(
                VocabularyAttention(d_tar, gumbels=gumbels, dropout=dropout),
                d_tar, dropout=0.0, rezero=rezero)

        self.self_attn = ResidualBlock(
            MultiHeadAttention(d_tar, d_tar, d_tar, d_tar, d_tar, nhead,
                               dropout=dropout, bias=True, gumbels=gumbels,
                               relative_clip=relative_clip, device=device,
                               use_wo=use_wo),
            d_tar, dropout, rezero=rezero, postnorm=postnorm)

        if use_pos_attn and position_encoding_layer is not None:
            self.position_encoding_layer = position_encoding_layer
            self.pos_selfattn = ResidualBlock(
                MultiHeadAttention(d_tar, d_tar, d_tar, d_tar, d_tar, nhead,
                                   dropout=dropout, bias=True, gumbels=gumbels,
                                   relative_clip=relative_clip, device=device,
                                   use_wo=use_wo),
                d_tar, dropout, rezero=rezero, postnorm=postnorm)

        self.src_attn = ResidualBlock(
            MultiHeadAttention(d_tar, d_src, d_src, d_tar, d_tar, nhead,
                               dropout=dropout, bias=True, gumbels=gumbels,
                               relative_clip=relative_clip, device=device,
                               use_wo=use_wo),
            d_tar, dropout, rezero=rezero, postnorm=postnorm)

        self.d_tar = d_tar
        self.max_sent_length = max_sent_length
        if highway:
            self.feedforward = HighwayBlock(
                FeedForward(d_tar, dim_feedforward, dropout=dropout),
                d_tar, dropout=dropout, rezero=rezero)
        else:
            self.feedforward = ResidualBlock(
                FeedForward(d_tar, dim_feedforward, dropout=dropout),
                d_tar, dropout=dropout, rezero=rezero, postnorm=postnorm)

    def forward(self, tgt, src, embedding=None, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None):
        if self.use_vocab_attn and embedding is not None:
            tgt = self.vocab_attn_layer(tgt, tgt, embedding)  # B x l_tar x d_tar
        self_attn_out = self.self_attn(
            tgt, tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)  # B x l_tar x d_tar
        if self.use_pos_attn:
            pos_encoding_out = self.position_encoding_layer(self_attn_out)
            self_attn_out = self.pos_selfattn(
                self_attn_out, pos_encoding_out, pos_encoding_out, self_attn_out,
                attn_mask=None, key_padding_mask=tgt_key_padding_mask)  # B x l_tar x d_tar
            src_attn_out = self.src_attn(
                self_attn_out, self_attn_out, src, src, attn_mask=memory_mask)  # B x l_tar x d_tar
        else:
            src_attn_out = self.src_attn(
                self_attn_out, self_attn_out, src, src, attn_mask=memory_mask)  # B x l_tar x d_tar
        out = self.feedforward(src_attn_out, src_attn_out)  # B x l_tar x d_tar
        return out  # B x l_tar x d_tar


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, nlayers=6, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.nlayers = nlayers
        self.dropout = nn.Dropout(dropout)
        self.transformer_decoder = nn.ModuleList([
            decoder_layer for i in range(nlayers)])
        self.layernorm = nn.LayerNorm(decoder_layer.d_tar)

    def forward(self, tgt, memory, embedding=None, tgt_mask=None, memory_mask=None,
                tgt_lengths=None, tgt_key_padding_mask=None):
        x = self.dropout(tgt)
        if tgt_lengths is not None and tgt_key_padding_mask is None:
            tgt_key_padding_mask = generate_key_padding_mask(
                tgt.shape[1], tgt_lengths)
        xs = []
        for layer in self.transformer_decoder:
            x = layer(x, memory, embedding=embedding, tgt_mask=tgt_mask,
                      memory_mask=memory_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask)
            xs.append(x)
        x = self.layernorm(x)
        return xs, x
