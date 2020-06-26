import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable, Function
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence


class GumbelSoftmax(nn.Module):
    def __init__(self, dim=None, tau=1):
        super(GumbelSoftmax, self).__init__()
        self.dim = dim
        self.tau = tau

    def forward(self, logits):
        return F.gumbel_softmax(logits, tau=self.tau, dim=self.dim)

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)


class LengthPredictor(nn.Module):
    def __init__(self, embed_size, min_value=-20, max_value=20, max_length=500,
                 tau=1, gumbels=False, device=None):
        super(LengthPredictor, self).__init__()
        self.embed_size = embed_size
        self.out_size = max_value - min_value + 1
        self.pooling_layer = nn.AdaptiveMaxPool1d(output_size=1)
        self.mlp = nn.Linear(self.embed_size, self.out_size)
        self.min_value = min_value
        self.device = device
        self.max_length = max_length
        max_length = max_length + max_value + 1
        range_vec_i = torch.arange(max_length).float().to(self.device)
        range_vec_j = torch.arange(max_length).float().to(self.device)
        distance_mat = F.softmax(-torch.abs(range_vec_i[None, :] - range_vec_j[:, None]) / tau, dim=-1)
        self.W = nn.Parameter(distance_mat)
        if gumbels:
            self.soft_max = GumbelSoftmax(dim=-1)
        else:
            self.soft_max = nn.Softmax(dim=-1)

    def forward(self, x, tgt_length=-1):
        '''
        in: B x length x embed_size
        out: B x new_length
        '''
        lengths = torch.Tensor([sentence.shape[0] for sentence in x]).long().to(self.device)
        out = self.pooling_layer(x.permute(0, 2, 1)).squeeze(2)  # out: B x embed_size
        out = self.soft_max(self.mlp(out))  # out: B x [-m, m]
        ms = torch.argmax(out, dim=1) + lengths + self.min_value   # out: B
        if tgt_length > 0:
            fix_m = tgt_length
        else:
            fix_m = torch.max(torch.clamp(ms, max=self.max_length+lengths[0], min=1).long())
        output = x.permute(0, 2, 1).matmul(self.W[:x.shape[1], :fix_m]).permute(0, 2, 1)
        return output, out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512, device=None):
        super(PositionalEncoding, self).__init__()
        self.multier = -math.log(10000.0)
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (self.multier / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.position_encoding = nn.Embedding(max_len, d_model)
        self.position_encoding.weight = nn.Parameter(pe.to(self.device), requires_grad=True)

    def forward(self, x):
        '''
        :param x: B x L x E
        return: B x L x E
        '''
        lens = [seq.shape[0] for seq in x]
        max_len = max(lens)
        input_pos = torch.LongTensor(
          [list(range(1, len + 1)) + [0] * (max_len - len) for len in lens]).to(self.device)
        out = self.position_encoding(input_pos).squeeze(2) + x
        return self.dropout(out)


class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position, device=None):
        '''
        :param num_units: d_a
        :param max_relative_position: k
        '''
        super(RelativePosition, self).__init__()
        self.num_units = num_units
        self.device = device
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        '''
        for self-att: length_q == length_k == length_x
        return: embeddings: length_q x length_k x d_a
        '''
        range_vec_q = torch.arange(length_q).to(self.device)
        range_vec_k = torch.arange(length_k).to(self.device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        final_mat = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        # final_mat = final_mat + self.max_relative_position
        embeddings = self.embeddings_table[final_mat.long()]
        return embeddings


class MultiHeadAttention(nn.Module):
    '''MultiHeadAttention with relative-position-encoding'''
    def __init__(self, d_k, d_q, d_v, d_key, d_value, nhead=8, dropout=0., bias=True,
                 activation=F.relu, relative=True, max_relative_position=4, gumbels=False,
                 device=None):
        super(MultiHeadAttention, self).__init__()
        self.nhead = nhead
        self.head_dim_key = d_key // nhead
        self.head_dim_value = d_value // nhead
        self.d_key = d_key
        self.d_value = d_value
        assert self.head_dim_key * nhead == d_key, "d_key must be divisible by nhead"
        assert self.head_dim_value * nhead == d_value, "d_value must be divisible by nhead"
        self.activation = activation
        self.relative = relative
        if relative:
            self.relative_position_k = RelativePosition(self.head_dim_key, max_relative_position, device)
            self.relative_position_v = RelativePosition(self.head_dim_value, max_relative_position, device)
        self.w_q = nn.Linear(d_q, d_key, bias)
        self.w_k = nn.Linear(d_k, d_key, bias)
        self.w_v = nn.Linear(d_v, d_value, bias)
        self.w_o = nn.Linear(d_value, d_key, bias)
        if gumbels:
            self.soft_max = GumbelSoftmax(dim=-1)
        else:
            self.soft_max = nn.Softmax(dim=-1)
        self._init_weight()

    def _reshape_to_batches(self, x):
        batch_size, seq_len, embed_dim = x.size()
        head_dim = embed_dim // self.nhead
        return x.reshape(batch_size, seq_len, self.nhead, head_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.nhead, seq_len, head_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.nhead
        return x.reshape(batch_size, self.nhead, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, in_feature*self.nhead)

    def _scaled_dot_product_attn(self, q, k, v, mask=None):
        scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(self.d_key)
        if mask is not None:
            scores = scores.mask_fill(mask == 0, -1e9)
        attn = self.soft_max(scores)
        return attn.matmul(v)

    def _relative_attn(self, q, k, v, mask=None):
        _, length_q, _ = q.size()
        _, length_k, head_dim_k = k.size()  # (B x L x Da)
        _, length_v, _ = v.size()  # (B x L x Da)
        r_k = self.relative_position_k(length_q, length_k)  # (L x L x Da)
        r_v = self.relative_position_v(length_q, length_v)  # (L x L x Da)
        relative = q.unsqueeze(2).matmul(r_k.transpose(1, 2)).squeeze(2)
        dot = (q.unsqueeze(2) * k.unsqueeze(1)).sum(-1)
        scores = (relative + dot) / math.sqrt(self.d_key)
        if mask is not None:
            scores = scores.mask_fill(mask == 0, -1e9)
        attn = self.soft_max(scores)
        attn = attn.matmul(v) + attn.unsqueeze(3).mul(r_v).sum(dim=2)
        return attn

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)

    def forward(self, q, k, v, mask=None):
        '''
        :param input: B x L x E
        :param output: B x L x E
        '''
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        if self.activation is not None:
            q, k, v = self.activation(q), self.activation(k), self.activation(v)
        q, k, v = self._reshape_to_batches(q), self._reshape_to_batches(k), self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.nhead, 1, 1)
        if self.relative:
            y = self._relative_attn(q, k, v, mask)
        else:
            y = self._scaled_dot_product_attn(q, k, v, mask)
        y = self._scaled_dot_product_attn(q, k, v, mask)
        y = self._reshape_from_batches(y)
        y = self.w_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y


class VocabularyAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.0, gumbels=False):
        super(VocabularyAttention, self).__init__()
        self.mlp = nn.Linear(embed_dim*2, embed_dim)
        self.dropout = nn.Dropout(dropout)
        if gumbels:
            self.soft_max = GumbelSoftmax(dim=-1)
        else:
            self.soft_max = nn.Softmax(dim=-1)

    def forward(self, x, embedding):
        A = self.soft_max(x.matmul(embedding.transpose(0, 1))).matmul(embedding)
        out = self.dropout(self.mlp(torch.cat([x, A], -1)))
        return out


class ResidualBlock(nn.Module):
    def __init__(self, layer, d_model, dropout=0., rezero=False, prenorm=False):
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout)
        self.prenorm = prenorm
        self.rezero = rezero
        if rezero is False:
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, *param, **kwargs):
        if self.rezero is False:
            return x + self.dropout(self.layer(*param, **kwargs)) * self.alpha
        elif self.prenorm:
            return self.layernorm(x + self.dropout(self.layer(*param, **kwargs)))
        else:
            return x + self.layernorm(self.dropout(self.layer(*param, **kwargs)))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, sn=False):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        return self.linear2(self.leaky_relu(self.linear1(x)))


class HighwayBlock(nn.Module):
    def __init__(self, layer, d_model, dropout=0., rezero=False):
        super(HighwayBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout)
        self.rezero = rezero
        if rezero is False:
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.layernorm = nn.LayerNorm(d_model)
            self.gate = FeedForward(d_model, 1)

    def forward(self, x, *param, **kwargs):
        if self.rezero is False:
            return x + self.dropout(self.layer(*param, **kwargs)) * self.alpha  # back to residual
        else:
            g = torch.sigmoid(self.gate(x))
            return self.layernorm(x * g + self.dropout(self.layer(*param, **kwargs)) * (1 - g))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embedding, d_tar, d_src, nhead, position_encoding=False,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 rezero=False, max_sent_length=512, relative_clip=4,
                 gumbels=False, device=None):
        super(TransformerDecoderLayer, self).__init__()
        self.position_encoding = position_encoding

        if position_encoding:
            self.position_encoding_layer = PositionalEncoding(d_tar, max_len=max_sent_length, device=device)

        self.vocab_attn_layer = ResidualBlock(
            VocabularyAttention(d_tar, gumbels=gumbels),
            d_tar, dropout, rezero=rezero)

        self.self_attn = ResidualBlock(
            MultiHeadAttention(d_tar, d_tar, d_tar, d_tar, d_tar, nhead,
                               dropout=dropout, bias=True, gumbels=gumbels,
                               max_relative_position=relative_clip, device=device),
            d_tar, dropout, rezero=rezero)

        self.pos_selfattn = ResidualBlock(
            MultiHeadAttention(d_tar, d_tar, d_tar, d_tar, d_tar, nhead,
                               dropout=dropout, bias=True, gumbels=gumbels,
                               max_relative_position=relative_clip, device=device),
            d_tar, dropout, rezero=rezero)

        self.src_attn = ResidualBlock(
            MultiHeadAttention(d_tar, d_src, d_src, d_tar, d_tar, nhead,
                               dropout=dropout, bias=True, gumbels=gumbels,
                               max_relative_position=relative_clip, device=device),
            d_tar, dropout, rezero=rezero)

        self.d_tar = d_tar
        self.max_sent_length = max_sent_length

        self.feedforward = HighwayBlock(
            FeedForward(d_tar, dim_feedforward),
            d_tar, dropout, rezero=rezero)

    def forward(self, tgt, src, embedding, mask_src=None, mask_tar=None):
        tgt = self.vocab_attn_layer(tgt, tgt, embedding)  # B x l_tar x d_tar
        tgt = self.self_attn(tgt, tgt, tgt, tgt, mask_tar)  # B x l_tar x d_tar
        if self.position_encoding:
            pos_emb = self.position_encoding_layer(tgt)
            tgt = self.pos_selfattn(tgt, pos_emb, pos_emb, tgt, mask_tar)  # B x l_tar x d_tar
        else:
            tgt = self.pos_selfattn(tgt, tgt, tgt, tgt, mask_tar)  # B x l_tar x d_tar
        tgt = self.src_attn(tgt, tgt, src, src, mask_src)  # B x l_tar x d_tar
        out = self.feedforward(tgt, tgt)  # B x l_tar x d_tar
        return out  # B x l_tar x d_tar


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, nlayers=6, dropout=0.):
        super(TransformerDecoder, self).__init__()
        self.nlayers = nlayers
        self.transformer_decoder = nn.ModuleList([
            decoder_layer for i in range(nlayers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, src, embedding, src_length=None, tar_length=None, mask_src=None, mask_tar=None):
        x = self.dropout(tgt)
        xs = []
        for layer in self.transformer_decoder:
            x = layer(x, src, embedding, mask_src=mask_src, mask_tar=mask_tar)
            xs.append(x)
        return xs
