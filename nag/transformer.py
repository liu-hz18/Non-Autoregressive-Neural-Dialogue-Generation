import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable, Function
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

from .utils import generate_key_padding_mask, generate_triu_mask, summary


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
    def __init__(self, embed_size, min_value=-20, max_value=20, max_length=250,
                 tau=1, diffrentable=False, device=None):
        super(LengthPredictor, self).__init__()
        self.embed_size = embed_size
        self.out_size = max_value - min_value + 1
        self.pooling_layer = nn.AdaptiveMaxPool1d(output_size=1)
        self.mlp = nn.Linear(self.embed_size, self.out_size)
        self.min_value = min_value
        self.device = device
        self.max_length = max_length
        self.diffrentable = diffrentable
        self.min_length = torch.Tensor([1]).to(self.device)
        if self.diffrentable:
            max_length = max_length + max_value + 1
            range_vec_i = torch.arange(max_length).float().to(self.device)
            range_vec_j = torch.arange(max_length).float().to(self.device)
            distance_mat = F.softmax(-torch.abs(range_vec_i[None, :] - range_vec_j[:, None]) / tau, dim=-1)
            self.W = nn.Parameter(distance_mat, requires_grad=False)

    def forward(self, x, src_length, tgt_length=None):
        '''
        in: B x length x embed_size
        out: B x new_length
        '''
        input_len = x.shape[1]
        out = self.pooling_layer(x.permute(0, 2, 1)).squeeze(2)  # out: B x embed_size
        len_out_prob = self.mlp(out)  # out: B x [-m, m]
        if tgt_length is not None:
            output_len = torch.max(tgt_length)
        else:
            ms = torch.argmax(len_out_prob.detach(), dim=1) + src_length + self.min_value   # out: B
            output_len = torch.max(ms)
            if output_len < 2:
                output_len = self.min_length
        if self.diffrentable:
            output = x.permute(0, 2, 1).matmul(self.W[:input_len, :output_len]).permute(0, 2, 1)
        else:
            output = self._soft_copy(x.cpu().detach().numpy(), input_len, output_len.item())
        return output, len_out_prob

    def _soft_copy(self, hidden, input_len, output_len):
        output = torch.Tensor([self._soft_copy_per_sentence(seq, input_len, output_len) for seq in hidden]).to(self.device)
        return output

    def _soft_copy_per_sentence(self, hidden, input_len, output_len):
        return [hidden[i*input_len // output_len] for i in range(output_len)]


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
        self.position_encoding = nn.Embedding.from_pretrained(pe.to(self.device), freeze=not requires_grad)
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
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position*2+1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        '''
        for self-att: length_q == length_k == length_x
        return: embeddings: length_q x length_k x d_a
        '''
        range_vec_q = torch.arange(start=0, end=length_q).to(self.device)
        range_vec_k = torch.arange(start=0, end=length_k).to(self.device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        final_mat = torch.clamp(
            distance_mat, min=-self.max_relative_position, max=self.max_relative_position)
        # final_mat = final_mat + self.max_relative_position
        embeddings = self.embeddings_table[final_mat.long()]
        return embeddings


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


class VocabularyAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, gumbels=False):
        super(VocabularyAttention, self).__init__()
        if gumbels:
            self.soft_max = GumbelSoftmax(dim=-1)
        else:
            self.soft_max = nn.Softmax(dim=-1)
        self.mlp = nn.Linear(embed_dim*2, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, embedding):
        A = self.soft_max(x.matmul(embedding.transpose(0, 1))).matmul(embedding)
        out = self.dropout(self.mlp(torch.cat([x, A], -1)))
        return out


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


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, negative_slope=0.01, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.leaky_relu(self.linear1(x))))


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


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_tar, d_src, nhead, gumbels=False, rezero=False,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 max_sent_length=64, relative_clip=4, device=None, use_wo=True,
                 use_pos_attn=False, use_vocab_attn=False, highway=False,
                 postnorm=True):
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

        if use_pos_attn:
            self.position_encoding_layer = PositionalEncoding(
                d_tar, max_len=max_sent_length, device=device,
                residual=False, requires_grad=False)
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


class Transformer(nn.Module):

    def __init__(self, ntoken, d_model, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, postnorm=True, dropout=0.1, gumbels=False,
                 use_src_mask=False, use_tgt_mask=False, use_memory_mask=False,
                 activation='relu', use_vocab_attn=False, use_pos_attn=False,
                 relative_clip=0, highway=False, device=None, max_sent_length=64,
                 share_input_output_embedding=False, share_encoder_decoder_embedding=False,
                 share_vocab_embedding=False, fix_pos_encoding=True, need_tgt_embed=True):
        super(Transformer, self).__init__()
        self.use_src_mask = use_src_mask
        self.use_tgt_mask = use_tgt_mask
        self.use_memory_mask = use_memory_mask
        self.device = device
        self.factor = math.sqrt(d_model)
        # src embedding
        self.src_embedding = nn.Embedding(ntoken, d_model)
        # output embedding
        self.share_input_output_embedding = share_input_output_embedding
        if not share_input_output_embedding:
            self.out_projection = nn.Linear(d_model, ntoken)
        # tgt embedding
        self.share_encoder_decoder_embedding = share_encoder_decoder_embedding
        self.need_tgt_embed = need_tgt_embed
        if (not share_encoder_decoder_embedding) and need_tgt_embed:
            self.tgt_embedding = nn.Embedding(ntoken, d_model)
        # vocab attention
        self.use_vocab_attn = use_vocab_attn
        self.share_vocab_embedding = share_vocab_embedding
        if use_vocab_attn and not share_vocab_embedding:
            self.vocab_embed = nn.Parameter(torch.Tensor(ntoken, d_model))
            nn.init.xavier_uniform_(self.vocab_embed)
        # pos embedding
        self.pos_encoder = PositionalEncoding(
            d_model, dropout, residual=True, device=device, requires_grad=not fix_pos_encoding)
        self.pos_decoder = PositionalEncoding(
            d_model, dropout, residual=True, device=device, requires_grad=not fix_pos_encoding)
        # build model
        encoder_layer = TransformerEncoderLayer(
            d_src=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            gumbels=gumbels,
            relative_clip=relative_clip,
            use_wo=True,
            rezero=False,
            device=device,
            use_vocab_attn=use_vocab_attn,
            highway=highway,
            postnorm=postnorm,
        )
        decoder_layer = TransformerDecoderLayer(
            d_tar=d_model,
            d_src=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            gumbels=gumbels,
            relative_clip=relative_clip,
            use_wo=True,
            rezero=False,
            device=device,
            max_sent_length=max_sent_length,
            use_vocab_attn=use_vocab_attn,
            use_pos_attn=use_pos_attn,
            highway=highway,
            postnorm=postnorm,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, dropout=dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, dropout=dropout)

    def forward(self, src, tgt, src_lengths=None, tgt_lengths=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_embed = self.src_embedding(src) * self.factor
        src_embed = self.pos_encoder(src_embed)
        if self.need_tgt_embed is True:
            if not self.share_encoder_decoder_embedding:
                tgt_embed = self.tgt_embedding(tgt) * self.factor
            else:
                tgt_embed = self.src_embedding(tgt) * self.factor
            tgt_embed = self.pos_decoder(tgt_embed)
        if self.use_vocab_attn:
            if self.share_vocab_embedding:
                embedding = self.src_embedding.weight
            else:
                embedding = self.vocab_embed
        else:
            embedding = None
        # generate mask
        src_len, tgt_len = src_embed.shape[1], tgt_embed.shape[1]
        src_mask = self._subsequent_mask(
            src_len, src_len, self.use_src_mask, self.device)
        tgt_mask = self._subsequent_mask(
            tgt_len, tgt_len, self.use_tgt_mask, self.device)
        memory_mask = self._subsequent_mask(
            src_len, tgt_len, self.use_memory_mask, self.device)
        # forward
        encoder_hidden, encoder_output = self.encoder(
            src_embed, embedding=embedding, src_mask=src_mask, src_lengths=src_lengths,
            src_key_padding_mask=src_key_padding_mask)
        decoder_hidden, decoder_output = self.decoder(
            tgt_embed, encoder_output, embedding=embedding, tgt_mask=tgt_mask,
            memory_mask=memory_mask, tgt_lengths=tgt_lengths,
            tgt_key_padding_mask=tgt_key_padding_mask)
        if not self.share_input_output_embedding:
            output = self.out_projection(decoder_output)
        else:
            output = F.linear(decoder_output, self.src_embedding.weight)
        return output, encoder_output  # need transpose for CE Loss ! ! ! e.g. output.permute(0, 2, 1)

    def _subsequent_mask(self, src_len, tgt_len, use_mask, device=None):
        if use_mask:
            return generate_triu_mask(src_len, tgt_len, device=device)
        else:
            return None

    def show_graph(self):
        summary(self, type_size=4)

    def beam_search(self, src, tgt_begin, src_length, eos_token_id, beam_size=2, max_length=32):  # for eval mode, bz = 1
        '''
        src: 1 x L, torch.LongTensor()
        tgt_begin: 1 x 1, torch.LongTensor()
        src_length: 1, torch.LongTensor()
        '''
        # init step: 1 -> beam_size
        out_probs = []
        select_path = []
        candidates = []
        outputs, _ = self.forward(src, tgt_begin, src_lengths=src_length)  # 1 x L x V
        out_prob = outputs[:, -1, :]  # 1 x V
        out_probs.append(out_prob)
        pred_probs, pred_tokens = torch.topk(-F.log_softmax(out_prob, dim=-1), dim=1, k=beam_size)  # beam_size x beam_size
        pred_tokens = torch.flatten(pred_tokens)
        for indice in pred_tokens:
            candidates.append(torch.cat((tgt_begin[0], indice.unsqueeze(0)), dim=-1))
        tgts = torch.stack(candidates, dim=0)
        srcs = src.repeat(beam_size, 1)
        accumulate_probs = pred_probs.repeat(beam_size, 1)
        src_lengths = src_length.repeat(beam_size)
        # next step: beam_size -> beam_size^2
        for i in range(max_length):  # O(beam_size x length)
            candidates = []
            outputs, _ = self.forward(srcs, tgts, src_lengths=src_lengths)  # beam_size x L x V
            out_prob = outputs[:, -1, :]  # beam_size x V
            out_probs.append(out_prob)
            pred_probs, pred_tokens = torch.topk(-F.log_softmax(out_prob, dim=-1), dim=1, k=beam_size)  # beam_size x beam_size
            pred_tokens = torch.flatten(pred_tokens)
            accumulate_probs += pred_probs
            topk_probs, topk_indices = torch.topk(torch.flatten(accumulate_probs), dim=0, k=beam_size)
            accumulate_probs = topk_probs.repeat(beam_size, 1)
            for indice in topk_indices:
                new_tgt = torch.cat((tgts[indice.item()//beam_size], pred_tokens[indice.item()].unsqueeze(0)), dim=-1)
                candidates.append(new_tgt)
            select_path.append(topk_indices[0]//beam_size)
            tgts = torch.stack(candidates, dim=0)
            if pred_tokens[0].item() == eos_token_id:
                break
        out_probs = torch.stack([out_probs[0][0]] + [out_prob[path] for out_prob, path in zip(out_probs[1:], select_path)], dim=0)
        return tgts[0][1:].unsqueeze(0), out_probs.unsqueeze(0)

    def greedy(self, src, tgt_begin, src_length, eos_token_id, max_length=32):  # for eval mode, bz = 1
        '''
        src: 1 x L, torch.LongTensor()
        tgt_begin: 1 x 1, torch.LongTensor()
        src_length: 1, torch.LongTensor()
        '''
        tgt = tgt_begin
        out_probs = []
        for i in range(max_length):
            output, _ = self.forward(src, tgt, src_lengths=src_length)  # 1 x L x V
            out_prob = output[:, -1, :]
            out_probs.append(out_prob[0])
            pred_token = torch.argmax(out_prob, dim=1)
            tgt = torch.cat((tgt, pred_token.unsqueeze(0)), dim=1)  # 1 x (L+1)
            if pred_token.item() == eos_token_id:
                break
        return tgt[:, 1:], torch.stack(out_probs, dim=0).unsqueeze(0)


class TransformerNonAutoRegressive(Transformer):

    def __init__(self, ntoken, d_model, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, postnorm=True, dropout=0.1, gumbels=False,
                 use_src_mask=False, use_tgt_mask=False, use_memory_mask=False,
                 activation='relu', use_vocab_attn=False, use_pos_attn=False,
                 relative_clip=0, highway=False, device=None, max_sent_length=64,
                 share_input_output_embedding=False, share_encoder_decoder_embedding=False,
                 share_vocab_embedding=False, fix_pos_encoding=True,
                 min_length_change=-20, max_length_change=20):
        use_src_mask = use_tgt_mask = use_memory_mask = False
        super(TransformerNonAutoRegressive, self).__init__(
            ntoken, d_model, nhead, num_encoder_layers, num_decoder_layers,
            dim_feedforward, postnorm, dropout, gumbels,
            use_src_mask, use_tgt_mask, use_memory_mask,
            activation, use_vocab_attn, use_pos_attn,
            relative_clip, highway, device, max_sent_length,
            share_input_output_embedding, share_encoder_decoder_embedding,
            share_vocab_embedding, fix_pos_encoding, need_tgt_embed=False)
        self.length_predictor = LengthPredictor(
            d_model, min_value=min_length_change, max_value=max_length_change,
            device=device, diffrentable=False)

    def forward(self, src, src_lengths=None, tgt_lengths=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        '''
        use length-predictor to predict target length
        '''
        src_embed = self.src_embedding(src) * self.factor
        src_embed = self.pos_encoder(src_embed)
        if self.use_vocab_attn:
            if self.share_vocab_embedding:
                embedding = self.src_embedding.weight
            else:
                embedding = self.vocab_embed
        else:
            embedding = None
        # forward
        encoder_hidden, encoder_output = self.encoder(
            src_embed, embedding=embedding, src_mask=None, src_lengths=src_lengths,
            src_key_padding_mask=src_key_padding_mask)

        decoder_input, delta_length_probs = self.length_predictor(
            encoder_output, src_lengths, tgt_lengths)  # B x L x E

        tgt_embed = self.pos_decoder(decoder_input)
        decoder_hidden, decoder_output = self.decoder(
            tgt_embed, encoder_output, embedding=embedding, tgt_mask=None,
            memory_mask=None, tgt_lengths=tgt_lengths,
            tgt_key_padding_mask=tgt_key_padding_mask)

        if not self.share_input_output_embedding:
            output = self.out_projection(decoder_output)
        else:
            output = F.linear(decoder_output, self.src_embedding.weight)
        return output, delta_length_probs  # need transpose for CE Loss ! ! ! e.g. output.permute(0, 2, 1)


class TransformerConditionalMasked(nn.Module):
    """docstring for Conditional Masked Transformer"""
    def __init__(self, ntoken, d_model, nhead=8, max_sent_length=64,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, postnorm=True, dropout=0.1, gumbels=False,
                 activation='relu', relative_clip=0, highway=False, device=None,
                 share_input_output_embedding=False, share_encoder_decoder_embedding=False):
        super(TransformerConditionalMasked, self).__init__()
        self.cls_token_id = ntoken
        self.mask_token_id = ntoken+1
        self.device = device
        self.max_sent_length = max_sent_length
        self.transformer = Transformer(
            ntoken+2, d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, postnorm=postnorm, dropout=dropout, gumbels=gumbels,
            use_src_mask=False, use_tgt_mask=False, use_memory_mask=False,
            activation=activation, use_vocab_attn=True, use_pos_attn=True,
            relative_clip=4, highway=False, device=device, max_sent_length=max_sent_length,
            share_input_output_embedding=share_input_output_embedding,
            share_encoder_decoder_embedding=share_encoder_decoder_embedding,
            share_vocab_embedding=True, fix_pos_encoding=True, need_tgt_embed=True)
        self.length_projector = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, max_sent_length),
        )
        self.init_tgt = torch.LongTensor([self.mask_token_id]).to(self.device)

    def forward(self, src, tgt=None, src_lengths=None, tgt_lengths=None, mask_iter=3):
        '''
        use [CLS] to predict target length
        '''
        bz = src.shape[0]
        src = torch.cat(
            (torch.LongTensor([self.cls_token_id]).to(src.device).repeat(bz, 1), src), dim=1)
        if src_lengths is not None:
            src_lengths += 1
        if tgt is not None:  # train
            mask = torch.rand_like(tgt.float(), device=tgt.device)
            tgt = tgt.masked_fill(mask < torch.rand((1), device=tgt.device), self.mask_token_id)
            output, encoder_output = self.transformer(src, tgt, src_lengths, tgt_lengths)
            pred_lengths_probs = self.length_projector(encoder_output[:, 0, :])  # B x max_sent_length
        else:  # eval
            # init step
            tgt = self.init_tgt.expand(bz, self.max_sent_length)
            output, encoder_output = self.transformer(src, tgt, src_lengths)
            pred_lengths_probs = self.length_projector(encoder_output[:, 0, :]) # B x max_sent_length
            pred_lengths = torch.argmax(pred_lengths_probs, dim=-1)
            output = output[:, :torch.max(pred_lengths), :]
            for counter in range(1, mask_iter):
                tgt_probs = F.softmax(output, dim=-1)
                tgt_tokens = torch.argmax(tgt_probs, dim=-1)
                tgt_probs = torch.max(tgt_probs, dim=-1)[0]
                num_mask = (pred_lengths.float() * (1.0 - (counter / mask_iter))).long()
                mask = self._generate_worst_mask(tgt_probs, num_mask)
                tgt = tgt_tokens.masked_fill(mask == 0, self.mask_token_id)
                output, encoder_output = self.transformer(src, tgt, src_lengths, pred_lengths)
        return output, pred_lengths_probs

    def show_graph(self):
        summary(self, type_size=4)

    def _generate_worst_mask(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        return torch.cat([torch.ones((1, seq_len)).to(token_probs.device).index_fill(1, mask, value=0) for mask in masks], dim=0)


class TransformerTorch(nn.Module):

    def __init__(self, ntoken, d_model, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, postnorm=True, dropout=0.1, gumbels=False,
                 use_src_mask=False, use_tgt_mask=False, use_memory_mask=False,
                 activation='relu', use_vocab_attn=False, use_pos_attn=False,
                 relative_clip=0, highway=False, device=None, max_sent_length=64,
                 share_input_output_embedding=False, share_encoder_decoder_embedding=False,
                 share_vocab_embedding=False, fix_pos_encoding=True):
        super(TransformerTorch, self).__init__()
        self.use_src_mask = use_src_mask
        self.use_tgt_mask = use_tgt_mask
        self.use_memory_mask = use_memory_mask
        self.device = device
        self.factor = math.sqrt(d_model)
        # src embedding
        self.src_embedding = nn.Embedding(ntoken, d_model)
        # output embedding
        self.share_input_output_embedding = share_input_output_embedding
        if not share_input_output_embedding:
            self.out_projection = nn.Linear(d_model, ntoken)
        # tgt embedding
        self.share_encoder_decoder_embedding = share_encoder_decoder_embedding
        if not share_encoder_decoder_embedding:
            self.tgt_embedding = nn.Embedding(ntoken, d_model)
        # pos embedding
        self.pos_encoder = PositionalEncoding(
            d_model, dropout, residual=True, device=device, requires_grad=not fix_pos_encoding)
        self.pos_decoder = PositionalEncoding(
            d_model, dropout, residual=True, device=device, requires_grad=not fix_pos_encoding)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout)

    def forward(self, src, tgt, src_lengths=None, tgt_lengths=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_embed = self.src_embedding(src) * self.factor
        src_embed = self.pos_encoder(src_embed)
        if not self.share_encoder_decoder_embedding:
            tgt_embed = self.tgt_embedding(tgt) * self.factor
        else:
            tgt_embed = self.src_embedding(tgt) * self.factor
        tgt_embed = self.pos_decoder(tgt_embed)
        # generate mask
        src_len, tgt_len = src_embed.shape[1], tgt_embed.shape[1]
        src_mask = self._subsequent_mask(
            src_len, src_len, self.use_src_mask, self.device)
        tgt_mask = self._subsequent_mask(
            tgt_len, tgt_len, self.use_tgt_mask, self.device)
        memory_mask = self._subsequent_mask(
            src_len, tgt_len, self.use_memory_mask, self.device)
        if src_lengths is not None and src_key_padding_mask is None:
            src_key_padding_mask = ~generate_key_padding_mask(
                src.shape[1], src_lengths)
        if tgt_lengths is not None and tgt_key_padding_mask is None:
            tgt_key_padding_mask = ~generate_key_padding_mask(
                tgt.shape[1], tgt_lengths)
        # forward
        output = self.transformer(
            src_embed.permute(1, 0, 2), tgt_embed.permute(1, 0, 2),
            src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask)\
            .permute(1, 0, 2)
        if not self.share_input_output_embedding:
            output = self.out_projection(output)
        else:
            output = F.linear(output, self.src_embedding.weight)
        return output, None  # need transpose for CE Loss ! ! ! e.g. output.permute(0, 2, 1)

    def _subsequent_mask(self, src_len, tgt_len, use_mask, device=None):
        if use_mask:
            mask = generate_triu_mask(src_len, tgt_len, device=device)
            return mask.float().masked_fill(mask == 0, float(1e-7)).masked_fill(mask == 1, float(0.0))
        else:
            return None

    def show_graph(self):
        summary(self, type_size=4)

    def beam_search(self, src, tgt_begin, src_length, eos_token_id, beam_size=2, max_length=32):  # for eval mode, bz = 1
        '''
        src: 1 x L, torch.LongTensor()
        tgt_begin: 1 x 1, torch.LongTensor()
        src_length: 1, torch.LongTensor()
        '''
        # init step: 1 -> beam_size
        out_probs = []
        select_path = []
        candidates = []
        outputs, _ = self.forward(src, tgt_begin, src_lengths=src_length)  # 1 x L x V
        out_prob = outputs[:, -1, :]  # 1 x V
        out_probs.append(out_prob)
        pred_probs, pred_tokens = torch.topk(-F.log_softmax(out_prob, dim=-1), dim=1, k=beam_size)  # beam_size x beam_size
        pred_tokens = torch.flatten(pred_tokens)
        for indice in pred_tokens:
            candidates.append(torch.cat((tgt_begin[0], indice.unsqueeze(0)), dim=-1))
        tgts = torch.stack(candidates, dim=0)
        srcs = src.repeat(beam_size, 1)
        accumulate_probs = pred_probs.repeat(beam_size, 1)
        src_lengths = src_length.repeat(beam_size)
        # next step: beam_size -> beam_size^2
        for i in range(max_length):  # O(beam_size x length)
            candidates = []
            outputs, _ = self.forward(srcs, tgts, src_lengths=src_lengths)  # beam_size x L x V
            out_prob = outputs[:, -1, :]  # beam_size x V
            out_probs.append(out_prob)
            pred_probs, pred_tokens = torch.topk(-F.log_softmax(out_prob, dim=-1), dim=1, k=beam_size)  # beam_size x beam_size
            pred_tokens = torch.flatten(pred_tokens)
            accumulate_probs += pred_probs
            topk_probs, topk_indices = torch.topk(torch.flatten(accumulate_probs), dim=0, k=beam_size)
            accumulate_probs = topk_probs.repeat(beam_size, 1)
            for indice in topk_indices:
                new_tgt = torch.cat((tgts[indice.item()//beam_size], pred_tokens[indice.item()].unsqueeze(0)), dim=-1)
                candidates.append(new_tgt)
            select_path.append(topk_indices[0]//beam_size)
            tgts = torch.stack(candidates, dim=0)
            if pred_tokens[0].item() == eos_token_id:
                break
        out_probs = torch.stack([out_probs[0][0]] + [out_prob[path] for out_prob, path in zip(out_probs[1:], select_path)], dim=0)
        return tgts[0][1:].unsqueeze(0), out_probs.unsqueeze(0)

    def greedy(self, src, tgt_begin, src_length, eos_token_id, max_length=32):  # for eval mode, bz = 1
        '''
        src: 1 x L, torch.LongTensor()
        tgt_begin: 1 x 1, torch.LongTensor()
        src_length: 1, torch.LongTensor()
        '''
        tgt = tgt_begin
        out_probs = []
        for i in range(max_length):
            output, _ = self.forward(src, tgt, src_lengths=src_length)  # 1 x L x V
            out_prob = output[:, -1, :]
            out_probs.append(out_prob[0])
            pred_token = torch.argmax(out_prob, dim=1)
            tgt = torch.cat((tgt, pred_token.unsqueeze(0)), dim=1)  # 1 x (L+1)
            if pred_token.item() == eos_token_id:
                break
        return tgt[:, 1:], torch.stack(out_probs, dim=0).unsqueeze(0)
