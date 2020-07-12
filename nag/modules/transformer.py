import math
import torch
from torch import nn
from torch.nn import functional as F

from .sinusoidal_position_embedding import PositionalEncoding
from .transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from .operators import StraightThroughLogits, StraightThroughSoftmax, GumbelSoftmax
from ..utils import generate_key_padding_mask, generate_triu_mask, summary


class TransformerBase(nn.Module):
    """docstring for TransformerBase"""
    def __init__(self):
        super(TransformerBase, self).__init__()

    def forward(self, src, tgt, src_lengths=None, tgt_lengths=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        raise NotImplementedError

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
        pred_probs, pred_tokens = torch.topk(
            F.log_softmax(out_prob, dim=-1), dim=1, k=beam_size)  # beam_size x beam_size
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
            pred_probs, pred_tokens = torch.topk(
                F.log_softmax(out_prob, dim=-1), dim=1, k=beam_size)  # beam_size x beam_size
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


class Transformer(TransformerBase):

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
        self.src_embedding = nn.Embedding(ntoken, d_model, padding_idx=0)
        # output embedding
        self.share_input_output_embedding = share_input_output_embedding
        if not share_input_output_embedding:
            self.out_projection = nn.Linear(d_model, ntoken)
        # tgt embedding
        self.share_encoder_decoder_embedding = share_encoder_decoder_embedding
        self.need_tgt_embed = need_tgt_embed
        if (not share_encoder_decoder_embedding) and need_tgt_embed:
            self.tgt_embedding = nn.Embedding(ntoken, d_model, padding_idx=0)
        # vocab attention
        self.use_vocab_attn = use_vocab_attn
        self.share_vocab_embedding = share_vocab_embedding
        if use_vocab_attn and (not share_vocab_embedding):
            self.vocab_embed = nn.Parameter(torch.Tensor(ntoken, d_model))
            nn.init.xavier_uniform_(self.vocab_embed)
        # pos embedding
        self.pos_encoder = PositionalEncoding(
            d_model, dropout, residual=True, device=device, requires_grad=not fix_pos_encoding)
        self.pos_decoder = PositionalEncoding(
            d_model, dropout, residual=True, device=device, requires_grad=not fix_pos_encoding)
        if use_pos_attn:
            self.position_encoding_layer = PositionalEncoding(
                d_model, max_len=max_sent_length, device=device,
                residual=False, requires_grad=False)
        else:
            self.position_encoding_layer = None
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
            position_encoding_layer=self.position_encoding_layer,
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
        return self.forward_after_embed(
            src_embed, tgt_embed, src_lengths, tgt_lengths,
            src_key_padding_mask, tgt_key_padding_mask)

    def forward_after_embed(self, src_embed, tgt_embed, src_lengths=None, tgt_lengths=None,
                            src_key_padding_mask=None, tgt_key_padding_mask=None):
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


class TransformerTorch(TransformerBase):

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
        self.src_embedding = nn.Embedding(ntoken, d_model, padding_idx=0)
        # output embedding
        self.share_input_output_embedding = share_input_output_embedding
        if not share_input_output_embedding:
            self.out_projection = nn.Linear(d_model, ntoken)
        # tgt embedding
        self.share_encoder_decoder_embedding = share_encoder_decoder_embedding
        if not share_encoder_decoder_embedding:
            self.tgt_embedding = nn.Embedding(ntoken, d_model, padding_idx=0)
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


class TransformerContinuous(Transformer):
    """docstring for TransformerContinuous"""
    operator_map = {
        'SX': nn.Softmax(dim=2),
        'STL': StraightThroughLogits(),
        'SG': GumbelSoftmax(hard=True, tau=1, dim=-1),
        'ST': StraightThroughSoftmax(dim=-1),
        'GX': GumbelSoftmax(hard=False, tau=1, dim=-1),
    }

    def __init__(self, ntoken, d_model, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, postnorm=True, dropout=0.1, gumbels=False,
                 use_src_mask=False, use_tgt_mask=False, use_memory_mask=False,
                 activation='relu', use_vocab_attn=False, use_pos_attn=False,
                 relative_clip=0, highway=False, device=None, max_sent_length=64,
                 share_input_output_embedding=False, share_encoder_decoder_embedding=False,
                 share_vocab_embedding=False, fix_pos_encoding=True, need_tgt_embed=True,
                 bos_token=2, tgt_operator='SX', dis_operator='SX'):
        super(TransformerContinuous, self).__init__(
            ntoken, d_model, nhead=nhead, gumbels=gumbels,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, postnorm=postnorm, dropout=dropout,
            use_src_mask=use_src_mask, use_tgt_mask=use_tgt_mask, use_memory_mask=use_memory_mask,
            activation=activation, use_vocab_attn=use_vocab_attn, use_pos_attn=use_pos_attn,
            relative_clip=relative_clip, highway=highway, device=device, max_sent_length=max_sent_length,
            share_input_output_embedding=share_input_output_embedding,
            share_encoder_decoder_embedding=share_encoder_decoder_embedding,
            share_vocab_embedding=share_vocab_embedding,
            fix_pos_encoding=fix_pos_encoding, need_tgt_embed=True)
        self.bos_token = bos_token
        self.input_operator = self.operator_map[tgt_operator]
        self.ref_operator = self.operator_map[dis_operator]

    def forward(self, src, tgt, src_lengths=None, tgt_lengths=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        '''
        src: LongTensor of shape (B x L)
        tgt: FloatTensor of shape (B x L x V)
        output: FloatTensor of shape (B x L x V)
        '''
        bz = src.shape[0]
        src_embed = self.src_embedding(src) * self.factor
        src_embed = self.pos_encoder(src_embed)
        if self.need_tgt_embed is True:
            tgt = self.input_operator(tgt)
            bos_token = torch.LongTensor(bz, 1).fill_(self.bos_token).to(src.device)
            if not self.share_encoder_decoder_embedding:
                bos_embed = self.tgt_embedding(bos_token)
                tgt_embed = torch.matmul(tgt[:, :-1, :], self.tgt_embedding.weight)
            else:
                bos_embed = self.src_embedding(bos_token)
                tgt_embed = torch.matmul(tgt[:, :-1, :], self.src_embedding.weight)
            tgt_embed = torch.cat((bos_embed, tgt_embed), 1) * self.factor
            tgt_embed = self.pos_decoder(tgt_embed)
        return self.forward_after_embed(
            src_embed, tgt_embed, src_lengths, tgt_lengths,
            src_key_padding_mask, tgt_key_padding_mask)

    def energy(self, src, energy_input, inf_output, src_lengths=None, tgt_lengths=None,
               src_key_padding_mask=None, tgt_key_padding_mask=None):
        output, _ = self.forward(
            src, energy_input, src_lengths=src_lengths, tgt_lengths=tgt_lengths,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask)
        energy_pred = torch.argmax(output, dim=2)
        # output: (B x L x V)
        prev_output = self.ref_operator(inf_output)  # (B x L x V)
        scores = F.log_softmax(output, dim=2)
        energy = (-torch.sum(scores * prev_output, dim=(1, 2))).mean() - scores.mean()  # B
        return energy, energy_pred
