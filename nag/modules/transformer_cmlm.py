import torch
from torch import nn
from torch.nn import functional as F

from .transformer import Transformer
from ..utils import summary


class TransformerConditionalMasked(nn.Module):
    """docstring for Conditional Masked Transformer"""
    def __init__(self, ntoken, d_model, nhead=8, max_sent_length=64,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, postnorm=True, dropout=0.1, gumbels=False,
                 activation='relu', relative_clip=0, highway=False, device=None,
                 share_input_output_embedding=False, share_encoder_decoder_embedding=False,
                 other_model_embedding=None):
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
            activation=activation, use_vocab_attn=False, use_pos_attn=True,
            relative_clip=0, highway=False, device=device, max_sent_length=max_sent_length,
            share_input_output_embedding=share_input_output_embedding,
            share_encoder_decoder_embedding=share_encoder_decoder_embedding,
            share_vocab_embedding=True, fix_pos_encoding=True, need_tgt_embed=True)
        if other_model_embedding is not None:
            self.transformer.src_embedding = other_model_embedding
        self.length_projector = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, max_sent_length),
        )
        self.init_tgt = torch.LongTensor([self.mask_token_id]).to(self.device)

    def forward(self, src, tgt=None, src_lengths=None, tgt_lengths=None, mask_iter=3, mask_all=False):
        '''
        use [CLS] to predict target length
        '''
        bz = src.shape[0]
        src = torch.cat(
            (torch.LongTensor([self.cls_token_id]).to(src.device).repeat(bz, 1), src), dim=1)
        if src_lengths is not None:
            src_lengths += 1
        if tgt is not None:  # train
            if mask_all is False:
                mask = torch.rand_like(tgt.float(), device=tgt.device)
                tgt = tgt.masked_fill(mask < torch.rand((1), device=tgt.device), self.mask_token_id)
            else:
                mask = torch.ones_like(tgt, dtype=torch.bool, device=tgt.device)
                tgt = tgt.masked_fill(mask == 1, self.mask_token_id)
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
                # print(torch.argmax(output, dim=-1))
        return output, pred_lengths_probs

    def show_graph(self):
        summary(self, type_size=4)

    def _generate_worst_mask(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        return torch.cat([torch.ones((1, seq_len)).to(token_probs.device).index_fill(1, mask, value=0) for mask in masks], dim=0)
