import os
import math
import torch
from torch import nn
from torch.nn import functional as F

from .transformer import GumbelSoftmax, LengthPredictor, PositionalEncoding,\
    TransformerDecoderLayer, TransformerDecoder
from .utils import summary


class NAGModel(nn.Module):

    def __init__(self, vocab_size, embed_size, nhead=8, n_encoder_layers=6,
                 n_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 rezero=False, gumbels=False, device=None,
                 min_length_change=-20, max_length_change=20,
                 pos_embedding_perlayer=False):
        super(NAGModel, self).__init__()
        self.net_name = 'No-AutoRegressive Transformer Dialogue Generation'
        self.dropout = dropout
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout, device=device)

        encoder_layer = nn.TransformerEncoderLayer(embed_size, nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        encoder_norm = nn.LayerNorm(embed_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers,
                                             encoder_norm)

        self.length_predictor = LengthPredictor(embed_size, device=device,
                                                min_value=min_length_change,
                                                max_value=max_length_change)

        decoder_layer = TransformerDecoderLayer(self.embedding, embed_size, embed_size, nhead,
                                                dim_feedforward=dim_feedforward, dropout=dropout,
                                                rezero=rezero, gumbels=gumbels, device=device,
                                                position_encoding=pos_embedding_perlayer)
        self.decoder = TransformerDecoder(decoder_layer, n_decoder_layers)

        self.mlp = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        if gumbels:
            self.soft_max = GumbelSoftmax(dim=-1)
        else:
            self.soft_max = nn.Softmax(dim=-1)
        self.device = device

    def forward(self, src, tgt_length=-1):
        embeded = self.embedding(src) * math.sqrt(self.embed_size)  # B x L x E
        embeded = self.pos_encoder(embeded)  # B x L x E
        encoder_output = self.encoder(embeded)  # B x L x E
        decoder_input, delta_length = self.length_predictor(encoder_output, tgt_length)  # B x L x E
        decoder_out = self.decoder(decoder_input, encoder_output, self.embedding.weight.detach())[-1]
        out = self.mlp(decoder_out)
        out = self.soft_max(out).permute(0, 2, 1)
        return out, delta_length

    def sample(self, niter=10, seq_length=50):
        print('device: ', self.device)
        test_input = torch.zeros(32, seq_length).long().to(self.device)  # B x L
        summary(self, type_size=4)
        print('Input: (B x src_len)(long)', test_input.shape, type(test_input))
        for i in range(niter):
            test_output, out_lengths = self.forward(test_input, tgt_length=-1)
        print('Output: (B x tgt_len x V)(float), (B x 1)(long)', test_output.shape, type(test_output), out_lengths.shape)
        # plot graph
        from torchviz import make_dot
        test_output, out_lengths = self.forward(test_input, tgt_length=-1)
        g = make_dot((test_output, out_lengths), params=dict(self.named_parameters()))
        g.render(os.path.join('.', self.net_name), view=False)
