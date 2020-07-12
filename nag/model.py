import math
import torch
from torch import nn
from torch.nn import functional as F

from .modules import GumbelSoftmax, LengthPredictor, PositionalEncoding,\
    TransformerDecoderLayer, TransformerDecoder
from .utils import summary, generate_key_padding_mask


class NAGModel(nn.Module):

    def __init__(self, vocab_size, embed_size, nhead=8, n_encoder_layers=6,
                 n_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 rezero=False, gumbels=False, device=None,
                 min_length_change=-20, max_length_change=20,
                 use_pos_attn=False, use_vocab_attn=False, fix_pos_encoding=True):
        super(NAGModel, self).__init__()
        self.net_name = 'No-AutoRegressive Transformer Dialogue Generation'
        self.dropout = dropout
        self.embed_size = embed_size
        self.device = device
        # word embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # transformer encoder
        self.pos_encoder = PositionalEncoding(
            embed_size, dropout, residual=True, device=device, requires_grad=not fix_pos_encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_size, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(embed_size)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, n_encoder_layers, encoder_norm)
        # length predictor
        self.length_predictor = LengthPredictor(
            embed_size, min_value=min_length_change, max_value=max_length_change,
            device=device, diffrentable=False)
        # vocab representation matrix
        self.vocab_embed = nn.Parameter(torch.Tensor(vocab_size, embed_size))
        # transformer decoder
        self.pos_decoder = PositionalEncoding(
            embed_size, dropout, residual=True, device=device, requires_grad=not fix_pos_encoding)
        decoder_layer = TransformerDecoderLayer(
            embed_size, embed_size, nhead, dropout=dropout,
            dim_feedforward=dim_feedforward, rezero=rezero, max_sent_length=64,
            gumbels=gumbels, use_pos_attn=use_pos_attn, use_vocab_attn=use_vocab_attn,
            device=device, highway=False)
        self.decoder = TransformerDecoder(
            decoder_layer, n_decoder_layers, dropout=dropout)
        # tranlate
        self.mlp = nn.Linear(embed_size, vocab_size)
        nn.init.xavier_uniform_(self.vocab_embed)

    def forward(self, src, src_length=None, tgt_length=None):
        mask_src = None
        if src_length is not None:
            mask_src = generate_key_padding_mask(
                srclen=src.shape[1], lengths=src_length, device=src.device, mask_is=True)
        embeded = self.embedding(src) * math.sqrt(self.embed_size)  # B x L x E
        encoder_input = self.pos_encoder(embeded, src_length)  # B x L x E
        encoder_output = self.encoder(
            encoder_input.permute(1, 0, 2), src_key_padding_mask=mask_src).permute(1, 0, 2)  # B x L x E

        decoder_input, delta_length = self.length_predictor(
            encoder_output, src_length, tgt_length)  # B x L x E

        decoder_input = self.pos_decoder(decoder_input, tgt_length)
        decoder_hidden, decoder_out = self.decoder(
            decoder_input, encoder_output, embedding=self.vocab_embed,
            src_length=src_length, tar_length=tgt_length)
        out = self.mlp(decoder_out).permute(0, 2, 1)
        return out, decoder_out, delta_length, self.embedding.weight.detach()

    def sample(self, niter=10, seq_length=50):
        print('device: ', self.device)
        # test_input = torch.zeros(32, seq_length).long().to(self.device)  # B x L
        summary(self, type_size=4)
        # print('Input: (B x src_len)(long)', test_input.shape, type(test_input))
        # for i in range(niter):
        #     test_output, _, out_lengths, _ = self.forward(test_input)
        # print('Output: (B x tgt_len x V)(float), (B x 1)(long)', test_output.shape, type(test_output), out_lengths.shape)
        # plot graph
        # from torchviz import make_dot
        # test_output, out_lengths = self.forward(test_input, tgt_length=-1)
        # g = make_dot((test_output, out_lengths), params=dict(self.named_parameters()))
        # g.render(os.path.join('.', self.net_name), view=False)


class NATransformer(nn.Module):

    def __init__(self, vocab_size, embed_size, nhead=8, dropout=0.1,
                 n_decoder_layers=6, n_encoder_layers=6, dim_feedforward=2048,
                 min_length_change=-20, max_length_change=20, fix_pos_encoding=True,
                 gumbels=False, activation='relu', device=None):
        super(NATransformer, self).__init__()
        self.net_name = 'No-AutoRegressive Transformer Dialogue Generation'
        self.dropout = dropout
        self.embed_size = embed_size
        self.device = device
        # word embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # transformer encoder
        self.pos_encoder = PositionalEncoding(
            embed_size, dropout, residual=True, device=device, requires_grad=not fix_pos_encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_size, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm = nn.LayerNorm(embed_size)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, n_encoder_layers, norm=encoder_norm)
        # length predictor
        self.length_predictor = LengthPredictor(
            embed_size, device=device, min_value=min_length_change,
            max_value=max_length_change, diffrentable=False)
        # transformer decoder
        self.pos_decoder = PositionalEncoding(
            embed_size, dropout, residual=True, device=device, requires_grad=not fix_pos_encoding)
        decoder_layer = nn.TransformerDecoderLayer(
            embed_size, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        decoder_norm = nn.LayerNorm(embed_size)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, n_decoder_layers, norm=decoder_norm)
        # tranlate
        self.mlp = nn.Linear(embed_size, vocab_size)

    def forward(self, src, src_length=None, tgt_length=None):
        if src_length is not None:
            mask_src = padding_mask(
                srclen=src.shape[1], lengths=src_length, device=src.device, mask_is=True)
        embeded = self.embedding(src) * math.sqrt(self.embed_size)  # B x L x E
        encoder_input = self.pos_encoder(embeded, src_length)  # B x L x E
        encoder_output = self.encoder(
            encoder_input.permute(1, 0, 2), src_key_padding_mask=mask_src).permute(1, 0, 2)  # B x L x E

        decoder_input, delta_length = self.length_predictor(encoder_output, src_length, tgt_length)  # B x L x E

        decoder_input = self.pos_decoder(decoder_input, tgt_length)
        decoder_out = self.decoder(
            decoder_input.permute(1, 0, 2), encoder_output.permute(1, 0, 2),
            memory_key_padding_mask=mask_src).permute(1, 0, 2)
        out = self.mlp(decoder_out).permute(0, 2, 1)
        return out, decoder_out, delta_length, self.embedding.weight.detach()
