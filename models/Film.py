import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.FourierCorrelation import SpectralConv1d, SpectralConvCross1d, SpectralConv1d_local, \
    SpectralConvCross1d_local
from layers.mwt import MWT_CZ1d_cross, mwt_transform
from layers.SelfAttention_Family import FullAttention, ProbAttention
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, \
    series_decomp_multi
import math
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        # self.decomp = series_decomp(kernel_size)
        kernel_size = [kernel_size]
        self.decomp = series_decomp_multi(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        configs.ab = 2

        if configs.ab == 0:
            encoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                              seq_len=self.seq_len, modes1=configs.modes1)
            decoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                              seq_len=self.seq_len // 2 + self.pred_len, modes1=configs.modes1)
            decoder_cross_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                                    seq_len_q=self.seq_len // 2 + self.pred_len,
                                                    seq_len_kv=self.seq_len, modes1=configs.modes1)
        elif configs.ab == 1:
            encoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                              seq_len=self.seq_len)
            decoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                              seq_len=self.seq_len // 2 + self.pred_len)
            decoder_cross_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                                output_attention=False, configs=configs)
        elif configs.ab == 2:
            encoder_self_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            decoder_self_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention)
            decoder_cross_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                                    seq_len_q=self.seq_len // 2 + self.pred_len,
                                                    seq_len_kv=self.seq_len)
        elif configs.ab == 3:
            encoder_self_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention, configs=configs)
            decoder_self_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention, configs=configs)
            decoder_cross_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                                output_attention=False, configs=configs)
        elif configs.ab == 4:
            encoder_self_att = FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=configs.output_attention)
            decoder_self_att = FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=configs.output_attention)
            decoder_cross_att = FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                              output_attention=configs.output_attention)
        elif configs.ab == 8:
            encoder_self_att = ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=configs.output_attention)
            decoder_self_att = ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                             output_attention=configs.output_attention)
            decoder_cross_att = ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                              output_attention=configs.output_attention)
        elif configs.ab == 5:
            encoder_self_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                                   seq_len_q=self.seq_len, seq_len_kv=self.seq_len)
            decoder_self_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                                   seq_len_q=self.seq_len // 2 + self.pred_len,
                                                   seq_len_kv=self.seq_len // 2 + self.pred_len)
            decoder_cross_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                                    seq_len_q=self.seq_len // 2 + self.pred_len,
                                                    seq_len_kv=self.seq_len)
        elif configs.ab == 6:
            encoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                              seq_len=self.seq_len, modes1=configs.modes1)
            decoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model,
                                              seq_len=self.seq_len // 2 + self.pred_len, modes1=configs.modes1)
            decoder_cross_att = SpectralConvCross1d_local(in_channels=configs.d_model, out_channels=configs.d_model,
                                                          seq_len_q=self.seq_len // 2 + self.pred_len,
                                                          seq_len_kv=self.seq_len, modes1=configs.modes1)
        elif configs.ab == 7:
            # encoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model, seq_len=self.seq_len, modes1=configs.modes1)
            # decoder_self_att = SpectralConv1d(in_channels=configs.d_model, out_channels=configs.d_model, seq_len=self.seq_len//2+self.pred_len, modes1=configs.modes1)
            # encoder_self_att = mwt_transform(ich=configs.d_model, L=3, alpha=int(self.pred_len/2+1))
            # decoder_self_att = mwt_transform(ich=configs.d_model, L=3, alpha=int(self.pred_len/2+1))
            encoder_self_att = mwt_transform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_self_att = mwt_transform(ich=configs.d_model, L=configs.L, base=configs.base)
            # decoder_cross_att = SpectralConvCross1d(in_channels=configs.d_model, out_channels=configs.d_model,seq_len_q=self.seq_len//2+self.pred_len, seq_len_kv=self.seq_len, modes1=configs.modes1)
            decoder_cross_att = MWT_CZ1d_cross(in_channels=configs.d_model, out_channels=configs.d_model,
                                               seq_len_q=self.seq_len // 2 + self.pred_len, seq_len_kv=self.seq_len,
                                               modes1=configs.modes1, ich=configs.d_model, base=configs.base,
                                               activation=configs.cross_activation)
        elif config.ab == 8:
            encoder_self_att = SpectralConv1d_local(in_channels=configs.d_model, out_channels=configs.d_model,
                                                    seq_len=self.seq_len, modes1=configs.modes1)
            decoder_self_att = SpectralConv1d_local(in_channels=configs.d_model, out_channels=configs.d_model,
                                                    seq_len=self.seq_len // 2 + self.pred_len, modes1=configs.modes1)
            decoder_cross_att = SpectralConvCross1d_local(in_channels=configs.d_model, out_channels=configs.d_model,
                                                          seq_len_q=self.seq_len // 2 + self.pred_len,
                                                          seq_len_kv=self.seq_len, modes1=configs.modes1)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        a = 2

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], 4).to(x_enc.device)
        x_dec = torch.zeros(x_enc.shape[0], 48+720, x_enc.shape[2]).to(x_enc.device)
        x_mark_dec = torch.zeros(x_enc.shape[0], 48+720, 4).to(x_enc.device)
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # seasonal_init1 = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == '__main__':
    class Configs(object):
        ab = 0
        modes1 = 32
        seq_len = 336
        label_len = 48
        pred_len = 720
        output_attention = True
        enc_in = 7
        dec_in = 7
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        moving_avg = [25]
        c_out = 7
        activation = 'gelu'
        wavelet = 0


    configs = Configs()
    model = Model(configs)

    enc = torch.randn([32, configs.seq_len, 7])
    enc_mark = torch.randn([32, configs.seq_len, 4])

    dec = torch.randn([32, configs.label_len + configs.pred_len, 7])
    dec_mark = torch.randn([32, configs.label_len + configs.pred_len, 4])
    out = model.forward(enc, enc_mark, dec, dec_mark)
    print('input shape', enc.shape)
    print('output shape', out[0].shape)
    a = 1


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print('model size', count_parameters(model) / (1024 * 1024))