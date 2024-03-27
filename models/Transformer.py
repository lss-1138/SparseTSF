import torch
import torch.nn as nn
from layers.Embed import PositionalEmbedding
import numpy as np



class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.d_model = 128
        self.n_heads = 4
        self.e_layers = 2
        self.d_layers = 2
        self.d_ff = 256

        self.transformer_model = nn.Transformer(d_model=self.d_model, nhead=self.n_heads, num_encoder_layers=self.e_layers,
                                                num_decoder_layers=self.d_layers, dim_feedforward=self.d_ff, batch_first=True)

        self.pe = PositionalEmbedding(self.d_model)

        self.input = nn.Linear(self.enc_in, self.d_model)
        self.output = nn.Linear(self.d_model, self.enc_in)



    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        batch_size = x_enc.shape[0]

        enc_inp = self.input(x_enc)
        enc_inp = enc_inp + self.pe(enc_inp)
        dec_inp = torch.zeros(batch_size, self.pred_len, self.d_model).float().to(x_enc.device)
        dec_inp = dec_inp + self.pe(dec_inp)

        out = self.transformer_model(enc_inp, dec_inp)

        y = self.output(out)

        return y



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
# from layers.SelfAttention_Family import FullAttention, AttentionLayer
# from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
# import numpy as np
#
#
# class Model(nn.Module):
#     """
#     Vanilla Transformer with O(L^2) complexity
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention
#
#         # Embedding
#         if configs.embed_type == 0:
#             self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                             configs.dropout)
#             self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                            configs.dropout)
#         elif configs.embed_type == 1:
#             self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         elif configs.embed_type == 2:
#             self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#
#         elif configs.embed_type == 3:
#             self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         elif configs.embed_type == 4:
#             self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         # Encoder
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=configs.output_attention), configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )
#         # Decoder
#         self.decoder = Decoder(
#             [
#                 DecoderLayer(
#                     AttentionLayer(
#                         FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
#                         configs.d_model, configs.n_heads),
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
#                         configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation,
#                 )
#                 for l in range(configs.d_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model),
#             projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
#         )
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
#                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
#
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
#
#         dec_out = self.dec_embedding(x_dec, x_mark_dec)
#         dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
#
#         if self.output_attention:
#             return dec_out[:, -self.pred_len:, :], attns
#         else:
#             return dec_out[:, -self.pred_len:, :]  # [B, L, D]
