import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class ScaleDotProductAttention(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(ScaleDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, k_mask=None, return_attn=False):
        r"""
        :param q: (b_sz, nq, d)
        :param k: (b_sz, nk, d)
        :param v: (b_sz, nk, d)
        :param k_mask: (b_sz, nk)
        :param return_attn: bool
        :return:
        """
        d = q.size(-1)
        # t: (b_sz, nq, nk)
        t = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(float(d))
        if k_mask:
            k_mask = k_mask.unsqueeze(1)  # (b_sz, 1, nk)
            t = t.masked_fill(k_mask, -1e6)
        attn = F.softmax(t, -1)
        res = torch.matmul(self.dropout(attn), v)
        if return_attn:
            return res, attn
        else:
            return res


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.proj_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.proj_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.proj_v = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaleDotProductAttention(dropout_rate)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, k_mask=None, return_attn=False):
        b_sz, n_q, d_model, n_k = q.size(0), q.size(1), q.size(-1), k.size(1)
        d_k, d_v = self.d_k, self.d_v
        n_head, d_model = self.n_head, self.d_model
        residual = q
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)
        q, k, v = q.reshape(b_sz, n_q, n_head, d_k), k.reshape(b_sz, n_k, n_head, d_k), \
                  v.reshape(b_sz, n_k, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if return_attn:
            res, attn = self.attention(q, k, v, k_mask, return_attn)
        else:
            res = self.attention(q, k, v, k_mask, return_attn)
        res = res.transpose(1, 2).reshape(b_sz, n_q, -1)
        res = self.dropout(self.fc(res))
        res += residual
        res = self.layernorm(res)
        if return_attn:
            return res, attn
        else:
            return res


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_in, d_hid)
        self.w2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(d_in)

    def forward(self, x):
        residual = x
        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layernorm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, d_hid, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout_rate)
        self.pff = PositionwiseFeedForward(d_model, d_hid, dropout_rate)

    def forward(self, enc_inputs, k_mask, return_attn=False):
        if return_attn:
            res, attn = self.slf_attn(enc_inputs, enc_inputs, enc_inputs, k_mask, return_attn)
        else:
            res = self.slf_attn(enc_inputs, enc_inputs, enc_inputs, k_mask, return_attn)
        res = self.pff(res)
        if return_attn:
            return res, attn
        else:
            return res


class Bert(nn.Module):
    def __init__(self, n_layers, d_model, d_k, d_v, n_head, d_hid, vocab_size, max_len, n_classes, dropout_rate=0.1):
        super(Bert, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, n_head, d_hid, dropout_rate)
                                     for _ in range(n_layers)])
        self.embedding = nn.Embedding(vocab_size, d_model, 0)
        self.segment_embed = nn.Embedding(2, d_model)
        self.position_embed = nn.Embedding(max_len, d_model)
        self.cls_output = nn.Linear(d_model, n_classes)

    def forward(self, seq, segments, seq_pad_mask=None, return_attn=False):
        r"""
        :param seq:(b_sz, n)
        :param segments:(b_sz, n)
        :param seq_pad_mask:(b_sz, n)
        :param return_attn:bool
        :return:
        """
        b_sz, len_seq = seq.size(0), seq.size(1)
        seq_embed = self.embedding(seq)
        pos_embed = self.position_embed(torch.arange(len_seq))
        seg_embed = self.segment_embed(segments)
        # inputs:(b_sz, len_seq, embed_size)
        inputs = seq_embed + pos_embed + seg_embed
        for layer in self.layers:
            if return_attn:
                inputs, attn = layer(inputs, seq_pad_mask, return_attn)
            else:
                inputs = layer(inputs, seq_pad_mask, return_attn)

        v = inputs[:, 0, :]
        logits = self.cls_output(v)
        if return_attn:
            return logits, attn
        return logits

