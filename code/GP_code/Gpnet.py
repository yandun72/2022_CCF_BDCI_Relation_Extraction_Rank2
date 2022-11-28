import torch
import numpy as np
import torch.nn as nn

import math
def sparse_multilabel_categorical_crossentropy(y_true=None, y_pred=None, mask_zero=False):
    '''
    稀疏多标签交叉熵损失的torch实现
    '''
    shape = y_pred.shape  # [batch_size, ent_type_size, seq_len, seq_len]
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        infs = zeros + 1e12
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)
    neg_loss = all_loss + torch.log(aux_loss)
    loss = torch.mean(torch.sum(pos_loss + neg_loss))
    #loss = torch.sum(pos_loss + neg_loss,dim=1)
    return loss


class RawGlobalPointer(nn.Module):
    def __init__(self, hiddensize, ent_type_size, inner_dim=64, RoPE=True, tril_mask=True):
        '''
        :param encoder: BERT
        :param ent_type_size: 实体数目
        :param inner_dim: 64
        '''
        super(RawGlobalPointer, self).__init__()
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hiddensize
        self.dense_gp = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        self.RoPE = RoPE
        self.trail_mask = tril_mask
        self.dp = nn.Dropout(0.2)
    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)  # [seq_len, 1]

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)  # [output_dim//2]
        indices = torch.pow(10000, -2 * indices / output_dim)  # 做幂相乘 [output_dim/2]
        embeddings = position_ids * indices  # [seq_len, out_dim/2]，每一行内容是position_ids值乘indices的值
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)  # [seq_len, out_dim/2, 2]
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))  # [batch_size, seq_len, out_dim/2, 2]
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim)) # [batch_size, seq_len, out_dim]
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, context_outputs, attention_mask):
        self.device = attention_mask.device
        last_hidden_state = context_outputs[0]  # [batch, seq_len, hidden_size]
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]
        outputs = self.dense_gp(self.dp(last_hidden_state))  # [batch_size, seq_len, ent_type_size * inner_dim * 2]
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)  # 将outputs在最后一个维度上切分，每一份分别是[batch_size, seq_len, inner_dim*2]，即有ent_type_size份
        outputs = torch.stack(outputs, dim=-2)  # [batch_size, seq_len, ent_type_size, inner_dim*2]
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]  # qw -->> kw -->> [batch_size, seq_len, ent_type_size, inner_dim]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)  # 按dim的维度重复2次
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)  # [batch_size, seq_len, inner_dim]
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)  # [batch_size, seq_len, ent_type_size, inner_dim//2, 2]
            qw2 = qw2.reshape(qw.shape)  # [batch_size, seq_len, ent_type_size, inner_dim]
            qw = qw * cos_pos + qw2 * sin_pos  # [batch_size, seq_len, ent_type_size, inner_dim]
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos  # [batch_size, seq_len, ent_type_size, inner_dim]
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        # padding mask  # [batch_size, ent_type_size, seq_len, seq_len]
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12   # padding部分置为负无穷
        # 排除下三角
        if self.trail_mask:
            mask = torch.tril(torch.ones_like(logits), -1)  # 下三角（不包括斜对角）
            logits = logits - mask * 1e12  # 下三角部分置为负无穷

        return logits / self.inner_dim ** 0.5

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' sinusoid编码
        Returns: [seq_len, d_hid]
    '''
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
    embeddings_table = torch.zeros(n_position, d_hid)
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    return embeddings_table

    # 第二种实现
    position_ids = torch.arange(0, n_position).unsqueeze(1)
    position_ids = position_ids.expand(-1, d_hid)
    indices = torch.arange(0, d_hid)
    position_ids = position_ids * torch.pow(10000, -2 * torch.true_divide(torch.floor_divide(indices, 2), d_hid))
    position_ids[:, ::2] = torch.sin(position_ids[:, ::2])
    position_ids[:, 1::2] = torch.cos(position_ids[:, 1::2])
    return position_ids
class RoPEPositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265
    """
    def __init__(self, max_position, embedding_size):
        super(RoPEPositionEncoding, self).__init__()
        position_embeddings = get_sinusoid_encoding_table(max_position, embedding_size)  # [seq_len, hdsz]
        cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)
        sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)
        # register_buffer是为了最外层model.to(device)，不用内部指定device
        self.register_buffer('cos_position', cos_position)
        self.register_buffer('sin_position', sin_position)
    
    def forward(self, qw, seq_dim=-2):
        # 默认最后两个维度为[seq_len, hdsz]
        seq_len = qw.shape[seq_dim]
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
        return qw * self.cos_position[:seq_len] + qw2 * self.sin_position[:seq_len]


class EfficientGlobalPointer(nn.Module):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    """
    def __init__(self, hidden_size, heads, head_size, RoPE=True, max_len=512, use_bias=True, tril_mask=False):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dropout = nn.Dropout(0.2)
        self.p_dense = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.q_dense = nn.Linear(head_size * 2, heads * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, inputs, mask=None):
        ''' inputs: [..., hdsz]
            mask: [bez, seq_len], padding部分为0
        '''
        try:
            sequence_output = self.p_dense(self.dropout(inputs))  # [..., head_size*2]
        except Exception as e:
            sequence_output = self.p_dense(self.dropout(inputs[0]))
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]  # [..., head_size]

        # ROPE编码
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)

        # 计算内积
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size**0.5  # [btz, seq_len, seq_len], 是否是实体的打分
        bias_input = self.q_dense(sequence_output)  # [..., heads*2]
        bias = torch.stack(torch.chunk(bias_input, self.heads, dim=-1), dim=-2).transpose(1,2)  # [btz, heads, seq_len, 2]
        logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2, 3)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits

