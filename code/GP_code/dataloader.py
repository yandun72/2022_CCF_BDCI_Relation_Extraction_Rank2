# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2022/6/11 21:39
# software: PyCharm

"""
文件说明：

"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_name(filename):
    """{"ID": "AT0010", "text": "故障现象：车速达到45Km/h时中央门锁不能落锁。",
    "spo_list": [{"h": {"name": "中央门锁", "pos": [16, 20]}, "t": {"name": "不能落锁", "pos": [20, 24]}, "relation": "部件故障"}]}
    """
    D = []
    data = json.load(open(filename, 'r', encoding='utf-8'))
    for line in data:
        D.append({
            'text': line['text'],
            'spo_list': [
                (spo['h']['name'], tuple(spo['h']['pos']), spo['relation'], spo['t']['name'], tuple(spo['t']['pos']))
                for spo in line['spo_list']]
        })
    return D


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """numpy函数，将序列padding到同一长度"""
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


def sequence_padding_entity(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """numpy函数，将序列padding到同一长度"""
    if length is None:
        length = np.max([np.shape(x[0])[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0][0])]

    outputs = []
    for x in inputs:
        x = x[0][slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


def search(pattern, sequence, pos):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    h_pos = pos[0]
    n = len(pattern)
    c = [-1]
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern and i <= h_pos + 1:
            c.append(i)
    return c[-1]


from utils import *


class data_generator(Dataset):
    def __init__(self, data, tokenizer, max_len, schema):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema  # spo
        # self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item['text']
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text['input_ids']
        attention_mask = encoder_text['attention_mask']
        offset_mapping = encoder_text.encodings[0].offsets
        spoes = set()
        for s, s_pos, p, o, o_pos in item['spo_list']:
            p = self.schema[p]
            sh = search(s, input_ids, s_pos)
            oh = search(o, input_ids, o_pos)
            sh, st, oh, ot = -1, -1, -1, -1
            for i in range(len(offset_mapping)): #此处代码，借鉴了vx好友“芥”的思想
                if offset_mapping[i][0] <= s_pos[0] and s_pos[0] < offset_mapping[i][1]:
                    sh = i
                if offset_mapping[i][0] < s_pos[1] and s_pos[1] <= offset_mapping[i][1]:
                    st = i
                if offset_mapping[i][0] <= o_pos[0] and o_pos[0] < offset_mapping[i][1]:
                    oh = i
                if offset_mapping[i][0] < o_pos[1] and o_pos[1] <= offset_mapping[i][1]:
                    ot = i

            try:
                if sh != -1 and oh != -1:
                    spoes.add((sh, st, p, oh, ot))
            except Exception as e:
                a = 1

        entity_labels = [set() for i in range(2)]
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]

        for sh, st, p, oh, ot in spoes:
            entity_labels[0].add((sh, st))  
            entity_labels[1].add((oh, ot))
            head_labels[p].add((sh, oh))  
            tail_labels[p].add((st, ot))

        for label in entity_labels + head_labels + tail_labels:
            if not label:
                label.add((0, 0))


        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])

        # weight = 1.0 if item['is_train'] == 1 else 0.5
        return entity_labels, head_labels, tail_labels, \
               input_ids, attention_mask

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)

    @staticmethod
    def collate(examples):

        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []

        for item in examples:
            entity_labels, head_labels, tail_labels, \
            input_ids, attention_mask = item

            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()

        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()

        return batch_token_ids, batch_mask_ids, \
               batch_entity_labels, batch_head_labels, batch_tail_labels
