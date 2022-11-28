import os
import torch
import random
import numpy as np
import argparse
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
import os
from torch.utils.data import Dataset
import numpy as np
import json
from transformers import BertTokenizerFast, BertModel, AutoTokenizer, BertConfig, BertPreTrainedModel, BertTokenizer
import torch.nn as nn
import torch
import copy
import json
import re
import pickle
from tqdm import tqdm
import logging


def get_root_logger(file_name, log_level=logging.INFO):
    """get root logger"""
    logger = logging.getLogger()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(filename=file_name),
            logging.StreamHandler()
        ]
    )
    return logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


from model import *
from Gpnet import *

bert_path = None
import time
import logging
import argparse

scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast
device = 'cuda'
schema = {'部件故障': 0, '性能故障': 1, '检测工具': 2, '组成': 3}
id2schema = {}
for k, v in schema.items():
    id2schema[v] = k


def read_rwa_data():
    file = open('../../raw_data/train.json', 'r', encoding='utf-8')
    raw_sample_cnt = 0
    raw_data = []
    for i in file.readlines():
        dic_single = {}
        arr_single = []
        raw_sample_cnt += 1
        data = json.loads(i)
        id = data['ID']
        text = data['text']
        spo_list = data['spo_list']
        dic_single['id'] = id
        dic_single['text'] = text
        dic_single['spo_list'] = spo_list
        dic_single['is_train'] = 1
        raw_data.append(dic_single)
    return raw_data


def get_waibu_data():
    file = open('../../user_data/train_ccl2022.json', 'r', encoding='utf-8')
    raw_sample_cnt = 0
    datas = []
    for i in file.readlines():
        id = 'xxx'
        dic_single = {}
        data = json.loads(i)
        text = data['text']
        head = data['h']
        tail = data['t']
        relation = data['relation']
        dic_single['text'] = text
        dic_single['id'] = id
        dic_single['h'] = head
        dic_single['t'] = tail
        dic_single['relation'] = relation
        dic_single['is_train'] = 0
        datas.append(dic_single)

    datas2 = []
    text_apear = set()
    from tqdm import tqdm
    for t in datas:
        text_apear.add(t['text'])
    text_apear = list(text_apear)

    spo_list = []
    for t in tqdm(text_apear):
        tmp = []
        for y in datas:
            if t == y['text']:
                tmp.append((y['h'], y['relation'], y['t']))
        spo_list.append(tmp)

    data3 = []
    for x, y in zip(text_apear, spo_list):
        # if len(x) > 200:
        #     continue
        tmp = {}
        tmp['text'] = x
        spo = []
        for h, r, t in y:
            tt = {}
            tt['h'] = {'name': h['name'], 'pos': h['pos']}
            tt['relation'] = r
            tt['t'] = {'name': t['name'], 'pos': t['pos']}
            # spo['spo_list'] = tt
            spo.append(tt)
        tmp['spo_list'] = spo
        tmp['is_train'] = 0
        data3.append(tmp)
    # D = []
    # for line in data3:
    #     D.append({
    #         'text': line['text'],
    #         'spo_list': [(spo['h']['name'], tuple(spo['h']['pos']), spo['relation'], spo['t']['name'], tuple(spo['t']['pos']))
    #                      for spo in line['spo_list']]
    #     })
    return data3


def get_huangchuang_data(data):
    arr_all = []
    cut_pattern = re.compile(r'([，。！？、])')
    raw_sample_cnt = 0
    for d in data:
        dic_single = {}
        arr_single = []
        raw_sample_cnt += 1

        # id = d['id']
        text = d['text']
        spo_list = d['spo_list']

        # dic_single['id'] = id
        dic_single['text'] = text
        dic_single['is_train'] = d['is_train']
        dic_single['spo_list'] = []

        if text in arr_all:
            continue

        if len(text) > 200:
            for spo in spo_list:
                h = spo['h']
                t = spo['t']
                relation = spo['relation']
                line = [(h['pos'][0], h['pos'][1], h['name']), relation, (t['pos'][0], t['pos'][1], t['name'])]
                arr_single.append(line)
            # dict_all[text] = arr_single
            spos = sorted(arr_single)

            split_blocks = cut_pattern.split(text)
            split_blocks.append("")
            split_blocks = ["".join(i) for i in zip(split_blocks[0::2], split_blocks[1::2])]
            current_text = ""
            total_blocks = []
            for block in split_blocks:
                if len(current_text + block) > 200:
                    total_blocks.append(current_text)
                    current_text = block
                else:
                    current_text += block

            if len(current_text) > 0:
                total_blocks.append(current_text)
            start_idx = 0
            end_idx = 0
            for t_idx, block_text in enumerate(total_blocks):

                end_idx += len(block_text)
                new_spos = []
                for spo in spos:

                    h_sidx, h_eidx, h_name = spo[0]
                    t_sidx, t_eidx, t_name = spo[2]

                    if start_idx <= h_eidx < end_idx and start_idx <= t_eidx <= end_idx:
                        new_spos.append(spo)

                if t_idx == 0:
                    line = {"text": block_text, "spo_list": new_spos, 'is_train': d['is_train']}
                    # if len(line['spos']):
                    arr_all.append(line)
                    # else:
                    #     print(line)
                else:
                    new_spos2 = []
                    for spo in new_spos:
                        h_sidx, h_eidx, h_name = spo[0]
                        relation = spo[1]
                        t_sidx, t_eidx, t_name = spo[2]
                        tmp = []
                        tmp.append((h_sidx - start_idx, h_eidx - start_idx, h_name))
                        tmp.append(relation)
                        tmp.append((t_sidx - start_idx, t_eidx - start_idx, t_name))
                        new_spos2.append(tmp)

                    line = {"text": block_text, "spo_list": new_spos2, 'is_train': d['is_train']}
                    # if len(line['spos']):
                    arr_all.append(line)
                    # else:
                    #     print(line)
                start_idx = end_idx

        else:
            for spo in spo_list:
                h = spo['h']
                t = spo['t']
                relation = spo['relation']

                arr_h = []
                arr_h.append(h['pos'][0])
                arr_h.append(h['pos'][1])
                arr_h.append(h['name'])

                arr_t = []
                arr_t.append(t['pos'][0])
                arr_t.append(t['pos'][1])
                arr_t.append(t['name'])

                arr_spo = []
                arr_spo.append(arr_h)
                arr_spo.append(relation)
                arr_spo.append(arr_t)
                dic_single['spo_list'].append(arr_spo)
            # if len(dic_single['spos']):
            arr_all.append(dic_single)
    print(f'==========raw_sample_cnt:{raw_sample_cnt}===========')
    print(f'==========now_sample_cnt:{len(arr_all)}===========')
    arr_all2 = []
    for x in arr_all:
        tmp = {}
        tmp['text'] = x['text']
        spo_list = []
        for z in x['spo_list']:
            t = {}
            t['h'] = {'name': z[0][-1], 'pos': z[0][0:2]}
            t['relation'] = z[1]
            t['t'] = {'name': z[2][-1], 'pos': z[2][0:2]}
            spo_list.append(t)
        tmp['spo_list'] = spo_list
        tmp['is_train'] = x['is_train']
        arr_all2.append(tmp)
        if len(x['text']) > 200:
            # print(x['text'])
            print(len(x['text']))
            print('**********************')
    D = []
    # data = json.load(open(filename, 'r', encoding='utf-8'))
    for line in arr_all2:
        D.append({
            'is_train': line['is_train'],
            'text': line['text'],
            'spo_list': [
                (spo['h']['name'], tuple(spo['h']['pos']), spo['relation'], spo['t']['name'], tuple(spo['t']['pos']))
                for spo in line['spo_list']]
        })
    return D


def del_leak_data(waibui_data, val_data):
    new_data = []
    for x in waibui_data:
        flag_add = 1
        for y in val_data:
            if x['text'] in y['text'] or y['text'] in x['text']:
                flag_add = 0
                break
        if flag_add == 1:
            new_data.append(x)
    return new_data


def extract_spoes(text, threshold=0, model=None):
    """抽取输入text中所包含的三元组"""
    max_seq_len = 220
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=max_seq_len)['offset_mapping']
    new_span, entities = [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:  # 单个字
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])  # 闭区间
    encoder_txt = tokenizer.encode_plus(text, max_length=max_seq_len)
    input_ids = torch.tensor(encoder_txt['input_ids']).long().unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)

    with autocast():
        scores = model(input_ids, attention_mask)
    outputs = [o[0].data.cpu().numpy() for o in scores]  # list类型，每个位置形状[ent_type_size, seq_len, seq_len]
    # 抽取subject和object
    subjects, objects = set(), set()
    outputs[0][:, [0, -1]] -= np.inf  # 在seq_len维度首尾取负无穷
    outputs[0][:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(outputs[0] > 0)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    # 识别对应的predicate
    spoes = set()
    for sh, st in subjects:
        for oh, ot in objects:
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]  # outputs[1]表示head,[:, sh, oh]查对应的4个关系里面对应的实体首是不是都大于0
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            ps = set(p1s) & set(p2s)  # 取交集，因为实体首部和尾部有可能被识别的关系类型不一样，取交集就可以保证了
            for p in ps:
                spoes.add((
                    text[new_span[sh][0]:new_span[st][-1] + 1], (new_span[sh][0], new_span[st][-1] + 1), id2schema[p],
                    text[new_span[oh][0]:new_span[ot][-1] + 1], (new_span[oh][0], new_span[ot][-1] + 1)
                ))
    return list(spoes)


from config import *

#args = parse_args()

tokenizer = AutoTokenizer.from_pretrained('../../user_data/opensource_models/chinese-roberta-wwm-ext-large', do_lower_case=True) #针对于所有模型的词表一致


class SPO(tuple):
    """用来存三元组的类，表现跟tuple基本一致，重写了两个特殊方法，使得在判断两个三元组是否等价时容错性更好"""

    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0], add_special_tokens=False)),
            tuple(spo[1]),
            spo[2],
            tuple(tokenizer.tokenize(spo[3], add_special_tokens=False)),
            tuple(spo[4])
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(args, data, val_loader, model,fold):
    """评估函数，计算f1、Precision、Recall"""
    losses = AverageMeter()
    model.eval()
    for i, batch in enumerate(tqdm(val_loader)):
        batch_token_ids, batch_mask_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = batch
        batch_token_ids, batch_mask_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = batch_token_ids.to(
            device), batch_mask_ids.to(device), batch_entity_labels.to(device), batch_head_labels.to(
            device), batch_tail_labels.to(device)
        with autocast():
            logits_entity, logits_head, logits_tail = model(batch_token_ids, batch_mask_ids)

            loss_entity = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels,
                                                                     y_pred=logits_entity, mask_zero=True)
            loss_head = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits_head,
                                                                   mask_zero=True)
            loss_tail = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits_tail,
                                                                   mask_zero=True)

            loss = (loss_entity + loss_head + loss_tail) / 3
        losses.update(loss.item(), args.batch_size)

    X, Y, Z = 1e-10, 1e-10, 1e-10
    correct_bujian, predict_bujian, gold_bujian = 1e-10, 1e-10, 1e-10
    correct_xingneng, predict_xingneng, gold_xingneng = 1e-10, 1e-10, 1e-10
    correct_jiance, predict_jiance, gold_jiance = 1e-10, 1e-10, 1e-10
    correct_zucheng, predict_zucheng, gold_zucheng = 1e-10, 1e-10, 1e-10

    f = open(f'{args.dev_pred_dir}/dev_pred_fold_{fold}.json', 'w', encoding='utf-8')
    bujian = 0
    xingneng = 0
    jiance = 0
    zucheng = 0

    for d in tqdm(data, desc='Evaluation', total=len(data)):
        R = set([SPO(spo) for spo in extract_spoes(d['text'], threshold=0, model=model)])
        # print(R)
        T = set([SPO(spo) for spo in d['spo_list']])
        # print(T)
        X += len(R & T)  # 抽取三元组和标注三元组匹配的个数，包括h.name,t.name,h.pos,t.pos以及relation都相同
        Y += len(R)  # 抽取三元组个数
        Z += len(T)  # 标注三元组个数

        bujian_pred, bujian_gold = [], []
        xingneng_pred, xingneng_gold = [], []
        jiance_pred, jiance_gold = [], []
        zucheng_pred, zucheng_gold = [], []
        for item in list(R):
            if item[2] == '部件故障':
                bujian_pred.append((item[0], item[1], item[-2], item[-1]))
            elif item[2] == '性能故障':
                xingneng_pred.append((item[0], item[1], item[-2], item[-1]))
            elif item[2] == "检测工具":
                jiance_pred.append((item[0], item[1], item[-2], item[-1]))
            else:
                zucheng_pred.append((item[0], item[1], item[-2], item[-1]))

        for dom in list(T):
            if dom[2] == '部件故障':
                bujian += 1
                bujian_gold.append((dom[0], dom[1], dom[-2], dom[-1]))
            if dom[2] == '性能故障':
                xingneng += 1
                xingneng_gold.append((dom[0], dom[1], dom[-2], dom[-1]))
            if dom[2] == '检测工具':
                jiance += 1
                jiance_gold.append((dom[0], dom[1], dom[-2], dom[-1]))
            if dom[2] == '组成':
                zucheng += 1
                zucheng_gold.append((dom[0], dom[1], dom[-2], dom[-1]))

        correct_bujian += len([t for t in bujian_pred if t in bujian_gold])
        predict_bujian += len(bujian_pred)
        gold_bujian += len(bujian_gold)

        correct_xingneng += len([t for t in xingneng_pred if t in xingneng_gold])
        predict_xingneng += len(xingneng_pred)
        gold_xingneng += len(xingneng_gold)

        correct_jiance += len([t for t in jiance_pred if t in jiance_gold])
        predict_jiance += len(jiance_pred)
        gold_jiance += len(jiance_gold)

        correct_zucheng += len([t for t in zucheng_pred if t in zucheng_gold])
        predict_zucheng += len(zucheng_pred)
        gold_zucheng += len(zucheng_gold)

        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
            ensure_ascii=False,
            indent=4)
        f.write(s + '\n')

    bujian_p = correct_bujian / predict_bujian
    bujian_r = correct_bujian / gold_bujian
    bujian_f = 2 * bujian_p * bujian_r / (bujian_p + bujian_r)

    xingneng_p = correct_xingneng / predict_xingneng
    xingneng_r = correct_xingneng / gold_xingneng
    xingneng_f = 2 * xingneng_p * xingneng_r / (xingneng_p + xingneng_r)

    jiance_p = correct_jiance / predict_jiance
    jiance_r = correct_jiance / gold_jiance
    jiance_f = 2 * jiance_p * jiance_r / (jiance_p + jiance_r)

    zucheng_p = correct_zucheng / predict_zucheng
    zucheng_r = correct_zucheng / gold_zucheng
    zucheng_f = 2 * zucheng_p * zucheng_r / (zucheng_p + zucheng_r)
    model.train()

    # print('bujian f1:', bujian_f, 'xingneng f1:', xingneng_f, 'jiance f1: ', jiance_f, 'zucheng f1:', zucheng_f)
    # print('bujian P:', bujian_p, 'xingneng P:', xingneng_p, 'jiance P: ', jiance_p, 'zucheng P:', zucheng_p)
    # print('bujian R:', bujian_r, 'xingneng R:', xingneng_r, 'jiance R: ', jiance_r, 'zucheng R:', zucheng_r)
    micro_f1 = (bujian_f * bujian + xingneng_f * xingneng + jiance_f * jiance + zucheng_f * zucheng) / (
            bujian + xingneng + jiance + zucheng)
    micro_r = (bujian_r * bujian + xingneng_r * xingneng + jiance_r * jiance + zucheng_r * zucheng) / (
            bujian + xingneng + jiance + zucheng)
    micro_p = (bujian_p * bujian + xingneng_p * xingneng + jiance_p * jiance + zucheng_p * zucheng) / (
            bujian + xingneng + jiance + zucheng)
    # print('micro_f1:',micro_f1,'micro_r:',micro_r,'micro_p:',micro_p,'loss',losses.avg)
    f.close()
    return micro_p, micro_r, micro_f1, losses.avg


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and 'video_embeddings' not in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and 'video_embeddings' not in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
