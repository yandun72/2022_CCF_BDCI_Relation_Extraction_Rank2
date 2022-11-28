from functools import partial
import numpy as np
import scipy as sp
from sklearn.metrics import f1_score

import copy
import os

import torch
import tqdm

from util import *
import json

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



class OptimizedF1(object):
    '''
    调用时
    op = OptimizedF1()
    op.fit(logits,labels)
    logits = op.coefficients()*logits
    '''
    def __init__(self):
        self.coef_ = []

    def _kappa_loss(self, coef, X, y):
        """
        y_hat = argmax(coef*X, axis=-1)
        :param coef: (1D array) weights
        :param X: (2D array)logits
        :param y: (1D array) label
        :return: -f1
        """
        X_p = np.copy(X)
        X_p = coef*X_p
        ll = f1_score(y, np.argmax(X_p, axis=-1), average='macro')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [1. for _ in range(200)]#权重都初始化为1
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, y):
        X_p = np.copy(X)
        X_p = self.coef_['x'] * X_p
        return f1_score(y, np.argmax(X_p, axis=-1), average='macro')

    def coefficients(self):
        return self.coef_['x']

def evaluate(args, tokenizer, id2predicate, id2label, label2id, model, dataloader, evl_path):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(evl_path, 'w', encoding='utf-8')
    # pbar = tqdm()
    for batch in dataloader:
        batch_ex = batch[-1]
        batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
        batch_token_ids, batch_mask = batch

        batch_spo = extract_spo_list(args, tokenizer, id2predicate, id2label, label2id, model, batch_ex, batch_token_ids,
                                     batch_mask)
        for i, ex in enumerate(batch_ex):
            one = batch_spo[i]
            R = set([(tuple(item[0]), item[1], tuple(item[2])) for item in one])
            T = set([(tuple(item[0]), item[1], tuple(item[2])) for item in ex['spos']])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            # pbar.update()
            # pbar.set_description(
            #     'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
            # )
            s = json.dumps({
                'text': ex['text'],
                'spos': list(T),
                'spos_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            }, ensure_ascii=False)
            f.write(s + '\n')
    # pbar.close()
    f.close()
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def swa(args,tokenizer, id2predicate, id2label, label2id, test_pred_path, model, val_dataloader, model_dir, fold, start=2, end=6):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list=os.listdir(model_dir)
    model_path_list=[model_dir + '/'+ model_name for model_name in model_path_list]
    tmp=[]
    for model_path in model_path_list:
        if '.ipynb_checkpoints' not in model_path and f'model_tmp' in model_path and f'model_{fold}.pth' not in model_path:
            tmp.append(model_path)
    model_path_list=tmp
    print(model_path_list)

    assert 0 <= start < len(model_path_list) - 1, \
        f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list[start:end]:
            print(_ckpt)
            # logger.info(f'Load model from {_ckpt}')
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    swa_model.to('cuda')
    # swa_model = torch.nn.parallel.DataParallel(swa_model.to('cuda'))
    return swa_model



# def swa(args,model,val_dataloader,model_dir,start=2,end=6):
#     """
#     swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
#     """
#     model_path_list=os.listdir(model_dir)
#     model_path_list=[model_dir + '/'+ model_name for model_name in model_path_list]
#     tmp=[]
#     for model_path in model_path_list:
#         # if '.ipynb_checkpoints' not in model_path and 'swa' not in model_path and '.bin' in model_path:
#         if '.ipynb_checkpoints' not in model_path and '.bin' in model_path:
#             tmp.append(model_path)
#     model_path_list=tmp
#     print(model_path_list)

#     assert 0 <= start < len(model_path_list) - 1, \
#         f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'

#     swa_model = copy.deepcopy(model)
#     swa_n = 0.

#     with torch.no_grad():
#         for _ckpt in model_path_list[start:end]:
#             print(_ckpt)
#             # logger.info(f'Load model from {_ckpt}')
#             model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu'))['model_state_dict'])
#             tmp_para_dict = dict(model.named_parameters())

#             alpha = 1. / (swa_n + 1.)

#             for name, para in swa_model.named_parameters():
#                 para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

#             swa_n += 1
#     if args.device == 'cuda':
#         swa_model = torch.nn.parallel.DataParallel(swa_model.to(args.device))
#     loss, results = validate(swa_model, val_dataloader)
#     results = {k: round(v, 4) for k, v in results.items()}
#     print(f"{results}")
#     mean_f1 = results['mean_f1']

#     torch.save({'epoch': 100, 'model_state_dict': swa_model.module.state_dict(), 'mean_f1': mean_f1},
#                f'{model_dir}/model_swa_{start}_{end}_mean_f1_{mean_f1}.bin')

#     return f'{model_dir}/model_swa_{start}_{end}_mean_f1_{mean_f1}.bin'
