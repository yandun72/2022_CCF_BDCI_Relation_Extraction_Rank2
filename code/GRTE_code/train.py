# import argparse
from config import parse_args
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from bert4keras.tokenizers import Tokenizer
from sklearn.model_selection import KFold

from model import GRTE
from util import *
from tqdm import tqdm
import os
import json
from transformers import BertModel, BertConfig, BertPreTrainedModel
import torch.nn as nn
import torch

from adversarial import FGM,PGD,AWP
from optimize_trick import EMA,swa

from contrastive_loss import compute_kl_loss

import re

def get_idx(f1_list):
    min_f1 = 1000
    idx = 0
    for i in range(len(f1_list)):
        f1 = f1_list[i]
        if f1 < min_f1:
            min_f1 = f1
            idx = i
    return idx

def train():
    set_seed(2022)
    output_path = os.path.join(args.output_path)
    train_path = '../../user_data/GRTE_data/train.json'

    if args.merge_ccl:
        train_ccl2022_path = '../../user_data/GRTE_data/train_ccl_no_merge.json'
    else:
        train_ccl2022_path = '../../user_data/GRTE_data/train_ccl.json'

    train_pseudo_path = '../../user_data/GRTE_data/train_pseudoA.json'

    rel2id_path = '../../user_data/GRTE_data/rel2id.json'

    log_path = os.path.join(output_path, "log.txt")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # label
    label_list = ["N/A", "SMH", "SMT", "SS", "MMH", "MMT", "MSH", "MST"]
    id2label, label2id = {}, {}
    for i, l in enumerate(label_list):
        id2label[str(i)] = l
        label2id[l] = i

    train_data = json.load(open(train_path,encoding='utf8'))

    train_ccl2022_data = json.load(open(train_ccl2022_path,encoding='utf8'))

    train_pseudo_data = json.load(open(train_pseudo_path,encoding='utf8'))

    #     train_data += train_ccl2022_data

    #     train_data += train_pseudo_data

    id2predicate, predicate2id = json.load(open(rel2id_path,encoding='utf8'))
    all_data = np.array(train_data)

    train_ccl2022_data = np.array(train_ccl2022_data)

    train_pseudo_data = np.array(train_pseudo_data)

    kf = KFold(n_splits=args.k_num, shuffle=True, random_state=42)
    fold = 0
    for train_index, val_index in kf.split(all_data):
        fold += 1
        #         if fold<5:
        #             continue
        print("="*80)
        print(f"正在训练第 {fold} 折的数据")
        #         train_data = all_data[train_index]
        #         val_data = all_data[val_index]
        train_data = []
        val_data = []
        for i in all_data[train_index]:
            for ii in i:
                train_data.append(ii)

        for j in all_data[val_index]:
            for jj in j:
                val_data.append(jj)

        print(len(train_data))
        if args.use_pseudoA:
            cut_pattern = re.compile(r'([，。！？、])')
            for i in train_pseudo_data:
                for ii in i:
                    flag = 1
                    split_blocks = cut_pattern.split(ii['text'])
                    for val in val_data:
                        for block in split_blocks:
                            if len(block)>=10 and (val['text'] in block or block in val['text']):
                                flag = 0
                                break
                        if flag==0:
                            break
                    if flag:
                        train_data.append(ii)
            print(len(train_data))

        if args.use_ccl:
            #add ccl2022 data for train
            for i in train_ccl2022_data:
                for ii in i:
                    flag = 1
                    for train_ in train_data:
                        if ii['text'] in train_['text'] or train_['text'] in ii['text']:
                            flag = 0
                            break
                    for val in val_data:
                        if flag==0:
                            break
                        if ii['text'] in val['text'] or val['text'] in ii['text']:
                            flag = 0
                            break
                    if flag:
                        train_data.append(ii)

            print(len(train_data))

        tokenizer = Tokenizer(args.bert_vocab_path)
        config = BertConfig.from_pretrained(args.pretrained_model_path)
        config.num_p = len(id2predicate)
        config.num_label = len(label_list)
        config.rounds = args.rounds
        config.fix_bert_embeddings = args.fix_bert_embeddings

        train_model = GRTE.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_path, config=config)
        train_model.to("cuda")

        #加入对抗
        if args.use_attack == 'FGM':
            fgm = FGM(train_model, emb_name="word_embeddings", epsilon=1)

        if args.use_ema:
            ema = EMA(train_model, 0.999)
            ema.register()

        scaler = torch.cuda.amp.GradScaler()

        dataloader = data_generator(args, train_data, tokenizer, [predicate2id, id2predicate], [label2id, id2label],
                                    args.batch_size, random=True)
        val_dataloader = data_generator(args, val_data, tokenizer, [predicate2id, id2predicate], [label2id, id2label],
                                        args.val_batch_size, random=False, is_train=False)
        t_total = len(dataloader) * args.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]
        #         optimizer_grouped_parameters = [
        #             {
        #                 "params": [p for n, p in train_model.named_parameters() if not any(nd in n for nd in no_decay)],
        #                 "weight_decay": args.weight_decay,
        #             },
        #             {"params": [p for n, p in train_model.named_parameters() if any(nd in n for nd in no_decay)],
        #              "weight_decay": 0.0},
        #         ]

        optimizer_grouped_parameters = [
            {'params': [p for n, p in train_model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n],
             'weight_decay': args.weight_decay,'lr': args.bert_learning_rate},
            {'params': [p for n, p in train_model.named_parameters() if not any(nd in n for nd in no_decay) and not 'bert' in n],
             'weight_decay': args.weight_decay,'lr': args.learning_rate},
            {'params': [p for n, p in train_model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' in n],
             'weight_decay': 0.0,'lr': args.bert_learning_rate},
            {'params': [p for n, p in train_model.named_parameters() if any(nd in n for nd in no_decay) and not 'bert' in n],
             'weight_decay': 0.0,'lr': args.learning_rate},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.min_num)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
        )
        #         T_mult = 1
        #         rewarm_epoch_num = 1
        #         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #             optimizer, t_total // args.num_train_epochs * rewarm_epoch_num, T_mult, eta_min=5e-6, last_epoch=-1)

        best_f1 = -1.0
        step = 0
        crossentropy = nn.CrossEntropyLoss(reduction="none")

        if args.use_attack == 'AWP':
            awp = AWP(train_model,
                      optimizer,
                      adv_lr=args.adv_lr,
                      adv_eps=args.adv_eps,
                      start_epoch=args.awp_start,
                      scaler=scaler
                      )

        test_pred_path = os.path.join(args.result_path, f"{fold}.json")

        tmp_f1_list = [0, 0, 0]

        for epoch in range(args.num_train_epochs):
            #             if epoch>=6:
            #                 break
            print("current epoch:", epoch)
            train_model.train()
            epoch_loss = 0
            with tqdm(total=dataloader.__len__()) as t:
                for i, batch in enumerate(dataloader):
                    batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
                    batch_token_ids, batch_mask, batch_label, batch_mask_label = batch

                    batch_label = batch_label.reshape([-1])

                    #                     with torch.cuda.amp.autocast():
                    if args.r_drop==False:
                        table = train_model(batch_token_ids, batch_mask)
                        table = table.reshape([-1, len(label_list)])
                        loss = crossentropy(table, batch_label.long())
                        loss = (loss * batch_mask_label.reshape([-1])).sum()
                    if args.r_drop:
                        # keep dropout and forward twice
                        table1 = train_model(batch_token_ids, batch_mask)
                        table1 = table1.reshape([-1, len(label_list)])
                        table2 = train_model(batch_token_ids, batch_mask)
                        table2 = table2.reshape([-1, len(label_list)])
                        # cross entropy loss for classifier
                        ce_loss1 = crossentropy(table1, batch_label.long())
                        ce_loss1 = (ce_loss1 * batch_mask_label.reshape([-1])).sum()

                        ce_loss2 = crossentropy(table2, batch_label.long())
                        ce_loss2 = (ce_loss2 * batch_mask_label.reshape([-1])).sum()

                        ce_loss = 0.5 * ce_loss1 + 0.5 * ce_loss2

                        kl_loss = compute_kl_loss(table1, table2)

                        # carefully choose hyper-parameters
                        loss = (1 - args.kl_alpha) * ce_loss + args.kl_alpha * kl_loss

                    scaler.scale(loss).backward()

                    if args.use_attack == 'FGM':
                        # 对抗训练
                        #                         with torch.cuda.amp.autocast():
                        fgm.attack()  # 在embedding上添加对抗扰动
                        table_adv = train_model(batch_token_ids, batch_mask)
                        table_adv = table_adv.reshape([-1, len(label_list)])
                        loss_adv = crossentropy(table_adv, batch_label.long())
                        loss_adv = (loss_adv * batch_mask_label.reshape([-1])).sum()

                        scaler.scale(loss_adv).backward()
                        fgm.restore()  # 恢复embedding参数

                    if args.use_attack == 'AWP' and args.awp_start <= epoch:
                        awp.attack_backward(batch_token_ids, batch_mask,batch_label,batch_mask_label,label_list,crossentropy,epoch)

                    step += 1
                    epoch_loss += loss.item()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    if args.use_ema and epoch>=args.ema_start:
                        ema.update()

                    scheduler.step()  # Update learning rate schedule
                    train_model.zero_grad()

                    t.set_postfix(loss="%.4lf" % (loss.cpu().item()))
                    t.update(1)

            if args.use_ema and epoch>=args.ema_start:
                ema.apply_shadow()

            f1, precision, recall = evaluate(args, tokenizer, id2predicate, id2label, label2id, train_model,
                                             val_dataloader, test_pred_path)

            if f1 > best_f1:
                # Save model checkpoint
                best_f1 = f1
                torch.save(train_model.state_dict(),
                           f=f"{args.output_path}/model_{fold}.pth")

            #             if args.use_swa and epoch>=args.swa_start and epoch<=args.swa_end:
            if args.use_swa:
                tmp_idx = get_idx(tmp_f1_list)
                if tmp_f1_list[tmp_idx] < f1:
                    tmp_f1_list[tmp_idx] = f1
                    torch.save(train_model.state_dict(),f=f"{args.output_path}/model_tmp_{tmp_idx}.pth")

            if args.use_ema and epoch>=args.ema_start:
                ema.restore()

            epoch_loss = epoch_loss / dataloader.__len__()
            with open(log_path, "a", encoding="utf-8") as f:
                print("epoch is:%d\tloss is:%f\tf1 is:%f\tprecision is:%f\trecall is:%f\tbest_f1 is:%f\t" % (
                    int(epoch), epoch_loss, f1, precision, recall, best_f1), file=f)

        #swa
        if args.use_swa:
            swa_model = swa(args,tokenizer, id2predicate, id2label, label2id, test_pred_path, train_model, val_dataloader, args.output_path, fold, start=0, end=3)
            f1, precision, recall = evaluate(args, tokenizer, id2predicate, id2label, label2id, swa_model,
                                             val_dataloader,
                                             test_pred_path)
            if f1>0.6:
                torch.save(swa_model.state_dict(),f=f"{args.output_path}/model_swa_{fold}.pth")

            print("ema: f1 is:%f\tprecision is:%f\trecall is:%f\t" % (f1, precision, recall))
            with open(log_path, "a", encoding="utf-8") as f:
                print("ema: f1 is:%f\tprecision is:%f\trecall is:%f\t" % (f1, precision, recall), file=f)

        torch.cuda.empty_cache()
        if args.use_swa:
            del swa_model
        del train_model

def evaluate(args, tokenizer, id2predicate, id2label, label2id, model, dataloader, evl_path):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open(evl_path, 'w', encoding='utf-8')
    pbar = tqdm()
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
            pbar.update()
            pbar.set_description(
                'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
            )
            s = json.dumps({
                'text': ex['text'],
                'spos': list(T),
                'spos_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            }, ensure_ascii=False)
            f.write(s + '\n')
    pbar.close()
    f.close()
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


if __name__ == '__main__':
    args = parse_args()

    train()
