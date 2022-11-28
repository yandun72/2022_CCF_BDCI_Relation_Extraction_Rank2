from utils import *
import os
import pdb
import sys
import json
import torch
import configparser
import torch.nn as nn
from model import *
from tqdm import tqdm
from optim import *
from Gpnet import *
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, AutoTokenizer
from config import *
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


def train(args, train_loader, val_loader, valid_loader, model, optimizer, scheduler, num_total_steps, device, logger,
          fold):
    total_loss, total_f1 = 0., 0.
    total_step = 0
    best_score = 0.
    ema_start = False
    step = 0
    num_total_steps = num_total_steps
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

    if args.use_ema:
        model_ema = EMA(model, 0.999)

    if args.use_fgm:
        model_adv = FGM(model)
    cnt_lower_best = 0
    for epoch in range(args.num_epochs):
        if epoch >= 15: # 25 epoch没完整跑完，费时间
            break
        for i, batch in enumerate(tqdm(train_loader)):
            model.train()

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

            scaler.scale(loss).backward()

            if args.use_fgm and epoch >= 3:
                model_adv.attack()
                with autocast():
                    logits_entity_adv, logits_head_adv, logits_tail_adv = model(batch_token_ids, batch_mask_ids)
                    loss_entity_adv = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels,
                                                                                 y_pred=logits_entity_adv,
                                                                                 mask_zero=True)
                    loss_head_adv = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels,
                                                                               y_pred=logits_head_adv, mask_zero=True)
                    loss_tail_adv = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels,
                                                                               y_pred=logits_tail_adv, mask_zero=True)

                    loss_adv = (loss_entity_adv + loss_head_adv + loss_tail_adv) / 3

                if args.gradient_accumulation_steps > 1:
                    loss_adv = loss_adv / args.gradient_accumulation_steps
                scaler.scale(loss_adv).backward()

                model_adv.restore()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.step(optimizer)
                    # optimizer.step()
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                if ema_start:
                    model_ema.update()
                model.zero_grad()
                step += 1
                total_loss += loss.item()

                if args.use_ema and epoch >= 3 and not ema_start:
                    ema_start = True
                    model_ema.register()
                    logger.info("***** EMA START *****")

                if step % args.logging_steps == 0:
                    logger.info(f"Fold: {fold} Epoch {epoch} step {step}: loss {loss:.3f}")


        #if epoch >= 3:
        if ema_start:
            model_ema.apply_shadow()

        P, R, f1, losses = evaluate(args, valid_loader, val_loader, model,fold)

        logger.info(
            f"Fold: {fold} step [{step}|{num_total_steps}] epoch [{epoch}|{args.num_epochs}]: P {P:.3f} R {R:.3f} F1 {f1:.3f} loss {losses:.3f}")
        # save best checkpoint
        model_to_save = model.module if hasattr(model, "module") else model
        # best_score_pre = best_score
        state_dict = model_to_save.state_dict()
        if f1 > best_score:
            best_score = f1
            torch.save({'step': step, 'model_state_dict': state_dict, 'R': R, 'f1': f1},
                       f'{args.save_dir}/fold_{fold}.bin')
        if ema_start:
            model_ema.restore()
        model.train()



from bert_optimization import *





from optim import *
from modeling_nezha import *
from dataloader import *

if __name__ == '__main__':
    args = parse_args()
    print(args)
    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.dev_pred_dir, exist_ok=True)
    
    
    for i in range(args.folds):
        os.makedirs(args.save_dir + f'fold_{i}', exist_ok=True)

    device = 'cuda'
    train_datas = read_rwa_data()

    schema = {'部件故障': 0, '性能故障': 1, '检测工具': 2, '组成': 3}
    id2schema = {}
    for k, v in schema.items():
        id2schema[v] = k
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_path, do_lower_case=True)
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained('../../user_data/opensource_models/chinese-roberta-wwm-ext-large', do_lower_case=True) #针对于所有模型的词表一致

    all_data = np.array(train_datas) #所有训练集

    kf = KFold(n_splits=args.folds, random_state=42, shuffle=True)
    for n, (train_idx, valid_idx) in enumerate(kf.split(all_data)):

        waibui_data = get_waibu_data() #取出外部数据

        train_data = all_data[train_idx]
        val_data = all_data[valid_idx]

        delet_leak_val_data = np.array(del_leak_data(waibui_data, val_data)) #对外部数据过滤掉含有验证集的数据
        train_data = np.concatenate((train_data, delet_leak_val_data))

        train_data = np.array(get_huangchuang_data(train_data)) #整体进行滑窗
        val_data = get_huangchuang_data(val_data)


        print(len(train_data), len(val_data))
        train_data = data_generator(train_data, tokenizer, max_len=220,
                                    schema=schema)
        val_data2 = data_generator(val_data, tokenizer, max_len=220,
                                   schema=schema)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=args.num_workers, collate_fn=train_data.collate)

        val_loader = DataLoader(val_data2, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                num_workers=args.num_workers, collate_fn=val_data2.collate)

        if 'nezha' not in args.bert_path: 
            encoder = AutoModel.from_pretrained(args.bert_path)
        else:
            encoder = NeZhaModel.from_pretrained(args.bert_path)
            
        mention_detect = RawGlobalPointer(hiddensize=args.cls_hidden_size, ent_type_size=2, inner_dim=64).to(
            device) 
        s_o_head = RawGlobalPointer(hiddensize=args.cls_hidden_size, ent_type_size=len(args.schema), inner_dim=64,
                                    RoPE=True,
                                    tril_mask=False).to(
            device)
        s_o_tail = RawGlobalPointer(hiddensize=args.cls_hidden_size, ent_type_size=len(args.schema), inner_dim=64,
                                    RoPE=True,
                                    tril_mask=False).to(
            device)

        model = E2ENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)

        num_total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
        

        optimizer, scheduler = build_optimizer(args, model, num_total_steps)
        
        prefix = "team_weiguan_"
        args.model_name = 'GPLinker'
        logger = get_root_logger(file_name=args.log_dir + f"{prefix}_{args.model_name}.log")

        train(args, train_loader, val_loader, val_data, model, optimizer, scheduler, num_total_steps, device, logger, n)


