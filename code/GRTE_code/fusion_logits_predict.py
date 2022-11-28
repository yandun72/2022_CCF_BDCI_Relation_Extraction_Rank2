from bert4keras.tokenizers import Tokenizer
from result_generator import resultGenerator
from model import GRTE
from util import *
from tqdm import tqdm
import os
import gc
import json
from transformers import BertModel, BertConfig, BertPreTrainedModel
import torch

from config import parse_args

def extract_spo_list(args, tokenizer, id2predicate, id2label, label2id, model_list, batch_ex, batch_token_ids, batch_mask, test_mode=False):
    table_list = []
    for model in model_list:
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model.to("cuda")
        model.eval()

        with torch.no_grad():
            table_fold = model(batch_token_ids, batch_mask)
            table_fold = torch.softmax(table_fold, dim=-1)
            table_fold = table_fold.cpu().detach().numpy() 
#             if len(table_list)>=5:
#                 table_fold *= 1.2
            table_list.append(table_fold)
    table = sum(table_list)/len(table_list)
    torch.cuda.empty_cache()
    def get_pred_id(table, all_tokens):

        B, L, _, R, _ = table.shape
        res = []
        for i in range(B):
            res.append([])
        table_tmp = table
        table = table.argmax(axis=-1)  # BLLR
        all_loc = np.where(table != label2id["N/A"])
        #         print(all_loc)

        res_dict = []
        for i in range(B):
            res_dict.append([])
        for i in range(len(all_loc[0])):
            token_n = len(all_tokens[all_loc[0][i]])
            if token_n - 1 <= all_loc[1][i] \
                    or token_n - 1 <= all_loc[2][i] \
                    or 0 in [all_loc[1][i], all_loc[2][i]]:
                continue
            res_dict[all_loc[0][i]].append([all_loc[1][i], all_loc[2][i], all_loc[3][i]])

        for i in range(B):
            for l1, l2, r in res_dict[i]:
                max_score = -10000.0
                now_spo = []
                if table[i, l1, l2, r] == label2id["SS"]:
                    res[i].append([l1, l1, r, l2, l2])
                elif table[i, l1, l2, r] == label2id["SMH"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "SMT"] and l1_ == l1 and l2_ > l2 and (l1_ < l2 or l1 > l2_):
                            if test_mode and l1 > l2_ and r<=1:
                                continue
                            res[i].append([l1, l1, r, l2, l2_])
                            break

                elif table[i, l1, l2, r] == label2id["MMH"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "MMT"] and l1_ > l1 and l2_ > l2 and (l1_ < l2 or l1 > l2_):
                            if test_mode and l1 > l2_ and r<=1:
                                continue
                            res[i].append([l1, l1_, r, l2, l2_])
                            break


                elif table[i, l1, l2, r] == label2id["MSH"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "MST"] and l1_ > l1 and l2_ == l2 and (l1_ < l2 or l1 > l2_):
                            if test_mode and l1 > l2_ and r<=1:
                                continue
                            res[i].append([l1, l1_, r, l2, l2_])
                            break

        return res

    all_tokens = []
    for ex in batch_ex:
        tokens = tokenizer.tokenize(ex["text"], maxlen=args.max_len)
        all_tokens.append(tokens)

    res_id = get_pred_id(table, all_tokens)
    batch_spo = [[] for _ in range(len(batch_ex))]
    batch_spo1 = [[] for _ in range(len(batch_ex))]
    batch_spo2 = [[] for _ in range(len(batch_ex))]
    for b, ex in enumerate(batch_ex):
        text = ex["text"]
        tokens = all_tokens[b]
        mapping = tokenizer.rematch(text, tokens)
        tmp_dict={}
        tmp_dict_o={}
        for sh, st, r, oh, ot in res_id[b]:
            s = (mapping[sh][0], mapping[st][-1])
            o = (mapping[oh][0], mapping[ot][-1])

            batch_spo[b].append(
                ((s[0], s[1] + 1, text[s[0]:s[1] + 1]), id2predicate[str(r)], (o[0], o[1] + 1, text[o[0]:o[1] + 1]))
            )

            if ((o[0], o[1] + 1, text[o[0]:o[1] + 1]), id2predicate[str(r)]) not in tmp_dict_o:
                tmp_dict_o[((o[0], o[1] + 1, text[o[0]:o[1] + 1]), id2predicate[str(r)])] = \
                    [((s[0], s[1] + 1, text[s[0]:s[1] + 1]), id2predicate[str(r)], (o[0], o[1] + 1, text[o[0]:o[1] + 1]))]
            else:
                tmp_dict_o[((o[0], o[1] + 1, text[o[0]:o[1] + 1]), id2predicate[str(r)])].append(
                    ((s[0], s[1] + 1, text[s[0]:s[1] + 1]), id2predicate[str(r)], (o[0], o[1] + 1, text[o[0]:o[1] + 1])))


            if ((s[0], s[1] + 1, text[s[0]:s[1] + 1]), id2predicate[str(r)]) not in tmp_dict:
                tmp_dict[((s[0], s[1] + 1, text[s[0]:s[1] + 1]), id2predicate[str(r)])] = \
                    [((s[0], s[1] + 1, text[s[0]:s[1] + 1]), id2predicate[str(r)], (o[0], o[1] + 1, text[o[0]:o[1] + 1]))]
            else:
                tmp_dict[((s[0], s[1] + 1, text[s[0]:s[1] + 1]), id2predicate[str(r)])].append(
                    ((s[0], s[1] + 1, text[s[0]:s[1] + 1]), id2predicate[str(r)], (o[0], o[1] + 1, text[o[0]:o[1] + 1])))

        for k,v in tmp_dict_o.items():
            if len(v)>1:
                min_len = 10000
                idx = -1
                flag=[0]*len(v)
                for i in range(1,len(v)):
                    if v[i][0][0] < v[i-1][0][1]:
                        flag[i]=1
                        flag[i-1]=1
                for i in range(0,len(v)):
                    if flag[i]==0:
                        batch_spo1[b].append(v[i])
                    else:
                        if min_len > v[i][0][1]-v[i][0][0]:
                            min_len = v[i][0][1]-v[i][0][0]
                            idx = i
                if idx!=-1:
                    batch_spo1[b].append(v[idx])
            else:
                batch_spo1[b].extend(v)


        for k,v in tmp_dict.items():
            if len(v)>1:
                min_len = 10000
                idx = -1
                flag=[0]*len(v)
                for i in range(1,len(v)):
                    if v[i][2][0] < v[i-1][2][1]:
                        flag[i]=1
                        flag[i-1]=1
                for i in range(0,len(v)):
                    if flag[i]==0:
                        batch_spo2[b].append(v[i])
                    else:
                        if min_len > v[i][2][1]-v[i][2][0]:
                            min_len = v[i][2][1]-v[i][2][0]
                            idx = i
                if idx!=-1:
                    batch_spo2[b].append(v[idx])
            else:
                batch_spo2[b].extend(v)

        if test_mode:
            batch_spo[b] = list( set(batch_spo1[b]) & set(batch_spo2[b]) )

    return batch_spo

def evaluate(args, tokenizer, id2predicate, id2label, label2id, model_list, dataloader):
    # model.to("cuda")
    test_pred_path = os.path.join(args.result_path, args.result_name)
    f = open(test_pred_path, 'w', encoding='utf-8')
    total = {}
    for batch in tqdm(dataloader):
        batch_ex = batch[-1]
        batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
        batch_token_ids, batch_mask = batch
        batch_spo = extract_spo_list(args, tokenizer, id2predicate, id2label, label2id, model_list, batch_ex, batch_token_ids,
                                     batch_mask, test_mode=True)
        for i, ex in enumerate(batch_ex):
            R = batch_spo[i]
            id = ex['id']
            spo_list = list(R)
            triples = []
            for spo in spo_list:
                s = spo[0]
                p = spo[1]
                o = spo[2]
                triple = {"s": tuple(s), "p": p, "o": tuple(o)}
                triples.append(triple)
            total[id] = triples
    json.dump(total, f, ensure_ascii=False, indent=2)


def predict():
    output_path = os.path.join(args.output_path)
    test_path = '../../user_data/GRTE_data/' + args.test_file
    rel2id_path = '../../user_data/GRTE_data/rel2id.json'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # label
    label_list = ["N/A", "SMH", "SMT", "SS", "MMH", "MMT", "MSH", "MST"]
    id2label, label2id = {}, {}
    for i, l in enumerate(label_list):
        id2label[str(i)] = l
        label2id[l] = i

    test_data = json.load(open(test_path,encoding='utf-8'))
    id2predicate, predicate2id = json.load(open(rel2id_path,encoding='utf-8'))
    tokenizer = Tokenizer(args.bert_vocab_path)
    test_dataloader = data_generator(args, test_data, tokenizer, [predicate2id, id2predicate], [label2id, id2label],
                                     args.test_batch_size, random=False, is_train=False)


    model_list = []
    for fold in range(1, args.k_num+1):
        config = BertConfig.from_pretrained(args.pretrained_model_path)
        config.num_p = len(id2predicate)
        config.num_label = len(label_list)
        config.rounds = args.rounds
        config.fix_bert_embeddings = args.fix_bert_embeddings
        train_model = GRTE.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_path, config=config)
        train_model.to("cuda")
        save_dir = f"{args.output_path}"
#         if fold==6:
#             train_model.load_state_dict(torch.load(args.ckpt_file, map_location="cuda"))
#         else:
        train_model.load_state_dict(
                torch.load(f"{save_dir}/model_{fold}.pth", map_location="cuda"))
        model_list.append(train_model)
        
        train_model.load_state_dict(
                torch.load(f"{save_dir}/model_swa_{fold}.pth", map_location="cuda"))
        model_list.append(train_model)
        
        del train_model;gc.collect()
        torch.cuda.empty_cache()
    print(len(model_list))

#     config = BertConfig.from_pretrained('../autodl-tmp/chinese-macbert-large')
#     config.num_p = len(id2predicate)
#     config.num_label = len(label_list)
#     config.rounds = args.rounds
#     config.fix_bert_embeddings = args.fix_bert_embeddings
#     train_model = GRTE.from_pretrained(pretrained_model_name_or_path='../autodl-tmp/chinese-macbert-large', config=config)
#     print(torch.cuda.is_available())
#     train_model.to("cuda")
#     ckpt_file = f"../autodl-tmp/model_storage/macbert_large_all/model_swa.pth"
#     train_model.load_state_dict(torch.load(ckpt_file, map_location="cuda"))
#     model_list.append(train_model)
#     del train_model;gc.collect()
#     torch.cuda.empty_cache()

#     config = BertConfig.from_pretrained('../autodl-tmp/uer-large')
#     config.num_p = len(id2predicate)
#     config.num_label = len(label_list)
#     config.rounds = args.rounds
#     config.fix_bert_embeddings = args.fix_bert_embeddings
#     train_model = GRTE.from_pretrained(pretrained_model_name_or_path='../autodl-tmp/uer-large', config=config)
#     print(torch.cuda.is_available())
#     train_model.to("cuda")
#     ckpt_file = f"../autodl-tmp/model_storage/uer_large_all/model_swa.pth"
#     train_model.load_state_dict(torch.load(ckpt_file, map_location="cuda"))
#     model_list.append(train_model)
#     del train_model;gc.collect()
#     torch.cuda.empty_cache()
    
    evaluate(args, tokenizer, id2predicate, id2label, label2id, model_list, test_dataloader)


if __name__ == '__main__':
    args = parse_args()
    args.test_batch_size = 2
    args.max_len=384
    predict()

    rg = resultGenerator()
    rg.load_result()
    # rg.merge_k_fold()
    rg.merge_text()
