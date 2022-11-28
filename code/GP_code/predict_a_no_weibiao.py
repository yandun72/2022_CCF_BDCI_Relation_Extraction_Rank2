import os
from tqdm import tqdm
from config import *
from utils import *
from Gpnet import *
from model import *
import json
import re
from torch.utils.data import Dataset, DataLoader
from modeling_nezha import *

def get_model_path_list(base_dir):
    """
    从文件夹中获取 model.pt 的路径
    """
    model_lists = []
    for root, dirs, files in os.walk(base_dir):
        for _file in files:
            if '.bin' in _file:
                model_lists.append(os.path.join(root, _file))
    return model_lists


def test_generator():
    cut_pattern = re.compile(r'([，。！？、])')
    file = open('../../raw_data/evalA.json', 'r', encoding='utf-8')
    datas = []
    for i in file.readlines():
        dic_single = {}
        arr_single = []
        case_data = json.loads(i)

        idx = case_data['ID']
        txt = case_data['text']
        if len(txt) > 510:
            split_blocks = cut_pattern.split(txt)
            split_blocks.append("")

            split_blocks = ["".join(i) for i in zip(split_blocks[0::2], split_blocks[1::2])]
            current_text = ""
            total_blocks = []
            for block in split_blocks:
                if len(current_text + block) > 510:
                    total_blocks.append(current_text)
                    current_text = block
                else:
                    current_text += block

            if len(current_text) > 0:
                total_blocks.append(current_text)

            for sub_idx, block in enumerate(total_blocks):
                line = {"id": str(idx) + "_{}".format(sub_idx), "text": block}
                datas.append(line)
        else:
            line = {"id": str(idx), "text": txt}
            datas.append(line)
    datas2 = []
    tmp = set()
    start = 0
    for i, x in enumerate(datas):
        if x['id'].split('_')[0] not in tmp:
            tmp.add(x['id'].split('_')[0])
            x['start'] = 0
            x['lens'] = len(x['text'])
        else:
            x['lens'] = len(x['text'])
            x['start'] = datas[i - 1]['start'] + datas[i - 1]['lens']
        datas2.append(x)
    arr_all2 = []
    for x in datas2:
        tmp = {}
        tmp['ID'] = x['id']
        tmp['text'] = x['text']
        tmp['start'] = x['start']
        tmp['lens'] = x['lens']
        arr_all2.append(tmp)
    print(len(arr_all2))
    return arr_all2
args = parse_args()
set_seed(42)

model_path = get_model_path_list(args.save_dir)
for x in model_path:
    print(x)
test_data = test_generator()
text_list = [(x['text'], x['ID'], x['start'], x['lens']) for x in test_data]
print(text_list[0:3])

test_data = test_generator()
scaler = torch.cuda.amp.GradScaler()
autocast = torch.cuda.amp.autocast


try:
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path, do_lower_case=True)
except Exception as e:
    tokenizer = AutoTokenizer.from_pretrained('../../user_data/opensource_models/chinese-roberta-wwm-ext-large', do_lower_case=True) #针对于所有模型的词表一致
if 'nezha' not in args.bert_path: 
    encoder = AutoModel.from_pretrained(args.bert_path)
else:
    encoder = NeZhaModel.from_pretrained(args.bert_path)
    
# subject_type表明subject是什么类型，object_type同理
schema = {'部件故障': 0, '性能故障': 1, '检测工具': 2, '组成': 3}
id2schema = {}
for k, v in schema.items():
    id2schema[v] = k
device = torch.device('cuda')

print(args.bert_path)
mention_detect = RawGlobalPointer(hiddensize=1024, ent_type_size=2, inner_dim=64).to(device)  # 实体关系抽取任务默认不提取实体类型
s_o_head = RawGlobalPointer(hiddensize=1024, ent_type_size=len(schema), inner_dim=64, RoPE=True,
                            tril_mask=False).to(device)
s_o_tail = RawGlobalPointer(hiddensize=1024, ent_type_size=len(schema), inner_dim=64, RoPE=True,
                            tril_mask=False).to(device)
model = E2ENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)

five_model_score = []
for paths in model_path:
    print(paths)
    check_point = torch.load(paths)
    model.load_state_dict(check_point['model_state_dict'])
    model.eval()
    this_model_pred = []
    for u, (text, id_, start, lens) in tqdm(enumerate(text_list)):
        encoder_txt = tokenizer.encode_plus(text, max_length=510)
        input_ids = torch.tensor(encoder_txt['input_ids']).long().unsqueeze(0).to(device)
        token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
        # print(text)
        # print(input_ids)
        with torch.no_grad():
            with autocast():
                scores = model(input_ids, attention_mask)
        tmp = [scores[j].cpu().numpy() for j in range(3)]
        this_model_pred.append(tmp)

    five_model_score.append(this_model_pred)
    
pred_spos = []
data = []
threshold = 0.0
for u, (text, id_, start, lens) in tqdm(enumerate(text_list)):

    result = {}
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=512)['offset_mapping']
    new_span, entities = [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])
    ner_tmp = (five_model_score[0][u][0] + five_model_score[1][u][0] + five_model_score[2][u][0]+ five_model_score[3][u][0] + five_model_score[4][u][0] ) / 5.0
    start_tmp = (five_model_score[0][u][1] + five_model_score[1][u][1] + five_model_score[2][u][1] + five_model_score[3][u][1] + five_model_score[4][u][1] ) / 5.0
    end_tmp = (five_model_score[0][u][2] + five_model_score[1][u][2] + five_model_score[2][u][2]+ five_model_score[3][u][2] + five_model_score[4][u][2]) / 5.0

    tmp = [ner_tmp, start_tmp, end_tmp]
    outputs = [o[0] for o in tmp]  # list类型，每个位置形状[ent_type_size, seq_len, seq_len]
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
    # print(subjects)
    # print(objects)
    spoes = set()
    for sh, st in subjects:
        for oh, ot in objects:
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]  # outputs[1]表示head,[:, sh, oh]查对应的4个关系里面对应的实体首是不是都大于0
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            ps = set(p1s) & set(p2s)  # 取交集，因为实体首部和尾部有可能被识别的关系类型不一样，取交集就可以保证了
            # print(ps,p2s,ps)
            # print(p1s)
            for p in ps:
                spoes.add((
                    text[new_span[sh][0]:new_span[st][-1] + 1], (new_span[sh][0], new_span[st][-1] + 1), id2schema[p],
                    text[new_span[oh][0]:new_span[ot][-1] + 1], (new_span[oh][0], new_span[ot][-1] + 1)
                ))
    pred_spos.append(list(spoes))
    spo_list = []
    result['ID'] = id_
    result['text'] = text
    result['start'] = start
    result['lens'] = lens

    for spo in list(spoes):
        spo_list.append({'h': {'name': spo[0], 'pos': list(spo[1])}, 't': {'name': spo[3], 'pos': list(spo[4])},
                         'relation': spo[2]})
    result["spo_list"] = spo_list
    # data.append(json.dumps(result, ensure_ascii=False))
    data.append(result)
print(data[0])

# data_not_split存放的是test_a中文本不是很长，没有被滑窗的样本
#data_split存放的是test_a中文本很长，被滑窗的样本

data_not_split = []
data_split = []
for x in data:
    if '_' not in x['ID']:
        data_not_split.append(x)
    else:
        data_split.append(x)
print(len(data_not_split), len(data_split))


'''
#####################################################################################################################################################
对被滑窗的预测样本按照原始的ID进行合并回原始的样本
'''
keys = {}
for x in data_split:
    # print(x['ID'].split('_')[0])
    if x['ID'].split('_')[0] not in keys:
        keys[x['ID'].split('_')[0]] = {'cnt': 1, 'pred': [x]}

    else:
        keys[x['ID'].split('_')[0]]['cnt'] += 1
        keys[x['ID'].split('_')[0]]['pred'].append(x)
data_split_merge = []
for k in keys:
    cnt = keys[k]['cnt']
    tmp = {'ID': k}
    text = keys[k]['pred'][0]['text']
    spo_list = keys[k]['pred'][0]['spo_list']

    for c in range(1, cnt):
        tt = keys[k]['pred'][c]
        tmp_spo = []
        lens = len(text)
        for spo in tt['spo_list']:
            ww = {
                'h': {'name': spo['h']['name'],
                      'pos': [spo['h']['pos'][0] + lens, spo['h']['pos'][1] + lens]},
                't': {'name': spo['t']['name'],
                      'pos': [spo['t']['pos'][0] + lens, spo['t']['pos'][1] + lens]},
                'relation': spo['relation']
            }
            tmp_spo.append(ww)
        text += tt['text']
        spo_list.extend(tmp_spo)
    tmp['text'] = text
    tmp['spo_list'] = spo_list
    data_split_merge.append(tmp)
print(len(data_split_merge))
'''
#####################################################################################################################################################
对被滑窗的预测样本按照原始的ID进行合并回原始的样本
'''

all_res = []

for x in data_split_merge + data_not_split:
    all_res.append(json.dumps(x, ensure_ascii=False))
print(len(all_res))
with open(args.predict_dir, 'w', encoding='utf-8') as w:
    for line in all_res:
        w.write(line)
        w.write('\n')

print('Finish!!!!')