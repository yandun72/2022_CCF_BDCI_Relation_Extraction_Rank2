import json
import re
import os

def train_generator():
    fr = open('../../user_data/train_ccl2022.json', encoding='utf8').readlines()
    fw = open('../../user_data/GRTE_data/train_ccl_no_merge.json', 'w', encoding='utf8')
    num = 0
    arr_all = []
    for i in fr:
        i = i.strip()
        if i == "":
            continue

        dic_single = {}
        arr_single = []

        data = json.loads(i)
        text = data['text']
        spo_list = [
            {
                'h':data['h'],
                't':data['t'],
                'relation':data['relation']
            }
        ]

        for value in spo_list:
            if value['h']['name'][0] == " ":
                value['h']['name'] = value['h']['name'][1:]
                value['h']['pos'][0] = value['h']['pos'][0]+1
            if value['t']['name'][0] == " ":
                value['t']['name'] = value['t']['name'][1:]
                value['t']['pos'][0] = value['t']['pos'][0] + 1

        dic_single['text'] = text
        dic_single['spos'] = []

        if text in arr_all:
            continue

        if len(text) > 200:
            num+=1
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
            if total_blocks[0] == '':
                total_blocks = total_blocks[1:]
            start_idx = 0
            end_idx = 0
            arr_current = []
            for t_idx, block_text in enumerate(total_blocks):

                end_idx += len(block_text)
                new_spos = []
                for spo in spos:

                    h_sidx, h_eidx, h_name = spo[0]
                    t_sidx, t_eidx, t_name = spo[2]

                    if start_idx <= h_eidx < end_idx and start_idx <= t_eidx <= end_idx:
                        new_spos.append(spo)

                if t_idx == 0:
                    line = {"text": block_text, "spos": new_spos}
                    arr_current.append(line)

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

                    line = {"text": block_text, "spos": new_spos2}
                    arr_current.append(line)
                start_idx = end_idx
            arr_all.append(arr_current)

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
                dic_single['spos'].append(arr_spo)
            arr_all.append([dic_single])
    print(num)
    fw.writelines(json.dumps(arr_all, ensure_ascii=False, indent=2))


rel2id_path = os.path.join("../../user_data/GRTE_data/rel2id.json")
id2predicate, predicate2id = json.load(open(rel2id_path,encoding='utf8'))

def get_spo_key(spo):
    return (spo['h']['pos'][0], spo['h']['pos'][1], predicate2id[spo['relation']], spo['t']['pos'][0], spo['t']['pos'][1])

def solve(data):
    spo_num_dict = {}
    spo_tmp_dict = {}

    for spo in data['spo_list']:
        spo_key = get_spo_key(spo)
        if spo_key not in spo_num_dict:
            spo_num_dict[spo_key] = 0
            spo_tmp_dict[spo_key] = spo

        spo_num_dict[spo_key] += 1

    train_ccl_path = '../../user_data/GRTE_data/train_ccl.json'
    with open(train_ccl_path,encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data_ccl = json.loads(line)
            if data_ccl['text'] != data['text']:
                continue
            for spo in data_ccl['spo_list']:
                spo_key = get_spo_key(spo)
                if spo_key not in spo_num_dict:
                    spo_num_dict[spo_key] = 0
                    spo_tmp_dict[spo_key] = spo

                spo_num_dict[spo_key] += 1

    merged_spo_list = []
    for spo_key,num in spo_num_dict.items():
        if num>=1:
            spo = spo_tmp_dict[spo_key]
            merged_spo_list.append(spo)
    data['spo_list'] = merged_spo_list
    return data

def merge_ccl():
    ccl_path = '../../user_data/GRTE_data/train_ccl.json'
    merged_path = "../../user_data/GRTE_data/train_ccl_tmp.json"

    fw = open(merged_path, 'w', encoding='utf8')
    with open(ccl_path,encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            data = solve(data)

            fw.write(json.dumps(data, ensure_ascii=False)+"\n")

def train_generator_ccl_merge():
    fr = open('../../user_data/GRTE_data/train_ccl_tmp.json', encoding='utf8').readlines()
    fw = open('../../user_data/GRTE_data/train_ccl.json', 'w', encoding='utf8')

    arr_all = []
    text_all = []
    for i in fr:
        i = i.strip()
        if i == "":
            continue

        dic_single = {}
        arr_single = []

        data = json.loads(i)
        id = data['ID']
        text = data['text']
        spo_list = data['spo_list']

        for value in spo_list:
            if value['h']['name'][0] == " ":
                value['h']['name'] = value['h']['name'][1:]
                value['h']['pos'][0] = value['h']['pos'][0]+1
            if value['t']['name'][0] == " ":
                value['t']['name'] = value['t']['name'][1:]
                value['t']['pos'][0] = value['t']['pos'][0] + 1
        dic_single['id'] = id
        dic_single['text'] = text
        dic_single['spos'] = []

        if text in text_all:
            continue
        text_all.append(text)

        if len(text) > 200:
            for spo in spo_list:
                h = spo['h']
                t = spo['t']
                relation = spo['relation']
                line = [(h['pos'][0], h['pos'][1], h['name']), relation, (t['pos'][0], t['pos'][1], t['name'])]
                arr_single.append(line)

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
            if total_blocks[0] == '':
                total_blocks = total_blocks[1:]
            start_idx = 0
            end_idx = 0
            arr_current = []
            for t_idx, block_text in enumerate(total_blocks):

                end_idx += len(block_text)
                new_spos = []
                for spo in spos:

                    h_sidx, h_eidx, h_name = spo[0]
                    t_sidx, t_eidx, t_name = spo[2]

                    if start_idx <= h_eidx < end_idx and start_idx <= t_eidx <= end_idx:
                        new_spos.append(spo)

                if t_idx == 0:
                    line = {"id": id, "text": block_text, "spos": new_spos}
                    arr_current.append(line)

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

                    line = {"id": id, "text": block_text, "spos": new_spos2}
                    arr_current.append(line)
                start_idx = end_idx
            arr_all.append(arr_current)


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
                dic_single['spos'].append(arr_spo)

            arr_all.append([dic_single])

    fw.writelines(json.dumps(arr_all, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    cut_pattern = re.compile(r'([，。！？])')
    train_generator()

    merge_ccl()
    train_generator_ccl_merge()