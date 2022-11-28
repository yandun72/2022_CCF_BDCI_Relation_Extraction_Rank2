import json
import os
import argparse

def post1(data):
    spo_list = data['spo_list']
    ans = []
    for spo in spo_list:
        h_pos = spo['h']['pos']
        t_pos = spo['t']['pos']
        if t_pos[1] <= h_pos[0] or t_pos[0] >= h_pos[1]:
            ans.append(spo)
        else:
            print(spo)
    data['spo_list'] = ans
    return data

def post2(data):
    spo_list = data['spo_list']
    ans = []
    for spo in spo_list:
        h_pos = spo['h']['pos']
        t_pos = spo['t']['pos']
        if h_pos[0] > t_pos[1] and (spo['relation']=='部件故障' or spo['relation']=='性能故障'):
            print(spo)
            continue
        ans.append(spo)
    data['spo_list'] = ans
    return data

rel2id_path = os.path.join("../user_data/GRTE_data/rel2id.json")
id2predicate, predicate2id = json.load(open(rel2id_path,encoding='utf8'))

def get_spo_key(spo):
    return (spo['h']['pos'][0], spo['h']['pos'][1], predicate2id[spo['relation']], spo['t']['pos'][0], spo['t']['pos'][1])

def post3(data):
    spo_list = data['spo_list']
    spo_dict = {}
    ans = []
    for spo in spo_list:
        h_pos = spo['h']['pos']
        t_pos = spo['t']['pos']

        if (h_pos[1],t_pos[0]) not in spo_dict or spo['relation'] == '组成':
            ans.append(spo)
            spo_dict[(h_pos[1],t_pos[0])]=len(ans)-1
        else:
            idx = spo_dict[(h_pos[1],t_pos[0])]
            tmp_spo = ans[idx]
            tmp_h_pos = tmp_spo['h']['pos']
            tmp_t_pos = tmp_spo['t']['pos']
            if tmp_t_pos[1]-tmp_h_pos[0] > t_pos[1]-h_pos[0]:
                print(tmp_spo)
                print(spo)
                ans[idx] = spo

    data['spo_list'] = ans


    spo_list = data['spo_list']
    spo_dict = {}
    ans = []
    for spo in spo_list:
        h_pos = spo['h']['pos']
        t_pos = spo['t']['pos']

        if (h_pos[0],t_pos[1]) not in spo_dict or spo['relation'] == '组成':
            ans.append(spo)
            spo_dict[(h_pos[0],t_pos[1])]=len(ans)-1
        else:
            idx = spo_dict[(h_pos[0],t_pos[1])]
            tmp_spo = ans[idx]
            tmp_h_pos = tmp_spo['h']['pos']
            tmp_t_pos = tmp_spo['t']['pos']
            if tmp_t_pos[0]-tmp_h_pos[1] < t_pos[0]-h_pos[1]:
                print(tmp_spo)
                print(spo)
                ans[idx] = spo

    data['spo_list'] = ans

    return data

def judge_space(text):
    space_num = 0
    for s in text:
        if s==' ':
            space_num+=1
    return space_num

def post4(data):
    spo_list = []
    for spo in data['spo_list']:
        text_h = spo['h']['name']
        text_t = spo['t']['name']
        if judge_space(text_h)>0 or judge_space(text_t)>0:
            print(text_h,"   ",text_t)
            continue
        spo_list.append(spo)
    data['spo_list'] = spo_list
    return data

def post5(data):
    spo_list = data['spo_list']
    spo_dict = {}
    ans = []
    for spo in spo_list:
        h_pos = spo['h']['pos']
        t_pos = spo['t']['pos']

        if (h_pos[1],t_pos[0]) not in spo_dict or not (t_pos[1] <= h_pos[0] or t_pos[0] >= h_pos[1]):
            ans.append(spo)
            spo_dict[(h_pos[1],t_pos[0])]=len(ans)-1
        else:
            idx = spo_dict[(h_pos[1],t_pos[0])]
            tmp_spo = ans[idx]
            tmp_h_pos = tmp_spo['h']['pos']
            tmp_t_pos = tmp_spo['t']['pos']
            if tmp_t_pos[1]-tmp_h_pos[0] < t_pos[1]-h_pos[0]:
                print(tmp_spo)
                print(spo)
                ans[idx] = spo

    data['spo_list'] = ans


    spo_list = data['spo_list']
    spo_dict = {}
    ans = []
    for spo in spo_list:
        h_pos = spo['h']['pos']
        t_pos = spo['t']['pos']

        if (h_pos[0],t_pos[1]) not in spo_dict or not (t_pos[1] <= h_pos[0] or t_pos[0] >= h_pos[1]):
            ans.append(spo)
            spo_dict[(h_pos[0],t_pos[1])]=len(ans)-1
        else:
            idx = spo_dict[(h_pos[0],t_pos[1])]
            tmp_spo = ans[idx]
            tmp_h_pos = tmp_spo['h']['pos']
            tmp_t_pos = tmp_spo['t']['pos']
            if tmp_t_pos[0]-tmp_h_pos[1] > t_pos[0]-h_pos[1]:
                print(tmp_spo)
                print(spo)
                ans[idx] = spo

    data['spo_list'] = ans
    return data

def post6(data):
    spo_list = data['spo_list']
    ans = []
    for spo in spo_list:
        h_pos = spo['h']['pos']
        t_pos = spo['t']['pos']
        if t_pos[0] - h_pos[1] >= 30:
            print(spo)
        else:
            ans.append(spo)
    data['spo_list'] = ans
    return data

def parse_args():
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--file_path', default="../prediction_result/merged.json", type=str)
    parser.add_argument('--final_path', default="../prediction_result/result.json", type=str)
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    fw = open(args.final_path, 'w', encoding='utf8')
    with open(args.file_path,encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            data = post1(data)
            # data = post2(data)
            data = post3(data)
            data = post4(data)
            # data = post5(data)
            # data = post6(data)
            fw.write(json.dumps(data, ensure_ascii=False)+"\n")

