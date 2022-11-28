import json
import os
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--vote_dir', default="../prediction_result", type=str)
    return parser.parse_args()

args = parse_args()

subs = []
listdir = os.listdir(args.vote_dir)
for file in listdir:
    if 'json' not in file:
        continue
    print(file)
    subs.append(args.vote_dir + '/' + file)
print(len(subs))
print(subs)

threshold = int(len(subs) / 2)

out_path = args.vote_dir + '/' + 'merged.json'

id2spo = {}

for sub in subs:
    with open(sub,encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            if l['ID'] not in id2spo:
                id2spo[l['ID']] = l['spo_list']
            else:
                for spo in l['spo_list']:
                    id2spo[l['ID']].append(spo)


vote = []
with open(subs[0],encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        tmp_spo_list = []
        for spo in id2spo[l['ID']]:
            if id2spo[l['ID']].count(spo) >= threshold and spo not in tmp_spo_list:
                tmp_spo_list.append(spo)
        l['spo_list'] = tmp_spo_list
        vote.append(l)

with open(out_path, 'w',encoding='utf-8') as f:
    for l in vote:
        f.write(json.dumps(l, ensure_ascii=False) + '\n')



