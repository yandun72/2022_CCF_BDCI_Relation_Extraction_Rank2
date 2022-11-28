import json

from config import parse_args
args = parse_args()

class resultGenerator:
    def __init__(self):
        self.result = {}

    def load_result(self):
        with open(args.result_path + '/' +  args.result_name, 'r', encoding='utf8') as f:
            datas = eval(f.read())
            self.result = datas

        origin_dataset = open('../../user_data/GRTE_data' + '/' + args.test_file, 'r', encoding='utf8')
        origins = json.load(origin_dataset)
        self.id2text = {}
        for data in origins:
            id = data['id']
            text = data['text']
            self.id2text[id] = text

    def merge_text(self):
        # 将切分的句子进行合并
        final = {}
        for k, v in self.result.items():
            text = self.id2text[k]
            real_id = k.split("_")[0]
            if real_id in final:
                final[real_id].append({"text": text, "spos": v})
            else:
                final[real_id] = [{"text": text, "spos": v}]

        fout = open(args.result_path + '/' +  args.result_name, 'w', encoding='utf8')
        for k, vv in final.items():
            if len(vv) == 1:
                text = vv[0]['text']
                spo_list = []
                spos = vv[0]['spos']
                for spo in spos:
                    s_sidx, s_eidx, s_entity = spo['s']
                    p = spo['p']
                    o_sidx, o_eidx, o_entity = spo['o']

                    one = {"h": {"name": s_entity, "pos": [s_sidx, s_eidx]}, "t": {"name": o_entity, "pos": [o_sidx, o_eidx]}, "relation": p}
                    spo_list.append(one)
                line = {"ID": k, "text": text, "spo_list": spo_list}
                fout.write(json.dumps(line, ensure_ascii=False)+"\n")
            elif len(vv) > 1:
                spo_list = []
                total_text = ""
                for v_idx, v in enumerate(vv):
                    text = v['text']
                    spos = v['spos']
                    if v_idx == 0:
                        for spo in spos:
                            s_sidx, s_eidx, s_entity = spo['s']
                            p = spo['p']
                            o_sidx, o_eidx, o_entity = spo['o']

                            one = {"h": {"name": s_entity, "pos": [s_sidx, s_eidx]},
                                   "t": {"name": o_entity, "pos": [o_sidx, o_eidx]}, "relation": p}
                            spo_list.append(one)
                    else:
                        for spo in spos:
                            s_sidx, s_eidx, s_entity = spo['s']
                            p = spo['p']
                            o_sidx, o_eidx, o_entity = spo['o']

                            one = {"h": {"name": s_entity, "pos": [s_sidx+len(total_text), s_eidx+len(total_text)]},
                                   "t": {"name": o_entity, "pos": [o_sidx+len(total_text), o_eidx+len(total_text)]}, "relation": p}
                            spo_list.append(one)
                    total_text += text
                line = {"ID": k, "text": total_text, "spo_list": spo_list}
                fout.write(json.dumps(line, ensure_ascii=False)+"\n")


