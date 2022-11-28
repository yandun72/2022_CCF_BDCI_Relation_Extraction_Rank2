
import json
import os
ccl_path = '../../user_data/train_ccl2022.json'
ccl_train_path = '../../user_data/GRTE_data/train_ccl.json'
# ccl_test_path = './data/bdci/test_ccl2022_zxc.json'

fw1 = open(ccl_train_path, 'w', encoding='utf8')
# fw2 = open(ccl_test_path, 'w', encoding='utf8')

def get_id(num):
    if num < 10:
        return 'CCL000' + str(num)
    elif num <100:
        return 'CCL00' + str(num)
    elif num <1000:
        return 'CCL0' + str(num)
    else:
        return 'CCL' + str(num)
num = 1
with open(ccl_path,encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        del data['h']['id']
        del data['t']['id']
        train_data ={
            'ID':get_id(num),
            'text':data['text'],
            'spo_list':[
                {
                    'h':data['h'],
                    't':data['t'],
                    'relation':data['relation']
                }
            ]
        }

        test_data ={
            'ID':get_id(num),
            'text':data['text'],
        }
        num += 1
        fw1.write(json.dumps(train_data, ensure_ascii=False)+"\n")
        # fw2.write(json.dumps(test_data, ensure_ascii=False)+"\n")


