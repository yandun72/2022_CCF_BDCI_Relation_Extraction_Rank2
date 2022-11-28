---
license: afl-3.0
---
**Please use 'Bert' related tokenizer classes and 'Nezha' related model classes**

[NEZHA: Neural Contextualized Representation for Chinese Language Understanding](https://arxiv.org/abs/1909.00204)
Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, Yasheng Wang, Jiashu Lin, Xin Jiang, Xiao Chen and Qun Liu.

The original checkpoints can be found [here](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-PyTorch)

## Example Usage

```
from transformers import BertTokenizer, NezhaModel
tokenizer = BertTokenizer.from_pretrained("sijunhe/nezha-large-wwm")
model = NezhaModel.from_pretrained("sijunhe/nezha-large-wwm")
text = "我爱北京天安门"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```