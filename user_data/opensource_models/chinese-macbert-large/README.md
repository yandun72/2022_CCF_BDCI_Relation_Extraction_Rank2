---
language: 
- zh
tags:
- bert
license: "apache-2.0"
---
<p align="center">
    <br>
    <img src="https://github.com/ymcui/MacBERT/raw/master/pics/banner.png" width="500"/>
    <br>
</p>
<p align="center">
    <a href="https://github.com/ymcui/MacBERT/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/MacBERT.svg?color=blue&style=flat-square">
    </a>
</p>

# Please use 'Bert' related functions to load this model!

This repository contains the resources in our paper **"Revisiting Pre-trained Models for Chinese Natural Language Processing"**, which will be published in "[Findings of EMNLP](https://2020.emnlp.org)". You can read our camera-ready paper through [ACL Anthology](#) or [arXiv pre-print](https://arxiv.org/abs/2004.13922).

**[Revisiting Pre-trained Models for Chinese Natural Language Processing](https://arxiv.org/abs/2004.13922)**  
*Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Shijin Wang, Guoping Hu*

You may also interested in,
- Chinese BERT series: https://github.com/ymcui/Chinese-BERT-wwm
- Chinese ELECTRA: https://github.com/ymcui/Chinese-ELECTRA
- Chinese XLNet: https://github.com/ymcui/Chinese-XLNet
- Knowledge Distillation Toolkit - TextBrewer: https://github.com/airaria/TextBrewer

More resources by HFL: https://github.com/ymcui/HFL-Anthology

## Introduction
**MacBERT** is an improved BERT with novel **M**LM **a**s **c**orrection pre-training task, which mitigates the discrepancy of pre-training and fine-tuning.

Instead of masking with [MASK] token, which never appears in the ﬁne-tuning stage, **we propose to use similar words for the masking purpose**. A similar word is obtained by using [Synonyms toolkit (Wang and Hu, 2017)](https://github.com/chatopera/Synonyms), which is based on word2vec (Mikolov et al., 2013) similarity calculations. If an N-gram is selected to mask, we will ﬁnd similar words individually. In rare cases, when there is no similar word, we will degrade to use random word replacement.

Here is an example of our pre-training task.
|  | Example       |
| -------------- | ----------------- |
| **Original Sentence**  | we use a language model to predict the probability of the next word. |
|  **MLM** | we use a language [M] to [M] ##di ##ct the pro [M] ##bility of the next word . |
| **Whole word masking**   | we use a language [M] to [M] [M] [M] the [M] [M] [M] of the next word . |
| **N-gram masking** | we use a [M] [M] to [M] [M] [M] the [M] [M] [M] [M] [M] next word . |
| **MLM as correction** | we use a text system to ca ##lc ##ulate the po ##si ##bility of the next word . |

Except for the new pre-training task, we also incorporate the following techniques.

- Whole Word Masking (WWM)
- N-gram masking
- Sentence-Order Prediction (SOP)

**Note that our MacBERT can be directly replaced with the original BERT as there is no differences in the main neural architecture.**

For more technical details, please check our paper: [Revisiting Pre-trained Models for Chinese Natural Language Processing](https://arxiv.org/abs/2004.13922)


## Citation
If you find our resource or paper is useful, please consider including the following citation in your paper.
- https://arxiv.org/abs/2004.13922
```
@inproceedings{cui-etal-2020-revisiting,
    title = "Revisiting Pre-Trained Models for {C}hinese Natural Language Processing",
    author = "Cui, Yiming  and
      Che, Wanxiang  and
      Liu, Ting  and
      Qin, Bing  and
      Wang, Shijin  and
      Hu, Guoping",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.58",
    pages = "657--668",
}
```