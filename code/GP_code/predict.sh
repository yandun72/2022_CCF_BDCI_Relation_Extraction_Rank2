#!/usr/bin/env bash

#=============================================================================================================
# GRTE predict
cd ../data/code/GRTE_code

# uer no ccl no A pseudo
python fusion_logits_predict.py \
--pretrained_model_path="../../user_data/opensource_models/uer-large" \
--bert_vocab_path="../../user_data/opensource_models/uer-large/vocab.txt" \
--output_path="../../user_data/GRTE_model_storage/uer_no_ccl_no_A" \
--result_name="GRTE_uer_no_ccl_no_A.json"

# roberta no ccl no A pseudo
python fusion_logits_predict.py \
--pretrained_model_path="../../user_data/opensource_models/chinese-roberta-wwm-ext-large" \
--bert_vocab_path="../../user_data/opensource_models/chinese-roberta-wwm-ext-large/vocab.txt" \
--output_path="../../user_data/GRTE_model_storage/roberta_no_ccl_no_A" \
--result_name="GRTE_roberta_no_ccl_no_A.json"

#roberta add ccl add A pseudo
python fusion_logits_predict.py \
--pretrained_model_path="../../user_data/opensource_models/chinese-roberta-wwm-ext-large" \
--bert_vocab_path="../../user_data/opensource_models/chinese-roberta-wwm-ext-large/vocab.txt" \
--output_path="../../user_data/GRTE_model_storage/roberta_pseudoA" \
--result_name="GRTE_roberta_pseudoA.json"

#uer add ccl add A pseudo
python fusion_logits_predict.py \
--pretrained_model_path="../../user_data/opensource_models/uer-large" \
--bert_vocab_path="../../user_data/opensource_models/uer-large/vocab.txt" \
--output_path="../../user_data/GRTE_model_storage/uer_pseudoA" \
--result_name="GRTE_uer_pseudoA.json"

#nezha add ccl add A pseudo
python fusion_logits_predict_nezha.py \
--pretrained_model_path="../../user_data/opensource_models/nezha-large-wwm" \
--bert_vocab_path="../../user_data/opensource_models/nezha-large-wwm/vocab.txt" \
--output_path="../../user_data/GRTE_model_storage/nezha_pseudoA" \
--result_name="GRTE_nezha_pseudoA.json"

#=============================================================================================================
# GP predict
cd ../Global_code
ls
#2_uer_weibiao_7fold
cd 2_uer_weibiao_7fold
python predict.py
cd ..

#31_roberta_7fold_weibiao
cd 31_roberta_7fold_weibiao
python predict.py
cd ..

#32_macbert_egp__ccl_a_weibiao_7fold
cd 32_macbert_egp__ccl_a_weibiao_7fold
python predict.py
cd ..

#5_anew_add_entity_
cd 5_anew_add_entity_
python predict.py
cd ..

#7_Roberta
cd 7_Roberta
python predict.py
cd ..

#32_Nezha_ccl_a_gp_7fold
cd 32_Nezha_ccl_a_gp_7fold
python predict.py
cd ..

#=============================================================================================================

# vote
cd ..
python vote.py
#post process
python post_process.py

#=============================================================================================================