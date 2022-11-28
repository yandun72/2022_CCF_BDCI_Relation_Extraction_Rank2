#准备数据GRTE数据
#=============================================================================================================
python data_generator.py
python process_ccl_data.py
python data_generator_ccl.py
#=============================================================================================================

# uer add ccl
python train.py \
--pretrained_model_path="../../user_data/opensource_models/uer-large" \
--bert_vocab_path="../../user_data/opensource_models/uer-large/vocab.txt" \
--output_path="../../user_data/GRTE_model_storage_tmp/uer_tmp" \
--use_ccl='True' \
--merge_ccl='True'

#roberta add ccl
python train.py \
--pretrained_model_path="../../user_data/opensource_models/chinese-roberta-wwm-ext-large" \
--bert_vocab_path="../../user_data/opensource_models/chinese-roberta-wwm-ext-large/vocab.txt" \
--output_path="../../user_data/GRTE_model_storage_tmp/roberta_tmp" \
--use_ccl='True' \
--merge_ccl='True'

#macbert add ccl
python train.py \
--pretrained_model_path="../../user_data/opensource_models/chinese-macbert-large" \
--bert_vocab_path="../../user_data/opensource_models/chinese-macbert-large/vocab.txt" \
--output_path="../../user_data/GRTE_model_storage_tmp/macbert_tmp" \
--use_ccl='True' \
--merge_ccl='True'

#nezha add ccl
python train_nezha.py \
--pretrained_model_path="../../user_data/opensource_models/nezha-large-wwm" \
--bert_vocab_path="../../user_data/opensource_models/nezha-large-wwm/vocab.txt" \
--output_path="../../user_data/GRTE_model_storage_tmp/nezha_tmp" \
--use_ccl='True' \
--merge_ccl='True'