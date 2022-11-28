python fusion_logits_predict.py \
--pretrained_model_path="../../user_data/opensource_models/uer-large" \
--bert_vocab_path="../../user_data/opensource_models/uer-large/vocab.txt" \
--output_path="../../user_data/GRTE_model_storage_tmp/uer_tmp" \
--test_file="testA.json" \
--result_path="../../user_data/resultA_pseudo" \
--result_name="GRTE_uer.json"

python fusion_logits_predict.py \
--pretrained_model_path="../../user_data/opensource_models/chinese-roberta-wwm-ext-large" \
--bert_vocab_path="../../user_data/opensource_models/chinese-roberta-wwm-ext-large/vocab.txt" \
--output_path="../../user_data/GRTE_model_storage_tmp/roberta_tmp" \
--test_file="testA.json" \
--result_path="../../user_data/resultA_pseudo" \
--result_name="GRTE_roberta.json"


python fusion_logits_predict.py \
--pretrained_model_path="../../user_data/opensource_models/chinese-macbert-large" \
--bert_vocab_path="../../user_data/opensource_models/chinese-macbert-large/vocab.txt" \
--output_path="../../user_data/GRTE_model_storage_tmp/macbert_tmp" \
--test_file="testA.json" \
--result_path="../../user_data/resultA_pseudo" \
--result_name="GRTE_macbert.json"

python fusion_logits_predict_nezha.py \
--pretrained_model_path="../../user_data/opensource_models/nezha-large-wwm" \
--bert_vocab_path="../../user_data/opensource_models/nezha-large-wwm/vocab.txt" \
--output_path="../../user_data/GRTE_model_storage_tmp/nezha_tmp" \
--test_file="testA.json" \
--result_path="../../user_data/resultA_pseudo" \
--result_name="GRTE_nezha.json"