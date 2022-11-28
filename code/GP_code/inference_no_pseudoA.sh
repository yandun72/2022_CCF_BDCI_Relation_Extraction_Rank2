# python predict_a_no_weibiao.py \
# --bert_path="../../user_data/opensource_models/chinese-roberta-wwm-ext-large" \
# --save_dir="../../user_data/gp_model_weight_no_pesudoA/1_roberta/output/ckpt/" \
# --log_dir="../../user_data/gp_model_weight_no_pesudoA/1_roberta/output/logs" \
# --dev_pred_dir="../../user_data/gp_model_weight_no_pesudoA/1_roberta/output/dev_pred" \
# --predict_dir="../../user_data/resultA_pseudo/gp_result_roberta_a.json" \

# python predict_a_no_weibiao.py \
# --bert_path="../../user_data/opensource_models/uer-large" \
# --save_dir="../../user_data/gp_model_weight_no_pesudoA/2_uer/output/ckpt/" \
# --log_dir="../../user_data/gp_model_weight_no_pesudoA/2_uer/output/logs" \
# --dev_pred_dir="../../user_data/gp_model_weight_no_pesudoA/2_uer/output/dev_pred" \
# --predict_dir="../../user_data/resultA_pseudo/gp_result_uer_a.json" \

# python predict_a_no_weibiao.py \
# --bert_path="../../user_data/opensource_models/chinese-macbert-large" \
# --save_dir="../../user_data/gp_model_weight_no_pesudoA/3_macbert/output/ckpt/" \
# --log_dir="../../user_data/gp_model_weight_no_pesudoA/3_macbert/output/logs" \
# --dev_pred_dir="../../user_data/gp_model_weight_no_pesudoA/3_macbert/output/dev_pred" \
# --predict_dir="../../user_data/resultA_pseudo/gp_result_macbert_a.json" \

python predict_a_no_weibiao.py \
--bert_path="../../user_data/opensource_models/nezha_wwm_xy" \
--save_dir="../../user_data/gp_model_weight_no_pesudoA/4_nezha_wwm/output/ckpt/" \
--log_dir="../../user_data/gp_model_weight_no_pesudoA/4_nezha_wwm/output/logs" \
--dev_pred_dir="../../user_data/gp_model_weight_no_pesudoA/4_nezha_wwm/output/dev_pred" \
--predict_dir="../../user_data/resultA_pseudo/gp_result_nezha_wwm_a.json" \

python predict_a_no_weibiao.py \
--bert_path="../../user_data/opensource_models/nezha_large" \
--save_dir="../../user_data/gp_model_weight_no_pesudoA/5_nezha_cn/output/ckpt/" \
--log_dir="../../user_data/gp_model_weight_no_pesudoA/5_nezha_cn/output/logs" \
--dev_pred_dir="../../user_data/gp_model_weight_no_pesudoA/5_nezha_cn/output/dev_pred" \
--predict_dir="../../user_data/resultA_pseudo/gp_result_nezha_large_cn_a.json"