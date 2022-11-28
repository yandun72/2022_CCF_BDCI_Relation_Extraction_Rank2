#当前目录：存放了post_process.py 和 vote.py的地方


# step1 GRTE训练4个五折
cd GRTE_code
sh train_no_pseudoA.sh


# step2 GRTE预测，4个结果存放结果和GP一致，用于后续的投票和后处理
sh inference_no_pseudoA.sh


#step3 GP训练5个五折
cd ../GP_code
sh train_no_pseudoA.sh


#step4 GP预测
sh inference_no_pseudoA.sh


#step5 投票
cd ..
python vote.py \
--vote_dir="../user_data/resultA_pseudo"

#step6 后处理
python post_process.py \
--file_path="../user_data/resultA_pseudo/merged.json" \
--final_path="../user_data/resultA_pseudo/A_pseudo.json"