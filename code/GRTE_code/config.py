import argparse

def str2bool(str):
    return True if str.lower() == 'true' else False

def parse_args():
    
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--rounds', default=4, type=int)
    parser.add_argument('--k_num', default=5, type=int)
    parser.add_argument('--max_len', default=200, type=int)
    parser.add_argument('--dataset', default='bdci', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--val_batch_size', default=4, type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--num_train_epochs', default=10, type=int)
    parser.add_argument('--fix_bert_embeddings', default=False, type=bool)
    
    # #roberta
    # parser.add_argument('--bert_vocab_path', default="../nas-private/chinese-roberta-wwm-ext-large/vocab.txt", type=str)
    # parser.add_argument('--pretrained_model_path', default="../nas-private/chinese-roberta-wwm-ext-large", type=str)

    # #macbert
    # parser.add_argument('--bert_vocab_path', default="../nas-private/chinese-macbert-large/vocab.txt", type=str)
    # parser.add_argument('--pretrained_model_path', default="../nas-private/chinese-macbert-large", type=str)

    # #uer
    # parser.add_argument('--bert_vocab_path', default="../nas-private/uer-large/vocab.txt", type=str)
    # parser.add_argument('--pretrained_model_path', default="../nas-private/uer-large", type=str)
    
    #nehza
    parser.add_argument('--bert_vocab_path', default="../nas-private/nezha-large-wwm/vocab.txt", type=str)
    parser.add_argument('--pretrained_model_path', default="../nas-private/nezha-large-wwm", type=str)
    
    # #pert
    # parser.add_argument('--bert_vocab_path', default="../nas-private/chinese-pert-large/vocab.txt", type=str)
    # parser.add_argument('--pretrained_model_path', default="../nas-private/chinese-pert-large", type=str)
    
    
    parser.add_argument('--warmup', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--min_num', default=1e-7, type=float)
    parser.add_argument('--base_path', default="data", type=str)
    
    parser.add_argument('--output_path', default="../../user_data/GRTE_model_storage/nezha_pseudoA", type=str)
    
    parser.add_argument('--train_file', default="train.json", type=str)
    parser.add_argument('--test_file', default="testB.json", type=str)
    parser.add_argument('--result_path', default="../../prediction_result", type=str)
    parser.add_argument('--result_name', default="GRTE_nezha_pseudo.json", type=str)
    parser.add_argument('--test_batch_size', default=4, type=int)
    
    parser.add_argument('--use_attack', default="None", type=str)
    parser.add_argument('--use_ema', default=True, type=bool)
    parser.add_argument('--ema_start', default=1, type=int)
    parser.add_argument('--use_swa', default=True, type=bool)
    parser.add_argument('--swa_start', default=7, type=int)
    parser.add_argument('--swa_end', default=9, type=int)
    parser.add_argument('--bert_learning_rate', default=2e-5, type=float)
    parser.add_argument('--threshold', default=3, type=int)
    parser.add_argument('--awp_start', default=2, type=int)
    parser.add_argument('--adv_lr', default=0.1, type=float)
    parser.add_argument('--adv_eps', default=0.002, type=float)
    
    parser.add_argument('--r_drop', default=False, type=bool)
    parser.add_argument('--kl_alpha', default=0.15, type=float)

    parser.add_argument("--use_ccl", type=str2bool, default='False')
    parser.add_argument("--use_pseudoA", type=str2bool, default='False')
    parser.add_argument("--merge_ccl", type=str2bool, default='False')
    
    parser.add_argument('--ckpt_file', default="../autodl-tmp/model_storage/roberta_large_all/model_swa.pth", type=str)
    
    
    return parser.parse_args()