import argparse
import time
def update_args(args):
    time_str = f"{time.strftime('%m-%d-%H-%M')}"
    return args
def parse_args():
    schema = {'部件故障': 0, '性能故障': 1, '检测工具': 2, '组成': 3}
    id2schema = {}
    for k, v in schema.items():
        id2schema[v] = k

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--schema', type=dict, default=schema)
    parser.add_argument('--id2schema', type=dict, default=id2schema)
    
    
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")

    
    parser.add_argument('--bert_path', type=str, default='')
    parser.add_argument('--cls_hidden_size', type=int, default=1024)
    parser.add_argument('--max_len', type=int, default=220)
    parser.add_argument('--folds', type=int, default=5)

    
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    parser.add_argument('--use_fgm', type=bool, default=True)

    parser.add_argument('--use_ema', type=bool, default=True)


    parser.add_argument('--batch_size', default=8, type=int, help="use for training duration per worker")
    

    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--logging_steps', type=int, default=100)
    

    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='')
    parser.add_argument('--predict_dir', type=str, default='')
    parser.add_argument('--dev_pred_dir', type=str, default='')
    

    parser.add_argument('--learning_rate', type=float, default=3e-5)
    
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    parser.add_argument('--warmup_ratio', type=float, default=0.15)

    #args = parser.parse_args(args=[])
    args = parser.parse_args()
    args = update_args(args)
    return args
