from transformers import AdamW
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
def build_optimizer(args, model, num_total_steps):

    no_decay = ["bias", "LayerNorm.weight"]


    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-6)

    warmup_steps = int(num_total_steps*args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_total_steps, num_warmup_steps=warmup_steps)
    
    return optimizer, scheduler
