import torch.nn.functional as F
import torch

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss = p_loss*pad_mask
        q_loss = q_loss*pad_mask

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

if __name__ == '__main__':
    x = torch.tensor([
        [1,2,3],
        [4,3,5]
    ],dtype=float)
    y = torch.tensor([
        [1,4,3],
        [2,3,4]
    ],dtype=float)

    kl_loss = compute_kl_loss(x,y)
    print(kl_loss)