import torch
import torch.nn as nn
# from torchcrf import CRF
from transformers import AutoModel


class E2ENet(nn.Module):
    def __init__(self, encoder, a, b, c):
        super(E2ENet, self).__init__()
        self.mention_detect = a  
        self.s_o_head = b 
        self.s_o_tail = c  
        self.encoder = encoder  

    def forward(self, batch_token_ids, batch_mask_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids)
        mention_outputs = self.mention_detect(outputs, batch_mask_ids)  
        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return mention_outputs, so_head_outputs, so_tail_outputs
