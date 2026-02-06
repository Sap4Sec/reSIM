"""
Custom trainer and collator for margin ranking loss.
"""

from typing import Dict, Any, Optional, Tuple

import torch
from torch import nn
from transformers import Trainer


class MarginRankingTrainer(Trainer):
    def __init__(self, *args, margin=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_loss = nn.MarginRankingLoss(margin=margin)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = torch.cat([
            inputs["input_ids_pos"], 
            inputs["input_ids_neg"]
        ], dim=0)
        
        attention_mask = torch.cat([
            inputs["attention_mask_pos"], 
            inputs["attention_mask_neg"]
        ], dim=0)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)
        
        batch_size = inputs["input_ids_pos"].size(0)
        s_pos = logits[:batch_size]
        s_neg = logits[batch_size:]
        
        target = torch.ones_like(s_pos)
        loss = self.rank_loss(s_pos, s_neg, target)
        
        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        return (loss, None, None)


class MarginRankingCollator:
    
    def __init__(self, tokenizer, padding=True, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        batch = {
            "input_ids_pos": torch.stack([f["input_ids_pos"] for f in features]),
            "attention_mask_pos": torch.stack([f["attention_mask_pos"] for f in features]),
            "input_ids_neg": torch.stack([f["input_ids_neg"] for f in features]),
            "attention_mask_neg": torch.stack([f["attention_mask_neg"] for f in features]),
        }
        
        # Ensure tensors don't have gradients
        for key in batch:
            batch[key] = batch[key].detach()
            
        return batch
