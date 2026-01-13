"""
Custom trainer and collator for margin ranking loss.
"""

from typing import Dict, Any, Optional, Tuple

import torch
from torch import nn
from transformers import Trainer


class MarginRankingTrainer(Trainer):
    """
    Custom Trainer that uses MarginRankingLoss for reranking.
    Handles positive/negative pair inputs rather than standard labels.
    """
    
    def __init__(self, *args, margin=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_loss = nn.MarginRankingLoss(margin=margin)
        
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        input_ids_pos = inputs["input_ids_pos"]
        attn_pos = inputs["attention_mask_pos"]
        input_ids_neg = inputs["input_ids_neg"]
        attn_neg = inputs["attention_mask_neg"]

        # Check if model has no_sync (i.e., is wrapped in DDP)
        if hasattr(model, 'no_sync'):
            # Training mode with DDP - sync gradients only on second forward pass
            with model.no_sync():
                output_pos = model(input_ids=input_ids_pos, attention_mask=attn_pos)
                s_pos = output_pos.logits.squeeze(-1)

            output_neg = model(input_ids=input_ids_neg, attention_mask=attn_neg)
            s_neg = output_neg.logits.squeeze(-1)
        else:
            # Evaluation mode or single GPU
            output_pos = model(input_ids=input_ids_pos, attention_mask=attn_pos)
            s_pos = output_pos.logits.squeeze(-1)

            output_neg = model(input_ids=input_ids_neg, attention_mask=attn_neg)
            s_neg = output_neg.logits.squeeze(-1)

        # Target: positive should score higher than negative
        target = torch.ones(s_pos.size(0), device=s_pos.device, dtype=s_pos.dtype)
        loss = self.rank_loss(s_pos, s_neg, target)

        if return_outputs:
            return loss, output_pos
        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None
    ):
        """Compute loss for evaluation without predictions."""
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)
        
        return (loss, None, None)


class MarginRankingCollator:
    """
    Data collator for margin ranking dataset.
    Stacks positive and negative pairs into batches.
    """
    
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
