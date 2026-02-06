"""
Dataset classes for reranker finetuning with triplet margin loss.
"""

import random
from typing import Dict, Any, Tuple, List

import pandas as pd
import torch
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    
    def __init__(
        self,
        dataset_path,
        tokenizer,
        max_seq_length,
        model_type="bert",
        use_half=False,
        random_seed=1000
    ):
        random.seed(random_seed)
        
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.model_type = model_type.lower()
        
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = getattr(tokenizer, 'cls_token_id', None)
        
        # Load data
        df = pd.read_csv(dataset_path, sep="\t").fillna("")
        samples = df[["anchor", "pos", "neg"]].values.tolist()
        
        # Optionally use only half of the dataset
        if use_half:
            samples = samples[:int(len(samples) / 2)]
        
        self.samples = samples
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor, pos, neg = self.samples[idx]
        
        input_ids_pos, attention_mask_pos = self._tokenize_pair(anchor, pos)
        input_ids_neg, attention_mask_neg = self._tokenize_pair(anchor, neg)

        return {
            "input_ids_pos": torch.tensor(input_ids_pos, dtype=torch.long),
            "attention_mask_pos": torch.tensor(attention_mask_pos, dtype=torch.long),
            "input_ids_neg": torch.tensor(input_ids_neg, dtype=torch.long),
            "attention_mask_neg": torch.tensor(attention_mask_neg, dtype=torch.long)
        }

    def _tokenize_pair(self, f1, f2):
        """Tokenize a pair of functions according to model type."""
        f1_ids = self.tokenizer.encode(f1, add_special_tokens=False)
        f2_ids = self.tokenizer.encode(f2, add_special_tokens=False)
        
        if self.model_type == "bert":
            return self._tokenize_bert(f1_ids, f2_ids)
        else:
            return self._tokenize_deepseek(f1_ids, f2_ids)
    
    def _tokenize_deepseek(self, f1_ids, f2_ids):
        """DeepSeek tokenization: f1 [SEP] f2 [PAD]"""
        reserve = 2  # SEP, PAD
        allowed = max(2, self.max_seq_length - reserve)
        
        half_allowed = max(1, allowed // 2)
        f1_keep = min(len(f1_ids), half_allowed)
        f2_keep = min(len(f2_ids), max(1, allowed - f1_keep))
        
        # Truncate from the beginning (keep end)
        f1_ids = f1_ids[-f1_keep:]
        f2_ids = f2_ids[-f2_keep:]
        
        tokens = f1_ids + [self.sep_token_id] + f2_ids + [self.pad_token_id]
        tokens = tokens[-self.max_seq_length:]
        
        pad_len = max(0, self.max_seq_length - len(tokens))
        input_ids = tokens + [self.pad_token_id] * pad_len
        attention_mask = [1] * len(tokens) + [0] * pad_len
        
        return input_ids, attention_mask
