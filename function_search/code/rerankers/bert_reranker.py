"""BERT-based reranker implementation (reBERT)."""

import os
import pickle
from typing import List, Dict, Optional
from time import perf_counter

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, BertTokenizerFast

from .base import RerankerBase

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_utils import initialize_logger


MAX_LENGTH = 512


class BertReranker(RerankerBase):
    """BERT-based cross-encoder reranker.
    
    Uses BertTokenizerFast and AutoModelForSequenceClassification
    with truncation from the left side.
    """
    
    def __init__(self, max_length=MAX_LENGTH):
        super().__init__(max_length)
    
    def load_model(
        self,
        model_path,
        tokenizer_path,
        gpu_id=0
    ):
        """Load BERT model and tokenizer.
        
        Args:
            model_path: Path to finetuned model checkpoint
            tokenizer_path: Path to tokenizer
            gpu_id: GPU device ID
        """
        self.device = f"cuda:{gpu_id}"
        
        print(f"Loading Tokenizer -> {tokenizer_path}")
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        self.tokenizer.truncation_side = "left"
        
        print(f"Loading Model -> {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            output_hidden_states=True
        )
        
        self.model.to(self.device)
        self.model.eval()
    
    def tokenize(
        self,
        queries,
        candidates
    ):
        """Tokenize query-candidate pairs with balanced truncation.
        
        Splits available tokens evenly between query and candidate,
        truncating from the left to preserve function endings.
        
        Args:
            queries: List of query assembly strings
            candidates: List of candidate assembly strings
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id
        cls_id = self.tokenizer.cls_token_id
        
        reserve = 3  # [CLS], [SEP], [PAD]
        allowed = max(2, self.max_length - reserve)
        half_allowed = max(1, allowed // 2)
        
        input_id_list = []
        attention_mask_list = []
        
        for query, candidate in zip(queries, candidates):
            query_ids = self.tokenizer.encode(query, add_special_tokens=False)
            candidate_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            
            # Balanced truncation
            query_keep = min(len(query_ids), half_allowed)
            candidate_keep = max(1, allowed - query_keep)
            candidate_keep = min(len(candidate_ids), candidate_keep)
            
            # Truncate from left (keep end of functions)
            query_ids = query_ids[-query_keep:] if query_keep < len(query_ids) else query_ids
            candidate_ids = candidate_ids[-candidate_keep:] if candidate_keep < len(candidate_ids) else candidate_ids
            
            # Build sequence: [CLS] query [SEP] candidate [PAD]
            tokens = [cls_id] + query_ids + [sep_id] + candidate_ids + [pad_id]
            tokens = tokens[:self.max_length]
            
            # Pad to max_length
            pad_len = max(0, self.max_length - len(tokens))
            input_ids = tokens + [pad_id] * pad_len
            attention_mask = [1] * len(tokens) + [0] * pad_len
            
            input_id_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(input_id_list, dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long, device=self.device)
        }
    
    def score(self, inputs):
        """Score tokenized inputs.
        
        Args:
            inputs: Dictionary with input_ids and attention_mask
            
        Returns:
            List of similarity scores
        """
        with torch.no_grad():
            logits = self.model(**inputs).logits
            scores = logits.squeeze()
            if scores.dim() == 0:
                return [scores.cpu().item()]
            return scores.cpu().tolist()


def worker_process(
    gpu_id,
    model_path,
    tokenizer_path,
    jobs,
    results,
    log_dir,
    reranker_name,
    ctx,
    k_value
):
    """Worker process for parallel reranking on a single GPU.
    
    Runs in a separate process, pulling jobs from a queue and
    pushing results back.
    
    Args:
        gpu_id: GPU ID to use
        model_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer
        jobs: Multiprocessing Queue with jobs
        results: Multiprocessing Queue for results
        log_dir: Directory for log files
        reranker_name: Name of the reranker
        ctx: Multiprocessing context
        k_value: K value for search depth
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    log = initialize_logger(log_dir, f"{reranker_name}_gpu{gpu_id}_at_{k_value}.log", 
                           gpu_id=gpu_id, context=ctx)
    
    torch.cuda.set_device(0)  # Always 0 since we set CUDA_VISIBLE_DEVICES
    
    reranker = BertReranker(max_length=MAX_LENGTH)
    reranker.load_model(model_path, tokenizer_path, gpu_id=0)
    
    while True:
        job = jobs.get()
        if job is None:
            break
        
        idx, queries, candidates, top_ids, q_gts, batch_size = job
        probabilities = []
        
        start_time = perf_counter()
        
        for start in range(0, len(queries), batch_size):
            batch_start_time = perf_counter()
            end = start + batch_size
            
            inputs = reranker.tokenize(queries[start:end], candidates[start:end])
            scores = reranker.score(inputs)
            probabilities.extend(scores)
            
            log.info("Completed batch on gpu_{} in: {:.3f}s", gpu_id, perf_counter() - batch_start_time)
        
        log.info("Completed job on gpu_{} in: {:.3f}s", gpu_id, perf_counter() - start_time)
        
        # Sort by score (descending)
        prob_array = np.array(probabilities)
        index_sorted = np.argsort(-prob_array)
        
        sorted_top_k_ids = [top_ids[i] for i in index_sorted]
        true_labels = [1 if id_ in q_gts else 0 for id_ in sorted_top_k_ids]
        
        results.put((idx, probabilities, (true_labels, sorted_top_k_ids, len(q_gts))))
        
        # Save intermediate results
        with open(os.path.join(log_dir, f"{reranker_name}_probs_for_gpu{gpu_id}_at_{k_value}.pkl"), 'ab') as f:
            pickle.dump(probabilities, f)
        
        with open(os.path.join(log_dir, f"{reranker_name}_preds_for_gpu{gpu_id}_at_{k_value}.pkl"), 'ab') as f:
            pickle.dump((true_labels, sorted_top_k_ids, len(q_gts)), f)
    
    log.info("Worker on gpu_{} exiting.", gpu_id)
