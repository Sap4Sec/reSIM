"""Reranker module with proper GPU isolation.

IMPORTANT: torch/transformers are imported INSIDE worker_process,
AFTER setting CUDA_VISIBLE_DEVICES, to ensure proper GPU isolation.
"""

import os
import pickle
from time import perf_counter

import numpy as np

import sys
sys.path.append("../")

from logging_utils import initialize_logger
from loguru import logger


MAX_LENGTH = 2048


def worker_process(gpu_id, model_name, tokenizer_path, jobs, results, log_dir, reranker_name, ctx, k_value):
    """Worker process for GPU-specific reranking.
    
    Args:
        gpu_id: The actual CUDA GPU ID (e.g., 4, 5, 6, 7)
    """
    # CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing torch
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # NOW import torch and related modules - they will only see the single GPU
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftConfig, PeftModel
    
    initialize_logger(log_dir, f"{reranker_name}_gpu{gpu_id}_at_{k_value}.log", context=ctx)
    log = logger.bind(gpu=gpu_id)
    
    log.info(f"Worker starting. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # Since only one GPU is visible, use cuda:0
    torch.cuda.set_device(0)
    device = "cuda:0"
    
    # Load model
    log.info("Loading reranker model...")
    model, tokenizer = _load_reranker_model(model_name, device, torch, AutoModelForSequenceClassification, 
                                             AutoTokenizer, BitsAndBytesConfig, PeftConfig, PeftModel)
    log.info("Model loaded successfully")
    
    while True:
        job = jobs.get()
        if job is None:
            break

        idx, queries, candidates, top_ids, q_gts, batch_size = job
        probabilities = []

        start_time = perf_counter()

        for start in range(0, len(queries), batch_size):
            start_time_batch = perf_counter()
            end = start + batch_size
            batch_queries = queries[start:end]
            batch_candidates = candidates[start:end]

            inputs = _tokenize_input(tokenizer, batch_queries, batch_candidates, MAX_LENGTH, device, torch)

            with torch.no_grad():
                logits = model(**inputs).logits
                scores = logits.squeeze()
                if scores.dim() == 0:
                    probabilities.append(scores.cpu().item())
                else:
                    probabilities.extend(scores.cpu().tolist())

            log.info("Completed batch on gpu_{} in: {:.3f}s", gpu_id, perf_counter()-start_time_batch)
        
        log.info("Completed job on gpu_{} in: {:.3f}s", gpu_id, perf_counter()-start_time)
        
        prob_array = np.array(probabilities)
        index_sorted = np.argsort(-prob_array)

        sorted_top_k_ids = [top_ids[i] for i in index_sorted]
        gt_ids = q_gts

        true_labels = [1 if id_ in gt_ids else 0 for id_ in sorted_top_k_ids]
        
        results.put((idx, probabilities, (true_labels, sorted_top_k_ids, len(gt_ids))))
        
        with open(os.path.join(log_dir, f"{reranker_name}_probs_for_gpu{gpu_id}_at_{k_value}.pkl"), 'ab') as file:
            pickle.dump(probabilities, file)

        with open(os.path.join(log_dir, f"{reranker_name}_preds_for_gpu{gpu_id}_at_{k_value}.pkl"), "ab") as file:
            pickle.dump((true_labels, sorted_top_k_ids, len(gt_ids)), file)

    log.info("Worker on gpu_{} exiting.", gpu_id)


def _load_reranker_model(model_name, device, torch, AutoModelForSequenceClassification, 
                         AutoTokenizer, BitsAndBytesConfig, PeftConfig, PeftModel):
    """Load the reranker model with quantization."""
    
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_storage=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    peft_cfg = PeftConfig.from_pretrained(model_name)
    base_name = peft_cfg.base_model_name_or_path
    
    device_map = {"": device}

    print(f"Loading base model: {base_name}")
    base = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=1,
        low_cpu_mem_usage=True,
        quantization_config=bnb,
        device_map=device_map,
        torch_dtype=torch.bfloat16
    )
    
    vocab_len = len(tokenizer)
    if base.get_input_embeddings().num_embeddings != vocab_len:
        base.resize_token_embeddings(vocab_len)

    print(f"Loading PEFT adapter: {model_name}")
    model = PeftModel.from_pretrained(base, model_name, torch_dtype=torch.bfloat16)

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        
    model.config.use_cache = False
    model.eval()

    return model, tokenizer


def _tokenize_input(tokenizer, queries, candidates, max_length, device, torch):
    """Tokenize query-candidate pairs."""
    
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    reserve = 2
    allowed = max(2, max_length - reserve)
    half_allowed = max(1, allowed // 2)

    input_id_list = []
    attention_mask_list = []

    for query, candidate in zip(queries, candidates):
        query_ids = tokenizer.encode(query, add_special_tokens=False)
        candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
        
        query_keep = min(len(query_ids), half_allowed)
        candidate_keep = max(1, allowed - query_keep)
        candidate_keep = min(len(candidate_ids), candidate_keep)

        query_ids = query_ids[-query_keep:]
        candidate_ids = candidate_ids[-candidate_keep:]
            
        tokens = query_ids + [sep_id] + candidate_ids + [pad_id]
        tokens = tokens[-max_length:]
        
        pad_len = max(0, max_length - len(tokens))
        input_ids = tokens + [pad_id] * pad_len
        attention_mask = [1] * len(tokens) + [0] * pad_len

        input_id_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    input_ids = torch.tensor(input_id_list, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.long, device=device)

    return {"input_ids": input_ids, "attention_mask": attention_mask}
