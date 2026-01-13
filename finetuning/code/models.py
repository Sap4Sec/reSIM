"""
Model loading functions for reranker finetuning.
Supports BERT (full finetuning) and DeepSeek (QLoRA).
"""

import os
from typing import Dict, Any, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def get_device():
    """Get the appropriate CUDA device based on LOCAL_RANK."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def load_bert_model(config):
    """
    Load BERT model for full finetuning (no LoRA).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    device = get_device()
    
    bert_config = config["model"]["bert"]
    model_path = bert_config["model_path"]
    tokenizer_path = bert_config["tokenizer_path"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    model.to(device)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer


def load_deepseek_model(config):
    """
    Load DeepSeek model with QLoRA (4-bit quantization + LoRA adapters).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    device = get_device()
    
    deepseek_config = config["model"]["deepseek"]
    model_name = deepseek_config["model_name"]
    sep_token = deepseek_config["sep_token"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Add special tokens
    added = tokenizer.add_special_tokens({"additional_special_tokens": [sep_token]})
    tokenizer.sep_token = sep_token
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)
    
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Quantization config
    quant_config = config["quantization"]
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    compute_dtype = dtype_map.get(quant_config["compute_dtype"], torch.bfloat16)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config["load_in_4bit"],
        bnb_4bit_use_double_quant=quant_config["use_double_quant"],
        bnb_4bit_quant_type=quant_config["quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=compute_dtype,
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    model.to(device)
    
    # Determine classification head name
    head_name = "score" if hasattr(model, "score") else ("classifier" if hasattr(model, "classifier") else None)
    
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # LoRA config
    lora_config = config["lora"]
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config["bias"],
        init_lora_weights=True,
        modules_to_save=[head_name] if head_name is not None else None,
    )
    
    # Apply LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_cfg)
    
    return model, tokenizer


def get_model_and_tokenizer(config):
    """
    Factory function to load model and tokenizer based on config.
    
    Args:
        config: Configuration dictionary with model.type field
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_type = config["model"]["type"].lower()
    
    if model_type == "bert":
        return load_bert_model(config)
    elif model_type == "deepseek":
        return load_deepseek_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'bert' or 'deepseek'.")
