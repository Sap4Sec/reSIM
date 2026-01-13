#!/usr/bin/env python3
"""
Reranker Finetuning Script

Usage:
    python train.py                          # Use default config.yaml
    python train.py --config path/to/config.yaml  # Use custom config
    
For multi-GPU training (with 2 GPUs) with accelerate:
    accelerate launch --num_processes=2 train.py --config config.yaml
"""

import os
import argparse

import torch
from transformers import TrainingArguments

from utils import load_config, setup_environment, get_output_dir, CSVLoggerCallback
from dataset import TripletDataset
from models import get_model_and_tokenizer
from trainer import MarginRankingTrainer, MarginRankingCollator


def parse_args():
    parser = argparse.ArgumentParser(description="Reranker Finetuning")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file"
    )
    return parser.parse_args()


def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    
    # Setup environment (CUDA devices, etc.)
    setup_environment(config)
    
    # Generate output directory
    output_dir = get_output_dir(config)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(config)
    print(f"Model loaded: {config['model']['type']}")
    
    # Prepare dataset paths
    base_path = config["data"]["base_path"]
    train_path = os.path.join(base_path, config["data"]["train_file"])
    val_path = os.path.join(base_path, config["data"]["val_file"])
    
    # Load datasets
    model_type = config["model"]["type"].lower()
    
    # Get model-specific training config merged with shared
    model_training = config["training"][model_type]
    shared_training = config["training"]["shared"]
    
    max_seq_length = model_training["max_seq_length"]
    use_half = config["data"]["use_half_dataset"]
    random_seed = config["data"]["random_seed"]
    
    train_dataset = TripletDataset(
        dataset_path=train_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        model_type=model_type,
        use_half=use_half,
        random_seed=random_seed
    )
    
    val_dataset = TripletDataset(
        dataset_path=val_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        model_type=model_type,
        use_half=False,  # Always use full validation set
        random_seed=random_seed
    )
    
    print(f"Datasets loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Create collator
    collator = MarginRankingCollator(tokenizer, padding=True, pad_to_multiple_of=8)
    
    # Training arguments from merged config
    hardware_config = config["hardware"]
    gradient_checkpointing = model_training.get("gradient_checkpointing", False)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=model_training["num_epochs"],
        learning_rate=model_training["learning_rate"],
        warmup_ratio=shared_training["warmup_ratio"],
        per_device_train_batch_size=model_training["per_device_train_batch_size"],
        gradient_accumulation_steps=model_training["gradient_accumulation_steps"],
        per_device_eval_batch_size=model_training["per_device_eval_batch_size"],
        eval_accumulation_steps=shared_training["eval_accumulation_steps"],
        save_strategy=shared_training["save_strategy"],
        save_steps=shared_training["save_steps"],
        eval_strategy=shared_training["eval_strategy"],
        eval_steps=shared_training["eval_steps"],
        save_total_limit=shared_training["save_total_limit"],
        logging_strategy="steps",
        logging_steps=shared_training["logging_steps"],
        report_to=None,
        load_best_model_at_end=shared_training["load_best_model_at_end"],
        metric_for_best_model=shared_training["metric_for_best_model"],
        bf16=shared_training["bf16"],
        fp16=shared_training["fp16"],
        tf32=shared_training["tf32"],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        prediction_loss_only=False,
        dataloader_num_workers=hardware_config["dataloader_num_workers"],
        dataloader_pin_memory=hardware_config["dataloader_pin_memory"],
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
    )
    
    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Setup CSV logger
    log_path = os.path.join(output_dir, "logs", "training_metrics.csv")
    csv_logger = CSVLoggerCallback(log_path)
    
    # Create trainer
    margin = shared_training["margin"]
    trainer = MarginRankingTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        args=training_args,
        margin=margin,
        callbacks=[csv_logger],
    )
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nTraining completed! Model saved to: {output_dir}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    main()
