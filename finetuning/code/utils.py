"""
Utility functions and callbacks for reranker finetuning.
"""

import os
import csv
import yaml
from typing import Dict, Any

from transformers import TrainerCallback


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment(config):
    """Set up environment variables for training."""
    # GPU settings
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_SHM_DISABLE"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    
    # Set CUDA visible devices from config
    cuda_devices = config.get("hardware", {}).get("cuda_visible_devices", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices


def get_output_dir(config):
    """Generate output directory path from config."""
    model_type = config["model"]["type"].lower()
    
    base_path = config["data"]["base_path"]
    base_output = config["output"]["base_output_dir"]
    db_name = config["output"]["db_name"]
    
    # Get model-specific training config
    model_training = config["training"][model_type]
    model_name = model_training["output_model_name"]
    epochs = model_training["num_epochs"]
    lr = model_training["learning_rate"]
    
    out_name = f"{model_name}_marginloss_finetuning_{epochs}_epochs_{lr}_lr"
    output_dir = os.path.join(base_path, base_output, model_name, db_name, out_name)
    
    return output_dir


class CSVLoggerCallback(TrainerCallback):
    """Callback to log training metrics to CSV file."""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.file_handle = None
        self.writer = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        # Ensure log directory exists
        log_dir = os.path.dirname(self.log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        self.file_handle = open(self.log_file, 'w', newline='')
        self.writer = csv.writer(self.file_handle)
        self.writer.writerow(['step', 'epoch', 'loss', 'learning_rate', 'eval_loss'])
        self.file_handle.flush()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self.writer is not None:
            step = state.global_step
            epoch = logs.get('epoch', '')
            loss = logs.get('loss', '')
            lr = logs.get('learning_rate', '')
            eval_loss = logs.get('eval_loss', '')
            self.writer.writerow([step, epoch, loss, lr, eval_loss])
            self.file_handle.flush()
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.file_handle:
            self.file_handle.close()
