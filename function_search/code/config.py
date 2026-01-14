"""Configuration loader for the function search pipeline."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml


class RerankerConfig:

    def __init__(self, name, model_path, tokenizer_path, max_length):

        self.name = name
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length


class Config:

    def __init__(self, base_path, db_name, pool_size, search_depths, 
                    batch_size, gpus, bfs_models, rerankers, 
                    output_base_dir, bfs_subdir, pools_subdir, 
                    pandas_dataset_dir, embeddings_subdir, models_dir):

        # Base paths
        self.base_path = base_path
        self.db_name = db_name
        
        # Experiment settings
        self.pool_size = pool_size
        self.search_depths = search_depths
        self.batch_size = batch_size
        
        # GPU configuration
        self.gpus = gpus
        
        # Models
        self.bfs_models = bfs_models
        self.rerankers = rerankers
        
        # Output paths
        self.output_base_dir = output_base_dir
        self.bfs_subdir = bfs_subdir
        self.pools_subdir = pools_subdir
        
        # Data paths
        self.pandas_dataset_dir = pandas_dataset_dir
        self.embeddings_subdir = embeddings_subdir
        self.models_dir = models_dir
    
    def pool_suffix(self):
        """Generate pool suffix for file naming."""
        return "" if self.pool_size is None else f"pool_{self.pool_size}"
    
    def num_gpus(self):
        """Number of GPUs available."""
        return len(self.gpus)
    
    def get_reranker_full_path(self, reranker_name):
        """Get full path to reranker model."""
        return os.path.join(self.base_path, self.rerankers[reranker_name].model_path)
    
    def get_tokenizer_full_path(self, reranker_name):
        """Get full path to tokenizer."""
        tok_path = self.rerankers[reranker_name].tokenizer_path
        if tok_path is None:
            return None
        return os.path.join(self.base_path, tok_path)
    
    def get_data_path(self):
        """Get path to pandas dataset."""
        dir_template = self.pandas_dataset_dir.replace("{db_name}", self.db_name)
        return os.path.join(self.base_path, dir_template)
    
    def get_models_path(self, model_name=None):
        """Get path to bi-encoder models directory."""
        if model_name:
            return os.path.join(self.base_path, self.models_dir, model_name)
        return os.path.join(self.base_path, self.models_dir)
    
    def get_bfs_results_path(self, bfs_model):
        """Get path to BFS results for a model."""
        return os.path.join(
            self.base_path, self.output_base_dir, self.db_name,
            self.bfs_subdir, f"{self.db_name}_test", self.pools_subdir, bfs_model
        )
    
    def get_reranker_results_path(self, reranker_name, bfs_model, search_depth):
        """Get path to save reranker results."""
        return os.path.join(
            self.base_path, self.output_base_dir, self.db_name,
            reranker_name, "marginloss", f"{self.db_name}_test", 
            self.pools_subdir, "window", str(search_depth), bfs_model
        )


def load_config(config_path):
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the config.yaml file
        
    Returns:
        Config object with all settings
    """
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Parse reranker configs
    rerankers = {}
    for name, cfg in raw_config.get('rerankers', {}).items():
        rerankers[name] = RerankerConfig(
            name=cfg['name'],
            model_path=cfg['model_path'],
            tokenizer_path=cfg.get('tokenizer_path'),
            max_length=cfg.get('max_length', 512)
        )
    
    return Config(
        base_path=raw_config['base_path'],
        db_name=raw_config['db_name'],
        pool_size=raw_config.get('pool_size'),
        search_depths=raw_config.get('search_depths', [30, 50, 100, 200]),
        batch_size=raw_config.get('batch_size', 50),
        gpus=raw_config.get('gpus', [0]),
        bfs_models=raw_config.get('bfs_models', []),
        rerankers=rerankers,
        output_base_dir=raw_config.get('output', {}).get('base_dir', 'tests/similarity/functions/search_results'),
        bfs_subdir=raw_config.get('output', {}).get('bfs_subdir', 'BFS'),
        pools_subdir=raw_config.get('output', {}).get('pools_subdir', 'POOLS_FOR_TESTING'),
        pandas_dataset_dir=raw_config.get('data', {}).get('pandas_dataset_dir', ''),
        embeddings_subdir=raw_config.get('data', {}).get('embeddings_subdir', 'embeddings'),
        models_dir=raw_config.get('data', {}).get('models_dir', 'models')
    )
