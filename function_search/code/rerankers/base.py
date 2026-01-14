"""Abstract base class for reranker models."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

import torch


class RerankerBase(ABC):
    """Abstract base class for cross-encoder rerankers.
    
    Subclasses must implement model loading, tokenization, and scoring.
    """
    
    def __init__(self, max_length=2048):
        """Initialize reranker.
        
        Args:
            max_length: Maximum sequence length for tokenization
        """
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.device = None
    
    @abstractmethod
    def load_model(self, model_path, tokenizer_path, gpu_id=0):
        """Load model and tokenizer.
        
        Args:
            model_path: Path to the model checkpoint
            tokenizer_path: Path to the tokenizer (optional, some models use internal tokenizer)
            gpu_id: GPU device ID
        """
        pass
    
    @abstractmethod
    def tokenize(self, queries, candidates):
        """Tokenize query-candidate pairs.
        
        Args:
            queries: List of query assembly strings
            candidates: List of candidate assembly strings
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        pass
    
    @abstractmethod
    def score(self, inputs):
        """Score tokenized inputs.
        
        Args:
            inputs: Dictionary with 'input_ids' and 'attention_mask' tensors
            
        Returns:
            List of similarity scores
        """
        pass
    
    def rerank(
        self,
        queries,
        candidates,
        batch_size=64
    ):
        """Rerank candidates for queries.
        
        Args:
            queries: List of query assembly strings (same length as candidates)
            candidates: List of candidate assembly strings
            batch_size: Inference batch size
            
        Returns:
            List of similarity scores
        """
        all_scores = []
        
        for start in range(0, len(queries), batch_size):
            end = start + batch_size
            batch_queries = queries[start:end]
            batch_candidates = candidates[start:end]
            
            inputs = self.tokenize(batch_queries, batch_candidates)
            scores = self.score(inputs)
            all_scores.extend(scores)
        
        return all_scores
