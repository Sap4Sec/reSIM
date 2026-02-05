"""Search engine for bi-encoder based function similarity search."""

import os
import json
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


class SearchEngine:
    
    def __init__(self, checkpoint_dir, ids_filename, embeddings_filename):
        """Initialize the search engine.
        
        Args:
            checkpoint_dir: Directory containing IDs and embeddings files
            ids_filename: JSON file with list of function IDs
            embeddings_filename: PyTorch tensor file with embeddings
        """
        ids_path = os.path.join(checkpoint_dir, ids_filename)
        embeddings_path = os.path.join(checkpoint_dir, embeddings_filename)
        
        print(f"Loading IDs from: {ids_path}")
        with open(ids_path, "r") as f:
            self.ids = json.load(f)
        
        print(f"Loading embeddings from: {embeddings_path}")
        self.matrix = torch.load(embeddings_path).to("cuda")
        
        print(f"Matrix shape: {self.matrix.shape}")
        print(f"Loaded {len(self.ids)} IDs")
    
    def find_top_k(self, query_id, k):
        """Find top-k most similar functions for a single query.
        
        Args:
            query_id: Function ID to query
            k: Number of results to return
            
        Returns:
            Tuple of (list of IDs, list of similarity scores)
        """

        res = []
        scores = []
        idx = self.ids.index(query_id)
        
        embedding_to_query = torch.clone(self.matrix[idx]).to("cuda")

        dist = F.cosine_similarity(self.matrix, embedding_to_query)

        index_sorted = torch.argsort(dist, descending=True)
        top_k = index_sorted[:k]

        res.extend([self.ids[k] for k in top_k])
        scores.extend([dist[k].item() for k in top_k])
        
        return res, scores
    
    def find_top_k_batch(self, query_ids, k):
        """Find top-k most similar functions for a batch of queries.
        
        Args:
            query_ids: List of function IDs to query
            k: Number of results per query
            
        Returns:
            Tuple of (list of ID lists, list of score lists)
        """
        # Get indices for all queries
        res = []
        scores = []
        
        idxs = []
        for idx in query_ids:
            idxs.append(self.ids.index(idx))
        
        m_query = []
        for idx in idxs:
            m_query.append(torch.clone(self.matrix[idx]))
        
        m_query = torch.stack(m_query).to("cuda")
        
        a_norm = self.matrix / self.matrix.norm(dim=-1)[:, None]
        b_norm = m_query / m_query.norm(dim=-1)[:, None]

        dist = torch.mm(b_norm, a_norm.transpose(0,1))

        for i in range(len(query_ids)):
            index_sorted = torch.argsort(dist[i], descending=True)

            top_k = index_sorted[:k]

            res.append([self.ids[k] for k in top_k])
            scores.append([dist[i][k].item() for k in top_k])
        
        return res, scores


def find_top_k_similar(
    data_path,
    df,
    batch_size,
    k,
    ids_file,
    embeddings_file
):
    """Find top-k similar functions for all queries with ground truth.
    
    Args:
        data_path: Path to embeddings directory
        df: DataFrame with queries (must have 'function_id' and 'ground_truth' columns)
        batch_size: Number of queries to process at once
        k: Number of results per query
        ids_file: Filename for IDs JSON
        embeddings_file: Filename for embeddings tensor
        
    Returns:
        List of tuples: (query_id, returned_ids, true_labels, num_gt, scores)
    """
    # Filter to queries with ground truth
    queries = df[~df['ground_truth'].isnull()]
    
    search_engine = SearchEngine(data_path, ids_file, embeddings_file)
    
    answers = []
    batch_ids = []
    batch_gts = []
    
    for i, (_, row) in enumerate(tqdm(queries.iterrows(), total=len(queries))):
        # Parse ground truth
        ground_truth = json.loads(row['ground_truth'])
        ground_truth = [int(x) for x in ground_truth]
        
        batch_ids.append(row["function_id"])
        batch_gts.append(ground_truth)
        
        # Process batch when full or at end
        if len(batch_ids) == batch_size or i == len(queries) - 1:
            top_k_lists, scores = search_engine.find_top_k_batch(batch_ids, k)
            
            for j, (ids_list, score_list) in enumerate(zip(top_k_lists, scores)):
                gt = batch_gts[j]
                true_labels = [1 if fid in gt else 0 for fid in ids_list]
                answers.append((batch_ids[j], ids_list, true_labels, len(gt), score_list))
            
            batch_ids = []
            batch_gts = []
    
    return answers
