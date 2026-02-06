"""Data utilities for loading datasets and creating pairs."""

import os
import json
import pickle
from typing import List, Tuple, Optional

import pandas as pd


def load_csv(path):
    df = pd.read_csv(path, sep='\t')
    df = df.where(pd.notnull(df), None)
    return df


def load_bfs_predictions(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_bfs_metrics(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def create_pairs(df, full_df, sim_results, k_value):

    q_seq = []
    top_seq = []
    top_ids = []
    gts = []
    
    # Create indexed views for fast lookup
    df_index = df.set_index("function_id")
    full_index = full_df.set_index("function_id")
    
    # Extract query IDs and their top-k candidates
    queries = [el[0] for el in sim_results]
    top_k_ids_all = [el[1][:k_value] for el in sim_results]
    
    for idx, query_id in enumerate(queries):
        # Get query assembly
        query_asm = df_index.at[int(query_id), "reranker_parsed"]
        
        # Get ground truth
        gt_str = df_index.at[int(query_id), "ground_truth"]
        ground_truth = json.loads(gt_str)
        ground_truth = [int(x) for x in ground_truth]
        gts.append(ground_truth)
        
        # Get candidate IDs and their assemblies
        candidate_ids = top_k_ids_all[idx]
        
        # Look up candidate assemblies
        candidate_data = (
            full_index.loc[candidate_ids][["reranker_parsed"]]
            .reset_index()
            .values
            .tolist()
        )
        
        candidate_ids_ordered = [el[0] for el in candidate_data]
        candidate_asms = [el[1] for el in candidate_data]
        
        # Create pairs: query repeated for each candidate
        q_seq.append([query_asm] * len(candidate_asms))
        top_seq.append(candidate_asms)
        top_ids.append(candidate_ids_ordered)
    
    return q_seq, top_seq, top_ids, gts


def save_predictions(filepath, predictions):
    with open(filepath, 'wb') as f:
        pickle.dump(predictions, f)


def save_metrics(filepath, metrics):
    with open(filepath, 'w') as f:
        json.dump(metrics, f)
