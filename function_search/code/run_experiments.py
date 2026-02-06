"""Main experiment runner for function search pipeline."""

import os
import json
import pickle
from typing import List, Optional

import torch.multiprocessing as mp
from tqdm import tqdm

from config import Config, load_config
from data_utils import load_csv, create_pairs, load_bfs_predictions, save_predictions, save_metrics
from metrics import extract_info_with_optimum

from rerankers.deep_reranker import worker_process


def compute_batch_logits_reranker_parallel(
    config,
    reranker_name,
    df,
    full_df,
    sim_results,
    base_res_path,
    k_value,
    batch_size
):

    ctx = mp.get_context("spawn")
    
    # Create pairs for reranking
    q_seq, top_seq, top_ids, q_gts = create_pairs(df, full_df, sim_results, k_value=k_value)
    
    # Get model paths
    model_path = config.get_reranker_full_path(reranker_name)
    tokenizer_path = config.get_tokenizer_full_path(reranker_name)
    
    # Setup queues
    jobs = ctx.Queue()
    results = ctx.Queue()
    
    print(f"Starting {config.num_gpus} GPU workers for {reranker_name}")
    
    # Start worker processes
    processes = []
    for i, gpu_id in enumerate(config.gpus):
        p = ctx.Process(
            target=worker_process,
            args=(gpu_id, model_path, tokenizer_path, jobs, results,
                  base_res_path, reranker_name, ctx, k_value),
            daemon=False
        )
        p.start()
        processes.append(p)
    
    # Submit jobs
    for idx in range(len(q_seq)):
        jobs.put((idx, q_seq[idx], top_seq[idx], top_ids[idx], q_gts[idx], batch_size))
    
    # Send termination signals
    num_gpus = config.num_gpus()
    for _ in range(num_gpus):
        jobs.put(None)
    
    # Collect results
    predictions = [None] * len(q_seq)
    answers = [None] * len(q_seq)
    
    for _ in tqdm(range(len(q_seq)), desc="Collecting results"):
        idx, probabilities, answer = results.get()
        predictions[idx] = probabilities
        answers[idx] = answer
    
    # Wait for workers to finish
    for p in processes:
        p.join()
    
    # Save final results
    pool_suffix = config.pool_suffix
    save_predictions(
        os.path.join(base_res_path, f"{reranker_name}_probs_{pool_suffix}_at_{k_value}.pkl"),
        predictions
    )
    save_predictions(
        os.path.join(base_res_path, f"{reranker_name}_preds_{pool_suffix}_at_{k_value}.pkl"),
        answers
    )
    
    return answers


def load_bfs_results(config, bfs_model, df):

    from search_engine import find_top_k_similar
    from metrics import extract_info
    
    base_path = config.get_bfs_results_path(bfs_model)
    os.makedirs(base_path, exist_ok=True)
    
    pred_file = os.path.join(base_path, f"{bfs_model}_preds_pool_{config.pool_size}_at_200.pkl")
    metrics_file = os.path.join(base_path, f"{bfs_model}_metrics_pool_{config.pool_size}_at_200.json")
    
    # Check if files exist
    if os.path.exists(pred_file) and os.path.exists(metrics_file):
        print(f"Loading BFS results from: {pred_file}")
        with open(pred_file, 'rb') as f:
            predictions = pickle.load(f)
        print(f"Loading BFS metrics from: {metrics_file}")
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return predictions, metrics
    
    # Files don't exist - compute BFS results
    print(f"BFS results not found, computing for {bfs_model}...")
    
    embeddings_dir = os.path.join(config.get_data_path(), config.embeddings_subdir, bfs_model)
    
    ids_file = f"{bfs_model}_pool_test_{config.pool_size}_functions_ids.json"
    embeddings_file = f"{bfs_model}_pool_test_{config.pool_size}_embeddings_matrix.pt"
    
    print(f"Loading embeddings from: {embeddings_dir}")
    print(f"  IDs file: {ids_file}")
    print(f"  Embeddings file: {embeddings_file}")
    
    # Compute BFS results using search engine
    k_value = 200  # Standard retrieval depth for BFS
    batch_size = 100
    
    answers = find_top_k_similar(
        embeddings_dir,
        df,
        batch_size,
        k_value,
        ids_file,
        embeddings_file
    )
    
    # Convert answers to predictions format: list of (query_id, returned_ids)
    predictions = [(ans[0], ans[1]) for ans in answers]
    
    # Compute metrics
    # answers format: (query_id, returned_ids, true_labels, num_gt, scores)
    # extract_info expects: (true_labels, sorted_ids, num_gt)
    metrics_input = [(ans[2], ans[1], ans[3]) for ans in answers]
    avg_ndcg, avg_recall, avg_precision, avg_mrr = extract_info(metrics_input, max_k=k_value)
    
    metrics = {
        'avg_ndcg': list(avg_ndcg),
        'avg_recall': list(avg_recall),
        'avg_precision': list(avg_precision),
        'avg_mrr': list(avg_mrr)
    }
    
    # Save results
    print(f"Saving BFS predictions to: {pred_file}")
    save_predictions(pred_file, predictions)
    
    print(f"Saving BFS metrics to: {metrics_file}")
    save_metrics(metrics_file, metrics)
    
    return predictions, metrics


def run_experiments(config, bfs_model, reranker_name):

    print(f"\n{'='*60}")
    print(f"Running experiment: {bfs_model} -> {reranker_name}")
    print(f"{'='*60}")
    
    # Load dataset
    data_path = config.get_data_path()

    test_file = os.path.join(bfs_model, f"{config.db_name}_{bfs_model}_pool_{config.pool_size}.csv")
    
    test_path = os.path.join(data_path, test_file)
    print(f"Loading data from: {test_path}")
    
    df = load_csv(test_path)
    full_df = load_csv(test_path)
    
    bfs_predictions, bfs_metrics = load_bfs_results(config, bfs_model, df)
    
    # Run for each search depth
    for search_depth in config.search_depths:
        print(f"\n--- Search depth: {search_depth} ---")
        
        # Create output directory
        res_path = config.get_reranker_results_path(reranker_name, bfs_model, search_depth)
        os.makedirs(res_path, exist_ok=True)
        
        # Run reranking
        reranker_answers = compute_batch_logits_reranker_parallel(
            config,
            reranker_name,
            df, full_df,
            bfs_predictions,
            res_path,
            k_value=search_depth,
            batch_size=config.batch_size
        )
        
        # Compute metrics
        metrics = extract_info_with_optimum(reranker_answers, search_depth)
        
        # Save metrics
        if config.pool_size is None:
            metrics_file = f"{reranker_name}_metrics_at_{search_depth}.json"
        else:
            metrics_file = f"{reranker_name}_metrics_pool_{config.pool_size}_at_{search_depth}.json"
        
        metrics_path = os.path.join(res_path, metrics_file)
        save_metrics(metrics_path, [
            metrics['avg_ndcg'],
            metrics['avg_recall'],
            metrics['avg_precision'],
            metrics['optimal_ndcg'],
            metrics['optimal_recall'],
            metrics['optimal_precision'],
        ])
        
        print(f"Metrics saved to: {metrics_path}")
        print(f"  nDCG@{search_depth}: {metrics['avg_ndcg'][-1]:.4f}")
        print(f"  Recall@{search_depth}: {metrics['avg_recall'][-1]:.4f}")
        print(f"  Precision@{search_depth}: {metrics['avg_precision'][-1]:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run function search experiments")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--model", default=None, help="Specific bi-encoder model to test")
    parser.add_argument("--reranker", default=None, help="Specific reranker to use")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    models = [args.model] if args.model else config.bfs_models
    rerankers = [args.reranker] if args.reranker else list(config.rerankers.keys())
    
    for model in models:
        for reranker in rerankers:
            run_experiments(config, model, reranker)
