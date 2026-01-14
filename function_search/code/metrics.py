"""Metrics computation for function search evaluation."""

import math
from typing import List, Tuple, Dict, Any
import numpy as np

from tqdm import tqdm


def find_dcg(relevance_list):
    dcg_score = 0.0
    for j, sim in enumerate(relevance_list):
        dcg_score += float(sim) / math.log(j + 2)
    return dcg_score


def count_ones(element_list):
    return len([x for x in element_list if x == 1])


def extract_info(answers, max_k=100):
    
    performance1 = []
    average_recall_k1 = []
    precision_at_k1 = []
    mrr_at_k1 = []
    
    for f_index in tqdm(range(0, len(answers))):
        data = answers[f_index]
        
        f1 = data[0]  # ranked list of 1s and 0s
        pf1 = data[2] # total number of true positives in ground truth

        tp1 = []
        recall_p1 = []
        precision_p1 = []
        mrr_p1 = []

        max_recall = []

        recalls_to_ret = {}

        for k in range(1, max_k + 1):
            cut1 = f1[:k]
            dcg1 = find_dcg(cut1)
            ideal1 = find_dcg(([1] * (pf1) + [0] * (k - pf1))[:k])

            p1k = float(count_ones(cut1))

            tp1.append(dcg1 / ideal1)
            recall_p1.append(p1k / pf1)
            
            precision_p1.append(p1k / k)
            
            # calculate MRR@K
            try:
                first_relevant_position = cut1.index(1) + 1
                mrr_p1.append(1 / first_relevant_position)
            except ValueError:
                mrr_p1.append(0)

        max_recall.append(recall_p1[-1])
        
        performance1.append(tp1)
        average_recall_k1.append(recall_p1)
        precision_at_k1.append(precision_p1)
        mrr_at_k1.append(mrr_p1)
    
    avg_p1 = np.average(performance1, axis=0)
    avg_recall = np.average(average_recall_k1, axis=0)
    avg_precision = np.average(precision_at_k1, axis=0)
    avg_mrr = np.average(mrr_at_k1, axis=0)
    
    for k in [1] + list(range(5, 30 + 1, 5)):
        i = k - 1
        print(f"Recall@{k}: {avg_recall[i]}")
        print(f"Precision@{k}: {avg_precision[i]}")
        print(f"nDCG@{k}: {avg_p1[i]}")
    
    return list(avg_p1), list(avg_recall), list(avg_precision), list(avg_mrr)


def extract_info_with_optimum(answers, max_k=100):
    
    performance1 = []
    average_recall_k1 = []
    precision_at_k1 = []
    mrr_at_k1 = []
    
    # Store optimal/ideal values per query per k
    optimal_recall_per_query = []
    optimal_precision_per_query = []
    optimal_ndcg_per_query = []
    
    for f_index in tqdm(range(0, len(answers))):
        data = answers[f_index]
        
        f1 = data[0]  # ranked list of 1s and 0s
        pf1 = data[2] # total number of true positives in ground truth

        tp1 = []
        recall_p1 = []
        precision_p1 = []
        
        # Optimal values for each k for this query
        optimal_recall_k = []
        optimal_precision_k = []
        optimal_ndcg_k = []
        
        # Count how many TPs were actually retrieved by the bi-encoder in total
        total_retrieved_tps = count_ones(f1)

        for k in range(1, max_k + 1):
            cut1 = f1[:k]
            dcg1 = find_dcg(cut1)
            ideal1 = find_dcg(([1] * (pf1) + [0] * (k - pf1))[:k])

            p1k = float(count_ones(cut1))

            tp1.append(dcg1 / ideal1)
            recall_p1.append(p1k / pf1)
            precision_p1.append(p1k / k)
            
            # Optimal values at this k:
            available_tps_at_k = total_retrieved_tps if k >= pf1 else (k if total_retrieved_tps > k else total_retrieved_tps)
            
            # Optimal recall@k: all available TPs in top-k / total ground truth TPs
            optimal_recall_k.append(available_tps_at_k / pf1)
            
            # Optimal precision@k: all available TPs at top / k
            optimal_precision_k.append(available_tps_at_k / k)
            
            # Optimal nDCG@k: all available TPs ranked first
            ideal_ranking_k = [1] * available_tps_at_k + [0] * (k - available_tps_at_k)
            optimal_dcg_k = find_dcg(ideal_ranking_k)
            ideal_dcg_k = find_dcg(([1] * (pf1) + [0] * (k - pf1))[:k])
            optimal_ndcg_k.append(optimal_dcg_k / ideal_dcg_k if ideal_dcg_k > 0 else 0)
        
        performance1.append(tp1)
        average_recall_k1.append(recall_p1)
        precision_at_k1.append(precision_p1)
        
        optimal_recall_per_query.append(optimal_recall_k)
        optimal_precision_per_query.append(optimal_precision_k)
        optimal_ndcg_per_query.append(optimal_ndcg_k)
    
    avg_p1 = np.average(performance1, axis=0)
    avg_recall = np.average(average_recall_k1, axis=0)
    avg_precision = np.average(precision_at_k1, axis=0)
    
    avg_optimal_recall = np.average(optimal_recall_per_query, axis=0)
    avg_optimal_precision = np.average(optimal_precision_per_query, axis=0)
    avg_optimal_ndcg = np.average(optimal_ndcg_per_query, axis=0)
    
    for k in [1] + list(range(5, 30 + 1, 5)):
        i = k - 1
        print(f"Recall@{k}: {avg_recall[i]} (Optimal: {avg_optimal_recall[i]:.4f})")
        print(f"Precision@{k}: {avg_precision[i]} (Optimal: {avg_optimal_precision[i]:.4f})")
        print(f"nDCG@{k}: {avg_p1[i]} (Optimal: {avg_optimal_ndcg[i]:.4f})")
    
    return {
        'avg_ndcg': list(avg_p1),
        'avg_recall': list(avg_recall),
        'avg_precision': list(avg_precision),
        'optimal_ndcg': list(avg_optimal_ndcg),
        'optimal_recall': list(avg_optimal_recall),
        'optimal_precision': list(avg_optimal_precision),
    }
