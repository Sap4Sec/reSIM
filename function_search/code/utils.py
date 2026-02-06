import os

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.lines import Line2D
from IPython.display import display, HTML


BASE_PATH = "/app/vol/reSAFE_code"

# Dataset configurations
DATASETS = {
    "bincorp": {
        "pool_size": 5000,
        "n_queries": 5000,
        "n_pool_functions": 25000,
        "search_depth": 200,
    },
    "multicomp": {  # This is the "multicomp" dataset in the paper
        "pool_size": 1000,
        "n_queries": 1000, 
        "n_pool_functions": 11622,
        "search_depth": 200,
    },
}

# Models evaluated
BFS_MODELS = ["GEMINI", "SAFE", "JTRANS", "CLAP", "BINBERT", "PALMTREE", "TREX"]
RERANKER_NAME = "reDEEP"

# k values for tables
K_VALUES = [5, 10, 15, 20, 25, 30]
K_INDICES = [k - 1 for k in K_VALUES]


SAVE_FIGURES = True
FIGURE_OUTPUT_DIR = os.path.join(BASE_PATH, "function_search", 
                                  "search_results", "PAPER_FIGURES")
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)



def get_bfs_metrics_path(db_name, bfs_model):

    pool_size = DATASETS[db_name]["pool_size"]
    
    base_folder = os.path.join(
        BASE_PATH, "function_search", "search_results",
        db_name, "BFS", f"{db_name}_test", "POOLS_FOR_TESTING", bfs_model
    )
    
    filename = f"{bfs_model}_metrics_pool_{pool_size}_at_200.json"
    
    return os.path.join(base_folder, filename)


def get_reranker_metrics_path(db_name, bfs_model, window_size=200, checkpoint=None):

    pool_size = DATASETS[db_name]["pool_size"]
    
    if checkpoint:
        base_folder = os.path.join(
            BASE_PATH, "function_search", "search_results",
            db_name, RERANKER_NAME, "marginloss", f"{db_name}_test",
            "POOLS_FOR_TESTING", "window", str(window_size), checkpoint
        )
    else:
        base_folder = os.path.join(
            BASE_PATH, "function_search", "search_results",
            db_name, RERANKER_NAME, "marginloss", f"{db_name}_test",
            "POOLS_FOR_TESTING", "window", str(window_size), bfs_model
        )
    
    filename = f"{RERANKER_NAME}_metrics_pool_{pool_size}_at_{window_size}.json"
    
    return os.path.join(base_folder, filename)


def load_metrics(filepath):

    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return {
        'ndcg': data[0],
        'recall': data[1],
        'precision': data[2],
        'ndcg_opt': data[3] if len(data) > 3 else None,
        'recall_opt': data[4] if len(data) > 4 else None,
        'precision_opt': data[5] if len(data) > 5 else None,
    }


def load_bfs_metrics(filepath):

    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return {
        'ndcg': data[0],
        'recall': data[1],
        'precision': data[2],
        'mrr': data[3] if len(data) > 3 else None,
    }


def format_table(df, title):
    """Pretty-print a DataFrame with styling."""
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    styled = df.style.format(precision=2).set_properties(**{'text-align': 'right'})
    display(styled)
    return styled


def calculate_improvement(baseline, reranked):
    """Calculate percentage improvement."""
    if baseline == 0:
        return 0
    return ((reranked - baseline) / baseline) * 100


# RQ1 UTILS

def load_rq1_data(db_name):

    results = {}
    
    for model in BFS_MODELS:
        results[model] = {"baseline": None, "reranked": None, "optimal": None}
        
        try:
            baseline_path = get_bfs_metrics_path(db_name, model)
            baseline_data = load_bfs_metrics(baseline_path)
            results[model]["baseline"] = {
                "ndcg": [baseline_data['ndcg'][k] for k in K_INDICES],
                "recall": [baseline_data['recall'][k] for k in K_INDICES],
            }
        except FileNotFoundError:
            print(f"Warning: Baseline not found for {model} on {db_name}")
            continue
            
        try:
            reranked_path = get_reranker_metrics_path(db_name, model, window_size=200)
            reranked_data = load_metrics(reranked_path)
            results[model]["reranked"] = {
                "ndcg": [reranked_data['ndcg'][k] for k in K_INDICES],
                "recall": [reranked_data['recall'][k] for k in K_INDICES],
            }
            
            if reranked_data['ndcg_opt'] is not None:
                results[model]["optimal"] = {
                    "ndcg": [reranked_data['ndcg_opt'][k] for k in K_INDICES],
                    "recall": [reranked_data['recall_opt'][k] for k in K_INDICES],
                }
        except FileNotFoundError:
            print(f"Warning: Reranked results not found for {model} on {db_name}")
    
    return results


def generate_rq1_table(db_name, results):

    rows = []
    
    for model in BFS_MODELS:
        if results[model]["baseline"] is None:
            continue
            
        baseline = results[model]["baseline"]
        row_base = {
            "Dataset": db_name.upper() if model == BFS_MODELS[0] else "",
            "BFS Model": model,
            "Reranker": "✗",
        }
        for i, k in enumerate(K_VALUES):
            row_base[f"nDCG@{k}"] = baseline["ndcg"][i]
            row_base[f"Recall@{k}"] = baseline["recall"][i]
        rows.append(row_base)
        
        if results[model]["reranked"] is not None:
            reranked = results[model]["reranked"]
            row_rerank = {
                "Dataset": "",
                "BFS Model": "",
                "Reranker": "✓",
            }
            for i, k in enumerate(K_VALUES):
                row_rerank[f"nDCG@{k}"] = reranked["ndcg"][i]
                row_rerank[f"Recall@{k}"] = reranked["recall"][i]
            rows.append(row_rerank)
    
    df = pd.DataFrame(rows)
    format_table(df, f"Table 1: nDCG and Recall for {db_name.upper()} (w=200)")
    
    return df


def generate_improvement_table(results_bincorp, results_multicomp):

    rows = []
    k_subset = [5, 10, 20, 30]
    k_indices_subset = [K_VALUES.index(k) for k in k_subset]
    
    for db_name, results in [("bincorp", results_bincorp), ("multicomp", results_multicomp)]:
        display_name = "BINCORP" if db_name == "bincorp" else "MULTICOMP"
        
        # Collect all improvements for computing AVG row
        all_ndcg_improvements = {k: [] for k in k_subset}
        all_recall_improvements = {k: [] for k in k_subset}
        all_ndcg_avg = []
        all_recall_avg = []
        
        for model in BFS_MODELS:
            if results[model]["baseline"] is None or results[model]["reranked"] is None:
                continue
                
            baseline = results[model]["baseline"]
            reranked = results[model]["reranked"]
            
            row = {"Dataset": display_name if model == BFS_MODELS[0] else "", "BFS Model": model}
            
            ndcg_improvements = []
            recall_improvements = []
            
            for idx in k_indices_subset:
                baseline_ndcg = round(baseline["ndcg"][idx], 2)
                reranked_ndcg = round(reranked["ndcg"][idx], 2)
                baseline_recall = round(baseline["recall"][idx], 2)
                reranked_recall = round(reranked["recall"][idx], 2)
                
                ndcg_imp = calculate_improvement(baseline_ndcg, reranked_ndcg)
                recall_imp = calculate_improvement(baseline_recall, reranked_recall)
                
                k = K_VALUES[idx]
                row[f"nDCG@{k}"] = f"+{ndcg_imp:.1f}%"
                row[f"Recall@{k}"] = f"+{recall_imp:.1f}%"
                
                ndcg_improvements.append(ndcg_imp)
                recall_improvements.append(recall_imp)
                
                # Track for AVG row
                all_ndcg_improvements[k].append(ndcg_imp)
                all_recall_improvements[k].append(recall_imp)
            
            row["nDCG AVG"] = f"+{np.mean(ndcg_improvements):.1f}%"
            row["Recall AVG"] = f"+{np.mean(recall_improvements):.1f}%"
            
            all_ndcg_avg.append(np.mean(ndcg_improvements))
            all_recall_avg.append(np.mean(recall_improvements))
            
            rows.append(row)
        
        # Add AVG row for this dataset
        avg_row = {"Dataset": "", "BFS Model": "AVG"}
        for k in k_subset:
            if all_ndcg_improvements[k]:
                avg_row[f"nDCG@{k}"] = f"+{np.mean(all_ndcg_improvements[k]):.1f}%"
            else:
                avg_row[f"nDCG@{k}"] = "N/A"
            if all_recall_improvements[k]:
                avg_row[f"Recall@{k}"] = f"+{np.mean(all_recall_improvements[k]):.1f}%"
            else:
                avg_row[f"Recall@{k}"] = "N/A"
        
        if all_ndcg_avg:
            avg_row["nDCG AVG"] = f"+{np.mean(all_ndcg_avg):.1f}%"
        else:
            avg_row["nDCG AVG"] = "N/A"
        if all_recall_avg:
            avg_row["Recall AVG"] = f"+{np.mean(all_recall_avg):.1f}%"
        else:
            avg_row["Recall AVG"] = "N/A"
        
        rows.append(avg_row)
    
    df = pd.DataFrame(rows)
    format_table(df, "Table 2: Improvement (%) after applying reDEEP reranker")
    
    return df


def plot_rq1_curves(db_name, results, models_to_plot=None, metric="recall", save_path=None):

    if models_to_plot is None:
        models_to_plot = ["CLAP", "BINBERT"]
    
    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    
    ax.set_xlabel("Number of Nearest Results (k)")
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(0.2, 1.0)
    ax.set_xlim(1, 30)
    
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticks([1] + list(range(5, 31, 5)))
    
    ax.grid(which='major', axis='x', linewidth=0.8, alpha=0.5)
    ax.grid(which='major', axis='y', linewidth=0.8, alpha=0.5)
    ax.grid(which='minor', axis='y', linewidth=0.5, alpha=0.25)
    
    # Color map for BFS models
    cycle_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    color_map = {m: cycle_colors[i % len(cycle_colors)] for i, m in enumerate(models_to_plot)}
    
    # Line styles for methods
    method_styles = {
        "base": "solid",
        "reDEEP": (0, (5, 1)),
        "opt": "dotted",
    }
    
    x = list(range(1, 31))
    
    for model in models_to_plot:
        if results[model]["baseline"] is None:
            continue
            
        color = color_map.get(model, "tab:blue")
        
        try:
            baseline_path = get_bfs_metrics_path(db_name, model)
            baseline_data = load_bfs_metrics(baseline_path)
            y_baseline = baseline_data[metric][:30]
            ax.plot(x, y_baseline, linestyle=method_styles["base"], linewidth=1.6, color=color)
        except:
            pass
        
        try:
            reranked_path = get_reranker_metrics_path(db_name, model, window_size=200)
            reranked_data = load_metrics(reranked_path)
            y_reranked = reranked_data[metric][:30]
            ax.plot(x, y_reranked, linestyle=method_styles["reDEEP"], linewidth=1.4, color=color)
            
            opt_key = f"{metric}_opt"
            if reranked_data[opt_key] is not None:
                y_optimal = reranked_data[opt_key][:30]
                ax.plot(x, y_optimal, linestyle=method_styles["opt"], linewidth=1.2, color=color)
        except:
            pass
    

    base_handles = [Line2D([0], [0], color=color_map[m], lw=1.8, linestyle="solid", label=m) 
                    for m in models_to_plot if results[m]["baseline"] is not None]
    color_leg = ax.legend(handles=base_handles, title="BFS Base", ncol=1, prop={'size': 7}, loc="lower right")
    ax.add_artist(color_leg)
    
    method_handles = [
        Line2D([0], [0], color="0.2", lw=1.8, linestyle=method_styles["base"], label="BFS"),
        Line2D([0], [0], color="0.2", lw=1.8, linestyle=method_styles["reDEEP"], label="DEEP"),
        Line2D([0], [0], color="0.2", lw=1.8, linestyle=method_styles["opt"], label="Optimal"),
    ]
    ax.legend(handles=method_handles, title="Models", prop={'size': 7}, loc="lower left")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, format="pdf", bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


# RQ2 utils

def load_rq2_data(db_name, bfs_model, window_sizes=[30, 50, 100, 200]):

    results = {"baseline": None}
    
    try:
        baseline_path = get_bfs_metrics_path(db_name, bfs_model)
        baseline_data = load_bfs_metrics(baseline_path)
        results["baseline"] = {
            "ndcg": baseline_data['ndcg'][:30],
            "recall": baseline_data['recall'][:30],
        }
    except FileNotFoundError:
        print(f"Warning: Baseline not found for {bfs_model} on {db_name}")
    
    # Load each window size
    for w in window_sizes:
        try:
            reranked_path = get_reranker_metrics_path(db_name, bfs_model, window_size=w)
            reranked_data = load_metrics(reranked_path)
            results[f"w={w}"] = {
                "ndcg": reranked_data['ndcg'][:30],
                "recall": reranked_data['recall'][:30],
            }
        except FileNotFoundError:
            print(f"Warning: Window {w} not found for {bfs_model} on {db_name}")
    
    return results


def plot_window_comparison(results, bfs_model, db_name, metric="recall", save_path=None):

    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    
    ax.set_xlabel("Number of Nearest Results (k)")
    ax.set_ylabel(metric.capitalize())
    ax.set_ylim(0.2, 1.0)
    ax.set_xlim(1, 30)
    
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticks([1] + list(range(5, 31, 5)))
    
    ax.grid(which='major', axis='x', linewidth=0.8, alpha=0.5)
    ax.grid(which='major', axis='y', linewidth=0.8, alpha=0.5)
    ax.grid(which='minor', axis='y', linewidth=0.5, alpha=0.25)
    
    colors = {"JTRANS": "tab:blue", "CLAP": "tab:orange", "BINBERT": "tab:green"}
    base_color = colors.get(bfs_model, "tab:blue")
    
    # Style mapping for windows
    window_styles = {
        "baseline": ("solid", 1.6),
        "w=30": ("dashed", 1.4),
        "w=50": ("dashdot", 1.4),
        "w=100": ("dotted", 1.4),
        "w=200": ((0, (5, 1)), 1.4),
    }
    
    x = list(range(1, 31))
    plotted_keys = []
    
    for key, data in results.items():
        if data is None:
            continue
        y = data[metric][:30]
        style, lw = window_styles.get(key, ("dashed", 1.0))
        ax.plot(x, y, linestyle=style, linewidth=lw, color=base_color)
        plotted_keys.append(key)
    
    legend_labels = {"baseline": "BFS", "w=30": "w=30", "w=50": "w=50", "w=100": "w=100", "w=200": "w=200"}
    handles = [
        Line2D([0], [0], color=base_color, lw=1.8, linestyle=window_styles[k][0], label=legend_labels.get(k, k))
        for k in plotted_keys
    ]
    ax.legend(handles=handles, title=bfs_model, prop={'size': 7}, loc="lower right")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, format="pdf", bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def generate_rq2_table(db_name, bfs_model, results):
    """Generate table comparing window sizes."""
    rows = []
    k_subset = [5, 10, 20, 30]
    k_indices = [k - 1 for k in k_subset]
    
    for key, data in results.items():
        if data is None:
            continue
        
        row = {"Window": key}
        for k in k_subset:
            idx = k - 1
            row[f"nDCG@{k}"] = data["ndcg"][idx]
            row[f"Recall@{k}"] = data["recall"][idx]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    format_table(df, f"RQ2: Window Size Comparison for {bfs_model} on {db_name.upper()}")
    return df


# RQ3 utils

def get_rq3_aggregate_path(model1="jtrans", model2="binbert", top_k=100):

    return os.path.join(
        BASE_PATH, "function_search", "search_results",
        "bincorp", RERANKER_NAME, "marginloss", "bincorp_test",
        "POOLS_FOR_TESTING", "AGGREGATE",
        f"aggregated_stats_{model1}_{model2}_{top_k}_merged.json"
    )


def load_rq3_ensemble_data(model1="jtrans", model2="binbert", top_k=100):

    filepath = get_rq3_aggregate_path(model1, model2, top_k)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return {
        'ndcg': data[0],
        'recall': data[1],
        'precision': data[2],
    }


def generate_rq3_table():

    rows = []
    
    # Load ensemble data
    try:
        ensemble_data = load_rq3_ensemble_data("jtrans", "binbert", 100)
        
        # Create ensemble row
        row = {"BFS Model": "ENSEMBLE (jTrans+BinBERT)", "Reranker": "✓"}
        for k in K_VALUES:
            idx = k - 1
            row[f"nDCG@{k}"] = ensemble_data['ndcg'][idx]
            row[f"Recall@{k}"] = ensemble_data['recall'][idx]
        rows.append(row)
            
    except FileNotFoundError as e:
        print(f"Warning: Ensemble file not found: {e}")
        print("Using placeholder values...")
        rows.append({
            "BFS Model": "ENSEMBLE", "Reranker": "✓",
            "nDCG@5": "N/A", "nDCG@10": "N/A", "nDCG@15": "N/A", 
            "nDCG@20": "N/A", "nDCG@25": "N/A", "nDCG@30": "N/A",
            "Recall@5": "N/A", "Recall@10": "N/A", "Recall@15": "N/A",
            "Recall@20": "N/A", "Recall@25": "N/A", "Recall@30": "N/A",
        })
    
    df = pd.DataFrame(rows)
    format_table(df, "Table 3: Ensemble Results (jTrans + BinBERT, top-100 merged)")
    
    return df


# RQ4 utils

def load_rq4_data(db_name="bincorp", bfs_model="CLAP"):

    results = {}
    
    # Pretrained reDEEP checkpoint
    try:
        pretrained_path = get_reranker_metrics_path(
            db_name, bfs_model, window_size=200, 
            checkpoint="pretrained_checkpoint_5"  # Adjust checkpoint name as needed
        )
        pretrained_data = load_metrics(pretrained_path)
        results["pretrained"] = {
            "ndcg": pretrained_data['ndcg'],
            "recall": pretrained_data['recall'],
        }
    except FileNotFoundError:
        print("Pretrained checkpoint not found")
        results["pretrained"] = None
    
    # Random reDEEP checkpoint
    try:
        random_path = get_reranker_metrics_path(
            db_name, bfs_model, window_size=200,
            checkpoint="random_checkpoint_5"  # Adjust checkpoint name as needed
        )
        random_data = load_metrics(random_path)
        results["random"] = {
            "ndcg": random_data['ndcg'],
            "recall": random_data['recall'],
        }
    except FileNotFoundError:
        print("Random checkpoint not found")
        results["random"] = None
    
    return results


def display_rq4_results():

    print("=" * 80)
    print("RQ4: Pre-training Impact Comparison")
    print("=" * 80)
    print("\nComparison at checkpoint 5 (same data and training steps):")
    print("-" * 40)
    
    # Values from the paper (CLAP + reDEEP on BinCorp)
    data = {
        "Model": ["reDEEP (pretrained)", "random_reDEEP"],
        "nDCG (avg)": [0.92, 0.14],
        "Recall (avg)": [0.93, 0.43],
    }
    
    df = pd.DataFrame(data)
    display(df.style.format(precision=2).set_properties(**{'text-align': 'right'}))
    
    return df
