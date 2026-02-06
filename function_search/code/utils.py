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



def get_bfs_metrics_path(db_name, bfs_model, cve=None):

    if db_name == "binpool_refactored":
        base_folder = os.path.join(
            BASE_PATH, "function_search", "search_results",
            db_name, "BFS", f"{db_name}_vuln", "POOLS_FOR_TESTING", bfs_model, cve
        )

        filename = f"{bfs_model}_metrics_at_200_training.json"

    else:
        
        pool_size = DATASETS[db_name]["pool_size"]
        
        base_folder = os.path.join(
            BASE_PATH, "function_search", "search_results",
            db_name, "BFS", f"{db_name}_test", "POOLS_FOR_TESTING", bfs_model
        )
        
        filename = f"{bfs_model}_metrics_pool_{pool_size}_at_200.json"
    
    return os.path.join(base_folder, filename)


def get_reranker_metrics_path(db_name, bfs_model, window_size=200, checkpoint=None, cve=None):

    if db_name == "binpool_refactored":
        base_folder = os.path.join(
            BASE_PATH, "function_search", "search_results",
            db_name, RERANKER_NAME, "marginloss", f"{db_name}_vuln", "POOLS_FOR_TESTING", "window", str(window_size), bfs_model, cve
        )

        filename = f"{RERANKER_NAME}_metrics_at_200.json"

    else:
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
    

    base_handles = [
        Line2D([0], [0], color=color_map[m], lw=1.8, linestyle="solid", label=m)
        for m in models_to_plot if m in results and results[m] is not None
    ]
    if base_handles:
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


def plot_multi_base_results_with_optimum(curves, max_pos=30, file_name=None, label_y="Recall", image_title=None, stds=None):
    """Plot curves produced in the multi-base example snippets.

    `curves` should be a dict mapping base_model -> { method_name: metric_list, ..., 'opt': opt_list }
    """
    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    ax.set_xlabel("Number of Nearest Results (k)")
    ax.set_ylabel(label_y)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(1, max_pos)

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.grid(which='major', axis='x', linewidth=0.8, alpha=0.5)
    ax.grid(which='major', axis='y', linewidth=0.8, alpha=0.5)
    ax.grid(which='minor', axis='y', linewidth=0.5, alpha=0.25)

    cycle_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    method_styles = {
        "base": "solid",
        "reDEEP": (0, (5, 1)),
        "reDEEP_random": (0, (3, 1, 1, 1)),
        "scratch_DEEP": (0, (3, 1, 1, 1)),
        "opt": "dotted",
    }

    x = list(range(1, max_pos + 1))

    bases = list(curves.keys())
    for i, base in enumerate(bases):
        methods = curves[base]
        color = cycle_colors[i % len(cycle_colors)]

        # Baseline (if present)
        if "base" in methods and methods["base"] is not None:
            yb = methods["base"][:max_pos]
            ax.plot(x, yb, linestyle=method_styles["base"], linewidth=1.6, color=color)

        # Rerankers and others
        for mname, vals in methods.items():
            if vals is None:
                continue
            if mname == "base":
                continue
            if mname == "opt":
                try:
                    yopt = vals[:max_pos]
                    ax.plot(x, yopt, linestyle=method_styles.get("opt", "dotted"), linewidth=1.2, color=color)
                except Exception:
                    pass
                continue

            # Choose style: treat names starting with "reDEEP" as reDEEP
            style = method_styles.get(mname, method_styles.get("reDEEP", (0, (5, 1))))
            try:
                y = vals[:max_pos]
                ax.plot(x, y, linestyle=style, linewidth=1.4, color=color)
            except Exception:
                pass

    # Build legends: one for bases (colors) and one for method styles
    base_handles = [Line2D([0], [0], color=cycle_colors[i % len(cycle_colors)], lw=1.8, linestyle="solid", label=b)
                    for i, b in enumerate(bases)]
    if base_handles:
        color_leg = ax.legend(handles=base_handles, title="BFS Base", ncol=1, prop={'size': 7}, loc="lower right")
        ax.add_artist(color_leg)

    # Collect all method names and build legend with proper labels
    all_methods = set()
    for base, series in curves.items():
        all_methods.update(series.keys())
    
    meths_sorted = sorted(all_methods, key=lambda x: (x == "opt", x != "base", x))  # base first, opt last
    method_handles = []
    for m in meths_sorted:
        color = "0.2"
        label = "BFS" if m == "base" else "Optimal" if m == "opt" else m.replace("re", "")
        if label == "DEEP_random":
            label = "scratch_DEEP"
        style = method_styles.get(m, "solid")
        method_handles.append(
            Line2D([0], [0], color=color, lw=1.4, linestyle=style, label=label)
        )
    ax.legend(handles=method_handles, title="Models", prop={'size': 7}, loc="lower left")

    plt.tight_layout()

    if file_name:
        plt.savefig(file_name, dpi=500, format="pdf", bbox_inches="tight")
        print(f"Saved: {file_name}")

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

def load_rq4_curves_dict(db_name="bincorp", bases_to_plot=None, methods_to_plot=None, window_size=200):
    if bases_to_plot is None:
        bases_to_plot = ["CLAP"]
    if methods_to_plot is None:
        methods_to_plot = ["reDEEP", "scratch_DEEP"]
    
    curves_n = {}  # nDCG
    curves_r = {}  # Recall
    curves_p = {}  # Precision
    
    for base in bases_to_plot:
        curves_n[base] = {}
        curves_r[base] = {}
        curves_p[base] = {}
        
        for method in methods_to_plot:
            try:
                if method == "reDEEP":
                    # Pretrained
                    folder = os.path.join(
                        BASE_PATH, "function_search", "search_results",
                        db_name, method, "marginloss", f"{db_name}_test",
                        "POOLS_FOR_TESTING", "window", str(window_size),
                        f"{base}_margin_0.2_checkpoint_2590"
                    )
                    fname = f"{method}_metrics_pool_{DATASETS[db_name]['pool_size']}_at_{window_size}.json"
                elif method == "scratch_DEEP":
                    # Random
                    folder = os.path.join(
                        BASE_PATH, "function_search", "search_results",
                        db_name, "reDEEP_random", "marginloss", f"{db_name}_test",
                        "POOLS_FOR_TESTING", "window", str(window_size),
                        base
                    )
                    fname = f"reDEEP_random_metrics_pool_{DATASETS[db_name]['pool_size']}_at_{window_size}.json"
                else:
                    continue
                
                filepath = os.path.join(folder, fname)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # data format: [ndcg, recall, precision, ndcg_opt?, recall_opt?, precision_opt?]
                if isinstance(data, list) and len(data) >= 3:
                    curves_n[base][method] = data[0]
                    curves_r[base][method] = data[1]
                    curves_p[base][method] = data[2]
                    
                    # Store optimal if present
                    if len(data) > 3:
                        curves_n[base]["opt"] = data[3]
                        curves_r[base]["opt"] = data[4] if len(data) > 4 else None
                        curves_p[base]["opt"] = data[5] if len(data) > 5 else None
                else:
                    print(f"  Warning: unexpected data format for {method} in {base}")
            
            except FileNotFoundError as e:
                print(f"  Skipping {method} for {base}: {e}")
                continue
    
    return curves_n, curves_r, curves_p


def generate_rq4_table():   
    # Values from the paper (CLAP + reDEEP on BinCorp)
    data = {
        "Model": ["reDEEP (pretrained)", "scratch_DEEP"],
        "nDCG (avg)": [0.92, 0.14],
        "Recall (avg)": [0.93, 0.43],
    }
    
    df = pd.DataFrame(data)
    display(df.style.format(precision=2).set_properties(**{'text-align': 'right'}))
    
    return df


def plot_rq4_curves_multi(db_name, bases_to_plot=None, methods_to_plot=None, 
                          metrics=None, max_pos=30, save_dir=None):
    if bases_to_plot is None:
        bases_to_plot = ["CLAP"]
    if methods_to_plot is None:
        methods_to_plot = ["reDEEP", "scratch_DEEP"]
    if metrics is None:
        metrics = ["ndcg", "recall"]
    
    print(f"\nLoading RQ4 curves for {db_name}...")
    curves_n, curves_r, curves_p = load_rq4_curves_dict(db_name, bases_to_plot, methods_to_plot)
    
    metric_map = {
        "ndcg": (curves_n, "nDCG"),
        "recall": (curves_r, "Recall"),
        "precision": (curves_p, "Precision"),
    }
    
    for metric in metrics:
        if metric not in metric_map:
            print(f"Warning: unknown metric '{metric}', skipping")
            continue
        
        curves, label_y = metric_map[metric]
        
        if save_dir:
            save_path = os.path.join(save_dir, f"RQ4_{metric}_pretrain_vs_random.pdf")
        else:
            save_path = None
        
        plot_multi_base_results_with_optimum(
            curves=curves,
            max_pos=max_pos,
            file_name=save_path,
            label_y=label_y,
        )

# Vulnerability utils

def plot_vuln_curves(db_name, results, models_to_plot=None, metric="recall", save_path=None):

    if models_to_plot is None:
        models_to_plot = ["CLAP"]
    
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
        if model not in results or results[model] is None:
            continue

        color = color_map.get(model, "tab:blue")

        res = results[model]

        # Nested format with baseline/reranked/optimal
        if isinstance(res, dict) and "baseline" in res:
            try:
                if res.get("baseline") is not None:
                    y_baseline = res["baseline"][metric][:30]
                    ax.plot(x, y_baseline, linestyle=method_styles["base"], linewidth=1.6, color=color)
            except Exception as e:
                print(f"Error plotting baseline for {model}: {e}")

            try:
                if res.get("reranked") is not None:
                    y_reranked = res["reranked"][metric][:30]
                    ax.plot(x, y_reranked, linestyle=method_styles.get("reDEEP", method_styles["reDEEP"]), linewidth=1.4, color=color)
            except Exception as e:
                print(f"Error plotting reranked for {model}: {e}")

            try:
                if res.get("optimal") is not None:
                    y_optimal = res["optimal"][metric][:30]
                    ax.plot(x, y_optimal, linestyle=method_styles["opt"], linewidth=1.2, color=color)
            except Exception as e:
                print(f"Error plotting optimal for {model}: {e}")

        # Case: pretrained reDEEP + scratch (flat) -> treat pretrained as baseline and scratch as reranked
        elif isinstance(res, dict) and ("reDEEP" in res or "scratch_DEEP" in res) and not ("baseline" in res):
            try:
                if "reDEEP" in res and res.get("reDEEP") is not None and metric in res["reDEEP"]:
                    y_pre = res["reDEEP"][metric][:30]
                    ax.plot(x, y_pre, linestyle=method_styles["base"], linewidth=1.6, color=color)
            except Exception as e:
                print(f"Error plotting pretrained reDEEP baseline for {model}: {e}")

            try:
                if "scratch_DEEP" in res and res.get("scratch_DEEP") is not None and metric in res["scratch_DEEP"]:
                    y_scratch = res["scratch_DEEP"][metric][:30]
                    ax.plot(x, y_scratch, linestyle=method_styles.get("reDEEP", method_styles["reDEEP"]), linewidth=1.4, color=color)
            except Exception as e:
                print(f"Error plotting scratch_DEEP for {model}: {e}")

        # Flat format: {'ndcg': [...], 'recall': [...]} -> plot metric directly
        elif isinstance(res, dict) and metric in res:
            try:
                y = res[metric][:30]
                style = method_styles.get(model, method_styles.get("reDEEP"))
                ax.plot(x, y, linestyle=style, linewidth=1.6, color=color)
            except Exception as e:
                print(f"Error plotting flat results for {model}: {e}")
    

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


def load_vuln_data(db_name):

    results = {}
    model = "CLAP"
    
    results[model] = {"baseline": None, "reranked": None, "optimal": None}
    
    # Get CVE directories
    model_path = os.path.join(
        BASE_PATH, "function_search", "search_results",
        db_name, "BFS", f"{db_name}_vuln", "POOLS_FOR_TESTING", model
    )
    
    cves = [d for d in os.listdir(model_path) 
            if os.path.isdir(os.path.join(model_path, d))]
    
    # === Load Baseline ===
    print(f"  Loading baseline for {len(cves)} CVEs...")
    all_baseline_ndcg = []
    all_baseline_recall = []
    
    for cve in cves:
        try:
            baseline_path = get_bfs_metrics_path(db_name, model, cve)
            baseline_data = load_bfs_metrics(baseline_path)
            all_baseline_ndcg.append(baseline_data['ndcg'])
            all_baseline_recall.append(baseline_data['recall'])
        except FileNotFoundError as e:
            continue
    
    if all_baseline_ndcg:
        results[model]["baseline"] = {
            "ndcg": np.mean(all_baseline_ndcg, axis=0).tolist(),
            "recall": np.mean(all_baseline_recall, axis=0).tolist(),
        }
    
    # === Load Reranked ===
    
    all_reranked_ndcg = []
    all_reranked_recall = []
    all_optimal_ndcg = []
    all_optimal_recall = []
    
    for cve in cves:
        try:
            reranked_path = get_reranker_metrics_path(db_name, model, window_size=200, cve=cve)
            reranked_data = load_metrics(reranked_path)
            all_reranked_ndcg.append(reranked_data['ndcg'])
            all_reranked_recall.append(reranked_data['recall'])
            
            # Collect optimal metrics if available
            if reranked_data['ndcg_opt'] is not None:
                all_optimal_ndcg.append(reranked_data['ndcg_opt'])
                all_optimal_recall.append(reranked_data['recall_opt'])
            
        except FileNotFoundError as e:
            continue
    
    if all_reranked_ndcg:
        results[model]["reranked"] = {
            "ndcg": np.mean(all_reranked_ndcg, axis=0).tolist(),
            "recall": np.mean(all_reranked_recall, axis=0).tolist(),
        }
    
    if all_optimal_ndcg:
        results[model]["optimal"] = {
            "ndcg": np.mean(all_optimal_ndcg, axis=0).tolist(),
            "recall": np.mean(all_optimal_recall, axis=0).tolist(),
        }
    
    return results
