"""Entry point for the function search pipeline.

This script launches experiments for evaluating bi-encoder models
with cross-encoder reranking.
"""

import os
import sys
import argparse
import multiprocessing

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from run_experiments import run_experiments


def main():
    """Main entry point for launching experiments."""
    
    parser = argparse.ArgumentParser(
        description="Function Search Pipeline: Evaluate bi-encoders with cross-encoder reranking"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=None,
        help="Specific bi-encoder models to evaluate (default: all from config)"
    )
    parser.add_argument(
        "--rerankers", "-r",
        nargs="+",
        default=None,
        help="Specific rerankers to use (default: all from config)"
    )
    parser.add_argument(
        "--depths", "-d",
        nargs="+",
        type=int,
        default=None,
        help="Specific search depths to evaluate (default: all from config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    if not os.path.exists(config_path):
        config_path = args.config
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Override config with command line arguments if provided
    if args.depths:
        config.search_depths = args.depths
    
    # Determine models and rerankers to run
    models = args.models if args.models else config.bfs_models
    rerankers = args.rerankers if args.rerankers else list(config.rerankers.keys())
    
    print(f"\n{'='*60}")
    print("Function Search Pipeline")
    print(f"{'='*60}")
    print(f"Database: {config.db_name}")
    print(f"Pool size: {config.pool_size}")
    print(f"Search depths: {config.search_depths}")
    print(f"GPUs: {config.gpus}")
    print(f"Batch size: {config.batch_size}")
    print(f"\nModels to evaluate: {models}")
    print(f"Rerankers to use: {rerankers}")
    print(f"{'='*60}\n")
    
    # Run experiments
    for model in models:
        for reranker in [rerankers[0]]:
            try:
                print(f"\nLaunching experiment for {model} using {reranker} as reranker...")
                run_experiments(config, model, reranker)
            except Exception as e:
                print(f"Error running {model} + {reranker}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
