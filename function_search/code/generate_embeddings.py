"""Generate embeddings for bi-encoder models.

This script computes and saves embeddings for various bi-encoder models
(SAFE, PALMTREE, JTRANS, BINBERT, GEMINI, CLAP, TREX) from a dataset of functions.

Usage:
    python generate_embeddings.py --config config.yaml --model SAFE
    python generate_embeddings.py --config config.yaml --model JTRANS --pool-size 5000
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import load_config


def numpy_to_torch(name, output_path):
    """Convert numpy embeddings to PyTorch tensor.
    
    Args:
        name: Base name for the embedding files
        output_path: Directory containing the numpy file
    """
    npy_path = os.path.join(output_path, f"{name}_embeddings_matrix.npy")
    pt_path = os.path.join(output_path, f"{name}_embeddings_matrix.pt")
    
    embeddings_np = np.load(npy_path)
    embeddings_torch = torch.from_numpy(embeddings_np).float()
    torch.save(embeddings_torch, pt_path)
    
    print(f"Saved PyTorch tensor to: {pt_path}")


def save_embeddings_generic(
    embeddings,
    function_ids,
    name,
    output_path
):
    """Save embeddings and IDs to disk.
    
    Args:
        embeddings: numpy array of shape (n_functions, embedding_dim)
        function_ids: list of function IDs
        name: base name for output files
        output_path: directory to save files
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Save numpy embeddings
    npy_path = os.path.join(output_path, f"{name}_embeddings_matrix.npy")
    np.save(npy_path, embeddings)
    print(f"Saved numpy embeddings to: {npy_path}")
    
    # Save function IDs
    ids_path = os.path.join(output_path, f"{name}_function_ids.json")
    with open(ids_path, "w") as f:
        json.dump(function_ids, f)
    print(f"Saved function IDs to: {ids_path}")
    
    # Convert to PyTorch
    numpy_to_torch(name, output_path)


# ============================================================================
# SAFE Embeddings
# ============================================================================

def generate_safe_embeddings(config, df, output_path, name):
    """Generate SAFE embeddings.
    
    Requires: safe_embeddings module and SAFE model
    """
    try:
        import safe_embeddings
    except ImportError:
        print("Error: safe_embeddings module not found.")
        print("Please ensure the SAFE embedding module is in your Python path.")
        return
    
    safe_path = config.get_models_path("SAFE")
    safe_name = "safe.pb"
    
    # Filter to functions with SAFE representation
    safe_df = df[df['safe_function'].notna()]
    print(f"Processing {len(safe_df)} functions with SAFE representation")
    
    safe_model = safe_embeddings.initialize_model(safe_path, safe_name)
    safe_embeddings.save_embeddings_SAFE(safe_model, safe_df, name, output_path, check_gt=False)
    
    numpy_to_torch(name, output_path)


# ============================================================================
# PALMTREE Embeddings
# ============================================================================

def generate_palmtree_embeddings(config, df, output_path, name):
    """Generate PALMTREE embeddings.
    
    Requires: palmtree_embeddings module
    """
    try:
        import palmtree_embeddings
    except ImportError:
        print("Error: palmtree_embeddings module not found.")
        print("Please ensure the PALMTREE embedding module is in your Python path.")
        return
    
    print(f"Processing {len(df)} functions with PALMTREE")
    
    palmtree_model = palmtree_embeddings.initialize_model()
    palmtree_embeddings.save_embeddings(palmtree_model, df, name, output_path, check_gt=False)
    
    numpy_to_torch(name, output_path)


# ============================================================================
# JTRANS Embeddings
# ============================================================================

def generate_jtrans_embeddings(config, df, output_path, name):
    """Generate jTrans embeddings.
    
    Requires: jtrans_embeddings module and jTrans model
    """
    try:
        import jtrans_embeddings
    except ImportError:
        print("Error: jtrans_embeddings module not found.")
        print("Please ensure the jTrans embedding module is in your Python path.")
        return
    
    jtrans_path = config.get_models_path("JTRANS")
    tokenizer_path = os.path.join(jtrans_path, "jtrans_tokenizer")
    jtrans_finetuned_path = os.path.join(jtrans_path, "models", "jTrans-finetune")
    
    print(f"Processing {len(df)} functions with jTRANS")
    
    jtrans_obj = jtrans_embeddings.JTRANS(tokenizer_path, jtrans_finetuned_path)
    jtrans_embeddings.save_embeddings(jtrans_obj, df, name, output_path, check_gt=False)
    
    numpy_to_torch(name, output_path)


# ============================================================================
# BINBERT Embeddings
# ============================================================================

def generate_binbert_embeddings(config, df, output_path, name):
    """Generate BINBERT embeddings.
    
    Requires: binbert_embeddings module
    """
    try:
        import binbert_embeddings
    except ImportError:
        print("Error: binbert_embeddings module not found.")
        print("Please ensure the BINBERT embedding module is in your Python path.")
        return
    
    print(f"Processing {len(df)} functions with BINBERT")
    
    binbert_model = binbert_embeddings.initialize_model()
    binbert_embeddings.save_embeddings(binbert_model, df, name, output_path, check_gt=False)
    
    numpy_to_torch(name, output_path)


# ============================================================================
# GEMINI Embeddings
# ============================================================================

def generate_gemini_embeddings(config, df, output_path, name):
    """Generate GEMINI embeddings.
    
    Requires: gemini_embeddings module
    """
    try:
        import gemini_embeddings
    except ImportError:
        print("Error: gemini_embeddings module not found.")
        print("Please ensure the GEMINI embedding module is in your Python path.")
        return
    
    print(f"Processing {len(df)} functions with GEMINI")
    
    gemini_model = gemini_embeddings.initialize_model()
    gemini_embeddings.save_embeddings(gemini_model, df, name, output_path)
    
    numpy_to_torch(name, output_path)


# ============================================================================
# CLAP Embeddings
# ============================================================================

def generate_clap_embeddings(config, df, output_path, name):
    """Generate CLAP embeddings.
    
    Requires: clap_embeddings module
    """
    try:
        import clap_embeddings
    except ImportError:
        print("Error: clap_embeddings module not found.")
        print("Please ensure the CLAP embedding module is in your Python path.")
        return
    
    print(f"Processing {len(df)} functions with CLAP")
    
    clap_model = clap_embeddings.initialize_model()
    clap_embeddings.save_embeddings(clap_model, df, name, output_path, check_gt=False)
    
    numpy_to_torch(name, output_path)


# ============================================================================
# TREX Embeddings
# ============================================================================

def generate_trex_embeddings(config, df, output_path, name):
    """Generate TREX embeddings.
    
    Requires: trex_embeddings module
    """
    try:
        import trex_embeddings
    except ImportError:
        print("Error: trex_embeddings module not found.")
        print("Please ensure the TREX embedding module is in your Python path.")
        return
    
    print(f"Processing {len(df)} functions with TREX")
    
    trex_model = trex_embeddings.initialize_model()
    trex_embeddings.save_embeddings(trex_model, df, name, output_path, check_gt=False)
    
    numpy_to_torch(name, output_path)


# ============================================================================
# Main
# ============================================================================

GENERATORS = {
    "SAFE": generate_safe_embeddings,
    "PALMTREE": generate_palmtree_embeddings,
    "JTRANS": generate_jtrans_embeddings,
    "BINBERT": generate_binbert_embeddings,
    "GEMINI": generate_gemini_embeddings,
    "CLAP": generate_clap_embeddings,
    "TREX": generate_trex_embeddings,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for bi-encoder models"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=list(GENERATORS.keys()),
        help="Bi-encoder model to use"
    )
    parser.add_argument(
        "--pool-size", "-p",
        type=int,
        default=None,
        help="Pool size (overrides config)"
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Path to input CSV file (overrides default)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for embeddings (overrides default)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    if not os.path.exists(config_path):
        config_path = args.config
    
    config = load_config(config_path)
    
    # Override pool size if specified
    pool_size = args.pool_size if args.pool_size else config.pool_size
    
    # Determine input CSV path
    if args.input_csv:
        input_path = args.input_csv
    else:
        data_path = config.get_data_path()
        if pool_size is None:
            input_path = os.path.join(data_path, "inference_reranker_with_gt.csv")
        else:
            input_path = os.path.join(
                data_path, args.model, 
                f"{config.db_name}_{args.model}_pool_{pool_size}.csv"
            )
    
    # Determine output directory
    if args.output_dir:
        output_path = args.output_dir
    else:
        output_path = os.path.join(
            config.get_data_path(), 
            config.embeddings_subdir, 
            args.model
        )
    
    os.makedirs(output_path, exist_ok=True)
    
    # Determine embedding name
    if pool_size is None:
        name = f"{args.model}"
    else:
        name = f"{args.model}_pool_{pool_size}"
    
    print(f"{'='*60}")
    print(f"Generating {args.model} embeddings")
    print(f"{'='*60}")
    print(f"Input CSV: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Embedding name: {name}")
    print(f"{'='*60}")
    
    # Load dataset
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return
    
    df = pd.read_csv(input_path, sep='\t')
    df = df.where(pd.notnull(df), None)
    print(f"Loaded {len(df)} functions from CSV")
    
    # Generate embeddings
    generator = GENERATORS[args.model]
    generator(config, df, output_path, name)
    
    print(f"\n{'='*60}")
    print(f"Embedding generation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
