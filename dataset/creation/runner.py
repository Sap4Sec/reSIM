#!/usr/bin/env python3
"""
This code disassembles the binary files and creates the finetuning and evaluation datasets.

Usage:
    python runner.py --db-name <DB_NAME> -d <BINARY_FILES_DIR> [options]
"""

import sys
import os
import argparse
from loguru import logger

from config import Config
from pipeline import DisassemblyPipeline
from utils.helpers import scan_for_binaries

def main():
    parser = argparse.ArgumentParser(description="Binaries disassembler for dataset creation")
    
    parser.add_argument("--db-name", help="Database name")
    parser.add_argument("-d", "--input-dir", help="Input directory")
    parser.add_argument("-p", "--producers", type=int, default=4, help="Number of producers")
    parser.add_argument("-c", "--consumers", type=int, default=2, help="Number of consumers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of binaries to process (for testing)")
    
    args = parser.parse_args()
    
    # Load and update configuration
    try:
        config = Config.load()
    except Exception:
        config = Config(
             db_name=args.db_name,
             num_producers=args.producers,
             num_consumers=args.consumers,
             ghidra_path="./analyzers/ghidra", 
             ghidra_projects_path="./analyzers/ghidra_projects",
             ghidra_script_path="./analyzers/ghidra_script.py",
             libc_signatures_path="../extraction/data/libc_signatures.json"
        )

    config.db_name = args.db_name
    config.num_producers = args.producers
    config.num_consumers = args.consumers
    
    # Initialize Pipeline
    pipeline = DisassemblyPipeline(config)
    
    try:
        logger.info("=== Starting Comprehensive Pipeline Run ===")
        
        logger.info("Step 1: Setting up Database (Create DB + Tables)...")
        if not pipeline.setup_database(create_db=True):
            logger.error("Database setup failed. Exiting.")
            sys.exit(1)
            
        logger.info(f"Step 2: Processing binaries from {args.input_dir}...")
        
        logger.info(f"Scanning directory: {args.input_dir}")
        binary_paths = scan_for_binaries(args.input_dir)
        logger.info(f"Found {len(binary_paths)} binaries")
        
        if args.limit:
            logger.info(f"Limiting execution to first {args.limit} binaries")
            binary_paths = binary_paths[:args.limit]
            
        stats = pipeline.run(
            binary_paths,
            skip_existing=True
        )
        
        logger.info(f"=== Run Complete ===")
        logger.info(f"Processed: {stats.get('processed')}")
        logger.info(f"Skipped: {stats.get('skipped')}")
        logger.info(f"Final Table Counts: {stats.get('final_counts')}")
        
    except KeyboardInterrupt:
        logger.warning("\nPre-mature termination by user (KeyboardInterrupt).")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        pipeline.shutdown()

if __name__ == "__main__":
    main()
