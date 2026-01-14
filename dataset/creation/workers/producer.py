"""
Producer worker for binary analysis.

Producers analyze binary files and put results into the queue
for consumers to insert into the database.
"""

import os
import gc
import traceback
from typing import Dict, Any, Optional
from multiprocessing import Queue
from queue import Empty

from loguru import logger

from config import Config
from analyzers.extractor import BinaryExtractor
from utils.helpers import (
    parse_binary_path,
    parse_binary_path_binpool,
    prepare_function_data,
    prepare_basic_blocks,
    prepare_ghidra_blocks,
    prepare_palmtree_data,
    BinaryMetadata
)


def producer_worker(
    worker_id: int,
    file_queue: Queue,
    result_queue: Queue,
    config: Config,
    use_binpool_format: bool = False
) -> None:
    """
    Producer worker that analyzes binaries and puts results in queue.
    
    This function runs in a separate process. It:
    1. Gets binary paths from the file queue
    2. Analyzes each binary using BinaryExtractor
    3. Prepares the data for database insertion
    4. Puts results into the result queue for consumers
    
    Args:
        worker_id: Unique identifier for this worker
        file_queue: Queue of binary file paths to process
        result_queue: Queue to put results for consumers
        config: Pipeline configuration
        use_binpool_format: Use BinPool path parsing format
    """
    logger.info(f"[Producer {worker_id}] Starting")
    
    # Create extractor for this worker
    extractor = BinaryExtractor(config)
    
    processed = 0
    errors = 0
    
    while True:
        try:
            # Get next binary path (with timeout for graceful shutdown)
            binary_path = file_queue.get(timeout=30)
            
            # Check for poison pill (shutdown signal)
            if binary_path is None:
                logger.info(f"[Producer {worker_id}] Received shutdown signal")
                break
            
            logger.info(f"[Producer {worker_id}] Analyzing: {binary_path}")
            
            # DEBUG: Check if file exists
            if not os.path.exists(binary_path):
                logger.error(f"[Producer {worker_id}] File NOT FOUND: {binary_path}")
                # Try to list directory to see what's there
                try:
                    parent_dir = os.path.dirname(binary_path)
                    if os.path.exists(parent_dir):
                        logger.error(f"[Producer {worker_id}] Parent dir contents: {os.listdir(parent_dir)[:10]}")
                    else:
                        logger.error(f"[Producer {worker_id}] Parent dir also NOT FOUND: {parent_dir}")
                except Exception:
                    pass
                continue
            else:
                file_size = os.path.getsize(binary_path)
                logger.info(f"[Producer {worker_id}] File check OK: {binary_path} (Size: {file_size} bytes)")

            # Parse metadata from path
            if use_binpool_format:
                metadata = parse_binary_path_binpool(binary_path)
            else:
                metadata = parse_binary_path(binary_path)
            
            # Extract functions
            functions = extractor.extract(binary_path)
            
            if not functions:
                logger.warning(f"[Producer {worker_id}] No functions extracted: {binary_path}")
                continue
            
            # Prepare results for each function
            results = []
            for func_addr, func_info in functions.items():
                try:
                    result = prepare_function_result(func_addr, func_info, metadata)
                    results.append(result)
                except Exception as e:
                    logger.debug(f"Error preparing function {func_addr}: {e}")
            
            # Put all results for this binary in queue
            result_queue.put({
                'binary_path': binary_path,
                'metadata': metadata,
                'functions': results,
                'function_count': len(results)
            })
            
            processed += 1
            logger.info(
                f"[Producer {worker_id}] Completed: {binary_path} "
                f"({len(results)} functions)"
            )
            
            # Periodic garbage collection
            if processed % 10 == 0:
                gc.collect()
            
        except Empty:
            # No more items, check if we should continue
            continue
            
        except Exception as e:
            errors += 1
            logger.error(f"[Producer {worker_id}] Error processing binary: {e}")
            traceback.print_exc()
            continue
    
    logger.info(
        f"[Producer {worker_id}] Finished. Processed: {processed}, Errors: {errors}"
    )


def prepare_function_result(
    func_addr: int,
    func_info: Dict[str, Any],
    metadata: BinaryMetadata
) -> Dict[str, Any]:
    """
    Prepare a single function's data for database insertion.
    
    Args:
        func_addr: Function address
        func_info: Raw function info from extractor
        metadata: Binary metadata
        
    Returns:
        Dictionary with all data needed for database insertion
    """
    # Add address to func_info if not present
    func_info['address'] = func_addr
    
    return {
        'function_data': prepare_function_data(func_info, metadata),
        'basic_blocks': prepare_basic_blocks(func_info),
        'ghidra_blocks': prepare_ghidra_blocks(func_info),
        'palmtree_data': prepare_palmtree_data(func_info)
    }
