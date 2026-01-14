"""
Consumer worker for database insertion.

Consumers take results from the queue and insert them into
the database using the connection pool.
"""

import traceback
from typing import Dict, Any
from multiprocessing import Queue
from queue import Empty

from loguru import logger

from config import Config
from database.connection import DatabasePool
from database.operations import DatabaseOperations


def consumer_worker(
    worker_id: int,
    result_queue: Queue,
    config: Config
) -> None:
    """
    Consumer worker that inserts results into the database.
    
    This function runs in a separate process. It:
    1. Gets results from the result queue
    2. Obtains a connection from the shared pool
    3. Inserts all function data within a transaction
    4. Commits or rolls back based on success/failure
    
    The key concurrency fix is that connections come from a shared pool,
    and each batch of inserts happens within a single transaction that
    is committed at the end.
    
    Args:
        worker_id: Unique identifier for this worker
        result_queue: Queue of results from producers
        config: Pipeline configuration
    """
    logger.info(f"[Consumer {worker_id}] Starting")
    
    # Initialize database pool for this process
    # (Each process needs its own pool connection)
    if not DatabasePool.is_initialized():
        DatabasePool.initialize(config)
    
    processed_binaries = 0
    processed_functions = 0
    errors = 0
    
    while True:
        try:
            # Get next result (with timeout for graceful shutdown)
            result = result_queue.get(timeout=60)
            
            # Check for poison pill (shutdown signal)
            if result is None:
                logger.info(f"[Consumer {worker_id}] Received shutdown signal")
                break
            
            binary_path = result['binary_path']
            functions = result['functions']
            
            logger.debug(
                f"[Consumer {worker_id}] Inserting {len(functions)} functions "
                f"from {binary_path}"
            )
            
            # Insert all functions using connection from pool
            inserted = insert_binary_functions(
                functions,
                config.insert_batch_size,
                worker_id
            )
            
            processed_binaries += 1
            processed_functions += inserted
            
            if processed_binaries % 10 == 0:
                logger.info(
                    f"[Consumer {worker_id}] Progress: {processed_binaries} binaries, "
                    f"{processed_functions} functions inserted"
                )
            
        except Empty:
            # No more items, check if we should continue
            continue
            
        except Exception as e:
            errors += 1
            logger.error(f"[Consumer {worker_id}] Error: {e}")
            traceback.print_exc()
            continue
    
    logger.info(
        f"[Consumer {worker_id}] Finished. "
        f"Binaries: {processed_binaries}, Functions: {processed_functions}, "
        f"Errors: {errors}"
    )


def insert_binary_functions(
    functions: list,
    batch_size: int,
    worker_id: int
) -> int:
    """
    Insert all functions from a binary into the database.
    
    Uses the connection pool and inserts multiple functions
    within a single transaction.
    
    Args:
        functions: List of prepared function data
        batch_size: Batch size for bulk inserts
        worker_id: Worker ID for logging
        
    Returns:
        Number of functions successfully inserted
    """
    inserted = 0
    
    # Get connection from pool - this is the key concurrency fix
    # The connection is automatically committed on success,
    # rolled back on error, and returned to pool when done
    with DatabasePool.get_connection() as conn:
        for func_data in functions:
            try:
                function_id = DatabaseOperations.insert_function_with_all_data(
                    conn=conn,
                    function_data=func_data['function_data'],
                    basic_blocks=func_data['basic_blocks'],
                    ghidra_blocks=func_data['ghidra_blocks'],
                    palmtree_data=func_data['palmtree_data'],
                    batch_size=batch_size
                )
                
                if function_id is not None:
                    inserted += 1
                    
            except Exception as e:
                # Log but continue with other functions
                func_name = func_data['function_data'].get('function_name', 'unknown')
                logger.debug(
                    f"[Consumer {worker_id}] Error inserting function {func_name}: {e}"
                )
                # The transaction will be rolled back at the end of the context
                # manager if there are uncommitted changes, but we continue
                # trying other functions
    
    return inserted


def consumer_worker_with_init(
    worker_id: int,
    result_queue: Queue,
    config: Config
) -> None:
    """
    Consumer worker that reinitializes the database pool.
    
    Use this version when spawning consumers as separate processes
    where each needs its own pool initialization.
    """
    # Force reinitialize pool for this process
    DatabasePool._pool = None
    DatabasePool.initialize(config)
    
    # Call main consumer logic
    consumer_worker(worker_id, result_queue, config)
