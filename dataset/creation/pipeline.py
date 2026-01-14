"""
Main pipeline orchestrator for DisassemblyPipeline.
"""

import json
import random
from typing import List, Optional
from multiprocessing import Process, Queue, Manager
from pathlib import Path

from tqdm import tqdm
from loguru import logger

from config import Config
from database.connection import (
    DatabasePool,
    create_database_if_not_exists,
    test_connection
)
from database.models import create_tables, get_table_counts
from database.operations import DatabaseOperations
from workers.producer import producer_worker
from workers.consumer import consumer_worker_with_init
from utils.helpers import scan_for_binaries


class DisassemblyPipeline:
    
    def __init__(self, config: Config):
        self.config = config
        self.manager = Manager()
        self._initialized = False
    
    def setup_database(self, create_db: bool = False) -> bool:
        try:
            if create_db:
                logger.info("Checking/creating database...")
                create_database_if_not_exists(self.config)
            
            # Test connection
            if not test_connection(self.config):
                logger.error("Database connection failed")
                return False
            
            # Initialize pool
            DatabasePool.initialize(self.config)
            
            # Create tables
            logger.info("Creating/verifying database tables...")
            create_tables()
            
            # Log table counts
            counts = get_table_counts()
            logger.info(f"Current table counts: {counts}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
    
    def run(
        self,
        binary_paths: List[str],
        shuffle: bool = True,
        skip_existing: bool = True,
        use_binpool_format: bool = False
    ) -> dict:
        if not self._initialized:
            if not self.setup_database():
                raise RuntimeError("Database setup failed")
        
        if not binary_paths:
            logger.warning("No binaries to process")
            return {"processed": 0, "skipped": 0}
        
        logger.info(f"Pipeline starting with {len(binary_paths)} binaries")
        logger.info(str(self.config))
        
        # Filter out already processed binaries
        if skip_existing:
            original_count = len(binary_paths)
            binary_paths = self._filter_existing(binary_paths)
            skipped = original_count - len(binary_paths)
            if skipped > 0:
                logger.info(f"Skipping {skipped} already processed binaries")
        else:
            skipped = 0
        
        if not binary_paths:
            logger.info("All binaries already processed")
            return {"processed": 0, "skipped": skipped}
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(binary_paths)
        
        file_queue = self.manager.Queue()
        result_queue = self.manager.Queue()
        
        consumers = self._start_consumers(result_queue)
        
        producers = self._start_producers(
            file_queue, result_queue, use_binpool_format
        )
        
        logger.info("Queuing binaries for processing...")
        for path in tqdm(binary_paths, desc="Queuing"):
            file_queue.put(path)
        
        for _ in producers:
            file_queue.put(None)
        
        logger.info("Waiting for producers to complete...")
        for p in tqdm(producers, desc="Producers"):
            p.join()
        
        for _ in consumers:
            result_queue.put(None)
        
        logger.info("Waiting for consumers to complete...")
        for c in tqdm(consumers, desc="Consumers"):
            c.join()
        
        final_counts = get_table_counts()
        logger.info(f"Pipeline completed. Final counts: {final_counts}")
        
        return {
            "processed": len(binary_paths),
            "skipped": skipped,
            "final_counts": final_counts
        }
    
    def run_from_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        **kwargs
    ) -> dict:
        logger.info(f"Scanning directory: {directory}")
        binary_paths = scan_for_binaries(directory, extensions)
        logger.info(f"Found {len(binary_paths)} binaries")
        
        return self.run(binary_paths, **kwargs)
    
    def run_from_json(
        self,
        json_path: str,
        **kwargs
    ) -> dict:
        logger.info(f"Loading binary list from: {json_path}")
        
        with open(json_path, 'r') as f:
            binary_paths = json.load(f)
        
        logger.info(f"Loaded {len(binary_paths)} binary paths")
        
        return self.run(binary_paths, **kwargs)
    
    def _start_producers(
        self,
        file_queue: Queue,
        result_queue: Queue,
        use_binpool_format: bool
    ) -> List[Process]:
        """Start producer worker processes."""
        producers = []
        
        for i in range(self.config.num_producers):
            p = Process(
                target=producer_worker,
                args=(i, file_queue, result_queue, self.config, use_binpool_format)
            )
            p.start()
            producers.append(p)
        
        logger.info(f"Started {len(producers)} producer workers")
        return producers
    
    def _start_consumers(self, result_queue: Queue) -> List[Process]:
        """Start consumer worker processes."""
        consumers = []
        
        for i in range(self.config.num_consumers):
            c = Process(
                target=consumer_worker_with_init,
                args=(i, result_queue, self.config)
            )
            c.start()
            consumers.append(c)
        
        logger.info(f"Started {len(consumers)} consumer workers")
        return consumers
    
    def _filter_existing(self, binary_paths: List[str]) -> List[str]:
        """Filter out binaries that have already been processed."""
        try:
            with DatabasePool.get_connection() as conn:
                existing = DatabaseOperations.get_processed_binaries(conn)
            
            # Create lookup set of file paths
            existing_files = {f[3] for f in existing}  # file_name is index 3
            
            return [
                path for path in binary_paths
                if path not in existing_files
            ]
        except Exception as e:
            logger.warning(f"Could not check existing binaries: {e}")
            return binary_paths
    
    def shutdown(self) -> None:
        """Clean up resources."""
        try:
            DatabasePool.close()
            logger.info("Pipeline shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
