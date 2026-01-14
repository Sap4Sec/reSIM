"""
Command-line interface for the binary disassembly pipeline.
"""

import sys
import os
import argparse
from pathlib import Path

from loguru import logger

from config import Config, create_config_template
from pipeline import DisassemblyPipeline
from database.connection import DatabasePool
from database.models import get_table_counts, drop_tables, truncate_tables


def setup_logging(log_level: str, log_dir: str):
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>"
    )
    
    # Add file handler
    os.makedirs(log_dir, exist_ok=True)
    logger.add(
        os.path.join(log_dir, "pipeline_{time}.log"),
        level=log_level,
        rotation="100 MB"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Binary Disassembly Pipeline for ML Dataset Creation",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Database options
    db_group = parser.add_argument_group("Database")
    db_group.add_argument(
        "--db-host",
        default=os.environ.get("PIPELINE_DB_HOST", "postgres"),
        help="Database host (default: postgres)"
    )
    db_group.add_argument(
        "--db-port",
        type=int,
        default=int(os.environ.get("PIPELINE_DB_PORT", "5432")),
        help="Database port (default: 5432)"
    )
    db_group.add_argument(
        "--db-name",
        default=os.environ.get("PIPELINE_DB_NAME"),
        help="Database name (required)"
    )
    db_group.add_argument(
        "--db-user",
        default=os.environ.get("PIPELINE_DB_USER", "jovyan"),
        help="Database user (default: jovyan)"
    )
    db_group.add_argument(
        "--db-password",
        default=os.environ.get("PIPELINE_DB_PASSWORD", "secret"),
        help="Database password"
    )
    
    # Processing options
    proc_group = parser.add_argument_group("Processing")
    proc_group.add_argument(
        "-p", "--producers",
        type=int,
        default=4,
        help="Number of producer workers (default: 4)"
    )
    proc_group.add_argument(
        "-c", "--consumers",
        type=int,
        default=2,
        help="Number of consumer workers (default: 2)"
    )
    proc_group.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)"
    )
    proc_group.add_argument(
        "--insert-batch",
        type=int,
        default=20,
        help="Batch size for DB inserts (default: 20)"
    )
    
    # Input options
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "-i", "--input-file",
        type=str,
        help="JSON file with list of binary paths"
    )
    input_group.add_argument(
        "-d", "--input-dir",
        type=str,
        help="Directory to scan for binaries"
    )
    input_group.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        help="File extensions to include when scanning (e.g., .o .so)"
    )
    input_group.add_argument(
        "--binpool",
        action="store_true",
        help="Use BinPool CVE dataset path format"
    )
    input_group.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip already processed binaries"
    )
    
    # Configuration options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file"
    )
    config_group.add_argument(
        "--ghidra-path",
        type=str,
        help="Path to Ghidra installation"
    )
    config_group.add_argument(
        "--libc-signatures",
        type=str,
        help="Path to libc signatures JSON"
    )
    
    # Actions
    action_group = parser.add_argument_group("Actions")
    action_group.add_argument(
        "--create-tables",
        action="store_true",
        help="Create database tables"
    )
    action_group.add_argument(
        "--create-db",
        action="store_true",
        help="Create database if it doesn't exist"
    )
    action_group.add_argument(
        "--drop-tables",
        action="store_true",
        help="Drop all tables (DANGER: deletes all data)"
    )
    action_group.add_argument(
        "--truncate-tables",
        action="store_true",
        help="Truncate all tables (DANGER: deletes all data)"
    )
    action_group.add_argument(
        "--stats",
        action="store_true",
        help="Show table statistics"
    )
    action_group.add_argument(
        "--create-config",
        action="store_true",
        help="Create config.yaml template"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )
    parser.add_argument(
        "--log-dir",
        default="../logs",
        help="Log directory (default: ../logs)"
    )
    
    args = parser.parse_args()
    
    # Handle config file creation
    if args.create_config:
        create_config_template()
        return 0
    
    # Setup logging
    setup_logging(args.log_level, args.log_dir)
    
    # Build configuration
    config = Config.load(args.config) if args.config else Config.load()
    
    # Override with command-line arguments
    if args.db_host:
        config.db_host = args.db_host
    if args.db_port:
        config.db_port = args.db_port
    if args.db_name:
        config.db_name = args.db_name
    if args.db_user:
        config.db_user = args.db_user
    if args.db_password:
        config.db_password = args.db_password
    if args.producers:
        config.num_producers = args.producers
    if args.consumers:
        config.num_consumers = args.consumers
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.insert_batch:
        config.insert_batch_size = args.insert_batch
    if args.ghidra_path:
        config.ghidra_path = args.ghidra_path
    if args.libc_signatures:
        config.libc_signatures_path = args.libc_signatures
    config.log_level = args.log_level
    config.log_dir = args.log_dir
    
    # Validate required arguments
    if not config.db_name:
        parser.error("--db-name is required")
    
    # Handle special actions
    if args.drop_tables:
        if input("Are you sure you want to DROP all tables? (yes/no): ") == "yes":
            DatabasePool.initialize(config)
            drop_tables()
            logger.info("Tables dropped")
        return 0
    
    if args.truncate_tables:
        if input("Are you sure you want to TRUNCATE all tables? (yes/no): ") == "yes":
            DatabasePool.initialize(config)
            truncate_tables()
            logger.info("Tables truncated")
        return 0
    
    if args.stats:
        DatabasePool.initialize(config)
        counts = get_table_counts()
        print("\nTable Statistics:")
        for table, count in counts.items():
            print(f"  {table}: {count:,} rows")
        return 0
    
    # Validate input
    if not args.input_file and not args.input_dir:
        if not args.create_tables:
            parser.error("Either --input-file, --input-dir, or --create-tables is required")
    
    # Create and run pipeline
    pipeline = DisassemblyPipeline(config)
    
    try:
        # Setup database
        if not pipeline.setup_database(create_db=args.create_db):
            logger.error("Database setup failed")
            return 1
        
        # If only creating tables, we're done
        if not args.input_file and not args.input_dir:
            logger.info("Tables created successfully")
            return 0
        
        # Run pipeline
        if args.input_file:
            stats = pipeline.run_from_json(
                args.input_file,
                skip_existing=not args.no_skip,
                use_binpool_format=args.binpool
            )
        else:
            stats = pipeline.run_from_directory(
                args.input_dir,
                extensions=args.extensions,
                skip_existing=not args.no_skip,
                use_binpool_format=args.binpool
            )
        
        logger.info(f"Pipeline completed: {stats}")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    sys.exit(main())
