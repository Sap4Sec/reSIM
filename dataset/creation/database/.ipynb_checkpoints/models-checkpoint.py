"""
Database table definitions and schema management.
"""

from loguru import logger
from .connection import DatabasePool


# SQL for creating the functions table
CREATE_FUNCTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS functions (
    id SERIAL PRIMARY KEY,
    project VARCHAR(255),
    compiler VARCHAR(50),
    optimization VARCHAR(10),
    file_name TEXT,
    function_name TEXT,
    address BIGINT,
    num_instructions INTEGER,
    num_basic_blocks INTEGER,
    size INTEGER,
    signature TEXT,
    args INTEGER,
    edges TEXT,
    edges_ghidra TEXT,
    num_basic_blocks_ghidra INTEGER,
    offset_ghidra BIGINT
);
"""

# SQL for creating the basic_blocks table
CREATE_BASIC_BLOCKS_TABLE = """
CREATE TABLE IF NOT EXISTS basic_blocks (
    id SERIAL PRIMARY KEY,
    address BIGINT,
    num_instructions INTEGER,
    asm BYTEA,
    is_call_jump BOOLEAN,
    call_name TEXT,
    call_ins_address BIGINT,
    n_args_call INTEGER,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE
);
"""

# SQL for creating the basic_blocks_ghidra table
CREATE_BASIC_BLOCKS_GHIDRA_TABLE = """
CREATE TABLE IF NOT EXISTS basic_blocks_ghidra (
    id SERIAL PRIMARY KEY,
    address BIGINT,
    asm TEXT,
    calls TEXT,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE
);
"""

# SQL for creating the palmtree table
CREATE_PALMTREE_TABLE = """
CREATE TABLE IF NOT EXISTS palmtree (
    id SERIAL PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    strings TEXT,
    dfg_nodes TEXT,
    dfg_edges TEXT,
    cfg_nodes TEXT,
    call_map TEXT
);
"""

# Indexes for better query performance
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_functions_project ON functions(project);",
    "CREATE INDEX IF NOT EXISTS idx_functions_file_name ON functions(file_name);",
    "CREATE INDEX IF NOT EXISTS idx_functions_function_name ON functions(function_name);",
    "CREATE INDEX IF NOT EXISTS idx_functions_composite ON functions(project, compiler, optimization, file_name, function_name);",
    "CREATE INDEX IF NOT EXISTS idx_basic_blocks_function_id ON basic_blocks(function_id);",
    "CREATE INDEX IF NOT EXISTS idx_basic_blocks_address ON basic_blocks(address);",
    "CREATE INDEX IF NOT EXISTS idx_basic_blocks_ghidra_function_id ON basic_blocks_ghidra(function_id);",
    "CREATE INDEX IF NOT EXISTS idx_palmtree_function_id ON palmtree(function_id);",
]

# All table creation statements in order
ALL_TABLES = [
    ("functions", CREATE_FUNCTIONS_TABLE),
    ("basic_blocks", CREATE_BASIC_BLOCKS_TABLE),
    ("basic_blocks_ghidra", CREATE_BASIC_BLOCKS_GHIDRA_TABLE),
    ("palmtree", CREATE_PALMTREE_TABLE),
]


def create_tables() -> None:
    """
    Create all required database tables if they don't exist.
    
    Requires DatabasePool to be initialized first.
    """
    with DatabasePool.get_connection() as conn:
        with conn.cursor() as cursor:
            # Create tables
            for table_name, create_sql in ALL_TABLES:
                try:
                    cursor.execute(create_sql)
                    logger.debug(f"Table '{table_name}' created/verified")
                except Exception as e:
                    logger.error(f"Failed to create table '{table_name}': {e}")
                    raise
            
            # Create indexes
            for index_sql in CREATE_INDEXES:
                try:
                    cursor.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")
    
    logger.info("All database tables and indexes created/verified")


def drop_tables() -> None:
    """
    Drop all tables. USE WITH CAUTION - this will delete all data!
    """
    with DatabasePool.get_connection() as conn:
        with conn.cursor() as cursor:
            # Drop in reverse order due to foreign key constraints
            for table_name, _ in reversed(ALL_TABLES):
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
                    logger.debug(f"Table '{table_name}' dropped")
                except Exception as e:
                    logger.error(f"Failed to drop table '{table_name}': {e}")
                    raise
    
    logger.warning("All tables dropped")


def truncate_tables() -> None:
    """
    Truncate all tables (delete all data but keep structure).
    """
    with DatabasePool.get_connection() as conn:
        with conn.cursor() as cursor:
            # Truncate in reverse order due to foreign key constraints
            for table_name, _ in reversed(ALL_TABLES):
                try:
                    cursor.execute(f"TRUNCATE TABLE {table_name} CASCADE;")
                    logger.debug(f"Table '{table_name}' truncated")
                except Exception as e:
                    logger.error(f"Failed to truncate table '{table_name}': {e}")
                    raise
    
    logger.warning("All tables truncated")


def table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    with DatabasePool.get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table_name,))
            return cursor.fetchone()[0]


def get_table_counts() -> dict:
    """Get row counts for all tables."""
    counts = {}
    with DatabasePool.get_connection() as conn:
        with conn.cursor() as cursor:
            for table_name, _ in ALL_TABLES:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                    counts[table_name] = cursor.fetchone()[0]
                except Exception:
                    counts[table_name] = -1  # Table might not exist
    return counts
