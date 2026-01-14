"""
Thread-safe database connection pooling for PostgreSQL.
"""

import threading
from contextlib import contextmanager
from typing import Optional, Any, Generator

from psycopg2 import connect, sql
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extensions import connection as PgConnection
from loguru import logger

from ..config import Config


class DatabasePool:
    _instance: Optional["DatabasePool"] = None
    _lock = threading.Lock()
    _pool: Optional[ThreadedConnectionPool] = None
    _config: Optional[Config] = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, config: Config) -> None:
        """Initialize the connection pool. Should be called once at startup."""
        with cls._lock:
            if cls._pool is not None:
                logger.warning("DatabasePool already initialized, skipping")
                return
            
            cls._config = config
            
            try:
                cls._pool = ThreadedConnectionPool(
                    minconn=config.db_pool_min_conn,
                    maxconn=config.db_pool_max_conn,
                    host=config.db_host,
                    port=config.db_port,
                    database=config.db_name,
                    user=config.db_user,
                    password=config.db_password
                )
                logger.info(
                    f"Database pool initialized: {config.db_host}:{config.db_port}/{config.db_name} "
                    f"(min={config.db_pool_min_conn}, max={config.db_pool_max_conn})"
                )
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the pool has been initialized."""
        return cls._pool is not None
    
    @classmethod
    @contextmanager
    def get_connection(cls) -> Generator[PgConnection, None, None]:
        """Get a connection from the pool as a context manager."""
        if cls._pool is None:
            raise RuntimeError("DatabasePool not initialized. Call initialize() first.")
        
        conn = cls._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error, transaction rolled back: {e}")
            raise
        finally:
            cls._pool.putconn(conn)
    
    @classmethod
    def close(cls) -> None:
        """Close all connections in the pool."""
        with cls._lock:
            if cls._pool is not None:
                cls._pool.closeall()
                cls._pool = None
                logger.info("Database pool closed")
    
    @classmethod
    def get_stats(cls) -> dict:
        """Get pool statistics (for debugging)."""
        if cls._pool is None:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "min_connections": cls._config.db_pool_min_conn if cls._config else None,
            "max_connections": cls._config.db_pool_max_conn if cls._config else None,
        }


def create_database_if_not_exists(config: Config) -> None:
    """Create the database if it doesn't exist. 
    Connects to the 'postgres' database to check/create the target database."""
    conn = connect(
        host=config.db_host,
        port=config.db_port,
        database="postgres",
        user=config.db_user,
        password=config.db_password
    )
    conn.autocommit = True
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (config.db_name,)
            )
            exists = cursor.fetchone() is not None
            
            if not exists:
                # Create database
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(config.db_name)
                    )
                )
                logger.info(f"Created database: {config.db_name}")
            else:
                logger.debug(f"Database already exists: {config.db_name}")
    finally:
        conn.close()


def test_connection(config: Config) -> bool:
    """Test database connection with given config."""
    try:
        conn = connect(
            host=config.db_host,
            port=config.db_port,
            database=config.db_name,
            user=config.db_user,
            password=config.db_password
        )
        conn.close()
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False
