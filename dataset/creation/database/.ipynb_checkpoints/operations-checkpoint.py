"""
Database operations-
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from psycopg2 import sql
from psycopg2.extras import execute_values
from psycopg2.extensions import connection as PgConnection
from loguru import logger


# SQL Statements
INSERT_FUNCTION_SQL = """
    INSERT INTO functions (
        project, compiler, optimization, file_name, function_name, address,
        num_instructions, num_basic_blocks, size, signature, args, edges, 
        edges_ghidra, num_basic_blocks_ghidra, offset_ghidra
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id;
"""

INSERT_BASIC_BLOCKS_SQL = """
    INSERT INTO basic_blocks (
        address, num_instructions, asm, is_call_jump, call_name,
        call_ins_address, n_args_call, function_id
    ) VALUES %s
"""

INSERT_BASIC_BLOCKS_GHIDRA_SQL = """
    INSERT INTO basic_blocks_ghidra (address, asm, calls, function_id)
    VALUES %s
"""

INSERT_PALMTREE_SQL = """
    INSERT INTO palmtree (function_id, strings, dfg_nodes, dfg_edges, cfg_nodes, call_map)
    VALUES (%s, %s, %s, %s, %s, %s)
"""

GET_FUNCTION_ID_SQL = """
    SELECT id FROM functions 
    WHERE project = %s AND compiler = %s AND optimization = %s 
    AND file_name = %s AND function_name = %s
    LIMIT 1;
"""

CHECK_BINARY_EXISTS_SQL = """
    SELECT 1 FROM functions 
    WHERE project = %s AND compiler = %s AND optimization = %s AND file_name = %s
    LIMIT 1;
"""


class DatabaseOperations:    
    @staticmethod
    def insert_function(
        conn: PgConnection,
        project: str,
        compiler: str,
        optimization: str,
        file_name: str,
        function_name: str,
        address: int,
        num_instructions: int,
        num_basic_blocks: int,
        size: int,
        signature: str,
        args: int,
        edges: str,
        edges_ghidra: str,
        num_basic_blocks_ghidra: int,
        offset_ghidra: int
    ) -> int:
        """
        Insert a function record and return its ID.
        
        Uses RETURNING to get the ID without an extra query.
        """
        with conn.cursor() as cursor:
            cursor.execute(INSERT_FUNCTION_SQL, (
                project, compiler, optimization, file_name, function_name,
                address, num_instructions, num_basic_blocks, size, signature,
                args, edges, edges_ghidra, num_basic_blocks_ghidra, offset_ghidra
            ))
            result = cursor.fetchone()
            return result[0] if result else None
    
    @staticmethod
    def insert_basic_blocks_bulk(
        conn: PgConnection,
        basic_blocks: List[Tuple],
        page_size: int = 20
    ) -> None:
        """
        Bulk insert basic blocks.
        
        Args:
            conn: Database connection
            basic_blocks: List of tuples (address, num_instructions, asm, is_call_jump,
                                         call_name, call_ins_address, n_args_call, function_id)
            page_size: Number of records per batch (default 20)
        """
        if not basic_blocks:
            return
        
        with conn.cursor() as cursor:
            execute_values(
                cursor,
                INSERT_BASIC_BLOCKS_SQL,
                basic_blocks,
                page_size=page_size
            )
    
    @staticmethod
    def insert_basic_blocks_ghidra_bulk(
        conn: PgConnection,
        ghidra_blocks: List[Tuple],
        page_size: int = 20
    ) -> None:
        """
        Bulk insert Ghidra basic blocks.
        
        Args:
            conn: Database connection
            ghidra_blocks: List of tuples (address, asm, calls, function_id)
            page_size: Number of records per batch (default 20)
        """
        if not ghidra_blocks:
            return
        
        with conn.cursor() as cursor:
            execute_values(
                cursor,
                INSERT_BASIC_BLOCKS_GHIDRA_SQL,
                ghidra_blocks,
                page_size=page_size
            )
    
    @staticmethod
    def insert_palmtree(
        conn: PgConnection,
        function_id: int,
        strings: str,
        dfg_nodes: str,
        dfg_edges: str,
        cfg_nodes: str,
        call_map: str
    ) -> None:
        """Insert palmtree information for a function."""
        with conn.cursor() as cursor:
            cursor.execute(INSERT_PALMTREE_SQL, (
                function_id, strings, dfg_nodes, dfg_edges, cfg_nodes, call_map
            ))
    
    @staticmethod
    def insert_function_with_all_data(
        conn: PgConnection,
        function_data: Dict[str, Any],
        basic_blocks: List[Tuple],
        ghidra_blocks: List[Tuple],
        palmtree_data: Dict[str, str],
        batch_size: int = 20
    ) -> Optional[int]:
        """
        Insert a function and all its related data in a single transaction.
        
        This is the main entry point for inserting analysis results.
        All inserts happen within the same transaction, so either all
        succeed or all are rolled back.
        
        Args:
            conn: Database connection
            function_data: Dict with function metadata
            basic_blocks: List of basic block tuples
            ghidra_blocks: List of Ghidra block tuples
            palmtree_data: Dict with palmtree information
            batch_size: Batch size for bulk inserts
            
        Returns:
            The function_id if successful, None otherwise
        """
        try:
            # Insert function and get ID
            function_id = DatabaseOperations.insert_function(
                conn,
                project=function_data['project'],
                compiler=function_data['compiler'],
                optimization=function_data['optimization'],
                file_name=function_data['file_name'],
                function_name=function_data['function_name'],
                address=function_data['address'],
                num_instructions=function_data['num_instructions'],
                num_basic_blocks=function_data['num_basic_blocks'],
                size=function_data['size'],
                signature=function_data['signature'],
                args=function_data['args'],
                edges=function_data['edges'],
                edges_ghidra=function_data['edges_ghidra'],
                num_basic_blocks_ghidra=function_data['num_basic_blocks_ghidra'],
                offset_ghidra=function_data['offset_ghidra']
            )
            
            if function_id is None:
                logger.error(f"Failed to get function_id for {function_data['function_name']}")
                return None
            
            # Add function_id to all basic blocks
            bbs_with_id = [
                (bb[0], bb[1], bb[2], bb[3], bb[4], bb[5], bb[6], function_id)
                for bb in basic_blocks
            ]
            
            # Insert basic blocks
            if bbs_with_id:
                DatabaseOperations.insert_basic_blocks_bulk(
                    conn, bbs_with_id, page_size=batch_size
                )
            
            # Add function_id to all Ghidra blocks
            ghidra_with_id = [
                (gb[0], gb[1], gb[2], function_id)
                for gb in ghidra_blocks
            ]
            
            # Insert Ghidra blocks
            if ghidra_with_id:
                DatabaseOperations.insert_basic_blocks_ghidra_bulk(
                    conn, ghidra_with_id, page_size=batch_size
                )
            
            # Insert palmtree info
            DatabaseOperations.insert_palmtree(
                conn,
                function_id=function_id,
                strings=palmtree_data.get('strings', '{}'),
                dfg_nodes=palmtree_data.get('dfg_nodes', ''),
                dfg_edges=palmtree_data.get('dfg_edges', ''),
                cfg_nodes=palmtree_data.get('cfg_nodes', '[]'),
                call_map=palmtree_data.get('call_map', '{}')
            )
            
            return function_id
            
        except Exception as e:
            logger.error(f"Error inserting function {function_data.get('function_name', 'unknown')}: {e}")
            raise
    
    @staticmethod
    def binary_exists(conn: PgConnection, project: str, compiler: str, 
                      optimization: str, file_name: str) -> bool:
        """Check if a binary has already been processed."""
        with conn.cursor() as cursor:
            cursor.execute(CHECK_BINARY_EXISTS_SQL, (
                project, compiler, optimization, file_name
            ))
            return cursor.fetchone() is not None
    
    @staticmethod
    def get_function_id(
        conn: PgConnection,
        project: str,
        compiler: str,
        optimization: str,
        file_name: str,
        function_name: str
    ) -> Optional[int]:
        """Get function ID by its identifying attributes."""
        with conn.cursor() as cursor:
            cursor.execute(GET_FUNCTION_ID_SQL, (
                project, compiler, optimization, file_name, function_name
            ))
            result = cursor.fetchone()
            return result[0] if result else None
    
    @staticmethod
    def get_processed_binaries(conn: PgConnection) -> set:
        """Get set of already processed binary identifiers."""
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT DISTINCT project, compiler, optimization, file_name
                FROM functions
            """)
            return {
                (row[0], row[1], row[2], row[3])
                for row in cursor.fetchall()
            }
