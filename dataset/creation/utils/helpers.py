"""
Utility functions for the pipeline.
"""

import os
import re
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class BinaryMetadata:
    """Metadata extracted from binary file path."""
    project: str
    compiler: str
    optimization: str
    file_name: str
    full_path: str


def parse_binary_path(file_path: str) -> BinaryMetadata:
    """Parse binary file path to extract metadata."""
    tokens = file_path.split(os.sep)
    
    # Try to extract from path structure
    file_name = tokens[-1] if tokens else file_path
    optimization = tokens[-2] if len(tokens) >= 2 else "unknown"
    compiler = tokens[-3] if len(tokens) >= 3 else "unknown"
    project = tokens[-4] if len(tokens) >= 4 else "unknown"
    
    return BinaryMetadata(
        project=project,
        compiler=compiler,
        optimization=optimization,
        file_name=file_name,
        full_path=file_path
    )


def parse_binary_path_binpool(file_path: str) -> BinaryMetadata:
    """Parse binary file path for BinPool CVE dataset format."""
    tokens = file_path.split(os.sep)
    
    file_name = file_path  # Use full path as filename for uniqueness
    project = next((token for token in tokens if token.startswith('CVE-')), "Unknown")
    optimization = next((token for token in tokens if token.startswith('opt')), "Unknown")
    compiler = "clang"  # Default for BinPool
    
    return BinaryMetadata(
        project=project,
        compiler=compiler,
        optimization=optimization,
        file_name=file_name,
        full_path=file_path
    )


def format_nodes(nodes: list) -> List[Dict[str, Any]]:
    """Format angr block nodes into a serializable format."""
    formatted = []
    
    for node, jump_info in nodes:
        is_call_jump, call_name, call_ins_address, n_args_call = jump_info
        
        # Extract capstone instructions
        capstone_ins = []
        try:
            for ins in node.capstone.insns:
                capstone_ins.append({
                    'address': ins.address,
                    'mnemonic': ins.mnemonic,
                    'op_str': ins.op_str,
                    'bytes': bytes(ins.bytes).hex()
                })
        except Exception:
            pass
        
        formatted.append({
            'block_address': node.addr,
            'n_instructions': node.instructions,
            'bytes': bytes(node.bytes).hex(),
            'is_call_jump': is_call_jump,
            'call_name': call_name,
            'call_ins_address': call_ins_address,
            'n_args_call': n_args_call,
            'capstone_ins': capstone_ins
        })
    
    return formatted


def format_edges(edges: list) -> List[Tuple[int, int]]:
    """Format angr edges into a serializable format."""
    formatted = []
    
    for src, dst in edges:
        try:
            formatted.append((src.addr, dst.addr))
        except AttributeError:
            # Already formatted or invalid
            if isinstance(src, int) and isinstance(dst, int):
                formatted.append((src, dst))
    
    return formatted


def palmtree_instruction_parsing(
    ins: Dict[str, Any],
    call_map: Dict[int, str],
    func_strings: Dict[int, str]
) -> str:
    mnemonic = ins.get('mnemonic', '')
    op_str = ins.get('op_str', '')
    
    # Replace call targets with function names
    if mnemonic.lower() == 'call':
        for addr, name in call_map.items():
            if hex(addr) in op_str or str(addr) in op_str:
                op_str = name
                break
    
    # Replace string references
    for addr, string_val in func_strings.items():
        hex_addr = hex(addr)
        if hex_addr in op_str:
            # Replace with string literal (truncated)
            safe_str = string_val[:50].replace('"', "'")
            op_str = op_str.replace(hex_addr, f'"{safe_str}"')
    
    # Normalize immediate values (optional)
    # op_str = re.sub(r'0x[0-9a-fA-F]+', 'IMM', op_str)
    
    return f"{mnemonic} {op_str}".strip()


def prepare_function_data(
    function_info: Dict[str, Any],
    metadata: BinaryMetadata
) -> Dict[str, Any]:
    nodes = format_nodes(function_info.get('nodes', []))
    edges = format_edges(function_info.get('edges', []))
    ghidra_edges = function_info.get('edges_ghidra', [])
    ghidra_nodes = function_info.get('nodes_ghidra', [])
    
    # Compute total instructions
    num_instructions = sum(block.get('n_instructions', 0) for block in nodes)
    
    return {
        'project': metadata.project,
        'compiler': metadata.compiler,
        'optimization': metadata.optimization,
        'file_name': metadata.file_name,
        'function_name': function_info.get('name', 'unknown'),
        'address': function_info.get('address', 0),
        'num_instructions': num_instructions,
        'num_basic_blocks': len(nodes),
        'size': function_info.get('size', 0),
        'signature': function_info.get('signature', ''),
        'args': function_info.get('args', 0),
        'edges': json.dumps(edges),
        'edges_ghidra': json.dumps(ghidra_edges),
        'num_basic_blocks_ghidra': len(ghidra_nodes),
        'offset_ghidra': function_info.get('offset_ghidra', 0)
    }


def prepare_basic_blocks(
    function_info: Dict[str, Any]
) -> List[Tuple]:
    nodes = format_nodes(function_info.get('nodes', []))
    
    blocks = []
    for block in nodes:
        blocks.append((
            block['block_address'],
            block['n_instructions'],
            bytes.fromhex(block['bytes']),  # Convert hex string to bytes
            block['is_call_jump'],
            block['call_name'],
            block['call_ins_address'],
            block['n_args_call']
            # function_id added later
        ))
    
    return blocks


def prepare_ghidra_blocks(
    function_info: Dict[str, Any]
) -> List[Tuple]:
    ghidra_nodes = function_info.get('nodes_ghidra', [])
    
    blocks = []
    for node in ghidra_nodes:
        # node format: (address, bytecode, calls)
        if isinstance(node, (list, tuple)) and len(node) >= 3:
            blocks.append((
                node[0],           # address
                node[1],           # asm/bytecode
                json.dumps(node[2])  # calls
                # function_id added later
            ))
    
    return blocks


def prepare_palmtree_data(
    function_info: Dict[str, Any]
) -> Dict[str, str]:
    call_map = function_info.get('call_map', {})
    func_strings = function_info.get('strings', {})
    nodes = format_nodes(function_info.get('nodes', []))
    
    # Parse CFG nodes for PalmTree
    cfg_nodes = []
    for node in nodes:
        capstone_ins = node.get('capstone_ins', [])
        instructions = [
            palmtree_instruction_parsing(ins, call_map, func_strings)
            for ins in capstone_ins
        ]
        cfg_nodes.append((node['block_address'], instructions))
    
    return {
        'strings': json.dumps(func_strings),
        'dfg_nodes': '',  # DFG extraction is optional
        'dfg_edges': '',
        'cfg_nodes': json.dumps(cfg_nodes),
        'call_map': json.dumps(call_map)
    }


def scan_for_binaries(root_dir: str, extensions: Optional[List[str]] = None) -> List[str]:
    file_list = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.startswith('.'):
                continue
            
            if extensions:
                if not any(file.endswith(ext) for ext in extensions):
                    continue
            
            file_list.append(os.path.join(root, file))
    
    return file_list
