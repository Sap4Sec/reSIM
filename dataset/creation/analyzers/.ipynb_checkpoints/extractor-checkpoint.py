import os
import signal
import json
import logging
import cxxfilt
import re
import string
import claripy
from typing import Dict, Any, Optional, List, Set, Tuple

logging.getLogger("angr").setLevel(logging.CRITICAL)
logging.getLogger("cle").setLevel(logging.CRITICAL)
logging.getLogger("pyvex").setLevel(logging.CRITICAL)
logging.getLogger("claripy").setLevel(logging.CRITICAL)

import angr
from loguru import logger

from ..config import Config
from ..analyzers.ghidra_analyzer import GhidraAnalyzer
from ..utils.helpers import format_nodes, format_edges

class AngrTimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise AngrTimeoutException

def is_valid_block(node):
    try:
        tmp = node.instructions
        tmp = node.bytes
    except Exception:
        tmp = None
    return tmp is not None

def patched_string_references(function, blocks, constants, minimum_length=2, vex_only=False):
    strings = []
    memory = function._project.loader.memory

    known_executable_addresses = set()
    for block in blocks:
        known_executable_addresses.update(block.instruction_addrs)
    for func in function._function_manager.values():
        known_executable_addresses.update(set(x.addr for x in func.graph.nodes()))

    for addr in function.local_runtime_values if not vex_only else constants:
        if not isinstance(addr, claripy.fp.FSort) and not isinstance(addr, float) and addr in memory:
            try:
                possible_pointer = memory.unpack_word(addr)
                if addr not in known_executable_addresses and possible_pointer not in known_executable_addresses:
                    stn = ""
                    offset = 0
                    current_char = chr(memory[addr + offset])
                    while current_char in string.printable:
                        stn += current_char
                        offset += 1
                        current_char = chr(memory[addr + offset])

                    if current_char == "\x00" and len(stn) >= minimum_length:
                        strings.append((addr, stn))
            except KeyError:
                pass
    return strings

class BinaryExtractor:
    """
    Extracts function information using Angr and Ghidra.
    """
    def __init__(self, config: Config):
        self.config = config
        self.ghidra_analyzer = GhidraAnalyzer(config)
        
        self.func_black_list = {
            '_init', '__cxa_finalize', '_start', 'deregister_tm_clones',
            'register_tm_clones', '__do_global_dtors_aux', 'frame_dummy',
            '__libc_csu_init', '__libc_start_main', '_fini', '__libc_csu_fini'
        }
        
        self.libc_signatures = self._load_libc_signatures()
        
        self.proj = None
        self.cfg = None
        self.mangling_dict = {}
        # self.is_stripped = False
        self.offset_ghidra = 0
        self.func_cfg_dict = {}

    def _load_libc_signatures(self) -> Dict[str, Any]:
        try:
            with open(self.config.libc_signatures_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load libc signatures: {e}")
            return {}

    def extract(self, binary_path: str, base_addr_zero: bool = False, cfg_timeout_minutes: int = 20) -> Dict[int, Dict[str, Any]]:
        self.func_cfg_dict = {}
        try:
            self._init_angr(binary_path, base_addr_zero)
            
            self._build_cfg(cfg_timeout_minutes)
            
            self._build_mangling_dict()
            
            ghidra_functions = self.ghidra_analyzer.analyze(
                binary_path,
                entry_point_angr=self.proj.entry,
                base_addr_zero=base_addr_zero
            )
            
            if not ghidra_functions:
                logger.warning(f"No functions from Ghidra for: {binary_path}")
                return {}
            
            return self._extract_all_func_cfg(ghidra_functions, return_strings=True)
            
        except AngrTimeoutException:
            logger.warning(f"CFG analysis timeout for: {binary_path}")
            return {}
        except Exception as e:
            logger.error(f"Extraction error for {binary_path}: {e}")
            return {}
        finally:
            self._cleanup()

    def _init_angr(self, binary_path: str, base_addr_zero: bool) -> None:
        if base_addr_zero:
            self.proj = angr.Project(binary_path, auto_load_libs=False, main_opts={'base_addr': 0x0})
        else:
            self.proj = angr.Project(binary_path, auto_load_libs=False)
            
        # # Detect stripped
        # has_symtab = self.proj.loader.main_object.sections_map.get('.symtab') is not None
        # has_dynsym = self.proj.loader.main_object.sections_map.get('.dynsym') is not None
        # function_symbols = [s for s in self.proj.loader.main_object.symbols if s.is_function]
        # 
        # if has_dynsym and not has_symtab:
        #     self.is_stripped = True
        #     logger.info(f"Detected stripped shared library: {binary_path}")
        # elif not has_symtab and len(function_symbols) < 5:
        #     self.is_stripped = True
        #     logger.info(f"Detected stripped executable: {binary_path}")
        # else:
        #     self.is_stripped = False
        #     logger.info(f"Detected non-stripped binary: {binary_path}")
        # self.is_stripped = False

    def _build_cfg(self, timeout_minutes: int) -> None:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_minutes * 60)
        try:
            self.cfg = self.proj.analyses.CFGFast()
        finally:
            signal.alarm(0)

    def _build_mangling_dict(self) -> None:
        self.mangling_dict = {}
        # if self.is_stripped:
        #     return
            
        for sym in self.proj.loader.main_object.symbols:
            if sym.is_function and sym.name.startswith('_Z'):
                try:
                    demangled_name = cxxfilt.demangle(sym.name)
                    dict_key = demangled_name.split("(")[0]
                    # Cleanups from original code
                    for prefix in ["void ", "const int ", "const "]:
                        if dict_key.startswith(prefix):
                            dict_key = dict_key[len(prefix):]
                    dict_key = dict_key.replace(" ", "")
                    self.mangling_dict[dict_key] = sym.name
                except Exception:
                    pass
        logger.info(f"Built mangling dictionary with {len(self.mangling_dict)} entries")

    def _extract_all_func_cfg(self, ghidra_functions: List[Dict], return_strings: bool) -> Dict[int, Dict[str, Any]]:
        self.offset_ghidra = self.ghidra_analyzer.offset if self.ghidra_analyzer.offset is not None else 0
        
        logger.info(f"Functions retrieved from Ghidra: {len(ghidra_functions)}")
        
        for function_ghidra in ghidra_functions:
            name_to_use = function_ghidra["name"]
            
            function_angr_symbol = self.proj.loader.find_symbol(name_to_use)
            
            if function_angr_symbol is None:
                if name_to_use in self.mangling_dict:
                    mangled_name = self.mangling_dict[name_to_use]
                    sym = self.proj.loader.find_symbol(mangled_name)
                    if sym:
                         function_angr_symbol = sym

            function_angr_address = function_ghidra["address"] - self.offset_ghidra
            
            if function_angr_symbol is not None:
                function_angr_address = function_angr_symbol.rebased_addr
            
            function_angr = None
            try:
                function_angr = self.cfg.functions[function_angr_address]
            except KeyError:
                continue

            nodes = [(node, self._check_jump_kind(node)) for node in list(function_angr.blocks) if is_valid_block(node)]
            
            nodes_dict = {node.addr: node for node, _ in nodes}
            edges = [(nodes_dict[src.addr], nodes_dict[dst.addr]) for src, dst in function_angr.graph.edges()
                     if src.addr in nodes_dict and dst.addr in nodes_dict]

            nodes_ghidra = {}
            if "block_bytecode_dict" in function_ghidra:
                for node in function_ghidra["block_bytecode_dict"]:
                    nodes_ghidra[node[0]] = node
            
            edges_ghidra = []
            if "edges" in function_ghidra:
                for ed in function_ghidra["edges"]:
                    if ed[0] in nodes_ghidra and ed[1] in nodes_ghidra:
                        edges_ghidra.append(ed)

            entry_addr = function_ghidra["address"]
            self.func_cfg_dict[entry_addr] = {
                "name": name_to_use,
                "signature": function_ghidra.get("signature", ""),
                "size": function_ghidra.get("size", 0),
                "args": function_ghidra.get("args", 0),
                "address": entry_addr,
                "offset_ghidra": self.offset_ghidra,
                "nodes": nodes,
                "edges": edges,
                "nodes_ghidra": function_ghidra.get("block_bytecode_dict", []),
                "edges_ghidra": edges_ghidra
            }
            
            if return_strings:
                blocks = [node for node, _ in nodes]
                code_constants = [const.value for block in blocks for const in block.vex.constants]
                strings = patched_string_references(function_angr, blocks, code_constants, vex_only=True)
                string_map = {addr: value for addr, value in strings}
                self.func_cfg_dict[entry_addr]["strings"] = string_map
                
                call_map = {}
                if hasattr(function_angr, "functions_called"):
                    for func_obj in function_angr.functions_called():
                         call_map[func_obj.addr] = func_obj.name
                self.func_cfg_dict[entry_addr]["call_map"] = call_map

        return self.func_cfg_dict

    def _check_jump_kind(self, block):
        is_call_jump, call_name, call_ins_address, n_args_call = False, None, None, 0
        if block.vex.jumpkind == "Ijk_Call":
            is_call_jump = True
            try:
                call_ins = block.capstone.insns[-1]
                call_ins_address = call_ins.address
                if len(call_ins.operands) > 0:
                    called_address = call_ins.operands[0].imm
                    if called_address is not None:
                        if called_address in self.func_cfg_dict: 
                             n_args_call = self.func_cfg_dict[called_address]["args"]
                        else:
                            symbol = self.proj.loader.find_symbol(called_address)
                            try:
                                if symbol and symbol.name:
                                    call_name = symbol.name
                                else:
                                    call_name = self.proj.loader.main_object.reverse_plt.get(called_address)
                            except Exception:
                                pass
                            
                            if call_name and call_name in self.libc_signatures:
                                n_args_call = self.libc_signatures[call_name]["args"]
            except Exception:
                pass
        return is_call_jump, call_name, call_ins_address, n_args_call

    def _cleanup(self):
        self.proj = None
        self.cfg = None
