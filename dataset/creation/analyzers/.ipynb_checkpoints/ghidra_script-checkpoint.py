# Ghidra analysis script for function and basic block extraction
# This script is executed by Ghidra in headless mode

import json
import time
from ghidra.program.model.block import BasicBlockModel
from ghidra.program.model.listing import *
from ghidra.program.model.symbol import *
from ghidra.program.flatapi import FlatProgramAPI
from ghidra.util import Msg
from ghidra.program.model.symbol import SourceType


class GhidraAnalyzer:
    
    @staticmethod
    def get_function_list():
        functions = []
        
        for f in currentProgram.getFunctionManager().getFunctions(True):
            if not f.isThunk():
                original_symbol = f.getSymbol().getName()
                
                functions.append([
                    f.getName(),                    # [0] Current name
                    f.getName(True),                # [1] Full namespace name
                    f.getEntryPoint(),              # [2] Entry point
                    f,                              # [3] Function object
                    f.getSignature(),               # [4] Signature
                    f.getParameterCount(),          # [5] Parameter count
                    f.getEntryPoint().getPhysicalAddress(),  # [6] Physical address
                    original_symbol                 # [7] Original symbol (mangled if C++)
                ])
        
        return functions

    @staticmethod
    def analyze():
        Msg.info(None, "=== STARTING ANALYSIS ===")
        api = FlatProgramAPI(currentProgram)
        
        demangled_names = {}
        functionManager = currentProgram.getFunctionManager()
        all_funcs = functionManager.getFunctions(True)
        
        for func in all_funcs:
            if not func.isThunk():
                symbol = func.getSymbol()
                demangled_names[func.getEntryPoint()] = symbol.getName()
        
        rename_count = 0
        for func in functionManager.getFunctions(True):
            if func.isThunk():
                continue
            
            symbol = func.getSymbol()
            if symbol.getSource() in [SourceType.DEFAULT, SourceType.ANALYSIS]:
                addr = func.getEntryPoint()
                new_name = "FUN_" + addr.toString().replace(":", "")
                func.setName(new_name, SourceType.ANALYSIS)
                rename_count += 1
        
        Msg.info(None, "Renamed {} functions".format(rename_count))
        
        # Get function list
        retrieved_functions = GhidraAnalyzer.get_function_list()
        Msg.info(None, "Found {} functions".format(len(retrieved_functions)))
        
        # Functions to skip
        func_black_list = {
            '_init', '__cxa_finalize', '_start', 'deregister_tm_clones',
            'register_tm_clones', '__do_global_dtors_aux', 'frame_dummy',
            '__libc_csu_init', '__libc_start_main', '_fini', '__libc_csu_fini'
        }
        
        functions = []
        
        for f in retrieved_functions:
            if "<EXTERNAL>" in f[1] or f[0] in func_black_list:
                continue
            
            entry_point = f[3].getEntryPoint()
            demangled_name = demangled_names.get(entry_point, f[0])
            
            basicBlockModel = BasicBlockModel(currentProgram)
            basicBlocks = basicBlockModel.getCodeBlocksContaining(f[3].getBody(), monitor)
            
            list_block_bytes = []
            f_size = 0
            edges = []
            list_block_ins = []
            
            for block in basicBlocks:
                block_bytecode = ""
                min_address = block.getMinAddress()
                max_address = block.getMaxAddress().next()
                length = int(str(max_address), 16) - int(str(min_address), 16)
                f_size += length
                
                successor_iter = block.getDestinations(monitor)
                while successor_iter.hasNext():
                    successor = successor_iter.next()
                    edges.append((
                        int(str(min_address), 16),
                        int(str(successor.getDestinationAddress()), 16)
                    ))
                
                listing = currentProgram.getListing()
                instruction = listing.getInstructionAt(min_address)
                
                block_calls = []
                block_ins = []
                
                while instruction and instruction.getMinAddress() < max_address:
                    block_ins.append(str(instruction))
                    address = instruction.getAddress()
                    mnemonic = instruction.getMnemonicString()
                    
                    if mnemonic == "CALL":
                        references = instruction.getReferencesFrom()
                        if references:
                            target = references[0].getToAddress()
                            function = getFunctionAt(target)
                            
                            if function:
                                call_target_name = demangled_names.get(
                                    function.getEntryPoint(),
                                    function.getName()
                                )
                                block_calls.append((str(address), str(call_target_name)))
                    
                    instruction = instruction.getNext()
                
                instrs_bytes = getBytes(min_address, length)
                
                res = []
                for byte in instrs_bytes:
                    if byte >= 0:
                        b = byte
                    else:
                        b = byte + 256
                    b = hex(b).replace("0x", "")
                    if len(b) < 2:
                        res.append("0" + str(b))
                    else:
                        res.append(str(b))
                
                block_bytecode = ''.join(res)
                
                bytecode_addr_block = (
                    int(str(min_address), 16),
                    block_bytecode,
                    block_calls
                )
                ins_addr_block = (int(str(min_address), 16), block_ins)
                
                list_block_bytes.append(bytecode_addr_block)
                list_block_ins.append(ins_addr_block)
            
            functions.append({
                "address": int(str(f[2]), 16),
                "name": str(demangled_name),
                "flagname": str(f[1]),
                "signature": str(f[4]),
                "size": f_size,
                "args": f[5],
                "block_bytecode_dict": list_block_bytes,
                "edges": edges,
                "block_instructions_dict": list_block_ins
            })
        
        return functions


if __name__ == "__main__":
    args = getScriptArgs()
    if len(args) > 0:
        file_name = str(args[0])
        if not file_name.startswith("/"):
            file_name = "./" + file_name
    else:
        file_name = "./ghidra_output.json"
    
    print("ghidra_script.py: Starting analysis")
    start_time = time.time()
    
    functions = GhidraAnalyzer.analyze()
    
    with open(file_name, 'w') as f:
        json.dump(functions, f)
    
    elapsed_time = time.time() - start_time
    print("ghidra_script.py: Analysis completed in {:.2f}s".format(elapsed_time))
    print("ghidra_script.py: Extracted {} functions".format(len(functions)))
