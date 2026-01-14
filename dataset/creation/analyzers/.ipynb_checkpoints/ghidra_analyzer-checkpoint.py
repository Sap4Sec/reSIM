import os
import shutil
import subprocess
import time
import json
import re
from loguru import logger
from ..config import Config

class GhidraAnalyzer:
    
    def __init__(self, config: Config):
        self.ghidra_path = config.ghidra_path
        self.projects_path = config.ghidra_projects_path
        self.ghidra_scripts_root_path = os.path.dirname(config.ghidra_script_path)
        self.ghidra_script = os.path.basename(config.ghidra_script_path)
        self.ghidra_timeout_minutes = config.ghidra_timeout_minutes
        
        os.makedirs(self.projects_path, exist_ok=True)
        
        self.filename = None
        self.entry_point_angr = None
        self.base_addr_zero = False
        self.project_name = None
        self.project_log = None
        self.result_json_file = None
        self.entry_point = None
        self.offset = None
        self.functions = []

    def analyze(self, filename, entry_point_angr, base_addr_zero=False):
        self.filename = filename
        self.entry_point_angr = entry_point_angr
        self.base_addr_zero = base_addr_zero
        
        # Unique project name to avoid conflicts
        self.project_name = '-'.join([str(os.getpid()), str(filename.split(os.sep)[-1])])
        
        log_path = os.path.join(self.projects_path, self.project_name + ".log")
        # logger.info(f"Opening project log in {log_path}")
        self.project_log = open(log_path, "w")
        
        self.result_json_file = self.project_name + ".json"
        
        try:
            self._start_ghidra()
            self.functions = self.get_functions()
        finally:
            if self.project_log:
                self.project_log.close()
            self.clean_ghidra_project()
            
        return self.functions

    def _start_ghidra(self):
        # logger.info("Starting ghidra analyzer")
        
        ghidra_exe = os.path.join(self.ghidra_path, "support", "analyzeHeadless")
        
        if self.base_addr_zero:
            process_statement = [
                ghidra_exe, self.projects_path, self.project_name, 
                "-import", self.filename, 
                "-loader", "ElfLoader", "-loader-imagebase", "0x0",
                "-scriptPath", self.ghidra_scripts_root_path, 
                "-postScript", self.ghidra_script, str(self.result_json_file)
            ]
        else:
            process_statement = [
                ghidra_exe, self.projects_path, self.project_name, 
                "-import", self.filename, 
                "-scriptPath", self.ghidra_scripts_root_path, 
                "-postScript", self.ghidra_script, str(self.result_json_file)
            ]
        
        timeout_seconds = self.ghidra_timeout_minutes * 60
        check_interval = 1
        
        try:
            ghidra_process = subprocess.Popen(process_statement, stdout=self.project_log, stderr=self.project_log)
            start_time = time.time()
            
            while True:
                elapsed_time = time.time() - start_time
                
                if ghidra_process.poll() is not None:  # Process completed
                    break
                
                if elapsed_time >= timeout_seconds:  
                    logger.warning(f"Ghidra/Angr timeout ({timeout_seconds}s) - Terminating")
                    ghidra_process.terminate()  # Graceful termination
                    time.sleep(5)  
                    
                    if ghidra_process.poll() is None:  # If still not terminated
                        ghidra_process.kill()  
                    
                    dict_file = "./" + self.result_json_file
                    if os.path.exists(dict_file):
                        os.remove(dict_file)
                    break
                    
                time.sleep(check_interval) 

        except Exception as e:
            logger.error(f"An error occurred in Ghidra execution: {e}")

    def clean_ghidra_project(self):
        files = [self.project_name + '.gpr', self.project_name + '.log']
        rep_folder = os.path.join(self.projects_path, self.project_name + '.rep')
        
        for file_name in files:
            file_path = os.path.join(self.projects_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                
        if os.path.exists(rep_folder):
            shutil.rmtree(rep_folder)
    
    def get_functions(self):
        try:
            # Result file is expected in current working directory as per script call
            dict_file = "./" + self.result_json_file
            
            if os.path.exists(dict_file):
                with open(dict_file, 'r') as file:
                    functions = json.load(file)
                os.remove(dict_file)
                
                if self.filename.endswith(".o"):
                    if functions:
                        min_address_function = min(functions, key=lambda x: x["address"])
                        self.entry_point = min_address_function["address"]
                        self.offset = self.entry_point - int(str(self.entry_point_angr), 16)
                    else:
                        self.entry_point = 0
                        self.offset = 0
                else: 
                    result = subprocess.run(['readelf', '-h', self.filename], capture_output=True, text=True)
        
                    if result.returncode != 0:
                        logger.error(f"Error executing readelf: {result.stderr}")
                        self.entry_point = None
                        self.offset = None
                    else:    
                        match = re.search(r'Entry point address:\s+([0-9a-fx]+)', result.stdout)
                        
                        if match:
                            self.entry_point = int(match.group(1), 16)
                            self.offset = self.entry_point - int(str(self.entry_point_angr), 16)
                        else:
                            self.entry_point = None
                            self.offset = None
                return functions
            else:
                logger.warning("Ghidra result JSON file not found")
                return []
        except Exception as e:
            logger.error(f"Exception in get_functions: {e}")
            self.entry_point = None
            self.offset = None
            return []
