"""
Configuration management for the binary disassembly pipeline.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from loguru import logger


@dataclass
class Config:
    # Database settings
    db_host: str = "postgres"
    db_port: int = 5432
    db_name: str = "resafe"
    db_user: str = "jovyan"
    db_password: str = "secret"
    db_pool_min_conn: int = 2
    db_pool_max_conn: int = 15
    
    # Ghidra settings
    ghidra_path: str = "./analyzers/ghidra"
    ghidra_projects_path: str = "./analyzers/ghidra_projects"
    ghidra_script_path: str = ""  # Will be set relative to package location
    ghidra_timeout_minutes: int = 20
    
    # Processing settings
    num_producers: int = 4
    num_consumers: int = 2
    batch_size: int = 10  # Number of binaries per batch for producers
    insert_batch_size: int = 20  # Number of records per batch for DB inserts
    
    # Paths
    libc_signatures_path: str = "../extraction/data/libc_signatures.json"
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "../logs"
    
    def __post_init__(self):
        if not self.ghidra_script_path:
            package_dir = Path(__file__).parent
            self.ghidra_script_path = str(package_dir / "analyzers" / "ghidra_script.py")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        config = cls()
        
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
            
            for key, value in yaml_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    @classmethod
    def from_env(cls, base_config: Optional["Config"] = None) -> "Config":
        config = base_config or cls()
        
        env_mapping = {
            'PIPELINE_DB_HOST': 'db_host',
            'PIPELINE_DB_PORT': 'db_port',
            'PIPELINE_DB_NAME': 'db_name',
            'PIPELINE_DB_USER': 'db_user',
            'PIPELINE_DB_PASSWORD': 'db_password',
            'PIPELINE_DB_POOL_MIN': 'db_pool_min_conn',
            'PIPELINE_DB_POOL_MAX': 'db_pool_max_conn',
            'PIPELINE_GHIDRA_PATH': 'ghidra_path',
            'PIPELINE_GHIDRA_PROJECTS': 'ghidra_projects_path',
            'PIPELINE_GHIDRA_SCRIPT': 'ghidra_script_path',
            'PIPELINE_GHIDRA_TIMEOUT': 'ghidra_timeout_minutes',
            'PIPELINE_NUM_PRODUCERS': 'num_producers',
            'PIPELINE_NUM_CONSUMERS': 'num_consumers',
            'PIPELINE_BATCH_SIZE': 'batch_size',
            'PIPELINE_INSERT_BATCH_SIZE': 'insert_batch_size',
            'PIPELINE_LIBC_SIGNATURES': 'libc_signatures_path',
            'PIPELINE_LOG_LEVEL': 'log_level',
            'PIPELINE_LOG_DIR': 'log_dir',
        }
        
        for env_var, attr in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert to appropriate type
                current_value = getattr(config, attr)
                if isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes')
                setattr(config, attr, value)
        
        return config
    
    @classmethod
    def load(cls, yaml_path: Optional[str] = None) -> "Config":
        config = cls()
        package_dir = Path(__file__).parent
        path = str(package_dir / "config.yaml")
        
        if path.exists():
            logger.info(f"Loading configuration from: {path.absolute()}")
            config = cls.from_yaml(str(path))
        
        # Override with environment variables
        config = cls.from_env(config)
        
        return config
    
    def to_dict(self) -> dict:
        return {
            'db_host': self.db_host,
            'db_port': self.db_port,
            'db_name': self.db_name,
            'db_user': self.db_user,
            'db_password': '***',  # Don't expose password
            'db_pool_min_conn': self.db_pool_min_conn,
            'db_pool_max_conn': self.db_pool_max_conn,
            'ghidra_path': self.ghidra_path,
            'ghidra_projects_path': self.ghidra_projects_path,
            'ghidra_script_path': self.ghidra_script_path,
            'ghidra_timeout_minutes': self.ghidra_timeout_minutes,
            'num_producers': self.num_producers,
            'num_consumers': self.num_consumers,
            'batch_size': self.batch_size,
            'insert_batch_size': self.insert_batch_size,
            'libc_signatures_path': self.libc_signatures_path,
            'log_level': self.log_level,
            'log_dir': self.log_dir,
        }
    
    def __str__(self) -> str:
        lines = ["Pipeline Configuration:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


# Default configuration template for YAML file
CONFIG_TEMPLATE = """# Pipeline Configuration
# All values can be overridden via environment variables prefixed with PIPELINE_

# Database settings
db_host: postgres
db_port: 5432
db_name: resafe #TODO: reSIM
db_user: jovyan
db_password: secret
db_pool_min_conn: 2
db_pool_max_conn: 15

# Ghidra settings
ghidra_path: ./analyzers/ghidra
ghidra_projects_path: ./analyzers/ghidra_projects
ghidra_timeout_minutes: 20

# Processing settings
num_producers: 4
num_consumers: 2
batch_size: 10
insert_batch_size: 20

# Paths
libc_signatures_path: ../extraction/data/libc_signatures.json

# Logging
log_level: INFO
log_dir: ../logs
"""


def create_config_template(path: str = "config.yaml"):
    with open(path, 'w') as f:
        f.write(CONFIG_TEMPLATE)
    print(f"Configuration template created at: {path}")
