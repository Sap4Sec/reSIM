"""Logging utilities for the function search pipeline."""

import os
from loguru import logger
from tqdm.auto import tqdm


class TqdmSink:
    """Loguru sink that writes through tqdm to avoid progress bar corruption."""
    
    def write(self, message):
        msg = message.rstrip()
        if msg:
            tqdm.write(msg)
    
    def flush(self) -> None:
        pass


def initialize_logger(log_dir, filename, gpu_id, context=None, console=False):
    """Initialize loguru logger with file and optional console output.
    
    Args:
        log_dir: Directory for log files
        filename: Log filename
        gpu_id: GPU ID for log format (optional)
        context: Multiprocessing context (optional)
        console: Whether to also log to console
        
    Returns:
        Configured logger instance
    """
    logger.remove()
    
    log_path = os.path.join(log_dir, filename)
    
    if gpu_id is not None:
        log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | GPU {extra[gpu]} | {message}"
    else:
        log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}"
    
    logger.add(
        log_path,
        enqueue=True,
        context=context,
        mode="w",
        encoding="utf-8",
        backtrace=False,
        diagnose=False,
        format=log_format,
        level="INFO",
    )
    
    if console:
        logger.add(TqdmSink(), level="INFO", enqueue=True)
    
    if gpu_id is not None:
        return logger.bind(gpu=gpu_id)
    return logger
