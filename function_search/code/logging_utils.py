import os
from loguru import logger
from tqdm.auto import tqdm


class _TqdmSink:
    def write(self, message):
        msg = message.rstrip()
        if msg:
            tqdm.write(msg)
    def flush(self):
        pass


def initialize_logger(logger_path, name, context=None, console=False):

    logger.remove()

    log_path = os.path.join(logger_path, name)
    logger.add(
        log_path,
        enqueue=True,
        context=context,
        mode="w",
        encoding="utf-8",
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | GPU {extra[gpu]} | {message}",
        level="INFO",
    )

    if console:
        logger.add(_TqdmSink(), level="INFO", enqueue=True)

    return logger

def initialize_logger_no_gpu(logger_path, name, context=None, console=False):

    logger.remove()

    log_path = os.path.join(logger_path, name)
    logger.add(
        log_path,
        enqueue=True,
        context=context,
        mode="w",
        encoding="utf-8",
        backtrace=False,
        diagnose=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}",
        level="INFO",
    )

    if console:
        logger.add(_TqdmSink(), level="INFO", enqueue=True)

    return logger