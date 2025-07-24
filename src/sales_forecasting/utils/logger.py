import logging
import sys
from pathlib import Path

def get_logger(name: str, log_file: str = "inference.log") -> logging.Logger:
    """
    Returns a logger with standard formatting and both stream & file handlers.
    If the logger already exists, returns the existing one to avoid duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_path = Path(log_file)
    file_handler = logging.FileHandler(file_path, mode="a")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
