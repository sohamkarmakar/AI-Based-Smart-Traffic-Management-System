import os
import logging
from pathlib import Path

def setup_logger(name: str = "traffic_management", log_file: str = "system.log") -> logging.Logger:
    """Sets up a logger with handlers for console and file output."""
    logger = logging.getLogger(name)
    
    # If logger already has handlers, return it to avoid duplicates
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # File Handler
    file_path = log_dir / log_file
    file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    return logger

# Default logger instance
logger = setup_logger()
