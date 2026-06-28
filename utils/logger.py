"""
Centralized logging configuration for the entire project.

Usage:
    from utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Training started")
    logger.error("Something went wrong")
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str = None,
    console: bool = True
) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_training_logger(output_dir: str) -> logging.Logger:
    """
    Setup logger specifically for training with file output.
    
    Args:
        output_dir: Directory to save training logs
        
    Returns:
        Configured logger
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(output_dir) / f"training_{timestamp}.log"
    
    return get_logger(
        "training",
        level=logging.INFO,
        log_file=str(log_file),
        console=True
    )


def setup_evaluation_logger(output_dir: str = None) -> logging.Logger:
    """
    Setup logger for evaluation scripts.
    
    Args:
        output_dir: Optional directory for log files
        
    Returns:
        Configured logger
    """
    if output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(output_dir) / f"evaluation_{timestamp}.log"
    else:
        log_file = None
    
    return get_logger(
        "evaluation",
        level=logging.INFO,
        log_file=str(log_file) if log_file else None,
        console=True
    )
