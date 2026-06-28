"""
Shared utilities package.

All common functions used across evaluation, training, and inference.
"""

from utils.logger import get_logger, setup_training_logger, setup_evaluation_logger
from utils.code_metrics import (
    extract_code_from_response,
    check_syntax,
    calculate_code_metrics,
    aggregate_metrics
)
from utils.io_utils import (
    load_model,
    load_dataset_samples,
    load_config,
    measure_memory_usage
)

__all__ = [
    # Logging
    "get_logger",
    "setup_training_logger",
    "setup_evaluation_logger",
    
    # Code metrics
    "extract_code_from_response",
    "check_syntax",
    "calculate_code_metrics",
    "aggregate_metrics",

    # IO Utils
    "load_model",
    "load_dataset_samples",
    "load_config",
    "measure_memory_usage"
]
