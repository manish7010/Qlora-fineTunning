"""
Shared utilities for model and dataset loading.

All scripts use these functions to ensure consistent initialization
of models and data across experiments and evaluations.
"""

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Dict
import psutil

from utils.logger import setup_evaluation_logger
from config.config import MODEL_NAME, NUM_SAMPLES, DATASET_NAME

logger = setup_evaluation_logger(output_dir="logs")


def load_model() -> pipeline:
    """Load model, tokenizer and return a text-generation pipeline."""
    logger.info(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    logger.info(f"Model loaded on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    return pipe


def load_dataset_samples() -> tuple[list, list]:
    """Load and shuffle dataset, return samples and prepared instructions."""
    logger.info("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split="train")
    samples = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))

    instructions = []
    for sample in samples:
        instruction = sample["instruction"]
        if sample.get("input"):
            instruction = f"{instruction}\n{sample['input']}"
        prompt = f"Instruction: {instruction}\nOutput:\n"
        instructions.append(prompt)
    logger.info(f"Loaded {len(samples)} samples")
    return samples, instructions

def load_config(config_path="training/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def measure_memory_usage() -> Dict[str, float]:
    """
    Measure current memory usage.
    
    Returns:
        Dictionary with memory statistics (MB)
    """
    memory_stats = {}
    
    # System RAM
    ram = psutil.virtual_memory()
    memory_stats["system_ram_used_mb"] = ram.used / (1024**2)
    memory_stats["system_ram_available_mb"] = ram.available / (1024**2)
    memory_stats["system_ram_percent"] = ram.percent
    
    # GPU memory (if available)
    if torch.cuda.is_available():
        memory_stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024**2)
        memory_stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024**2)
        memory_stats["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    
    return memory_stats

