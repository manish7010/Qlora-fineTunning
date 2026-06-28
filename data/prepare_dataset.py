"""
Dataset Preparation Script

Loads python_code_instructions_18k_alpaca and splits into:
- Train: 16,000 samples (89%)
- Validation: 1,000 samples (5.5%)
- Test: 1,000 samples (5.5%)

No overlap between splits.
"""

import json
from datasets import load_dataset
from pathlib import Path

from config.config import DATASET_NAME, DATASET_PATH
from utils.logger import setup_training_logger

logger = setup_training_logger(output_dir="logs")

def prepare_dataset(output_dir=DATASET_PATH):
    """Prepare train/val/test splits."""
    
    logger.info("="*80)
    logger.info("DATASET PREPARATION")
    logger.info("="*80)
    
    # Load full dataset
    logger.info("\n1. Loading dataset from Hugging Face...")
    dataset = load_dataset(DATASET_NAME, split="train")
    logger.info(f"   ✓ Loaded {len(dataset)} samples")
    
    # Shuffle with fixed seed for reproducibility
    logger.info("\n2. Shuffling dataset (seed=42)...")
    dataset = dataset.shuffle(seed=42)
    
    # Split indices
    total_samples = len(dataset)
    train_size = 16000
    val_size = 1000
    test_size = 1000
    
    logger.info(f"\n3. Splitting dataset:")
    logger.info(f"   • Train: {train_size} samples ({train_size/total_samples*100:.1f}%)")
    logger.info(f"   • Validation: {val_size} samples ({val_size/total_samples*100:.1f}%)")
    logger.info(f"   • Test: {test_size} samples ({test_size/total_samples*100:.1f}%)")
    
    # Create splits
    train_dataset = dataset.select(range(0, train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, train_size + val_size + test_size))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON files
    logger.info(f"\n4. Saving splits to {output_dir}/...")
    
    def save_split(dataset_split, filename):
        """Save dataset split to JSON."""
        data = []
        for sample in dataset_split:
            data.append({
                "instruction": sample["instruction"],
                "input": sample.get("input", ""),
                "output": sample["output"]
            })
        
        filepath = output_path / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"   ✓ {filename}: {len(data)} samples")
        return len(data)
    
    train_count = save_split(train_dataset, "train.json")
    val_count = save_split(val_dataset, "validation.json")
    test_count = save_split(test_dataset, "test.json")
    
    # Verification
    logger.info(f"\n5. Verification:")
    logger.info(f"   • Total samples: {train_count + val_count + test_count}")
    logger.info(f"   • No overlap: {len(set(range(train_size + val_size + test_size))) == train_count + val_count + test_count}")
    
    # logger.info sample
    logger.info(f"\n6. Sample from training set:")
    sample = train_dataset[0]
    logger.info(f"   Instruction: {sample['instruction'][:100]}...")
    logger.info(f"   Output: {sample['output'][:100]}...")
    
    logger.info("\n" + "="*80)
    logger.info("✓ Dataset preparation complete!")
    logger.info("="*80)
    
    return {
        "train_samples": train_count,
        "val_samples": val_count,
        "test_samples": test_count,
        "output_dir": str(output_path)
    }


if __name__ == "__main__":
    stats = prepare_dataset()
    
    # Save stats
    with open(f"{DATASET_PATH}/dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\n📊 Stats saved to {DATASET_PATH}/dataset_stats.json")
