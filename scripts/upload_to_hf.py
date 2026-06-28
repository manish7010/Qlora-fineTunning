"""
Upload fine-tuned model to HuggingFace Hub.

This allows:
- Sharing model publicly or privately
- Automatic downloads in CI/CD
- Version control for models
- No need to store large files in Git

Usage:
    python scripts/upload_to_hf.py \
        --model_path ./training/outputs/final_model \
        --repo_name username/qwen-python-finetuned \
        --private
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

from utils.logger import get_logger

logger = get_logger(__name__)


def create_model_card(
    model_path: str,
    base_model: str,
    dataset: str,
    training_stats: dict = None
) -> str:
    """
    Create a model card (README.md) for HuggingFace.
    
    Args:
        model_path: Path to fine-tuned model
        base_model: Base model name
        dataset: Training dataset name
        training_stats: Optional training statistics
        
    Returns:
        Model card content
    """
    
    card = f"""---
language: code
tags:
- code-generation
- python
- fine-tuned
- qlora
base_model: {base_model}
datasets:
- {dataset}
license: mit
---

# Qwen2.5-Coder-0.5B Python Fine-tuned

Fine-tuned version of [{base_model}](https://huggingface.co/{base_model}) for Python code generation.

## Model Details

- **Base Model**: {base_model}
- **Fine-tuning Method**: QLoRA (4-bit quantization + LoRA adapters)
- **Dataset**: {dataset}
- **Task**: Python code generation from natural language instructions

## Training Details

"""
    
    if training_stats:
        card += f"""
- **Training Samples**: {training_stats.get('train_samples', 'N/A')}
- **Validation Samples**: {training_stats.get('val_samples', 'N/A')}
- **Epochs**: {training_stats.get('total_epochs', 'N/A')}
- **Training Time**: {training_stats.get('training_time', 'N/A')}
- **Final Loss**: {training_stats.get('final_train_loss', 'N/A')}

## Performance

- **Syntax Validity**: {training_stats.get('syntax_validity', 'N/A')}%
- **Pass@1**: {training_stats.get('pass_at_1', 'N/A')}%
- **Verbosity Reduction**: {training_stats.get('verbosity_reduction', 'N/A')}%
"""
    
    card += """
## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("KpRT/qwen-python-finetuned")
tokenizer = AutoTokenizer.from_pretrained("KpRT/qwen-python-finetuned")

prompt = "Write a function to reverse a string"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(code)
```

## Citation

If you use this model, please cite:

```bibtex
@misc{qwen-python-finetuned,
  author = {K R T},
  title = {Qwen2.5-Coder Python Fine-tuned},
  year = {2026},
  publisher = {HuggingFace},
  url = {https://huggingface.co/KpRT/qwen-python-finetuned}
}
```
"""
    
    return card


def upload_model_to_hf(
    model_path: str,
    repo_name: str,
    private: bool = True,
    token: str = None
):
    """
    Upload model to HuggingFace Hub.
    
    Args:
        model_path: Local path to fine-tuned model
        repo_name: HuggingFace repo name (username/model-name)
        private: Whether to make repo private
        token: HuggingFace API token (or use HF_TOKEN env var)
    """
    
    logger.info("="*80)
    logger.info("UPLOADING MODEL TO HUGGINGFACE HUB")
    logger.info("="*80)
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Initialize API
    api = HfApi(token=token)
    
    # Create repository
    logger.info(f"Creating repository: {repo_name}")
    try:
        create_repo(
            repo_id=repo_name,
            private=private,
            token=token,
            exist_ok=True,
            repo_type="model"
        )
        logger.info(f"✓ Repository created/verified: {repo_name}")
    except Exception as e:
        logger.error(f"Failed to create repository: {str(e)}")
        raise
    
    # Load training stats if available
    stats_file = model_path.parent / "training_stats_modified.json"
    training_stats = None
    
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            training_stats = json.load(f)
        logger.info("✓ Loaded training statistics")
    
    # Create model card
    logger.info("Creating model card...")
    model_card = create_model_card(
        model_path=str(model_path),
        base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        dataset="iamtarun/python_code_instructions_18k_alpaca",
        training_stats=training_stats
    )
    
    # Save model card
    card_path = model_path / "README.md"
    with open(card_path, 'w') as f:
        f.write(model_card)
    logger.info("✓ Model card created")
    
    # Upload folder
    logger.info(f"Uploading model from {model_path}...")
    try:
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            token=token,
            commit_message="Upload fine-tuned model"
        )
        logger.info("✓ Model uploaded successfully")
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise
    
    # Provide link
    visibility = "private" if private else "public"
    logger.info("="*80)
    logger.info("✓ UPLOAD COMPLETE")
    logger.info("="*80)
    logger.info(f"Repository: https://huggingface.co/{repo_name} ({visibility})")
    logger.info(f"Download with: AutoModelForCausalLM.from_pretrained('{repo_name}')")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="HuggingFace repo name (username/model-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    upload_model_to_hf(
        model_path=args.model_path,
        repo_name=args.repo_name,
        private=args.private,
        token=args.token
    )


if __name__ == "__main__":
    main()
