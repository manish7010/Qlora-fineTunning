"""
QLoRA Fine-tuning Script for Qwen2.5-Coder-0.5B

Features:
- 4-bit quantization for memory efficiency
- LoRA adapters for parameter-efficient fine-tuning
- W&B integration for experiment tracking
- Custom callbacks for validation metrics
- Checkpointing for resumability
"""

import os
import json
import torch
import wandb
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import numpy as np

from callbacks import CustomValidationCallback, GPUMemoryCallback
from utils.logger import setup_training_logger
from utils.io_utils import load_config

logger = setup_training_logger(output_dir="logs")


def setup_wandb(config):
    """Initialize Weights & Biases."""
    wandb.init(
        project=config['wandb_project'],
        name=config['wandb_run_name'],
        config=config
    )


def load_and_prepare_model(config):
    """Load model with 4-bit quantization and apply LoRA."""

    logger.info("=" * 80)
    logger.info("MODEL LOADING & PREPARATION")
    logger.info("=" * 80)

    # Quantization config
    logger.info("Setting up 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['load_in_4bit'],
        bnb_4bit_quant_type=config['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=torch.bfloat16 if config['bnb_4bit_compute_dtype'] == "bfloat16" else torch.float16,
        bnb_4bit_use_double_quant=True  # Nested quantization for extra memory savings
    )

    # Load model
    logger.info(f"Loading model: {config['model_name']}...")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare for k-bit training
    logger.info("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    logger.info("Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        target_modules=config['lora_target_modules'],
        lora_dropout=config['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info("Model statistics:")
    logger.info(f"  Total parameters:     {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable %%:          {100 * trainable_params / total_params:.2f}%%")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logger.info("Tokenizer loaded successfully")

    return model, tokenizer


def load_datasets(config, tokenizer):
    """Load and preprocess training and validation datasets.

    Returns:
        train_dataset: Tokenized training dataset.
        val_dataset_tokenized: Tokenized validation dataset (for eval_loss).
        val_dataset_raw: Raw (un-tokenized) validation dataset (for CustomValidationCallback).
    """

    logger.info("=" * 80)
    logger.info("DATASET LOADING")
    logger.info("=" * 80)

    # Load datasets
    logger.info(f"Loading training data from {config['dataset_train']}...")
    train_dataset = load_dataset('json', data_files=config['dataset_train'], split='train')

    logger.info(f"Loading validation data from {config['dataset_val']}...")
    val_dataset_raw = load_dataset('json', data_files=config['dataset_val'], split='train')

    logger.info(f"Train samples:      {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset_raw)}")

    # Preprocessing function
    def preprocess_function(examples):
        """Format data for instruction fine-tuning."""

        formatted_texts = []
        for instruction, input_text, output in zip(
            examples['instruction'],
            examples['input'],
            examples['output']
        ):
            # Create prompt
            if input_text and input_text.strip():
                prompt = f"{instruction}\n{input_text}"
            else:
                prompt = instruction

            # Format: instruction + output (model learns to generate output)
            # Wrap output code in python block and append EOS token
            text = f"Instruction: {prompt}\nOutput:\n```python\n{output}\n```{tokenizer.eos_token}"
            formatted_texts.append(text)

        # Tokenize
        tokenized = tokenizer(
            formatted_texts,
            truncation=True,
            max_length=config['model_max_length'],
            padding=False,  # Dynamic padding in data collator
            return_tensors=None
        )

        # Ensure EOS at the end
        for i, seq in enumerate(tokenized['input_ids']):
            if seq[-1] != tokenizer.eos_token_id:
                seq[-1] = tokenizer.eos_token_id

        return tokenized

    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Processing training data"
    )

    val_dataset_tokenized = val_dataset_raw.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset_raw.column_names,
        desc="Processing validation data"
    )

    logger.info("Preprocessing complete")

    return train_dataset, val_dataset_tokenized, val_dataset_raw


def setup_trainer(model, tokenizer, train_dataset, val_dataset_tokenized, val_dataset_raw, config):
    """Setup Trainer with all configurations."""

    logger.info("=" * 80)
    logger.info("TRAINER SETUP")
    logger.info("=" * 80)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_ratio=config['warmup_ratio'],
        max_grad_norm=config['max_grad_norm'],
        lr_scheduler_type=config['lr_scheduler_type'],
        optim=config['optim'],
        logging_steps=config['logging_steps'],
        eval_strategy=config['eval_strategy'],
        eval_steps=config['eval_steps'],
        save_strategy=config['save_strategy'],
        save_steps=config['save_steps'],
        save_total_limit=config['save_total_limit'],
        load_best_model_at_end=config['load_best_model_at_end'],
        metric_for_best_model=config['metric_for_best_model'],
        greater_is_better=config['greater_is_better'],
        report_to=config['report_to'],
        run_name=config['run_name'],
        fp16=config['fp16'],
        bf16=config['bf16'],
        gradient_checkpointing=config['gradient_checkpointing'],
        seed=config['seed'],
        dataloader_num_workers=config['dataloader_num_workers'],
        remove_unused_columns=config['remove_unused_columns'],
        logging_first_step=True,
        disable_tqdm=False
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Callbacks
    validation_callback = CustomValidationCallback(tokenizer, val_dataset_raw)
    gpu_memory_callback = GPUMemoryCallback()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset_tokenized,
        data_collator=data_collator,
        callbacks=[validation_callback, gpu_memory_callback]
    )

    effective_batch = config['per_device_train_batch_size'] * config['gradient_accumulation_steps']
    total_steps = len(train_dataset) // effective_batch * config['num_train_epochs']

    logger.info("Trainer configured")
    logger.info(f"  Effective batch size:  {effective_batch}")
    logger.info(f"  Total training steps:  {total_steps}")

    return trainer


def main():
    """Main training pipeline."""

    logger.info("=" * 80)
    logger.info("QWEN 0.5B PYTHON CODE FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load config
    config = load_config()

    # Setup W&B
    setup_wandb(config)

    # Load model and tokenizer
    model, tokenizer = load_and_prepare_model(config)

    # Load datasets — raw val dataset kept separately for generation-based eval
    train_dataset, val_dataset_tokenized, val_dataset_raw = load_datasets(config, tokenizer)

    # Setup trainer
    trainer = setup_trainer(model, tokenizer, train_dataset, val_dataset_tokenized, val_dataset_raw, config)

    # Start training
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    try:
        trainer.train()

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)

        # Save final model
        logger.info("Saving final model...")
        final_model_path = os.path.join(config['output_dir'], "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

        # Save best model (already saved by trainer)
        best_model_path = os.path.join(config['output_dir'], "best_model")
        if os.path.exists(best_model_path):
            logger.info(f"Best model available at: {best_model_path}")

        # Save training stats
        training_stats = {
            "total_epochs": config['num_train_epochs'],
            "final_train_loss": trainer.state.log_history[-1].get('loss', 'N/A'),
            "best_eval_loss": trainer.state.best_metric,
            "total_steps": trainer.state.global_step,
            "training_time": (
                str(datetime.now() - datetime.fromisoformat(trainer.state.log_history[0]['time']))
                if 'time' in trainer.state.log_history[0] else 'N/A'
            )
        }

        stats_path = os.path.join(config['output_dir'], "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(training_stats, f, indent=2)

        logger.info(f"Training stats saved to: {stats_path}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        logger.info(
            f"Checkpoints saved. Resume with: "
            f"trainer.train(resume_from_checkpoint='{config['output_dir']}/checkpoint-XXX')"
        )

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

    finally:
        wandb.finish()

    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()