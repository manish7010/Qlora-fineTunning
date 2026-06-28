"""
Custom Training Callbacks

Provides additional monitoring during training:
- Validation metrics (syntax validity, token count, code ratio)
- GPU memory tracking
- Sample generation logging
"""

import torch
import wandb
from transformers import TrainerCallback
import numpy as np

from utils.code_metrics import extract_code_from_response, check_syntax
from utils.logger import setup_training_logger
from config.config import MAX_NEW_TOKENS, TEMPERATURE
from utils.io_utils import load_config

logger = setup_training_logger(output_dir="logs")


class CustomValidationCallback(TrainerCallback):
    """
    Custom callback to compute generation-based validation metrics during training.

    Accepts the raw (un-tokenized) validation dataset so that prompts are built
    from the instruction/input fields only — the output field is never included
    in the prompt, avoiding data leakage into generation.

    Metrics computed:
    - Syntax validity %
    - Average token count
    - Code-to-response ratio
    - Sample generations (logged to W&B)

    Generation strategy:
    - Attempts batched inference for GPU throughput (batch_size samples at once).
    - Falls back to sequential generation per sample on any batch-level error.
    """

    def __init__(self, tokenizer, val_dataset_raw, num_samples=50, batch_size=8):
        """
        Args:
            tokenizer: Tokenizer for encoding prompts and decoding generations.
            val_dataset_raw: Raw (un-tokenized) validation dataset with
                             'instruction', 'input', and 'output' fields.
            num_samples: Number of samples to evaluate per epoch.
            batch_size: Number of samples to generate in one forward pass.
                        Tune down if you hit OOM; the sequential fallback will
                        automatically handle any batch-level failures.
        """
        self.tokenizer = tokenizer
        self.val_dataset_raw = val_dataset_raw
        self.num_samples = min(num_samples, len(val_dataset_raw))
        self.batch_size = batch_size
        # Fix the sample indices once so each epoch evaluates the same examples
        self.eval_indices = np.random.choice(len(val_dataset_raw), self.num_samples, replace=False)
        # Load config
        self.config = load_config()

    def _build_prompt(self, sample) -> str:
        """Construct the prompt-only string from a raw dataset sample."""
        instruction = sample['instruction']
        input_text = sample.get('input', '')
        if input_text and input_text.strip():
            return f"Instruction: {instruction}\n{input_text}\nOutput:\n"
        return f"Instruction: {instruction}\nOutput:\n"

    def _generate_batch(self, model, prompts: list[str]) -> list[str]:
        """
        Tokenize and generate for a batch of prompts.

        Left-pads the batch so all sequences share the same length, then
        decodes only the newly generated tokens for each item.

        Args:
            model: The model currently being evaluated.
            prompts: List of prompt strings to generate from.

        Returns:
            List of decoded generated strings, one per prompt.

        Raises:
            Re-raises any exception so the caller can fall back to sequential.
        """
        # Left-pad so the model attends to real tokens from the right
        self.tokenizer.padding_side = "left"

        encodings = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['model_max_length']
        )
        input_ids      = encodings.input_ids.to(model.device)
        attention_mask = encodings.attention_mask.to(model.device)
        prompt_lengths = attention_mask.sum(dim=1)  # real token count per item

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Slice off the prompt tokens for each sequence individually
        generated_texts = []
        for i, out in enumerate(outputs):
            new_tokens = out[prompt_lengths[i]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def _generate_sequential(self, model, prompt: str) -> str:
        """
        Fallback: generate a single sample without batching.

        Args:
            model: The model currently being evaluated.
            prompt: A single prompt string.

        Returns:
            Decoded generated string (new tokens only).
        """
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['model_max_length']
        ).input_ids.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=self.tokenizer.pad_token_id
            )

        return self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )

    def _process_results(self, prompts, generated_texts, start_index):
        """
        Compute per-sample metrics from a list of (prompt, generated) pairs.

        Args:
            prompts: Original prompt strings for this chunk.
            generated_texts: Corresponding generated strings.
            start_index: Offset of the first item in the overall eval loop
                         (used to decide which items become W&B table rows).

        Returns:
            Tuple of:
                syntax_hits   – number of syntactically valid outputs
                token_counts  – list of int token lengths
                code_ratios   – list of float code/response ratios
                sample_outputs – list of dicts for the W&B table (may be empty)
        """
        syntax_hits    = 0
        token_counts   = []
        code_ratios    = []
        sample_outputs = []

        for local_i, (prompt, generated_text) in enumerate(zip(prompts, generated_texts)):
            global_i = start_index + local_i

            code     = extract_code_from_response(generated_text)
            is_valid = check_syntax(code)['valid']
            syntax_hits += is_valid

            num_tokens = len(self.tokenizer.encode(generated_text))
            token_counts.append(num_tokens)

            code_ratio = len(code) / len(generated_text) if generated_text else 0
            code_ratios.append(code_ratio)

            if global_i < 5:
                sample_outputs.append({
                    "sample_id": global_i,
                    "prompt":    prompt[:200],
                    "generated": generated_text[:200],
                    "syntax_valid": is_valid,
                    "tokens":    num_tokens
                })

        return syntax_hits, token_counts, code_ratios, sample_outputs

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Called after each evaluation step (end of each epoch)."""

        if state.epoch is None:
            return

        logger.info("=" * 60)
        logger.info(f"VALIDATION METRICS - Epoch {int(state.epoch)}")
        logger.info("=" * 60)

        syntax_valid_count = 0
        token_counts       = []
        code_ratios        = []
        all_sample_outputs = []

        model.eval()

        # ------------------------------------------------------------------ #
        # Iterate over the fixed eval indices in chunks of self.batch_size.   #
        # For each chunk we try batched generation first; on failure we fall   #
        # back to sequential generation sample-by-sample.                     #
        # ------------------------------------------------------------------ #
        index_batches = [
            self.eval_indices[i : i + self.batch_size]
            for i in range(0, self.num_samples, self.batch_size)
        ]

        for batch_start, batch_indices in enumerate(index_batches):
            global_start = batch_start * self.batch_size
            samples  = [self.val_dataset_raw[int(idx)] for idx in batch_indices]
            prompts  = [self._build_prompt(s) for s in samples]

            # ---- Attempt batch generation --------------------------------- #
            try:
                generated_texts = self._generate_batch(model, prompts)
                logger.debug(
                    f"Batch [{global_start}:{global_start + len(prompts)}] "
                    "generated successfully."
                )

            except Exception as batch_err:
                logger.warning(
                    f"Batch generation failed for indices "
                    f"{global_start}:{global_start + len(prompts)} "
                    f"({batch_err}). Falling back to sequential."
                )

                # ---- Sequential fallback ---------------------------------- #
                generated_texts = []
                for local_i, prompt in enumerate(prompts):
                    global_i = global_start + local_i
                    try:
                        text = self._generate_sequential(model, prompt)
                        generated_texts.append(text)
                    except Exception as seq_err:
                        logger.warning(
                            f"Sequential generation failed for sample "
                            f"{global_i}: {seq_err}. Skipping."
                        )
                        generated_texts.append("")  # sentinel; skipped below

            # ---- Compute metrics for this chunk --------------------------- #
            # Filter out empty sentinel strings from failed sequential calls
            valid_pairs = [
                (p, g) for p, g in zip(prompts, generated_texts) if g
            ]
            if not valid_pairs:
                continue

            valid_prompts, valid_generated = zip(*valid_pairs)

            hits, t_counts, c_ratios, s_outputs = self._process_results(
                list(valid_prompts),
                list(valid_generated),
                global_start
            )
            syntax_valid_count += hits
            token_counts.extend(t_counts)
            code_ratios.extend(c_ratios)
            all_sample_outputs.extend(s_outputs)

        # ------------------------------------------------------------------ #
        # Aggregate and log                                                   #
        # ------------------------------------------------------------------ #
        evaluated            = len(token_counts)
        syntax_validity_pct  = (syntax_valid_count / evaluated * 100) if evaluated else 0
        avg_token_count      = np.mean(token_counts) if token_counts else 0
        avg_code_ratio       = np.mean(code_ratios)  if code_ratios  else 0

        logger.info(f"Syntax Validity: {syntax_validity_pct:.1f}% ({syntax_valid_count}/{evaluated})")
        logger.info(f"Avg Token Count: {avg_token_count:.1f}")
        logger.info(f"Avg Code Ratio:  {avg_code_ratio:.2f}")

        wandb.log({
            "val/syntax_validity": syntax_validity_pct,
            "val/avg_token_count": avg_token_count,
            "val/avg_code_ratio":  avg_code_ratio,
            "epoch": state.epoch
        })

        if all_sample_outputs:
            wandb.log({
                "val/sample_generations": wandb.Table(
                    columns=["sample_id", "prompt", "generated", "syntax_valid", "tokens"],
                    data=[
                        [s["sample_id"], s["prompt"], s["generated"],
                         s["syntax_valid"], s["tokens"]]
                        for s in all_sample_outputs
                    ]
                ),
                "epoch": state.epoch
            })

        logger.info("=" * 60)

        model.train()


class GPUMemoryCallback(TrainerCallback):
    """Track and log GPU memory usage at each logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log GPU memory usage alongside training metrics."""
        if not torch.cuda.is_available():
            return

        memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
        memory_reserved  = torch.cuda.memory_reserved()  / 1024 ** 3  # GB

        logger.info(
            f"GPU memory — allocated: {memory_allocated:.2f} GB, "
            f"reserved: {memory_reserved:.2f} GB"
        )

        wandb.log({
            "gpu/memory_allocated_gb": memory_allocated,
            "gpu/memory_reserved_gb":  memory_reserved,
            "step": state.global_step
        })