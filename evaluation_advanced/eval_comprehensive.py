"""
Comprehensive Model Evaluation

Evaluates model on test set with multiple metrics:
- Syntax validity
- Compilation rate
- CodeBLEU
- Exact Match
- Token counts
- Code-to-response ratio
- Cyclomatic complexity
- Perplexity

Supports both base and fine-tuned model evaluation.
Supports ONNX model evaluation via --onnx flag (CPU only).
Supports recomputing metrics from existing results JSON via --results_json flag.
"""

import json
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from codebleu import calc_codebleu
import numpy as np

from utils.code_metrics import (
    extract_code_from_response,
    check_syntax,
    has_excess_explanation,
    calculate_code_to_response_ratio,
    calculate_cyclomatic_complexity,
)
from utils.logger import setup_evaluation_logger
from config.config import MAX_NEW_TOKENS, TEMPERATURE
from utils.io_utils import load_config


class ComprehensiveEvaluator:
    """Comprehensive evaluation for code generation models."""

    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        device: str = "auto",
        onnx: bool = False,
        tokenizer_only: bool = False,
    ):
        """
        Args:
            model_path: Path to model (base, fine-tuned, or ONNX directory)
            test_data_path: Path to test.json
            device: Device to run on (ignored when onnx=True, always CPU)
            onnx: If True, load model as ONNX via ORTModelForCausalLM
            tokenizer_only: If True, skip full model loading (used with --results_json)
        """
        self.logger = setup_evaluation_logger(output_dir="logs")
        self.onnx = onnx
        self.tokenizer_only = tokenizer_only

        self.logger.info("=" * 80)
        self.logger.info("LOADING MODEL FOR EVALUATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Model : {model_path}")

        if tokenizer_only:
            self.logger.info("Mode  : Tokenizer only (--results_json mode)")
        else:
            self.logger.info(f"Mode  : {'ONNX (CPU)' if onnx else 'PyTorch'}")

        self.model_path = model_path
        self.test_data_path = test_data_path
        self.config = load_config()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if tokenizer_only:
            # -----------------------------------------------------------------
            # Skip full model loading — only tokenizer needed for metrics
            # -----------------------------------------------------------------
            self.model = None
            self.device = "cpu"
            self.logger.info("Tokenizer loaded. Skipping full model load (--results_json mode).")

        elif onnx:
            # -----------------------------------------------------------------
            # ONNX path — use ORTModelForCausalLM with CPUExecutionProvider
            # device_map and torch_dtype are not applicable here
            # -----------------------------------------------------------------
            from optimum.onnxruntime import ORTModelForCausalLM

            self.model = ORTModelForCausalLM.from_pretrained(
                model_path,
                use_cache=True,
                provider="CPUExecutionProvider",
            )
            self.device = "cpu"
            self.logger.info("ONNX model loaded on CPU via ORTModelForCausalLM")

        else:
            # -----------------------------------------------------------------
            # Original PyTorch path — unchanged
            # -----------------------------------------------------------------
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device if device == "auto" else None
            )

            if device != "auto":
                self.model = self.model.to(device)

            self.device = next(self.model.parameters()).device
            self.logger.info(f"Model loaded on {self.device}")

        self.logger.info(f"Loading test data from {test_data_path}...")
        self.test_data = self._load_test_data()
        self.logger.info(f"Loaded {len(self.test_data)} test samples")

    def _load_test_data(self) -> List[Dict]:
        """Load test data from JSON."""
        with open(self.test_data_path, 'r') as f:
            return json.load(f)

    def evaluate(self, num_samples: int = 1000, batch_size: int = 8) -> Dict:
        """
        Run comprehensive evaluation.

        Args:
            num_samples: Number of samples to evaluate (max 1000)
            batch_size: Batch size for generation

        Returns:
            Dictionary with all metrics
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPREHENSIVE EVALUATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Samples   : {num_samples}")
        self.logger.info(f"Batch size: {batch_size}")

        samples = self.test_data[:num_samples]

        self.logger.info("Step 1: Generating responses...")
        start_time = time.time()
        results = self._generate_responses(samples, batch_size)
        generation_time = time.time() - start_time

        self.logger.info(f"Generated {len(results)} responses in {generation_time:.1f}s")
        self.logger.info(f"Avg time per sample: {generation_time / len(results):.2f}s")

        self.logger.info("Step 2: Computing metrics...")
        metrics = self._compute_all_metrics(results, samples)

        metrics['generation_time_sec'] = generation_time
        metrics['avg_time_per_sample'] = generation_time / len(results)
        metrics['samples_evaluated'] = len(results)

        return {
            "metadata": {
                "model_path": self.model_path,
                "model_mode": "onnx_cpu" if self.onnx else "pytorch",
                "test_data_path": self.test_data_path,
                "evaluation_date": datetime.now().isoformat(),
                "num_samples": len(results),
                "device": str(self.device)
            },
            "metrics": metrics,
            "detailed_results": results
        }

    def evaluate_from_results(self, existing: Dict) -> Dict:
        """
        Recompute metrics from an existing results JSON without regenerating responses.

        Preserves generation_time_sec, avg_time_per_sample, and samples_evaluated
        from the original metrics (if present). Overwrites all other metric fields.
        Preserves original metadata, only updating evaluation_date.

        Args:
            existing: Previously saved results dict containing 'detailed_results'

        Returns:
            Updated results dict with freshly computed metrics
        """
        self.logger.info("=" * 80)
        self.logger.info("RECOMPUTING METRICS FROM EXISTING RESULTS")
        self.logger.info("=" * 80)

        detailed_results = existing.get("detailed_results", [])
        if not detailed_results:
            raise ValueError("Provided JSON has no 'detailed_results' field or it is empty.")

        self.logger.info(f"Found {len(detailed_results)} existing responses. Skipping generation.")
        self.logger.info("Computing metrics...")

        metrics = self._compute_all_metrics(detailed_results, samples=None)

        # Preserve generation timing fields from original metrics if present
        original_metrics = existing.get("metrics", {})
        for field in ("generation_time_sec", "avg_time_per_sample", "samples_evaluated"):
            if field in original_metrics:
                metrics[field] = original_metrics[field]

        # Preserve original metadata, only refresh evaluation_date
        original_metadata = existing.get("metadata", {})
        updated_metadata = {**original_metadata, "evaluation_date": datetime.now().isoformat()}

        return {
            "metadata": updated_metadata,
            "metrics": metrics,
            "detailed_results": detailed_results,
        }

    def _generate_responses(self, samples: List[Dict], batch_size: int) -> List[Dict]:
        """Generate responses for all samples."""

        results = []

        prompts = []
        for sample in samples:
            instruction = sample['instruction']
            if sample.get('input', '').strip():
                instruction = f"{instruction}\n{sample['input']}"
            if self.model_path == "Qwen/Qwen2.5-Coder-0.5B-Instruct":
                modified_prompt = f"Task: {instruction}\n\nExample format:\ndef example():\n    return 'code only'\n\nNow write the code:"
            else:
                modified_prompt = f"Instruction: {instruction}\nOutput:\n"
            prompts.append(modified_prompt)

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_samples = samples[i:i + batch_size]

            self.logger.debug(f"Batch {i // batch_size + 1}/{(len(prompts) - 1) // batch_size + 1}")

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config['model_max_length']
            )

            # ------------------------------------------------------------------
            # For ONNX (CPU), do NOT move inputs to a CUDA device.
            # For PyTorch, move inputs to whatever device the model is on.
            # ------------------------------------------------------------------
            if not self.onnx:
                inputs = inputs.to(self.device)

            # ONNX model does not use torch.no_grad() context (it's not a
            # PyTorch model), but it is harmless to wrap it — keeping the
            # block unified here for simplicity.
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            for j, (output, sample) in enumerate(zip(outputs, batch_samples)):
                generated_ids = output[inputs['input_ids'][j].shape[0]:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                results.append({
                    "sample_id": i + j,
                    "instruction": batch_prompts[j],
                    "generated": generated_text,
                    "reference": sample['output']
                })

        return results

    def _compute_all_metrics(self, results: List[Dict], samples: Optional[List[Dict]]) -> Dict:
        """Compute all evaluation metrics."""

        metrics = {}

        # 1. Syntax validity
        self.logger.info("Computing syntax validity...")
        syntax_valid = [check_syntax(extract_code_from_response(r['generated']))['valid']
                        for r in results]
        metrics['syntax_validity_pct'] = (sum(syntax_valid) / len(syntax_valid)) * 100

        # 2. Compilation rate (same as syntax for Python)
        metrics['compilation_rate_pct'] = metrics['syntax_validity_pct']

        # 3. CodeBLEU (per-sample to avoid one bad sample zeroing the entire batch)
        self.logger.info("Computing CodeBLEU...")
        per_sample_scores = []
        for idx, r in enumerate(results):
            prediction = extract_code_from_response(r['generated'])
            reference = r['reference']
            if prediction and reference:
                try:
                    score = calc_codebleu(
                        [[reference]],
                        [prediction],
                        lang="python",
                        weights=(0.25, 0.25, 0.25, 0.25)
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Ignored score"
                        f"CodeBLEU failed for sample {idx}: {str(e)} | "
                        f"pred={prediction[:100]!r} | ref={reference[:100]!r}"
                    )
            else:
                self.logger.warning(
                    f"Ignored score"
                    f"[CodeBLEU Error] Sample {idx} failed due to empty strings | "
                    f"pred={prediction[:100]!r} | ref={reference[:100]!r}"
                )
            if score is not None:
                per_sample_scores.append(score['codebleu'])
        metrics['codebleu'] = float(np.mean(per_sample_scores))

        # 4. Exact Match
        self.logger.info("Computing Exact Match...")
        exact_matches = [
            extract_code_from_response(r['generated']).strip() == r['reference'].strip()
            for r in results
        ]
        metrics['exact_match_pct'] = (sum(exact_matches) / len(exact_matches)) * 100

        # 5. Token statistics
        self.logger.info("Computing token statistics...")
        token_counts = [len(self.tokenizer.encode(r['generated'])) for r in results]
        metrics['avg_tokens'] = np.mean(token_counts)
        metrics['median_tokens'] = np.median(token_counts)
        metrics['std_tokens'] = np.std(token_counts)

        # 6. Code-to-response ratio
        self.logger.info("Computing code ratio...")
        code_ratios = [
            calculate_code_to_response_ratio(r['generated'], extract_code_from_response(r['generated']))
            for r in results
        ]
        metrics['avg_code_ratio'] = np.mean(code_ratios)

        # 7. Verbosity analysis
        self.logger.info("Computing verbosity...")
        excess_explanation = [
            has_excess_explanation(r['generated'], extract_code_from_response(r['generated']))
            for r in results
        ]
        metrics['excess_explanation_pct'] = (sum(excess_explanation) / len(excess_explanation)) * 100

        # 8. Cyclomatic complexity
        self.logger.info("Computing cyclomatic complexity...")
        complexities = [
            calculate_cyclomatic_complexity(extract_code_from_response(r['generated']))
            for r in results
        ]
        complexities = [c for c in complexities if c is not None]
        metrics['avg_cyclomatic_complexity'] = np.mean(complexities) if complexities else 0.0

        # 9. Response length statistics
        response_lengths = [len(r['generated']) for r in results]
        metrics['avg_response_length_chars'] = np.mean(response_lengths)
        metrics['median_response_length_chars'] = np.median(response_lengths)

        self.logger.info("All metrics computed successfully")

        return metrics

    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to: {output_path}")


def main():
    """Main evaluation script."""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model. Required for normal evaluation. Optional when --results_json "
             "is provided (falls back to model_path stored in the JSON metadata)."
    )
    parser.add_argument("--test_data", type=str, default="data/datasets/test.json", help="Path to test data")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Load model as ONNX via ORTModelForCausalLM (CPU only). "
             "model_path should point to the ONNX model directory."
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default=None,
        help="Path to an existing results JSON containing 'detailed_results'. "
             "When provided, skips response generation and recomputes metrics only. "
             "model_path is optional in this mode (falls back to value in JSON metadata)."
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate args
    # ------------------------------------------------------------------
    if args.results_json is None and args.model_path is None:
        parser.error("--model_path is required when --results_json is not provided.")

    # ------------------------------------------------------------------
    # --results_json mode: recompute metrics only
    # ------------------------------------------------------------------
    if args.results_json:
        with open(args.results_json, 'r') as f:
            existing = json.load(f)

        # Resolve model_path: CLI arg takes priority, then fall back to JSON metadata
        model_path = args.model_path or existing.get("metadata", {}).get("model_path")
        if not model_path:
            parser.error(
                "Could not determine model_path. Provide --model_path or ensure the "
                "results JSON has a 'metadata.model_path' field."
            )

        evaluator = ComprehensiveEvaluator(
            model_path=model_path,
            test_data_path=args.test_data,
            tokenizer_only=True,
        )

        results = evaluator.evaluate_from_results(existing)
        evaluator.save_results(results, args.output)

    # ------------------------------------------------------------------
    # Normal mode: generate responses + compute metrics (original flow)
    # ------------------------------------------------------------------
    else:
        evaluator = ComprehensiveEvaluator(
            args.model_path,
            args.test_data,
            onnx=args.onnx,
        )
        results = evaluator.evaluate(num_samples=args.num_samples, batch_size=args.batch_size)
        evaluator.save_results(results, args.output)

    # ------------------------------------------------------------------
    # Summary log (shared by both modes)
    # ------------------------------------------------------------------
    logger = evaluator.logger
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    if args.results_json:
        logger.info("Mode             : Metrics-only (from existing results JSON)")
    else:
        logger.info(f"Mode             : {'ONNX (CPU)' if args.onnx else 'PyTorch'}")
    logger.info(f"Syntax Validity  : {results['metrics']['syntax_validity_pct']:.1f}%")
    logger.info(f"CodeBLEU         : {results['metrics']['codebleu']:.3f}")
    logger.info(f"Exact Match      : {results['metrics']['exact_match_pct']:.1f}%")
    logger.info(f"Avg Tokens       : {results['metrics']['avg_tokens']:.1f}")
    logger.info(f"Code Ratio       : {results['metrics']['avg_code_ratio']:.2f}")
    logger.info(f"Excess Explanation: {results['metrics']['excess_explanation_pct']:.1f}%")
    logger.info(f"Avg Complexity   : {results['metrics']['avg_cyclomatic_complexity']:.2f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()