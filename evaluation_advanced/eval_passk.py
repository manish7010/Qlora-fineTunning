"""
Pass@k Evaluation

Evaluates functional correctness using pass@k metric.
Uses HumanEval dataset (subset of 50 problems for faster evaluation).

Pass@k: Probability that at least one of k generated solutions passes unit tests.

Supports ONNX model evaluation via --onnx flag (CPU only).
"""

import json
import sys
import io
import signal
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from collections import defaultdict

from utils.code_metrics import extract_code_from_response
from utils.logger import setup_evaluation_logger
from config.config import MAX_NEW_TOKENS, TEMPERATURE
from utils.io_utils import load_config


def execute_code(code: str, test_code: str, entry_point: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Execute code with test cases.

    For HumanEval-style tests the test string only *defines* check(candidate) but
    never calls it.  We append the call ourselves using entry_point so the
    assertions actually run.

    Args:
        code: Generated code
        test_code: Test code — either bare assertions (synthetic) or a
                   check(candidate) definition (HumanEval)
        entry_point: Name of the function under test (e.g. "has_close_elements")
        timeout: Execution timeout in seconds

    Returns:
        (passed, error_message)
    """
    def _handler(signum, frame):
        raise TimeoutError()

    try:
        if "def check(" in test_code:
            full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})"
        else:
            full_code = f"{code}\n\n{test_code}"

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout)
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                namespace = {}
                exec(full_code, namespace)
        finally:
            signal.alarm(0)  # always cancel the alarm, even if exec raises

        return True, None

    except TimeoutError:
        return False, f"Execution timed out after {timeout}s"
    except AssertionError as e:
        return False, f"Assertion failed: {str(e)}"
    except Exception as e:
        return False, f"Execution error: {type(e).__name__}: {str(e)}"


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Estimate pass@k using the formula from the Codex paper.

    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: k in pass@k

    Returns:
        Estimated pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


class PassAtKEvaluator:
    """Evaluator for pass@k metric."""

    def __init__(self, model_path: str, device: str = "auto", onnx: bool = False):
        """
        Args:
            model_path: Path to model (or ONNX directory when onnx=True)
            device: Device to use (ignored when onnx=True, always CPU)
            onnx: If True, load model as ONNX via ORTModelForCausalLM
        """
        self.logger = setup_evaluation_logger(output_dir="logs")
        self.onnx = onnx

        self.logger.info("=" * 80)
        self.logger.info("LOADING MODEL FOR PASS@K EVALUATION")
        self.logger.info("=" * 80)
        self.logger.info("Model : %s", model_path)
        self.logger.info("Mode  : %s", "ONNX (CPU)" if onnx else "PyTorch")

        self.model_path = model_path
        self.config = load_config()

        # Load tokenizer
        self.logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.logger.debug("Tokenizer loaded. pad_token set to eos_token.")

        if onnx:
            # -----------------------------------------------------------------
            # ONNX path — use ORTModelForCausalLM with CPUExecutionProvider
            # -----------------------------------------------------------------
            from optimum.onnxruntime import ORTModelForCausalLM

            self.logger.info("Loading ONNX model weights...")
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
            self.logger.info("Loading model weights...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device if device == "auto" else None,
            )

            if device != "auto":
                self.model = self.model.to(device)

            self.device = next(self.model.parameters()).device
            self.logger.info("Model loaded on device: %s", self.device)

        # Load HumanEval
        self.logger.info("Loading HumanEval dataset...")
        try:
            self.dataset = load_dataset("openai_humaneval", split="test")
            self.logger.info("HumanEval loaded — %d problems available.", len(self.dataset))
        except Exception as exc:
            self.logger.warning("HumanEval not available (%s). Falling back to synthetic tests.", exc)
            self.dataset = self._create_synthetic_tests()

    # ── Fallback dataset ───────────────────────────────────────────────────
    def _create_synthetic_tests(self) -> List[Dict]:
        """
        Create synthetic test cases as fallback.

        Contains:
          - Two simple bare-assert problems (original)
          - HumanEval/0 added as-is from the real dataset to verify that the
            check(candidate) preprocessing works correctly before going online.
        """
        self.logger.debug("Creating synthetic test cases.")
        return [
            # ── Original bare-assert style problems ────────────────────────
            {
                "task_id": "test/0",
                "prompt": "def add_numbers(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    ",
                "test": "assert add_numbers(2, 3) == 5\nassert add_numbers(-1, 1) == 0",
                "entry_point": "add_numbers",
            },
            {
                "task_id": "test/1",
                "prompt": "def reverse_string(s):\n    \"\"\"Reverse a string.\"\"\"\n    ",
                "test": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''",
                "entry_point": "reverse_string",
            },
            # ── HumanEval/0 added verbatim (check-style test) ──────────────
            {
                "task_id": "HumanEval/0",
                "prompt": (
                    "from typing import List\n\n\n"
                    "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
                    "    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n"
                    "    given threshold.\n"
                    "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
                    "    False\n"
                    "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
                    "    True\n"
                    "    \"\"\"\n"
                ),
                "test": (
                    "\n\nMETADATA = {\n"
                    "    'author': 'jt',\n"
                    "    'dataset': 'test'\n"
                    "}\n\n\n"
                    "def check(candidate):\n"
                    "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
                    "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
                    "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n"
                    "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n"
                    "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n"
                    "    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n"
                    "    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n"
                ),
                "entry_point": "has_close_elements",
            },
        ]

    # ── Solution generation ────────────────────────────────────────────────
    def generate_solutions(self, problem: Dict, k: int = 10) -> List[str]:
        """
        Generate k solutions for a problem.

        Args:
            problem: Problem dict with 'prompt' field
            k: Number of solutions to generate

        Returns:
            List of generated code solutions
        """
        prompt = problem["prompt"]
        if self.model_path == "Qwen/Qwen2.5-Coder-0.5B-Instruct":
            prompt = f"Task: {prompt}\n\nExample format:\ndef example():\n    return 'code only'\n\nNow write the code:"
        else:
            prompt = f"Instruction: {prompt}\nOutput:\n"
        self.logger.debug("Generating %d solutions for task: %s", k, problem.get("task_id"))

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["model_max_length"],
        )

        # ------------------------------------------------------------------
        # For ONNX (CPU), do NOT move inputs to a CUDA device.
        # For PyTorch, move inputs to whatever device the model is on.
        # ------------------------------------------------------------------
        if not self.onnx:
            inputs = inputs.to(self.device)

        solutions = []
        for idx in range(k):
            self.logger.debug("  Generating solution %d / %d …", idx + 1, k)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_code = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            solutions.append(prompt + generated_code)

        self.logger.debug("Generated %d solutions for task %s.", len(solutions), problem.get("task_id"))
        return solutions

    # ── Main evaluation loop ───────────────────────────────────────────────
    def evaluate(self, num_problems: int = 50, k_values: List[int] = [1, 10]) -> Dict:
        """
        Evaluate pass@k on problems.

        Args:
            num_problems: Number of problems to evaluate
            k_values: List of k values to compute

        Returns:
            Evaluation results
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING PASS@K EVALUATION")
        self.logger.info("=" * 80)
        self.logger.info("Problems to evaluate : %d", num_problems)
        self.logger.info("k values             : %s", k_values)
        self.logger.info("Solutions per problem: %d (max k)", max(k_values))

        max_k = max(k_values)
        problems = list(self.dataset)[:num_problems]
        results_by_problem = []

        for i, problem in enumerate(problems, start=1):
            task_id = problem["task_id"]
            entry_point = problem["entry_point"]
            self.logger.info("Problem %d / %d — %s", i, len(problems), task_id)

            # Generate solutions
            solutions = self.generate_solutions(problem, k=max_k)

            # Test each solution
            passed_solutions = []
            error_occured = []
            for j, solution in enumerate(solutions, start=1):
                passed, error = execute_code(
                    extract_code_from_response(solution),
                    problem.get("test", ""),
                    entry_point=entry_point,
                    timeout=60,
                )
                passed_solutions.append(passed)
                error_occured.append(error)
                if passed:
                    self.logger.debug("  Solution %d: PASSED", j)
                else:
                    self.logger.debug("  Solution %d: FAILED — %s", j, error)

            num_passed = sum(passed_solutions)
            self.logger.info(
                "  %s → %d / %d solutions passed.", task_id, num_passed, len(solutions)
            )

            results_by_problem.append({
                "task_id": task_id,
                "num_generated": len(solutions),
                "num_passed": num_passed,
                "passed_solutions": passed_solutions,
                "solutions": solutions,
                "error_occured": error_occured
            })

        # Calculate pass@k
        pass_at_k_results = {}
        for k in k_values:
            per_problem = [
                estimate_pass_at_k(r["num_generated"], r["num_passed"], k)
                for r in results_by_problem
            ]
            avg = np.mean(per_problem) * 100
            pass_at_k_results[f"pass@{k}"] = avg
            self.logger.info("pass@%d : %.2f%%", k, avg)

        total_generated = sum(r["num_generated"] for r in results_by_problem)
        total_passed    = sum(r["num_passed"]    for r in results_by_problem)

        self.logger.info("-" * 80)
        self.logger.info(
            "Total solutions — generated: %d | passed: %d | pass rate: %.1f%%",
            total_generated,
            total_passed,
            (total_passed / total_generated * 100) if total_generated else 0,
        )
        self.logger.info("=" * 80)

        return {
            "metadata": {
                "model_path": self.model_path,
                "model_mode": "onnx_cpu" if self.onnx else "pytorch",
                "num_problems": len(problems),
                "k_values": k_values,
                "total_solutions_generated": total_generated,
                "total_solutions_passed": total_passed,
            },
            "pass_at_k": pass_at_k_results,
            "per_problem_results": results_by_problem,
        }

    # ── Persist results ────────────────────────────────────────────────────
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info("Results saved to: %s", output_path)


# ── CLI entry-point ───────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pass@k evaluation")
    parser.add_argument("--model_path",   type=str,            required=True,  help="Path to model")
    parser.add_argument("--output",       type=str,            required=True,  help="Output JSON path")
    parser.add_argument("--num_problems", type=int,            default=50,     help="Number of problems")
    parser.add_argument("--k_values",     type=int, nargs="+", default=[1, 10],help="k values for pass@k")
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Load model as ONNX via ORTModelForCausalLM (CPU only). "
             "model_path should point to the ONNX model directory."
    )

    args = parser.parse_args()

    evaluator = PassAtKEvaluator(args.model_path, onnx=args.onnx)
    results   = evaluator.evaluate(num_problems=args.num_problems, k_values=args.k_values)
    evaluator.save_results(results, args.output)

    evaluator.logger.info("=" * 80)
    evaluator.logger.info("PASS@K RESULTS SUMMARY")
    evaluator.logger.info("=" * 80)
    evaluator.logger.info("Mode : %s", "ONNX (CPU)" if args.onnx else "PyTorch")
    for k, score in results["pass_at_k"].items():
        evaluator.logger.info("  %-10s : %.1f%%", k, score)
    evaluator.logger.info("=" * 80)


if __name__ == "__main__":
    main()