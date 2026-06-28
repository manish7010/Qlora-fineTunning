"""
LLM-as-Judge Evaluation

Uses Groq (llama-3.3-70b-versatile) to evaluate code quality on dimensions:
- Correctness
- Code quality
- Best practices
- Readability
"""

import json
import os
import re
import time
from typing import Dict, List

from dotenv import load_dotenv
from groq import Groq

from utils.logger import setup_evaluation_logger

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()


# ── LLMJudge ──────────────────────────────────────────────────────────────────
class LLMJudge:
    """Evaluate code using Groq's llama-3.3-70b-versatile as a judge."""

    MODEL = "llama-3.3-70b-versatile" or os.environ.get("GROQ_MODEL")

    def __init__(self, api_key: str = None, log_dir: str = "logs"):
        """
        Args:
            api_key:  Groq API key. Falls back to GROQ_API_KEY env var (loaded from .env).
            log_dir:  Directory for log files.
        """
        self.logger = setup_evaluation_logger(output_dir=log_dir)
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")

        if not resolved_key:
            self.logger.error("No Groq API key found. Set GROQ_API_KEY in your .env file.")
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env or pass api_key=.")

        self.client = Groq(api_key=resolved_key)
        self.logger.info("LLMJudge initialised with model: %s", self.MODEL)

    def parse_judge_response(self, response_text: str) -> dict:
        """
        Robustly parse judge response with 3 fallback layers:
        1. Direct JSON parse (after stripping markdown fences)
        2. Regex extraction of individual fields → reconstruct JSON
        3. Raise a clear error with debug info
        """

        # --- Layer 1: Strip markdown fences and try direct JSON parse ---
        cleaned = response_text.strip()

        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        try:
            scores = json.loads(cleaned)
            self.logger.info("Layer 1 parse succeeded")
            return scores
        except json.JSONDecodeError as e:
            self.logger.warning("Layer 1 (direct JSON parse) failed: %s", e)

        # --- Layer 2: Regex extraction of individual fields ---
        self.logger.info("Attempting Layer 2: regex field extraction")

        # Matches:  "key": value  OR  "key": "string value"
        # Handles numbers, floats, booleans, and quoted strings
        field_patterns = {
            # Numeric fields (int or float)
            "correctness":    r'"correctness"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            "code_quality":   r'"code_quality"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            "best_practices": r'"best_practices"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            "readability":    r'"readability"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            "overall":        r'"overall"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            # String field — captures everything between the quotes, allows escaped quotes
            "reasoning":      r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"',
        }

        extracted = {}
        missing_fields = []

        for field, pattern in field_patterns.items():
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                raw_val = match.group(1)
                if field == "reasoning":
                    extracted[field] = raw_val.replace('\\"', '"')
                else:
                    # Parse as int if no decimal point, else float
                    extracted[field] = float(raw_val) if "." in raw_val else int(raw_val)
                self.logger.info("Extracted '%s': %s", field, extracted[field])
            else:
                missing_fields.append(field)
                self.logger.info("Field '%s' not found in response", field)

        # Succeed if we got at least the numeric score fields
        required_fields = {"correctness", "code_quality", "best_practices", "readability", "overall"}
        found_required = required_fields - set(missing_fields)

        if found_required == required_fields:
            self.logger.info("Layer 2 parse succeeded (all required fields found). Missing optional: %s", missing_fields)
            return extracted

        # --- Layer 3: Graceful failure ---
        raise ValueError(
            f"Failed to parse judge response after all extraction attempts.\n"
            f"Missing fields: {missing_fields}\n"
            f"Extracted so far: {extracted}\n"
            f"Raw response (first 500 chars): {response_text[:500]}"
        )

    # ── Single sample ──────────────────────────────────────────────────────
    def judge_sample(
        self,
        instruction: str,
        generated_code: str,
        reference_code: str = None,
    ) -> Dict:
        """
        Evaluate a single code sample.

        Args:
            instruction:    The task instruction.
            generated_code: Code generated by the model under test.
            reference_code: Reference / gold solution (optional).

        Returns:
            Dictionary with per-dimension scores and reasoning.
        """
        self.logger.debug("Judging sample | instruction: %.80s…", instruction)

        ref_block = (
            f"Reference Solution:\n```python\n{reference_code}\n```"
            if reference_code
            else ""
        )

        prompt = f"""You are a code quality evaluator. Evaluate the following code on these criteria:

Task: {instruction}

Generated Code:
```python
{generated_code}
```

{ref_block}

Rate on a scale of 0-10 for each criterion:

1. **Correctness**: Does it solve the task?
2. **Code Quality**: Is it clean, efficient, and well-structured?
3. **Best Practices**: Does it follow Python conventions?
4. **Readability**: Is it easy to understand?

Provide scores in JSON format:
{{
  "correctness": <score>,
  "code_quality": <score>,
  "best_practices": <score>,
  "readability": <score>,
  "overall": <average>,
  "reasoning": "<brief explanation>"
}}

Only output the JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.choices[0].message.content
            self.logger.debug("Raw judge response: %s", response_text[:200])

            scores = self.parse_judge_response(response_text)
            self.logger.debug(
                "Scores — correctness: %s, overall: %s",
                scores.get("correctness"),
                scores.get("overall"),
            )
            return scores

        except Exception as exc:
            self.logger.error("Error judging sample: %s", exc, exc_info=True)
            return {
                "correctness": 0,
                "code_quality": 0,
                "best_practices": 0,
                "readability": 0,
                "overall": 0,
                "reasoning": f"Error: {exc}",
                "raw_json": response_text
            }

    # ── Batch evaluation ───────────────────────────────────────────────────
    def evaluate_model(self, results: List[Dict], num_samples: int = 100) -> Dict:
        """
        Evaluate multiple samples and aggregate scores.

        Args:
            results:     List of dicts with keys: instruction, generated, reference.
            num_samples: Number of samples to evaluate.

        Returns:
            Dict with aggregate scores, sample count, and per-sample scores.
        """
        self.logger.info("=" * 60)
        self.logger.info("LLM-AS-JUDGE EVALUATION")
        self.logger.info("=" * 60)
        self.logger.info("Total samples available : %d", len(results))
        self.logger.info("Samples to evaluate     : %d", num_samples)

        samples_to_eval = results[:num_samples]
        all_scores: List[Dict] = []

        for i, sample in enumerate(samples_to_eval, start=1):
            self.logger.info("Evaluating sample %d / %d …", i, len(samples_to_eval))

            scores = self.judge_sample(
                instruction=sample.get("instruction", ""),
                generated_code=sample.get("generated", ""),
                reference_code=sample.get("reference", None),
            )
            all_scores.append(scores)

            # Groq has generous rate limits, but a small sleep avoids bursts
            time.sleep(0.5)

        n = len(all_scores)
        avg_scores = {
            "correctness":   sum(s["correctness"]   for s in all_scores) / n,
            "code_quality":  sum(s["code_quality"]  for s in all_scores) / n,
            "best_practices":sum(s["best_practices"] for s in all_scores) / n,
            "readability":   sum(s["readability"]   for s in all_scores) / n,
            "overall":       sum(s["overall"]       for s in all_scores) / n,
        }

        self.logger.info("-" * 60)
        self.logger.info("Aggregate results (%d samples):", n)
        for k, v in avg_scores.items():
            self.logger.info("  %-20s : %.2f / 10", k.replace("_", " ").title(), v)
        self.logger.info("=" * 60)

        return {
            "aggregate_scores": avg_scores,
            "num_samples_evaluated": n,
            "individual_scores": all_scores,
        }


# ── CLI entry-point ───────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation (Groq)")
    parser.add_argument("--results_file", required=True, help="Path to eval results JSON")
    parser.add_argument("--output",       required=True, help="Output path for judge results JSON")
    parser.add_argument("--num_samples",  type=int, default=100, help="Number of samples to judge")
    parser.add_argument("--api_key",      help="Groq API key (overrides .env / GROQ_API_KEY)")
    parser.add_argument("--log_dir",      default="logs", help="Directory for log files")
    args = parser.parse_args()

    logger = setup_evaluation_logger(output_dir="logs")

    # Load evaluation results
    logger.info("Loading results from: %s", args.results_file)
    with open(args.results_file, "r") as f:
        eval_data = json.load(f)

    results = eval_data.get("detailed_results", [])
    logger.info("Found %d samples in results file", len(results))

    # Initialise judge
    judge = LLMJudge(api_key=args.api_key, log_dir=args.log_dir)

    # Run evaluation
    judge_results = judge.evaluate_model(results, num_samples=args.num_samples)

    # Persist results
    with open(args.output, "w") as f:
        json.dump(judge_results, f, indent=2)

    logger.info("Results saved to: %s", args.output)


if __name__ == "__main__":
    main()