"""
Analysis script to interpret base_model_evaluation.json results.
Run after base_model_evaluation.py completes.
"""

import json
import sys
from typing import Dict

from utils.logger import setup_evaluation_logger


# ============================================================================
# DATA LOADING
# ============================================================================

def load_results(logger, filename: str = "base_model_evaluation.json") -> Dict:
    """Load evaluation results from JSON."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"{filename} not found. Run base_model_evaluation.py first.")
        sys.exit(1)


# ============================================================================
# REPORTING
# ============================================================================

def log_decision(data: Dict, logger) -> None:
    """Log recommendation with supporting reasoning."""
    rec = data["recommendations"]
    metrics = data["aggregate_metrics"]

    logger.info("=" * 80)
    logger.info("FINAL DECISION")
    logger.info("=" * 80)

    if rec["should_finetune"]:
        logger.info("RECOMMENDATION: PROCEED WITH FINE-TUNING")
        logger.info("Reasons:")
        for i, reason in enumerate(rec["reasons"], 1):
            logger.info(f"  {i}. {reason}")

        if "potential_improvements" in rec:
            logger.info("Expected improvements after fine-tuning:")
            for improvement in rec["potential_improvements"]:
                logger.info(f"  - {improvement}")
    else:
        logger.info("RECOMMENDATION: BASE MODEL IS SUFFICIENT")
        logger.info("Current performance:")
        logger.info(f"  - Syntax Validity:    {metrics['syntax_validity']['valid_percentage']}%")
        logger.info(f"  - Excess Explanation: {metrics['verbosity_analysis']['excess_explanation_percentage']}%")
        logger.info(f"  - Code Focus:         {metrics['verbosity_analysis']['avg_code_to_response_ratio']}")

        if "alternative_approaches" in rec:
            logger.info("Alternative approaches:")
            for i, alt in enumerate(rec["alternative_approaches"], 1):
                logger.info(f"  {i}. {alt}")


def log_detailed_metrics(data: Dict, logger) -> None:
    """Log detailed metric breakdown."""
    metrics = data["aggregate_metrics"]

    logger.info("=" * 80)
    logger.info("DETAILED METRICS BREAKDOWN")
    logger.info("=" * 80)

    # Syntax validity
    syntax_pct = metrics["syntax_validity"]["valid_percentage"]
    syntax_status = "PASS" if syntax_pct >= 80 else "FAIL"
    logger.info("Code Correctness:")
    logger.info(f"  - Valid syntax:  {syntax_pct}%  [{syntax_status} | threshold: 80%]")
    logger.info(f"  - Invalid count: {metrics['syntax_validity']['invalid_count']}/{metrics['total_samples']}")

    # Verbosity
    excess_pct = metrics["verbosity_analysis"]["excess_explanation_percentage"]
    verbosity_status = "PASS" if excess_pct < 60 else "FAIL"
    logger.info("Response Conciseness:")
    logger.info(f"  - Excess explanation: {excess_pct}%  [{verbosity_status} | threshold: <60%]")
    logger.info(f"  - Avg verbosity ratio:       {metrics['verbosity_analysis']['avg_verbosity_ratio']}")
    logger.info(f"  - Avg code-to-response ratio:{metrics['verbosity_analysis']['avg_code_to_response_ratio']}")

    # Performance
    logger.info("Generation Performance:")
    logger.info(f"  - Avg generation time: {metrics['performance']['avg_generation_time_sec']}s/sample")
    logger.info(f"  - Avg tokens/sec:      {metrics['performance']['avg_tokens_per_sec']}")
    logger.info(f"  - Est. time (100 samples): {metrics['performance']['total_time_for_100_samples_min']} min")


def log_sample_examples(data: Dict, logger, num_examples: int = 3) -> None:
    """Log representative sample outputs — one good, one verbose, one invalid."""
    results = data["detailed_results"]

    good_example = None
    verbose_example = None
    invalid_example = None

    for result in results:
        if "error" in result:
            continue
        m = result["metrics"]
        if m["syntax_valid"] and not m["has_excess_explanation"] and good_example is None:
            good_example = result
        elif m["has_excess_explanation"] and verbose_example is None:
            verbose_example = result
        elif not m["syntax_valid"] and invalid_example is None:
            invalid_example = result

    logger.info("=" * 80)
    logger.info(f"SAMPLE OUTPUTS (up to {num_examples} examples)")
    logger.info("=" * 80)

    if good_example:
        m = good_example["metrics"]
        logger.info("[GOOD] Valid syntax, concise response")
        logger.info(f"  Instruction:     {good_example['instruction'][:100]}...")
        logger.info(f"  Code length:     {m['code_length_chars']} chars")
        logger.info(f"  Response length: {m['response_length_chars']} chars")
        logger.info(f"  Verbosity ratio: {m['verbosity_ratio']}")
        logger.info(f"  Code preview:\n{good_example['extracted_code'][:200]}")

    if verbose_example:
        m = verbose_example["metrics"]
        logger.info("[VERBOSE] Valid syntax but excessive explanation")
        logger.info(f"  Instruction:     {verbose_example['instruction'][:100]}...")
        logger.info(f"  Code length:     {m['code_length_chars']} chars")
        logger.info(f"  Response length: {m['response_length_chars']} chars")
        logger.info(f"  Verbosity ratio: {m['verbosity_ratio']}")
        logger.info(f"  Response preview:\n{verbose_example['response'][:300]}")

    if invalid_example:
        m = invalid_example["metrics"]
        logger.info("[INVALID] Syntax error in generated code")
        logger.info(f"  Instruction:  {invalid_example['instruction'][:100]}...")
        logger.info(f"  Syntax error: {m['syntax_error']}")
        logger.info(f"  Code preview:\n{invalid_example['extracted_code'][:200]}")


def log_next_steps(data: Dict, logger) -> None:
    """Log recommended next steps based on decision."""
    logger.info("=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)

    if data["recommendations"]["should_finetune"]:
        logger.info("1. Begin fine-tuning pipeline setup")
        logger.info("2. Use QLoRA (4-bit quantization) for efficient training")
        logger.info("3. Train for 2-3 epochs")
        logger.info("4. Re-run comprehensive evaluation to measure improvements against this baseline")
    else:
        logger.info("1. Apply prompt engineering strategies to address identified issues")
        logger.info("2. Post-process outputs to extract code blocks where needed")
        logger.info("3. Consider fine-tuning only if domain-specific patterns are required")


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger = setup_evaluation_logger(output_dir="logs")

    logger.info("=" * 80)
    logger.info("BASE MODEL EVALUATION - RESULTS ANALYSIS")
    logger.info("=" * 80)

    data = load_results(logger)

    logger.info(f"Model:   {data['metadata']['model_name']}")
    logger.info(f"Date:    {data['metadata']['evaluation_date']}")
    logger.info(f"Samples: {data['metadata']['num_samples']}")

    log_decision(data, logger)
    log_detailed_metrics(data, logger)
    log_sample_examples(data, logger)
    log_next_steps(data, logger)

    logger.info("=" * 80)


if __name__ == "__main__":
    main()