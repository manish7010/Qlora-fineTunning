"""
Evaluation script to determine if Qwen2.5-Coder-0.5B needs fine-tuning.

Tests base model on 100 Python/ML samples and measures:
1. Code correctness (syntax validity)
2. Response conciseness (explanation overhead)
3. Generation speed
4. Task completion quality

Results saved to base_model_evaluation.json
"""

import json
import time
from datetime import datetime
from typing import Dict, List

from utils.code_metrics import (
    calculate_code_to_response_ratio,
    calculate_verbosity_ratio,
    check_syntax,
    extract_code_from_response,
    has_excess_explanation,
)
from utils.logger import setup_evaluation_logger
from utils.io_utils import load_model, load_dataset_samples
from config.config import MAX_NEW_TOKENS, MODEL_NAME, TEMPERATURE, NUM_SAMPLES

logger = setup_evaluation_logger(output_dir="logs")

# ============================================================================
# GENERATION
# ============================================================================

def run_batch_generation(pipe, instructions: list) -> tuple[list, float]:
    """
    Generate responses for all instructions in batches.

    Returns:
        Tuple of (raw outputs list, avg generation time per sample)
    """
    logger.info(f"Generating {len(instructions)} responses in batches...")
    start_time = time.time()

    outputs = pipe(
        instructions,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        return_full_text=False,
        batch_size=8,
    )

    total_time = time.time() - start_time
    avg_time = total_time / len(instructions)

    logger.info(f"Generated all responses in {total_time:.1f}s (avg {avg_time:.2f}s/sample)")
    return outputs, avg_time


def run_sequential_generation(pipe, instructions: list) -> tuple[list, list]:
    """
    Fallback: generate responses one by one.

    Returns:
        Tuple of (raw outputs list, per-sample generation times)
    """
    logger.info("Running sequential generation fallback...")
    outputs = []
    times = []

    for i, instruction in enumerate(instructions):
        logger.debug(f"Generating sample {i + 1}/{len(instructions)}")
        start = time.time()
        output = pipe(
            instruction,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            return_full_text=False,
        )
        times.append(time.time() - start)
        outputs.append(output[0])

    return outputs, times


# ============================================================================
# RESULT PROCESSING
# ============================================================================

def build_sample_result(
    sample_id: int,
    instruction: str,
    response: str,
    generation_time: float,
    pipe,
    ground_truth: str,
) -> Dict:
    """Extract metrics from a single generated response and return a result dict."""
    code = extract_code_from_response(response)
    syntax_check = check_syntax(code)
    num_tokens = len(pipe.tokenizer.encode(response))

    return {
        "sample_id": sample_id,
        "instruction": instruction,
        "response": response,
        "extracted_code": code,
        "metrics": {
            "syntax_valid": syntax_check["valid"],
            "syntax_error": syntax_check["error"],
            "generation_time_sec": round(generation_time, 2),
            "response_length_chars": len(response),
            "code_length_chars": len(code),
            "verbosity_ratio": round(calculate_verbosity_ratio(response, code), 2),
            "has_excess_explanation": has_excess_explanation(response, code),
            "total_tokens": num_tokens,
            "tokens_per_sec": round(num_tokens / generation_time, 2) if generation_time > 0 else 0,
            "code_to_response_ratio": round(calculate_code_to_response_ratio(response, code), 2),
        },
        "ground_truth": ground_truth,
    }


def process_results(
    samples,
    instructions: list,
    outputs: list,
    generation_times,
    pipe
) -> List[Dict]:
    """
    Process raw pipeline outputs into structured result dicts.

    Args:
        generation_times: Either a single float (batch avg) or list of floats (sequential).
    """
    results = []
    is_batch_time = isinstance(generation_times, float)

    for i, (sample, output) in enumerate(zip(samples, outputs)):
        gen_time = generation_times if is_batch_time else generation_times[i]

        try:
            response = output["generated_text"] if isinstance(output, dict) else output[0]["generated_text"]
            result = build_sample_result(
                sample_id=i,
                instruction=instructions[i],
                response=response,
                generation_time=gen_time,
                pipe=pipe,
                ground_truth=sample.get("output", "N/A"),
            )
        except Exception as e:
            logger.error(f"Failed to process sample {i}: {e}")
            result = {
                "sample_id": i,
                "instruction": instructions[i],
                "error": str(e),
                "metrics": {"syntax_valid": False, "generation_failed": True},
            }

        results.append(result)

    logger.info(f"Processed {len(results)} results")
    return results


# ============================================================================
# AGGREGATION & RECOMMENDATIONS
# ============================================================================

def calculate_aggregate_metrics(results: List[Dict]) -> Dict:
    """Calculate aggregate statistics across all samples."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        return {"error": "All samples failed"}

    n = len(valid)
    syntax_valid_count = sum(1 for r in valid if r["metrics"]["syntax_valid"])
    excess_count = sum(1 for r in valid if r["metrics"]["has_excess_explanation"])

    return {
        "total_samples": len(results),
        "successful_generations": n,
        "failed_generations": len(results) - n,
        "syntax_validity": {
            "valid_count": syntax_valid_count,
            "valid_percentage": round(syntax_valid_count / n * 100, 2),
            "invalid_count": n - syntax_valid_count,
        },
        "verbosity_analysis": {
            "samples_with_excess_explanation": excess_count,
            "excess_explanation_percentage": round(excess_count / n * 100, 2),
            "avg_verbosity_ratio": round(sum(r["metrics"]["verbosity_ratio"] for r in valid) / n, 2),
            "avg_code_to_response_ratio": round(sum(r["metrics"]["code_to_response_ratio"] for r in valid) / n, 2),
        },
        "performance": {
            "avg_generation_time_sec": round(sum(r["metrics"]["generation_time_sec"] for r in valid) / n, 2),
            "avg_tokens_per_sec": round(sum(r["metrics"]["tokens_per_sec"] for r in valid) / n, 2),
            "total_time_for_100_samples_min": round(
                sum(r["metrics"]["generation_time_sec"] for r in valid) / n * 100 / 60, 2
            ),
        },
    }


def generate_recommendations(aggregate_metrics: Dict) -> Dict:
    """Generate recommendations based on evaluation results."""
    recommendations = {"should_finetune": False, "reasons": [], "potential_improvements": []}

    syntax_pct = aggregate_metrics["syntax_validity"]["valid_percentage"]
    excess_pct = aggregate_metrics["verbosity_analysis"]["excess_explanation_percentage"]
    code_ratio = aggregate_metrics["verbosity_analysis"]["avg_code_to_response_ratio"]

    if syntax_pct < 80:
        recommendations["should_finetune"] = True
        recommendations["reasons"].append(
            f"Low syntax validity ({syntax_pct}%) - fine-tuning can improve code correctness"
        )
        recommendations["potential_improvements"].append("syntax_validity: +15-25%")

    if excess_pct > 60:
        recommendations["should_finetune"] = True
        recommendations["reasons"].append(
            f"High verbosity ({excess_pct}% samples with excess explanation) - "
            "fine-tuning can produce more concise responses"
        )
        recommendations["potential_improvements"].append("reduce_explanation_overhead: 40-60%")

    if code_ratio < 0.5:
        recommendations["should_finetune"] = True
        recommendations["reasons"].append(
            f"Low code-to-response ratio ({code_ratio}) - too much non-code content"
        )
        recommendations["potential_improvements"].append("code_focus: +30-50%")

    if not recommendations["should_finetune"]:
        recommendations["reasons"].append("Base model performs adequately on Python/ML tasks")
        recommendations["alternative_approaches"] = [
            "Use prompt engineering to reduce explanations",
            "Post-process responses to extract code blocks",
            "Consider fine-tuning only if domain-specific patterns are needed",
        ]
    else:
        recommendations["expected_benefits"] = [
            "More concise responses (code-only output)",
            "Better syntax correctness",
            "Domain-specific patterns (ML libraries)",
            "Faster inference (fewer tokens generated)",
        ]

    return recommendations


# ============================================================================
# MAIN
# ============================================================================

def main():

    logger.info("=" * 80)
    logger.info("BASE MODEL EVALUATION - Qwen2.5-Coder-0.5B-Instruct")
    logger.info("=" * 80)
    logger.info(f"Model: {MODEL_NAME} | Samples: {NUM_SAMPLES} | Max new tokens: {MAX_NEW_TOKENS}")

    pipe = load_model()
    samples, instructions = load_dataset_samples()

    # Generation: attempt batch, fall back to sequential
    try:
        outputs, avg_time = run_batch_generation(pipe, instructions)
        results = process_results(samples, instructions, outputs, avg_time, pipe)
    except Exception as e:
        logger.warning(f"Batch generation failed ({e}), falling back to sequential...")
        outputs, times = run_sequential_generation(pipe, instructions)
        results = process_results(samples, instructions, outputs, times, pipe)

    # Aggregate
    aggregate_metrics = calculate_aggregate_metrics(results)
    recommendations = generate_recommendations(aggregate_metrics)

    # Save
    output = {
        "metadata": {
            "model_name": MODEL_NAME,
            "evaluation_date": datetime.now().isoformat(),
            "num_samples": NUM_SAMPLES,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
        },
        "aggregate_metrics": aggregate_metrics,
        "recommendations": recommendations,
        "detailed_results": results,
    }

    output_file = "base_model_evaluation_after_fine_tuning_eos.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Syntax Validity:            {aggregate_metrics['syntax_validity']['valid_percentage']}%")
    logger.info(f"Excess Explanation:         {aggregate_metrics['verbosity_analysis']['excess_explanation_percentage']}%")
    logger.info(f"Avg Code-to-Response Ratio: {aggregate_metrics['verbosity_analysis']['avg_code_to_response_ratio']}")
    logger.info(f"Avg Generation Time:        {aggregate_metrics['performance']['avg_generation_time_sec']}s")
    logger.info(f"Avg Tokens/sec:             {aggregate_metrics['performance']['avg_tokens_per_sec']}")
    logger.info(f"Recommendation:             {'FINE-TUNE' if recommendations['should_finetune'] else 'BASE MODEL SUFFICIENT'}")
    for reason in recommendations["reasons"]:
        logger.info(f"  - {reason}")
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()