"""
Prompt Engineering Baseline Test

Tests multiple prompt strategies to determine if fine-tuning is actually needed.
Compares results against base_model_evaluation.json metrics.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Tuple

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
# PROMPT STRATEGIES
# ============================================================================

PROMPT_STRATEGIES = {
    "baseline": {
        "template": "{instruction}",
        "description": "No prompt engineering (same as base evaluation)",
    },
    "code_only_simple": {
        "template": "{instruction}\n\nOutput only valid Python code. No explanations.",
        "description": "Simple instruction to avoid explanations",
    },
    "code_only_strict": {
        "template": (
            "Task: {instruction}\n\n"
            "RULES:\n"
            "- Output ONLY executable Python code\n"
            "- NO explanations before or after code\n"
            "- NO markdown formatting\n"
            "- NO comments in code\n"
            "- Start directly with code\n\n"
            "Code:"
        ),
        "description": "Strict formatting rules",
    },
    "code_only_example": {
        "template": (
            "Task: {instruction}\n\n"
            "Example format:\n"
            "def example():\n"
            "    return 'code only'\n\n"
            "Now write the code:"
        ),
        "description": "Shows example of expected format",
    },
    "role_system": {
        "template": (
            "You are a code generator that outputs only valid, executable Python code "
            "with no explanations.\n\n"
            "Task: {instruction}\n\n"
            "Output:"
        ),
        "description": "System role definition",
    },
    "json_structured": {
        "template": (
            "Task: {instruction}\n\n"
            "Respond ONLY with a valid JSON object. No text before or after it.\n"
            "The JSON must have exactly two keys:\n"
            "  - \"code\": a string containing only the executable Python code\n"
            "  - \"explanation\": a string containing only the explanation of the code\n\n"
            "Example format:\n"
            "{{\n"
            "  \"code\": \"def example():\\n    return 42\",\n"
            "  \"explanation\": \"This function returns the integer 42.\"\n"
            "}}\n\n"
            "JSON:"
        ),
        "description": "Outputs code and explanation as structured JSON with separate keys",
    },
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_base_metrics() -> Dict:
    """Load baseline metrics from base_model_evaluation.json."""
    try:
        with open("base_model_evaluation.json", "r") as f:
            return json.load(f)["aggregate_metrics"]
    except FileNotFoundError:
        logger.error("base_model_evaluation.json not found. Run base_model_evaluation.py first.")
        raise


# ============================================================================
# JSON STRATEGY PARSING
# ============================================================================

def parse_json_response(response: str) -> Dict:
    """
    Parse a model response expected to be a JSON object with 'code' and 'explanation' keys.

    Returns a dict with:
        - 'code': extracted code string (empty string on failure)
        - 'explanation': extracted explanation string (empty string on failure)
        - 'parse_success': bool indicating whether JSON parsing succeeded
        - 'parse_error': error message string if parsing failed, else None
    """
    # Strip any accidental markdown fences or leading/trailing whitespace
    cleaned = response.strip()
    for fence in ["```json", "```"]:
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        code = parsed.get("code", "")
        explanation = parsed.get("explanation", "")

        if not isinstance(code, str):
            code = str(code)
        if not isinstance(explanation, str):
            explanation = str(explanation)

        return {
            "code": code,
            "explanation": explanation,
            "parse_success": True,
            "parse_error": None,
        }

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"JSON parse failed: {e}. Falling back to code extraction.")
        return {
            "code": extract_code_from_response(response),
            "explanation": "",
            "parse_success": False,
            "parse_error": str(e),
        }


# ============================================================================
# GENERATION
# ============================================================================

def build_prompted_inputs(samples, template: str) -> Tuple[list, list]:
    """Build prompted inputs and raw instructions from samples."""
    instructions, prompted_inputs = [], []
    for sample in samples:
        instruction = sample["instruction"]
        if sample.get("input"):
            instruction = f"### Instructions:\n{instruction}\n### Input:\n{sample['input']}"
        instructions.append(instruction)
        prompted_inputs.append(template.format(instruction=instruction))
    return instructions, prompted_inputs


def generate_responses(pipe, prompted_inputs: list) -> Tuple[list, float]:
    """
    Generate responses with batch fallback to sequential.

    Returns:
        Tuple of (list of response strings, avg generation time per sample)
    """
    logger.info(f"Generating {len(prompted_inputs)} responses...")
    start = time.time()

    try:
        outputs = pipe(
            prompted_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            return_full_text=False,
            batch_size=8,
        )
        total_time = time.time() - start
        avg_time = total_time / len(prompted_inputs)
        responses = [output["generated_text"] if isinstance(output, dict) else output[0]["generated_text"] for output in outputs]
        logger.info(f"Batch generation done in {total_time:.1f}s (avg {avg_time:.2f}s/sample)")

    except Exception as e:
        logger.warning(f"Batch generation failed ({e}), falling back to sequential...")
        responses, times = [], []
        for i, prompt in enumerate(prompted_inputs):
            logger.debug(f"Sequential sample {i + 1}/{len(prompted_inputs)}")
            t = time.time()
            out = pipe(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                return_full_text=False,
            )
            times.append(time.time() - t)
            responses.append(out[0]["generated_text"])
        avg_time = sum(times) / len(times)

    return responses, avg_time


# ============================================================================
# RESULT PROCESSING & METRICS
# ============================================================================

def build_sample_result(
    sample_id: int,
    instruction: str,
    prompted_input: str,
    response: str,
    generation_time: float,
    pipe,
    strategy_name: str = "",
) -> Dict:
    """Build a structured result dict for a single sample."""

    # --- Code & explanation extraction ---
    if strategy_name == "json_structured":
        parsed = parse_json_response(response)
        code = parsed["code"]
        explanation = parsed["explanation"]
        json_parse_success = parsed["parse_success"]
        json_parse_error = parsed["parse_error"]
    else:
        code = extract_code_from_response(response)
        explanation = None
        json_parse_success = None
        json_parse_error = None

    # --- Metrics always computed on code only (not full response) ---
    syntax_check = check_syntax(code)
    num_tokens = len(pipe.tokenizer.encode(response))

    result = {
        "sample_id": sample_id,
        "instruction": instruction,
        "prompted_input": prompted_input,
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
    }

    # --- json_structured exclusive fields ---
    if strategy_name == "json_structured":
        result["explanation"] = explanation
        result["metrics"]["json_parse_success"] = json_parse_success
        result["metrics"]["json_parse_error"] = json_parse_error
        result["metrics"]["explanation_length_chars"] = len(explanation)

    return result


def calculate_strategy_metrics(results: List[Dict], strategy_name: str = "") -> Dict:
    """Aggregate metrics for a single strategy's results."""
    valid = [r for r in results if "error" not in r]

    if not valid:
        return {"error": "All samples failed"}

    n = len(valid)
    syntax_valid_count = sum(1 for r in valid if r["metrics"]["syntax_valid"])
    excess_count = sum(1 for r in valid if r["metrics"]["has_excess_explanation"])

    metrics = {
        "total_samples": len(results),
        "successful_generations": n,
        "syntax_validity": {
            "valid_count": syntax_valid_count,
            "valid_percentage": round(syntax_valid_count / n * 100, 2),
        },
        "verbosity_analysis": {
            "samples_with_excess_explanation": excess_count,
            "excess_explanation_percentage": round(excess_count / n * 100, 2),
            "avg_code_to_response_ratio": round(
                sum(r["metrics"]["code_to_response_ratio"] for r in valid) / n, 2
            ),
        },
        "performance": {
            "avg_generation_time_sec": round(
                sum(r["metrics"]["generation_time_sec"] for r in valid) / n, 2
            ),
            "avg_tokens_per_sec": round(
                sum(r["metrics"]["tokens_per_sec"] for r in valid) / n, 2
            ),
        },
    }

    # --- json_structured: add JSON-specific aggregate metrics ---
    if strategy_name == "json_structured":
        parse_successes = [
            r for r in valid if r["metrics"].get("json_parse_success") is True
        ]
        parse_success_count = len(parse_successes)
        avg_explanation_len = (
            round(
                sum(r["metrics"]["explanation_length_chars"] for r in valid) / n, 2
            )
            if n > 0
            else 0
        )
        metrics["json_parsing"] = {
            "parse_success_count": parse_success_count,
            "parse_success_percentage": round(parse_success_count / n * 100, 2),
            "parse_failure_count": n - parse_success_count,
            "avg_explanation_length_chars": avg_explanation_len,
        }

    return metrics


# ============================================================================
# STRATEGY EVALUATION
# ============================================================================

def evaluate_strategy(pipe, samples, strategy_name: str, strategy_info: Dict) -> List[Dict]:
    """Run evaluation for a single prompt strategy."""
    logger.info(f"Evaluating strategy: {strategy_name} — {strategy_info['description']}")

    instructions, prompted_inputs = build_prompted_inputs(samples, strategy_info["template"])
    responses, avg_time = generate_responses(pipe, prompted_inputs)

    results = []
    for i, (instruction, prompted_input, response) in enumerate(
        zip(instructions, prompted_inputs, responses)
    ):
        try:
            result = build_sample_result(
                i, instruction, prompted_input, response, avg_time, pipe,
                strategy_name=strategy_name,
            )
        except Exception as e:
            logger.error(f"Failed to process sample {i} for strategy '{strategy_name}': {e}")
            result = {
                "sample_id": i,
                "instruction": instruction,
                "error": str(e),
                "metrics": {"syntax_valid": False, "generation_failed": True},
            }
        results.append(result)

    return results


# ============================================================================
# COMPARISON & DECISION
# ============================================================================

def compare_strategies(all_strategy_results: Dict, base_metrics: Dict) -> Dict:
    """Compare all strategies against baseline and determine best approach."""
    comparison = {}
    best_strategy = None
    best_score = 0

    for name, results in all_strategy_results.items():
        metrics = calculate_strategy_metrics(results, strategy_name=name)
        if "error" in metrics:
            logger.warning(f"Strategy '{name}' produced no valid results, skipping.")
            continue

        syntax_delta = (
            metrics["syntax_validity"]["valid_percentage"]
            - base_metrics["syntax_validity"]["valid_percentage"]
        )
        verbosity_delta = (
            base_metrics["verbosity_analysis"]["excess_explanation_percentage"]
            - metrics["verbosity_analysis"]["excess_explanation_percentage"]
        )
        speed_delta = (
            base_metrics["performance"]["avg_generation_time_sec"]
            - metrics["performance"]["avg_generation_time_sec"]
        )
        score = syntax_delta + (verbosity_delta * 0.5) + (speed_delta / 10)

        comparison[name] = {
            "metrics": metrics,
            "improvements": {
                "syntax_validity_delta": round(syntax_delta, 2),
                "verbosity_reduction_delta": round(verbosity_delta, 2),
                "speed_improvement_delta": round(speed_delta, 2),
                "combined_score": round(score, 2),
            },
        }

        if score > best_score:
            best_score = score
            best_strategy = name

    decision = make_final_decision(comparison, best_strategy, best_score)

    return {
        "comparison": comparison,
        "best_strategy": best_strategy,
        "best_improvement_score": round(best_score, 2),
        "decision": decision,
    }


def make_final_decision(
    comparison: Dict, best_strategy: str, best_score: float
) -> Dict:
    """Determine whether fine-tuning is warranted based on prompt engineering results."""
    if not best_strategy or best_strategy == "baseline":
        logger.info("No prompt strategy improved over baseline — fine-tuning recommended.")
        return {
            "should_finetune": True,
            "reason": "No prompt strategy improved over baseline",
            "confidence": "HIGH",
            "evidence": "Prompt engineering ineffective — fine-tuning is necessary",
        }

    best_metrics = comparison[best_strategy]["metrics"]
    improvements = comparison[best_strategy]["improvements"]

    syntax_valid = best_metrics["syntax_validity"]["valid_percentage"]
    verbosity = best_metrics["verbosity_analysis"]["excess_explanation_percentage"]

    issues = []
    if syntax_valid < 85:
        issues.append(f"Syntax validity still low ({syntax_valid}% < 85% threshold)")
    if verbosity > 40:
        issues.append(f"Verbosity still high ({verbosity}% > 40% threshold)")

    if issues:
        return {
            "should_finetune": True,
            "reason": f"Best prompt strategy ('{best_strategy}') does not meet quality thresholds",
            "remaining_issues": issues,
            "confidence": "HIGH",
            "evidence": (
                f"Prompt engineering achieved {improvements['syntax_validity_delta']:+.1f}% syntax improvement "
                f"and {improvements['verbosity_reduction_delta']:+.1f}% verbosity reduction — still below targets"
            ),
        }

    return {
        "should_finetune": False,
        "reason": f"Prompt strategy '{best_strategy}' meets quality thresholds",
        "confidence": "HIGH",
        "evidence": (
            f"Achieved {syntax_valid}% syntax validity and {verbosity}% excess explanation "
            "through prompt engineering alone"
        ),
        "recommendation": f"Use '{best_strategy}' prompt strategy instead of fine-tuning",
    }


# ============================================================================
# REPORTING
# ============================================================================

def log_comparison_table(final_analysis: Dict, base_metrics: Dict) -> None:
    """Log strategy comparison as a structured table."""
    logger.info("=" * 80)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 80)

    header = f"{'Strategy':<25} {'Syntax %':<12} {'Delta':<8} {'Verbose %':<12} {'Delta':<8} {'Score':<8}"
    logger.info(header)
    logger.info("-" * 80)

    baseline_syntax = base_metrics["syntax_validity"]["valid_percentage"]
    baseline_verbose = base_metrics["verbosity_analysis"]["excess_explanation_percentage"]
    logger.info(
        f"{'BASELINE':<25} {baseline_syntax:<12.1f} {'-':<8} {baseline_verbose:<12.1f} {'-':<8} {'-':<8}"
    )

    for name, data in final_analysis["comparison"].items():
        if name == "baseline":
            continue
        m = data["metrics"]
        imp = data["improvements"]

        # For json_structured, append parse success rate to the strategy name in the table
        display_name = name
        if name == "json_structured" and "json_parsing" in m:
            parse_pct = m["json_parsing"]["parse_success_percentage"]
            display_name = f"{name}({parse_pct:.0f}%json)"

        logger.info(
            f"{display_name:<25} "
            f"{m['syntax_validity']['valid_percentage']:<12.1f} "
            f"{imp['syntax_validity_delta']:+7.1f}  "
            f"{m['verbosity_analysis']['excess_explanation_percentage']:<12.1f} "
            f"{imp['verbosity_reduction_delta']:+7.1f}  "
            f"{imp['combined_score']:<8.1f}"
        )


def log_decision(final_analysis: Dict) -> None:
    """Log final decision with supporting evidence."""
    decision = final_analysis["decision"]

    logger.info("=" * 80)
    logger.info("FINAL DECISION")
    logger.info("=" * 80)

    if decision["should_finetune"]:
        logger.info("RECOMMENDATION: PROCEED WITH FINE-TUNING")
        logger.info(f"Confidence: {decision['confidence']}")
        logger.info(f"Reason: {decision['reason']}")
        logger.info(f"Evidence: {decision['evidence']}")
        if "remaining_issues" in decision:
            logger.info("Remaining issues after best prompt strategy:")
            for issue in decision["remaining_issues"]:
                logger.info(f"  - {issue}")
    else:
        logger.info("RECOMMENDATION: PROMPT ENGINEERING IS SUFFICIENT")
        logger.info(f"Confidence: {decision['confidence']}")
        logger.info(f"Reason: {decision['reason']}")
        logger.info(f"Evidence: {decision['evidence']}")
        logger.info(f"Best strategy: '{final_analysis['best_strategy']}'")


# ============================================================================
# MAIN
# ============================================================================

def main():

    logger.info("=" * 80)
    logger.info("PROMPT ENGINEERING BASELINE TEST")
    logger.info("=" * 80)

    base_metrics = load_base_metrics()
    logger.info("Baseline metrics loaded:")
    logger.info(f"  - Syntax Validity:    {base_metrics['syntax_validity']['valid_percentage']}%")
    logger.info(f"  - Excess Explanation: {base_metrics['verbosity_analysis']['excess_explanation_percentage']}%")
    logger.info(f"  - Avg Gen Time:       {base_metrics['performance']['avg_generation_time_sec']}s")

    pipe = load_model()
    samples, _ = load_dataset_samples()

    logger.info(f"Testing {len(PROMPT_STRATEGIES)} strategies on {NUM_SAMPLES} samples")

    # Evaluate each strategy
    all_strategy_results = {}
    for strategy_name, strategy_info in PROMPT_STRATEGIES.items():
        all_strategy_results[strategy_name] = evaluate_strategy(
            pipe, samples, strategy_name, strategy_info
        )

    # Compare and decide
    final_analysis = compare_strategies(all_strategy_results, base_metrics)

    log_comparison_table(final_analysis, base_metrics)
    log_decision(final_analysis)

    # Save results
    output = {
        "metadata": {
            "model_name": MODEL_NAME,
            "evaluation_date": datetime.now().isoformat(),
            "num_samples": NUM_SAMPLES,
            "strategies_tested": len(PROMPT_STRATEGIES),
        },
        "baseline_metrics": base_metrics,
        "prompt_strategies": {
            name: {
                "description": PROMPT_STRATEGIES[name]["description"],
                "template": PROMPT_STRATEGIES[name]["template"],
                "metrics": final_analysis["comparison"].get(name, {}).get("metrics", {}),
                "improvements": final_analysis["comparison"].get(name, {}).get("improvements", {}),
            }
            for name in PROMPT_STRATEGIES
        },
        "best_strategy": final_analysis["best_strategy"],
        "decision": final_analysis["decision"],
        "detailed_results": all_strategy_results,
    }

    output_file = "prompt_engineering_comparison_after_fine_tuning.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()