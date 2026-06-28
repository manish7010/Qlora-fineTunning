# Evaluation Results

This directory stores all evaluation outputs produced by the three evaluation scripts.

---

## Overview of Evaluation Scripts

| Script | What it measures | Output file |
|---|---|---|
| `eval_comprehensive.py` | Syntax validity, CodeBLEU, Exact Match, token stats, code ratio, cyclomatic complexity | `*_results.json` |
| `eval_llm_judge.py` | LLM-as-judge scores (correctness, quality, best practices, readability) via Groq | `llm_judge_results.json` |
| `eval_pass_at_k.py` | Functional correctness via HumanEval pass@k | `pass_at_k_results.json` |

---

## Expected Files

After running all evaluations, this directory will contain:
results/
├── base_model_results.json         # Base model metrics on test set
├── finetuned_model_results.json    # Fine-tuned model metrics on test set
├── pass_at_k_results.json          # Pass@k functional correctness (HumanEval)
└── llm_judge_results.json          # LLM-as-judge scores (optional)

---

## Script Usage & Commands

### 1. `eval_comprehensive.py` — Comprehensive Metrics

Evaluates a model on your test set across 9 metric categories.

**Basic usage (fine-tuned model):**
```bash
python eval_comprehensive.py \
  --model_path training/outputs/final_model \
  --test_data data/datasets/test.json \
  --output results/finetuned_model_results.json
```

**Base model:**
```bash
python eval_comprehensive.py \
  --model_path Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --test_data data/datasets/test.json \
  --output results/base_model_results.json
```

**ONNX model (CPU only):**
```bash
python eval_comprehensive.py \
  --model_path training/outputs/onnx_model \
  --output results/onnx_model_results.json \
  --onnx
```

**Recompute metrics from an existing results file (no regeneration):**
```bash
python eval_comprehensive.py \
  --results_json results/finetuned_model_results.json \
  --output results/finetuned_model_results_recomputed.json
```

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--model_path` | required* | Path to model or ONNX directory. Optional if `--results_json` is provided and JSON has `metadata.model_path`. |
| `--test_data` | `data/datasets/test.json` | Path to `test.json` |
| `--output` | required | Output JSON path |
| `--num_samples` | `1000` | Number of test samples to evaluate |
| `--batch_size` | `8` | Generation batch size |
| `--onnx` | `False` | Load model via `ORTModelForCausalLM` (CPU only) |
| `--results_json` | `None` | Skip generation; recompute metrics from an existing results JSON |

---

### 2. `eval_llm_judge.py` — LLM-as-Judge

Uses `llama-3.3-70b-versatile` via Groq to score generated code on four dimensions. Requires a `GROQ_API_KEY` in your `.env` file.

**Setup:**
```bash
echo "GROQ_API_KEY=your_key_here" >> .env
```

**Run:**
```bash
python eval_llm_judge.py \
  --results_file results/finetuned_model_results.json \
  --output results/llm_judge_results.json \
  --num_samples 100
```

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--results_file` | required | Path to a comprehensive eval results JSON (must have `detailed_results`) |
| `--output` | required | Output path for judge results JSON |
| `--num_samples` | `100` | Number of samples to send to the judge |
| `--api_key` | env | Groq API key — overrides `GROQ_API_KEY` in `.env` |
| `--log_dir` | `logs` | Directory for log files |

> **Note:** Groq has generous rate limits but the script adds a 0.5 s sleep between calls to avoid bursts. Evaluating 100 samples takes roughly 1–2 minutes.

---

### 3. `eval_pass_at_k.py` — Pass@k (Functional Correctness)

Evaluates functional correctness on HumanEval problems. Generates multiple solutions per problem and uses the unbiased pass@k estimator from the Codex paper.

**Run:**
```bash
python eval_pass_at_k.py \
  --model_path training/outputs/final_model \
  --output results/pass_at_k_results.json \
  --num_problems 50 \
  --k_values 1 10
```

**ONNX model:**
```bash
python eval_pass_at_k.py \
  --model_path training/outputs/onnx_model \
  --output results/pass_at_k_results.json \
  --onnx
```

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--model_path` | required | Path to model or ONNX directory |
| `--output` | required | Output JSON path |
| `--num_problems` | `50` | Number of HumanEval problems to evaluate |
| `--k_values` | `1 10` | Space-separated list of k values (e.g. `1 5 10`) |
| `--onnx` | `False` | Load model via `ORTModelForCausalLM` (CPU only) |

> **Note:** Each problem generates `max(k_values)` solutions. With `--k_values 1 10` and `--num_problems 50`, this produces 500 total inference calls — plan GPU/CPU time accordingly.

---

## Output File Structures

### Comprehensive results (`*_results.json`)

```json
{
  "metadata": {
    "model_path": "training/outputs/final_model",
    "model_mode": "pytorch",
    "test_data_path": "data/datasets/test.json",
    "evaluation_date": "2025-01-15T14:30:00",
    "num_samples": 1000,
    "device": "cuda:0"
  },
  "metrics": {
    "syntax_validity_pct": 93.2,
    "compilation_rate_pct": 93.2,
    "codebleu": 0.724,
    "exact_match_pct": 12.4,
    "avg_tokens": 87.3,
    "median_tokens": 72.0,
    "std_tokens": 45.1,
    "avg_code_ratio": 0.91,
    "excess_explanation_pct": 8.7,
    "avg_cyclomatic_complexity": 2.3,
    "avg_response_length_chars": 412.5,
    "median_response_length_chars": 348.0,
    "generation_time_sec": 840.2,
    "avg_time_per_sample": 0.84,
    "samples_evaluated": 1000
  },
  "detailed_results": [
    {
      "sample_id": 0,
      "instruction": "Instruction: Write a function...\nOutput:\n",
      "generated": "def example(): ...",
      "reference": "def example(): ..."
    }
  ]
}
```

### LLM judge results (`llm_judge_results.json`)

```json
{
  "aggregate_scores": {
    "correctness": 7.4,
    "code_quality": 7.1,
    "best_practices": 6.9,
    "readability": 7.6,
    "overall": 7.25
  },
  "num_samples_evaluated": 100,
  "individual_scores": [
    {
      "correctness": 8,
      "code_quality": 7,
      "best_practices": 7,
      "readability": 8,
      "overall": 7.5,
      "reasoning": "Correct solution, clean structure, follows PEP8."
    }
  ]
}
```

### Pass@k results (`pass_at_k_results.json`)

```json
{
  "metadata": {
    "model_path": "training/outputs/final_model",
    "model_mode": "pytorch",
    "num_problems": 50,
    "k_values": [1, 10],
    "total_solutions_generated": 500,
    "total_solutions_passed": 312
  },
  "pass_at_k": {
    "pass@1": 54.3,
    "pass@10": 78.9
  },
  "per_problem_results": [
    {
      "task_id": "HumanEval/0",
      "num_generated": 10,
      "num_passed": 7,
      "passed_solutions": [true, true, false, true, true, false, true, true, false, true],
      "solutions": ["..."],
      "error_occured": [null, null, "Assertion failed: ...", ...]
    }
  ]
}
```

---

## Metric Reference

| Metric | Script | Range | What a good score looks like |
|---|---|---|---|
| `syntax_validity_pct` | Comprehensive | 0–100% | > 90% |
| `codebleu` | Comprehensive | 0.0–1.0 | > 0.65 is strong for fine-tuned models |
| `exact_match_pct` | Comprehensive | 0–100% | Typically low (5–20%); useful for tracking relative gains |
| `avg_code_ratio` | Comprehensive | 0.0–1.0 | Close to 1.0 means responses are mostly code, little prose |
| `excess_explanation_pct` | Comprehensive | 0–100% | Lower is better for instruction-following |
| `avg_cyclomatic_complexity` | Comprehensive | ≥ 1 | 1–5 is typical for small functions |
| `overall` (judge) | LLM Judge | 0–10 | > 7.0 indicates good general quality |
| `pass@1` | Pass@k | 0–100% | > 30% is a solid baseline; > 60% is strong |
| `pass@10` | Pass@k | 0–100% | > 70% is a solid baseline |

---

## Notes

- All result files are gitignored (too large for version control) — keep local copies.
- The `--results_json` flag in `eval_comprehensive.py` lets you re-run or update metrics without re-generating model outputs, which is useful when adding new metric functions.
- If HumanEval is unavailable offline, `eval_pass_at_k.py` automatically falls back to a small built-in synthetic test suite.
- ONNX evaluation always runs on CPU regardless of `--device`. Expect slower inference but more portable deployment testing.