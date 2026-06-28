"""
merge_export_quantize_onnx.py

Full pipeline: LoRA Adapter → Merged Model → ONNX Export → INT8 Quantization

Steps:
    1. merge   — Merge LoRA adapter into base model, save to inference/merged_model/
    2. onnx    — Export merged model to ONNX format
    3. quantize — Apply INT8 quantization (platform-aware: arm64 / avx512 / avx2)
    4. benchmark (optional) — Compare merged vs quantized inference speed

Usage:
    # Full pipeline
    python inference/merge_export_quantize_onnx.py \
        --adapter_path training/outputs/final_model \
        --base_model Qwen/Qwen2.5-Coder-0.5B-Instruct \
        --output_dir inference \
        --steps all

    # Only merge
    python inference/merge_export_quantize.py --steps merge

    # Only ONNX export + quantize (merged model already exists)
    python inference/merge_export_quantize.py --steps onnx quantize

    # Full pipeline with benchmark
    python inference/merge_export_quantize.py --steps all --benchmark
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from utils.logger import get_logger

logger = get_logger(__name__)


def section(title: str):
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def ok(msg: str):
    logger.info(f"  ✓ {msg}")


def warn(msg: str):
    logger.warning(f"  ⚠  {msg}")


def info(msg: str):
    logger.info(f"  • {msg}")


# ---------------------------------------------------------------------------
# Platform detection for quantization config
# ---------------------------------------------------------------------------

def detect_quantization_backend() -> str:
    """
    Detect the best quantization backend for the current hardware.

    Returns one of: 'arm64', 'avx512', 'avx2', 'basic'
    """
    machine = platform.machine().lower()

    # ARM (Apple Silicon, ARM Linux servers)
    if machine in ("arm64", "aarch64"):
        info("Detected ARM64 architecture → using arm64 quantization config")
        return "arm64"

    # x86 — check CPU flags
    try:
        import cpuinfo  # optional: pip install py-cpuinfo
        flags = cpuinfo.get_cpu_info().get("flags", [])
        if "avx512f" in flags:
            info("Detected AVX-512 support → using avx512 quantization config")
            return "avx512"
        if "avx2" in flags:
            info("Detected AVX2 support → using avx2 quantization config")
            return "avx2"
    except ImportError:
        warn("py-cpuinfo not installed — falling back to basic quantization")
        warn("Install with: pip install py-cpuinfo  (for better CPU-optimized quantization)")

    info("Using basic dynamic quantization config")
    return "basic"


def build_quantization_config(backend: str):
    """Return an AutoQuantizationConfig for the detected backend."""
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    if backend == "arm64":
        return AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    elif backend == "avx512":
        return AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    elif backend == "avx2":
        return AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    else:
        # Fallback: avx2 config works on most modern x86 CPUs even without AVX2
        return AutoQuantizationConfig.avx2(is_static=False, per_channel=False)


# ---------------------------------------------------------------------------
# Step 1 — Merge LoRA adapter into base model
# ---------------------------------------------------------------------------

def step_merge(
    adapter_path: str,
    base_model_name: str,
    merged_model_dir: Path,
) -> None:
    """
    Load the base model + LoRA adapter, merge weights, save full model.

    Args:
        adapter_path:     Path to the fine-tuned adapter directory
                          (contains adapter_config.json + adapter_model.safetensors)
        base_model_name:  HuggingFace model ID or local path of the base model
        merged_model_dir: Where to save the merged full model
    """
    section("STEP 1 — MERGE LORA ADAPTER INTO BASE MODEL")
    info(f"Adapter   : {adapter_path}")
    info(f"Base model: {base_model_name}")
    info(f"Output    : {merged_model_dir}")

    if merged_model_dir.exists() and (merged_model_dir / "config.json").exists():
        warn("Merged model already exists — skipping merge step")
        warn("Delete inference/merged_model/ to force re-merge")
        return

    merged_model_dir.mkdir(parents=True, exist_ok=True)

    # --- Load tokenizer from adapter dir (has chat_template.jinja etc.) ----
    logger.info("")
    logger.info("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_model_dir))
    ok("Tokenizer saved")

    # --- Load base model in float32 (safe for CPU merge) -------------------
    logger.info("")
    logger.info("  Loading base model (this may download ~1 GB on first run)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,   # float32 required for stable LoRA merge
        device_map="cpu",            # always merge on CPU regardless of GPU
        trust_remote_code=True,
    )
    ok("Base model loaded")

    # --- Attach LoRA adapter ------------------------------------------------
    logger.info("")
    logger.info("  Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    ok("Adapter attached")

    # --- Merge and unload ---------------------------------------------------
    logger.info("")
    logger.info("  Merging adapter weights into base model...")
    model = model.merge_and_unload()
    ok("Weights merged")

    # --- Save merged model --------------------------------------------------
    logger.info("")
    logger.info("  Saving merged model to disk...")
    model.save_pretrained(str(merged_model_dir), safe_serialization=True)
    ok(f"Merged model saved → {merged_model_dir}")

    # --- Verify config.json has model_type ----------------------------------
    config_path = merged_model_dir / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    model_type = cfg.get("model_type", "MISSING")
    ok(f"config.json verified — model_type: {model_type}")

    # Cleanup to free RAM before next step
    del model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("")
    ok("MERGE COMPLETE")


# ---------------------------------------------------------------------------
# Step 2 — Export merged model to ONNX
# ---------------------------------------------------------------------------

def step_onnx_export(
    merged_model_dir: Path,
    onnx_model_dir: Path,
) -> None:
    """
    Export the merged PyTorch model to ONNX format using optimum.

    Args:
        merged_model_dir: Path to the merged full model
        onnx_model_dir:   Where to save the ONNX model files
    """
    section("STEP 2 — EXPORT TO ONNX")
    info(f"Input : {merged_model_dir}")
    info(f"Output: {onnx_model_dir}")

    if not merged_model_dir.exists() or not (merged_model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Merged model not found at {merged_model_dir}. "
            "Run with --steps merge first."
        )

    onnx_model_dir.mkdir(parents=True, exist_ok=True)

    from optimum.onnxruntime import ORTModelForCausalLM

    logger.info("")
    logger.info("  Exporting to ONNX (this may take a few minutes)...")

    try:
        ort_model = ORTModelForCausalLM.from_pretrained(
            str(merged_model_dir),
            export=True,
            use_cache=True,
            library_name="transformers",
            task="text-generation-with-past",
            provider="CPUExecutionProvider",
        )
        ort_model.save_pretrained(str(onnx_model_dir))
        ok("ONNX export successful (primary method)")

    except Exception as e:
        warn(f"Primary export failed: {e}")
        logger.info("  Trying optimum CLI-style export as fallback...")

        from optimum.exporters.onnx import main_export
        main_export(
            model_name_or_path=str(merged_model_dir),
            output=str(onnx_model_dir),
            task="text-generation-with-past",
            library_name="transformers",
        )
        ok("ONNX export successful (fallback method)")

    # Save tokenizer alongside ONNX model
    tokenizer = AutoTokenizer.from_pretrained(str(merged_model_dir))
    tokenizer.save_pretrained(str(onnx_model_dir))
    ok("Tokenizer saved to ONNX directory")

    onnx_files = list(onnx_model_dir.rglob("*.onnx"))
    info(f"ONNX files written: {[f.name for f in onnx_files]}")

    logger.info("")
    ok("ONNX EXPORT COMPLETE")


# ---------------------------------------------------------------------------
# Step 3 — Quantize ONNX model (INT8, platform-aware)
# ---------------------------------------------------------------------------

def step_quantize(
    onnx_model_dir: Path,
    quantized_model_dir: Path,
    merged_model_dir: Path,
) -> dict:
    """
    Apply INT8 dynamic quantization to the ONNX model.

    Args:
        onnx_model_dir:       Path to unquantized ONNX model
        quantized_model_dir:  Where to save quantized ONNX model
        merged_model_dir:     Used only for measuring original model size

    Returns:
        metadata dict with size and compression info
    """
    section("STEP 3 — INT8 QUANTIZATION")
    info(f"Input : {onnx_model_dir}")
    info(f"Output: {quantized_model_dir}")

    if not onnx_model_dir.exists() or not list(onnx_model_dir.rglob("*.onnx")):
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_model_dir}. "
            "Run with --steps onnx first."
        )

    quantized_model_dir.mkdir(parents=True, exist_ok=True)

    from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer

    # --- Detect platform and build config -----------------------------------
    logger.info("")
    logger.info("  Detecting hardware for quantization config...")
    backend = detect_quantization_backend()
    qconfig = build_quantization_config(backend)
    ok(f"Quantization backend: {backend}")

    # --- Load unquantized ONNX model ----------------------------------------
    logger.info("")
    logger.info("  Loading unquantized ONNX model...")
    ort_model = ORTModelForCausalLM.from_pretrained(
        str(onnx_model_dir),
        use_cache=True,
        provider="CPUExecutionProvider",
    )
    ok("ONNX model loaded")

    # --- Quantize -----------------------------------------------------------
    logger.info("")
    logger.info("  Applying dynamic INT8 quantization...")
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    quantizer.quantize(
        save_dir=str(quantized_model_dir),
        quantization_config=qconfig,
    )
    ok("Quantization applied")

    # Copy tokenizer files to quantized dir
    tokenizer = AutoTokenizer.from_pretrained(str(onnx_model_dir), fix_mistral_regex=True)
    tokenizer.save_pretrained(str(quantized_model_dir))
    ok("Tokenizer saved to quantized directory")

    # --- Size comparison ----------------------------------------------------
    logger.info("")
    logger.info("  Measuring model sizes...")

    original_mb = sum(
        f.stat().st_size for f in merged_model_dir.rglob("*.safetensors")
    ) / (1024 ** 2)

    onnx_mb = sum(
        f.stat().st_size for f in onnx_model_dir.rglob("*.onnx")
    ) / (1024 ** 2)

    quantized_mb = sum(
        f.stat().st_size for f in quantized_model_dir.rglob("*.onnx")
    ) / (1024 ** 2)

    compression = original_mb / quantized_mb if quantized_mb > 0 else 0

    info(f"Original merged model : {original_mb:.1f} MB")
    info(f"Unquantized ONNX      : {onnx_mb:.1f} MB")
    info(f"Quantized ONNX (INT8) : {quantized_mb:.1f} MB")
    info(f"Compression ratio     : {compression:.2f}x vs original")

    # --- Save metadata ------------------------------------------------------
    metadata = {
        "base_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "merged_model": str(merged_model_dir),
        "quantization_type": "dynamic_int8",
        "quantization_backend": backend,
        "original_merged_mb": round(original_mb, 2),
        "onnx_unquantized_mb": round(onnx_mb, 2),
        "onnx_quantized_mb": round(quantized_mb, 2),
        "compression_ratio": round(compression, 2),
        "target_device": "CPU",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }

    with open(quantized_model_dir / "quantization_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    ok("Metadata saved → quantization_metadata.json")

    logger.info("")
    ok("QUANTIZATION COMPLETE")
    return metadata


# ---------------------------------------------------------------------------
# Step 4 — Benchmark (optional)
# ---------------------------------------------------------------------------

def step_benchmark(
    merged_model_dir: Path,
    quantized_model_dir: Path,
    num_samples: int = 5,
) -> None:
    """
    Compare inference speed of merged PyTorch model vs quantized ONNX.

    Args:
        merged_model_dir:     Path to merged full PyTorch model
        quantized_model_dir:  Path to quantized ONNX model
        num_samples:          Number of test prompts to run
    """
    section("STEP 4 — INFERENCE BENCHMARK")
    info(f"Samples per model: {num_samples}")

    from optimum.onnxruntime import ORTModelForCausalLM

    test_prompts = [
        "Write a Python function to reverse a string",
        "Implement binary search in Python",
        "Create a bubble sort algorithm",
        "Write a function to check if a number is prime",
        "Implement a stack using Python list",
    ][:num_samples]

    # --- Load tokenizer (shared) -------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(str(merged_model_dir))

    # --- Benchmark PyTorch merged model ------------------------------------
    logger.info("")
    logger.info("  Loading merged PyTorch model...")
    pytorch_model = AutoModelForCausalLM.from_pretrained(
        str(merged_model_dir),
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    pytorch_model.eval()
    ok("PyTorch model loaded")

    logger.info("")
    logger.info(f"  Benchmarking PyTorch model ({num_samples} prompts)...")
    pytorch_times = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        start = time.time()
        with torch.no_grad():
            pytorch_model.generate(**inputs, max_new_tokens=512, do_sample=False)
        pytorch_times.append(time.time() - start)

    avg_pytorch = sum(pytorch_times) / len(pytorch_times)
    ok(f"PyTorch average: {avg_pytorch:.3f}s per prompt")

    del pytorch_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Benchmark quantized ONNX model ------------------------------------
    logger.info("")
    logger.info("  Loading quantized ONNX model...")
    onnx_model = ORTModelForCausalLM.from_pretrained(
        str(quantized_model_dir),
        use_cache=True,
        file_name="model_quantized.onnx", 
        provider="CPUExecutionProvider",
    )
    ok("ONNX model loaded")

    logger.info("")
    logger.info(f"  Benchmarking ONNX model ({num_samples} prompts)...")
    onnx_times = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        start = time.time()
        onnx_model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True)
        onnx_times.append(time.time() - start)

    avg_onnx = sum(onnx_times) / len(onnx_times)
    ok(f"ONNX average: {avg_onnx:.3f}s per prompt")

    # --- Results ------------------------------------------------------------
    speedup = avg_pytorch / avg_onnx if avg_onnx > 0 else 0

    logger.info("")
    logger.info("  ┌─────────────────────────────────────────┐")
    logger.info("  │           BENCHMARK RESULTS             │")
    logger.info("  ├─────────────────────────────────────────┤")
    logger.info(f"  │  PyTorch (merged) : {avg_pytorch:>7.3f}s per prompt  │")
    logger.info(f"  │  ONNX (quantized) : {avg_onnx:>7.3f}s per prompt  │")
    logger.info(f"  │  Speedup          : {speedup:>7.2f}x               │")
    logger.info("  └─────────────────────────────────────────┘")

    # Save benchmark results
    benchmark_results = {
        "pytorch_avg_seconds": round(avg_pytorch, 4),
        "onnx_avg_seconds": round(avg_onnx, 4),
        "speedup_x": round(speedup, 2),
        "num_samples": num_samples,
        "max_new_tokens": 50,
    }
    results_path = quantized_model_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    ok(f"Benchmark results saved → {results_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter → Export ONNX → Quantize INT8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        default="training/outputs/final_model",
        help="Path to LoRA adapter directory (default: training/outputs/final_model)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="Base model HuggingFace ID or local path (default: Qwen/Qwen2.5-Coder-0.5B-Instruct)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference",
        help="Root output directory (default: inference). "
             "Creates merged_model/ and onnx_model/ inside it.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["merge", "onnx", "quantize", "all"],
        default=["all"],
        help=(
            "Which steps to run (default: all). "
            "Options: merge onnx quantize all. "
            "Example: --steps merge onnx"
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark after quantization (compares PyTorch vs ONNX)",
    )
    parser.add_argument(
        "--benchmark_samples",
        type=int,
        default=5,
        help="Number of prompts to use for benchmarking (default: 5)",
    )

    return parser.parse_args()


def resolve_steps(steps: list) -> list:
    """Expand 'all' into the full ordered step list."""
    if "all" in steps:
        return ["merge", "onnx", "quantize"]
    # Preserve order: merge → onnx → quantize
    ordered = ["merge", "onnx", "quantize"]
    return [s for s in ordered if s in steps]


def main():
    args = parse_args()
    steps = resolve_steps(args.steps)

    output_dir     = Path(args.output_dir)
    merged_dir     = output_dir / "merged_model"
    onnx_dir       = output_dir / "onnx_model"
    quantized_dir  = output_dir / "onnx_model" / "quantized"

    section("PIPELINE CONFIG")
    info(f"Adapter path   : {args.adapter_path}")
    info(f"Base model     : {args.base_model}")
    info(f"Output root    : {output_dir}")
    info(f"Merged model   : {merged_dir}")
    info(f"ONNX model     : {onnx_dir}")
    info(f"Quantized model: {quantized_dir}")
    info(f"Steps          : {' → '.join(steps)}")
    info(f"Benchmark      : {args.benchmark}")
    info(f"Platform       : {platform.platform()}")
    info(f"PyTorch        : {torch.__version__}")
    info(f"CUDA available : {torch.cuda.is_available()}")

    pipeline_start = time.time()

    # -----------------------------------------------------------------------
    if "merge" in steps:
        step_merge(
            adapter_path=args.adapter_path,
            base_model_name=args.base_model,
            merged_model_dir=merged_dir,
        )

    if "onnx" in steps:
        step_onnx_export(
            merged_model_dir=merged_dir,
            onnx_model_dir=onnx_dir,
        )

    if "quantize" in steps:
        metadata = step_quantize(
            onnx_model_dir=onnx_dir,
            quantized_model_dir=quantized_dir,
            merged_model_dir=merged_dir,
        )

    if args.benchmark:
        if not quantized_dir.exists():
            warn("Benchmark skipped — quantized model not found. Run --steps quantize first.")
        else:
            step_benchmark(
                merged_model_dir=merged_dir,
                quantized_model_dir=quantized_dir,
                num_samples=args.benchmark_samples,
            )

    # -----------------------------------------------------------------------
    total = time.time() - pipeline_start
    section("PIPELINE COMPLETE")
    info(f"Total time : {total:.1f}s  ({total/60:.1f} min)")
    info(f"Merged model    → {merged_dir}")
    info(f"ONNX model      → {onnx_dir}")
    info(f"Quantized model → {quantized_dir}")

if __name__ == "__main__":
    main()