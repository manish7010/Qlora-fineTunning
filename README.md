# Qwen2.5-Coder Python Fine-tuning

Fine-tuning [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) on a Python code generation dataset using QLoRA. The fine-tuned model is available on Hugging Face at [`KpRT/qwen-python-finetuned`](https://huggingface.co/KpRT/qwen-python-finetuned).

---

## Results

| Metric | Base Model | Fine-tuned Model |
|---|---|---|
| Syntax Validity | 96.8% | 95.2 |
| CodeBLEU | 0.298 | 0.392 |
| Avg Code Ratio | 0.43 | 0.92 |
| Excess Explanation | 96.1% | 0.5%  |
| Pass@1 (HumanEval) | 22.0% | 54.4% |
| Pass@3 (HumanEval) | 48.2% | 71.4% |
| Pass@5 (HumanEval) | 62.0% | 78.0% |
| LLM Judge Overall | 6.25 / 10 | 6.68 / 10 |

> The primary objective of fine-tuning was to eliminate verbose explanations from model outputs and produce clean, code-only responses. This goal was achieved — excess explanation dropped from 96.1% in the base model to just 0.5% after fine-tuning, and avg code ratio improved from 0.43 to 0.92, meaning the model now outputs mostly code with minimal prose. Beyond the core objective, the fine-tuned model also outperforms the base across all functional benchmarks: Pass@1 improved from 22.0% to 54.4%, Pass@3 from 48.2% to 71.4%, Pass@5 from 62.0% to 78.0%, and LLM Judge overall score from 6.25 to 6.68. Evaluation plots are available in `plots/`. For a detailed breakdown of how to interpret each metric, see [`evaluation_advanced/results/README.md`](evaluation_advanced/results/README.md).

---

## Project Structure

```
qwen-python-finetuning/
├── evaluation/                  # Pre-training evidence phase
│   ├── evaluate_base_model.py
│   ├── test_prompt_engineering.py
│   └── analyze_results.py
├── config/
│   └── config.py     
├── data/
│   └── prepare_dataset.py       # Train / val / test split
├── api/
│   └── main.py      
├── training/
│   ├── train_qlora.py           # QLoRA fine-tuning
│   ├── callbacks.py             # Validation callbacks
│   └── config.yaml              # Hyperparameters
├── evaluation_advanced/
│   ├── eval_comprehensive.py    # 14-metric evaluation
│   ├── eval_passk.py            # HumanEval pass@k
│   └── eval_llm_judge.py        # LLM-as-judge scoring
├── inference/
│   └── merge_export_quantize_onnx.py
├── scripts/
│   ├── model_evaluation_analysis.ipynb
│   └── upload_to_hf.py
├── utils/
│   ├── code_metrics.py
│   ├── io_utils.py
│   └── logger.py
├── demo/
│   ├── streamlit_product.py
│   └── streamlit_developer.py
├── requirements.txt
└── README.md
```

---

## Quickstart — Use the Fine-tuned Model

If you just want to run inference, pull the model from Hugging Face directly — no training required.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "KpRT/qwen-python-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

prompt = "Write a Python function that checks if a number is prime."
modified_prompt = f"Instruction: {prompt}\nOutput:\n"
inputs = tokenizer(modified_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Reproducing from Scratch

Follow these steps in order.

### 1. Clone the Repository

```bash
git clone https://github.com/KRT2002/qwen-python-finetuning.git
cd qwen-python-finetuning
```

### 2. Set Up the Environment

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 3. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
HF_TOKEN=your_huggingface_token_here
```

- **GROQ_API_KEY** — required only for `eval_llm_judge.py`. Get one at [console.groq.com](https://console.groq.com).
- **GROQ_MODEL** — the Groq model used as judge. Default: `llama-3.3-70b-versatile`.
- **HF_TOKEN** — required to push a fine-tuned model to the Hugging Face Hub. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 4. Prepare the Dataset

```bash
cd data
python prepare_dataset.py
```

This splits the source dataset into `train.json` (16k samples), `validation.json` (1k), and `test.json` (1k) under `data/datasets/`.

### 5. (Optional) Evaluate the Base Model

Run this before training to establish baseline metrics and validate your prompt strategy.

```bash
cd evaluation
python evaluate_base_model.py
python test_prompt_engineering.py
python analyze_results.py
```

Results are written to `evaluation/results/`.

### 6. Train

```bash
cd training
python train_qlora.py
```

Training uses QLoRA (4-bit quantization + LoRA adapters). Checkpoints are saved to `training/outputs/`. A T4 GPU takes roughly 3–4 hours.

To track training with Weights & Biases:

```bash
wandb login
# then run train_qlora.py as above
```

Hyperparameters are in `training/config.yaml`.

### 7. Evaluate the Fine-tuned Model

```bash
cd evaluation_advanced

# Comprehensive metrics (syntax, CodeBLEU, code ratio, complexity, etc.)
python eval_comprehensive.py \
  --model_path ../training/outputs/final_model \
  --test_data ../data/datasets/test.json \
  --output results/finetuned_model_results.json

# HumanEval pass@k
python eval_passk.py \
  --model_path ../training/outputs/final_model \
  --output results/pass_at_k_results.json \
  --num_problems 50 \
  --k_values 1 10

# LLM-as-judge (requires GROQ_API_KEY in .env)
python eval_llm_judge.py \
  --results_file results/finetuned_model_results.json \
  --output results/llm_judge_results.json \
  --num_samples 100
```

See [`evaluation_advanced/results/README.md`](evaluation_advanced/results/README.md) for full documentation of all flags, output file schemas, and metric reference.

### 8. (Optional) Export to ONNX

```bash
cd inference
python merge_export_quantize_onnx.py
```

### 9. Run the Demo

```bash
cd demo

# Product-facing demo
streamlit run streamlit_product.py

# Developer/metrics view
streamlit run streamlit_developer.py
```

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | Qwen2.5-Coder-0.5B-Instruct |
| Quantization | 4-bit (bitsandbytes NF4) |
| LoRA rank | 64 |
| LoRA alpha | 16 |
| LoRA target modules | q_proj, v_proj |
| Learning rate | 2e-4 |
| Batch size | 4 (grad accum 4) |
| Epochs | 3 |
| Optimizer | paged_adamw_32bit |

---

## Requirements

- Python 3.10+
- CUDA-capable GPU with at least 16 GB VRAM for training (inference works on smaller GPUs)
- See `requirements.txt` for full dependency list

---

## License

This project is released under the MIT License. The base model is subject to [Qwen's model license](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/blob/main/LICENSE).