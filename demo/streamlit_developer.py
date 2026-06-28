"""
Developer Dashboard - Model Comparison & Evaluation

For developers and interviews. Shows:
- Base vs Fine-tuned vs ONNX comparison
- Performance metrics (inference time, throughput, memory)
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import time

from utils.logger import get_logger
from utils.io_utils import measure_memory_usage
from utils.code_metrics import calculate_code_metrics

logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="Developer Dashboard - Model Comparison",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_models():
    """Load all three models for comparison."""
    
    models = {}
    
    # Base model
    with st.spinner("Loading base model..."):
        base_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        models["base"] = {
            "model": AutoModelForCausalLM.from_pretrained(
                base_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            ),
            "tokenizer": AutoTokenizer.from_pretrained(base_path),
            "type": "PyTorch FP16/32",
            "path": base_path
        }
    
    # Fine-tuned model
    with st.spinner("Loading fine-tuned model..."):
        # Replace with your HuggingFace repo
        ft_path = "KpRT/qwen-python-finetuned"
        try:
            models["finetuned"] = {
                "model": AutoModelForCausalLM.from_pretrained(
                    ft_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                ),
                "tokenizer": AutoTokenizer.from_pretrained(ft_path),
                "type": "PyTorch FP16/32 (Fine-tuned)",
                "path": ft_path
            }
        except:
            st.warning("Could not load fine-tuned model from HuggingFace. Using local fallback.")
            ft_path = "inference/merged_model"
            models["finetuned"] = {
                "model": AutoModelForCausalLM.from_pretrained(
                    ft_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                ),
                "tokenizer": AutoTokenizer.from_pretrained(ft_path),
                "type": "PyTorch FP16/32 (Fine-tuned)",
                "path": ft_path
            }
    
    # ONNX model
    with st.spinner("Loading onnx model..."):
        onnx_path = "inference/onnx_model"
        try:
            models["onnx"] = {
                "model": ORTModelForCausalLM.from_pretrained(
                    onnx_path,
                    use_cache=True,
                    provider="CPUExecutionProvider"
                ),
                "tokenizer": AutoTokenizer.from_pretrained(onnx_path),
                "type": "ONNX",
                "path": onnx_path
            }
        except:
            st.info("ONNX model not found.")
            models["onnx"] = None
    
    return models

def generate_with_metrics(model, tokenizer, prompt, model_path, max_tokens=256):
    """
    Generate code and measure performance metrics.
    
    Returns:
        Tuple of (generated_code, metrics_dict)
    """
    
    # Prompt formatting based on model
    if model_path == "Qwen/Qwen2.5-Coder-0.5B-Instruct":
        modified_prompt = (
            f"Task: {prompt}\n\n"
            f"Example format:\n"
            f"def example():\n"
            f"    return 'code only'\n\n"
            f"Now write the code:"
        )
    else:
        modified_prompt = f"Instruction: {prompt}\nOutput:\n"
    
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
    
    # Measure memory before
    mem_before = measure_memory_usage()
    
    # Tokenize
    inputs = tokenizer(modified_prompt, return_tensors="pt").to(device)
    
    # Generate with timing
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
    
    inference_time = time.time() - start_time
    
    # Decode
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Measure memory after
    mem_after = measure_memory_usage()
    
    # Calculate metrics
    num_tokens = len(generated_ids)
    throughput = num_tokens / inference_time if inference_time > 0 else 0
    
    metrics = {
        "inference_time_sec": inference_time,
        "tokens_generated": num_tokens,
        "throughput_tokens_per_sec": throughput,
        "memory_before_mb": mem_before.get("system_ram_used_mb", 0),
        "memory_after_mb": mem_after.get("system_ram_used_mb", 0),
        "memory_delta_mb": mem_after.get("system_ram_used_mb", 0) - mem_before.get("system_ram_used_mb", 0)
    }
    
    if "gpu_allocated_mb" in mem_after:
        metrics["gpu_memory_mb"] = mem_after["gpu_allocated_mb"]
    
    return generated_code, metrics


def main():
    """Main dashboard app."""
    
    st.title("📊 Developer Dashboard - Model Comparison")
    st.markdown("Compare Base, Fine-tuned, and ONNX models")
    
    # Load models
    models = load_models()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        max_tokens = st.slider("Max Tokens", 50, 512, 256)
        
        st.markdown("---")
        
        st.header("📝 Sample Prompts")
        samples = {
            "PyTorch DataLoader": "Create a PyTorch DataLoader for image classification",
            "K-Fold CV": "Implement k-fold cross-validation using sklearn",
            "Confusion Matrix": "Plot confusion matrix with matplotlib"
        }
        
        selected_sample = st.selectbox("Choose sample:", [""] + list(samples.keys()))
        
        if selected_sample and st.button("Load Sample"):
            st.session_state['prompt'] = samples[selected_sample]
    

    st.header("Live Model Comparison")
    
    prompt = st.text_area(
        "Enter instruction:",
        value=st.session_state.get('prompt', ''),
        height=100,
        key="prompt_input"
    )
    
    if st.button("🚀 Generate from All Models", type="primary"):
        if not prompt.strip():
            st.warning("Please enter a prompt")
        else:
            # Generate from each model
            results = {}
            
            for model_name, model_data in models.items():
                if model_data is None:
                    continue
                
                with st.spinner(f"Generating with {model_name} model..."):
                    code, metrics = generate_with_metrics(
                        model_data["model"],
                        model_data["tokenizer"],
                        prompt,
                        model_data["path"],
                        max_tokens
                    )
                    
                    results[model_name] = {
                        "code": code,
                        "metrics": metrics,
                        "type": model_data["type"]
                    }
            
            # Display comparison
            st.markdown("### 📊 Performance Comparison")
            
            # Metrics table
            metrics_df = pd.DataFrame({
                "Model": [name.capitalize() for name in results.keys()],
                "Inference Time (s)": [r["metrics"]["inference_time_sec"] for r in results.values()],
                "Throughput (tok/s)": [r["metrics"]["throughput_tokens_per_sec"] for r in results.values()],
                "Tokens Generated": [r["metrics"]["tokens_generated"] for r in results.values()],
                "Memory (MB)": [r["metrics"].get("memory_delta_mb", 0) for r in results.values()]
            })
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Inference time comparison
                fig1 = go.Figure(data=[
                    go.Bar(
                        x=list(results.keys()),
                        y=[r["metrics"]["inference_time_sec"] for r in results.values()],
                        marker_color=['lightblue', 'lightgreen', 'lightyellow'][:len(results)]
                    )
                ])
                fig1.update_layout(
                    title="Inference Time (Lower is Better)",
                    yaxis_title="Seconds",
                    height=300
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Throughput comparison
                fig2 = go.Figure(data=[
                    go.Bar(
                        x=list(results.keys()),
                        y=[r["metrics"]["throughput_tokens_per_sec"] for r in results.values()],
                        marker_color=['lightcoral', 'lightgreen', 'lightyellow'][:len(results)]
                    )
                ])
                fig2.update_layout(
                    title="Throughput (Higher is Better)",
                    yaxis_title="Tokens/Second",
                    height=300
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Code outputs
            st.markdown("### 📝 Generated Code")
            
            cols = st.columns(len(results))
            for idx, (model_name, result) in enumerate(results.items()):
                with cols[idx]:
                    st.markdown(f"**{model_name.capitalize()}**")
                    st.code(result["code"], language='python')
                    
                    # Code quality metrics
                    code_metrics = calculate_code_metrics(result["code"])
                    st.caption(f"Syntax Valid: {'✅' if code_metrics['syntax_valid'] else '❌'}")

if __name__ == "__main__":
    main()