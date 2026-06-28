"""
Simple Product Demo - Python Code Generator

User-facing demo focused on functionality, not metrics.
Clean interface for end users.
"""

import streamlit as st
import requests
from typing import Optional

# Page config
st.set_page_config(
    page_title="Python Code Generator",
    page_icon="🐍",
    layout="centered"
)

# API endpoint
API_URL = "http://localhost:8001"


def call_api(instruction: str, max_tokens: int = 256, temperature: float = 0.7) -> Optional[dict]:
    """
    Call FastAPI backend to generate code.
    
    Args:
        instruction: User instruction
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        API response or None if failed
    """
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={
                "instruction": instruction,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "model_type": "finetuned"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please start the API server first:")
        st.code("uvicorn api.main:app --host 0.0.0.0 --port 8000")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def main():
    """Main app."""
    
    # Header
    st.title("🐍 Python Code Generator")
    st.markdown("Generate Python code from natural language descriptions")
    
    # Sample prompts in sidebar
    with st.sidebar:
        st.header("📚 Example Prompts")
        
        examples = {
            "String Reversal": "Write a function to reverse a string",
            "Binary Search": "Implement binary search algorithm",
            "List Comprehension": "Create a list of squares using list comprehension",
            "File Reading": "Write code to read a CSV file using pandas",
            "API Request": "Make a GET request to an API using requests library"
        }
        
        selected_example = st.selectbox(
            "Choose an example:",
            [""] + list(examples.keys())
        )
        
        if selected_example:
            if st.button("📝 Use This Example"):
                st.session_state['instruction'] = examples[selected_example]
    
    # Main interface
    st.markdown("### ✍️ Describe what you want")
    
    instruction = st.text_area(
        "Enter your instruction:",
        value=st.session_state.get('instruction', ''),
        height=100,
        placeholder="Example: Write a function to calculate factorial",
        key="instruction_input"
    )
    
    # Advanced options (collapsed)
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_tokens = st.slider(
                "Max Length",
                min_value=50,
                max_value=512,
                value=256,
                step=50
            )
        
        with col2:
            temperature = st.slider(
                "Creativity",
                min_value=0.1,
                max_value=1.5,
                value=0.7,
                step=0.1,
                help="Higher = more creative, Lower = more deterministic"
            )
    
    # Generate button
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        generate_button = st.button("🚀 Generate Code", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state['instruction'] = ''
            st.rerun()
    
    # Generation
    if generate_button and instruction.strip():
        with st.spinner("Generating code..."):
            result = call_api(instruction, max_tokens, temperature)
            
            if result:
                st.markdown("### ✅ Generated Code")
                
                # Display code
                st.code(result['generated_code'], language='python')
                
                # Stats (minimal, non-intrusive)
                with st.expander("ℹ️ Generation Info"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Tokens", result['tokens_generated'])
                    with col2:
                        st.metric("Time", f"{result['inference_time_ms']:.0f}ms")
    
    elif generate_button:
        st.warning("⚠️ Please enter an instruction first")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Powered by Qwen2.5-Coder Fine-tuned | "
        "<a href='/developer'>Developer Dashboard</a>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
