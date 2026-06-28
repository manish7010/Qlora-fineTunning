"""
FastAPI backend for Python code generation.

Provides REST API endpoints for:
- Code generation
- Model information
- Health checks

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from contextlib import asynccontextmanager

from utils.logger import get_logger
from utils.io_utils import measure_memory_usage

logger = get_logger(__name__)

# Global model storage
models = {}


class CodeRequest(BaseModel):
    """Request schema for code generation."""
    
    instruction: str = Field(
        ...,
        description="Natural language instruction for code generation",
        example="Write a function to reverse a string"
    )
    max_tokens: int = Field(
        default=256,
        ge=50,
        le=512,
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.1,
        le=2.0,
        description="Sampling temperature"
    )
    model_type: str = Field(
        default="finetuned",
        description="Model to use: 'base', 'finetuned', or 'onnx'"
    )


class CodeResponse(BaseModel):
    """Response schema for code generation."""
    
    generated_code: str = Field(..., description="Generated Python code")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    model_used: str = Field(..., description="Model type used")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")


class ModelInfo(BaseModel):
    """Model information schema."""
    
    model_name: str
    model_type: str
    parameters: str
    loaded: bool


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    models_loaded: Dict[str, bool]
    memory_usage: Dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models on startup, cleanup on shutdown.
    """
    logger.info("Starting API server...")
    
    # Load fine-tuned model
    logger.info("Loading fine-tuned model...")
    try:
        # Replace with your HuggingFace repo
        model_name = "KpRT/qwen-python-finetuned"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        models["finetuned"] = {
            "model": model,
            "tokenizer": tokenizer,
            "name": model_name
        }
        
        logger.info("✓ Fine-tuned model loaded")
        
    except Exception as e:
        logger.warning(f"Could not load fine-tuned model: {str(e)}")
        logger.info("Falling back to base model")
        
        # Fallback to base model
        base_model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        models["finetuned"] = {
            "model": model,
            "tokenizer": tokenizer,
            "name": base_model_name
        }
    
    logger.info("API server ready")
    
    yield
    
    # Cleanup
    logger.info("Shutting down API server...")
    models.clear()


# Initialize FastAPI app
app = FastAPI(
    title="Python Code Generator API",
    description="REST API for generating Python code from natural language",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Python Code Generator API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns server status and model availability.
    """
    memory = measure_memory_usage()
    
    return HealthResponse(
        status="healthy",
        models_loaded={
            model_type: model_data is not None
            for model_type, model_data in models.items()
        },
        memory_usage=memory
    )


@app.get("/models", response_model=Dict[str, ModelInfo])
async def list_models():
    """
    List available models.
    """
    model_info = {}
    
    for model_type, model_data in models.items():
        if model_data:
            model_info[model_type] = ModelInfo(
                model_name=model_data["name"],
                model_type=model_type,
                parameters="500M",
                loaded=True
            )
    
    return model_info


@app.post("/generate", response_model=CodeResponse)
async def generate_code(request: CodeRequest):
    """
    Generate Python code from natural language instruction.
    
    Args:
        request: Code generation request
        
    Returns:
        Generated code and metadata
        
    Raises:
        HTTPException: If model not available or generation fails
    """
    # Get model
    model_type = request.model_type
    
    if model_type not in models or models[model_type] is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_type}' not available. Available models: {list(models.keys())}"
        )
    
    model_data = models[model_type]
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]
    
    try:
        # Measure memory before
        mem_before = measure_memory_usage()
        
        # Reformat instruction
        formatted_instruction = f"Instruction: {request.instruction}\nOutput:\n"
        
        # Tokenize input
        inputs = tokenizer(
            formatted_instruction,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=request.temperature,
                pad_token_id=tokenizer.pad_token_id
            )
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_code = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Measure memory after
        mem_after = measure_memory_usage()
        memory_delta = mem_after.get("system_ram_used_mb", 0) - mem_before.get("system_ram_used_mb", 0)
        
        # Prepare response
        response = CodeResponse(
            generated_code=generated_code,
            tokens_generated=len(generated_ids),
            inference_time_ms=round(inference_time, 2),
            model_used=model_type,
            memory_usage_mb=round(memory_delta, 2)
        )
        
        logger.info(
            f"Generated {len(generated_ids)} tokens in {inference_time:.2f}ms "
            f"({len(generated_ids)/(inference_time/1000):.2f} tokens/sec)"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Code generation failed: {str(e)}"
        )


@app.post("/generate/batch")
async def generate_code_batch(requests: list[CodeRequest]):
    """
    Generate code for multiple instructions.
    
    Args:
        requests: List of code generation requests
        
    Returns:
        List of generated code responses
    """
    results = []
    
    for req in requests:
        try:
            result = await generate_code(req)
            results.append(result)
        except HTTPException as e:
            results.append({
                "error": e.detail,
                "instruction": req.instruction
            })
    
    return results


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )