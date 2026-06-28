"""
Shared utilities for code quality evaluation.

All evaluation scripts use these functions to ensure consistency.
"""

import re
from typing import Dict, Optional
from radon.complexity import cc_visit
import numpy as np


def extract_code_from_response(response: str) -> str:
    """
    Extract code from model response.
    
    Handles:
    - Markdown code blocks (```python ... ```)
    - Code-like content (def, class, import statements)
    - Plain text fallback
    
    Args:
        response: Model generated text
        
    Returns:
        Extracted code string
    """
    # Try markdown code blocks first
    code_block_pattern = r"```(?:python)?\n(.*?)```"
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Try to find code-like content
    lines = response.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        # Detect code start
        if line.strip().startswith(('def ', 'class ', 'import ', 'from ', '@')):
            in_code = True
        
        # Collect code lines
        if in_code:
            code_lines.append(line)
        
        # Stop at empty line after code (heuristic)
        elif in_code and not line.strip():
            # Check if next lines are still code
            continue
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    # Fallback: return entire response
    return response.strip()


def check_syntax(code: str) -> Dict[str, any]:
    """
    Check if code is syntactically valid Python.
    
    Args:
        code: Python code string
        
    Returns:
        Dictionary with 'valid' (bool) and 'error' (str or None)
    """
    try:
        compile(code, '<string>', 'exec')
        return {"valid": True, "error": None}
    except SyntaxError as e:
        return {"valid": False, "error": f"Syntax error: {str(e)}"}
    except Exception as e:
        return {"valid": False, "error": f"Compilation error: {str(e)}"}


def calculate_verbosity_ratio(response: str, code: str) -> float:
    """
    Calculate ratio of explanation text to code.
    
    Higher ratio = more verbose (more explanation relative to code)
    
    Args:
        response: Full model response
        code: Extracted code portion
        
    Returns:
        Verbosity ratio (explanation_chars / code_chars)
    """
    total_chars = len(response)
    code_chars = len(code)
    
    if code_chars == 0:
        return float('inf')
    
    explanation_chars = total_chars - code_chars
    return explanation_chars / code_chars


def has_excess_explanation(
    response: str, 
    code: str, 
    threshold: int = 100
) -> bool:
    """
    Detect if response has excessive explanation.
    
    Args:
        response: Full model response
        code: Extracted code portion
        threshold: Character threshold for excess explanation
        
    Returns:
        True if explanation exceeds threshold
    """
    explanation = response.replace(code, '').strip()
    
    # Remove common short phrases
    explanation = re.sub(
        r'(Here\'s|Here is|This code|The code|```python|```)',
        '',
        explanation
    )
    
    return len(explanation.strip()) > threshold


def calculate_code_to_response_ratio(response: str, code: str) -> float:
    """
    Calculate what percentage of response is actual code.
    
    Args:
        response: Full model response
        code: Extracted code portion
        
    Returns:
        Ratio (0.0 to 1.0) of code to total response
    """
    if len(response) == 0:
        return 0.0
    
    return len(code) / len(response)


def calculate_cyclomatic_complexity(code: str) -> Optional[float]:
    """
    Calculate average cyclomatic complexity of code.
    
    Lower complexity = simpler, more maintainable code.
    
    Args:
        code: Python code string
        
    Returns:
        Average complexity score, or None if calculation fails
    """
    try:
        results = cc_visit(code)
        
        if not results:
            return None
        
        complexities = [block.complexity for block in results]
        return float(np.mean(complexities))
        
    except Exception:
        return None


def calculate_code_metrics(response: str) -> Dict[str, any]:
    """
    Calculate all code quality metrics for a response.
    
    Args:
        response: Model generated response
        
    Returns:
        Dictionary with all metrics
    """
    # Extract code
    code = extract_code_from_response(response)
    
    # Syntax check
    syntax_result = check_syntax(code)
    
    # Calculate metrics
    metrics = {
        "response_length": len(response),
        "code_length": len(code),
        "syntax_valid": syntax_result["valid"],
        "syntax_error": syntax_result["error"],
        "verbosity_ratio": calculate_verbosity_ratio(response, code),
        "has_excess_explanation": has_excess_explanation(response, code),
        "code_to_response_ratio": calculate_code_to_response_ratio(response, code),
        "cyclomatic_complexity": calculate_cyclomatic_complexity(code)
    }
    
    return metrics


def aggregate_metrics(results: list) -> Dict[str, float]:
    """
    Aggregate metrics across multiple samples.
    
    Args:
        results: List of result dictionaries with 'metrics' key
        
    Returns:
        Aggregated statistics
    """
    valid_results = [r for r in results if 'metrics' in r]
    
    if not valid_results:
        return {}
    
    # Extract metric values
    syntax_valid = [r['metrics']['syntax_valid'] for r in valid_results]
    excess_explanation = [r['metrics']['has_excess_explanation'] for r in valid_results]
    code_ratios = [r['metrics']['code_to_response_ratio'] for r in valid_results]
    complexities = [
        r['metrics']['cyclomatic_complexity'] 
        for r in valid_results 
        if r['metrics']['cyclomatic_complexity'] is not None
    ]
    
    # Calculate aggregates
    aggregated = {
        "total_samples": len(valid_results),
        "syntax_validity_pct": (sum(syntax_valid) / len(syntax_valid)) * 100,
        "excess_explanation_pct": (sum(excess_explanation) / len(excess_explanation)) * 100,
        "avg_code_to_response_ratio": np.mean(code_ratios),
        "median_code_to_response_ratio": np.median(code_ratios),
        "avg_cyclomatic_complexity": np.mean(complexities) if complexities else None
    }
    
    return aggregated
