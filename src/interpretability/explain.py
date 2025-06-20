"""
explain.py
Module for generating model interpretability explanations using SHAP and LIME.
"""
import shap
import numpy as np
from typing import List, Any
from transformers import pipeline

def explain_with_shap(model: Any, tokenizer: Any, sentences: List[str]) -> Any:
    """
    Generates SHAP explanations for a given set of sentences.

    This function wraps the model and tokenizer in a Hugging Face pipeline and
    uses the SHAP Explainer to understand token importance.

    Args:
        model: A fine-tuned Hugging Face model for token classification.
        tokenizer: The tokenizer corresponding to the model.
        sentences: A list of sentences (strings) to explain.

    Returns:
        A SHAP explanation object that can be used for plotting or analysis.
        Returns None if an error occurs.
    """
    print("--- Generating SHAP Explanations ---")
    try:
        # Create a token classification pipeline
        ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer)
        
        # Create a SHAP explainer
        explainer = shap.Explainer(ner_pipeline)
        
        # Get SHAP values
        shap_values = explainer(sentences)
        
        print("SHAP explanations generated successfully.")
        return shap_values
    except Exception as e:
        print(f"[ERROR] Failed to generate SHAP explanations: {e}")
        return None

# Note: LIME is often challenging to apply directly to transformer token-level tasks
# due to its reliance on perturbing interpretable inputs (like whole words), which
# doesn't align well with subword tokenization. A robust implementation would be
# complex. For this project, we will focus on SHAP, which is better suited for
# this type of model and task. If LIME is a strict requirement, a more specialized
# implementation would be needed. 