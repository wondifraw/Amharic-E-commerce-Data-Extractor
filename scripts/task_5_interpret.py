"""
task_5_interpret.py
Main script to generate and save interpretability reports for the best NER model.

This script performs the following steps:
1.  Identifies the best-performing model from the comparison results.
2.  Loads the fine-tuned model and its tokenizer.
3.  Generates SHAP explanations for a set of sample sentences.
4.  Saves the SHAP visualizations to a directory for analysis.
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForTokenClassification, AutoTokenizer
from src.interpretability.explain import explain_with_shap

# --- Configuration ---
COMPARISON_SUMMARY_PATH = 'models/comparison_results/comparison_summary.json'
INTERPRETABILITY_REPORTS_DIR = 'reports/interpretability'
SAMPLE_SENTENCES = [
    "Iphone 13 ዋጋ 85,000 ብር",
    "አድራሻ: አዲስ አበባ, ቦሌ",
    "ሹራብ በ 500 ብር ብቻ",
    "ላፕቶፕ እና ስልክ መሸጫ"
]

def main(summary_path: str):
    """
    Main function to execute the model interpretability pipeline.
    """
    print("--- Starting Model Interpretability Pipeline ---")
    
    # 1. Find the best model from the comparison summary
    if not os.path.exists(summary_path):
        print(f"[ERROR] Comparison summary not found at '{summary_path}'. Please run Task 4 first.")
        return
        
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        best_model_name = summary_data.get("best_model", {}).get("name")
        if not best_model_name:
            print("[ERROR] Could not determine the best model from the summary file.")
            return
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[ERROR] Failed to read or parse the summary file '{summary_path}': {e}")
        return

    model_dir = os.path.join('models/comparison_results', best_model_name.replace("/", "_"))
    print(f"Loading best model '{best_model_name}' from directory: {model_dir}")

    # 2. Load the model and tokenizer
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception as e:
        print(f"[ERROR] Failed to load the model or tokenizer from '{model_dir}': {e}")
        return

    # 3. Generate SHAP explanations
    shap_values = explain_with_shap(model, tokenizer, SAMPLE_SENTENCES)
    
    if shap_values is None:
        print("[ERROR] SHAP explanation failed. Aborting report generation.")
        return

    # 4. Save SHAP plots to files
    os.makedirs(INTERPRETABILITY_REPORTS_DIR, exist_ok=True)
    print(f"Saving SHAP plots to '{INTERPRETABILITY_REPORTS_DIR}'...")
    
    try:
        for i, sentence in enumerate(SAMPLE_SENTENCES):
            # Generate a plot for each sentence
            shap.plots.text(shap_values[i], show=False)
            
            # Save the current plot to a file
            plot_path = os.path.join(INTERPRETABILITY_REPORTS_DIR, f"shap_explanation_sentence_{i+1}.png")
            plt.savefig(plot_path, bbox_inches='tight', dpi=150)
            plt.close() # Close the plot to avoid displaying it in a headless environment
            print(f"Saved plot: {plot_path}")
            
    except Exception as e:
        print(f"[ERROR] Failed to save SHAP plots: {e}")

    print("\n--- Model Interpretability Pipeline Finished Successfully! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interpretability reports for the best NER model.")
    parser.add_argument(
        "--summary_path",
        type=str,
        default=COMPARISON_SUMMARY_PATH,
        help="Path to the model comparison summary JSON file."
    )
    args = parser.parse_args()
    main(args.summary_path) 