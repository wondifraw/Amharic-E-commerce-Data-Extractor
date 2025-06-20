"""
task_4_compare.py
Main script to fine-tune and compare multiple NER models.

This script automates the process of:
1.  Training several NER models on the same dataset.
2.  Evaluating their performance using standard metrics.
3.  Saving the results and identifying the best-performing model.
"""
import os
import json
import argparse
from transformers import AutoTokenizer
from src.ner.dataset import load_conll_data, create_ner_datasets
from src.ner.train import train_ner_model

# --- Configuration ---
CONLL_PATH = 'data/processed/labeled_dataset.conll'
MODELS_TO_COMPARE = [
    'xlm-roberta-base',
    'distilbert-base-multilingual-cased',
    'Davlan/bert-tiny-amharic',
]
LABEL_LIST = ['O', 'B-Product', 'I-Product', 'B-LOC', 'I-LOC', 'B-PRICE', 'I-PRICE']
COMPARISON_OUTPUT_DIR = 'models/comparison_results'

def main(models_to_compare: list):
    """
    Main function to execute the model comparison pipeline.
    """
    print("--- Starting NER Model Comparison Pipeline ---")
    
    # 1. Load labeled data
    print(f"Loading data from {CONLL_PATH}...")
    all_tokens, all_labels = load_conll_data(CONLL_PATH)
    if not all_tokens:
        print("[ERROR] No data was loaded. Please run Task 2 to create a labeled dataset.")
        return

    all_metrics = {}

    # 2. Train and evaluate each model
    for model_name in models_to_compare:
        print(f"\n--- Processing Model: {model_name} ---")
        model_output_dir = os.path.join(COMPARISON_OUTPUT_DIR, model_name.replace("/", "_"))
        
        # Load tokenizer
        print(f"Loading tokenizer for '{model_name}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"[ERROR] Could not load tokenizer for '{model_name}'. Skipping. Error: {e}")
            continue

        # Create datasets (using the same split logic ensures fair comparison)
        print("Creating and preprocessing datasets...")
        train_dataset, eval_dataset, label_to_id, id_to_label = create_ner_datasets(
            all_tokens, all_labels, LABEL_LIST, tokenizer
        )
        
        # Train the model
        train_ner_model(
            model_name,
            train_dataset,
            eval_dataset,
            label_to_id,
            id_to_label,
            model_output_dir,
        )
        
        # Load and store evaluation metrics
        metrics_path = os.path.join(model_output_dir, "evaluation_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                all_metrics[model_name] = json.load(f)

    # 3. Compare results and find the best model
    if not all_metrics:
        print("\n[ERROR] No models were successfully trained and evaluated. Cannot compare.")
        return

    print("\n--- Model Comparison Results ---")
    for model_name, metrics in all_metrics.items():
        f1_score = metrics.get('eval_f1', 0)
        accuracy = metrics.get('eval_accuracy', 0)
        print(f"Model: {model_name:<40} | F1-Score: {f1_score:.4f} | Accuracy: {accuracy:.4f}")

    best_model_name = max(all_metrics, key=lambda m: all_metrics[m].get("eval_f1", 0))
    best_model_metrics = all_metrics[best_model_name]

    print(f"\nBest performing model (by F1-score): {best_model_name}")

    # 4. Save comparison summary
    summary = {
        "best_model": {
            "name": best_model_name,
            "metrics": best_model_metrics
        },
        "all_results": all_metrics
    }
    
    summary_path = os.path.join(COMPARISON_OUTPUT_DIR, "comparison_summary.json")
    os.makedirs(COMPARISON_OUTPUT_DIR, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nComparison summary saved to: {summary_path}")
    print("--- Model Comparison Pipeline Finished Successfully! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune and compare multiple NER models.")
    parser.add_argument(
        "--models",
        nargs='+',
        default=MODELS_TO_COMPARE,
        help="A list of pretrained model names to compare (from Hugging Face Hub)."
    )
    args = parser.parse_args()
    
    main(args.models) 