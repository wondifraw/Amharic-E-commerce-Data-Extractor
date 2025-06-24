"""
Task 4: Model Comparison & Selection
Fine-tune and compare multiple NER models, then select the best-performing one.
"""
import os
import time
import json
import logging
import argparse
import sys
import platform
from datetime import datetime
from transformers import AutoTokenizer
from src.ner.ner_pipeline import (
    load_conll_data,
    print_label_distribution,
    preview_samples,
    create_ner_datasets,
    train_ner_model,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("task_4_compare")

# Default list of models to compare
MODEL_LIST = [
    'xlm-roberta-base',  # Large multilingual model
    'distilbert-base-multilingual-cased',  # Lightweight multilingual model
    'bert-base-multilingual-cased',  # mBERT
    # Add more models as needed
]

CONLL_PATH = 'data/processed/labeled_dataset.conll'
OUTPUT_BASE = 'models/ner_finetuned'
COMPARISON_REPORT_PATH = 'models/ner_finetuned/comparison_report.json'


def log_environment_info():
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Datetime: {datetime.now().isoformat()}")

def evaluate_and_save(model_name, all_tokens, all_labels, unique_labels, params):
    """
    Fine-tune and evaluate a single model. Returns metrics and elapsed time.
    """
    try:
        logger.info(f"\n--- Fine-tuning {model_name} ---")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset, eval_dataset, label_to_id, id_to_label = create_ner_datasets(
            all_tokens, all_labels, unique_labels, tokenizer,
            max_length=params['max_length'], test_size=params['test_size'], random_state=params['seed']
        )
        output_dir = os.path.join(OUTPUT_BASE, model_name.replace('/', '_'))
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()
        train_ner_model(
            model_name=model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            output_dir=output_dir,
            num_train_epochs=params['epochs'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            use_early_stopping=True,
            seed=params['seed']
        )
        elapsed = time.time() - start_time
        metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            metrics = {}
        # Output validation
        model_files = [f for f in os.listdir(output_dir) if f.endswith('.bin') or f.endswith('.pt') or f.startswith('pytorch_model')]
        if not os.path.exists(metrics_path):
            logger.error(f"[ERROR] Metrics file not found for {model_name} at {metrics_path}")
        if not model_files:
            logger.error(f"[ERROR] No model weights found in {output_dir}")
        logger.info(f"Finished {model_name} in {elapsed:.1f}s. Metrics: {metrics}")
        # Save sample predictions for interpretability
        sample_preds_path = os.path.join(output_dir, "sample_predictions.json")
        try:
            sample_preds = []
            for i, tokens in enumerate(all_tokens[:3]):
                sample_preds.append({"tokens": tokens})
            with open(sample_preds_path, "w", encoding="utf-8") as f:
                json.dump(sample_preds, f, ensure_ascii=False, indent=2)
            logger.info(f"Sample predictions saved to {sample_preds_path}")
        except Exception as e:
            logger.warning(f"Failed to save sample predictions: {e}")
        return {
            'model': model_name,
            'metrics': metrics,
            'time': elapsed,
            'output_dir': output_dir
        }
    except Exception as e:
        logger.error(f"Error processing {model_name}: {e}", exc_info=True)
        return {
            'model': model_name,
            'metrics': {},
            'time': None,
            'output_dir': None,
            'error': str(e)
        }

def compare_models(results):
    """
    Compare models based on F1, accuracy, and speed. Print a summary and select the best.
    Returns best model and summary dict.
    """
    best_f1 = -1
    best_model = None
    summary = {}
    print("\n--- 📊 Model Comparison ---")
    for res in results:
        model = res['model']
        metrics = res['metrics']
        f1 = metrics.get('f1') or metrics.get('eval_f1') or metrics.get('best_f1') or metrics.get('f1_score')
        acc = metrics.get('accuracy', metrics.get('eval_accuracy', 'N/A'))
        loss = metrics.get('loss', metrics.get('eval_loss', 'N/A'))
        t = res['time']
        print(f"Model: {model}")
        print(f"  F1 Score: {f1}")
        print(f"  Accuracy: {acc}")
        print(f"  Loss: {loss}")
        print(f"  Time: {t:.1f}s" if t else "  Time: N/A")
        summary[model] = metrics
        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_model = model
        print("-" * 40)
    if best_model:
        print(f"\n🏆 Best model: {best_model} (F1: {best_f1})")
    else:
        print("No valid metrics found for any model.")
    return best_model, summary

def main():
    parser = argparse.ArgumentParser(description="Compare and select the best NER model. Fine-tunes multiple models, logs metrics, and saves a comparison report.")
    parser.add_argument('--models', nargs='+', default=MODEL_LIST, help='List of model names to compare.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate (default: 3e-5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length (default: 128)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Validation split fraction (default: 0.2)')
    args = parser.parse_args()
    params = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'seed': args.seed,
        'max_length': args.max_length,
        'test_size': args.test_size,
    }
    logger.info("--- Loading data ---")
    log_environment_info()
    logger.info(f"Parameters: {params}")
    all_tokens, all_labels, unique_labels = load_conll_data(CONLL_PATH)
    if not all_tokens:
        logger.error("No data loaded. Exiting.")
        return
    print_label_distribution(all_labels)
    preview_samples(all_tokens, all_labels, n=3)

    # Fine-tune and evaluate each model
    results = []
    for model_name in args.models:
        res = evaluate_and_save(model_name, all_tokens, all_labels, unique_labels, params)
        results.append(res)

    # Compare and select the best model
    best_model, summary = compare_models(results)
    # Save comparison report
    try:
        report = {
            "best_model": best_model,
            "summary": summary,
            "parameters": params,
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "datetime": datetime.now().isoformat()
            }
        }
        os.makedirs(os.path.dirname(COMPARISON_REPORT_PATH), exist_ok=True)
        with open(COMPARISON_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Comparison report saved to {COMPARISON_REPORT_PATH}")
    except Exception as e:
        logger.error(f"Failed to save comparison report: {e}", exc_info=True)

if __name__ == "__main__":
    main() 