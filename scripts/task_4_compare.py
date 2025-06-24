"""
Task 4: Model Comparison & Selection
Fine-tune and compare multiple NER models, then select the best-performing one.
"""
import os
import time
import json
import logging
from transformers import AutoTokenizer
from src.ner.ner_pipeline import (
    load_conll_data,
    print_label_distribution,
    preview_samples,
    create_ner_datasets,
    train_ner_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task_4_compare")

# List of models to compare
MODEL_LIST = [
    'xlm-roberta-base',  # Large multilingual model
    'distilbert-base-multilingual-cased',  # Lightweight multilingual model
    'bert-base-multilingual-cased',  # mBERT
    # Add more models as needed
]

CONLL_PATH = 'data/processed/labeled_dataset.conll'
OUTPUT_BASE = 'models/ner_finetuned'


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
        logger.info(f"Finished {model_name} in {elapsed:.1f}s. Metrics: {metrics}")
        return {
            'model': model_name,
            'metrics': metrics,
            'time': elapsed,
            'output_dir': output_dir
        }
    except Exception as e:
        logger.error(f"Error processing {model_name}: {e}")
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
    """
    best_f1 = -1
    best_model = None
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
        if f1 is not None and f1 > best_f1:
            best_f1 = f1
            best_model = model
        print("-" * 40)
    if best_model:
        print(f"\n🏆 Best model: {best_model} (F1: {best_f1})")
    else:
        print("No valid metrics found for any model.")


def main():
    # Hyperparameters and settings
    params = {
        'epochs': 5,
        'batch_size': 8,
        'learning_rate': 3e-5,
        'seed': 42,
        'max_length': 128,
        'test_size': 0.2,
    }
    logger.info("--- Loading data ---")
    all_tokens, all_labels, unique_labels = load_conll_data(CONLL_PATH)
    if not all_tokens:
        logger.error("No data loaded. Exiting.")
        return
    print_label_distribution(all_labels)
    preview_samples(all_tokens, all_labels, n=3)

    # Fine-tune and evaluate each model
    results = []
    for model_name in MODEL_LIST:
        res = evaluate_and_save(model_name, all_tokens, all_labels, unique_labels, params)
        results.append(res)

    # Compare and select the best model
    compare_models(results)

if __name__ == "__main__":
    main() 