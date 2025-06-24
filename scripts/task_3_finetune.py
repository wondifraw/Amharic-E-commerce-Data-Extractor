"""
task_3_finetune.py
Main script to fine-tune and compare NER models.
Enhanced for flexibility, logging, and user experience.
"""
import os
import argparse
import logging
from transformers import AutoTokenizer
from src.ner.dataset import load_conll_data, create_ner_datasets, print_label_distribution, preview_samples
from src.ner.train import train_ner_model
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task_3_finetune")

# --- Configuration ---
CONLL_PATH = 'data/processed/labeled_dataset.conll'
OUTPUT_BASE = 'models/ner_finetuned'

# A list of models to train and compare
MODEL_LIST = [
    'xlm-roberta-base',
    'Davlan/bert-base-multilingual-cased-finetuned-amharic',
    'Davlan/distilbert-base-multilingual-cased-ner-hrl',
]

def compare_and_print_results(model_names, output_base):
    """
    Compare models based on F1 score and print a summary.
    """
    best_f1 = -1
    best_model = None
    logger.info("\n--- 📊 Model Performance Comparison ---")
    for model_name in model_names:
        model_folder = model_name.replace('/', '_')
        metrics_path = os.path.join(output_base, model_folder, "evaluation_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            f1 = metrics.get('f1') or metrics.get('eval_f1') or metrics.get('best_f1') or metrics.get('f1_score')
            logger.info(f"Model: {model_name}")
            logger.info(f"  F1 Score: {f1}")
            logger.info(f"  Accuracy: {metrics.get('accuracy', metrics.get('eval_accuracy', 'N/A'))}")
            logger.info(f"  Loss:     {metrics.get('loss', metrics.get('eval_loss', 'N/A'))}")
            if f1 is not None and f1 > best_f1:
                best_f1 = f1
                best_model = model_name
        else:
            logger.warning(f"Metrics not found for model: {model_name}")
        logger.info("-" * 40)
    if best_model:
        logger.info(f"\n🏆 Best model: {best_model} (F1: {best_f1})")
    else:
        logger.warning("No valid metrics found for any model.")

def main(models_to_train, epochs, batch_size, learning_rate, seed, max_length, test_size):
    """
    Main function to execute the NER model fine-tuning pipeline for a list of models.
    Enhanced for flexibility, logging, and user experience.
    """
    logger.info("--- Starting NER Model Fine-Tuning Pipeline ---")

    # 1. Load labeled data and dynamically extract unique labels
    logger.info(f"Loading data from {CONLL_PATH}...")
    all_tokens, all_labels, unique_labels = load_conll_data(CONLL_PATH)
    if not all_tokens:
        logger.error("[ERROR] No data was loaded. Please ensure the CoNLL file is correct and not empty.")
        return
    logger.info(f"Loaded {len(all_tokens)} samples. Found labels: {unique_labels}")
    print_label_distribution(all_labels)
    preview_samples(all_tokens, all_labels, n=3)

    # 2. Loop through each model, fine-tune, and save results
    for model_name in models_to_train:
        logger.info(f"\n--- Processing model: {model_name} ---")
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Create datasets
            train_dataset, eval_dataset, label_to_id, id_to_label = create_ner_datasets(
                all_tokens, all_labels, unique_labels, tokenizer,
                max_length=max_length, test_size=test_size, random_state=seed
            )
            logger.info(f"Dataset created. Training samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")
            # Define output directory for the current model
            safe_model_name = model_name.replace("/", "_")
            output_dir = os.path.join(OUTPUT_BASE, safe_model_name)
            os.makedirs(output_dir, exist_ok=True)
            # Train the model
            train_ner_model(
                model_name=model_name,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                label_to_id=label_to_id,
                id_to_label=id_to_label,
                output_dir=output_dir,
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                use_early_stopping=True,
                seed=seed
            )
            logger.info(f"✅ Finished fine-tuning {model_name}. Results are in: {output_dir}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to process model '{model_name}'. Error: {e}")
            continue # Move to the next model

    logger.info("\n--- NER Model Fine-Tuning Pipeline Finished! ---")
    compare_and_print_results(models_to_train, OUTPUT_BASE)
    logger.info("Compare 'evaluation_metrics.json' in each model's output directory to select the best one.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune and compare NER models.")
    parser.add_argument(
        "--models",
        nargs='+',
        default=MODEL_LIST,
        help=f"A list of pretrained models to fine-tune. Defaults to: {MODEL_LIST}"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate (default: 3e-5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length (default: 128)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Validation split fraction (default: 0.2)")
    args = parser.parse_args()
    main(
        models_to_train=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_length=args.max_length,
        test_size=args.test_size
    ) 