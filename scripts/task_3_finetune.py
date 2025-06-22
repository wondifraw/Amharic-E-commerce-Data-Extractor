"""
task_3_finetune.py
Main script to fine-tune a Named Entity Recognition (NER) model.

This script orchestrates the entire fine-tuning process:
1.  Loads the labeled data in CoNLL format.
2.  Prepares the data using a tokenizer.
3.  Splits the data into training and validation sets.
4.  Initiates the training and evaluation process.
5.  Saves the fine-tuned model and its performance metrics.
"""
import os
import argparse
from transformers import AutoTokenizer
from src.ner.dataset import load_conll_data, create_ner_datasets
from src.ner.train import train_ner_model

# --- Configuration ---
CONLL_PATH = 'data/processed/labeled_dataset.conll'
DEFAULT_MODEL_NAME = 'xlm-roberta-base'
DEFAULT_OUTPUT_DIR = 'models/ner_finetuned'
LABEL_LIST = ['O', 'B-Product', 'I-Product', 'B-LOC', 'I-LOC', 'B-PRICE', 'I-PRICE']

def main(model_name: str, output_dir: str):
    """
    Main function to execute the NER model fine-tuning pipeline.
    """
    print("--- Starting NER Model Fine-Tuning Pipeline ---")
    
    # 1. Load labeled data
    print(f"Loading data from {CONLL_PATH}...")
    all_tokens, all_labels = load_conll_data(CONLL_PATH)
    if not all_tokens:
        print("[ERROR] No data was loaded. Please ensure the CoNLL file is correct and not empty.")
        return

    # 2. Load tokenizer
    print(f"Loading tokenizer for '{model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"[ERROR] Could not load the tokenizer for '{model_name}'. Please check the model name. Error: {e}")
        return

    # 3. Create datasets
    print("Creating and preprocessing datasets...")
    train_dataset, eval_dataset, label_to_id, id_to_label = create_ner_datasets(
        all_tokens, all_labels, LABEL_LIST, tokenizer
    )
    print(f"Dataset created. Training samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")

    # 4. Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 5. Train the model
    train_ner_model(
        model_name,
        train_dataset,
        eval_dataset,
        label_to_id,
        id_to_label,
        output_dir,
    )

    print("\n--- NER Model Fine-Tuning Pipeline Finished Successfully! ---")
    print(f"The fine-tuned model and evaluation results are saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a NER model on a CoNLL dataset.")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="The name of the pretrained model to fine-tune (from Hugging Face Hub)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="The directory where the fine-tuned model and results will be saved."
    )
    args = parser.parse_args()
    
    main(args.model_name, args.output_dir) 