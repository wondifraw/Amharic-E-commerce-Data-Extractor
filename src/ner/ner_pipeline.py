"""
ner_pipeline.py
Unified module for loading, preparing, and fine-tuning NER models using Hugging Face Transformers.
Includes data utilities, training, and evaluation with robust logging and reproducibility.
"""

import os
import json
import numpy as np
from typing import List, Tuple, Dict, Optional
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter
import logging

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ner.pipeline")

# --- Data Utilities ---
def load_conll_data(file_path: str) -> Tuple[List[List[str]], List[List[str]], List[str]]:
    """
    Loads data from a CoNLL-formatted file and extracts unique labels.
    """
    sentences, labels = [], []
    current_tokens, current_labels = [], []
    label_set = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if current_tokens:
                        sentences.append(current_tokens)
                        labels.append(current_labels)
                        label_set.update(current_labels)
                        current_tokens, current_labels = [], []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        current_tokens.append(parts[0])
                        current_labels.append(parts[1])
            if current_tokens:
                sentences.append(current_tokens)
                labels.append(current_labels)
                label_set.update(current_labels)
    except FileNotFoundError:
        logger.error(f"The file was not found at {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while reading {file_path}: {e}")
    if not sentences or not labels:
        logger.warning(f"No data loaded from {file_path}. Check file format.")
    unique_labels = sorted(list(label_set))
    logger.info(f"Loaded {len(sentences)} sentences from {file_path}. Found {len(unique_labels)} unique labels.")
    return sentences, labels, unique_labels

def print_label_distribution(all_labels: List[List[str]]):
    """
    Print the distribution of labels in the dataset.
    """
    flat_labels = [label for seq in all_labels for label in seq]
    counter = Counter(flat_labels)
    logger.info("Label distribution:")
    for label, count in counter.items():
        logger.info(f"  {label}: {count}")

def preview_samples(all_tokens: List[List[str]], all_labels: List[List[str]], n: int = 3):
    """
    Print a preview of a few samples from the dataset.
    """
    logger.info(f"Previewing {min(n, len(all_tokens))} samples:")
    for i in range(min(n, len(all_tokens))):
        logger.info(f"Tokens: {all_tokens[i]}")
        logger.info(f"Labels: {all_labels[i]}")

def create_ner_datasets(
    all_tokens: List[List[str]],
    all_labels: List[List[str]],
    label_list: List[str],
    tokenizer,
    max_length: int = 128,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Creates and preprocesses training and validation datasets for NER.
    """
    if not all_tokens or not all_labels:
        logger.error("Empty token or label list provided to create_ner_datasets.")
        raise ValueError("No data to create datasets.")
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for i, label in enumerate(label_list)}

    train_tokens, val_tokens, train_labels, val_labels = train_test_split(
        all_tokens, all_labels, test_size=test_size, random_state=random_state
    )
    logger.info(f"Split data: {len(train_tokens)} train, {len(val_tokens)} val samples.")

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=max_length
        )
        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id.get(label[word_idx], -100))
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    train_dataset = Dataset.from_dict({"tokens": train_tokens, "labels": train_labels})
    val_dataset = Dataset.from_dict({"tokens": val_tokens, "labels": val_labels})

    try:
        tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
        tokenized_val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        raise

    return tokenized_train_dataset, tokenized_val_dataset, label_to_id, id_to_label

# --- Training Utilities ---
def compute_metrics(p, id_to_label: Dict[int, str]) -> Dict[str, float]:
    """
    Computes evaluation metrics (accuracy, f1, precision, recall) for token classification.
    Ignores the padded tokens (label -100).
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    flat_true_labels = [label for sublist in true_labels for label in sublist]
    flat_true_predictions = [pred for sublist in true_predictions for pred in sublist]
    if not flat_true_labels:
        return {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0}
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_true_labels, flat_true_predictions, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(flat_true_labels, flat_true_predictions)
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

def train_ner_model(
    model_name: str,
    train_dataset,
    eval_dataset,
    label_to_id,
    id_to_label,
    output_dir: str,
    num_train_epochs: int = 5,
    learning_rate: float = 3e-5,
    batch_size: int = 8,
    use_early_stopping: bool = True,
    seed: int = 42,
    custom_callbacks: Optional[List] = None,
):
    """
    Configures and runs the fine-tuning process for a NER model.
    Enhanced for reproducibility, logging, and usability.
    """
    logger.info(f"--- Starting NER Model Training for {model_name} ---")
    set_seed(seed)
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(label_to_id), id2label=id_to_label, label2id=label_to_id
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load model or tokenizer for {model_name}: {e}")
        raise
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        save_total_limit=2,
        seed=seed,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if use_early_stopping else []
    if custom_callbacks:
        callbacks.extend(custom_callbacks)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id_to_label),
        callbacks=callbacks,
    )
    try:
        trainer.train()
        eval_metrics = trainer.evaluate()
        logger.info(f"Evaluation Metrics: {eval_metrics}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "training_args.json"), "w", encoding="utf-8") as f:
            json.dump(training_args.to_dict(), f, indent=2)
        with open(os.path.join(output_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2)
        logger.info(f"Model, tokenizer, and metrics saved to {output_dir}")
    except Exception as e:
        logger.error(f"Training failed for {model_name}: {e}")
        raise 