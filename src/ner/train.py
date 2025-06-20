"""
train.py
Module for fine-tuning a NER model using the Hugging Face Trainer API.
"""
import os
import json
import numpy as np
from typing import Dict, List, Tuple
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(p: Tuple[np.ndarray, np.ndarray], id_to_label: Dict[int, str]) -> Dict[str, float]:
    """
    Computes evaluation metrics (accuracy, f1, precision, recall) for token classification.
    Ignores the padded tokens (label -100).
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (-100) and convert predictions and labels to strings
    true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Flatten the lists
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
    train_dataset: Dataset,
    eval_dataset: Dataset,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    output_dir: str,
):
    """
    Configures and runs the fine-tuning process for a NER model.

    Args:
        model_name: The name of the pretrained model from Hugging Face Hub.
        train_dataset: The tokenized training dataset.
        eval_dataset: The tokenized evaluation dataset.
        label_to_id: A dictionary mapping label names to integer IDs.
        id_to_label: A dictionary mapping integer IDs to label names.
        output_dir: The directory to save the fine-tuned model and results.
    """
    print(f"--- Starting NER Model Training for {model_name} ---")
    
    # 1. Load Model and Tokenizer
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(label_to_id), id2label=id_to_label, label2id=label_to_id
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"[ERROR] Failed to load model or tokenizer '{model_name}': {e}")
        return

    # 2. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3, # Keep it low for a quick example run
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none", # Disable wandb/tensorboard reporting for simplicity
    )

    # 3. Initialize Trainer
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id_to_label),
    )

    # 4. Train and Evaluate
    try:
        print("Starting training...")
        trainer.train()
        print("Training finished. Evaluating...")
        eval_metrics = trainer.evaluate()
        print(f"Evaluation Metrics: {eval_metrics}")
        
        # 5. Save the final model and metrics
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")
        
        metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"Evaluation metrics saved to {metrics_path}")

    except Exception as e:
        print(f"[ERROR] An error occurred during training or evaluation: {e}") 