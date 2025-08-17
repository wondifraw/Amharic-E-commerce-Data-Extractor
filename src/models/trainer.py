"""Professional NER model trainer."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    Trainer, TrainingArguments, DataCollatorForTokenClassification
)
from seqeval.metrics import f1_score, precision_score, recall_score


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    model_name: str = "xlm-roberta-base"
    output_dir: str = "models/trained"
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100


class NERTrainer:
    """Professional NER model trainer with comprehensive evaluation."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.label_list = None
        
    def setup_model(self, label_list: list):
        """Initialize tokenizer and model."""
        self.label_list = label_list
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        id2label = {i: label for i, label in enumerate(label_list)}
        label2id = {label: i for i, label in enumerate(label_list)}
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id
        )
        
    def tokenize_and_align_labels(self, examples):
        """Tokenize and align labels for transformer models."""
        tokenized_inputs = self.tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True, 
            padding=True
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
                
            labels.append(label_ids)
            
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
        
    def compute_metrics(self, eval_prediction):
        """Compute evaluation metrics."""
        predictions, labels = eval_prediction
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
        
    def train(self, datasets) -> str:
        """Train the NER model."""
        if not self.tokenizer or not self.model:
            raise ValueError("Model not initialized. Call setup_model() first.")
            
        # Tokenize datasets
        tokenized_datasets = datasets.map(
            self.tokenize_and_align_labels, 
            batched=True
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=DataCollatorForTokenClassification(
                tokenizer=self.tokenizer, padding=True
            ),
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        # Train model
        trainer.train()
        
        # Save model
        model_path = Path(self.config.output_dir) / "final_model"
        trainer.save_model(str(model_path))
        
        return str(model_path)