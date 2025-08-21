"""
NER Model Training for Ethiopian Telegram Data
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from sklearn.metrics import classification_report, f1_score
from loguru import logger
import yaml


class NERModelTrainer:
    """Fine-tune NER models for Ethiopian e-commerce data"""
    
    def __init__(self, config_path: str = "../config/config.yaml"):
        try:
            self.config = self._load_config(config_path)
            self.label_list = self.config['ner']['entity_labels']
        except:
            # Default labels matching the actual dataset
            self.label_list = ['O', 'B-Product', 'I-Product', 'B-PRICE', 'I-PRICE', 'B-LOC', 'I-LOC']
        
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.tokenizer = None
        self.model = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def load_conll_data(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        """Load CoNLL format data"""
        sentences = []
        labels = []
        current_tokens = []
        current_labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('#') or not line:
                    if current_tokens:
                        sentences.append(current_tokens.copy())
                        labels.append(current_labels.copy())
                        current_tokens.clear()
                        current_labels.clear()
                else:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        current_tokens.append(parts[0])
                        current_labels.append(parts[1])
        
        if current_tokens:
            sentences.append(current_tokens)
            labels.append(current_labels)
        
        logger.info(f"Loaded {len(sentences)} sentences")
        return sentences, labels
    
    def tokenize_and_align_labels(self, examples, tokenizer):
        """Tokenize and align labels with subword tokens"""
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=True,
            max_length=512
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
                    label_ids.append(self.label2id[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def initialize_model(self, model_name: str):
        """Initialize tokenizer and model"""
        logger.info(f"Initializing model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id
        )
    
    def prepare_dataset(self, sentences: List[List[str]], labels: List[List[str]]) -> Dataset:
        """Prepare single dataset"""
        data = {
            "tokens": sentences,
            "labels": labels
        }
        return Dataset.from_dict(data)
    
    def train_model(self, train_dataset: Dataset, eval_dataset: Dataset, output_dir: str) -> str:
        """Train NER model"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        logger.info(f"Training model with {len(train_dataset)} training samples")
        
        # Tokenize datasets
        train_tokenized = train_dataset.map(
            lambda x: self.tokenize_and_align_labels(x, self.tokenizer),
            batched=True
        )
        
        eval_tokenized = eval_dataset.map(
            lambda x: self.tokenize_and_align_labels(x, self.tokenizer),
            batched=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            warmup_steps=100,
            logging_steps=10,
            save_total_limit=2,
            report_to=None
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        trainer.train()
        
        # Save model
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
        return output_dir
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Flatten for sklearn metrics
        flat_true_labels = [label for sublist in true_labels for label in sublist]
        flat_predictions = [pred for sublist in true_predictions for pred in sublist]
        
        # Calculate F1 score
        f1 = f1_score(flat_true_labels, flat_predictions, average='weighted')
        
        return {"f1": f1}
    
    def evaluate_model(self, model_path: str, test_sentences: List[List[str]], 
                      test_labels: List[List[str]]) -> Dict:
        """Evaluate trained model"""
        from transformers import pipeline
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        # Create pipeline
        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple"
        )
        
        # Evaluate on test data
        all_predictions = []
        all_true_labels = []
        
        for tokens, labels in zip(test_sentences, test_labels):
            text = " ".join(tokens)
            predictions = ner_pipeline(text)
            
            # Convert predictions back to token-level labels
            pred_labels = self._align_predictions_with_tokens(tokens, predictions)
            
            all_predictions.extend(pred_labels)
            all_true_labels.extend(labels)
        
        # Calculate metrics
        report = classification_report(all_true_labels, all_predictions, output_dict=True)
        
        return {
            "classification_report": report,
            "f1_score": report['weighted avg']['f1-score'],
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall']
        }
    
    def _align_predictions_with_tokens(self, tokens: List[str], predictions: List[Dict]) -> List[str]:
        """Align NER pipeline predictions with original tokens"""
        pred_labels = ['O'] * len(tokens)
        
        for pred in predictions:
            entity_label = pred['entity_group']
            start_char = pred['start']
            end_char = pred['end']
            
            # Find corresponding tokens (simplified approach)
            text = " ".join(tokens)
            entity_text = text[start_char:end_char]
            
            # Find token indices (basic implementation)
            for i, token in enumerate(tokens):
                if token in entity_text:
                    pred_labels[i] = f"B-{entity_label}" if pred_labels[i] == 'O' else f"I-{entity_label}"
        
        return pred_labels


def main():
    """Test the trainer"""
    trainer = NERModelTrainer()
    
    # Create sample data file first
    from src.preprocessing.conll_labeler import CoNLLLabeler
    
    labeler = CoNLLLabeler()
    sample_data = labeler.create_sample_labeled_data()
    conll_file = labeler.save_conll_format(sample_data)
    
    # Load and prepare data
    sentences, labels = trainer.load_conll_data(conll_file)
    train_dataset, val_dataset = trainer.prepare_dataset(sentences, labels)
    
    # Train a small model for testing
    model_path = trainer.train_model("distilbert-base-multilingual-cased", train_dataset, val_dataset)
    
    print(f"Model trained and saved to: {model_path}")


if __name__ == "__main__":
    main()