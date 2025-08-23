"""NER model training utilities."""

import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from datasets import Dataset
import pandas as pd
from typing import List, Dict, Tuple
import logging
from sklearn.metrics import classification_report
from seqeval.metrics import f1_score, precision_score, recall_score

from config.config import model_config, data_config

logger = logging.getLogger(__name__)

class NERTrainer:
    """Trainer for NER models."""
    
    def __init__(self, model_name: str = 'xlm-roberta-base'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.labels = data_config.entity_labels
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        
    def load_model(self) -> None:
        """Load tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.labels),
                id2label=self.id_to_label,
                label2id=self.label_to_id
            )
            logger.info(f"Loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def parse_conll_data(self, conll_text: str) -> List[Dict]:
        """Parse CoNLL format data."""
        examples = []
        current_tokens = []
        current_labels = []
        
        lines = conll_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            if not line:  # Empty line indicates end of sentence
                if current_tokens:
                    examples.append({
                        'tokens': current_tokens,
                        'labels': current_labels
                    })
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, label = parts
                    current_tokens.append(token)
                    current_labels.append(label)
        
        # Add last example if exists
        if current_tokens:
            examples.append({
                'tokens': current_tokens,
                'labels': current_labels
            })
        
        return examples
    
    def tokenize_and_align_labels(self, examples: Dict) -> Dict:
        """Tokenize and align labels with subword tokens."""
        tokenized_inputs = self.tokenizer(
            examples['tokens'],
            truncation=True,
            is_split_into_words=True,
            max_length=model_config.max_length,
            padding=True
        )
        
        labels = []
        for i, label_list in enumerate(examples['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_to_id[label_list[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    
    def prepare_dataset(self, conll_data: str) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets."""
        examples = self.parse_conll_data(conll_data)
        
        # Split into train/val
        split_idx = int(0.8 * len(examples))
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        # Convert to datasets
        train_dataset = Dataset.from_dict({
            'tokens': [ex['tokens'] for ex in train_examples],
            'labels': [ex['labels'] for ex in train_examples]
        })
        
        val_dataset = Dataset.from_dict({
            'tokens': [ex['tokens'] for ex in val_examples],
            'labels': [ex['labels'] for ex in val_examples]
        })
        
        # Tokenize
        train_dataset = train_dataset.map(self.tokenize_and_align_labels, batched=True)
        val_dataset = val_dataset.map(self.tokenize_and_align_labels, batched=True)
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        return {
            'precision': precision_score(true_labels, true_predictions),
            'recall': recall_score(true_labels, true_predictions),
            'f1': f1_score(true_labels, true_predictions),
        }
    
    def train(self, conll_data: str, output_dir: str) -> None:
        """Train the NER model."""
        try:
            if not self.model:
                self.load_model()
            
            train_dataset, val_dataset = self.prepare_dataset(conll_data)
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=model_config.num_epochs,
                per_device_train_batch_size=model_config.batch_size,
                per_device_eval_batch_size=model_config.batch_size,
                warmup_steps=model_config.warmup_steps,
                weight_decay=0.01,
                logging_dir=f'{output_dir}/logs',
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                learning_rate=model_config.learning_rate,
            )
            
            data_collator = DataCollatorForTokenClassification(self.tokenizer)
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )
            
            trainer.train()
            trainer.save_model()
            
            logger.info(f"Model training completed and saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            raise