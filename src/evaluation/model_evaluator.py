"""Model evaluation and comparison utilities."""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
from typing import List, Dict, Tuple
import time
import logging
from sklearn.metrics import classification_report
import json

from config.config import model_config

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluator for comparing NER models."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_model(self, model_path: str, model_name: str) -> None:
        """Load a trained model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            
            self.models[model_name] = {
                'tokenizer': tokenizer,
                'model': model,
                'pipeline': pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple')
            }
            
            logger.info(f"Loaded model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
    
    def evaluate_model_speed(self, model_name: str, test_texts: List[str]) -> float:
        """Evaluate model inference speed."""
        if model_name not in self.models:
            return 0.0
        
        pipeline_model = self.models[model_name]['pipeline']
        
        start_time = time.time()
        
        for text in test_texts:
            try:
                _ = pipeline_model(text)
            except Exception as e:
                logger.warning(f"Error processing text with {model_name}: {e}")
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / len(test_texts)
        return avg_time
    
    def evaluate_model_accuracy(self, model_name: str, test_data: List[Dict]) -> Dict:
        """Evaluate model accuracy on test data."""
        if model_name not in self.models:
            return {}
        
        pipeline_model = self.models[model_name]['pipeline']
        
        true_labels = []
        pred_labels = []
        
        for example in test_data:
            text = ' '.join(example['tokens'])
            true_entities = self._extract_entities_from_labels(example['tokens'], example['labels'])
            
            try:
                predictions = pipeline_model(text)
                pred_entities = self._extract_entities_from_predictions(predictions)
                
                # Convert to label sequences for comparison
                true_seq = self._entities_to_sequence(example['tokens'], true_entities)
                pred_seq = self._entities_to_sequence(example['tokens'], pred_entities)
                
                true_labels.extend(true_seq)
                pred_labels.extend(pred_seq)
                
            except Exception as e:
                logger.warning(f"Error evaluating example with {model_name}: {e}")
        
        # Calculate metrics
        if true_labels and pred_labels:
            report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
            return {
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score'],
                'accuracy': report['accuracy']
            }
        
        return {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}
    
    def _extract_entities_from_labels(self, tokens: List[str], labels: List[str]) -> List[Dict]:
        """Extract entities from BIO labels."""
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'start': i,
                    'end': i + 1,
                    'label': label[2:],
                    'text': token
                }
            elif label.startswith('I-') and current_entity:
                current_entity['end'] = i + 1
                current_entity['text'] += ' ' + token
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _extract_entities_from_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Extract entities from model predictions."""
        entities = []
        
        for pred in predictions:
            entities.append({
                'start': pred.get('start', 0),
                'end': pred.get('end', 0),
                'label': pred.get('entity_group', 'O'),
                'text': pred.get('word', ''),
                'confidence': pred.get('score', 0.0)
            })
        
        return entities
    
    def _entities_to_sequence(self, tokens: List[str], entities: List[Dict]) -> List[str]:
        """Convert entities back to BIO sequence."""
        labels = ['O'] * len(tokens)
        
        for entity in entities:
            start = entity.get('start', 0)
            end = entity.get('end', start + 1)
            label_type = entity.get('label', 'O')
            
            if start < len(labels):
                labels[start] = f'B-{label_type}'
                for i in range(start + 1, min(end, len(labels))):
                    labels[i] = f'I-{label_type}'
        
        return labels
    
    def compare_models(self, test_data: List[Dict], test_texts: List[str]) -> pd.DataFrame:
        """Compare all loaded models."""
        comparison_results = []
        
        for model_name in self.models.keys():
            logger.info(f"Evaluating {model_name}...")
            
            # Accuracy metrics
            accuracy_metrics = self.evaluate_model_accuracy(model_name, test_data)
            
            # Speed metrics
            avg_inference_time = self.evaluate_model_speed(model_name, test_texts[:10])  # Sample for speed
            
            result = {
                'model': model_name,
                'precision': accuracy_metrics.get('precision', 0),
                'recall': accuracy_metrics.get('recall', 0),
                'f1_score': accuracy_metrics.get('f1', 0),
                'accuracy': accuracy_metrics.get('accuracy', 0),
                'avg_inference_time': avg_inference_time,
                'speed_score': 1 / (avg_inference_time + 0.001)  # Higher is better
            }
            
            comparison_results.append(result)
        
        df = pd.DataFrame(comparison_results)
        
        # Calculate overall score
        if not df.empty:
            df['overall_score'] = (
                df['f1_score'] * 0.4 + 
                df['accuracy'] * 0.3 + 
                (df['speed_score'] / df['speed_score'].max()) * 0.3
            )
            
            df = df.sort_values('overall_score', ascending=False)
        
        return df
    
    def save_comparison_results(self, results_df: pd.DataFrame, filepath: str) -> None:
        """Save comparison results."""
        try:
            results_df.to_csv(filepath, index=False)
            logger.info(f"Comparison results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")