"""
Model Evaluation and Comparison System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import json
import os


class NERModelEvaluator:
    """Evaluate and compare NER models"""
    
    def __init__(self):
        self.evaluation_results = {}
        self.model_comparisons = {}
    
    def evaluate_predictions(self, true_labels: List[List[str]], 
                           predicted_labels: List[List[str]], 
                           model_name: str) -> Dict:
        """Evaluate model predictions against true labels"""
        
        # Flatten labels for sklearn metrics
        flat_true = [label for sentence in true_labels for label in sentence]
        flat_pred = [label for sentence in predicted_labels for label in sentence]
        
        # Ensure same length
        min_len = min(len(flat_true), len(flat_pred))
        flat_true = flat_true[:min_len]
        flat_pred = flat_pred[:min_len]
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            flat_true, flat_pred, average=None, labels=list(set(flat_true))
        )
        
        # Overall metrics
        overall_precision = precision_recall_fscore_support(flat_true, flat_pred, average='weighted')[0]
        overall_recall = precision_recall_fscore_support(flat_true, flat_pred, average='weighted')[1]
        overall_f1 = precision_recall_fscore_support(flat_true, flat_pred, average='weighted')[2]
        
        # Entity-level metrics
        entity_metrics = self._calculate_entity_metrics(true_labels, predicted_labels)
        
        # Classification report
        class_report = classification_report(flat_true, flat_pred, output_dict=True)
        
        evaluation_result = {
            'model_name': model_name,
            'overall_metrics': {
                'precision': float(overall_precision),
                'recall': float(overall_recall),
                'f1_score': float(overall_f1),
                'accuracy': float(sum(t == p for t, p in zip(flat_true, flat_pred)) / len(flat_true))
            },
            'entity_metrics': entity_metrics,
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(flat_true, flat_pred).tolist(),
            'label_names': list(set(flat_true))
        }
        
        self.evaluation_results[model_name] = evaluation_result
        logger.info(f"Evaluated model {model_name}: F1={overall_f1:.3f}")
        
        return evaluation_result
    
    def _calculate_entity_metrics(self, true_labels: List[List[str]], 
                                 predicted_labels: List[List[str]]) -> Dict:
        """Calculate entity-level precision, recall, and F1"""
        
        def extract_entities(labels: List[str]) -> List[Tuple[str, int, int]]:
            """Extract entities from BIO labels"""
            entities = []
            current_entity = None
            start_idx = 0
            
            for i, label in enumerate(labels):
                if label.startswith('B-'):
                    if current_entity:
                        entities.append((current_entity, start_idx, i-1))
                    current_entity = label[2:]
                    start_idx = i
                elif label.startswith('I-'):
                    if current_entity != label[2:]:
                        if current_entity:
                            entities.append((current_entity, start_idx, i-1))
                        current_entity = label[2:]
                        start_idx = i
                else:  # O label
                    if current_entity:
                        entities.append((current_entity, start_idx, i-1))
                        current_entity = None
            
            if current_entity:
                entities.append((current_entity, start_idx, len(labels)-1))
            
            return entities
        
        # Extract entities from all sentences
        true_entities = []
        pred_entities = []
        
        for true_sent, pred_sent in zip(true_labels, predicted_labels):
            true_entities.extend(extract_entities(true_sent))
            pred_entities.extend(extract_entities(pred_sent))
        
        # Calculate metrics by entity type
        entity_types = set([ent[0] for ent in true_entities + pred_entities])
        entity_metrics = {}
        
        for entity_type in entity_types:
            true_type_entities = set([ent for ent in true_entities if ent[0] == entity_type])
            pred_type_entities = set([ent for ent in pred_entities if ent[0] == entity_type])
            
            tp = len(true_type_entities & pred_type_entities)
            fp = len(pred_type_entities - true_type_entities)
            fn = len(true_type_entities - pred_type_entities)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            entity_metrics[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': len(true_type_entities)
            }
        
        return entity_metrics
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict:
        """Compare multiple model evaluation results"""
        
        comparison = {
            'models_compared': list(model_results.keys()),
            'overall_comparison': {},
            'entity_comparison': {},
            'best_model': None
        }
        
        # Overall metrics comparison
        for metric in ['precision', 'recall', 'f1_score', 'accuracy']:
            comparison['overall_comparison'][metric] = {}
            for model_name, results in model_results.items():
                comparison['overall_comparison'][metric][model_name] = \
                    results['overall_metrics'][metric]
        
        # Entity-level comparison
        all_entities = set()
        for results in model_results.values():
            all_entities.update(results['entity_metrics'].keys())
        
        for entity in all_entities:
            comparison['entity_comparison'][entity] = {}
            for model_name, results in model_results.items():
                if entity in results['entity_metrics']:
                    comparison['entity_comparison'][entity][model_name] = \
                        results['entity_metrics'][entity]
                else:
                    comparison['entity_comparison'][entity][model_name] = {
                        'precision': 0, 'recall': 0, 'f1_score': 0, 'support': 0
                    }
        
        # Determine best model based on overall F1 score
        best_f1 = 0
        best_model = None
        for model_name, results in model_results.items():
            f1 = results['overall_metrics']['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_model = model_name
        
        comparison['best_model'] = {
            'name': best_model,
            'f1_score': best_f1
        }
        
        self.model_comparisons = comparison
        logger.info(f"Best model: {best_model} with F1: {best_f1:.3f}")
        
        return comparison
    
    def create_confusion_matrix_plot(self, model_name: str, save_path: Optional[str] = None):
        """Create confusion matrix visualization"""
        
        if model_name not in self.evaluation_results:
            logger.error(f"No evaluation results found for model: {model_name}")
            return
        
        results = self.evaluation_results[model_name]
        cm = np.array(results['confusion_matrix'])
        labels = results['label_names']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def create_model_comparison_plot(self, save_path: Optional[str] = None):
        """Create model comparison visualization"""
        
        if not self.model_comparisons:
            logger.error("No model comparisons available. Run compare_models first.")
            return
        
        # Prepare data for plotting
        models = self.model_comparisons['models_compared']
        metrics = ['precision', 'recall', 'f1_score', 'accuracy']
        
        data = []
        for metric in metrics:
            for model in models:
                data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': self.model_comparisons['overall_comparison'][metric][model]
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Model', y='Score', hue='Metric')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.legend(title='Metrics')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        ax = plt.gca()
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', rotation=90, padding=3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def create_entity_performance_plot(self, save_path: Optional[str] = None):
        """Create entity-level performance visualization"""
        
        if not self.model_comparisons:
            logger.error("No model comparisons available. Run compare_models first.")
            return
        
        # Prepare data
        entity_data = []
        for entity, model_scores in self.model_comparisons['entity_comparison'].items():
            for model, scores in model_scores.items():
                entity_data.append({
                    'Entity': entity,
                    'Model': model,
                    'F1 Score': scores['f1_score'],
                    'Precision': scores['precision'],
                    'Recall': scores['recall']
                })
        
        df = pd.DataFrame(entity_data)
        
        # Create heatmap
        pivot_df = df.pivot_table(values='F1 Score', index='Entity', columns='Model')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', fmt='.3f', 
                   cbar_kws={'label': 'F1 Score'})
        plt.title('Entity-Level F1 Scores by Model')
        plt.xlabel('Model')
        plt.ylabel('Entity Type')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Entity performance plot saved to {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, output_dir: str = "reports/evaluation"):
        """Generate comprehensive evaluation report"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations
        for model_name in self.evaluation_results.keys():
            cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
            self.create_confusion_matrix_plot(model_name, cm_path)
        
        if self.model_comparisons:
            comp_path = os.path.join(output_dir, "model_comparison.png")
            self.create_model_comparison_plot(comp_path)
            
            entity_path = os.path.join(output_dir, "entity_performance.png")
            self.create_entity_performance_plot(entity_path)
        
        # Compile report data
        report = {
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'models_evaluated': list(self.evaluation_results.keys()),
            'evaluation_results': self.evaluation_results,
            'model_comparisons': self.model_comparisons,
            'summary': self._generate_summary()
        }
        
        # Save report
        report_path = os.path.join(output_dir, "evaluation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report_path
    
    def _generate_summary(self) -> Dict:
        """Generate evaluation summary"""
        
        if not self.evaluation_results:
            return {"message": "No evaluation results available"}
        
        summary = {
            'total_models_evaluated': len(self.evaluation_results),
            'best_performing_model': None,
            'average_performance': {},
            'entity_performance_summary': {}
        }
        
        # Find best model
        if self.model_comparisons and 'best_model' in self.model_comparisons:
            summary['best_performing_model'] = self.model_comparisons['best_model']
        
        # Calculate average performance
        all_f1_scores = [
            results['overall_metrics']['f1_score'] 
            for results in self.evaluation_results.values()
        ]
        all_precision_scores = [
            results['overall_metrics']['precision'] 
            for results in self.evaluation_results.values()
        ]
        all_recall_scores = [
            results['overall_metrics']['recall'] 
            for results in self.evaluation_results.values()
        ]
        
        summary['average_performance'] = {
            'avg_f1_score': np.mean(all_f1_scores),
            'avg_precision': np.mean(all_precision_scores),
            'avg_recall': np.mean(all_recall_scores),
            'f1_std': np.std(all_f1_scores)
        }
        
        # Entity performance summary
        if self.model_comparisons and 'entity_comparison' in self.model_comparisons:
            entity_summary = {}
            for entity, model_scores in self.model_comparisons['entity_comparison'].items():
                f1_scores = [scores['f1_score'] for scores in model_scores.values()]
                entity_summary[entity] = {
                    'avg_f1': np.mean(f1_scores),
                    'best_f1': max(f1_scores),
                    'worst_f1': min(f1_scores)
                }
            summary['entity_performance_summary'] = entity_summary
        
        return summary


def main():
    """Test model evaluation"""
    
    # Sample data for testing
    true_labels = [
        ['B-Product', 'I-Product', 'B-PRICE', 'I-PRICE', 'O', 'B-LOC'],
        ['O', 'B-Product', 'O', 'B-PRICE', 'I-PRICE', 'B-LOC', 'I-LOC'],
        ['B-LOC', 'I-LOC', 'O', 'B-Product', 'B-PRICE', 'I-PRICE']
    ]
    
    # Simulate predictions from different models
    pred_model1 = [
        ['B-Product', 'I-Product', 'B-PRICE', 'I-PRICE', 'O', 'B-LOC'],
        ['O', 'B-Product', 'O', 'B-PRICE', 'I-PRICE', 'B-LOC', 'O'],
        ['B-LOC', 'I-LOC', 'O', 'B-Product', 'B-PRICE', 'I-PRICE']
    ]
    
    pred_model2 = [
        ['B-Product', 'I-Product', 'B-PRICE', 'O', 'O', 'B-LOC'],
        ['O', 'B-Product', 'O', 'B-PRICE', 'I-PRICE', 'B-LOC', 'I-LOC'],
        ['B-LOC', 'O', 'O', 'B-Product', 'B-PRICE', 'I-PRICE']
    ]
    
    # Initialize evaluator
    evaluator = NERModelEvaluator()
    
    # Evaluate models
    result1 = evaluator.evaluate_predictions(true_labels, pred_model1, "XLM-RoBERTa")
    result2 = evaluator.evaluate_predictions(true_labels, pred_model2, "DistilBERT")
    
    # Compare models
    comparison = evaluator.compare_models({
        "XLM-RoBERTa": result1,
        "DistilBERT": result2
    })
    
    # Generate report
    report_path = evaluator.generate_evaluation_report()
    
    print(f"Evaluation completed. Report saved to: {report_path}")
    print(f"Best model: {comparison['best_model']['name']} "
          f"(F1: {comparison['best_model']['f1_score']:.3f})")


if __name__ == "__main__":
    main()