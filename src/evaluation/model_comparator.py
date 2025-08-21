"""
Model Comparison and Selection System
"""

import os
import json
import pandas as pd
from typing import Dict, List, Tuple
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from src.ner.model_trainer import NERModelTrainer


class ModelComparator:
    """Compare and select best NER models"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.trainer = NERModelTrainer(config_path)
        self.results = {}
    
    def compare_models(self, data_path: str, models: List[str] = None) -> Dict:
        """Compare multiple models on the same dataset"""
        if models is None:
            models = self.trainer.config['ner']['model_names']
        
        # Load data
        sentences, labels = self.trainer.load_conll_data(data_path)
        train_dataset, val_dataset = self.trainer.prepare_dataset(sentences, labels)
        
        # Test data (use validation for now)
        test_sentences = sentences[int(0.8 * len(sentences)):]
        test_labels = labels[int(0.8 * len(sentences)):]
        
        comparison_results = {}
        
        for model_name in models:
            logger.info(f"Training and evaluating {model_name}")
            
            try:
                # Train model
                model_path = self.trainer.train_model(model_name, train_dataset, val_dataset)
                
                # Evaluate model
                metrics = self.trainer.evaluate_model(model_path, test_sentences, test_labels)
                
                comparison_results[model_name] = {
                    'model_path': model_path,
                    'metrics': metrics,
                    'f1_score': metrics['f1_score'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall']
                }
                
                logger.info(f"{model_name} - F1: {metrics['f1_score']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                comparison_results[model_name] = {'error': str(e)}
        
        self.results = comparison_results
        return comparison_results
    
    def select_best_model(self) -> Tuple[str, Dict]:
        """Select best model based on F1 score"""
        if not self.results:
            raise ValueError("No comparison results available. Run compare_models first.")
        
        best_model = None
        best_f1 = 0
        
        for model_name, result in self.results.items():
            if 'error' not in result and result['f1_score'] > best_f1:
                best_f1 = result['f1_score']
                best_model = model_name
        
        if best_model is None:
            raise ValueError("No successful model training found")
        
        logger.info(f"Best model: {best_model} with F1 score: {best_f1:.3f}")
        return best_model, self.results[best_model]
    
    def create_comparison_report(self, output_path: str = "models/comparison_report.json"):
        """Create detailed comparison report"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = {
            'comparison_date': pd.Timestamp.now().isoformat(),
            'models_compared': list(self.results.keys()),
            'results': self.results
        }
        
        # Add summary statistics
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if successful_results:
            report['summary'] = {
                'best_model': max(successful_results.keys(), key=lambda x: successful_results[x]['f1_score']),
                'avg_f1': sum(r['f1_score'] for r in successful_results.values()) / len(successful_results),
                'avg_precision': sum(r['precision'] for r in successful_results.values()) / len(successful_results),
                'avg_recall': sum(r['recall'] for r in successful_results.values()) / len(successful_results)
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparison report saved to {output_path}")
        return report
    
    def visualize_comparison(self, output_dir: str = "models/plots/"):
        """Create visualization of model comparison"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for plotting
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not successful_results:
            logger.warning("No successful results to visualize")
            return
        
        models = list(successful_results.keys())
        f1_scores = [successful_results[m]['f1_score'] for m in models]
        precision_scores = [successful_results[m]['precision'] for m in models]
        recall_scores = [successful_results[m]['recall'] for m in models]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of F1 scores
        ax1.bar(models, f1_scores, color='skyblue', alpha=0.7)
        ax1.set_title('Model F1 Scores Comparison')
        ax1.set_ylabel('F1 Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(f1_scores):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Precision-Recall scatter plot
        ax2.scatter(precision_scores, recall_scores, s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax2.annotate(model.split('/')[-1], (precision_scores[i], recall_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Precision')
        ax2.set_ylabel('Recall')
        ax2.set_title('Precision vs Recall')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed metrics heatmap
        metrics_df = pd.DataFrame({
            'F1 Score': f1_scores,
            'Precision': precision_scores,
            'Recall': recall_scores
        }, index=[m.split('/')[-1] for m in models])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics_df.T, annot=True, cmap='Blues', fmt='.3f', cbar_kws={'label': 'Score'})
        plt.title('Model Performance Metrics Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def get_model_recommendations(self) -> Dict:
        """Get model recommendations based on different criteria"""
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not successful_results:
            return {"error": "No successful model results available"}
        
        recommendations = {}
        
        # Best overall performance
        best_f1_model = max(successful_results.keys(), key=lambda x: successful_results[x]['f1_score'])
        recommendations['best_overall'] = {
            'model': best_f1_model,
            'reason': 'Highest F1 score',
            'f1_score': successful_results[best_f1_model]['f1_score']
        }
        
        # Best precision
        best_precision_model = max(successful_results.keys(), key=lambda x: successful_results[x]['precision'])
        recommendations['best_precision'] = {
            'model': best_precision_model,
            'reason': 'Highest precision (fewer false positives)',
            'precision': successful_results[best_precision_model]['precision']
        }
        
        # Best recall
        best_recall_model = max(successful_results.keys(), key=lambda x: successful_results[x]['recall'])
        recommendations['best_recall'] = {
            'model': best_recall_model,
            'reason': 'Highest recall (fewer false negatives)',
            'recall': successful_results[best_recall_model]['recall']
        }
        
        # Balanced model (closest to equal precision and recall)
        balanced_scores = {}
        for model, result in successful_results.items():
            balance_score = 1 - abs(result['precision'] - result['recall'])
            balanced_scores[model] = balance_score
        
        most_balanced_model = max(balanced_scores.keys(), key=lambda x: balanced_scores[x])
        recommendations['most_balanced'] = {
            'model': most_balanced_model,
            'reason': 'Most balanced precision and recall',
            'balance_score': balanced_scores[most_balanced_model]
        }
        
        return recommendations


def main():
    """Test model comparison"""
    # First create sample data
    from src.preprocessing.conll_labeler import CoNLLLabeler
    
    labeler = CoNLLLabeler()
    sample_data = labeler.create_sample_labeled_data()
    conll_file = labeler.save_conll_format(sample_data)
    
    # Compare models (using smaller models for testing)
    comparator = ModelComparator()
    
    # Use smaller/faster models for testing
    test_models = ["distilbert-base-multilingual-cased"]
    
    results = comparator.compare_models(conll_file, test_models)
    
    # Generate report and visualizations
    report = comparator.create_comparison_report()
    comparator.visualize_comparison()
    
    # Get recommendations
    recommendations = comparator.get_model_recommendations()
    
    print("Model Comparison Results:")
    for model, result in results.items():
        if 'error' not in result:
            print(f"{model}: F1={result['f1_score']:.3f}")
    
    print("\nRecommendations:")
    for category, rec in recommendations.items():
        print(f"{category}: {rec['model']} - {rec['reason']}")


if __name__ == "__main__":
    main()