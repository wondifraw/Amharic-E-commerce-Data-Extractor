"""
Model Interpretability using SHAP and LIME
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install with: pip install lime")


class NERModelExplainer:
    """Explain NER model predictions using SHAP and LIME"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
        
        # Get label mappings
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        # Initialize explainers
        self.lime_explainer = None
        if LIME_AVAILABLE:
            self.lime_explainer = LimeTextExplainer(class_names=list(self.label2id.keys()))
    
    def predict_with_confidence(self, texts: List[str]) -> List[Dict]:
        """Get predictions with confidence scores"""
        results = []
        
        for text in texts:
            predictions = self.pipeline(text)
            
            # Calculate overall confidence
            confidences = [pred['score'] for pred in predictions]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            results.append({
                'text': text,
                'predictions': predictions,
                'avg_confidence': avg_confidence,
                'num_entities': len(predictions)
            })
        
        return results
    
    def explain_with_lime(self, text: str, num_features: int = 10) -> Dict:
        """Explain prediction using LIME"""
        if not LIME_AVAILABLE:
            return {"error": "LIME not available"}
        
        def predict_proba(texts):
            """Prediction function for LIME"""
            results = []
            for t in texts:
                preds = self.pipeline(t)
                # Simplified: return binary classification (has_entities, no_entities)
                has_entities = 1.0 if preds else 0.0
                results.append([1 - has_entities, has_entities])
            return np.array(results)
        
        try:
            explanation = self.lime_explainer.explain_instance(
                text, predict_proba, num_features=num_features
            )
            
            return {
                'explanation': explanation,
                'features': explanation.as_list(),
                'html': explanation.as_html()
            }
        except Exception as e:
            logger.error(f"LIME explanation failed: {str(e)}")
            return {"error": str(e)}
    
    def explain_with_shap(self, texts: List[str], max_evals: int = 100) -> Dict:
        """Explain predictions using SHAP"""
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not available"}
        
        try:
            # Create a simple prediction function for SHAP
            def model_predict(texts):
                results = []
                for text in texts:
                    preds = self.pipeline(text)
                    # Return entity count as a simple metric
                    entity_count = len(preds)
                    results.append(entity_count)
                return np.array(results)
            
            # Use SHAP's text explainer
            explainer = shap.Explainer(model_predict, self.tokenizer)
            shap_values = explainer(texts[:min(len(texts), max_evals)])
            
            return {
                'shap_values': shap_values,
                'base_values': shap_values.base_values,
                'data': shap_values.data
            }
        except Exception as e:
            logger.error(f"SHAP explanation failed: {str(e)}")
            return {"error": str(e)}
    
    def analyze_difficult_cases(self, texts: List[str], confidence_threshold: float = 0.5) -> Dict:
        """Identify and analyze difficult cases"""
        predictions = self.predict_with_confidence(texts)
        
        # Categorize cases
        easy_cases = []
        difficult_cases = []
        no_entity_cases = []
        
        for pred in predictions:
            if pred['num_entities'] == 0:
                no_entity_cases.append(pred)
            elif pred['avg_confidence'] >= confidence_threshold:
                easy_cases.append(pred)
            else:
                difficult_cases.append(pred)
        
        analysis = {
            'total_cases': len(predictions),
            'easy_cases': len(easy_cases),
            'difficult_cases': len(difficult_cases),
            'no_entity_cases': len(no_entity_cases),
            'difficult_case_examples': difficult_cases[:5],  # Top 5 difficult cases
            'confidence_distribution': [p['avg_confidence'] for p in predictions]
        }
        
        return analysis
    
    def create_confidence_visualization(self, texts: List[str], output_path: str = "models/plots/confidence_analysis.png"):
        """Create confidence analysis visualization"""
        predictions = self.predict_with_confidence(texts)
        
        # Extract confidence scores
        confidences = [p['avg_confidence'] for p in predictions]
        entity_counts = [p['num_entities'] for p in predictions]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confidence distribution
        ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Average Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Prediction Confidence')
        ax1.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
        ax1.legend()
        
        # Confidence vs Entity Count
        ax2.scatter(confidences, entity_counts, alpha=0.6)
        ax2.set_xlabel('Average Confidence Score')
        ax2.set_ylabel('Number of Entities Detected')
        ax2.set_title('Confidence vs Entity Count')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confidence visualization saved to {output_path}")
    
    def generate_interpretability_report(self, texts: List[str], output_dir: str = "models/interpretability/") -> Dict:
        """Generate comprehensive interpretability report"""
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'model_path': self.model_path,
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_texts_analyzed': len(texts)
        }
        
        # Confidence analysis
        logger.info("Analyzing prediction confidence...")
        difficult_cases = self.analyze_difficult_cases(texts)
        report['difficult_cases_analysis'] = difficult_cases
        
        # Create visualizations
        self.create_confidence_visualization(texts, os.path.join(output_dir, 'confidence_analysis.png'))
        
        # LIME explanations for difficult cases
        if LIME_AVAILABLE and difficult_cases['difficult_cases'] > 0:
            logger.info("Generating LIME explanations...")
            lime_explanations = []
            
            for case in difficult_cases['difficult_case_examples'][:3]:  # Top 3 difficult cases
                explanation = self.explain_with_lime(case['text'])
                if 'error' not in explanation:
                    lime_explanations.append({
                        'text': case['text'],
                        'confidence': case['avg_confidence'],
                        'features': explanation['features']
                    })
                    
                    # Save HTML explanation
                    html_path = os.path.join(output_dir, f'lime_explanation_{len(lime_explanations)}.html')
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(explanation['html'])
            
            report['lime_explanations'] = lime_explanations
        
        # SHAP analysis
        if SHAP_AVAILABLE:
            logger.info("Generating SHAP explanations...")
            shap_results = self.explain_with_shap(texts[:10])  # Analyze first 10 texts
            if 'error' not in shap_results:
                report['shap_analysis'] = {
                    'analyzed_texts': len(texts[:10]),
                    'base_values_mean': float(np.mean(shap_results['base_values'])) if hasattr(shap_results['base_values'], '__iter__') else float(shap_results['base_values'])
                }
        
        # Save report
        report_path = os.path.join(output_dir, 'interpretability_report.json')
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Interpretability report saved to {report_path}")
        return report
    
    def explain_single_prediction(self, text: str) -> Dict:
        """Provide detailed explanation for a single prediction"""
        # Get prediction
        predictions = self.pipeline(text)
        
        # Tokenize for analysis
        tokens = self.tokenizer.tokenize(text)
        
        explanation = {
            'text': text,
            'tokens': tokens,
            'predictions': predictions,
            'token_count': len(tokens),
            'entity_count': len(predictions)
        }
        
        # Add confidence analysis
        if predictions:
            confidences = [pred['score'] for pred in predictions]
            explanation['confidence_stats'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        
        # Add LIME explanation if available
        if LIME_AVAILABLE:
            lime_result = self.explain_with_lime(text)
            if 'error' not in lime_result:
                explanation['lime_features'] = lime_result['features']
        
        return explanation


def main():
    """Test interpretability system"""
    # Sample texts for testing
    sample_texts = [
        "ሰላም! የሕፃናት ጠርሙስ ዋጋ 150 ብር ነው። ቦሌ አካባቢ ነው።",
        "አዲስ አበባ ውስጥ የሚሸጥ ልብስ በ 200 ብር",
        "መርካቶ ውስጥ ጫማ 300 ብር",
        "Hello world this is a test message",
        "ፒያሳ አካባቢ ስልክ ETB 5000"
    ]
    
    # Note: This requires a trained model
    try:
        # First train a simple model for testing
        from src.preprocessing.conll_labeler import CoNLLLabeler
        from src.ner.model_trainer import NERModelTrainer
        
        labeler = CoNLLLabeler()
        sample_data = labeler.create_sample_labeled_data()
        conll_file = labeler.save_conll_format(sample_data)
        
        trainer = NERModelTrainer()
        sentences, labels = trainer.load_conll_data(conll_file)
        train_dataset, val_dataset = trainer.prepare_dataset(sentences, labels)
        
        model_path = trainer.train_model("distilbert-base-multilingual-cased", train_dataset, val_dataset)
        
        # Now test interpretability
        explainer = NERModelExplainer(model_path)
        
        # Generate interpretability report
        report = explainer.generate_interpretability_report(sample_texts)
        
        # Explain single prediction
        single_explanation = explainer.explain_single_prediction(sample_texts[0])
        
        print("Interpretability Analysis Complete!")
        print(f"Analyzed {len(sample_texts)} texts")
        print(f"Difficult cases: {report.get('difficult_cases_analysis', {}).get('difficult_cases', 0)}")
        
    except Exception as e:
        logger.error(f"Error in interpretability testing: {str(e)}")
        print("Note: This requires a trained model. Run the training pipeline first.")


if __name__ == "__main__":
    main()