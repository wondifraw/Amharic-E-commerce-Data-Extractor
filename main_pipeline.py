"""Main pipeline for Amharic NER project."""

import asyncio
import pandas as pd
import logging
from pathlib import Path

from src.data_ingestion.telegram_scraper import TelegramScraper
from src.preprocessing.text_processor import AmharicTextProcessor
from src.labeling.conll_labeler import CoNLLLabeler
from src.training.ner_trainer import NERTrainer
from src.evaluation.model_evaluator import ModelEvaluator
from src.interpretability.model_explainer import NERExplainer
from src.vendor_analytics.scorecard import VendorAnalytics
from config.config import model_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmharicNERPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self):
        self.scraper = TelegramScraper()
        self.processor = AmharicTextProcessor()
        self.labeler = CoNLLLabeler()
        self.evaluator = ModelEvaluator()
        self.analytics = VendorAnalytics()
        
    async def run_data_ingestion(self) -> pd.DataFrame:
        """Task 1: Data ingestion and preprocessing."""
        logger.info("Starting data ingestion...")
        
        # Scrape data
        df = await self.scraper.scrape_all_channels(limit_per_channel=200)
        
        if df.empty:
            logger.error("No data scraped")
            return df
            
        # Save raw data
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        df.to_csv("data/raw/telegram_messages.csv", index=False)
        
        # Preprocess
        df = self.processor.preprocess_dataset(df)
        
        # Save processed data
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        df.to_csv("data/processed/cleaned_messages.csv", index=False)
        
        logger.info(f"Data ingestion completed: {len(df)} messages")
        return df
    
    def run_labeling(self, df: pd.DataFrame) -> str:
        """Task 2: Label dataset in CoNLL format."""
        logger.info("Starting data labeling...")
        
        conll_data = self.labeler.label_dataset_sample(df, num_messages=50)
        
        # Save labeled data
        Path("data/labeled").mkdir(parents=True, exist_ok=True)
        self.labeler.save_conll_data(conll_data, "data/labeled/train_data.conll")
        
        logger.info("Data labeling completed")
        return conll_data
    
    def run_training(self, conll_data: str) -> None:
        """Task 3: Fine-tune NER models."""
        logger.info("Starting model training...")
        
        Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
        
        for model_name in model_config.model_names:
            try:
                logger.info(f"Training {model_name}...")
                trainer = NERTrainer(model_name)
                output_dir = f"models/checkpoints/{model_name.replace('/', '_')}"
                trainer.train(conll_data, output_dir)
            except Exception as e:
                logger.error(f"Training failed for {model_name}: {e}")
    
    def run_evaluation(self, conll_data: str) -> pd.DataFrame:
        """Task 4: Model comparison and selection."""
        logger.info("Starting model evaluation...")
        
        # Load trained models
        for model_name in model_config.model_names:
            model_path = f"models/checkpoints/{model_name.replace('/', '_')}"
            if Path(model_path).exists():
                self.evaluator.load_model(model_path, model_name)
        
        # Prepare test data
        test_data = self.labeler.parse_conll_data(conll_data)
        test_texts = [' '.join(ex['tokens']) for ex in test_data]
        
        # Compare models
        results_df = self.evaluator.compare_models(test_data, test_texts)
        
        # Save results
        self.evaluator.save_comparison_results(results_df, "models/comparison_results.csv")
        
        logger.info("Model evaluation completed")
        return results_df
    
    def run_interpretability(self, conll_data: str, best_model_name: str) -> None:
        """Task 5: Model interpretability analysis."""
        logger.info("Starting interpretability analysis...")
        
        try:
            from transformers import pipeline
            model_path = f"models/checkpoints/{best_model_name.replace('/', '_')}"
            
            if Path(model_path).exists():
                model_pipeline = pipeline('ner', model=model_path, aggregation_strategy='simple')
                explainer = NERExplainer(model_pipeline)
                
                test_data = self.labeler.parse_conll_data(conll_data)
                explainer.generate_interpretability_report(test_data, "models/interpretability_report.json")
                
                logger.info("Interpretability analysis completed")
            else:
                logger.warning(f"Model not found: {model_path}")
                
        except Exception as e:
            logger.error(f"Interpretability analysis failed: {e}")
    
    def run_vendor_analytics(self, df: pd.DataFrame) -> None:
        """Task 6: Vendor scorecard generation."""
        logger.info("Starting vendor analytics...")
        
        # Generate scorecard
        scorecard_df = self.analytics.generate_vendor_scorecard(df)
        
        # Generate detailed report
        self.analytics.generate_detailed_report(df, "data/processed/vendor_analytics_report.json")
        
        # Display summary
        if not scorecard_df.empty:
            print("\n=== VENDOR SCORECARD SUMMARY ===")
            print(scorecard_df[['Vendor_Channel', 'Avg_Views_Per_Post', 'Posts_Per_Week', 
                              'Avg_Price_ETB', 'Lending_Score', 'Risk_Category']].head(10))
        
        logger.info("Vendor analytics completed")
    
    async def run_full_pipeline(self) -> None:
        """Run complete pipeline."""
        try:
            # Task 1: Data Ingestion
            df = await self.run_data_ingestion()
            if df.empty:
                return
            
            # Task 2: Labeling
            conll_data = self.run_labeling(df)
            
            # Task 3: Training
            self.run_training(conll_data)
            
            # Task 4: Evaluation
            results_df = self.run_evaluation(conll_data)
            
            # Task 5: Interpretability
            if not results_df.empty:
                best_model = results_df.iloc[0]['model']
                self.run_interpretability(conll_data, best_model)
            
            # Task 6: Vendor Analytics
            self.run_vendor_analytics(df)
            
            logger.info("Full pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")

async def main():
    """Main entry point."""
    pipeline = AmharicNERPipeline()
    await pipeline.run_full_pipeline()

if __name__ == "__main__":
    asyncio.run(main())