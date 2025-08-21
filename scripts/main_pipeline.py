"""
Main Pipeline for Ethiopian Telegram NER System
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from loguru import logger
from src.data_ingestion.telegram_scraper import TelegramScraper
from src.preprocessing.text_preprocessor import AmharicTextPreprocessor
from src.preprocessing.conll_labeler import CoNLLLabeler
from src.ner.model_trainer import NERModelTrainer
from src.evaluation.model_comparator import ModelComparator
from src.interpretability.model_explainer import NERModelExplainer
from src.vendor_analytics.scorecard import VendorScorecard


class EthiopianNERPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.scraper = None
        self.preprocessor = AmharicTextPreprocessor()
        self.labeler = CoNLLLabeler()
        self.trainer = NERModelTrainer()
        self.comparator = ModelComparator()
        self.scorecard = VendorScorecard()
        
        # Setup logging
        logger.add("logs/pipeline_{time}.log", rotation="1 day", retention="7 days")
    
    async def run_data_ingestion(self, limit: int = 1000):
        """Run data ingestion step"""
        logger.info("Starting data ingestion...")
        
        self.scraper = TelegramScraper()
        await self.scraper.initialize_client()
        
        try:
            df = await self.scraper.scrape_all_channels(limit_per_channel=limit)
            output_path = await self.scraper.save_raw_data(df)
            logger.info(f"Data ingestion complete. Saved to {output_path}")
            return output_path
        finally:
            await self.scraper.close()
    
    def run_preprocessing(self, data_path: str):
        """Run preprocessing step"""
        logger.info("Starting preprocessing...")
        
        import pandas as pd
        df = pd.read_csv(data_path)
        
        # Preprocess data
        processed_df = self.preprocessor.preprocess_dataframe(df)
        
        # Save processed data
        output_path = data_path.replace('raw', 'processed').replace('.csv', '_processed.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Prepare sample for labeling
        sample_df = self.preprocessor.prepare_for_labeling(processed_df, sample_size=50)
        sample_path = output_path.replace('_processed.csv', '_sample_for_labeling.csv')
        sample_df.to_csv(sample_path, index=False, encoding='utf-8')
        
        logger.info(f"Preprocessing complete. Processed data: {output_path}")
        logger.info(f"Sample for labeling: {sample_path}")
        
        return output_path, sample_path
    
    def run_labeling(self, sample_path: str = None):
        """Run CoNLL labeling step"""
        logger.info("Starting CoNLL labeling...")
        
        if sample_path:
            import pandas as pd
            sample_df = pd.read_csv(sample_path)
            messages = sample_df['cleaned_text'].tolist()
            labeled_data = self.labeler.create_extended_dataset(messages, target_size=50)
        else:
            labeled_data = self.labeler.create_sample_labeled_data()
        
        # Save in CoNLL format
        conll_path = "data/labeled/dataset.txt"
        self.labeler.save_conll_format(labeled_data, conll_path)
        
        logger.info(f"CoNLL labeling complete. Dataset saved to {conll_path}")
        return conll_path
    
    def run_training(self, data_path: str):
        """Run model training step"""
        logger.info("Starting model training...")
        
        # Load and prepare data
        sentences, labels = self.trainer.load_conll_data(data_path)
        train_dataset, val_dataset = self.trainer.prepare_dataset(sentences, labels)
        
        # Train best model (start with one for testing)
        model_name = "distilbert-base-multilingual-cased"
        model_path = self.trainer.train_model(model_name, train_dataset, val_dataset)
        
        logger.info(f"Model training complete. Model saved to {model_path}")
        return model_path
    
    def run_model_comparison(self, data_path: str):
        """Run model comparison step"""
        logger.info("Starting model comparison...")
        
        # Compare models (use subset for faster testing)
        test_models = [
            "distilbert-base-multilingual-cased",
            # Add more models as needed
        ]
        
        results = self.comparator.compare_models(data_path, test_models)
        
        # Generate report and visualizations
        report = self.comparator.create_comparison_report()
        self.comparator.visualize_comparison()
        
        # Get best model
        best_model, best_result = self.comparator.select_best_model()
        
        logger.info(f"Model comparison complete. Best model: {best_model}")
        return best_model, best_result
    
    def run_interpretability(self, model_path: str, sample_texts: list = None):
        """Run model interpretability analysis"""
        logger.info("Starting interpretability analysis...")
        
        if sample_texts is None:
            sample_texts = [
                "ሰላም! የሕፃናት ጠርሙስ ዋጋ 150 ብር ነው። ቦሌ አካባቢ ነው።",
                "አዲስ አበባ ውስጥ የሚሸጥ ልብስ በ 200 ብር",
                "መርካቶ ውስጥ ጫማ 300 ብር",
                "ፒያሳ አካባቢ ስልክ ETB 5000",
                "Baby bottle for sale 150 birr in Bole"
            ]
        
        explainer = NERModelExplainer(model_path)
        report = explainer.generate_interpretability_report(sample_texts)
        
        logger.info("Interpretability analysis complete")
        return report
    
    def run_vendor_analytics(self, data_path: str):
        """Run vendor analytics step"""
        logger.info("Starting vendor analytics...")
        
        import pandas as pd
        df = pd.read_csv(data_path)
        
        # Analyze all vendors
        vendor_analyses = self.scorecard.analyze_all_vendors(df)
        
        # Create comparison table
        comparison_table = self.scorecard.create_vendor_comparison_table(vendor_analyses)
        
        # Save comparison table
        table_path = "models/vendor_analytics/vendor_comparison.csv"
        os.makedirs(os.path.dirname(table_path), exist_ok=True)
        comparison_table.to_csv(table_path, index=False)
        
        # Create visualizations
        self.scorecard.create_visualizations(vendor_analyses)
        
        # Generate report
        report = self.scorecard.generate_lending_report(vendor_analyses)
        
        logger.info("Vendor analytics complete")
        return vendor_analyses, comparison_table
    
    async def run_full_pipeline(self, limit: int = 500):
        """Run the complete pipeline"""
        logger.info("Starting full pipeline...")
        
        try:
            # Step 1: Data Ingestion
            raw_data_path = await self.run_data_ingestion(limit)
            
            # Step 2: Preprocessing
            processed_data_path, sample_path = self.run_preprocessing(raw_data_path)
            
            # Step 3: Labeling
            conll_path = self.run_labeling(sample_path)
            
            # Step 4: Training
            model_path = self.run_training(conll_path)
            
            # Step 5: Model Comparison (if multiple models)
            # best_model, best_result = self.run_model_comparison(conll_path)
            
            # Step 6: Interpretability
            interpretability_report = self.run_interpretability(model_path)
            
            # Step 7: Vendor Analytics
            vendor_analyses, comparison_table = self.run_vendor_analytics(processed_data_path)
            
            logger.info("Full pipeline completed successfully!")
            
            # Print summary
            print("\n" + "="*50)
            print("PIPELINE EXECUTION SUMMARY")
            print("="*50)
            print(f"✅ Data ingested and saved to: {raw_data_path}")
            print(f"✅ Data preprocessed and saved to: {processed_data_path}")
            print(f"✅ CoNLL dataset created: {conll_path}")
            print(f"✅ Model trained and saved to: {model_path}")
            print(f"✅ Interpretability analysis completed")
            print(f"✅ Vendor analytics completed")
            print("\n📊 Vendor Scorecard Summary:")
            print(comparison_table.to_string(index=False))
            print("\n🎯 Next steps:")
            print("- Run the Streamlit dashboard: streamlit run src/dashboard/streamlit_app.py")
            print("- Check model performance in models/comparison_report.json")
            print("- Review vendor analytics in models/vendor_analytics/")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


async def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Ethiopian Telegram NER Pipeline")
    parser.add_argument("--step", choices=["full", "ingestion", "preprocessing", "labeling", "training", "comparison", "interpretability", "analytics"], 
                       default="full", help="Pipeline step to run")
    parser.add_argument("--limit", type=int, default=500, help="Limit for data ingestion")
    parser.add_argument("--data-path", type=str, help="Path to data file")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    
    args = parser.parse_args()
    
    pipeline = EthiopianNERPipeline()
    
    try:
        if args.step == "full":
            await pipeline.run_full_pipeline(args.limit)
        
        elif args.step == "ingestion":
            await pipeline.run_data_ingestion(args.limit)
        
        elif args.step == "preprocessing":
            if not args.data_path:
                print("Error: --data-path required for preprocessing step")
                return
            pipeline.run_preprocessing(args.data_path)
        
        elif args.step == "labeling":
            pipeline.run_labeling(args.data_path)
        
        elif args.step == "training":
            if not args.data_path:
                print("Error: --data-path required for training step")
                return
            pipeline.run_training(args.data_path)
        
        elif args.step == "comparison":
            if not args.data_path:
                print("Error: --data-path required for comparison step")
                return
            pipeline.run_model_comparison(args.data_path)
        
        elif args.step == "interpretability":
            if not args.model_path:
                print("Error: --model-path required for interpretability step")
                return
            pipeline.run_interpretability(args.model_path)
        
        elif args.step == "analytics":
            if not args.data_path:
                print("Error: --data-path required for analytics step")
                return
            pipeline.run_vendor_analytics(args.data_path)
    
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())