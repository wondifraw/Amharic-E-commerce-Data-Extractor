"""
Quick Start Script for Ethiopian Telegram NER System
Demonstrates the complete pipeline with sample data
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.text_preprocessor import AmharicTextPreprocessor
from preprocessing.conll_labeler import CoNLLLabeler
from ner.model_trainer import NERModelTrainer
from evaluation.model_comparator import ModelComparator
from vendor_analytics.scorecard import VendorScorecard
import pandas as pd
import numpy as np
from loguru import logger


def create_sample_data():
    """Create sample data for demonstration"""
    logger.info("Creating sample data...")
    
    # Sample Ethiopian e-commerce messages
    sample_messages = [
        "ሰላም! የሕፃናት ጠርሙስ ዋጋ 150 ብር ነው። ቦሌ አካባቢ ነው።",
        "አዲስ አበባ ውስጥ የሚሸጥ ልብስ በ 200 ብር",
        "መርካቶ ውስጥ ጫማ 300 ብር ነው",
        "ፒያሳ አካባቢ ስልክ ETB 5000",
        "ሃያ ሁለት ላይ መጽሐፍ 50 ብር ነው",
        "Baby bottle for sale 150 birr in Bole area",
        "Phone available at Merkato 4000 birr",
        "Clothes በ 180 ብር አዲስ አበባ ውስጥ",
        "ካዛንቺስ አካባቢ shoes 250 ብር",
        "ጀሞ ላይ የሚሸጥ ልብስ ዋጋ 120 ብር",
        "ሰሚት አካባቢ የሚሸጥ ስልክ 3000 ብር",
        "መጽሐፍ በ 45 ብር ቦሌ ላይ",
        "አዲስ አበባ ውስጥ ጫማ ETB 280",
        "ልብስ ዋጋ 160 ብር መርካቶ ውስጥ",
        "ፒያሳ ላይ baby bottle 140 birr"
    ] * 4  # Repeat to have more data
    
    # Create sample DataFrame
    channels = ["@ethio_market_place", "@addis_shopping", "@bole_market", "@merkato_online", "@ethiopia_deals"]
    
    data = {
        'id': range(1, len(sample_messages) + 1),
        'channel': np.random.choice(channels, len(sample_messages)),
        'text': sample_messages,
        'date': pd.date_range('2024-01-01', periods=len(sample_messages), freq='H'),
        'views': np.random.randint(50, 1000, len(sample_messages)),
        'forwards': np.random.randint(0, 50, len(sample_messages)),
        'replies': np.random.randint(0, 20, len(sample_messages)),
        'sender_id': np.random.randint(1000, 9999, len(sample_messages)),
        'has_media': np.random.choice([True, False], len(sample_messages), p=[0.3, 0.7]),
        'media_type': np.random.choice(['photo', 'document', None], len(sample_messages), p=[0.2, 0.1, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Save sample data
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/sample_telegram_data.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"Sample data created: {output_path}")
    return output_path


def demonstrate_preprocessing(data_path):
    """Demonstrate text preprocessing"""
    logger.info("Demonstrating text preprocessing...")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize preprocessor
    preprocessor = AmharicTextPreprocessor()
    
    # Preprocess data
    processed_df = preprocessor.preprocess_dataframe(df)
    
    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    processed_path = "data/processed/sample_processed_data.csv"
    processed_df.to_csv(processed_path, index=False, encoding='utf-8')
    
    # Show sample results
    print("\n" + "="*50)
    print("PREPROCESSING RESULTS")
    print("="*50)
    print(f"Original messages: {len(df)}")
    print(f"Processed messages: {len(processed_df)}")
    print(f"Messages with Amharic: {processed_df['has_amharic'].sum()}")
    print(f"Average token count: {processed_df['token_count'].mean():.1f}")
    
    # Show sample
    print("\nSample processed message:")
    sample = processed_df.iloc[0]
    print(f"Original: {sample['text'][:100]}...")
    print(f"Cleaned: {sample['cleaned_text'][:100]}...")
    print(f"Tokens: {sample['tokens'][:10]}")
    print(f"Price hints: {sample['price_hints']}")
    print(f"Location hints: {sample['location_hints']}")
    
    return processed_path


def demonstrate_conll_labeling():
    """Demonstrate CoNLL format labeling"""
    logger.info("Demonstrating CoNLL labeling...")
    
    # Initialize labeler
    labeler = CoNLLLabeler()
    
    # Create labeled dataset
    labeled_data = labeler.create_sample_labeled_data()
    
    # Extend with more examples
    additional_messages = [
        "ሰሚት አካባቢ የሚሸጥ ስልክ 3000 ብር",
        "መጽሐፍ በ 45 ብር ቦሌ ላይ",
        "አዲስ አበባ ውስጥ ጫማ ETB 280"
    ]
    
    extended_data = labeler.create_extended_dataset(additional_messages, target_size=30)
    
    # Save CoNLL format
    os.makedirs("data/labeled", exist_ok=True)
    conll_path = "data/labeled/sample_dataset.txt"
    labeler.save_conll_format(extended_data, conll_path)
    
    print("\n" + "="*50)
    print("CONLL LABELING RESULTS")
    print("="*50)
    print(f"Labeled sentences: {len(extended_data)}")
    
    # Show sample labeled data
    print("\nSample labeled sentence:")
    message, tokens, labels = extended_data[0]
    print(f"Message: {message}")
    print("Tokens and Labels:")
    for token, label in zip(tokens[:10], labels[:10]):
        print(f"  {token:<15} {label}")
    
    return conll_path


def demonstrate_model_training(conll_path):
    """Demonstrate model training (simplified)"""
    logger.info("Demonstrating model training setup...")
    
    # Initialize trainer
    trainer = NERModelTrainer()
    
    # Load CoNLL data
    sentences, labels = trainer.load_conll_data(conll_path)
    
    # Prepare datasets
    train_dataset, val_dataset = trainer.prepare_dataset(sentences, labels)
    
    print("\n" + "="*50)
    print("MODEL TRAINING SETUP")
    print("="*50)
    print(f"Total sentences: {len(sentences)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Entity labels: {trainer.label_list}")
    
    # Show sample training data structure
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"\nSample training item keys: {list(sample.keys())}")
        print(f"Sample tokens: {sample['tokens'][:10]}")
        print(f"Sample labels: {sample['labels'][:10]}")
    
    print("\nNote: Actual model training skipped in quick start demo")
    print("To train models, run: python scripts/main_pipeline.py --step training")
    
    return "models/sample_model"  # Placeholder


def demonstrate_vendor_analytics(processed_data_path):
    """Demonstrate vendor analytics"""
    logger.info("Demonstrating vendor analytics...")
    
    # Load processed data
    df = pd.read_csv(processed_data_path)
    
    # Initialize scorecard
    scorecard = VendorScorecard()
    
    # Analyze all vendors
    vendor_analyses = scorecard.analyze_all_vendors(df)
    
    # Create comparison table
    comparison_table = scorecard.create_vendor_comparison_table(vendor_analyses)
    
    # Save results
    os.makedirs("models/vendor_analytics", exist_ok=True)
    comparison_table.to_csv("models/vendor_analytics/sample_vendor_comparison.csv", index=False)
    
    print("\n" + "="*50)
    print("VENDOR ANALYTICS RESULTS")
    print("="*50)
    print(f"Vendors analyzed: {len(vendor_analyses)}")
    
    # Show comparison table
    print("\nVendor Scorecard:")
    print(comparison_table.to_string(index=False))
    
    # Show detailed analysis for top vendor
    if len(comparison_table) > 0:
        top_vendor_name = comparison_table.iloc[0]['Vendor']
        top_vendor_channel = f"@{top_vendor_name}"
        
        if top_vendor_channel in vendor_analyses:
            analysis = vendor_analyses[top_vendor_channel]
            print(f"\nDetailed Analysis for Top Vendor ({top_vendor_name}):")
            print(f"  Lending Score: {analysis['lending_score']:.1f}")
            print(f"  Risk Category: {analysis['risk_category']}")
            print(f"  Total Posts: {analysis['total_posts']}")
            print(f"  Posting Frequency: {analysis['posting_frequency']:.1f} posts/week")
            print(f"  Average Views: {analysis['engagement']['avg_views']:.0f}")
            print(f"  Average Price: {analysis['price_metrics']['avg_price']:.0f} ETB")
    
    return vendor_analyses


def demonstrate_interpretability():
    """Demonstrate model interpretability (simplified)"""
    logger.info("Demonstrating interpretability concepts...")
    
    print("\n" + "="*50)
    print("MODEL INTERPRETABILITY")
    print("="*50)
    
    sample_texts = [
        "ሰላም! የሕፃናት ጠርሙስ ዋጋ 150 ብር ነው። ቦሌ አካባቢ ነው።",
        "አዲስ አበባ ውስጥ የሚሸጥ ልብስ በ 200 ብር",
        "Hello world this is a test message"
    ]
    
    print("Sample texts for interpretability analysis:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    
    print("\nInterpretability features available:")
    print("✓ SHAP (SHapley Additive exPlanations)")
    print("✓ LIME (Local Interpretable Model-agnostic Explanations)")
    print("✓ Confidence analysis")
    print("✓ Difficult case identification")
    
    print("\nNote: Full interpretability analysis requires trained model")
    print("To run interpretability analysis, use a trained model with:")
    print("python scripts/main_pipeline.py --step interpretability --model-path <model_path>")


def main():
    """Run the complete quick start demonstration"""
    print("🛍️ Ethiopian Telegram NER System - Quick Start Demo")
    print("="*60)
    
    try:
        # Step 1: Create sample data
        data_path = create_sample_data()
        
        # Step 2: Demonstrate preprocessing
        processed_path = demonstrate_preprocessing(data_path)
        
        # Step 3: Demonstrate CoNLL labeling
        conll_path = demonstrate_conll_labeling()
        
        # Step 4: Demonstrate model training setup
        model_path = demonstrate_model_training(conll_path)
        
        # Step 5: Demonstrate vendor analytics
        vendor_analyses = demonstrate_vendor_analytics(processed_path)
        
        # Step 6: Demonstrate interpretability concepts
        demonstrate_interpretability()
        
        # Summary
        print("\n" + "="*60)
        print("🎉 QUICK START DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📁 Generated Files:")
        print(f"  • Raw data: {data_path}")
        print(f"  • Processed data: {processed_path}")
        print(f"  • CoNLL dataset: {conll_path}")
        print(f"  • Vendor comparison: models/vendor_analytics/sample_vendor_comparison.csv")
        
        print("\n🚀 Next Steps:")
        print("1. Configure Telegram API credentials in config/config.yaml")
        print("2. Run full pipeline: python scripts/main_pipeline.py --step full")
        print("3. Start dashboard: streamlit run src/dashboard/streamlit_app.py")
        print("4. Train models: python scripts/main_pipeline.py --step training")
        print("5. Run tests: make test")
        
        print("\n📚 Documentation:")
        print("  • README.md - Project overview and setup")
        print("  • CONTRIBUTING.md - Development guidelines")
        print("  • config/config.yaml - Configuration options")
        
    except Exception as e:
        logger.error(f"Quick start demo failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease check the logs and ensure all dependencies are installed.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())