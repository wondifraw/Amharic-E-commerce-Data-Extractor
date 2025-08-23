"""Script to run individual pipeline tasks."""

import asyncio
import argparse
import pandas as pd
from pathlib import Path

from main_pipeline import AmharicNERPipeline

async def main():
    parser = argparse.ArgumentParser(description='Run individual pipeline tasks')
    parser.add_argument('--task', choices=['scrape', 'label', 'train', 'evaluate', 'analytics'], 
                       required=True, help='Task to run')
    parser.add_argument('--input', help='Input file path')
    parser.add_argument('--model', help='Model name for training/evaluation')
    
    args = parser.parse_args()
    pipeline = AmharicNERPipeline()
    
    if args.task == 'scrape':
        df = await pipeline.run_data_ingestion()
        print(f"Scraped {len(df)} messages")
        
    elif args.task == 'label':
        if args.input and Path(args.input).exists():
            df = pd.read_csv(args.input)
            conll_data = pipeline.run_labeling(df)
            print("Labeling completed")
        else:
            print("Input file required for labeling")
            
    elif args.task == 'train':
        if args.input and Path(args.input).exists():
            with open(args.input, 'r', encoding='utf-8') as f:
                conll_data = f.read()
            pipeline.run_training(conll_data)
            print("Training completed")
        else:
            print("CoNLL input file required for training")
            
    elif args.task == 'evaluate':
        if args.input and Path(args.input).exists():
            with open(args.input, 'r', encoding='utf-8') as f:
                conll_data = f.read()
            results = pipeline.run_evaluation(conll_data)
            print("Evaluation completed")
            print(results)
        else:
            print("CoNLL input file required for evaluation")
            
    elif args.task == 'analytics':
        if args.input and Path(args.input).exists():
            df = pd.read_csv(args.input)
            pipeline.run_vendor_analytics(df)
            print("Analytics completed")
        else:
            print("CSV input file required for analytics")

if __name__ == "__main__":
    asyncio.run(main())