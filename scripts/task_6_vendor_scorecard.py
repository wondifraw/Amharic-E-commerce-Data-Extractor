"""
task_6_vendor_scorecard.py
Main script to generate a vendor scorecard for micro-lending decisions.

This script performs the following steps for each vendor:
1.  Loads the processed data (and optionally, NER results).
2.  Computes key performance metrics (e.g., posting frequency, average views).
3.  Calculates a "Lending Score".
4.  Compiles the results into a final summary CSV file.
"""
import os
import json
import glob
import argparse
import logging
import sys
import platform
from datetime import datetime
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from src.vendor.analytics import compute_vendor_metrics, calculate_lending_score

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("task_6_vendor_scorecard")

# --- Configuration ---
DEFAULT_PROCESSED_DATA_DIR = 'data/processed'
DEFAULT_COMPARISON_SUMMARY_PATH = 'models/comparison_results/comparison_summary.json'
DEFAULT_SCORECARD_OUTPUT_PATH = 'reports/vendor_scorecard.csv'
DEFAULT_SCORECARD_REPORT_PATH = 'reports/vendor_scorecard_report.json'

def log_environment_info():
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Datetime: {datetime.now().isoformat()}")

def load_best_ner_pipeline(summary_path: str):
    """
    Loads the best fine-tuned NER model as a Hugging Face pipeline.
    """
    if not os.path.exists(summary_path):
        logger.warning(f"Model comparison summary not found at '{summary_path}'. Cannot perform NER.")
        return None
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        best_model_name = summary.get("best_model", {}).get("name")
        if not best_model_name:
            logger.warning("Could not determine the best model from the summary file.")
            return None
        model_dir = os.path.join('models/comparison_results', best_model_name.replace("/", "_"))
        logger.info(f"Loading best NER model '{best_model_name}' from: {model_dir}")
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    except Exception as e:
        logger.error(f"Failed to load the best NER model: {e}", exc_info=True)
        return None

def run_ner_on_messages(ner_pipeline, messages: list) -> list:
    """
    Runs NER inference on a list of message texts.
    """
    if not ner_pipeline:
        return [{} for _ in messages] # Return empty results if no pipeline
    texts = [msg.get('cleaned_text', '') for msg in messages]
    valid_texts = [text for text in texts if text]
    if not valid_texts:
        return [{} for _ in messages]
    logger.info(f"Running NER on {len(valid_texts)} messages...")
    try:
        ner_results = ner_pipeline(valid_texts)
        result_iter = iter(ner_results)
        return [{"entities": next(result_iter, [])} for text in texts if text]
    except Exception as e:
        logger.error(f"An error occurred during NER inference: {e}", exc_info=True)
        return [{} for _ in messages]

def main(processed_data_dir, summary_path, scorecard_output_path, scorecard_report_path, use_ner: bool):
    """
    Main function to execute the vendor analytics and scorecard pipeline.
    """
    logger.info("--- Starting Vendor Scorecard Generation ---")
    log_environment_info()
    logger.info(f"Parameters: processed_data_dir={processed_data_dir}, summary_path={summary_path}, scorecard_output_path={scorecard_output_path}, use_ner={use_ner}")
    ner_pipeline = None
    if use_ner:
        ner_pipeline = load_best_ner_pipeline(summary_path)
    vendor_scorecards = []
    processed_files = glob.glob(os.path.join(processed_data_dir, '*_processed.json'))
    if not processed_files:
        logger.error("No processed data files found. Please run Task 1 first.")
        return
    for file_path in processed_files:
        vendor_name = os.path.basename(file_path).replace('_processed.json', '')
        logger.info(f"--- Analyzing Vendor: {vendor_name} ---")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                messages = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load or parse data for {vendor_name}. Skipping. Error: {e}")
            continue
        ner_results = run_ner_on_messages(ner_pipeline, messages) if use_ner else None
        metrics = compute_vendor_metrics(messages, ner_results)
        lending_score = calculate_lending_score(metrics)
        scorecard = {"Vendor": vendor_name, **metrics, "Lending Score": lending_score}
        vendor_scorecards.append(scorecard)
    if not vendor_scorecards:
        logger.error("No vendor scorecards could be generated.")
        return
    # Create and save the final DataFrame
    df = pd.DataFrame(vendor_scorecards)
    # Output validation
    required_columns = ["Vendor", "Lending Score", "avg_views_per_post", "posting_frequency_per_week", "avg_price_point", "total_posts", "top_performing_post"]
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing column in scorecard: {col}")
            df[col] = None
    df = df[required_columns]
    df = df.sort_values(by="Lending Score", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(scorecard_output_path), exist_ok=True)
    df.to_csv(scorecard_output_path, index=False)
    logger.info(f"Scorecard saved to: {scorecard_output_path}")
    # Summary statistics
    logger.info("\nTop 5 Vendors by Lending Score:")
    logger.info(f"\n{df.head()}\n")
    logger.info(f"Scorecard shape: {df.shape}")
    logger.info(f"Lending Score stats: min={df['Lending Score'].min()}, max={df['Lending Score'].max()}, mean={df['Lending Score'].mean()}")
    # Save detailed report
    try:
        report = {
            "parameters": {
                "processed_data_dir": processed_data_dir,
                "summary_path": summary_path,
                "scorecard_output_path": scorecard_output_path,
                "use_ner": use_ner
            },
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "datetime": datetime.now().isoformat()
            },
            "scorecard_shape": df.shape,
            "lending_score_stats": {
                "min": df['Lending Score'].min(),
                "max": df['Lending Score'].max(),
                "mean": df['Lending Score'].mean()
            },
            "top_5_vendors": df.head().to_dict(orient='records')
        }
        os.makedirs(os.path.dirname(scorecard_report_path), exist_ok=True)
        with open(scorecard_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed scorecard report saved to {scorecard_report_path}")
    except Exception as e:
        logger.error(f"Failed to save scorecard report: {e}", exc_info=True)
    logger.info("--- Vendor Scorecard Generation Finished Successfully! ---")
    print("\n--- Vendor Scorecard Generation Finished Successfully! ---")
    print(f"Scorecard saved to: {scorecard_output_path}")
    print("\nTop 5 Vendors by Lending Score:")
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a vendor scorecard for micro-lending.")
    parser.add_argument('--processed_data_dir', type=str, default=DEFAULT_PROCESSED_DATA_DIR, help='Directory containing processed vendor data.')
    parser.add_argument('--summary_path', type=str, default=DEFAULT_COMPARISON_SUMMARY_PATH, help='Path to model comparison summary JSON.')
    parser.add_argument('--scorecard_output_path', type=str, default=DEFAULT_SCORECARD_OUTPUT_PATH, help='Path to save the vendor scorecard CSV.')
    parser.add_argument('--scorecard_report_path', type=str, default=DEFAULT_SCORECARD_REPORT_PATH, help='Path to save the detailed scorecard report JSON.')
    parser.add_argument('--disable_ner', action='store_true', help='Disable NER inference. The scorecard will be generated without price information.')
    args = parser.parse_args()
    main(
        processed_data_dir=args.processed_data_dir,
        summary_path=args.summary_path,
        scorecard_output_path=args.scorecard_output_path,
        scorecard_report_path=args.scorecard_report_path,
        use_ner=not args.disable_ner
    ) 