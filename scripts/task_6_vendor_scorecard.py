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
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from src.vendor.analytics import compute_vendor_metrics, calculate_lending_score

# --- Configuration ---
PROCESSED_DATA_DIR = 'data/processed'
COMPARISON_SUMMARY_PATH = 'models/comparison_results/comparison_summary.json'
SCORECARD_OUTPUT_PATH = 'reports/vendor_scorecard.csv'

def load_best_ner_pipeline(summary_path: str):
    """
    Loads the best fine-tuned NER model as a Hugging Face pipeline.
    """
    if not os.path.exists(summary_path):
        print(f"[WARNING] Model comparison summary not found at '{summary_path}'. Cannot perform NER.")
        return None
        
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        best_model_name = summary.get("best_model", {}).get("name")
        if not best_model_name:
            print("[WARNING] Could not determine the best model from the summary file.")
            return None
            
        model_dir = os.path.join('models/comparison_results', best_model_name.replace("/", "_"))
        print(f"Loading best NER model '{best_model_name}' from: {model_dir}")
        
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        
    except Exception as e:
        print(f"[ERROR] Failed to load the best NER model: {e}")
        return None

def run_ner_on_messages(ner_pipeline, messages: list) -> list:
    """
    Runs NER inference on a list of message texts.
    """
    if not ner_pipeline:
        return [{} for _ in messages] # Return empty results if no pipeline
        
    texts = [msg.get('cleaned_text', '') for msg in messages]
    # Filter out empty texts to avoid sending them to the pipeline
    valid_texts = [text for text in texts if text]
    
    if not valid_texts:
        return [{} for _ in messages]

    print(f"Running NER on {len(valid_texts)} messages...")
    try:
        ner_results = ner_pipeline(valid_texts)
        # Re-align results with original messages, handling empty texts
        result_iter = iter(ner_results)
        return [{"entities": next(result_iter, [])} for text in texts if text]
    except Exception as e:
        print(f"[ERROR] An error occurred during NER inference: {e}")
        return [{} for _ in messages]

def main(use_ner: bool):
    """
    Main function to execute the vendor analytics and scorecard pipeline.
    """
    print("--- Starting Vendor Scorecard Generation ---")
    
    ner_pipeline = None
    if use_ner:
        ner_pipeline = load_best_ner_pipeline(COMPARISON_SUMMARY_PATH)

    vendor_scorecards = []
    processed_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, '*_processed.json'))

    if not processed_files:
        print("[ERROR] No processed data files found. Please run Task 1 first.")
        return

    for file_path in processed_files:
        vendor_name = os.path.basename(file_path).replace('_processed.json', '')
        print(f"\n--- Analyzing Vendor: {vendor_name} ---")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                messages = json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not load or parse data for {vendor_name}. Skipping. Error: {e}")
            continue

        ner_results = run_ner_on_messages(ner_pipeline, messages) if use_ner else None
        
        metrics = compute_vendor_metrics(messages, ner_results)
        lending_score = calculate_lending_score(metrics)
        
        scorecard = {"Vendor": vendor_name, **metrics, "Lending Score": lending_score}
        vendor_scorecards.append(scorecard)
        
    if not vendor_scorecards:
        print("[ERROR] No vendor scorecards could be generated.")
        return

    # Create and save the final DataFrame
    df = pd.DataFrame(vendor_scorecards)
    # Reorder columns for better readability
    column_order = ["Vendor", "Lending Score", "avg_views_per_post", "posting_frequency_per_week", "avg_price_point", "total_posts", "top_performing_post"]
    df = df[column_order]
    df = df.sort_values(by="Lending Score", ascending=False).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(SCORECARD_OUTPUT_PATH), exist_ok=True)
    df.to_csv(SCORECARD_OUTPUT_PATH, index=False)
    
    print("\n--- Vendor Scorecard Generation Finished Successfully! ---")
    print(f"Scorecard saved to: {SCORECARD_OUTPUT_PATH}")
    print("\nTop 5 Vendors by Lending Score:")
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a vendor scorecard for micro-lending.")
    parser.add_argument(
        "--disable_ner",
        action="store_true",
        help="Disable NER inference. The scorecard will be generated without price information."
    )
    args = parser.parse_args()
    main(use_ner=not args.disable_ner) 