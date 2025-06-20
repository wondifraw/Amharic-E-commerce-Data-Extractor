"""
task_2_label.py
Script to generate a labeling template and convert labeled data to CoNLL format.

This script provides a two-step workflow for data labeling:
1.  `export`: Creates a JSON file with a sample of messages for manual labeling.
2.  `convert`: Reads the manually labeled JSON file and converts it to CoNLL format.
"""
import json
import os
import random
import argparse
import glob
from src.labeling.conll_formatter import to_conll

PROCESSED_DATA_DIR = 'data/processed'
LABELING_TEMPLATE_PATH = os.path.join(PROCESSED_DATA_DIR, 'labeling_template.json')
CONLL_OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, 'labeled_dataset.conll')
NUM_SAMPLES = 50

LABELING_INSTRUCTIONS = """
// Welcome to the labeling task for NER!
// Please label each token in the 'tokens' list with its corresponding entity tag in the 'labels' list.
// 
// The entity tags are:
//   - B-Product: Beginning of a product name (e.g., "iPhone")
//   - I-Product: Inside a product name (e.g., "13" in "iPhone 13")
//   - B-LOC: Beginning of a location (e.g., "Addis")
//   - I-LOC: Inside a location (e.g., "Ababa" in "Addis Ababa")
//   - B-PRICE: Beginning of a price (e.g., "1000")
//   - I-PRICE: Inside a price (e.g., "ብር" in "1000 ብር")
//   - O: Outside of any entity (for all other tokens)
//
// Example:
//   {"tokens": ["iPhone", "13", "ዋጋ", "1000", "ብር"], "labels": ["B-Product", "I-Product", "O", "B-PRICE", "I-PRICE"]}
//
// After you finish labeling, save this file and run the convert command.
"""

def export_labeling_template():
    """
    Exports a JSON template file with a random sample of messages for labeling.
    """
    print(f"--- Exporting Labeling Template (Sample of {NUM_SAMPLES} messages) ---")
    all_messages = []
    
    # Find all processed JSON files
    processed_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, '*_processed.json'))
    if not processed_files:
        print("[ERROR] No processed data files found in 'data/processed'. Please run Task 1 first.")
        return

    for file_path in processed_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_messages.extend(json.load(f))
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] Could not read or parse {file_path}: {e}")
    
    # Filter for messages that have tokens
    messages_with_tokens = [msg for msg in all_messages if msg.get('tokens')]
    
    if not messages_with_tokens:
        print("[ERROR] No messages with tokens found in the processed data.")
        return
        
    # Get a random sample
    sample_size = min(NUM_SAMPLES, len(messages_with_tokens))
    random_sample = random.sample(messages_with_tokens, sample_size)
    
    # Create the template structure
    labeling_template = [
        {"id": msg.get("id"), "channel": msg.get("channel"), "tokens": msg["tokens"], "labels": ["O"] * len(msg["tokens"])}
        for msg in random_sample
    ]
    
    try:
        with open(LABELING_TEMPLATE_PATH, 'w', encoding='utf-8') as f:
            f.write(LABELING_INSTRUCTIONS)
            json.dump(labeling_template, f, ensure_ascii=False, indent=2)
        print(f"\nSuccessfully created labeling template at: {LABELING_TEMPLATE_PATH}")
        print("Please open this file and update the 'labels' for each token.")
    except IOError as e:
        print(f"[ERROR] Could not write template file: {e}")

def convert_to_conll():
    """
    Reads the labeled JSON template and converts it to a CoNLL file.
    """
    print("--- Converting Labeled Template to CoNLL Format ---")
    if not os.path.exists(LABELING_TEMPLATE_PATH):
        print(f"[ERROR] Labeling template not found at {LABELING_TEMPLATE_PATH}. Please run the 'export' command first.")
        return
        
    try:
        with open(LABELING_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
            # Find where the JSON array starts to ignore the instructions block
            json_start_index = content.find('[')
            if json_start_index == -1:
                raise json.JSONDecodeError("Could not find the start of the JSON array.", content, 0)
            
            labeled_data = json.loads(content[json_start_index:])
            
        to_conll(labeled_data, CONLL_OUTPUT_PATH)
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {LABELING_TEMPLATE_PATH}. Please check the file for syntax errors: {e}")
    except IOError as e:
        print(f"[ERROR] Could not read template file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to help with NER data labeling.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Command to export the template
    parser_export = subparsers.add_parser("export", help="Export a JSON template for manual labeling.")
    parser_export.set_defaults(func=export_labeling_template)

    # Command to convert to CoNLL
    parser_convert = subparsers.add_parser("convert", help="Convert the labeled JSON template to CoNLL format.")
    parser_convert.set_defaults(func=convert_to_conll)

    args = parser.parse_args()
    args.func() 