"""
conll_formatter.py
Module for converting labeled data into CoNLL format for NER tasks.
"""
from typing import List, Dict

def to_conll(labeled_messages: List[Dict], output_path: str):
    """
    Converts a list of labeled message dictionaries to a CoNLL formatted file.

    Each message is a dictionary containing 'tokens' and 'labels'.
    The output file will have one token and its corresponding label per line,
    with a blank line separating sentences/messages.

    Args:
        labeled_messages (List[Dict]): A list of dictionaries, each with 'tokens' and 'labels'.
        output_path (str): The file path to save the CoNLL output.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for msg in labeled_messages:
                tokens = msg.get('tokens', [])
                labels = msg.get('labels', [])
                if len(tokens) != len(labels):
                    print(f"[WARNING] Skipping message due to mismatched tokens and labels: {msg.get('id', 'N/A')}")
                    continue
                
                for token, label in zip(tokens, labels):
                    f.write(f"{token} {label}\n")
                f.write("\n")  # Blank line separates messages
        print(f"Successfully saved CoNLL data to {output_path}")
    except IOError as e:
        print(f"[ERROR] Could not write to file {output_path}: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during CoNLL conversion: {e}") 