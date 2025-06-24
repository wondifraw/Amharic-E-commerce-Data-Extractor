import re
import os
import sys
import argparse
import logging
import pandas as pd
from collections import Counter

# --- Logging setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("conll_formatter")

# -------------------------------
# Entity Patterns 
# -------------------------------
location_keywords = [
    'Addis Ababa', 'ቦሌ', 'መኪናአይነት', 'ብስራቱ', 'ገብረኤል', 'ጉባኤ',
    'ኢህትሪያል', 'ህንፃ', 'ፕላዛ', '4ኛዎ', 'ጎዳና', 'ላቅቶ', 'ሞል', 'ስላሴ',
    'ማኅበራት', 'ሀብታም', 'አዲስ አበባ', 'ኢህትሪያል', 'ሳልታ', 'ሆስፒታል',
    'ሎጎ', 'ማህበረሰብ', 'አልቦርድ', 'ቅድሚያ', 'አህያ ህንፃ', 'ማኅበራት ፕላዛ',
    'ማህበራት ፕላዛ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
    'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ', 'ማህበራት ህንፃ',
]

# -------------------------------
# Function: Label Single Message
# -------------------------------
def label_message_conll(message):
    """
    Label a single message in CoNLL format.
    Returns a list of (token, label) tuples.
    """
    if not isinstance(message, str) or message.strip() == "":
        logger.warning("Empty or invalid message encountered.")
        return []

    labeled_tokens = []

    # Split first line and remaining
    if '\n' in message:
        first_line, rest = message.split('\n', 1)
    else:
        first_line, rest = message, ""

    # Label first line as PRODUCT
    first_tokens = re.findall(r'\S+', first_line)
    if first_tokens:
        labeled_tokens.append((first_tokens[0], "B-PRODUCT"))
        for token in first_tokens[1:]:
            labeled_tokens.append((token, "I-PRODUCT"))

    # Process remaining lines
    for line in rest.split('\n'):
        tokens = re.findall(r'\S+', line)
        for token in tokens:
            if re.match(r'^\d{10}$', token):
                label = "I-PHONE"
            elif re.match(r'^\d+(\.\d{1,2})?$', token) or any(x in token for x in ['ETB', '$', 'birr', 'ብር', 'ዋጋ']):
                label = "I-PRICE"
            elif any(loc in token for loc in location_keywords):
                label = "I-LOC"
            else:
                label = "O"
            labeled_tokens.append((token, label))

    return labeled_tokens

# -------------------------------
# Save to CoNLL format
# -------------------------------
def save_conll(data, path):
    """
    Save labeled data to CoNLL format at the given path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sentence in data:
            for token, label in sentence:
                f.write(f"{token}\t{label}\n")
            f.write("\n")
    logger.info(f"Saved {len(data)} sentences to {path}")

# -------------------------------
# Metrics: Label Distribution
# -------------------------------
def label_distribution(labeled_dataset):
    """
    Print and return the distribution of labels in the dataset.
    """
    counter = Counter()
    for sentence in labeled_dataset:
        for _, label in sentence:
            counter[label] += 1
    logger.info(f"Label distribution: {dict(counter)}")
    return counter

# -------------------------------
# CLI for batch labeling
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch label messages and save in CoNLL format.")
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file with a Message column.')
    parser.add_argument('--output', type=str, required=True, help='Path to output CoNLL file.')
    parser.add_argument('--limit', type=int, default=50, help='Number of messages to label (default: 50).')
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
        messages = df['Message'].dropna().astype(str)
        labeled_dataset = []
        for i, msg in enumerate(messages[:args.limit]):
            labeled_sentence = label_message_conll(msg)
            labeled_dataset.append(labeled_sentence)
        save_conll(labeled_dataset, args.output)
        label_distribution(labeled_dataset)
        logger.info(f"✅ Labeled {len(labeled_dataset)} messages and saved to {args.output}")
    except Exception as e:
        logger.error(f"Failed to label and save data: {e}", exc_info=True)

if __name__ == "__main__":
    main()