"""
amharic_text.py
Module for preprocessing Amharic text data.
"""
import re
import unicodedata
from typing import List, Dict

def normalize_and_clean(text: str) -> str:
    """
    Normalizes and cleans Amharic text by handling Unicode, removing unwanted
    characters, and standardizing whitespace.

    Args:
        text (str): The input Amharic text.

    Returns:
        str: The cleaned and normalized text.
    """
    if not isinstance(text, str):
        return ""
    try:
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'(\+?251|0)?[97]\d{8}\b', '', text)
        text = re.sub(r'[^\w\s\u1200-\u137F፡።፣፤፥፦፧፨]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"[ERROR] Text cleaning failed for text '{text[:30]}...': {e}")
        return ""

def tokenize(text: str) -> List[str]:
    """
    Tokenizes Amharic text into words.

    Args:
        text (str): The input Amharic text.

    Returns:
        List[str]: A list of tokens.
    """
    if not text:
        return []
    try:
        tokens = re.findall(r'\b\w+[\፡\።\፣\፤\፥\፦\፧\፨]*\b|\b\w+\b', text, re.UNICODE)
        return tokens
    except Exception as e:
        print(f"[ERROR] Tokenization failed for text '{text[:30]}...': {e}")
        return []

def preprocess_messages(messages: List[Dict]) -> List[Dict]:
    """
    Applies cleaning and tokenization to a list of message dictionaries.

    Args:
        messages (List[Dict]): A list of raw message dictionaries.

    Returns:
        List[Dict]: A list of processed message dictionaries with 'cleaned_text' and 'tokens'.
    """
    processed = []
    for msg in messages:
        cleaned_text = normalize_and_clean(msg.get('text', ''))
        tokens = tokenize(cleaned_text)
        
        processed_msg = msg.copy()
        processed_msg['cleaned_text'] = cleaned_text
        processed_msg['tokens'] = tokens
        processed.append(processed_msg)
    return processed 