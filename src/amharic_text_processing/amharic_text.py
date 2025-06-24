import pandas as pd
import re

from etnltk.lang.am.normalizer import (
    normalize_labialized,
    normalize_shortened,
    normalize_punct,
    normalize_char,
)
from etnltk.tokenize.am import word_tokenize

# Optional: emoji cleaner
def remove_emoji(text):
    emoji_pattern = re.compile("[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\u2600-\u26FF"          # miscellaneous symbols
        u"\u2700-\u27BF"          # dingbats
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# --------------------------
# Amharic Preprocessing Function
# --------------------------
def preprocess_amharic_message(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Step 1: Remove emojis
    text = remove_emoji(text)

    # Step 2: Normalize
    text = normalize_labialized(text)
    text = normalize_shortened(text)
    text = normalize_punct(text)
    text = normalize_char(text)

    # Step 3: Tokenize
    tokens = word_tokenize(text)

    return ' '.join(tokens).strip()