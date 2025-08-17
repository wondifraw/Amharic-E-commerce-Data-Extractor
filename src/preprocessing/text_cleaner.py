"""Text cleaning utilities for Amharic text processing."""

import re
from typing import List, Optional
import demoji
from amseg.amharicNormalizer import AmharicNormalizer


class TextCleaner:
    """Professional text cleaning for Amharic e-commerce data."""
    
    def __init__(self):
        self.normalizer = AmharicNormalizer()
        self.unwanted_patterns = [
            r'@\w+',  # Mentions
            r'http[s]?://\S+',  # URLs
            r'www\.\S+',  # WWW URLs
            r'\d{10,}',  # Long numbers (likely phone numbers)
        ]
        self.unwanted_words = {
            '@classy', 'ብርands', 'ብርandseller', '@sami_twa', '@kingsmarque'
        }
        
    def remove_emoji(self, text: str) -> str:
        """Remove emojis from text."""
        try:
            return demoji.replace(text, repl="")
        except Exception:
            return text
            
    def remove_unwanted_patterns(self, text: str) -> str:
        """Remove unwanted patterns like URLs, mentions, etc."""
        for pattern in self.unwanted_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text.strip()
        
    def normalize_text(self, text: str) -> str:
        """Normalize Amharic text."""
        try:
            return self.normalizer.normalize(text)
        except Exception:
            return text
            
    def clean_text(self, text: str) -> str:
        """Complete text cleaning pipeline."""
        if not text or not isinstance(text, str):
            return ""
            
        # Remove emojis
        text = self.remove_emoji(text)
        
        # Remove unwanted patterns
        text = self.remove_unwanted_patterns(text)
        
        # Normalize Amharic text
        text = self.normalize_text(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def filter_unwanted_words(self, words: List[str]) -> List[str]:
        """Filter out unwanted words from token list."""
        return [word for word in words if word.lower() not in self.unwanted_words]