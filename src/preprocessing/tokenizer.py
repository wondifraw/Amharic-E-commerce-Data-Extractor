"""Amharic tokenization utilities."""

import re
from typing import List, Set
from amseg.amharicSegmenter import AmharicSegmenter


class AmharicTokenizer:
    """Professional Amharic tokenizer for e-commerce text."""
    
    def __init__(self):
        self.punctuation = {
            '።', '፤', '፡', '!', '?', '፥', '፦', '፧', '(', ')', ',', '.', '-',
            ':', ';', '"', "'", '[', ']', '{', '}', '/', '\\', '|'
        }
        self.segmenter = AmharicSegmenter([], [])
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Amharic text into words."""
        if not text:
            return []
            
        try:
            tokens = self.segmenter.amharic_tokenizer(text)
            return self._filter_tokens(tokens)
        except Exception:
            # Fallback to simple whitespace tokenization
            return text.split()
            
    def _filter_tokens(self, tokens: List[str]) -> List[str]:
        """Filter out unwanted tokens."""
        filtered = []
        for token in tokens:
            if self._is_valid_token(token):
                filtered.append(token)
        return filtered
        
    def _is_valid_token(self, token: str) -> bool:
        """Check if token is valid for processing."""
        if not token or token.strip() == '':
            return False
            
        # Skip punctuation
        if token in self.punctuation:
            return False
            
        # Skip pure English words (keep mixed)
        if re.match(r'^[a-zA-Z]+$', token):
            return False
            
        # Skip pure numbers
        if re.match(r'^\d+$', token):
            return False
            
        return True