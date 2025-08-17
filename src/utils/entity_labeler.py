"""Professional entity labeling for CONLL format."""

from typing import List, Tuple, Dict
import re


class EntityLabeler:
    """Professional entity labeling system for Amharic NER."""
    
    def __init__(self):
        self.price_indicators = {'ዋጋ', 'ብር', 'ETB', '$', 'ዶላር'}
        self.location_indicators = {'አድራሻ', 'ቦታ', 'አካባቢ'}
        self.product_indicators = {'ምርት', 'እቃ', 'ሸቀጥ'}
        
    def label_conll_format(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Generate CONLL format labels with improved logic."""
        labeled_data = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if self._is_price_context(token, tokens, i):
                labels = self._label_price_sequence(tokens, i)
                labeled_data.extend(labels)
                i += len(labels)
            elif self._is_location_context(token, tokens, i):
                labels = self._label_location_sequence(tokens, i)
                labeled_data.extend(labels)
                i += len(labels)
            elif self._is_product_context(token, tokens, i):
                labels = self._label_product_sequence(tokens, i)
                labeled_data.extend(labels)
                i += len(labels)
            else:
                labeled_data.append((token, "O"))
                i += 1
                
        return labeled_data
        
    def _is_price_context(self, token: str, tokens: List[str], index: int) -> bool:
        """Check if token is in price context."""
        if token in self.price_indicators:
            return True
        if re.match(r'\d+', token) and index + 1 < len(tokens):
            return tokens[index + 1] in self.price_indicators
        return False
        
    def _is_location_context(self, token: str, tokens: List[str], index: int) -> bool:
        """Check if token is in location context."""
        return token in self.location_indicators
        
    def _is_product_context(self, token: str, tokens: List[str], index: int) -> bool:
        """Check if token is in product context."""
        return token in self.product_indicators
        
    def _label_price_sequence(self, tokens: List[str], start_idx: int) -> List[Tuple[str, str]]:
        """Label price-related token sequence."""
        labels = []
        current_token = tokens[start_idx]
        
        if current_token == "ዋጋ":
            labels.append((current_token, "B-PRICE"))
            # Look for following price and currency
            for i in range(start_idx + 1, min(start_idx + 3, len(tokens))):
                if i < len(tokens):
                    labels.append((tokens[i], "I-PRICE"))
        elif re.match(r'\d+', current_token):
            labels.append((current_token, "B-PRICE"))
            # Check for currency indicator
            if start_idx + 1 < len(tokens) and tokens[start_idx + 1] in self.price_indicators:
                labels.append((tokens[start_idx + 1], "I-PRICE"))
        else:
            labels.append((current_token, "B-PRICE"))
            
        return labels
        
    def _label_location_sequence(self, tokens: List[str], start_idx: int) -> List[Tuple[str, str]]:
        """Label location-related token sequence."""
        labels = []
        labels.append((tokens[start_idx], "O"))  # Location indicator is not part of location
        
        # Label following tokens as location
        for i in range(start_idx + 1, min(start_idx + 5, len(tokens))):
            if i < len(tokens):
                if i == start_idx + 1:
                    labels.append((tokens[i], "B-LOC"))
                else:
                    labels.append((tokens[i], "I-LOC"))
                    
        return labels
        
    def _label_product_sequence(self, tokens: List[str], start_idx: int) -> List[Tuple[str, str]]:
        """Label product-related token sequence."""
        labels = []
        labels.append((tokens[start_idx], "O"))  # Product indicator
        
        # Label following tokens as product
        for i in range(start_idx + 1, min(start_idx + 4, len(tokens))):
            if i < len(tokens):
                if i == start_idx + 1:
                    labels.append((tokens[i], "B-PRODUCT"))
                else:
                    labels.append((tokens[i], "I-PRODUCT"))
                    
        return labels