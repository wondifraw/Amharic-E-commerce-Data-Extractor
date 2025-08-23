"""Text preprocessing utilities for Amharic text."""

import re
import pandas as pd
from typing import List, Dict
import logging
from src.utils.etnltk_helper import etnltk_helper

logger = logging.getLogger(__name__)

class AmharicTextProcessor:
    """Processor for Amharic text preprocessing."""
    
    def __init__(self):
        # Amharic character ranges
        self.amharic_pattern = re.compile(r'[\u1200-\u137F]+')
        self.price_patterns = [
            r'(\d+)\s*ብር',
            r'ዋጋ\s*(\d+)',
            r'በ\s*(\d+)\s*ብር',
            r'(\d+)\s*birr'
        ]
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize Amharic text."""
        if not isinstance(text, str):
            return ""
        
        # Use etnltk normalization
        text = etnltk_helper.normalize(text)
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def tokenize_amharic(self, text: str) -> List[str]:
        """Tokenize Amharic text using etnltk."""
        return etnltk_helper.tokenize(text)
    
    def extract_entities_hints(self, text: str) -> Dict[str, List[str]]:
        """Extract potential entities using pattern matching."""
        entities = {
            'prices': [],
            'locations': [],
            'products': []
        }
        
        # Extract prices
        for pattern in self.price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['prices'].extend(matches)
        
        # Common location indicators
        location_keywords = ['አዲስ', 'ቦሌ', 'መርካቶ', 'ፒያሳ', 'ካዛንቺስ']
        for keyword in location_keywords:
            if keyword in text:
                # Extract surrounding context
                pattern = rf'\b\w*{keyword}\w*\b'
                matches = re.findall(pattern, text)
                entities['locations'].extend(matches)
        
        return entities
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess entire dataset."""
        try:
            # Clean text
            df['cleaned_text'] = df['text'].apply(self.clean_text)
            
            # Filter out empty texts
            df = df[df['cleaned_text'].str.len() > 0]
            
            # Add text length
            df['text_length'] = df['cleaned_text'].str.len()
            
            # Extract entity hints
            df['entity_hints'] = df['cleaned_text'].apply(self.extract_entities_hints)
            
            logger.info(f"Preprocessed {len(df)} messages")
            return df
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return df