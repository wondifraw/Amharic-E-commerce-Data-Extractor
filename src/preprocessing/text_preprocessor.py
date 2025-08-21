"""
Text Preprocessing for Amharic and Ethiopian Languages
"""

# import re
# import pandas as pd
from typing import List, Dict, Tuple
import unicodedata
from loguru import logger

import os
import pandas as pd
from etnltk import Amharic
from etnltk.lang.am import clean_amharic, normalize
from etnltk.tokenize.am import word_tokenize
import re


class AmharicTextPreprocessor:
    """Preprocessor for Amharic and Ethiopian text data"""
    
    def __init__(self):
        # Amharic Unicode range
        self.amharic_range = (0x1200, 0x137F)
        
        # Common Amharic stopwords (basic set)
        self.amharic_stopwords = {
            'እና', 'ወይም', 'ነው', 'ናት', 'ናቸው', 'አለ', 'አላት', 'አላቸው',
            'በ', 'ከ', 'ለ', 'እስከ', 'ወደ', 'ላይ', 'ስር', 'ውስጥ', 'ጋር'
        }
        
        # Price patterns in Amharic
        self.price_patterns = [
            r'ዋጋ\s*\d+\s*ብር',
            r'በ\s*\d+\s*ብር',
            r'\d+\s*ብር',
            r'ETB\s*\d+',
            r'\d+\s*birr'
        ]
    
    def preprocess_amharic_text(self,text):
        if not isinstance(text, str):
            return '', ''
        # Clean and normalize text using etnltk
        cleaned = clean_amharic(text, keep_numbers=True)
        normalized = normalize(cleaned)
        # Tokenize using etnltk's Amharic word_tokenize
        tokens = word_tokenize(normalized)
        return normalized, ' '.join(tokens)
    
    def custom_clean_amharic(self,text):
        """
        Custom cleaner for Amharic text:
        - Keeps Amharic characters, numbers, and basic punctuation (፡።፣፤፥፦፧፨, . , ! ?)
        - Removes Latin and other scripts
        - Normalizes whitespace
        """
        if not isinstance(text, str):
            return ''
        # Keep Amharic, numbers, and basic punctuation
        cleaned = re.findall(r'[\u1200-\u137F0-9]+', text)
        cleaned_text = ' '.join(cleaned)
        # Normalize whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def custom_preprocess_amharic_text(self,text):
        """
        Returns cleaned and tokenized Amharic text (space-separated tokens)
        """
        cleaned = self.custom_clean_amharic(text)
        tokens = cleaned.split()
        return cleaned, ' '.join(tokens)

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        return unicodedata.normalize('NFC', text)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Normalize Unicode
        text = self.normalize_unicode(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags (keep the text part)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def tokenize_amharic(self, text: str) -> List[str]:
        """Tokenize Amharic text"""
        # Basic tokenization by whitespace and punctuation
        tokens = re.findall(r'\S+', text)
        
        # Further split on punctuation while preserving Amharic characters
        refined_tokens = []
        for token in tokens:
            # Split on punctuation but keep Amharic characters together
            parts = re.split(r'([^\w\u1200-\u137F]+)', token)
            refined_tokens.extend([part for part in parts if part.strip()])
        
        return refined_tokens
    
    def extract_entities_hints(self, text: str) -> Dict[str, List[str]]:
        """Extract potential entities using pattern matching"""
        entities = {
            'prices': [],
            'locations': [],
            'products': []
        }
        
        # Extract prices
        for pattern in self.price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['prices'].extend(matches)
        
        # Extract potential locations (Ethiopian cities/areas)
        location_patterns = [
            r'አዲስ\s*አበባ', r'አዲስ\s*አበባ', r'ቦሌ', r'መርካቶ', r'ፒያሳ',
            r'ካዛንቺስ', r'ጀሞ', r'ሰሚት', r'ሃያ\s*ሁለት', r'ሳሪስ'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['locations'].extend(matches)
        
        return entities
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Preprocess entire dataframe"""
        logger.info(f"Preprocessing {len(df)} messages")
        
        # Create processed dataframe
        processed_df = df.copy()
        
        # Clean text
        processed_df['cleaned_text'] = processed_df[text_column].apply(self.clean_text)
        
        # Tokenize
        processed_df['tokens'] = processed_df['cleaned_text'].apply(self.tokenize_amharic)
        processed_df['token_count'] = processed_df['tokens'].apply(len)
        
        # Extract entity hints
        entity_hints = processed_df['cleaned_text'].apply(self.extract_entities_hints)
        processed_df['price_hints'] = entity_hints.apply(lambda x: x['prices'])
        processed_df['location_hints'] = entity_hints.apply(lambda x: x['locations'])
        
        # Add metadata
        processed_df['has_amharic'] = processed_df['cleaned_text'].apply(self._contains_amharic)
        processed_df['text_length'] = processed_df['cleaned_text'].apply(len)
        
        # Filter out very short messages
        processed_df = processed_df[processed_df['token_count'] >= 3]
        
        logger.info(f"Preprocessing complete. {len(processed_df)} messages retained")
        return processed_df
    
    def _contains_amharic(self, text: str) -> bool:
        """Check if text contains Amharic characters"""
        for char in text:
            if self.amharic_range[0] <= ord(char) <= self.amharic_range[1]:
                return True
        return False
    
    def prepare_for_labeling(self, df: pd.DataFrame, sample_size: int = 50) -> pd.DataFrame:
        """Prepare a sample of data for manual labeling"""
        # Filter for messages with potential entities
        candidates = df[
            (df['has_amharic']) & 
            (df['token_count'].between(5, 50)) &
            ((df['price_hints'].apply(len) > 0) | (df['location_hints'].apply(len) > 0))
        ]
        
        # Sample for labeling
        if len(candidates) > sample_size:
            sample_df = candidates.sample(n=sample_size, random_state=42)
        else:
            sample_df = candidates
        
        logger.info(f"Selected {len(sample_df)} messages for labeling")
        return sample_df[['id', 'channel', 'cleaned_text', 'tokens', 'price_hints', 'location_hints']]


def main():
    """Test the preprocessor"""
    # Sample data for testing
    sample_data = {
        'text': [
            'ሰላም! የሕፃናት ጠርሙስ ዋጋ 150 ብር ነው። ቦሌ አካባቢ ነው።',
            'አዲስ አበባ ውስጥ የሚሸጥ ልብስ በ 200 ብር',
            'Hello! Baby bottle price is 150 birr. Located in Bole area.'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    preprocessor = AmharicTextPreprocessor()
    
    processed_df = preprocessor.preprocess_dataframe(df)
    print(processed_df[['cleaned_text', 'tokens', 'price_hints', 'location_hints']])


if __name__ == "__main__":
    main()