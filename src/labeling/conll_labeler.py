"""CoNLL format labeling utilities for Amharic NER."""

import pandas as pd
import re
from typing import List, Tuple, Dict
import logging
from src.utils.etnltk_helper import etnltk_helper
# Hardcoded entity labels to avoid import conflicts
ENTITY_LABELS = ['O', 'B-Product', 'I-Product', 'B-LOC', 'I-LOC', 'B-PRICE', 'I-PRICE']

logger = logging.getLogger(__name__)

class CoNLLLabeler:
    """Labeler for creating CoNLL format annotations."""
    
    def __init__(self):
        self.labels = ENTITY_LABELS
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        
    def tokenize_for_labeling(self, text: str) -> List[str]:
        """Tokenize text for labeling using etnltk."""
        tokens = etnltk_helper.tokenize(text)
        return [token.strip('.,!?;:()[]{}') for token in tokens if token.strip()]
    
    def auto_label_message(self, text: str) -> List[Tuple[str, str]]:
        """Automatically label a message using pattern matching."""
        tokens = self.tokenize_for_labeling(text)
        labels = ['O'] * len(tokens)
        
        # Price patterns
        price_patterns = [
            (r'ዋጋ', r'\d+', r'ብር'),
            (r'በ', r'\d+', r'ብር'),
            (r'\d+', r'ብር')
        ]
        
        # Location patterns
        location_keywords = ['አዲስ', 'አበባ', 'ቦሌ', 'መርካቶ', 'ፒያሳ', 'ካዛንቺስ', 'ሜክሲኮ']
        
        # Product indicators
        product_keywords = ['ሻንጣ', 'ጫማ', 'ልብስ', 'ስልክ', 'ላፕቶፕ', 'መኪና', 'ቤት']
        
        i = 0
        while i < len(tokens):
            token = tokens[i].lower()
            
            # Check for price patterns
            if any(keyword in token for keyword in ['ዋጋ', 'ብር']) or token.isdigit():
                if self._is_price_context(tokens, i):
                    labels[i] = 'B-PRICE'
                    # Look for continuation
                    j = i + 1
                    while j < len(tokens) and self._continues_price(tokens[j]):
                        labels[j] = 'I-PRICE'
                        j += 1
                    i = j
                    continue
            
            # Check for locations
            if any(keyword in token for keyword in location_keywords):
                labels[i] = 'B-LOC'
                # Look for continuation
                j = i + 1
                while j < len(tokens) and self._continues_location(tokens[j]):
                    labels[j] = 'I-LOC'
                    j += 1
                i = j
                continue
            
            # Check for products
            if any(keyword in token for keyword in product_keywords):
                labels[i] = 'B-Product'
                # Look for continuation
                j = i + 1
                while j < len(tokens) and self._continues_product(tokens[j]):
                    labels[j] = 'I-Product'
                    j += 1
                i = j
                continue
            
            i += 1
        
        return list(zip(tokens, labels))
    
    def _is_price_context(self, tokens: List[str], index: int) -> bool:
        """Check if token is in price context."""
        token = tokens[index].lower()
        
        # Direct price indicators
        if 'ዋጋ' in token or 'ብር' in token:
            return True
        
        # Number with price context
        if token.isdigit():
            # Check surrounding tokens
            context = []
            for i in range(max(0, index-2), min(len(tokens), index+3)):
                context.append(tokens[i].lower())
            
            return any(price_word in ' '.join(context) for price_word in ['ዋጋ', 'ብር', 'በ'])
        
        return False
    
    def _continues_price(self, token: str) -> bool:
        """Check if token continues a price entity."""
        return token.isdigit() or 'ብር' in token.lower()
    
    def _continues_location(self, token: str) -> bool:
        """Check if token continues a location entity."""
        location_parts = ['አበባ', 'ከተማ', 'ወረዳ']
        return any(part in token.lower() for part in location_parts)
    
    def _continues_product(self, token: str) -> bool:
        """Check if token continues a product entity."""
        # Simple heuristic: if it's not punctuation and follows product pattern
        return len(token) > 1 and not token.isdigit()
    
    def create_conll_format(self, labeled_data: List[Tuple[str, str]]) -> str:
        """Convert labeled data to CoNLL format."""
        conll_lines = []
        
        for token, label in labeled_data:
            conll_lines.append(f"{token}\t{label}")
        
        conll_lines.append("")  # Empty line to separate messages
        
        return "\n".join(conll_lines)
    
    def label_dataset_sample(self, df: pd.DataFrame, num_messages: int = 50) -> str:
        """Label a sample of the dataset in CoNLL format."""
        try:
            # Sample messages
            sample_df = df.sample(n=min(num_messages, len(df)))
            
            all_conll_data = []
            
            for _, row in sample_df.iterrows():
                text = row['cleaned_text'] if 'cleaned_text' in row else row['text']
                labeled_tokens = self.auto_label_message(text)
                conll_format = self.create_conll_format(labeled_tokens)
                all_conll_data.append(conll_format)
            
            logger.info(f"Labeled {len(sample_df)} messages in CoNLL format")
            
            return "\n".join(all_conll_data)
            
        except Exception as e:
            logger.error(f"Error in labeling: {e}")
            return ""
    
    def parse_conll_data(self, conll_text: str) -> List[Dict]:
        """Parse CoNLL format data."""
        examples = []
        current_tokens = []
        current_labels = []
        
        lines = conll_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            if not line:  # Empty line indicates end of sentence
                if current_tokens:
                    examples.append({
                        'tokens': current_tokens,
                        'labels': current_labels
                    })
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, label = parts
                    current_tokens.append(token)
                    current_labels.append(label)
        
        # Add last example if exists
        if current_tokens:
            examples.append({
                'tokens': current_tokens,
                'labels': current_labels
            })
        
        return examples
    
    def save_conll_data(self, conll_data: str, filepath: str) -> None:
        """Save CoNLL formatted data to file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(conll_data)
            logger.info(f"CoNLL data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving CoNLL data: {e}")