"""
CoNLL Format Labeling System for Ethiopian NER
"""

import pandas as pd
import re
from typing import List, Tuple, Dict
from loguru import logger
import os


class CoNLLLabeler:
    """Creates CoNLL format labels for NER training"""
    
    def __init__(self):
        self.entity_patterns = {
            'PRICE': [
                (r'ዋጋ\s*(\d+)\s*ብር', ['B-PRICE', 'I-PRICE', 'I-PRICE']),
                (r'በ\s*(\d+)\s*ብር', ['B-PRICE', 'I-PRICE', 'I-PRICE']),
                (r'(\d+)\s*ብር', ['B-PRICE', 'I-PRICE']),
                (r'ETB\s*(\d+)', ['B-PRICE', 'I-PRICE']),
                (r'(\d+)\s*birr', ['B-PRICE', 'I-PRICE'])
            ],
            'LOC': [
                (r'አዲስ\s*አበባ', ['B-LOC', 'I-LOC']),
                (r'ቦሌ', ['B-LOC']),
                (r'መርካቶ', ['B-LOC']),
                (r'ፒያሳ', ['B-LOC']),
                (r'ካዛንቺስ', ['B-LOC']),
                (r'ሃያ\s*ሁለት', ['B-LOC', 'I-LOC']),
                (r'ሰሚት', ['B-LOC']),
                (r'ጀሞ', ['B-LOC'])
            ],
            'Product': [
                (r'የሕፃናት\s*ጠርሙስ', ['B-Product', 'I-Product']),
                (r'ልብስ', ['B-Product']),
                (r'ጫማ', ['B-Product']),
                (r'ስልክ', ['B-Product']),
                (r'መጽሐፍ', ['B-Product']),
                (r'baby\s*bottle', ['B-Product', 'I-Product']),
                (r'phone', ['B-Product']),
                (r'shoes', ['B-Product']),
                (r'clothes', ['B-Product'])
            ]
        }
    
    def tokenize_for_conll(self, text: str) -> List[str]:
        """Tokenize text for CoNLL format"""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\S+', text)
        
        # Further refine tokenization
        refined_tokens = []
        for token in tokens:
            # Handle punctuation
            if re.match(r'^[^\w\u1200-\u137F]+$', token):
                refined_tokens.append(token)
            else:
                # Split punctuation from words
                parts = re.split(r'([^\w\u1200-\u137F]+)', token)
                refined_tokens.extend([part for part in parts if part.strip()])
        
        return refined_tokens
    
    def auto_label_entities(self, tokens: List[str]) -> List[str]:
        """Automatically label entities using patterns"""
        labels = ['O'] * len(tokens)
        text = ' '.join(tokens)
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern, pattern_labels in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                
                for match in matches:
                    # Find token positions for the match
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Map character positions to token positions
                    token_positions = self._map_char_to_token_positions(text, tokens, start_pos, end_pos)
                    
                    # Apply labels
                    for i, token_idx in enumerate(token_positions):
                        if i < len(pattern_labels) and token_idx < len(labels):
                            labels[token_idx] = pattern_labels[i]
        
        return labels
    
    def _map_char_to_token_positions(self, text: str, tokens: List[str], start: int, end: int) -> List[int]:
        """Map character positions to token indices"""
        token_positions = []
        current_pos = 0
        
        for i, token in enumerate(tokens):
            token_start = text.find(token, current_pos)
            token_end = token_start + len(token)
            
            # Check if token overlaps with the match
            if not (token_end <= start or token_start >= end):
                token_positions.append(i)
            
            current_pos = token_end
        
        return token_positions
    
    def create_sample_labeled_data(self) -> List[Tuple[str, List[str], List[str]]]:
        """Create sample labeled data for demonstration"""
        sample_messages = [
            "ሰላም! የሕፃናት ጠርሙስ ዋጋ 150 ብር ነው። ቦሌ አካባቢ ነው።",
            "አዲስ አበባ ውስጥ የሚሸጥ ልብስ በ 200 ብር",
            "መርካቶ ውስጥ ጫማ 300 ብር",
            "ፒያሳ አካባቢ ስልክ ETB 5000",
            "ሃያ ሁለት ላይ መጽሐፍ 50 ብር ነው",
            "Baby bottle for sale 150 birr in Bole",
            "Phone available at Merkato 4000 birr",
            "Clothes በ 180 ብር አዲስ አበባ",
            "ካዛንቺስ አካባቢ shoes 250 ብር",
            "ጀሞ ላይ የሚሸጥ ልብስ ዋጋ 120 ብር"
        ]
        
        labeled_data = []
        
        for message in sample_messages:
            tokens = self.tokenize_for_conll(message)
            labels = self.auto_label_entities(tokens)
            labeled_data.append((message, tokens, labels))
        
        return labeled_data
    
    def save_conll_format(self, labeled_data: List[Tuple[str, List[str], List[str]]], 
                         output_path: str = "data/labeled/sample_conll.txt"):
        """Save data in CoNLL format"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for message, tokens, labels in labeled_data:
                # Write comment with original message
                f.write(f"# {message}\n")
                
                # Write tokens and labels
                for token, label in zip(tokens, labels):
                    f.write(f"{token}\t{label}\n")
                
                # Empty line to separate sentences
                f.write("\n")
        
        logger.info(f"CoNLL format data saved to {output_path}")
        return output_path
    
    def load_conll_format(self, file_path: str) -> List[Tuple[List[str], List[str]]]:
        """Load CoNLL format data"""
        sentences = []
        current_tokens = []
        current_labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if line.startswith('#'):
                    continue
                
                if not line:
                    # End of sentence
                    if current_tokens:
                        sentences.append((current_tokens.copy(), current_labels.copy()))
                        current_tokens.clear()
                        current_labels.clear()
                else:
                    # Parse token and label
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        current_tokens.append(parts[0])
                        current_labels.append(parts[1])
        
        # Add last sentence if exists
        if current_tokens:
            sentences.append((current_tokens, current_labels))
        
        logger.info(f"Loaded {len(sentences)} sentences from {file_path}")
        return sentences
    
    def create_extended_dataset(self, base_messages: List[str], target_size: int = 50) -> List[Tuple[str, List[str], List[str]]]:
        """Create extended dataset with variations"""
        labeled_data = []
        
        # Start with sample data
        sample_data = self.create_sample_labeled_data()
        labeled_data.extend(sample_data)
        
        # Add base messages if provided
        for message in base_messages[:target_size - len(sample_data)]:
            tokens = self.tokenize_for_conll(message)
            labels = self.auto_label_entities(tokens)
            labeled_data.append((message, tokens, labels))
        
        return labeled_data[:target_size]
    
    def validate_labels(self, tokens: List[str], labels: List[str]) -> bool:
        """Validate BIO tagging consistency"""
        if len(tokens) != len(labels):
            return False
        
        for i, label in enumerate(labels):
            if label.startswith('I-'):
                # I- tag must follow B- or I- of same type
                if i == 0:
                    return False
                
                prev_label = labels[i-1]
                entity_type = label[2:]
                
                if not (prev_label == f'B-{entity_type}' or prev_label == f'I-{entity_type}'):
                    return False
        
        return True


def main():
    """Create sample CoNLL dataset"""
    labeler = CoNLLLabeler()
    
    # Create sample labeled data
    labeled_data = labeler.create_sample_labeled_data()
    
    # Add more synthetic examples
    additional_messages = [
        "ሰሚት አካባቢ የሚሸጥ ስልክ 3000 ብር",
        "መጽሐፍ በ 45 ብር ቦሌ ላይ",
        "አዲስ አበባ ውስጥ ጫማ ETB 280",
        "ልብስ ዋጋ 160 ብር መርካቶ",
        "ፒያሳ ላይ baby bottle 140 birr"
    ]
    
    extended_data = labeler.create_extended_dataset(additional_messages, target_size=50)
    
    # Save in CoNLL format
    output_path = labeler.save_conll_format(extended_data)
    
    # Validate the created dataset
    loaded_data = labeler.load_conll_format(output_path)
    
    print(f"Created dataset with {len(loaded_data)} sentences")
    
    # Show sample
    for i, (tokens, labels) in enumerate(loaded_data[:3]):
        print(f"\nSample {i+1}:")
        for token, label in zip(tokens, labels):
            print(f"{token}\t{label}")
        print()


if __name__ == "__main__":
    main()