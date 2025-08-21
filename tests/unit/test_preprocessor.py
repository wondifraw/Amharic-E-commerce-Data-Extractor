"""
Unit tests for text preprocessor
"""

import pytest
import pandas as pd
from src.preprocessing.text_preprocessor import AmharicTextPreprocessor


class TestAmharicTextPreprocessor:
    """Test cases for AmharicTextPreprocessor"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = AmharicTextPreprocessor()
    
    def test_normalize_unicode(self):
        """Test Unicode normalization"""
        text = "ሰላም"
        normalized = self.preprocessor.normalize_unicode(text)
        assert isinstance(normalized, str)
        assert len(normalized) > 0
    
    def test_clean_text(self):
        """Test text cleaning"""
        dirty_text = "  ሰላም!!!   http://example.com  @user #hashtag  "
        cleaned = self.preprocessor.clean_text(dirty_text)
        
        assert "http://example.com" not in cleaned
        assert "@user" not in cleaned
        assert cleaned.strip() == cleaned  # No leading/trailing whitespace
        assert "!!!" not in cleaned or cleaned.count("!") == 1
    
    def test_tokenize_amharic(self):
        """Test Amharic tokenization"""
        text = "ሰላም ዓለም! እንዴት ነህ?"
        tokens = self.preprocessor.tokenize_amharic(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "ሰላም" in tokens
        assert "ዓለም" in tokens
    
    def test_extract_entities_hints(self):
        """Test entity hint extraction"""
        text = "የሕፃናት ጠርሙስ ዋጋ 150 ብር ቦሌ አካባቢ"
        entities = self.preprocessor.extract_entities_hints(text)
        
        assert 'prices' in entities
        assert 'locations' in entities
        assert 'products' in entities
        assert len(entities['prices']) > 0  # Should find price
    
    def test_preprocess_dataframe(self):
        """Test dataframe preprocessing"""
        sample_data = {
            'text': [
                'ሰላም! የሕፃናት ጠርሙስ ዋጋ 150 ብር ነው።',
                'Hello world',
                'አዲስ አበባ ውስጥ ልብስ በ 200 ብር'
            ]
        }
        df = pd.DataFrame(sample_data)
        
        processed_df = self.preprocessor.preprocess_dataframe(df)
        
        assert 'cleaned_text' in processed_df.columns
        assert 'tokens' in processed_df.columns
        assert 'token_count' in processed_df.columns
        assert 'has_amharic' in processed_df.columns
        assert len(processed_df) <= len(df)  # May filter short messages
    
    def test_contains_amharic(self):
        """Test Amharic detection"""
        amharic_text = "ሰላም ዓለም"
        english_text = "Hello world"
        mixed_text = "Hello ሰላም"
        
        assert self.preprocessor._contains_amharic(amharic_text) == True
        assert self.preprocessor._contains_amharic(english_text) == False
        assert self.preprocessor._contains_amharic(mixed_text) == True
    
    def test_prepare_for_labeling(self):
        """Test labeling preparation"""
        sample_data = {
            'text': ['ሰላም! ዋጋ 150 ብር ቦሌ'] * 10 + ['short'] * 5,
            'has_amharic': [True] * 10 + [False] * 5,
            'token_count': [8] * 10 + [1] * 5,
            'price_hints': [['150 ብር']] * 10 + [[]] * 5,
            'location_hints': [['ቦሌ']] * 10 + [[]] * 5
        }
        df = pd.DataFrame(sample_data)
        
        sample_df = self.preprocessor.prepare_for_labeling(df, sample_size=5)
        
        assert len(sample_df) <= 5
        assert all(sample_df['has_amharic'])  # Should only include Amharic text