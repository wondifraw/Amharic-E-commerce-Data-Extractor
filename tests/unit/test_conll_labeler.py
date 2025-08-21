"""
Unit tests for CoNLL labeler
"""

import pytest
from src.preprocessing.conll_labeler import CoNLLLabeler


class TestCoNLLLabeler:
    """Test cases for CoNLLLabeler"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.labeler = CoNLLLabeler()
    
    def test_tokenize_for_conll(self):
        """Test CoNLL tokenization"""
        text = "ሰላም! ዋጋ 150 ብር ነው።"
        tokens = self.labeler.tokenize_for_conll(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "ሰላም" in tokens
        assert "150" in tokens
    
    def test_auto_label_entities(self):
        """Test automatic entity labeling"""
        tokens = ["ሰላም", "ዋጋ", "150", "ብር", "ቦሌ", "አካባቢ"]
        labels = self.labeler.auto_label_entities(tokens)
        
        assert len(labels) == len(tokens)
        assert all(label in ['O', 'B-PRICE', 'I-PRICE', 'B-LOC', 'I-LOC', 'B-Product', 'I-Product'] for label in labels)
        
        # Should detect price
        price_labels = [i for i, label in enumerate(labels) if 'PRICE' in label]
        assert len(price_labels) > 0
    
    def test_create_sample_labeled_data(self):
        """Test sample data creation"""
        labeled_data = self.labeler.create_sample_labeled_data()
        
        assert isinstance(labeled_data, list)
        assert len(labeled_data) > 0
        
        # Check structure
        for message, tokens, labels in labeled_data:
            assert isinstance(message, str)
            assert isinstance(tokens, list)
            assert isinstance(labels, list)
            assert len(tokens) == len(labels)
    
    def test_validate_labels(self):
        """Test BIO label validation"""
        # Valid sequence
        tokens1 = ["ዋጋ", "150", "ብር"]
        labels1 = ["B-PRICE", "I-PRICE", "I-PRICE"]
        assert self.labeler.validate_labels(tokens1, labels1) == True
        
        # Invalid sequence (I- without B-)
        tokens2 = ["ዋጋ", "150", "ብር"]
        labels2 = ["I-PRICE", "I-PRICE", "I-PRICE"]
        assert self.labeler.validate_labels(tokens2, labels2) == False
        
        # Mismatched lengths
        tokens3 = ["ዋጋ", "150"]
        labels3 = ["B-PRICE"]
        assert self.labeler.validate_labels(tokens3, labels3) == False
    
    def test_save_and_load_conll_format(self, tmp_path):
        """Test saving and loading CoNLL format"""
        # Create sample data
        labeled_data = [
            ("ሰላም ዋጋ 150 ብር", ["ሰላም", "ዋጋ", "150", "ብር"], ["O", "B-PRICE", "I-PRICE", "I-PRICE"])
        ]
        
        # Save
        output_path = tmp_path / "test.conll"
        self.labeler.save_conll_format(labeled_data, str(output_path))
        
        # Load
        loaded_data = self.labeler.load_conll_format(str(output_path))
        
        assert len(loaded_data) == 1
        tokens, labels = loaded_data[0]
        assert tokens == ["ሰላም", "ዋጋ", "150", "ብር"]
        assert labels == ["O", "B-PRICE", "I-PRICE", "I-PRICE"]