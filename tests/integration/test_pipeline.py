"""
Integration tests for the complete pipeline
"""

import pytest
import pandas as pd
import os
import tempfile
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from preprocessing.text_preprocessor import AmharicTextPreprocessor
from preprocessing.conll_labeler import CoNLLLabeler
from ner.model_trainer import NERModelTrainer
from vendor_analytics.scorecard import VendorScorecard


class TestPipelineIntegration:
    """Integration tests for the complete pipeline"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.sample_data = pd.DataFrame({
            'channel': ['@test_channel'] * 20,
            'text': [
                'ሰላም! የሕፃናት ጠርሙስ ዋጋ 150 ብር ነው። ቦሌ አካባቢ ነው።',
                'አዲስ አበባ ውስጥ የሚሸጥ ልብስ በ 200 ብር',
                'መርካቶ ውስጥ ጫማ 300 ብር',
                'ፒያሳ አካባቢ ስልክ ETB 5000',
                'Baby bottle for sale 150 birr in Bole'
            ] * 4,
            'views': [100, 200, 300, 400, 500] * 4,
            'forwards': [5, 10, 15, 20, 25] * 4,
            'replies': [1, 2, 3, 4, 5] * 4,
            'date': pd.date_range('2024-01-01', periods=20, freq='D')
        })
    
    def test_preprocessing_to_labeling_pipeline(self):
        """Test preprocessing to labeling pipeline"""
        # Step 1: Preprocessing
        preprocessor = AmharicTextPreprocessor()
        processed_df = preprocessor.preprocess_dataframe(self.sample_data)
        
        assert 'cleaned_text' in processed_df.columns
        assert 'tokens' in processed_df.columns
        assert len(processed_df) > 0
        
        # Step 2: Prepare for labeling
        sample_df = preprocessor.prepare_for_labeling(processed_df, sample_size=10)
        
        assert len(sample_df) <= 10
        assert 'cleaned_text' in sample_df.columns
        
        # Step 3: CoNLL labeling
        labeler = CoNLLLabeler()
        messages = sample_df['cleaned_text'].tolist()
        labeled_data = labeler.create_extended_dataset(messages, target_size=15)
        
        assert len(labeled_data) == 15
        
        # Validate structure
        for message, tokens, labels in labeled_data:
            assert isinstance(message, str)
            assert isinstance(tokens, list)
            assert isinstance(labels, list)
            assert len(tokens) == len(labels)
    
    def test_labeling_to_training_pipeline(self):
        """Test labeling to training pipeline"""
        # Create labeled data
        labeler = CoNLLLabeler()
        labeled_data = labeler.create_sample_labeled_data()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            temp_path = f.name
            
        try:
            labeler.save_conll_format(labeled_data, temp_path)
            
            # Load with trainer
            trainer = NERModelTrainer()
            sentences, labels = trainer.load_conll_data(temp_path)
            
            assert len(sentences) == len(labeled_data)
            assert len(labels) == len(labeled_data)
            
            # Prepare dataset
            train_dataset, val_dataset = trainer.prepare_dataset(sentences, labels)
            
            assert train_dataset is not None
            assert val_dataset is not None
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_data_to_vendor_analytics_pipeline(self):
        """Test data to vendor analytics pipeline"""
        # Process data
        preprocessor = AmharicTextPreprocessor()
        processed_df = preprocessor.preprocess_dataframe(self.sample_data)
        
        # Vendor analytics
        scorecard = VendorScorecard()
        vendor_analyses = scorecard.analyze_all_vendors(processed_df)
        
        assert len(vendor_analyses) > 0
        assert '@test_channel' in vendor_analyses
        
        analysis = vendor_analyses['@test_channel']
        assert 'lending_score' in analysis
        assert 'risk_category' in analysis
        assert 'posting_frequency' in analysis
        
        # Create comparison table
        comparison_table = scorecard.create_vendor_comparison_table(vendor_analyses)
        
        assert isinstance(comparison_table, pd.DataFrame)
        assert len(comparison_table) > 0
        assert 'Lending Score' in comparison_table.columns
    
    def test_end_to_end_data_flow(self):
        """Test complete data flow from raw data to analytics"""
        # Step 1: Raw data (simulated)
        raw_data = self.sample_data.copy()
        
        # Step 2: Preprocessing
        preprocessor = AmharicTextPreprocessor()
        processed_data = preprocessor.preprocess_dataframe(raw_data)
        
        # Step 3: Sample for labeling
        sample_for_labeling = preprocessor.prepare_for_labeling(processed_data, sample_size=5)
        
        # Step 4: Create CoNLL dataset
        labeler = CoNLLLabeler()
        messages = sample_for_labeling['cleaned_text'].tolist()
        labeled_data = labeler.create_extended_dataset(messages, target_size=10)
        
        # Step 5: Vendor analytics
        scorecard = VendorScorecard()
        vendor_analyses = scorecard.analyze_all_vendors(processed_data)
        
        # Verify end-to-end flow
        assert len(processed_data) > 0
        assert len(sample_for_labeling) > 0
        assert len(labeled_data) == 10
        assert len(vendor_analyses) > 0
        
        # Check data consistency
        original_channels = set(raw_data['channel'].unique())
        processed_channels = set(processed_data['channel'].unique())
        analytics_channels = set(vendor_analyses.keys())
        
        assert original_channels == processed_channels
        assert processed_channels == analytics_channels
    
    @pytest.mark.slow
    def test_model_training_integration(self):
        """Test model training integration (marked as slow)"""
        # Create minimal dataset for training
        labeler = CoNLLLabeler()
        labeled_data = labeler.create_sample_labeled_data()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            temp_path = f.name
        
        try:
            labeler.save_conll_format(labeled_data, temp_path)
            
            # Train model (this would be slow in real scenario)
            trainer = NERModelTrainer()
            sentences, labels = trainer.load_conll_data(temp_path)
            train_dataset, val_dataset = trainer.prepare_dataset(sentences, labels)
            
            # Verify datasets are properly formatted
            assert len(train_dataset) > 0
            assert len(val_dataset) >= 0
            
            # Check dataset structure
            sample_item = train_dataset[0]
            assert 'tokens' in sample_item
            assert 'labels' in sample_item
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_error_handling_in_pipeline(self):
        """Test error handling throughout the pipeline"""
        # Test with empty data
        empty_df = pd.DataFrame()
        
        preprocessor = AmharicTextPreprocessor()
        processed_empty = preprocessor.preprocess_dataframe(empty_df)
        assert len(processed_empty) == 0
        
        # Test vendor analytics with insufficient data
        scorecard = VendorScorecard()
        small_data = self.sample_data.head(5)  # Less than minimum threshold
        
        vendor_analyses = scorecard.analyze_all_vendors(small_data)
        
        # Should handle insufficient data gracefully
        for analysis in vendor_analyses.values():
            if 'error' in analysis:
                assert 'Insufficient data' in analysis['error']
    
    def test_data_validation_across_components(self):
        """Test data validation across different components"""
        # Process data through multiple components
        preprocessor = AmharicTextPreprocessor()
        processed_df = preprocessor.preprocess_dataframe(self.sample_data)
        
        # Validate processed data structure
        required_columns = ['cleaned_text', 'tokens', 'token_count', 'has_amharic']
        assert all(col in processed_df.columns for col in required_columns)
        
        # Test with labeler
        labeler = CoNLLLabeler()
        sample_messages = processed_df['cleaned_text'].head(3).tolist()
        
        for message in sample_messages:
            tokens = labeler.tokenize_for_conll(message)
            labels = labeler.auto_label_entities(tokens)
            
            # Validate BIO consistency
            assert labeler.validate_labels(tokens, labels)
        
        # Test with vendor analytics
        scorecard = VendorScorecard()
        
        # Check price extraction
        for text in processed_df['cleaned_text'].head(5):
            prices = scorecard.extract_prices_from_text(text)
            assert isinstance(prices, list)
            assert all(isinstance(price, float) for price in prices)