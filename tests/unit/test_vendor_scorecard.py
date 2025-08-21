"""
Unit tests for vendor scorecard
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.vendor_analytics.scorecard import VendorScorecard


class TestVendorScorecard:
    """Test cases for VendorScorecard"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.scorecard = VendorScorecard()
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        self.sample_data = pd.DataFrame({
            'channel': ['@test_channel'] * 30,
            'text': [
                'ሰላም! የሕፃናት ጠርሙስ ዋጋ 150 ብር ነው። ቦሌ አካባቢ ነው።',
                'አዲስ አበባ ውስጥ የሚሸጥ ልብስ በ 200 ብር',
                'መርካቶ ውስጥ ጫማ 300 ብር'
            ] * 10,
            'views': np.random.randint(100, 1000, 30),
            'forwards': np.random.randint(0, 50, 30),
            'replies': np.random.randint(0, 20, 30),
            'date': dates
        })
    
    def test_calculate_posting_frequency(self):
        """Test posting frequency calculation"""
        frequency = self.scorecard.calculate_posting_frequency(self.sample_data)
        
        assert frequency > 0
        assert isinstance(frequency, float)
        # Should be approximately 30 posts / 4.3 weeks ≈ 7 posts/week
        assert 5 <= frequency <= 10
    
    def test_calculate_engagement_metrics(self):
        """Test engagement metrics calculation"""
        metrics = self.scorecard.calculate_engagement_metrics(self.sample_data)
        
        required_keys = ['avg_views', 'avg_forwards', 'avg_replies', 'engagement_rate']
        assert all(key in metrics for key in required_keys)
        assert all(isinstance(metrics[key], (int, float)) for key in required_keys)
        assert metrics['avg_views'] > 0
        assert metrics['engagement_rate'] >= 0
    
    def test_extract_prices_from_text(self):
        """Test price extraction"""
        text1 = "ሰላም! ዋጋ 150 ብር ነው"
        text2 = "በ 200 ብር የሚሸጥ ልብስ"
        text3 = "ETB 5000 ስልክ"
        text4 = "No prices here"
        
        prices1 = self.scorecard.extract_prices_from_text(text1)
        prices2 = self.scorecard.extract_prices_from_text(text2)
        prices3 = self.scorecard.extract_prices_from_text(text3)
        prices4 = self.scorecard.extract_prices_from_text(text4)
        
        assert 150.0 in prices1
        assert 200.0 in prices2
        assert 5000.0 in prices3
        assert len(prices4) == 0
    
    def test_calculate_price_metrics(self):
        """Test price metrics calculation"""
        metrics = self.scorecard.calculate_price_metrics(self.sample_data)
        
        required_keys = ['avg_price', 'price_std', 'price_consistency', 'price_range', 'total_products_with_prices']
        assert all(key in metrics for key in required_keys)
        assert metrics['avg_price'] > 0  # Should find prices in sample data
        assert metrics['total_products_with_prices'] > 0
        assert 0 <= metrics['price_consistency'] <= 1
    
    def test_find_top_performing_post(self):
        """Test top post identification"""
        top_post = self.scorecard.find_top_performing_post(self.sample_data)
        
        required_keys = ['views', 'text', 'date']
        assert all(key in top_post for key in required_keys)
        assert top_post['views'] == self.sample_data['views'].max()
        assert isinstance(top_post['text'], str)
    
    def test_calculate_lending_score(self):
        """Test lending score calculation"""
        vendor_metrics = {
            'posting_frequency': 7.0,
            'engagement': {
                'avg_views': 500,
                'engagement_rate': 0.05
            },
            'price_metrics': {
                'price_consistency': 0.8
            }
        }
        
        score = self.scorecard.calculate_lending_score(vendor_metrics)
        
        assert 0 <= score <= 100
        assert isinstance(score, float)
    
    def test_analyze_vendor(self):
        """Test comprehensive vendor analysis"""
        analysis = self.scorecard.analyze_vendor(self.sample_data, '@test_channel')
        
        required_keys = [
            'vendor_name', 'total_posts', 'posting_frequency', 'engagement',
            'price_metrics', 'top_performing_post', 'lending_score', 'risk_category'
        ]
        assert all(key in analysis for key in required_keys)
        assert analysis['vendor_name'] == '@test_channel'
        assert analysis['total_posts'] == 30
        assert 0 <= analysis['lending_score'] <= 100
        assert analysis['risk_category'] in ['High Priority', 'Medium Priority', 'Low Priority']
    
    def test_analyze_vendor_insufficient_data(self):
        """Test vendor analysis with insufficient data"""
        small_data = self.sample_data.head(5)  # Less than minimum threshold
        analysis = self.scorecard.analyze_vendor(small_data, '@test_channel')
        
        assert 'error' in analysis
        assert 'Insufficient data' in analysis['error']
    
    def test_create_vendor_comparison_table(self):
        """Test vendor comparison table creation"""
        # Create analyses for multiple vendors
        vendor_analyses = {
            '@vendor1': self.scorecard.analyze_vendor(self.sample_data, '@vendor1'),
            '@vendor2': self.scorecard.analyze_vendor(self.sample_data, '@vendor2')
        }
        
        comparison_table = self.scorecard.create_vendor_comparison_table(vendor_analyses)
        
        assert isinstance(comparison_table, pd.DataFrame)
        assert len(comparison_table) == 2
        assert 'Vendor' in comparison_table.columns
        assert 'Lending Score' in comparison_table.columns
        assert 'Risk Category' in comparison_table.columns
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes"""
        empty_df = pd.DataFrame()
        
        frequency = self.scorecard.calculate_posting_frequency(empty_df)
        assert frequency == 0.0
        
        engagement = self.scorecard.calculate_engagement_metrics(empty_df)
        assert engagement['avg_views'] == 0
        
        price_metrics = self.scorecard.calculate_price_metrics(empty_df)
        assert price_metrics['avg_price'] == 0