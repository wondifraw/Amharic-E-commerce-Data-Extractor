"""
Vendor Analytics and Scoring System for Micro-lending
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from loguru import logger
import yaml


class VendorAnalyticsEngine:
    """Analyze vendor performance for micro-lending decisions"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.vendor_metrics = {}
        self.scoring_weights = self.config['vendor_analytics']['metrics']
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def analyze_vendor_activity(self, df: pd.DataFrame, vendor_channel: str) -> Dict:
        """Analyze activity metrics for a specific vendor"""
        vendor_data = df[df['channel'] == vendor_channel].copy()
        
        if len(vendor_data) == 0:
            logger.warning(f"No data found for vendor: {vendor_channel}")
            return {}
        
        # Convert date column to datetime
        vendor_data['date'] = pd.to_datetime(vendor_data['date'])
        vendor_data = vendor_data.sort_values('date')
        
        # Calculate time span
        date_range = vendor_data['date'].max() - vendor_data['date'].min()
        weeks_active = max(1, date_range.days / 7)
        
        # Activity metrics
        total_posts = len(vendor_data)
        posts_per_week = total_posts / weeks_active
        
        # Engagement metrics
        avg_views = vendor_data['views'].mean()
        total_views = vendor_data['views'].sum()
        avg_forwards = vendor_data['forwards'].mean()
        avg_replies = vendor_data['replies'].mean()
        
        # Top performing post
        top_post = vendor_data.loc[vendor_data['views'].idxmax()]
        
        # Consistency metrics
        posts_by_week = vendor_data.groupby(vendor_data['date'].dt.isocalendar().week).size()
        posting_consistency = 1 - (posts_by_week.std() / posts_by_week.mean()) if posts_by_week.mean() > 0 else 0
        posting_consistency = max(0, min(1, posting_consistency))  # Normalize to 0-1
        
        metrics = {
            'vendor_channel': vendor_channel,
            'total_posts': total_posts,
            'weeks_active': weeks_active,
            'posts_per_week': posts_per_week,
            'avg_views_per_post': avg_views,
            'total_views': total_views,
            'avg_forwards': avg_forwards,
            'avg_replies': avg_replies,
            'posting_consistency': posting_consistency,
            'top_post': {
                'id': top_post['id'],
                'text': top_post['text'][:100] + '...' if len(top_post['text']) > 100 else top_post['text'],
                'views': top_post['views'],
                'date': top_post['date'].isoformat()
            }
        }
        
        return metrics
    
    def extract_price_info(self, df: pd.DataFrame, vendor_channel: str, 
                          ner_predictions: Optional[Dict] = None) -> Dict:
        """Extract price information from vendor posts"""
        vendor_data = df[df['channel'] == vendor_channel].copy()
        
        if len(vendor_data) == 0:
            return {'avg_price': 0, 'price_range': (0, 0), 'price_count': 0}
        
        prices = []
        
        # Use NER predictions if available
        if ner_predictions and vendor_channel in ner_predictions:
            for post_id, entities in ner_predictions[vendor_channel].items():
                for entity in entities:
                    if entity['label'] in ['B-PRICE', 'I-PRICE']:
                        # Extract numeric value from price entity
                        price_text = entity['text']
                        price_value = self._extract_price_value(price_text)
                        if price_value > 0:
                            prices.append(price_value)
        else:
            # Fallback to pattern matching
            import re
            price_patterns = [
                r'ዋጋ\s*(\d+)\s*ብር',
                r'በ\s*(\d+)\s*ብር',
                r'(\d+)\s*ብር',
                r'ETB\s*(\d+)',
                r'(\d+)\s*birr'
            ]
            
            for _, row in vendor_data.iterrows():
                text = str(row['text'])
                for pattern in price_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        try:
                            price = float(match)
                            if 10 <= price <= 100000:  # Reasonable price range
                                prices.append(price)
                        except ValueError:
                            continue
        
        if not prices:
            return {'avg_price': 0, 'price_range': (0, 0), 'price_count': 0}
        
        return {
            'avg_price': np.mean(prices),
            'price_range': (min(prices), max(prices)),
            'price_count': len(prices),
            'price_std': np.std(prices)
        }
    
    def _extract_price_value(self, price_text: str) -> float:
        """Extract numeric value from price text"""
        import re
        
        # Remove common Amharic price words
        price_text = re.sub(r'ዋጋ|ብር|birr|ETB|በ', '', price_text, flags=re.IGNORECASE)
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', price_text)
        
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        
        return 0.0
    
    def calculate_lending_score(self, vendor_metrics: Dict, price_info: Dict) -> float:
        """Calculate lending score based on vendor metrics"""
        
        # Normalize metrics to 0-1 scale
        normalized_metrics = {}
        
        # Posts per week (normalize to 0-1, cap at 10 posts/week)
        normalized_metrics['posting_frequency'] = min(1.0, vendor_metrics['posts_per_week'] / 10.0)
        
        # Average views (normalize to 0-1, cap at 10000 views)
        normalized_metrics['avg_views'] = min(1.0, vendor_metrics['avg_views_per_post'] / 10000.0)
        
        # Price consistency (higher std deviation = lower score)
        if price_info['price_count'] > 1:
            price_cv = price_info['price_std'] / price_info['avg_price'] if price_info['avg_price'] > 0 else 1
            normalized_metrics['price_consistency'] = max(0, 1 - price_cv)
        else:
            normalized_metrics['price_consistency'] = 0.5  # Neutral score for single price
        
        # Engagement score (combination of forwards and replies)
        engagement_score = (vendor_metrics['avg_forwards'] + vendor_metrics['avg_replies']) / 100
        normalized_metrics['engagement'] = min(1.0, engagement_score)
        
        # Calculate weighted score
        lending_score = (
            normalized_metrics['posting_frequency'] * self.scoring_weights['posting_frequency_weight'] +
            normalized_metrics['avg_views'] * self.scoring_weights['avg_views_weight'] +
            normalized_metrics['price_consistency'] * self.scoring_weights['price_consistency_weight'] +
            normalized_metrics['engagement'] * self.scoring_weights['engagement_weight']
        )
        
        # Scale to 0-100
        lending_score *= 100
        
        return round(lending_score, 2)
    
    def analyze_all_vendors(self, df: pd.DataFrame, 
                           ner_predictions: Optional[Dict] = None) -> pd.DataFrame:
        """Analyze all vendors in the dataset"""
        vendors = df['channel'].unique()
        vendor_scores = []
        
        for vendor in vendors:
            logger.info(f"Analyzing vendor: {vendor}")
            
            # Get activity metrics
            activity_metrics = self.analyze_vendor_activity(df, vendor)
            
            if not activity_metrics:
                continue
            
            # Get price information
            price_info = self.extract_price_info(df, vendor, ner_predictions)
            
            # Calculate lending score
            lending_score = self.calculate_lending_score(activity_metrics, price_info)
            
            # Compile vendor profile
            vendor_profile = {
                'vendor_channel': vendor,
                'total_posts': activity_metrics['total_posts'],
                'posts_per_week': round(activity_metrics['posts_per_week'], 2),
                'avg_views_per_post': round(activity_metrics['avg_views_per_post'], 0),
                'avg_price_etb': round(price_info['avg_price'], 0),
                'price_range_min': round(price_info['price_range'][0], 0),
                'price_range_max': round(price_info['price_range'][1], 0),
                'lending_score': lending_score,
                'top_post_views': activity_metrics['top_post']['views'],
                'top_post_text': activity_metrics['top_post']['text'],
                'weeks_active': round(activity_metrics['weeks_active'], 1),
                'posting_consistency': round(activity_metrics['posting_consistency'], 2)
            }
            
            vendor_scores.append(vendor_profile)
        
        # Create DataFrame and sort by lending score
        results_df = pd.DataFrame(vendor_scores)
        results_df = results_df.sort_values('lending_score', ascending=False)
        
        logger.info(f"Analyzed {len(results_df)} vendors")
        return results_df
    
    def generate_vendor_report(self, vendor_scores_df: pd.DataFrame, 
                              output_path: str = "data/processed/vendor_scorecard.json") -> str:
        """Generate comprehensive vendor report"""
        
        # Summary statistics
        summary = {
            'total_vendors_analyzed': len(vendor_scores_df),
            'avg_lending_score': vendor_scores_df['lending_score'].mean(),
            'top_vendor': {
                'channel': vendor_scores_df.iloc[0]['vendor_channel'],
                'score': vendor_scores_df.iloc[0]['lending_score']
            },
            'score_distribution': {
                'high_score_vendors': len(vendor_scores_df[vendor_scores_df['lending_score'] >= 70]),
                'medium_score_vendors': len(vendor_scores_df[
                    (vendor_scores_df['lending_score'] >= 40) & 
                    (vendor_scores_df['lending_score'] < 70)
                ]),
                'low_score_vendors': len(vendor_scores_df[vendor_scores_df['lending_score'] < 40])
            }
        }
        
        # Top 5 vendors
        top_vendors = vendor_scores_df.head(5).to_dict('records')
        
        # Compile report
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': summary,
            'top_vendors': top_vendors,
            'all_vendors': vendor_scores_df.to_dict('records'),
            'methodology': {
                'scoring_weights': self.scoring_weights,
                'metrics_explanation': {
                    'posting_frequency': 'Average posts per week',
                    'avg_views': 'Average views per post',
                    'price_consistency': 'Consistency in pricing (lower std dev = higher score)',
                    'engagement': 'Average forwards and replies per post'
                }
            }
        }
        
        # Save report
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Vendor report saved to {output_path}")
        return output_path
    
    def create_lending_recommendations(self, vendor_scores_df: pd.DataFrame) -> Dict:
        """Create lending recommendations based on scores"""
        
        recommendations = {
            'high_priority': [],  # Score >= 70
            'medium_priority': [],  # Score 40-69
            'low_priority': [],  # Score < 40
            'criteria': {
                'high_priority': 'Score >= 70, consistent posting, high engagement',
                'medium_priority': 'Score 40-69, moderate activity and engagement',
                'low_priority': 'Score < 40, low activity or engagement'
            }
        }
        
        for _, vendor in vendor_scores_df.iterrows():
            vendor_info = {
                'channel': vendor['vendor_channel'],
                'score': vendor['lending_score'],
                'avg_views': vendor['avg_views_per_post'],
                'posts_per_week': vendor['posts_per_week'],
                'avg_price': vendor['avg_price_etb']
            }
            
            if vendor['lending_score'] >= 70:
                recommendations['high_priority'].append(vendor_info)
            elif vendor['lending_score'] >= 40:
                recommendations['medium_priority'].append(vendor_info)
            else:
                recommendations['low_priority'].append(vendor_info)
        
        return recommendations


def main():
    """Test vendor analytics"""
    # Load sample data (you would use your actual scraped data)
    sample_data = {
        'channel': ['@vendor1', '@vendor1', '@vendor2', '@vendor2', '@vendor3'] * 20,
        'text': [
            'የሕፃናት ጠርሙስ ዋጋ 150 ብር ቦሌ አካባቢ',
            'ልብስ በ 200 ብር አዲስ አበባ',
            'ጫማ 300 ብር መርካቶ',
            'ስልክ ETB 5000 ፒያሳ',
            'መጽሐፍ 50 ብር ሃያ ሁለት'
        ] * 20,
        'views': np.random.randint(100, 5000, 100),
        'forwards': np.random.randint(0, 50, 100),
        'replies': np.random.randint(0, 20, 100),
        'date': pd.date_range('2024-01-01', periods=100, freq='D')
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize analytics engine
    analytics = VendorAnalyticsEngine()
    
    # Analyze all vendors
    vendor_scores = analytics.analyze_all_vendors(df)
    
    print("Vendor Scorecard:")
    print(vendor_scores[['vendor_channel', 'posts_per_week', 'avg_views_per_post', 
                        'avg_price_etb', 'lending_score']].to_string(index=False))
    
    # Generate report
    report_path = analytics.generate_vendor_report(vendor_scores)
    
    # Create recommendations
    recommendations = analytics.create_lending_recommendations(vendor_scores)
    
    print(f"\nHigh Priority Vendors: {len(recommendations['high_priority'])}")
    print(f"Medium Priority Vendors: {len(recommendations['medium_priority'])}")
    print(f"Low Priority Vendors: {len(recommendations['low_priority'])}")


if __name__ == "__main__":
    main()