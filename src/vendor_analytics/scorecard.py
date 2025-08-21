"""
Vendor Analytics and Scorecard for Micro-lending Decisions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import os
import json


class VendorScorecard:
    """Analyze vendor performance for micro-lending decisions"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # Scoring weights from config
        self.weights = self.config['vendor_analytics']['metrics']
        self.thresholds = self.config['vendor_analytics']['thresholds']
    
    def calculate_posting_frequency(self, df: pd.DataFrame) -> float:
        """Calculate average posts per week"""
        if len(df) == 0:
            return 0.0
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate date range
        date_range = (df['date'].max() - df['date'].min()).days
        weeks = max(date_range / 7, 1)  # At least 1 week
        
        posts_per_week = len(df) / weeks
        return posts_per_week
    
    def calculate_engagement_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate engagement metrics"""
        if len(df) == 0:
            return {'avg_views': 0, 'avg_forwards': 0, 'avg_replies': 0, 'engagement_rate': 0}
        
        # Handle missing values
        df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)
        df['forwards'] = pd.to_numeric(df['forwards'], errors='coerce').fillna(0)
        df['replies'] = pd.to_numeric(df['replies'], errors='coerce').fillna(0)
        
        avg_views = df['views'].mean()
        avg_forwards = df['forwards'].mean()
        avg_replies = df['replies'].mean()
        
        # Engagement rate: (forwards + replies) / views
        engagement_rate = (avg_forwards + avg_replies) / max(avg_views, 1)
        
        return {
            'avg_views': avg_views,
            'avg_forwards': avg_forwards,
            'avg_replies': avg_replies,
            'engagement_rate': engagement_rate
        }
    
    def extract_prices_from_text(self, text: str) -> List[float]:
        """Extract price values from text"""
        import re
        
        if not isinstance(text, str):
            return []
        
        prices = []
        
        # Price patterns
        patterns = [
            r'ዋጋ\s*(\d+)\s*ብር',
            r'በ\s*(\d+)\s*ብር',
            r'(\d+)\s*ብር',
            r'ETB\s*(\d+)',
            r'(\d+)\s*birr'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    price = float(match)
                    if 10 <= price <= 100000:  # Reasonable price range
                        prices.append(price)
                except ValueError:
                    continue
        
        return prices
    
    def calculate_price_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate price-related metrics"""
        all_prices = []
        
        for text in df['text'].fillna(''):
            prices = self.extract_prices_from_text(text)
            all_prices.extend(prices)
        
        if not all_prices:
            return {
                'avg_price': 0,
                'price_std': 0,
                'price_consistency': 0,
                'price_range': 0,
                'total_products_with_prices': 0
            }
        
        avg_price = np.mean(all_prices)
        price_std = np.std(all_prices)
        price_consistency = 1 - (price_std / max(avg_price, 1))  # Higher is more consistent
        price_range = max(all_prices) - min(all_prices)
        
        return {
            'avg_price': avg_price,
            'price_std': price_std,
            'price_consistency': max(0, price_consistency),  # Ensure non-negative
            'price_range': price_range,
            'total_products_with_prices': len(all_prices)
        }
    
    def find_top_performing_post(self, df: pd.DataFrame) -> Dict:
        """Find the post with highest views"""
        if len(df) == 0:
            return {'views': 0, 'text': '', 'date': None}
        
        df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)
        top_post_idx = df['views'].idxmax()
        top_post = df.loc[top_post_idx]
        
        # Extract price from top post
        prices = self.extract_prices_from_text(top_post['text'])
        top_price = prices[0] if prices else None
        
        return {
            'views': top_post['views'],
            'text': top_post['text'][:100] + '...' if len(top_post['text']) > 100 else top_post['text'],
            'date': top_post['date'],
            'price': top_price,
            'message_id': top_post.get('id', 'N/A')
        }
    
    def calculate_lending_score(self, vendor_metrics: Dict) -> float:
        """Calculate composite lending score (0-100)"""
        # Normalize metrics to 0-1 scale
        
        # Posting frequency (normalize to 0-1, assuming 10 posts/week is excellent)
        freq_score = min(vendor_metrics['posting_frequency'] / 10, 1.0)
        
        # Average views (normalize to 0-1, assuming 1000 views is excellent)
        views_score = min(vendor_metrics['engagement']['avg_views'] / 1000, 1.0)
        
        # Price consistency (already 0-1)
        price_score = vendor_metrics['price_metrics']['price_consistency']
        
        # Engagement rate (normalize to 0-1, assuming 0.1 is excellent)
        engagement_score = min(vendor_metrics['engagement']['engagement_rate'] / 0.1, 1.0)
        
        # Weighted composite score
        lending_score = (
            freq_score * self.weights['posting_frequency_weight'] +
            views_score * self.weights['avg_views_weight'] +
            price_score * self.weights['price_consistency_weight'] +
            engagement_score * self.weights['engagement_weight']
        ) * 100
        
        return min(lending_score, 100)  # Cap at 100
    
    def analyze_vendor(self, df: pd.DataFrame, vendor_name: str) -> Dict:
        """Comprehensive vendor analysis"""
        logger.info(f"Analyzing vendor: {vendor_name}")
        
        # Filter data for this vendor
        vendor_df = df[df['channel'] == vendor_name].copy()
        
        if len(vendor_df) < self.thresholds['min_posts_for_analysis']:
            logger.warning(f"Insufficient data for {vendor_name}: {len(vendor_df)} posts")
            return {
                'vendor_name': vendor_name,
                'error': f'Insufficient data: {len(vendor_df)} posts (minimum: {self.thresholds["min_posts_for_analysis"]})'
            }
        
        # Calculate all metrics
        posting_frequency = self.calculate_posting_frequency(vendor_df)
        engagement_metrics = self.calculate_engagement_metrics(vendor_df)
        price_metrics = self.calculate_price_metrics(vendor_df)
        top_post = self.find_top_performing_post(vendor_df)
        
        vendor_metrics = {
            'vendor_name': vendor_name,
            'total_posts': len(vendor_df),
            'posting_frequency': posting_frequency,
            'engagement': engagement_metrics,
            'price_metrics': price_metrics,
            'top_performing_post': top_post,
            'analysis_date': datetime.now().isoformat()
        }
        
        # Calculate lending score
        lending_score = self.calculate_lending_score(vendor_metrics)
        vendor_metrics['lending_score'] = lending_score
        
        # Risk classification
        if lending_score >= 70:
            risk_category = "High Priority"
            recommendation = "Ready for micro-lending"
        elif lending_score >= 40:
            risk_category = "Medium Priority"
            recommendation = "Requires additional assessment"
        else:
            risk_category = "Low Priority"
            recommendation = "High risk, not recommended"
        
        vendor_metrics['risk_category'] = risk_category
        vendor_metrics['recommendation'] = recommendation
        
        return vendor_metrics
    
    def analyze_all_vendors(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze all vendors in the dataset"""
        vendors = df['channel'].unique()
        vendor_analyses = {}
        
        for vendor in vendors:
            analysis = self.analyze_vendor(df, vendor)
            vendor_analyses[vendor] = analysis
        
        return vendor_analyses
    
    def create_vendor_comparison_table(self, vendor_analyses: Dict[str, Dict]) -> pd.DataFrame:
        """Create comparison table for all vendors"""
        comparison_data = []
        
        for vendor_name, analysis in vendor_analyses.items():
            if 'error' in analysis:
                continue
            
            comparison_data.append({
                'Vendor': vendor_name.replace('@', ''),
                'Avg. Views/Post': round(analysis['engagement']['avg_views'], 1),
                'Posts/Week': round(analysis['posting_frequency'], 1),
                'Avg. Price (ETB)': round(analysis['price_metrics']['avg_price'], 0) if analysis['price_metrics']['avg_price'] > 0 else 'N/A',
                'Price Consistency': round(analysis['price_metrics']['price_consistency'], 2),
                'Engagement Rate': round(analysis['engagement']['engagement_rate'], 3),
                'Lending Score': round(analysis['lending_score'], 1),
                'Risk Category': analysis['risk_category']
            })
        
        return pd.DataFrame(comparison_data).sort_values('Lending Score', ascending=False)
    
    def create_visualizations(self, vendor_analyses: Dict[str, Dict], output_dir: str = "models/vendor_analytics/"):
        """Create vendor analytics visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for plotting
        valid_analyses = {k: v for k, v in vendor_analyses.items() if 'error' not in v}
        
        if not valid_analyses:
            logger.warning("No valid vendor analyses for visualization")
            return
        
        vendors = list(valid_analyses.keys())
        lending_scores = [valid_analyses[v]['lending_score'] for v in vendors]
        avg_views = [valid_analyses[v]['engagement']['avg_views'] for v in vendors]
        posting_freq = [valid_analyses[v]['posting_frequency'] for v in vendors]
        avg_prices = [valid_analyses[v]['price_metrics']['avg_price'] for v in vendors]
        
        # Create comprehensive dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Lending Scores
        colors = ['green' if score >= 70 else 'orange' if score >= 40 else 'red' for score in lending_scores]
        bars1 = ax1.bar(range(len(vendors)), lending_scores, color=colors, alpha=0.7)
        ax1.set_title('Vendor Lending Scores', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Lending Score')
        ax1.set_xticks(range(len(vendors)))
        ax1.set_xticklabels([v.replace('@', '') for v in vendors], rotation=45, ha='right')
        ax1.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='High Priority (≥70)')
        ax1.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Medium Priority (≥40)')
        ax1.legend()
        
        # Add value labels on bars
        for bar, score in zip(bars1, lending_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Views vs Posting Frequency
        ax2.scatter(posting_freq, avg_views, s=100, alpha=0.7, c=lending_scores, cmap='RdYlGn')
        ax2.set_xlabel('Posts per Week')
        ax2.set_ylabel('Average Views per Post')
        ax2.set_title('Engagement: Views vs Posting Frequency')
        
        for i, vendor in enumerate(vendors):
            ax2.annotate(vendor.replace('@', ''), (posting_freq[i], avg_views[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 3. Price Distribution
        valid_prices = [p for p in avg_prices if p > 0]
        if valid_prices:
            ax3.hist(valid_prices, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_xlabel('Average Price (ETB)')
            ax3.set_ylabel('Number of Vendors')
            ax3.set_title('Distribution of Average Prices')
        else:
            ax3.text(0.5, 0.5, 'No price data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Price Distribution - No Data')
        
        # 4. Risk Category Distribution
        risk_categories = [valid_analyses[v]['risk_category'] for v in vendors]
        risk_counts = pd.Series(risk_categories).value_counts()
        
        colors_pie = {'High Priority': 'green', 'Medium Priority': 'orange', 'Low Priority': 'red'}
        pie_colors = [colors_pie.get(cat, 'gray') for cat in risk_counts.index]
        
        ax4.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                colors=pie_colors, startangle=90)
        ax4.set_title('Risk Category Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vendor_analytics_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Vendor analytics visualizations saved to {output_dir}")
    
    def generate_lending_report(self, vendor_analyses: Dict[str, Dict], output_path: str = "models/vendor_analytics/lending_report.json"):
        """Generate comprehensive lending report"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create summary statistics
        valid_analyses = {k: v for k, v in vendor_analyses.items() if 'error' not in v}
        
        if not valid_analyses:
            logger.warning("No valid analyses for report generation")
            return
        
        lending_scores = [v['lending_score'] for v in valid_analyses.values()]
        risk_categories = [v['risk_category'] for v in valid_analyses.values()]
        
        summary = {
            'report_date': datetime.now().isoformat(),
            'total_vendors_analyzed': len(valid_analyses),
            'lending_score_stats': {
                'mean': np.mean(lending_scores),
                'median': np.median(lending_scores),
                'std': np.std(lending_scores),
                'min': np.min(lending_scores),
                'max': np.max(lending_scores)
            },
            'risk_distribution': dict(pd.Series(risk_categories).value_counts()),
            'top_vendors': sorted(valid_analyses.items(), 
                                key=lambda x: x[1]['lending_score'], reverse=True)[:5]
        }
        
        report = {
            'summary': summary,
            'detailed_analyses': vendor_analyses,
            'methodology': {
                'scoring_weights': self.weights,
                'thresholds': self.thresholds,
                'score_interpretation': {
                    'high_priority': '≥70 - Ready for micro-lending',
                    'medium_priority': '40-69 - Requires additional assessment',
                    'low_priority': '<40 - High risk, not recommended'
                }
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Lending report saved to {output_path}")
        return report


def main():
    """Test vendor analytics system"""
    # Create sample data for testing
    sample_data = {
        'channel': ['@ethio_market_place'] * 20 + ['@addis_shopping'] * 15 + ['@bole_market'] * 25,
        'text': [
            'ሰላም! የሕፃናት ጠርሙስ ዋጋ 150 ብር ነው። ቦሌ አካባቢ ነው።',
            'አዲስ አበባ ውስጥ የሚሸጥ ልብስ በ 200 ብር',
            'መርካቶ ውስጥ ጫማ 300 ብር',
            'ፒያሳ አካባቢ ስልክ ETB 5000',
            'Baby bottle for sale 150 birr in Bole'
        ] * 12,
        'views': np.random.randint(50, 1000, 60),
        'forwards': np.random.randint(0, 50, 60),
        'replies': np.random.randint(0, 20, 60),
        'date': pd.date_range('2024-01-01', periods=60, freq='D')
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize scorecard
    scorecard = VendorScorecard()
    
    # Analyze all vendors
    vendor_analyses = scorecard.analyze_all_vendors(df)
    
    # Create comparison table
    comparison_table = scorecard.create_vendor_comparison_table(vendor_analyses)
    print("Vendor Scorecard:")
    print(comparison_table.to_string(index=False))
    
    # Create visualizations
    scorecard.create_visualizations(vendor_analyses)
    
    # Generate report
    report = scorecard.generate_lending_report(vendor_analyses)
    
    print(f"\nAnalyzed {len(vendor_analyses)} vendors")
    print(f"High Priority vendors: {len([v for v in vendor_analyses.values() if v.get('risk_category') == 'High Priority'])}")


if __name__ == "__main__":
    main()