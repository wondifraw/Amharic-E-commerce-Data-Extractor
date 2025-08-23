"""Vendor analytics and scorecard generation."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class VendorAnalytics:
    """Analytics engine for vendor scoring and evaluation."""
    
    def __init__(self):
        self.vendor_metrics = {}
        
    def calculate_vendor_metrics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate comprehensive metrics for each vendor/channel."""
        vendor_metrics = {}
        
        for channel in df['channel'].unique():
            channel_data = df[df['channel'] == channel].copy()
            
            if len(channel_data) == 0:
                continue
            
            # Basic activity metrics
            metrics = self._calculate_activity_metrics(channel_data)
            
            # Engagement metrics
            engagement_metrics = self._calculate_engagement_metrics(channel_data)
            metrics.update(engagement_metrics)
            
            # Business profile metrics (using NER results if available)
            business_metrics = self._calculate_business_metrics(channel_data)
            metrics.update(business_metrics)
            
            # Performance metrics
            performance_metrics = self._calculate_performance_metrics(channel_data)
            metrics.update(performance_metrics)
            
            vendor_metrics[channel] = metrics
        
        return vendor_metrics
    
    def _calculate_activity_metrics(self, channel_data: pd.DataFrame) -> Dict:
        """Calculate activity and consistency metrics."""
        try:
            # Convert date column if it's string
            if 'date' in channel_data.columns:
                channel_data['date'] = pd.to_datetime(channel_data['date'])
                
                # Calculate posting frequency
                date_range = (channel_data['date'].max() - channel_data['date'].min()).days
                if date_range > 0:
                    posts_per_week = len(channel_data) * 7 / date_range
                else:
                    posts_per_week = len(channel_data)
            else:
                posts_per_week = len(channel_data) / 4  # Assume 4 weeks of data
            
            return {
                'total_posts': len(channel_data),
                'posts_per_week': round(posts_per_week, 2),
                'activity_score': min(posts_per_week / 10, 1.0)  # Normalize to 0-1
            }
            
        except Exception as e:
            logger.warning(f"Error calculating activity metrics: {e}")
            return {
                'total_posts': len(channel_data),
                'posts_per_week': 0,
                'activity_score': 0
            }
    
    def _calculate_engagement_metrics(self, channel_data: pd.DataFrame) -> Dict:
        """Calculate engagement and reach metrics."""
        try:
            views = channel_data['views'].fillna(0)
            forwards = channel_data['forwards'].fillna(0)
            
            avg_views = views.mean()
            max_views = views.max()
            avg_forwards = forwards.mean()
            
            # Find top performing post
            top_post_idx = views.idxmax()
            top_post_text = channel_data.loc[top_post_idx, 'text'] if not views.empty else ""
            
            return {
                'avg_views_per_post': round(avg_views, 2),
                'max_views': int(max_views),
                'avg_forwards': round(avg_forwards, 2),
                'top_post_text': top_post_text[:100] + "..." if len(top_post_text) > 100 else top_post_text,
                'engagement_score': min(avg_views / 1000, 1.0)  # Normalize to 0-1
            }
            
        except Exception as e:
            logger.warning(f"Error calculating engagement metrics: {e}")
            return {
                'avg_views_per_post': 0,
                'max_views': 0,
                'avg_forwards': 0,
                'top_post_text': "",
                'engagement_score': 0
            }
    
    def _calculate_business_metrics(self, channel_data: pd.DataFrame) -> Dict:
        """Calculate business profile metrics using extracted entities."""
        try:
            prices = []
            products = []
            locations = []
            
            # Extract prices from entity hints if available
            if 'entity_hints' in channel_data.columns:
                for hints in channel_data['entity_hints'].fillna({}):
                    if isinstance(hints, dict):
                        prices.extend(hints.get('prices', []))
                        products.extend(hints.get('products', []))
                        locations.extend(hints.get('locations', []))
            
            # Convert price strings to numbers
            numeric_prices = []
            for price in prices:
                try:
                    # Extract numbers from price strings
                    import re
                    numbers = re.findall(r'\d+', str(price))
                    if numbers:
                        numeric_prices.append(int(numbers[0]))
                except:
                    continue
            
            avg_price = np.mean(numeric_prices) if numeric_prices else 0
            unique_products = len(set(products))
            unique_locations = len(set(locations))
            
            return {
                'avg_price_etb': round(avg_price, 2),
                'price_range': f"{min(numeric_prices)}-{max(numeric_prices)}" if numeric_prices else "N/A",
                'unique_products': unique_products,
                'unique_locations': unique_locations,
                'business_diversity_score': min((unique_products + unique_locations) / 10, 1.0)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating business metrics: {e}")
            return {
                'avg_price_etb': 0,
                'price_range': "N/A",
                'unique_products': 0,
                'unique_locations': 0,
                'business_diversity_score': 0
            }
    
    def _calculate_performance_metrics(self, channel_data: pd.DataFrame) -> Dict:
        """Calculate overall performance metrics."""
        try:
            # Text quality metrics
            avg_text_length = channel_data['text'].str.len().mean()
            
            # Consistency metrics (posting pattern)
            if 'date' in channel_data.columns:
                channel_data['date'] = pd.to_datetime(channel_data['date'])
                daily_posts = channel_data.groupby(channel_data['date'].dt.date).size()
                posting_consistency = 1 - (daily_posts.std() / (daily_posts.mean() + 1))
            else:
                posting_consistency = 0.5  # Default moderate consistency
            
            return {
                'avg_text_length': round(avg_text_length, 2),
                'posting_consistency': round(max(0, posting_consistency), 2),
                'content_quality_score': min(avg_text_length / 200, 1.0)  # Normalize
            }
            
        except Exception as e:
            logger.warning(f"Error calculating performance metrics: {e}")
            return {
                'avg_text_length': 0,
                'posting_consistency': 0,
                'content_quality_score': 0
            }
    
    def calculate_lending_score(self, metrics: Dict) -> float:
        """Calculate composite lending score."""
        try:
            # Weighted scoring formula
            score = (
                metrics.get('engagement_score', 0) * 0.4 +
                metrics.get('activity_score', 0) * 0.3 +
                metrics.get('business_diversity_score', 0) * 0.2 +
                metrics.get('content_quality_score', 0) * 0.1
            )
            
            return round(score * 100, 2)  # Convert to 0-100 scale
            
        except Exception as e:
            logger.warning(f"Error calculating lending score: {e}")
            return 0.0
    
    def generate_vendor_scorecard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive vendor scorecard."""
        try:
            vendor_metrics = self.calculate_vendor_metrics(df)
            
            scorecard_data = []
            
            for channel, metrics in vendor_metrics.items():
                lending_score = self.calculate_lending_score(metrics)
                
                scorecard_row = {
                    'Vendor_Channel': channel.replace('@', ''),
                    'Avg_Views_Per_Post': metrics.get('avg_views_per_post', 0),
                    'Posts_Per_Week': metrics.get('posts_per_week', 0),
                    'Avg_Price_ETB': metrics.get('avg_price_etb', 0),
                    'Total_Posts': metrics.get('total_posts', 0),
                    'Max_Views': metrics.get('max_views', 0),
                    'Unique_Products': metrics.get('unique_products', 0),
                    'Business_Diversity': metrics.get('business_diversity_score', 0),
                    'Content_Quality': metrics.get('content_quality_score', 0),
                    'Lending_Score': lending_score,
                    'Risk_Category': self._categorize_risk(lending_score)
                }
                
                scorecard_data.append(scorecard_row)
            
            scorecard_df = pd.DataFrame(scorecard_data)
            
            # Sort by lending score
            scorecard_df = scorecard_df.sort_values('Lending_Score', ascending=False)
            
            return scorecard_df
            
        except Exception as e:
            logger.error(f"Error generating scorecard: {e}")
            return pd.DataFrame()
    
    def _categorize_risk(self, lending_score: float) -> str:
        """Categorize vendor risk based on lending score."""
        if lending_score >= 80:
            return "Low Risk"
        elif lending_score >= 60:
            return "Medium Risk"
        elif lending_score >= 40:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def generate_detailed_report(self, df: pd.DataFrame, output_path: str) -> None:
        """Generate detailed vendor analytics report."""
        try:
            vendor_metrics = self.calculate_vendor_metrics(df)
            scorecard_df = self.generate_vendor_scorecard(df)
            
            # Create comprehensive report
            report = {
                'summary': {
                    'total_vendors_analyzed': len(vendor_metrics),
                    'avg_lending_score': scorecard_df['Lending_Score'].mean() if not scorecard_df.empty else 0,
                    'top_vendor': scorecard_df.iloc[0]['Vendor_Channel'] if not scorecard_df.empty else "N/A",
                    'analysis_date': datetime.now().isoformat()
                },
                'vendor_rankings': scorecard_df.to_dict('records'),
                'detailed_metrics': vendor_metrics,
                'recommendations': self._generate_lending_recommendations(scorecard_df)
            }
            
            # Save report
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Save scorecard CSV
            csv_path = output_path.replace('.json', '_scorecard.csv')
            scorecard_df.to_csv(csv_path, index=False)
            
            logger.info(f"Vendor analytics report saved to {output_path}")
            logger.info(f"Vendor scorecard saved to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error generating detailed report: {e}")
    
    def _generate_lending_recommendations(self, scorecard_df: pd.DataFrame) -> List[str]:
        """Generate lending recommendations based on analysis."""
        if scorecard_df.empty:
            return ["No vendor data available for analysis."]
        
        recommendations = []
        
        # Top performers
        top_vendors = scorecard_df[scorecard_df['Lending_Score'] >= 70]
        if not top_vendors.empty:
            recommendations.append(f"Consider priority lending to {len(top_vendors)} high-scoring vendors.")
        
        # Risk analysis
        high_risk = scorecard_df[scorecard_df['Risk_Category'].isin(['High Risk', 'Very High Risk'])]
        if not high_risk.empty:
            recommendations.append(f"Exercise caution with {len(high_risk)} high-risk vendors.")
        
        # Activity patterns
        active_vendors = scorecard_df[scorecard_df['Posts_Per_Week'] >= 5]
        if not active_vendors.empty:
            recommendations.append(f"{len(active_vendors)} vendors show high activity levels.")
        
        # Engagement insights
        high_engagement = scorecard_df[scorecard_df['Avg_Views_Per_Post'] >= 500]
        if not high_engagement.empty:
            recommendations.append(f"{len(high_engagement)} vendors demonstrate strong market reach.")
        
        return recommendations