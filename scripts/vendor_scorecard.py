import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class VendorScorecard:
    """
    Vendor Analytics Engine for Micro-Lending Scorecard
    Processes Telegram posts and NER outputs to calculate vendor performance metrics
    """
    
    def __init__(self, cleaned_data_path: str, conll_data_path: str):
        """
        Initialize the Vendor Scorecard system
        
        Args:
            cleaned_data_path: Path to cleaned Telegram messages CSV
            conll_data_path: Path to CONLL NER output file
        """
        self.cleaned_data_path = cleaned_data_path
        self.conll_data_path = conll_data_path
        self.telegram_data = None
        self.ner_data = None
        self.vendor_metrics = {}
        
    def load_data(self):
        """Load and prepare the data for analysis"""
        print("Loading Telegram data...")
        self.telegram_data = pd.read_csv(self.cleaned_data_path)
        
        # Convert date column to datetime
        self.telegram_data['Date'] = pd.to_datetime(self.telegram_data['Date'])
        
        print("Loading NER data...")
        self.ner_data = self._parse_conll_file()
        
        print(f"Loaded {len(self.telegram_data)} Telegram messages")
        print(f"Loaded {len(self.ner_data)} NER entities")
        
    def _parse_conll_file(self) -> List[Dict]:
        """
        Parse CONLL format NER output file
        Returns list of dictionaries with entity information
        """
        entities = []
        current_entity = {"text": "", "type": "", "start_pos": 0}
        
        with open(self.conll_data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:  # Empty line indicates end of entity
                    if current_entity["text"]:
                        entities.append(current_entity.copy())
                        current_entity = {"text": "", "type": "", "start_pos": 0}
                    continue
                
                # Parse CONLL format: word\tlabel
                parts = line.split('\t')
                if len(parts) >= 2:
                    word, label = parts[0], parts[1]
                    
                    if label.startswith('B-'):  # Beginning of entity
                        if current_entity["text"]:
                            entities.append(current_entity.copy())
                        current_entity = {
                            "text": word,
                            "type": label[2:],  # Remove B- prefix
                            "start_pos": line_num
                        }
                    elif label.startswith('I-') and current_entity["text"]:  # Inside entity
                        current_entity["text"] += " " + word
                    else:  # O (Outside) or other
                        if current_entity["text"]:
                            entities.append(current_entity.copy())
                            current_entity = {"text": "", "type": "", "start_pos": 0}
        
        # Add last entity if exists
        if current_entity["text"]:
            entities.append(current_entity)
            
        return entities
    
    def extract_prices_from_messages(self) -> Dict[int, List[float]]:
        """
        Extract price information from message text using regex patterns
        Returns dictionary mapping message ID to list of prices found
        """
        price_patterns = [
            r'(\d+(?:,\d+)*)\s*ብር',  # Numbers followed by "ብር" (birr)
            r'(\d+(?:,\d+)*)\s*ETB',  # Numbers followed by ETB
            r'(\d+(?:,\d+)*)\s*\$',   # Numbers followed by dollar sign
            r'(\d+(?:,\d+)*)\s*አማርኛ',  # Numbers followed by Amharic currency
        ]
        
        message_prices = {}
        
        for _, row in self.telegram_data.iterrows():
            message_id = row['ID']
            message_text = str(row['Message'])
            prices = []
            
            for pattern in price_patterns:
                matches = re.findall(pattern, message_text, re.IGNORECASE)
                for match in matches:
                    # Clean and convert price
                    price_str = match.replace(',', '')
                    try:
                        price = float(price_str)
                        if 1 <= price <= 1000000:  # Reasonable price range
                            prices.append(price)
                    except ValueError:
                        continue
            
            if prices:
                message_prices[message_id] = prices
                
        return message_prices
    
    def calculate_vendor_metrics(self) -> Dict[str, Dict]:
        """
        Calculate comprehensive metrics for each vendor
        Returns dictionary with vendor metrics
        """
        print("Calculating vendor metrics...")
        
        # Group data by vendor (Channel Username)
        vendor_groups = self.telegram_data.groupby('Channel Username')
        
        # Extract prices from messages
        message_prices = self.extract_prices_from_messages()
        
        for vendor, vendor_data in vendor_groups:
            print(f"Processing vendor: {vendor}")
            
            # Basic posting metrics
            total_posts = len(vendor_data)
            date_range = vendor_data['Date'].max() - vendor_data['Date'].min()
            weeks_active = max(1, date_range.days / 7)  # Minimum 1 week
            posts_per_week = total_posts / weeks_active
            
            # Price analysis
            vendor_prices = []
            for _, row in vendor_data.iterrows():
                message_id = row['ID']
                if message_id in message_prices:
                    vendor_prices.extend(message_prices[message_id])
            
            avg_price = np.mean(vendor_prices) if vendor_prices else 0
            max_price = max(vendor_prices) if vendor_prices else 0
            
            # Engagement analysis (simulated since no views data)
            # Using message length as proxy for engagement potential
            avg_message_length = vendor_data['Message'].str.len().mean()
            
            # Product diversity (from NER entities)
            vendor_products = [entity for entity in self.ner_data 
                             if entity['type'] == 'PRODUCT']
            product_diversity = len(set(entity['text'] for entity in vendor_products))
            
            # Calculate lending score
            lending_score = self._calculate_lending_score(
                posts_per_week=posts_per_week,
                avg_price=avg_price,
                avg_message_length=avg_message_length,
                product_diversity=product_diversity,
                total_posts=total_posts
            )
            
            # Store vendor metrics
            self.vendor_metrics[vendor] = {
                'total_posts': total_posts,
                'posts_per_week': round(posts_per_week, 2),
                'avg_price_etb': round(avg_price, 2),
                'max_price_etb': round(max_price, 2),
                'avg_message_length': round(avg_message_length, 0),
                'product_diversity': product_diversity,
                'lending_score': round(lending_score, 2),
                'weeks_active': round(weeks_active, 1),
                'date_range': {
                    'start': vendor_data['Date'].min().strftime('%Y-%m-%d'),
                    'end': vendor_data['Date'].max().strftime('%Y-%m-%d')
                }
            }
        
        return self.vendor_metrics
    
    def _calculate_lending_score(self, posts_per_week: float, avg_price: float, 
                                avg_message_length: float, product_diversity: int,
                                total_posts: int) -> float:
        """
        Calculate lending score based on multiple factors
        Returns score between 0-100
        """
        # Normalize each factor to 0-100 scale
        posting_score = min(100, (posts_per_week / 10) * 100)  # 10 posts/week = 100 score
        price_score = min(100, (avg_price / 10000) * 100)  # 10,000 ETB = 100 score
        engagement_score = min(100, (avg_message_length / 500) * 100)  # 500 chars = 100 score
        diversity_score = min(100, (product_diversity / 20) * 100)  # 20 products = 100 score
        activity_score = min(100, (total_posts / 50) * 100)  # 50 posts = 100 score
        
        # Weighted average (adjust weights based on business priorities)
        weights = {
            'posting': 0.25,      # Regular activity
            'price': 0.20,        # Price point
            'engagement': 0.20,   # Message quality
            'diversity': 0.15,    # Product variety
            'activity': 0.20      # Total activity
        }
        
        final_score = (
            posting_score * weights['posting'] +
            price_score * weights['price'] +
            engagement_score * weights['engagement'] +
            diversity_score * weights['diversity'] +
            activity_score * weights['activity']
        )
        
        return final_score
    
    def generate_scorecard_table(self) -> pd.DataFrame:
        """
        Generate the vendor scorecard summary table
        Returns DataFrame with key metrics for each vendor
        """
        if not self.vendor_metrics:
            self.calculate_vendor_metrics()
        
        # Create summary table
        scorecard_data = []
        for vendor, metrics in self.vendor_metrics.items():
            scorecard_data.append({
                'Vendor': vendor,
                'Posts/Week': metrics['posts_per_week'],
                'Avg. Price (ETB)': metrics['avg_price_etb'],
                'Lending Score': metrics['lending_score'],
                'Total Posts': metrics['total_posts'],
                'Product Diversity': metrics['product_diversity']
            })
        
        scorecard_df = pd.DataFrame(scorecard_data)
        
        # Sort by lending score (descending)
        scorecard_df = scorecard_df.sort_values('Lending Score', ascending=False)
        
        return scorecard_df
    
    def generate_detailed_report(self) -> Dict:
        """
        Generate detailed vendor analysis report
        Returns comprehensive report dictionary
        """
        if not self.vendor_metrics:
            self.calculate_vendor_metrics()
        
        # Top performing vendors
        sorted_vendors = sorted(self.vendor_metrics.items(), 
                              key=lambda x: x[1]['lending_score'], 
                              reverse=True)
        
        report = {
            'summary': {
                'total_vendors': len(self.vendor_metrics),
                'total_posts': sum(m['total_posts'] for m in self.vendor_metrics.values()),
                'avg_lending_score': np.mean([m['lending_score'] for m in self.vendor_metrics.values()]),
                'date_range': {
                    'start': self.telegram_data['Date'].min().strftime('%Y-%m-%d'),
                    'end': self.telegram_data['Date'].max().strftime('%Y-%m-%d')
                }
            },
            'top_vendors': sorted_vendors[:5],  # Top 5 vendors
            'vendor_metrics': self.vendor_metrics,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate lending recommendations based on vendor analysis"""
        recommendations = []
        
        if not self.vendor_metrics:
            return recommendations
        
        # Analyze overall patterns
        avg_score = np.mean([m['lending_score'] for m in self.vendor_metrics.values()])
        high_score_vendors = [v for v, m in self.vendor_metrics.items() 
                            if m['lending_score'] >= 70]
        low_activity_vendors = [v for v, m in self.vendor_metrics.items() 
                              if m['posts_per_week'] < 2]
        
        recommendations.append(f"Average lending score across all vendors: {avg_score:.1f}/100")
        recommendations.append(f"High-performing vendors (score ≥70): {len(high_score_vendors)}")
        recommendations.append(f"Low-activity vendors (<2 posts/week): {len(low_activity_vendors)}")
        
        if high_score_vendors:
            recommendations.append("Recommended lending candidates: " + ", ".join(high_score_vendors[:3]))
        
        return recommendations
    
    def save_results(self, output_dir: str = "data/vendor_scorecard"):
        """
        Save all results to files
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save scorecard table
        scorecard_df = self.generate_scorecard_table()
        scorecard_df.to_csv(f"{output_dir}/vendor_scorecard.csv", index=False)
        
        # Save detailed report
        detailed_report = self.generate_detailed_report()
        with open(f"{output_dir}/detailed_report.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, indent=2, ensure_ascii=False)
        
        # Save vendor metrics
        with open(f"{output_dir}/vendor_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(self.vendor_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_dir}/")
    
    def print_scorecard(self):
        """Print the vendor scorecard table to console"""
        scorecard_df = self.generate_scorecard_table()
        
        print("\n" + "="*80)
        print("VENDOR SCORECARD FOR MICRO-LENDING")
        print("="*80)
        print(scorecard_df.to_string(index=False))
        print("="*80)
        
        # Print recommendations
        report = self.generate_detailed_report()
        print("\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"• {rec}")


# def main():
#     """Main function to run the Vendor Scorecard analysis"""
#     # Initialize the scorecard system
#     scorecard = VendorScorecard(
#         cleaned_data_path="data/processed/telegram_cleaned_messages.csv",
#         conll_data_path="data/processed/conll_output.conll"
#     )
    
#     # Load and process data
#     scorecard.load_data()
    
#     # Calculate metrics
#     scorecard.calculate_vendor_metrics()
    
#     # Generate and display results
#     scorecard.print_scorecard()
    
#     # Save results
#     scorecard.save_results()
    
#     print("\nVendor Scorecard analysis completed successfully!")


# if __name__ == "__main__":
#     main() 