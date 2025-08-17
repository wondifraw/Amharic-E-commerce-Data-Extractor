"""Professional vendor analytics and scoring system."""

import pandas as pd
import numpy as np
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ScoringWeights:
    """Weights for different scoring factors."""
    posting_frequency: float = 0.25
    price_consistency: float = 0.20
    message_quality: float = 0.20
    product_diversity: float = 0.15
    total_activity: float = 0.20


class VendorScorecard:
    """Professional vendor analytics engine for micro-lending decisions."""
    
    def __init__(self, telegram_data_path: str, ner_data_path: str):
        self.telegram_data_path = telegram_data_path
        self.ner_data_path = ner_data_path
        self.telegram_data = None
        self.ner_entities = None
        self.vendor_metrics = {}
        self.scoring_weights = ScoringWeights()
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def load_data(self) -> None:
        """Load and validate input data."""
        try:
            self.logger.info("Loading Telegram data...")
            self.telegram_data = pd.read_csv(self.telegram_data_path)
            self.telegram_data['Date'] = pd.to_datetime(self.telegram_data['Date'])
            
            self.logger.info("Loading NER entities...")
            self.ner_entities = self._parse_conll_file()
            
            self.logger.info(f"Loaded {len(self.telegram_data)} messages from {self.telegram_data['Channel Username'].nunique()} vendors")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
            
    def _parse_conll_file(self) -> List[Dict]:
        """Parse CONLL format NER output."""
        entities = []
        current_entity = {"text": "", "type": "", "confidence": 1.0}
        
        try:
            with open(self.ner_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        if current_entity["text"]:
                            entities.append(current_entity.copy())
                            current_entity = {"text": "", "type": "", "confidence": 1.0}
                        continue
                        
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        word, label = parts[0], parts[1]
                        
                        if label.startswith('B-'):
                            if current_entity["text"]:
                                entities.append(current_entity.copy())
                            current_entity = {
                                "text": word,
                                "type": label[2:],
                                "confidence": 1.0
                            }
                        elif label.startswith('I-') and current_entity["text"]:
                            current_entity["text"] += " " + word
                            
                if current_entity["text"]:
                    entities.append(current_entity)
                    
        except Exception as e:
            self.logger.warning(f"Error parsing CONLL file: {e}")
            return []
            
        return entities
        
    def extract_prices(self) -> Dict[str, List[float]]:
        """Extract price information from messages."""
        price_patterns = [
            r'(\d+(?:,\d+)*)\s*ብር',
            r'(\d+(?:,\d+)*)\s*ETB',
            r'(\d+(?:,\d+)*)\s*\$',
            r'ዋጋ\s*(\d+(?:,\d+)*)',
        ]
        
        vendor_prices = {}
        
        for vendor in self.telegram_data['Channel Username'].unique():
            vendor_data = self.telegram_data[self.telegram_data['Channel Username'] == vendor]
            prices = []
            
            for _, row in vendor_data.iterrows():
                message = str(row['Message'])
                for pattern in price_patterns:
                    matches = re.findall(pattern, message, re.IGNORECASE)
                    for match in matches:
                        try:
                            price = float(match.replace(',', ''))
                            if 1 <= price <= 1000000:  # Reasonable range
                                prices.append(price)
                        except ValueError:
                            continue
                            
            vendor_prices[vendor] = prices
            
        return vendor_prices
        
    def calculate_vendor_metrics(self) -> Dict[str, Dict]:
        """Calculate comprehensive vendor metrics."""
        if self.telegram_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        self.logger.info("Calculating vendor metrics...")
        vendor_prices = self.extract_prices()
        
        for vendor in self.telegram_data['Channel Username'].unique():
            vendor_data = self.telegram_data[self.telegram_data['Channel Username'] == vendor]
            
            # Basic metrics
            total_posts = len(vendor_data)
            date_range = vendor_data['Date'].max() - vendor_data['Date'].min()
            weeks_active = max(1, date_range.days / 7)
            posts_per_week = total_posts / weeks_active
            
            # Price analysis
            prices = vendor_prices.get(vendor, [])
            avg_price = np.mean(prices) if prices else 0
            price_std = np.std(prices) if len(prices) > 1 else 0
            
            # Message quality
            avg_message_length = vendor_data['Message'].str.len().mean()
            
            # Product diversity (from NER)
            vendor_products = [e for e in self.ner_entities if e['type'] == 'PRODUCT']
            unique_products = len(set(e['text'] for e in vendor_products))
            
            # Calculate lending score
            lending_score = self._calculate_lending_score(
                posts_per_week, avg_price, avg_message_length, 
                unique_products, total_posts, price_std
            )
            
            self.vendor_metrics[vendor] = {
                'total_posts': total_posts,
                'posts_per_week': round(posts_per_week, 2),
                'avg_price_etb': round(avg_price, 2),
                'price_volatility': round(price_std, 2),
                'avg_message_length': round(avg_message_length, 0),
                'product_diversity': unique_products,
                'lending_score': round(lending_score, 2),
                'weeks_active': round(weeks_active, 1),
                'risk_category': self._categorize_risk(lending_score)
            }
            
        return self.vendor_metrics
        
    def _calculate_lending_score(self, posts_per_week: float, avg_price: float,
                               avg_message_length: float, product_diversity: int,
                               total_posts: int, price_volatility: float) -> float:
        """Calculate comprehensive lending score."""
        # Normalize metrics to 0-100 scale
        posting_score = min(100, (posts_per_week / 10) * 100)
        price_score = min(100, (avg_price / 10000) * 100)
        quality_score = min(100, (avg_message_length / 500) * 100)
        diversity_score = min(100, (product_diversity / 20) * 100)
        activity_score = min(100, (total_posts / 100) * 100)
        
        # Penalty for high price volatility
        volatility_penalty = min(20, (price_volatility / 1000) * 20)
        
        # Weighted score
        final_score = (
            posting_score * self.scoring_weights.posting_frequency +
            price_score * self.scoring_weights.price_consistency +
            quality_score * self.scoring_weights.message_quality +
            diversity_score * self.scoring_weights.product_diversity +
            activity_score * self.scoring_weights.total_activity
        ) - volatility_penalty
        
        return max(0, min(100, final_score))
        
    def _categorize_risk(self, score: float) -> str:
        """Categorize lending risk based on score."""
        if score >= 80:
            return "Low Risk"
        elif score >= 60:
            return "Medium Risk"
        elif score >= 40:
            return "High Risk"
        else:
            return "Very High Risk"
            
    def generate_scorecard_report(self) -> Dict:
        """Generate comprehensive scorecard report."""
        if not self.vendor_metrics:
            self.calculate_vendor_metrics()
            
        # Sort vendors by score
        sorted_vendors = sorted(
            self.vendor_metrics.items(),
            key=lambda x: x[1]['lending_score'],
            reverse=True
        )
        
        return {
            'summary': {
                'total_vendors': len(self.vendor_metrics),
                'avg_score': np.mean([m['lending_score'] for m in self.vendor_metrics.values()]),
                'low_risk_count': sum(1 for m in self.vendor_metrics.values() if m['risk_category'] == 'Low Risk'),
                'analysis_date': datetime.now().isoformat()
            },
            'top_vendors': sorted_vendors[:10],
            'risk_distribution': self._get_risk_distribution(),
            'recommendations': self._generate_recommendations()
        }
        
    def _get_risk_distribution(self) -> Dict[str, int]:
        """Get distribution of vendors by risk category."""
        distribution = {}
        for metrics in self.vendor_metrics.values():
            category = metrics['risk_category']
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
        
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        high_performers = [v for v, m in self.vendor_metrics.items() 
                          if m['lending_score'] >= 70]
        low_activity = [v for v, m in self.vendor_metrics.items() 
                       if m['posts_per_week'] < 1]
        
        if high_performers:
            recommendations.append(f"Consider priority lending to: {', '.join(high_performers[:3])}")
            
        if len(low_activity) > len(self.vendor_metrics) * 0.3:
            recommendations.append("High number of low-activity vendors detected - consider engagement strategies")
            
        return recommendations
        
    def save_results(self, output_dir: str = "reports/vendor_analytics") -> None:
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save scorecard report
        report = self.generate_scorecard_report()
        with open(output_path / "vendor_scorecard_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # Save metrics as CSV
        df = pd.DataFrame.from_dict(self.vendor_metrics, orient='index')
        df.index.name = 'Vendor'
        df.to_csv(output_path / "vendor_metrics.csv")
        
        self.logger.info(f"Results saved to {output_path}")
        
    def get_top_vendors(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top vendors by lending score."""
        if not self.vendor_metrics:
            self.calculate_vendor_metrics()
            
        return sorted(
            [(v, m['lending_score']) for v, m in self.vendor_metrics.items()],
            key=lambda x: x[1],
            reverse=True
        )[:limit]