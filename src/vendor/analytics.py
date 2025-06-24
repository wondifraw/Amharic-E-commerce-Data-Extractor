"""
analytics.py
Module for calculating vendor performance metrics and a lending score.
"""
import re
import numpy as np
from typing import List, Dict
from datetime import datetime

def extract_price(text: str) -> float:
    """
    Extracts a numerical price from a string.
    It removes commas and common currency symbols/words like 'ብር'.
    """
    if not text:
        return None
    try:
        # Remove commas and currency symbols/words
        cleaned_text = re.sub(r'[,ብር\s]', '', text, flags=re.IGNORECASE)
        # Find the first numerical sequence
        match = re.search(r'\d+\.?\d*', cleaned_text)
        if match:
            return float(match.group(0))
    except (ValueError, TypeError):
        return None
    return None

def compute_vendor_metrics(
    messages: List[Dict],
    ner_results: List[Dict] = None
) -> Dict:
    """
    Computes key performance metrics for a single vendor from their messages.

    Args:
        messages: A list of message dictionaries for the vendor.
        ner_results: A list of corresponding NER results for each message.

    Returns:
        A dictionary containing the vendor's performance metrics.
    """
    if not messages:
        return {}

    # Metric: Posting Frequency (posts per week)
    dates = [datetime.fromisoformat(m['date']) for m in messages if 'date' in m]
    if dates:
        weeks = max((max(dates) - min(dates)).days / 7, 1)
        posting_frequency = len(messages) / weeks
    else:
        posting_frequency = 0

    # Metric: Average Views per Post
    views = [m.get('views', 0) for m in messages if m.get('views') is not None]
    avg_views = np.mean(views) if views else 0

    # Metric: Average Price Point
    all_prices = []
    if ner_results:
        for result in ner_results:
            for entity in result.get('entities', []):
                if entity.get('label') in ['B-PRICE', 'I-PRICE']:
                    price = extract_price(entity.get('text'))
                    if price:
                        all_prices.append(price)
    avg_price = np.mean(all_prices) if all_prices else 0
    
    # Metric: Top Performing Post
    top_post_info = {}
    if messages:
        top_post = max(messages, key=lambda m: m.get('views', 0))
        top_post_info = {
            'message_id': top_post.get('id'),
            'views': top_post.get('views'),
            'text': top_post.get('text')
        }

    return {
        "total_posts": len(messages),
        "posting_frequency_per_week": round(posting_frequency, 2),
        "avg_views_per_post": round(avg_views, 2),
        "avg_price_point": round(avg_price, 2),
        "top_performing_post": top_post_info,
    }

def calculate_lending_score(metrics: Dict) -> float:
    """
    Calculates a simple, weighted "Lending Score" for a vendor.

    The score is based on a weighted combination of average views and posting frequency.
    This formula can be adjusted based on business priorities.
    """
    avg_views = metrics.get("avg_views_per_post", 0)
    posting_freq = metrics.get("posting_frequency_per_week", 0)
    
    # Normalize or scale factors might be needed in a real-world scenario
    # to prevent one metric from dominating the score.
    # For now, we use a simple weighted sum.
    score = (avg_views * 0.5) + (posting_freq * 0.5)
    return round(score, 2) 