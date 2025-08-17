"""
Amharic E-commerce Data Extractor & NER Fine-Tuning Package

A comprehensive toolkit for collecting, preprocessing, and analyzing Amharic e-commerce data
from Telegram channels with advanced Named Entity Recognition capabilities.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "wondebdu@gmail.com"

from .data_collection import TelegramScraper
from .preprocessing import DataProcessor, TextCleaner
from .models import NERTrainer, ModelEvaluator
from .analytics import VendorScorecard
from .utils import AmharicTokenizer, EntityLabeler

__all__ = [
    "TelegramScraper",
    "DataProcessor", 
    "TextCleaner",
    "NERTrainer",
    "ModelEvaluator", 
    "VendorScorecard",
    "AmharicTokenizer",
    "EntityLabeler"
]