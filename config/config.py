"""Configuration settings for Amharic NER project."""

import os
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TelegramConfig:
    """Telegram API configuration."""
    api_id: str = os.getenv('TELEGRAM_API_ID', '')
    api_hash: str = os.getenv('TELEGRAM_API_HASH', '')
    phone: str = os.getenv('TELEGRAM_PHONE', '')
    
    # Target channels for scraping
    channels: List[str] = None
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [
                '@ethio_market_place',
                '@addis_shopping',
                '@ethio_electronics',
                '@bole_market',
                '@merkato_online',
                '@zemenExpress',
                '@shewabrand',
                '@lobelia4cosmetics',
                '@yetenaweg'
            ]

@dataclass
class ModelConfig:
    """Model training configuration."""
    model_names: List[str] = None
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    
    def __post_init__(self):
        if self.model_names is None:
            self.model_names = [
                'xlm-roberta-base',
                'distilbert-base-multilingual-cased',
                'bert-base-multilingual-cased'
            ]

@dataclass
class DataConfig:
    """Data processing configuration."""
    entity_labels: List[str] = None
    min_messages: int = 30
    max_messages: int = 1000
    
    def __post_init__(self):
        if self.entity_labels is None:
            self.entity_labels = ['O', 'B-Product', 'I-Product', 'B-LOC', 'I-LOC', 'B-PRICE', 'I-PRICE']

# Global configuration instances
telegram_config = TelegramConfig()
model_config = ModelConfig()
data_config = DataConfig()