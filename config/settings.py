"""Application configuration settings."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class DataConfig:
    """Data-related configuration."""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    cache_dir: str = "data/cache"
    media_dir: str = "data/raw/media"
    
    
@dataclass
class ModelConfig:
    """Model training configuration."""
    models_dir: str = "models"
    supported_models: Dict[str, str] = None
    default_model: str = "xlm-roberta-base"
    max_sequence_length: int = 512
    
    def __post_init__(self):
        if self.supported_models is None:
            self.supported_models = {
                "xlm-roberta-base": "XLM-RoBERTa (Multilingual)",
                "distilbert-base-multilingual-cased": "DistilBERT (Multilingual)",
                "bert-base-multilingual-cased": "BERT (Multilingual)",
            }


@dataclass
class TelegramConfig:
    """Telegram scraping configuration."""
    api_id: str = os.getenv('TG_API_ID', '')
    api_hash: str = os.getenv('TG_API_HASH', '')
    session_name: str = 'amharic_scraper'
    max_messages_per_channel: int = 10000
    rate_limit_delay: float = 1.0
    download_media: bool = True
    
    
@dataclass
class EntityConfig:
    """Entity recognition configuration."""
    entity_types: List[str] = None
    price_indicators: List[str] = None
    location_indicators: List[str] = None
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = ["PRICE", "LOC", "PRODUCT", "VENDOR"]
        if self.price_indicators is None:
            self.price_indicators = ["ዋጋ", "ብር", "ETB", "$"]
        if self.location_indicators is None:
            self.location_indicators = ["አድራሻ", "ቦታ", "አካባቢ"]


@dataclass
class AppConfig:
    """Main application configuration."""
    project_root: Path = Path(__file__).parent.parent
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    telegram: TelegramConfig = TelegramConfig()
    entity: EntityConfig = EntityConfig()
    
    def __post_init__(self):
        # Ensure directories exist
        for dir_path in [
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.data.cache_dir,
            self.data.media_dir,
            self.model.models_dir,
            "logs",
            "reports"
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = AppConfig()