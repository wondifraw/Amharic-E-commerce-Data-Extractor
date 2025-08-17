"""Environment configuration management for the entire project."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

class Config:
    """Centralized configuration management."""
    
    # Telegram API
    TG_API_ID: str = os.getenv('TG_API_ID', '')
    TG_API_HASH: str = os.getenv('TG_API_HASH', '')
    PHONE: str = os.getenv('PHONE', '')
    
    # Model Configuration
    MODEL_CACHE_DIR: str = os.getenv('MODEL_CACHE_DIR', './models')
    DATA_CACHE_DIR: str = os.getenv('DATA_CACHE_DIR', './data/cache')
    
    # Training Configuration
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '16'))
    LEARNING_RATE: float = float(os.getenv('LEARNING_RATE', '2e-5'))
    NUM_EPOCHS: int = int(os.getenv('NUM_EPOCHS', '3'))
    
    # Data Paths
    CONLL_OUTPUT_PATH: str = os.getenv('CONLL_OUTPUT_PATH', './data/processed/conll_output.conll')
    TELEGRAM_DATA_PATH: str = os.getenv('TELEGRAM_DATA_PATH', './data/telegram_data.csv')
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', './logs/app.log')
    
    # Vendor Analytics
    VENDOR_SCORECARD_OUTPUT: str = os.getenv('VENDOR_SCORECARD_OUTPUT', './reports/vendor_analytics')
    
    # Performance
    USE_GPU: bool = os.getenv('USE_GPU', 'true').lower() == 'true'
    DATALOADER_WORKERS: int = int(os.getenv('DATALOADER_WORKERS', '4'))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        required = ['TG_API_ID', 'TG_API_HASH']
        missing = [key for key in required if not getattr(cls, key)]
        
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
        return True

# Global config instance
config = Config()