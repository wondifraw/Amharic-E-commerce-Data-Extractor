# 🔧 Environment Configuration Guide

## Quick Setup

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit with your credentials
nano .env  # or use any text editor

# 3. Validate configuration
python setup_env.py
```

## Required Variables

### Telegram API (Required)
```env
TG_API_ID=your_telegram_api_id
TG_API_HASH=your_telegram_api_hash
```

Get from: https://my.telegram.org/apps

### Optional Configuration
```env
# Model settings
BATCH_SIZE=16
LEARNING_RATE=2e-5
NUM_EPOCHS=3

# Data paths
CONLL_OUTPUT_PATH=./data/processed/conll_output.conll
TELEGRAM_DATA_PATH=./data/telegram_data.csv

# Performance
USE_GPU=true
DATALOADER_WORKERS=4
```

## Usage in Code

```python
from src.config import config

# Access configuration
api_id = config.TG_API_ID
batch_size = config.BATCH_SIZE
use_gpu = config.USE_GPU
```

## Validation

```bash
python setup_env.py  # Validates all required variables
```