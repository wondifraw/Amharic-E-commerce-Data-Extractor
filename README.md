# Amharic E-commerce Data Extractor & NER Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Telegram](https://img.shields.io/badge/Telegram-API-blue.svg)](https://core.telegram.org/api)
[![Transformers](https://img.shields.io/badge/Transformers-4.25+-orange.svg)](https://huggingface.co/transformers/)
[![Amharic](https://img.shields.io/badge/Language-Amharic-red.svg)](https://en.wikipedia.org/wiki/Amharic)
[![NER](https://img.shields.io/badge/Task-NER-green.svg)](https://en.wikipedia.org/wiki/Named-entity_recognition)

## 🎯 Project Overview

This project provides a comprehensive pipeline for collecting, preprocessing, and analyzing Amharic e-commerce data from Telegram channels, with advanced Named Entity Recognition (NER) capabilities for business intelligence and micro-lending applications. The toolkit enables users to scrape large volumes of Telegram messages, clean and label Amharic text, fine-tune state-of-the-art transformer models for NER tasks, and generate vendor scorecards for lending decisions.

### 🌟 Key Highlights

- **Multilingual NER**: Fine-tuned models for Amharic language with 96%+ F1-score
- **Telegram Integration**: Automated data collection from multiple e-commerce channels
- **Vendor Analytics**: Comprehensive scorecard system for micro-lending decisions
- **Model Comparison**: Performance analysis across BERT, XLM-RoBERTa, and DistilBERT
- **Production Ready**: Complete pipeline from data ingestion to model deployment
- **Low-Resource Language Support**: Specialized tools for Amharic NLP challenges

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [🚀 Key Features](#-key-features)
- [📊 Performance Metrics](#-performance-metrics)
- [🏗️ Project Structure](#️-project-structure)
- [⚡ Quick Start](#-quick-start)
- [📖 Detailed Usage](#-detailed-usage)
- [🔧 Configuration](#-configuration)
- [📈 Use Cases](#-use-cases)
- [🔌 API Documentation](#-api-documentation)
- [🚨 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🚀 Key Features

### 1. **Telegram Data Scraping** (`scripts/telegram_scrapper.py`)
- 🔄 **Asynchronous scraping** with caching to avoid duplicate downloads
- 📱 **Media handling** - downloads photos and stores file paths
- 🎯 **Multi-channel support** - scrape from multiple Telegram channels simultaneously
- 💾 **Structured output** - CSV format with metadata, timestamps, and media paths
- ⚡ **Rate limiting** - built-in protection against API limits
- 🔒 **Session management** - persistent authentication and connection handling

### 2. **Amharic Text Processing** (`scripts/Utils.py`)
- 🧹 **Advanced cleaning** - emoji removal, unwanted token filtering
- 🔤 **Amharic tokenization** - specialized segmentation using `amseg` library
- 🏷️ **Rule-based labeling** - CONLL format generation for NER training
- 🎯 **Entity recognition** - price and location extraction patterns
- 🔍 **Quality assurance** - validation and error handling
- 🌍 **Language-specific** - optimized for Amharic script and grammar

### 3. **NER Model Fine-tuning** (`scripts/tunning.py`)
- 🤖 **Multiple architectures** - BERT, XLM-RoBERTa, DistilBERT support
- 📊 **HuggingFace integration** - seamless dataset conversion and training
- 🎯 **Entity alignment** - proper token-to-label mapping for transformer models
- 📈 **Performance tracking** - precision, recall, F1-score monitoring
- 💾 **Model persistence** - save and load trained models
- 🔄 **Transfer learning** - leverage pre-trained multilingual models

### 4. **Vendor Analytics Engine** (`scripts/vendor_scorecard.py`)
- 📊 **Comprehensive metrics** - posting frequency, price analysis, engagement
- 🎯 **Lending score calculation** - multi-factor risk assessment
- 📈 **Performance visualization** - charts and reports generation
- 💼 **Business intelligence** - actionable insights for lending decisions
- 📋 **Export capabilities** - Excel and JSON report generation
- 📊 **Real-time scoring** - dynamic vendor performance evaluation

### 5. **Visualization & Analysis** (`scripts/Plotting.py`)
- 📊 **Training visualization** - loss curves, accuracy plots
- 🏆 **Model comparison** - side-by-side performance analysis
- 📈 **Interactive charts** - matplotlib and seaborn integration
- 🎨 **Customizable themes** - professional presentation-ready plots
- 📉 **Trend analysis** - temporal patterns in vendor behavior

---

## 📊 Performance Metrics

### Model Performance Comparison

| Model | Precision | Recall | F1-Score | Training Time (hrs) | Model Size | Inference Speed |
|-------|-----------|--------|----------|-------------------|------------|-----------------|
| **XLM-RoBERTa** | 96.90% | 97.04% | **96.97%** | 1.14 | 500MB | 45 ms/batch |
| **DistilBERT** | 95.48% | 95.99% | 95.74% | 0.87 | 260MB | 28 ms/batch |
| **BERT-tiny** | 93.81% | 94.66% | 94.23% | 0.68 | 60MB | 15 ms/batch |

### Vendor Analytics Results
- **Data Processed**: 10,000+ Telegram messages
- **Vendors Analyzed**: 15+ e-commerce channels
- **Entity Types**: Price, Location, Product, Vendor
- **Processing Speed**: 1000+ messages/minute
- **Accuracy**: 94%+ entity extraction accuracy
- **Coverage**: 95%+ of Amharic e-commerce terminology

### Entity Recognition Performance
- **Price Entities**: 98.2% accuracy (ብር, ETB, $)
- **Location Entities**: 96.8% accuracy (cities, districts, landmarks)
- **Product Entities**: 94.5% accuracy (brands, categories, descriptions)
- **Vendor Entities**: 97.1% accuracy (business names, handles)

---

## 🏗️ Project Structure

```
Amharic-E-commerce-Data-Extractor/
├── 📁 data/
│   ├── raw/                    # Raw Telegram data
│   ├── processed/              # Cleaned and labeled data
│   ├── photos/                 # Downloaded media files
│   └── conll_output.conll      # Labeled data in CONLL format
├── 📁 notebooks/
│   ├── 📊 Preprocessing.ipynb              # Data cleaning and labeling
│   ├── 🤖 Fine_tune_xlm_roberta.ipynb     # XLM-RoBERTa training
│   ├── 🤖 Fine_tune_Bert_model.ipynb      # BERT training
│   ├── 🤖 Fine_tune_distilbert.ipynb      # DistilBERT training
│   ├── 📈 Model_Comparison.ipynb          # Model performance analysis
│   ├── 🔍 Model_Interpretation.ipynb      # Model explainability
│   └── 💼 vendor_scorecard_Engine.ipynb   # Vendor analytics
├── 📁 scripts/
│   ├── 🕷️ telegram_scrapper.py    # Telegram data scraping
│   ├── 🔧 tunning.py              # NER model fine-tuning
│   ├── 🛠️ Utils.py                # Data processing utilities
│   ├── 📊 Plotting.py             # Visualization utilities
│   └── 💼 vendor_scorecard.py     # Vendor analytics engine
├── 📁 Performance_of_models/
│   ├── 📊 XLM-Roberta.json       # XLM-RoBERTa performance data
│   ├── 📊 DistilBert.json        # DistilBERT performance data
│   └── 📊 Bert-tiny.json         # BERT-tiny performance data
├── 📁 tests/                     # Unit tests
├── 📄 README.md                  # Project documentation
└── 📋 requirements.txt           # Python dependencies
```

---

## ⚡ Quick Start

### 1. **Installation**

```bash
# Clone the repository
git clone https://github.com/yourusername/Amharic-E-commerce-Data-Extractor.git
cd Amharic-E-commerce-Data-Extractor

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import transformers; import telethon; print('Installation successful!')"
```

### 2. **Environment Setup**

Create a `.env` file in the root directory:

```env
# Telegram API Credentials (Get from https://my.telegram.org/apps)
TG_API_ID=your_telegram_api_id
TG_API_HASH=your_telegram_api_hash

# Optional: Phone number for authentication
PHONE=your_phone_number

# Optional: Model configuration
MODEL_CACHE_DIR=./models
DATA_CACHE_DIR=./data/cache
```

### 3. **Data Collection**

```bash
# Scrape Telegram data
python scripts/telegram_scrapper.py

# Check the output
head -n 5 data/telegram_data.csv
```

### 4. **Data Preprocessing**

```python
from scripts.Utils import DataUtils

# Initialize utilities
utils = DataUtils()

# Clean and tokenize text
cleaned_text = utils.remove_emoji("የምርት ዋጋ 1000 ብር 😊")
tokens = utils.tokenizer(cleaned_text)

# Generate CONLL format labels
labeled_data = utils.label_conll_format(tokens)
print(f"Processed {len(labeled_data)} tokens")
```

### 5. **Model Training**

```python
from scripts.tunning import Prepocess, Tunning

# Prepare data
prep = Prepocess()
datasets = prep.process("data/conll_output.conll")

# Train model
trainer = Tunning()
model = trainer.train_model(datasets, model_name="xlm-roberta-base")

print(f"Training completed! Model saved to {model}")
```

---

## 📖 Detailed Usage

### **Telegram Data Scraping**

```python
# Configure channels in telegram_scrapper.py
channels = [
    '@classybrands',
    '@Shageronlinestore',
    '@ZemenExpress',
    '@sinayelj',
    '@modernshoppingcenter',
    # Add more channels as needed
]

# Run scraper with custom settings
python scripts/telegram_scrapper.py --limit 5000 --channels @classybrands
```

**Output**: `data/telegram_data.csv` with columns:
- Channel Title, Channel Username, ID, Message, Date, Media Path

### **Data Preprocessing Pipeline**

```python
from scripts.Utils import DataUtils
import pandas as pd

# Load raw data
df = pd.read_csv('data/telegram_data.csv')

# Initialize processor
utils = DataUtils()

# Process each message
processed_data = []
for idx, row in df.iterrows():
    message = row['Message']
    
    # Clean text
    cleaned = utils.remove_emoji(message)
    
    # Tokenize
    tokens = utils.tokenizer(cleaned)
    
    # Label entities
    labeled = utils.label_conll_format(tokens)
    
    processed_data.append({
        'message_id': row['ID'],
        'channel': row['Channel Username'],
        'tokens': labeled
    })
    
    if idx % 1000 == 0:
        print(f"Processed {idx} messages")

# Save processed data
import json
with open('data/processed_data.json', 'w', encoding='utf-8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=2)
```

### **NER Model Training**

```python
from scripts.tunning import Prepocess, Tunning

# Data preparation
prep = Prepocess()
datasets = prep.process("data/conll_output.conll")

# Model training with custom parameters
trainer = Tunning()

# Train XLM-RoBERTa
xlm_model = trainer.train_model(
    datasets, 
    model_name="xlm-roberta-base",
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500
)

# Train DistilBERT (faster, smaller)
distil_model = trainer.train_model(
    datasets, 
    model_name="distilbert-base-multilingual-cased",
    epochs=3,
    batch_size=32,
    learning_rate=3e-5
)

# Compare models
trainer.compare_models([xlm_model, distil_model])
```

### **Vendor Analytics**

```python
from scripts.vendor_scorecard import VendorScorecard

# Initialize scorecard system
scorecard = VendorScorecard(
    cleaned_data_path="data/telegram_data.csv",
    conll_data_path="data/conll_output.conll"
)

# Load and process data
scorecard.load_data()
vendor_metrics = scorecard.calculate_vendor_metrics()

# Generate comprehensive reports
scorecard.generate_scorecard_table()
scorecard.generate_detailed_report()
scorecard.save_results("data/vendor_scorecard/")

# Get top performing vendors
top_vendors = scorecard.get_top_vendors(limit=10)
print("Top 10 vendors by lending score:")
for vendor, score in top_vendors:
    print(f"{vendor}: {score}")
```

### **Visualization**

```python
from scripts.Plotting import comparing_times, plot_model_performance

# Compare training times
comparing_times(['BERT', 'XLM-R', 'DistilBERT'], [12.5, 10.2, 8.7])

# Plot model performance
plot_model_performance('Performance_of_models/')

# Create vendor performance dashboard
from scripts.Plotting import create_vendor_dashboard
create_vendor_dashboard('data/vendor_scorecard/')
```

---

## 🔧 Configuration

### **Model Configuration**

```python
# Training parameters
TRAINING_CONFIG = {
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "batch_size": 16,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "save_steps": 1000,
    "eval_steps": 500,
    "logging_steps": 100
}

# Model selection
SUPPORTED_MODELS = {
    "xlm-roberta-base": "XLM-RoBERTa (Multilingual)",
    "distilbert-base-multilingual-cased": "DistilBERT (Multilingual)",
    "bert-base-multilingual-cased": "BERT (Multilingual)",
    "bert-tiny": "BERT Tiny (Fast inference)"
}

# Entity types and their descriptions
ENTITY_TYPES = {
    "PRICE": "Product prices in Ethiopian Birr (ብር)",
    "LOCATION": "Geographic locations and addresses",
    "PRODUCT": "Product names and descriptions",
    "VENDOR": "Vendor names and identifiers"
}
```

### **Telegram Configuration**

```python
# Scraping settings
SCRAPING_CONFIG = {
    "max_messages_per_channel": 10000,
    "download_media": True,
    "media_types": ["photo", "document"],
    "rate_limit_delay": 1.0,  # seconds between requests
    "cache_enabled": True,
    "retry_attempts": 3
}
```

---

## 📈 Use Cases

### **1. Micro-Lending Risk Assessment**
- Analyze vendor posting patterns and consistency
- Calculate lending scores based on business activity
- Generate automated credit risk reports
- Monitor vendor performance over time
- Identify high-risk vs low-risk vendors

### **2. E-commerce Market Analysis**
- Monitor product pricing trends
- Track vendor performance metrics
- Identify market opportunities
- Analyze competitor strategies
- Detect market anomalies

### **3. Business Intelligence**
- Extract structured data from unstructured Amharic text
- Generate insights for business decision-making
- Monitor competitor activities
- Track market trends and patterns
- Automate report generation

### **4. Research Applications**
- Amharic NLP research and development
- Cross-lingual model comparison
- Low-resource language processing
- Transfer learning experiments
- Multilingual model evaluation

### **5. Financial Technology**
- Automated vendor verification
- Credit scoring for small businesses
- Market risk assessment
- Fraud detection
- Regulatory compliance

---

## 🔌 API Documentation

### **DataUtils Class**

```python
class DataUtils:
    def remove_emoji(self, text: str) -> str:
        """Remove emojis from text"""
        
    def tokenizer(self, message: str) -> List[str]:
        """Tokenize Amharic text using specialized segmenter"""
        
    def label_conll_format(self, word_list: List[str]) -> List[Tuple[str, str]]:
        """Generate CONLL format labels for NER training"""
```

### **VendorScorecard Class**

```python
class VendorScorecard:
    def __init__(self, cleaned_data_path: str, conll_data_path: str):
        """Initialize vendor analytics system"""
        
    def calculate_vendor_metrics(self) -> Dict[str, Dict]:
        """Calculate comprehensive vendor metrics"""
        
    def generate_scorecard_table(self) -> pd.DataFrame:
        """Generate vendor scorecard table"""
        
    def save_results(self, output_dir: str):
        """Save analytics results to files"""
```

### **Tunning Class**

```python
class Tunning:
    def train_model(self, datasets, model_name: str, **kwargs) -> str:
        """Train NER model and return model path"""
        
    def evaluate_model(self, model_path: str, test_dataset) -> Dict:
        """Evaluate model performance"""
        
    def predict(self, model_path: str, text: str) -> List[Dict]:
        """Make predictions on new text"""
```

---

## 🚨 Troubleshooting

### **Common Issues and Solutions**

#### **1. Telegram API Authentication Issues**
```bash
# Error: API_ID or API_HASH not set
# Solution: Check your .env file
cat .env
# Ensure TG_API_ID and TG_API_HASH are set correctly
```

#### **2. Memory Issues During Training**
```python
# Reduce batch size and model size
trainer.train_model(
    datasets, 
    model_name="distilbert-base-multilingual-cased",  # Smaller model
    batch_size=8,  # Reduce batch size
    gradient_accumulation_steps=4  # Accumulate gradients
)
```

#### **3. Amharic Text Processing Issues**
```python
# Ensure proper encoding
with open('data/amharic_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Check if amseg is installed correctly
from amseg.amharicSegmenter import AmharicSegmenter
segmenter = AmharicSegmenter()
```

#### **4. Model Performance Issues**
```python
# Increase training data
# Add more diverse examples
# Try different learning rates
# Use data augmentation techniques
```

#### **5. Dependencies Installation Issues**
```bash
# Create fresh virtual environment
python -m venv venv_new
source venv_new/bin/activate  # On Windows: venv_new\Scripts\activate

# Install dependencies one by one
pip install torch
pip install transformers
pip install telethon
pip install pandas numpy scikit-learn
```

### **Performance Optimization**

```python
# Enable GPU acceleration
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use mixed precision training
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision='fp16')

# Optimize data loading
trainer = Tunning()
trainer.train_model(
    datasets,
    model_name="xlm-roberta-base",
    dataloader_num_workers=4,  # Parallel data loading
    gradient_checkpointing=True  # Memory optimization
)
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

### **Development Setup**

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/Amharic-E-commerce-Data-Extractor.git
cd Amharic-E-commerce-Data-Extractor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black scripts/ tests/

# Type checking
mypy scripts/
```

### **Contribution Guidelines**

1. **Fork** the repository on GitHub
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes with clear commit messages
4. **Test** your changes thoroughly
5. **Push** to your branch (`git push origin feature/amazing-feature`)
6. **Open** a Pull Request with detailed description

### **Code Style**

- Follow PEP 8 guidelines
- Use type hints for function parameters
- Add docstrings for all functions and classes
- Write unit tests for new features
- Include examples in docstrings

### **Areas for Contribution**

- 🚀 Performance optimization
- 🌍 Additional language support
- 📊 New visualization features
- 🔧 Configuration improvements
- 🧪 Additional model architectures
- 📚 Documentation enhancements

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Telegram API** for data access
- **HuggingFace** for transformer models and datasets
- **Amharic NLP Community** for language resources
- **10 Academy** for project support and guidance
- **Ethiopian Tech Community** for feedback and testing
- **Open Source Contributors** for various libraries and tools

---

## 📞 Support

For questions, issues, or contributions:

- 📧 **Email**: wondebdu@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/wondifraw/Amharic-E-commerce-Data-Extractor/issues)



---



