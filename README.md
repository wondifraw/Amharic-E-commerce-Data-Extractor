# Amharic E-commerce Data Extractor & NER Fine-Tuning

## Project Summary

This project provides a robust pipeline for collecting, preprocessing, and analyzing Amharic e-commerce data from Telegram channels, with a focus on Named Entity Recognition (NER) for business intelligence and micro-lending applications. The toolkit enables users to scrape large volumes of Telegram messages, clean and label Amharic text, and fine-tune state-of-the-art transformer models for NER tasks. The project is designed for researchers, data scientists, and practitioners working with Amharic language data in the context of e-commerce and financial technology.

---

## Table of Contents

- [Amharic E-commerce Data Extractor \& NER Fine-Tuning](#amharic-e-commerce-data-extractor--ner-fine-tuning)
  - [Project Summary](#project-summary)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Key Features](#key-features)
    - [1. Telegram Data Scraping (`scripts/telegram_scrapper.py`)](#1-telegram-data-scraping-scriptstelegram_scrapperpy)
    - [2. Data Preprocessing \& Labeling (`scripts/Utils.py`)](#2-data-preprocessing--labeling-scriptsutilspy)
    - [3. NER Data Preparation \& Fine-Tuning (`scripts/tunning.py`)](#3-ner-data-preparation--fine-tuning-scriptstunningpy)
    - [4. Visualization (`scripts/Plotting.py`)](#4-visualization-scriptsplottingpy)
    - [5. Jupyter Notebooks (`notebooks/`)](#5-jupyter-notebooks-notebooks)
  - [Quick Start](#quick-start)
    - [1. Install Dependencies](#1-install-dependencies)
    - [2. Set Up Telegram API Credentials](#2-set-up-telegram-api-credentials)
    - [3. Scrape Telegram Data](#3-scrape-telegram-data)
    - [4. Preprocess and Label Data](#4-preprocess-and-label-data)
    - [5. Prepare Data \& Fine-tune NER Model](#5-prepare-data--fine-tune-ner-model)
    - [6. Visualize Results](#6-visualize-results)
  - [Example Usage](#example-usage)
    - [Data Cleaning \& Labeling](#data-cleaning--labeling)
    - [NER Data Preparation \& Training](#ner-data-preparation--training)
    - [Plotting](#plotting)
  - [Data Files](#data-files)
  - [Notebooks](#notebooks)
  - [Contributing](#contributing)

---
## Project Structure

```
Amharic-E-commerce-Data-Extractor/
├── data/
│   ├── raw/                    # Raw Telegram data
│   └── processed/              # Cleaned and labeled data
│   └── conll_output.conll      # Labeled data in CONLL format
├── notebooks/                  # Jupyter notebooks for model training and analysis
│   ├── Fine_tune_xlm_roberta.ipynb
│   ├── Fine_tune_Bert_model.ipynb
│   ├── Fine_tune_distilbert.ipynb
│   ├── Model_Comparison.ipynb
│   ├── Model_Interpretation.ipynb
│   └── Preprocessing.ipynb
├── scripts/                    # Main scripts for data and modeling
│   ├── telegram_scrapper.py    # Telegram data scraping
│   ├── tunning.py              # NER data preparation and model fine-tuning
│   ├── Utils.py                # Data cleaning, tokenization, and labeling utilities
│   └── Plotting.py             # Visualization utilities
├── README.md
└── requirements.txt
```

---

## Key Features

### 1. Telegram Data Scraping (`scripts/telegram_scrapper.py`)
- Efficiently scrapes messages and media from specified Telegram channels using the Telethon library.
- Stores structured data in CSV format, including channel metadata, message content, timestamps, and media file paths.
- Implements caching to avoid duplicate downloads and supports large-scale data collection for downstream analysis.

### 2. Data Preprocessing & Labeling (`scripts/Utils.py`)
- Cleans Amharic text by removing emojis and unwanted tokens, ensuring high-quality input for NER.
- Tokenizes Amharic messages using the `amseg` library, tailored for the unique structure of the Amharic language.
- Applies rule-based labeling to generate CONLL-formatted data, with special handling for price and location entities.

### 3. NER Data Preparation & Fine-Tuning (`scripts/tunning.py`)
- Reads CONLL-formatted data and converts it into HuggingFace Datasets for seamless integration with transformer models.
- Splits data into training, validation, and test sets, supporting robust model evaluation.
- Tokenizes and aligns entity labels for transformer-based NER models (e.g., BERT, XLM-RoBERTa).
- Fine-tunes models using the HuggingFace Trainer API, with built-in metrics for precision, recall, and F1-score.

### 4. Visualization (`scripts/Plotting.py`)
- Provides utilities for visualizing model training times and performance metrics using matplotlib.
- Enables clear comparison of different NER models and experimental results.

### 5. Jupyter Notebooks (`notebooks/`)
- End-to-end workflows for data preprocessing, model training, comparison, and interpretation.
- Designed for interactive experimentation, rapid prototyping, and in-depth analysis.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Telegram API Credentials
Create a `.env` file in the root directory with the following content:
```
TG_API_ID=your_telegram_api_id
TG_API_HASH=your_telegram_api_hash
```

### 3. Scrape Telegram Data
```bash
python scripts/telegram_scrapper.py
```
- Output: `data/telegram_data.csv` and downloaded media in `data/photos/`

### 4. Preprocess and Label Data
Use the utilities in `scripts/Utils.py` or the `Preprocessing.ipynb` notebook to clean and label your data for NER.

### 5. Prepare Data & Fine-tune NER Model
Use `scripts/tunning.py` or the provided notebooks to:
- Convert labeled data to HuggingFace Datasets
- Fine-tune transformer models for Amharic NER

### 6. Visualize Results
Use `scripts/Plotting.py` or the analysis notebooks to visualize training times and model performance.

---

## Example Usage

### Data Cleaning & Labeling
```python
from scripts.Utils import DataUtils

utils = DataUtils()
cleaned = utils.remove_emoji("some text 😊")
tokens = utils.tokenizer(cleaned)
labeled = utils.label_conll_format(tokens)
```

### NER Data Preparation & Training
```python
from scripts.tunning import Prepocess, Tunning

prep = Prepocess()
datasets = prep.process("data/conll_output.conll")
# ... then use Tunning for model training
```

### Plotting
```python
from scripts.Plotting import comparing_times
comparing_times(['BERT', 'XLM-R', 'DistilBERT'], [12.5, 10.2, 8.7])
```

---

## Data Files

- **Raw Data:** `data/raw/`
- **Processed Data:** `data/processed/`
- **Labeled Data:** `data/conll_output.conll`

---

## Notebooks

- **Preprocessing.ipynb:** Data cleaning and labeling
- **Fine_tune_xlm_roberta.ipynb, Fine_tune_Bert_model.ipynb, Fine_tune_distilbert.ipynb:** Model training and evaluation
- **Model_Comparison.ipynb:** Compare NER models
- **Model_Interpretation.ipynb:** Analyze and interpret model predictions

---

## Contributing

Contributions are welcome! To contribute to this project:

1. Fork the repository on GitHub.
2. Create a new feature branch (`git checkout -b feature/your-feature-name`).
3. Make your changes and commit them with clear messages.
4. Push your branch to your forked repository.
5. Open a Pull Request (PR) to the main repository, describing your changes and why they should be merged.
6. Ensure your code follows the existing style and passes all tests (if applicable).
