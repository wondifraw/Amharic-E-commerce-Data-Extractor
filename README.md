# Amharic E-commerce Data Extractor for NER

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.8+-brightgreen.svg)
![Contributions Welcome](https://img.shields.io/badge/Contributions-welcome-orange.svg)

This project provides an end-to-end pipeline for scraping, preprocessing, and preparing Amharic text data from Telegram e-commerce channels. The primary objective is to build a high-quality, labeled dataset for training a Named Entity Recognition (NER) model to automate product information extraction for **EthioMart**.

## Table of Contents
- [Project Goal](#project-goal)
- [Features](#features)
- [Workflow](#workflow)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage Guide](#usage-guide)
- [Contributing](#contributing)
- [License](#license)

## Project Goal

The core business challenge is to automate the identification of products, brands, and other key entities from unstructured Amharic posts on Telegram. This NER system will enable EthioMart to streamline its product cataloging process, improve search accuracy, and enhance customer experience.

## Features

- **Telegram Scraper**: Fetches messages, media, and metadata from specified channels using Telethon.
- **Amharic Text Preprocessor**: Leverages `etnltk` for robust, language-specific normalization and tokenization.
- **Data Cleaner**: Includes utilities to remove duplicate entries and empty values, ensuring data quality.
- **CoNLL Formatter**: Converts labeled data into the standard CoNLL format, ready for NER model training.

## Workflow

The project follows a sequential data pipeline, from raw data collection to a model-ready format.

```mermaid
graph TD
    A[📢 Telegram Channels] --> B(📨 Telegram Scraper);
    B --> C[/📄 Raw Data (.csv)/];
    C --> D(🧼 Preprocessing & Cleaning);
    D --> E[/📝 Cleaned Data (.csv)/];
    E --> F(🏷️ Manual Labeling);
    F --> G[/📦 CoNLL Formatted Data/];
    G --> H(🤖 NER Model Training);
```

## Project Structure

```
Amharic-E-commerce-Data-Extractor/
│
├── data/
│   ├── processed/      # Cleaned and labeled data
│   └── raw/            # Raw scraped data from Telegram
│
├── notebooks/
│   ├── 1_Data_Ingestion_and_Preprocessing.ipynb
│   └── 2_Labeling_and_CoNLL_Conversion.ipynb
│
├── src/
│   ├── ingestion/
│   │   └── telegram_scraper.py
│   ├── preprocessing/
│   │   └── amharic_text.py
│   └── labeling/
│       └── conll_formatter.py
│
├── requirements.txt
└── README.md
```

## Setup and Installation

### Prerequisites
- Python 3.8+
- Git

### Installation Steps

1.  **Clone the Repository:**
    ```sh
    git clone <your-repository-url>
    cd Amharic-E-commerce-Data-Extractor
    ```

2.  **Set Up a Virtual Environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage Guide

### 1. Data Ingestion & Preprocessing
- Open `notebooks/1_Data_Ingestion_and_Preprocessing.ipynb`.
- **Provide Credentials**: Replace `'YOUR_API_ID'` and `'YOUR_API_HASH'` with your Telegram API keys.
- **Run the Notebook**: Execute the cells to scrape raw data and save it to `data/raw/`. The notebook will then preprocess the data and save the cleaned version to `data/processed/`.

### 2. Labeling & CoNLL Formatting
- Open `notebooks/2_Labeling_and_CoNLL_Conversion.ipynb`.
- **Perform Manual Annotation**: This notebook loads the cleaned data. You must manually define your entity labels (e.g., `B-PRODUCT`, `I-PRODUCT`) in the `labels` column. A placeholder is provided, which labels all tokens as `O` (Outside).
- **Generate CoNLL File**: Run the final cells to format the data into a `.conll` file, which will be saved in `data/processed/` and is ready for model training.

## Contributing

Contributions are highly encouraged! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.