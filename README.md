# Amharic E-commerce Data Extractor for NER

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.8+-brightgreen.svg)
![Contributions Welcome](https://img.shields.io/badge/Contributions-welcome-orange.svg)

This project provides an end-to-end pipeline for scraping, preprocessing, and preparing Amharic text data from Telegram e-commerce channels. The primary objective is to build a high-quality, labeled dataset for training a Named Entity Recognition (NER) model to automate product information extraction for **EthioMart**.

## Table of Contents
- [Amharic E-commerce Data Extractor for NER](#amharic-e-commerce-data-extractor-for-ner)
  - [Table of Contents](#table-of-contents)
  - [Project Goal](#project-goal)
  - [Features](#features)
  - [Workflow](#workflow)
  - [Project Structure](#project-structure)
  - [Setup and Installation](#setup-and-installation)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
  - [Usage Guide](#usage-guide)
    - [1. Data Ingestion \& Preprocessing](#1-data-ingestion--preprocessing)
    - [2. Automated Labeling \& CoNLL Formatting](#2-automated-labeling--conll-formatting)
    - [Using Scripts (Alternative)](#using-scripts-alternative)
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
    B --> C[/📄 Raw Data (.json)/];
    C --> D(🧼 Preprocessing & Cleaning);
    D --> E[/📝 Cleaned Data (.csv)/];
    E --> F(🤖 Automated Labeling);
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
│   ├── 1_Data_Ingestion.ipynb
│   └── 2_Data_Labeling.ipynb
│
├── src/
│   ├── ingestion/
│   │   └── telegram_scraper.py
│   ├── preprocessing/
│   │   └── amharic_text.py
│   └── labeling/
│       └── conll_formatter.py
│
├── scripts/
|   ├── task_1_ingest.py
|   └── task_2_label.py
|
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
4.  **Telegram API Credentials:**
    You will need to get your own API credentials from [my.telegram.org](https://my.telegram.org).

## Usage Guide

The primary way to run the pipeline is through the Jupyter notebooks provided in the `notebooks/` directory.

### 1. Data Ingestion & Preprocessing
- Open `notebooks/1_Data_Ingestion.ipynb`.
- **Provide Credentials**: In the second cell, replace the placeholder values for `API_ID` and `API_HASH` with your Telegram API keys.
- **Run the Notebook**: Execute the cells sequentially. The notebook will:
    1. Scrape raw data from the predefined Telegram channels.
    2. Save the raw messages to `data/raw/`.
    3. Preprocess the text data.
    4. Save the cleaned, processed data to `data/processed/`.

### 2. Automated Labeling & CoNLL Formatting
- Open `notebooks/2_Data_Labeling.ipynb`.
- **Run the Notebook**: Execute the cells to load the cleaned data and apply rule-based labeling for entities like `PRODUCT`, `PRICE`, `PHONE`, and `LOCATION`.
- **Generate CoNLL File**: The final cell will format the labeled data into a `.conll` file and save it in `data/processed/`, ready for NER model training.

### Using Scripts (Alternative)
You can also run the pipeline using the Python scripts in the `scripts/` directory.

```sh
# Run data ingestion (requires API credentials as arguments)
python scripts/task_1_ingest.py --api_id YOUR_ID --api_hash YOUR_HASH

# Run data labeling
python scripts/task_2_label.py
```

## Contributing

Contributions are highly encouraged! Please open an issue or submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License.