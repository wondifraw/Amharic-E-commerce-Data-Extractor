# Amharic E-commerce Data Extractor

## Table of Contents
- [Amharic E-commerce Data Extractor](#amharic-e-commerce-data-extractor)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Objectives](#objectives)
  - [Main Features](#main-features)
    - [Data Ingestion](#data-ingestion)
    - [Preprocessing](#preprocessing)
    - [Data Labeling](#data-labeling)
    - [NER Model Fine-Tuning](#ner-model-fine-tuning)
    - [Model Comparison \& Selection](#model-comparison--selection)
    - [Explainability](#explainability)
    - [Vendor Scorecard](#vendor-scorecard)
  - [Directory Structure](#directory-structure)
  - [Getting Started](#getting-started)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Install Requirements](#2-install-requirements)
    - [3. Run Main Tasks](#3-run-main-tasks)
    - [4. Use in Jupyter Notebooks](#4-use-in-jupyter-notebooks)
  - [Requirements](#requirements)
  - [Contributing](#contributing)
  - [Contact](#contact)
  - [GitHub](#github)

---

## Overview
This project provides a robust, modular pipeline for extracting, labeling, and analyzing entities from Amharic e-commerce data. It leverages state-of-the-art multilingual NLP models (such as XLM-Roberta, DistilBERT, and mBERT) for Named Entity Recognition (NER), and includes tools for data ingestion, preprocessing, model fine-tuning, comparison, explainability, and vendor scorecard generation.

## Objectives
- **Extract structured information** (products, prices, locations, etc.) from Amharic e-commerce sources.
- **Fine-tune and compare multiple NER models** to select the best for production.
- **Provide explainability and scorecard tools** for model outputs and vendor evaluation.

## Main Features

### Data Ingestion
- **Telegram Scraper:** Collects messages, media, and metadata from Amharic e-commerce Telegram channels using Telethon.
- **Flexible Source Support:** Easily extendable to other data sources.
- **Script:** `scripts/task_1_ingest.py`

### Preprocessing
- **Amharic Text Normalization:** Utilizes `etnltk` and custom rules for robust tokenization and normalization.
- **Data Cleaning:** Removes duplicates, empty values, and irrelevant content.
- **Script:** `src/preprocessing/amharic_text.py`

### Data Labeling
- **Manual & Automated Labeling:** Jupyter notebooks and scripts for labeling entities (e.g., PRODUCT, PRICE, LOCATION).
- **CoNLL Formatting:** Converts labeled data into standard CoNLL format for NER training.
- **Notebook:** `notebooks/2_Data_Labeling.ipynb`

### NER Model Fine-Tuning
- **Multilingual Model Support:** Fine-tune XLM-Roberta, DistilBERT, mBERT, or any HuggingFace-compatible model.
- **Unified Pipeline:** All utilities in `src/ner/ner_pipeline.py`.
- **Script:** `scripts/task_3_finetune.py`
- **Notebook:** `notebooks/Fine_Tune_NER_Model.ipynb`

### Model Comparison & Selection
- **Automated Evaluation:** Fine-tune and evaluate multiple models on the same dataset.
- **Metrics:** Compare by F1, accuracy, speed, and robustness.
- **Script:** `scripts/task_4_compare.py`
- **Notebook:** (You can create one using the script logic)

### Explainability
- **Model Interpretation:** Tools for explaining and visualizing NER model predictions.
- **Script:** `scripts/model_explainability.py` (rename as needed)

### Vendor Scorecard
- **Vendor Analysis:** Generate scorecards for vendors based on extracted entities and model outputs.
- **Script:** `scripts/task_6_vendor_scorecard.py`

## Directory Structure
```
Amharic-E-commerce-Data-Extractor/
├── data/                # Raw and processed datasets
├── models/              # Saved and fine-tuned model checkpoints
├── notebooks/           # Jupyter notebooks for exploration and tasks
├── scripts/             # Main scripts for each pipeline task
├── src/
│   └── ner/
│       └── ner_pipeline.py  # Unified NER data and training utilities
│   └── ...             # Other modules (preprocessing, labeling, etc.)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
```

## Getting Started

### 1. Clone the Repository
```bash
git clone <repo-url>
cd Amharic-E-commerce-Data-Extractor
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run Main Tasks
- **Data Ingestion:**
  ```bash
  python scripts/task_1_ingest.py
  ```
- **Data Labeling:**
  Use the provided Jupyter notebook: `notebooks/2_Data_Labeling.ipynb`
- **NER Fine-Tuning:**
  ```bash
  python scripts/task_3_finetune.py
  ```
- **Model Comparison & Selection:**
  ```bash
  python scripts/task_4_compare.py
  ```
- **Vendor Scorecard:**
  ```bash
  python scripts/task_6_vendor_scorecard.py
  ```

### 4. Use in Jupyter Notebooks
- All major pipeline functions are available in `src/ner/ner_pipeline.py`.
- Example workflow:
  ```python
  from src.ner.ner_pipeline import (
      load_conll_data, print_label_distribution, preview_samples,
      create_ner_datasets, train_ner_model
  )
  conll_path = 'data/processed/labeled_dataset.conll'
  all_tokens, all_labels, unique_labels = load_conll_data(conll_path)
  print_label_distribution(all_labels)
  preview_samples(all_tokens, all_labels, n=3)
  # ... continue with dataset creation and training ...
  ```
- See `notebooks/Fine_Tune_NER_Model.ipynb` for a step-by-step example.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies (transformers, datasets, scikit-learn, etc.)

## Contributing
Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.

## Contact
For questions or support, please contact the project maintainer or open an issue on GitHub.

## GitHub
GitHub: [your-username](https://github.com/wondifraw)

---
**Professional, modular, and ready for production or research use.**