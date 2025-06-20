"""
dataset.py
Module for loading and preparing CoNLL-formatted data for NER tasks.
"""
from typing import List, Tuple, Dict
from datasets import Dataset
from sklearn.model_selection import train_test_split

def load_conll_data(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Loads data from a CoNLL-formatted file.

    Args:
        file_path (str): The path to the CoNLL file.

    Returns:
        A tuple containing two lists: one for sentences (lists of tokens) and one for
        their corresponding labels.
    """
    sentences, labels = [], []
    current_tokens, current_labels = [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if current_tokens:
                        sentences.append(current_tokens)
                        labels.append(current_labels)
                        current_tokens, current_labels = [], []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        current_tokens.append(parts[0])
                        current_labels.append(parts[1])
        
        # Add the last sentence if the file doesn't end with a newline
        if current_tokens:
            sentences.append(current_tokens)
            labels.append(current_labels)
            
    except FileNotFoundError:
        print(f"[ERROR] The file was not found at {file_path}")
    except Exception as e:
        print(f"[ERROR] An error occurred while reading {file_path}: {e}")
        
    return sentences, labels

def create_ner_datasets(
    all_tokens: List[List[str]],
    all_labels: List[List[str]],
    label_list: List[str],
    tokenizer
) -> Tuple[Dataset, Dataset, Dict, Dict]:
    """
    Creates and preprocesses training and validation datasets for NER.

    Args:
        all_tokens: A list of tokenized sentences.
        all_labels: A list of corresponding label sequences.
        label_list: The master list of all possible entity labels.
        tokenizer: A Hugging Face tokenizer instance.

    Returns:
        A tuple containing the tokenized train and validation datasets, and the
        label-to-ID and ID-to-label mapping dictionaries.
    """
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for i, label in enumerate(label_list)}

    # Split data into training and validation sets
    train_tokens, val_tokens, train_labels, val_labels = train_test_split(
        all_tokens, all_labels, test_size=0.2, random_state=42
    )

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128
        )
        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                else:
                    label_ids.append(-100) # Only label the first token of a word
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Create Hugging Face Dataset objects
    train_dataset = Dataset.from_dict({"tokens": train_tokens, "labels": train_labels})
    val_dataset = Dataset.from_dict({"tokens": val_tokens, "labels": val_labels})

    # Tokenize and align
    tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

    return tokenized_train_dataset, tokenized_val_dataset, label_to_id, id_to_label 