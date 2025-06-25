import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report


class Prepocess:

  def read_conll_file(self, file_path):
      with open(file_path, "r") as f:
          content = f.read().strip()
          sentences = content.split("\n\n")
          data = []
          for sentence in sentences:
              tokens = sentence.split("\n")
              token_data = []
              for token in tokens:
                  token_data.append(token.split())
              data.append(token_data)
      return data

  def convert_to_dataset(self, data, label_map, chunk_size=15):
      formatted_data = {"tokens": [], "ner_tags": []}

      for sentence in data:
          tokens = [token_data[0] for token_data in sentence]
          ner_tags = [label_map[token_data[1]] for token_data in sentence]

          # Split tokens and ner_tags into chunks
          for i in range(0, len(tokens), chunk_size):
              chunk_tokens = tokens[i:i + chunk_size]
              chunk_ner_tags = ner_tags[i:i + chunk_size]
              formatted_data["tokens"].append(chunk_tokens)
              formatted_data["ner_tags"].append(chunk_ner_tags)

      return Dataset.from_dict(formatted_data)


  def process(self, filepath):
      data = self.read_conll_file(filepath)

      label_list = sorted(list(set([token_data[1] for sentence in data for token_data in sentence])))
      label_map = {label: i for i, label in enumerate(label_list)}

      dataset = self.convert_to_dataset(data, label_map, chunk_size=15)

      datasets = dataset.train_test_split(test_size=0.2)

      train_valid = datasets['train'].train_test_split(test_size=0.2)
      final_datasets = DatasetDict({
                                  'train': train_valid['train'],
                                  'validation': train_valid['test'],
                                  'test': datasets['test']
                                  })

      return final_datasets



class Tunning:
  def compute_metrics(self, eval_prediction):
      predictions, labels = eval_prediction
      predictions = np.argmax(predictions, axis=2)


      # Remove ignored index (special tokens)
      true_predictions = [
          [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
          for prediction, label in zip(predictions, labels)
      ]
      true_labels = [
          [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
          for prediction, label in zip(predictions, labels)
      ]


      return {
          "precision": precision_score(true_labels, true_predictions),
          "recall": recall_score(true_labels, true_predictions),
          "f1": f1_score(true_labels, true_predictions),
      }

  def tokenize_and_align_labels(self,examples):
      tokenized_inputs = tokenizer(
          examples["tokens"], truncation=True, is_split_into_words=True, padding=True
      )
      labels = []
      for i, label in enumerate(examples["ner_tags"]):
          word_ids = tokenized_inputs.word_ids(batch_index=i)
          previous_word_idx = None
          label_ids = []
          for word_idx in word_ids:
              if word_idx is None:
                  label_ids.append(-100)
              elif word_idx != previous_word_idx:
                  label_ids.append(label[word_idx])
              else:
                  label_ids.append(-100)
              previous_word_idx = word_idx
          labels.append(label_ids)
      tokenized_inputs["labels"] = labels
      return tokenized_inputs



  def data_collator(self, data):
      input_ids = [torch.tensor(item["input_ids"]) for item in data]
      attention_mask = [torch.tensor(item["attention_mask"]) for item in data]
      labels = [torch.tensor(item["labels"]) for item in data]


      input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
      attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
      labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)


      return {
          "input_ids": input_ids,
          "attention_mask": attention_mask,
          "labels": labels,
      }

  def tokenize_train_args(self, datasets, save_strategy = 'epoch',epochs=1, eval_strategy='epoch'):
      self.tokenized_datasets = datasets.map(self.tokenize_and_align_labels, batched=True)


      self.training_args = TrainingArguments(
          output_dir="./results",
          evaluation_strategy=eval_strategy,
          save_strategy=save_strategy,
          eval_steps=500,
          save_steps=500,
          num_train_epochs=epochs,
          per_device_train_batch_size=8,
          per_device_eval_batch_size=8,
          logging_dir="./logs",
          logging_strategy='epoch',
          logging_steps=100,
          learning_rate=5e-5,
          load_best_model_at_end=True,
          metric_for_best_model="f1",
      )



  def train(self, tokenizer, model):
      trainer = Trainer(
          model=model,

          args=self.training_args,
          train_dataset=self.tokenized_datasets["train"],
          eval_dataset=self.tokenized_datasets["validation"],
          data_collator=self.data_collator,
          tokenizer=tokenizer,
          compute_metrics=self.compute_metrics,
      )

      return trainer