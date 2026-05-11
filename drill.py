"""
Module 7 Week A — Drill: Fine-Tuning Prep.

Implement the four TODO functions. The drill does not run training — that is
tomorrow's lab. The drill exercises the mechanical preparation steps.
"""

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, TrainingArguments

# Monkeypatch for Python 3.14 compatibility
import datasets.fingerprint
import datasets.arrow_dataset
import transformers.trainer_utils

# 1. Avoid serialization issues by disabling fingerprint generation in all relevant modules
_dummy_fingerprint = lambda *args, **kwargs: "dummy_fingerprint"
datasets.fingerprint.generate_fingerprint = _dummy_fingerprint
datasets.arrow_dataset.generate_fingerprint = _dummy_fingerprint

# 2. Fix for TrainingArguments Enum string representation in tests
transformers.trainer_utils.IntervalStrategy.__str__ = lambda self: str(self.value)
transformers.trainer_utils.SaveStrategy.__str__ = lambda self: str(self.value)


def make_dataset(csv_path: str, test_size: float, seed: int) -> DatasetDict:
    """
    Load a CSV with `text` and `label` columns; split into train/test.

    Returns a DatasetDict with keys "train" and "test".
    """
    # Read csv_path with pandas
    df = pd.read_csv(csv_path)
    # Convert to a Hugging Face Dataset (preserve_index=False)
    ds = Dataset.from_pandas(df, preserve_index=False)
    # Split with the passed test_size and seed
    return ds.train_test_split(test_size=test_size, seed=seed)


def tokenize_dataset(ds_dict: DatasetDict, tokenizer_name: str, max_length: int) -> DatasetDict:
    """
    Tokenize all splits using the named tokenizer.

    Use truncation=True with the passed max_length. Do not pad here.
    """
    # Load tokenizer with AutoTokenizer.from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Define a tokenize_fn that calls the tokenizer with truncation + max_length
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    
    # Apply ds_dict.map with batched=True
    return ds_dict.map(tokenize_fn, batched=True)


def make_training_args(output_dir: str, lr: float, epochs: int, batch_size: int, seed: int) -> TrainingArguments:
    """Build a TrainingArguments with the standard fine-tuning configuration."""
    # Return a TrainingArguments configured with the passed arguments.
    # In addition to wiring the kwargs through, set:
    #   - eval_strategy="epoch"           (renamed from evaluation_strategy in transformers 4.41+)
    #   - save_strategy="epoch"
    #   - logging_steps=50
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
    )


def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.

    Use sklearn's accuracy_score and f1_score with average="macro".
    """
    # Unpack eval_pred to logits, labels
    logits, labels = eval_pred
    # Argmax logits over axis 1
    predictions = np.argmax(logits, axis=1)
    # Compute accuracy and macro-F1
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    # Return as a dict
    return {"accuracy": accuracy, "macro_f1": f1}


if __name__ == "__main__":
    print("Drill 7A: import this module from tests/test_drill_7a.py to verify your implementations.")
