"""Data loading and formatting for training."""

from __future__ import annotations

import pandas as pd
from datasets import Dataset

from src.prompts import format_chat_train


def load_train_csv(path: str) -> pd.DataFrame:
    """Load the competition training CSV."""
    df = pd.read_csv(path)
    assert {"id", "prompt", "answer"}.issubset(df.columns), (
        f"Expected columns id, prompt, answer; got {list(df.columns)}"
    )
    return df


def load_test_csv(path: str) -> pd.DataFrame:
    """Load the competition test CSV."""
    df = pd.read_csv(path)
    assert {"id", "prompt"}.issubset(df.columns), (
        f"Expected columns id, prompt; got {list(df.columns)}"
    )
    return df


def build_sft_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_seq_length: int = 4096,
) -> Dataset:
    """Convert training DataFrame to a HuggingFace Dataset ready for SFTTrainer.

    Each row becomes a 'text' field with the full chat-formatted string.
    """

    def _format_row(row):
        messages = format_chat_train(row["prompt"], str(row["answer"]))
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(_format_row, remove_columns=dataset.column_names)

    # Filter out sequences that are too long
    def _filter_length(example):
        return len(tokenizer.encode(example["text"])) <= max_seq_length

    dataset = dataset.filter(_filter_length)
    return dataset
