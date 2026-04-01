"""Data loading and formatting for training."""

from __future__ import annotations

import pandas as pd
from datasets import Dataset

try:
    from src.prompts import METRIC_SUFFIX, format_chat_train
except ModuleNotFoundError:
    from prompts import METRIC_SUFFIX, format_chat_train


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


def classify_type(prompt_text: str) -> str:
    """Classify a prompt into one of 6 question types."""
    p = prompt_text.lower()
    if "bit manipulation" in p or "8-bit binary" in p:
        return "bit_ops"
    elif "encrypt" in p or "decrypt" in p:
        return "cipher"
    elif "gravitational" in p or "falling distance" in p:
        return "gravity"
    elif "numeral system" in p:
        return "numeral"
    elif "transformation rules" in p:
        return "symbol"
    elif "unit conversion" in p or "convert the following measurement" in p:
        return "unit_conv"
    return "unknown"


def stratified_sample(df: pd.DataFrame, n_per_type: int, seed: int) -> pd.DataFrame:
    """Stratified sampling by question type."""
    df = df.copy()
    if "qtype" not in df.columns:
        df["qtype"] = df["prompt"].apply(classify_type)
    return (
        df.groupby("qtype")
        .apply(lambda g: g.sample(n=min(n_per_type, len(g)), random_state=seed))
        .reset_index(drop=True)
    )


def build_sft_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_seq_length: int = 4096,
) -> Dataset:
    """Convert training DataFrame to a HuggingFace Dataset ready for SFTTrainer.

    Each row becomes a 'text' field with the full chat-formatted string.
    Uses METRIC_SUFFIX and enable_thinking for competition format.
    """

    def _format_row(row):
        user_msg = row["prompt"] + METRIC_SUFFIX
        assistant_msg = f'\\boxed{{{row["answer"]}}}'
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        for kwargs in [{"enable_thinking": True}, {}]:
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False, **kwargs
                )
                return {"text": text}
            except Exception:
                continue
        return {"text": f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"}

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(_format_row, remove_columns=dataset.column_names)

    # Filter out sequences that are too long
    def _filter_length(example):
        return len(tokenizer.encode(example["text"])) <= max_seq_length

    dataset = dataset.filter(_filter_length)
    return dataset


def build_grpo_dataset(df: pd.DataFrame) -> Dataset:
    """Build a GRPO dataset with prompts and gold answers.

    Returns a Dataset with 'prompt' (list of message dicts) and 'answer' (str) columns.
    """
    data = []
    for _, row in df.iterrows():
        data.append({
            "prompt": [{"role": "user", "content": row["prompt"] + METRIC_SUFFIX}],
            "answer": str(row["answer"]),
        })
    return Dataset.from_list(data)
