"""Evaluation utilities: answer extraction and scoring.

Mirrors the competition metric logic:
1. Extract answer from \\boxed{}
2. Fallback to last numeric value
3. Compare exact string or within relative tolerance 1e-2
"""

from __future__ import annotations

import re


def extract_boxed_answer(text: str) -> str | None:
    """Extract the content inside the last \\boxed{...} in the text."""
    # Handle nested braces by counting
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # Take the last match
    last_match = matches[-1]
    start = last_match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return None


def extract_last_number(text: str) -> str | None:
    """Extract the last numeric value from text."""
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else None


def extract_answer(text: str) -> str:
    """Extract the final answer using the competition's priority:
    1. Content in \\boxed{}
    2. Last numeric value
    3. Empty string
    """
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed
    num = extract_last_number(text)
    if num is not None:
        return num
    return ""


def is_correct(prediction: str, ground_truth: str, rtol: float = 1e-2) -> bool:
    """Check if prediction matches ground truth.

    Exact string match or numeric match within relative tolerance.
    """
    # Exact string match
    if prediction.strip() == str(ground_truth).strip():
        return True

    # Numeric comparison
    try:
        pred_val = float(prediction.strip())
        gt_val = float(str(ground_truth).strip())
        if gt_val == 0:
            return abs(pred_val) < rtol
        return abs(pred_val - gt_val) / abs(gt_val) < rtol
    except (ValueError, ZeroDivisionError):
        return False


def compute_accuracy(predictions: list[str], ground_truths: list[str]) -> dict[str, float]:
    """Compute accuracy score matching competition metric."""
    assert len(predictions) == len(ground_truths)
    correct = sum(
        is_correct(extract_answer(p), str(gt)) for p, gt in zip(predictions, ground_truths)
    )
    total = len(predictions)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    """GRPO reward function: score completions by correctness.

    Returns:
        +1.0 for correct \\boxed{} answer
        +0.1 for wrong but has \\boxed{}
        -0.5 for no \\boxed{} at all
    """
    rewards = []
    for comp, gold in zip(completions, answer):
        # trl 1.0 passes completions as list of message-dicts, not strings
        if isinstance(comp, list):
            comp = comp[-1]["content"] if comp else ""
        elif isinstance(comp, dict):
            comp = comp.get("content", "")
        pred = extract_boxed_answer(comp)
        if pred is not None and is_correct(pred, str(gold)):
            rewards.append(1.0)
        elif pred is not None:
            rewards.append(0.1)
        else:
            rewards.append(-0.5)
    return rewards
