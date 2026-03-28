"""Prompt templates for the Nemotron reasoning challenge."""

# System prompt that instructs the model to reason step-by-step and box its answer
SYSTEM_PROMPT = (
    "You are a helpful assistant that solves reasoning puzzles. "
    "Think step by step. Put your final answer within \\boxed{}."
)


# Chat template for training: produces the expected output format
def format_chat_train(prompt: str, answer: str) -> list[dict[str, str]]:
    """Format a training example as a chat conversation.

    Returns a list of messages suitable for tokenizer.apply_chat_template.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"The answer is \\boxed{{{answer}}}"},
    ]


def format_chat_inference(prompt: str) -> list[dict[str, str]]:
    """Format a test example for inference (no assistant turn)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
