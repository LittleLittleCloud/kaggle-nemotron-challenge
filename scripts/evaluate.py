"""Local evaluation: load model + LoRA, run inference, compute accuracy."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_train_csv
from src.eval_utils import compute_accuracy
from src.prompts import format_chat_inference


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate(
    config_path: str = "configs/default.yaml",
    adapter_path: str | None = None,
    num_samples: int = 50,
):
    """Run local evaluation on a subset of training data."""
    cfg = load_config(config_path)

    model_name = cfg["model"]["name"]
    if adapter_path is None:
        adapter_path = cfg["training"]["output_dir"]

    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load validation data
    train_df = load_train_csv(cfg["data"]["train_file"])
    eval_df = train_df.sample(n=min(num_samples, len(train_df)), random_state=42)

    inf_cfg = cfg.get("inference", {})
    max_tokens = inf_cfg.get("max_tokens", 7680)

    predictions = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating"):
        messages = format_chat_inference(row["prompt"])
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        predictions.append(response)

    # Score
    results = compute_accuracy(predictions, eval_df["answer"].astype(str).tolist())
    print("\nLocal Evaluation Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Correct:  {results['correct']}/{results['total']}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Local evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--num-samples", type=int, default=50)
    args = parser.parse_args()
    evaluate(args.config, args.adapter_path, args.num_samples)


if __name__ == "__main__":
    main()
