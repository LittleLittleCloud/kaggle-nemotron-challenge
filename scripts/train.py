"""LoRA SFT training for Nemotron-3-Nano-30B.

Supports:
  - Standard HuggingFace transformers + PEFT + TRL
  - 4-bit quantized loading for memory efficiency
  - Configurable via YAML config
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import build_sft_dataset, load_train_csv


def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(cfg: dict):
    """Load model with quantization and tokenizer."""
    model_name = cfg["model"]["name"]
    print(f"Loading model: {model_name}")

    # Quantization config for 4-bit
    bnb_config = None
    if cfg["model"].get("load_in_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, cfg["model"].get("torch_dtype", "bfloat16")),
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def setup_lora(model, cfg: dict):
    """Apply LoRA adapter to the model."""
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


def train(config_path: str = "configs/default.yaml"):
    """Run the full training pipeline."""
    cfg = load_config(config_path)

    # 1. Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(cfg)

    # 2. Setup LoRA
    model, peft_config = setup_lora(model, cfg)

    # 3. Load and format data
    train_df = load_train_csv(cfg["data"]["train_file"])
    print(f"Loaded {len(train_df)} training examples")

    # Split validation
    val_split = cfg["data"].get("val_split", 0.05)
    if val_split > 0:
        val_df = train_df.sample(frac=val_split, random_state=42)
        train_df = train_df.drop(val_df.index)
        val_dataset = build_sft_dataset(val_df, tokenizer, cfg["data"]["max_seq_length"])
        print(f"Validation set: {len(val_dataset)} examples")
    else:
        val_dataset = None

    train_dataset = build_sft_dataset(train_df, tokenizer, cfg["data"]["max_seq_length"])
    print(f"Training set: {len(train_dataset)} examples (after filtering)")

    # 4. Training arguments
    tcfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg["num_train_epochs"],
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        learning_rate=tcfg["learning_rate"],
        weight_decay=tcfg.get("weight_decay", 0.01),
        warmup_ratio=tcfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        logging_steps=tcfg.get("logging_steps", 10),
        save_strategy=tcfg.get("save_strategy", "epoch"),
        bf16=tcfg.get("bf16", True),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        optim=tcfg.get("optim", "paged_adamw_8bit"),
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        seed=tcfg.get("seed", 42),
        report_to="none",
        eval_strategy="epoch" if val_dataset else "no",
        remove_unused_columns=False,
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # 6. Train!
    print("Starting training...")
    trainer.train()

    # 7. Save LoRA adapter only
    output_dir = Path(tcfg["output_dir"])
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapter saved to {output_dir}")

    return output_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train LoRA adapter")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
