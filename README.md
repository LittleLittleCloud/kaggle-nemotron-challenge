# NVIDIA Nemotron Reasoning Challenge

LoRA fine-tuning pipeline for the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).

## Competition Summary

- **Task**: Improve reasoning accuracy on logic puzzles (bit manipulation, algebraic equations)
- **Model**: Nemotron-3-Nano-30B-A3B-BF16 (base, frozen)
- **Submission**: LoRA adapter (rank ≤ 32) as `submission.zip`
- **Metric**: Accuracy — answer extracted from `\boxed{}`, exact string or numeric match (rtol=1e-2)
- **Inference**: vLLM, temperature=0, max_tokens=7680, max_model_len=8192

See [REPO_LAYOUT.md](REPO_LAYOUT.md) for the full project structure.

## Quick Start

```bash
# 1. Download data (requires Kaggle credentials)
python main.py download

# 2. Explore data
python main.py explore

# 3. Train LoRA adapter (requires GPU)
python main.py train

# 4. Local evaluation
python main.py eval --num-samples 20

# 5. Package submission
python main.py package

# 6. Submit to Kaggle
python main.py submit --message "baseline v1"

# Or run everything:
python main.py all
```

## Kaggle Notebook

For running on Kaggle with GPU, use `notebooks/kaggle_e2e.ipynb`.

## Key Parameters (Competition Metric)

| Parameter | Value |
|---|---|
| max_lora_rank | 32 |
| max_tokens | 7680 |
| top_p | 1.0 |
| temperature | 0.0 |
| max_num_seqs | 64 |
| gpu_memory_utilization | 0.85 |
| max_model_len | 8192 |
