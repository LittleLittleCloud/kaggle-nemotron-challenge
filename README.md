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

The notebook installs `datasets`, `trl`, and the `nemo` library from offline wheels bundled in
the `bigmiao/nemo-offline-packages` Kaggle dataset.

### Offline Packages Workflow

```bash
# 1. Edit offline_packages/requirements.txt to add/update packages

# 2. Download wheels (no-deps to avoid bundling torch etc.)
pip download --no-deps -r offline_packages/requirements.txt \
    -d offline_packages/ \
    --python-version 3.12 --platform manylinux2014_x86_64 --only-binary=:all:

# 3. Build the nemo library wheel
uv build --wheel --out-dir offline_packages/

# 4. Upload new dataset version to Kaggle
uv run kaggle datasets version -p offline_packages/ -m "<description>"
```

The `kernel-metadata.json` references:
- `competition_sources`: competition train/test CSVs
- `dataset_sources`: `bigmiao/nemo-offline-packages` (offline wheels)
- `model_sources`: `metric/nemotron-3-nano-30b-a3b-bf16`
- `kernel_sources`: `ryanholbrook/nvidia-utility-script` (mamba_ssm, triton)

### Accelerator (machine_shape) Values

| Value | GPU | VRAM | Notes |
|---|---|---|---|
| `NvidiaRtxPro6000` | RTX PRO 6000 Blackwell | 96 GB | Supports bf16 |
| `NvidiaTeslaT4` | Tesla T4 | 16 GB | fp16 only, no bf16 |
| `NvidiaTeslaP100` | Tesla P100 | 16 GB | sm_60, incompatible with latest PyTorch |
| `Tpu1VmV38` | TPU v3-8 | — | TPU |

Push with accelerator:
```bash
uv run kaggle kernels push -p notebooks/ --accelerator NvidiaRtxPro6000     
```

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
