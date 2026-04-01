# Experiment: Baseline SFT-Only Pipeline
Date: 2026-03-28
Branch: exp/baseline-v1
Config: notebooks/kaggle_e2e.ipynb (cell 6 — Configuration)
Tag: sub-v39-lb0.60

## Hypothesis
SFT warmup on Nemotron-3-Nano-30B with LoRA produces a working baseline submission. GRPO disabled due to OOM; saved for future investigation.

## Changes
- Full rewrite of kaggle_e2e.ipynb: SFT → save → sanity check pipeline
- Packaged src/ as nemo wheel in bigmiao/nemo-offline-packages dataset
- Offline pip install with --no-index --no-deps
- Patches: mamba_ssm cutlass try/except, pure PyTorch rmsnorm, triton ptxas fix, is_fast_path_available=False
- correctness_reward handles trl 1.0 list/dict completion format
- PrintLossCallback logs loss/grad_norm/lr to stdout every 10 steps
- GRPO disabled for baseline (OOM at 95 GB with num_gen=2)

## Config
| Parameter | Value |
|---|---|
| Model | Nemotron-3-Nano-30B-A3B-BF16 |
| LoRA rank / alpha | 32 / 16 |
| LoRA target | all-linear |
| SFT samples | 5/type (E2E test, 30 total) |
| SFT max_seq_length | 1024 |
| SFT LR | 2e-4 |
| GRPO | **disabled** (OOM) |
| Batch size / grad accum | 1 / 4 |
| GPU | RTX PRO 6000 (95 GB usable) |

## Results
- v38: SFT-only, 600 samples (100/type), completed in ~43 min
- v39: SFT-only, 600 samples (100/type), completed in ~45 min
- SFT training: 150 steps, ~31 min
- Adapter: 3.4 GB safetensors (LoRA rank=32, all-linear)
- submission.zip: 3006.3 MB
- Loss: not logged (PrintLossCallback not in v38 notebook)
- **LB Score: 0.60 (v39) — baseline accepted**

### Submission History
| Version | Date | LB Score |
|---|---|---|
| v24 | 2026-03-24 | 0.52 |
| v29 | 2026-03-29 | 0.50 |
| v38a | 2026-04-01 03:37 | 0.49 |
| v38b | 2026-04-01 03:43 | 0.50 |
| v38c | 2026-04-01 04:59 | 0.60 |
| v39 | 2026-04-01 05:10 | pending |

## Iteration Log
| Version | Issue | Fix |
|---|---|---|
| v28-29 | Read-only filesystem, pip --ignore-installed fails | copytree to /tmp, --no-deps |
| v30 | No module named 'src' | try/except import fallback in data.py |
| v32 | max_seq_length not in trl 1.0 SFTConfig | Changed to max_length |
| v34 | TypeError: completions are list not str | isinstance checks in correctness_reward |
| v35 | CUDA OOM during GRPO (num_gen=4, max_comp=768) | Reduced to num_gen=2, max_comp=512 |
| v37 | CUDA OOM during GRPO (num_gen=2, max_comp=512, 92.7/95 GB) | Disabled GRPO for baseline |
| v38 | SFT-only baseline with loss logging | Current run |
| v39 | SFT-only baseline, pushed from local | **LB 0.60** — accepted as baseline |

## GRPO OOM Analysis
- v35: num_gen=4, max_comp=768 → OOM (tried to alloc 8 GB, 5.66 GB free)
- v37: num_gen=2, max_comp=512 → OOM (tried to alloc 7 GB, 2.22 GB free, 92.74/95 GB used)
- Base model + LoRA + optimizer ≈ 74-76 GB; GRPO generation adds 15-20 GB
- Future options: reduce LoRA rank to 16 (~3.5 GB savings), reduce max_comp to 256, or use vLLM generation

## Notes
- VS Code buffer edits do NOT persist to disk — must edit notebook JSON directly via Python scripts
- Kaggle dataset updates need ~2 min to propagate; push kernel after delay
- trl 1.0 API differs from older versions: max_length not max_seq_length, completions are list-of-dicts
- GRPO with num_generations=1 gives zero advantage (useless); minimum useful is 2

## What's Working
- SFT warmup with LoRA on Nemotron-3-Nano-30B produces reasonable outputs
- Offline package pipeline (nemo wheel + datasets/trl) works on Kaggle
- End-to-end pipeline: SFT → save adapter → generate → package → submit

## Bottlenecks
- **GRPO OOM**: Base model + LoRA + optimizer ≈ 74-76 GB; GRPO generation adds 15-20 GB → exceeds 95 GB
- **SFT data**: Only 600 samples (100/type), likely underfitting
- **No loss logging** in current notebook — hard to diagnose training dynamics

## Next Experiments

| Priority | Experiment | Hypothesis |
|---|---|---|
| P0 | Enable GRPO | RL alignment improves reasoning; need to solve OOM first |
| P0 | More SFT data | 600 samples is likely too few; scale to 2k-5k |
| P1 | Reduce LoRA rank to 16 | Saves ~3.5 GB VRAM, may enable GRPO |
| P1 | Prompt engineering (CoT) | Better prompts → better base outputs |
| P2 | Reduce max_completion_length | 256 instead of 512, save VRAM for GRPO |
| P2 | vLLM generation | More efficient inference, may enable GRPO |
| P3 | Data filtering/quality | Clean training data → fewer bad patterns |

### GRPO OOM Mitigation Options
1. Reduce LoRA rank 32→16 (~3.5 GB savings)
2. Reduce `max_completion_length` 512→256
3. Use vLLM for generation (separate memory pool)
4. Gradient checkpointing (if not already enabled)
5. Combine 1+2 for ~5-7 GB total savings
