# Experiment Tracker — NVIDIA Nemotron Reasoning Challenge

## Current Baseline

| Metric | Value |
|---|---|
| Best LB Score | **0.60** |
| Kernel Version | v39 |
| Branch | `exp/baseline-v1` |
| Tag | `sub-v39-lb0.60` |
| Model | Nemotron-3-Nano-30B-A3B-BF16 |
| Method | SFT-only (LoRA rank=32, all-linear) |
| Runtime | ~45 min on RTX PRO 6000 |

## Scoreboard

| Experiment | Branch | Date | LB Score | Tag | Notes |
|---|---|---|---|---|---|
| Baseline SFT-only | `exp/baseline-v1` | 2026-03-28 | **0.60** | `sub-v39-lb0.60` | SFT 600 samples, GRPO disabled (OOM) |

## Competition Strategy

### What's Working
- SFT warmup with LoRA on Nemotron-3-Nano-30B produces reasonable outputs
- Offline package pipeline (nemo wheel + datasets/trl) works on Kaggle
- End-to-end pipeline: SFT → save adapter → generate → package → submit

### Bottlenecks
- **GRPO OOM**: Base model + LoRA + optimizer ≈ 74-76 GB; GRPO generation adds 15-20 GB → exceeds 95 GB
- **SFT data**: Only 600 samples (100/type), likely underfitting
- **No loss logging** in current notebook — hard to diagnose training dynamics

### Planned Experiments

| Priority | Experiment | Hypothesis | Branch |
|---|---|---|---|
| P0 | Enable GRPO | RL alignment improves reasoning; need to solve OOM first | TBD |
| P0 | More SFT data | 600 samples is likely too few; scale to 2k-5k | TBD |
| P1 | Reduce LoRA rank to 16 | Saves ~3.5 GB VRAM, may enable GRPO | TBD |
| P1 | Prompt engineering (CoT) | Better prompts → better base outputs | TBD |
| P2 | Reduce max_completion_length | 256 instead of 512, save VRAM for GRPO | TBD |
| P2 | vLLM generation | More efficient inference, may enable GRPO | TBD |
| P3 | Data filtering/quality | Clean training data → fewer bad patterns | TBD |

### GRPO OOM Mitigation Options
1. Reduce LoRA rank 32→16 (~3.5 GB savings)
2. Reduce `max_completion_length` 512→256
3. Use vLLM for generation (separate memory pool)
4. Gradient checkpointing (if not already enabled)
5. Combine 1+2 for ~5-7 GB total savings

## Submission Log

| # | Date | Kernel Ver | LB Score | Description |
|---|---|---|---|---|
| 1 | 2026-03-24 | v24 | 0.52 | Early baseline |
| 2 | 2026-03-29 | v29 | 0.50 | — |
| 3 | 2026-04-01 | v38 | 0.49 | — |
| 4 | 2026-04-01 | v38 | 0.50 | — |
| 5 | 2026-04-01 | v38 | 0.60 | Best score, accepted as baseline |
| 6 | 2026-04-01 | v39 | pending | Submitted from local |
