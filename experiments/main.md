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

| Experiment | Branch | LB Score | Tag | Notes |
|---|---|---|---|---|
| Baseline SFT-only | `exp/baseline-v1` | **0.60** | `sub-v39-lb0.60` | SFT 600 samples, GRPO disabled (OOM) |

