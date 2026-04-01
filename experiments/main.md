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

## Submission Log

| # | Date | Kernel Ver | LB Score | Description |
|---|---|---|---|---|
| 1 | 2026-03-24 | v24 | 0.52 | Early baseline |
| 2 | 2026-03-29 | v29 | 0.50 | — |
| 3 | 2026-04-01 | v38 | 0.49 | — |
| 4 | 2026-04-01 | v38 | 0.50 | — |
| 5 | 2026-04-01 | v38 | 0.60 | Best score, accepted as baseline |
| 6 | 2026-04-01 | v39 | pending | Submitted from local |
