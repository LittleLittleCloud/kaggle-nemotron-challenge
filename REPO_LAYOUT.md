# Repository Layout

```
├── main.py                  # CLI orchestrator
├── configs/
│   └── default.yaml         # Training config
├── src/
│   ├── data.py              # Data loading & formatting
│   ├── eval_utils.py        # Answer extraction & scoring
│   └── prompts.py           # Prompt templates
├── scripts/
│   ├── download_data.py     # Download from Kaggle
│   ├── explore_data.py      # Quick EDA
│   ├── train.py             # LoRA SFT training
│   ├── evaluate.py          # Local evaluation
│   └── package.py           # Package submission.zip
└── notebooks/
    └── kaggle_e2e.ipynb     # End-to-end Kaggle notebook
```
