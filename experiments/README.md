# Experiments

Record experiment results here. Use one file per experiment.

## Git Branching Model

```
main                 ← 稳定 baseline，只接受验证过的改进
├── exp/sft-v1       ← 每个实验一个分支
├── exp/prompt-cot
├── exp/rl-grpo
└── exp/data-filter
```

### Workflow

```bash
# 1. 创建实验分支 + 追踪文件（一条命令搞定）
uv run python main.py new-exp sft-lr2e4
# → 创建 exp/sft-lr2e4 分支
# → 创建 experiments/2026-03-28_sft-lr2e4.md
# → 自动 commit

# 2. 做实验，随时 commit（message 写关键参数）
git add -A && git commit -m "sft: lr=2e-4, epoch=3"

# 3. 记录结果到 experiments/，打 tag
git tag exp-sft-lr2e4-acc0.52

# 4. 实验有提升 → merge 回 main
git checkout main
git merge exp/sft-lr2e4

# 5. 实验没用 → 保留分支不删，留作记录
```

### Tag Naming

- `exp-<描述>-acc<分数>` — 实验结果，例: `exp-sft-lr2e4-acc0.52`
- `sub-v<版本>-lb<分数>` — Kaggle 提交，例: `sub-v3-lb0.55`

### Rules

- 每个实验一个分支，代码和 config 修改完整可溯
- 打 tag 记录分数，`git tag` 比文件记录更可靠
- 不删失败分支，避免重复试错
- `main` 只 merge 有提升的实验，保持 baseline 干净

## Naming Convention

`YYYY-MM-DD_<short-description>.md`

## Template

```
# Experiment: <title>
Date: YYYY-MM-DD
Config: configs/xxx.yaml
Branch: exp/xxx
Tag: exp-xxx-acc0.xx

## Changes
- ...

## Results
- Accuracy:
- LB Score:

## Notes
- ...
```
