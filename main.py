"""
NVIDIA Nemotron Reasoning Challenge - Pipeline Orchestrator

Usage:
    python main.py download     # Download competition data
    python main.py explore      # Quick EDA
    python main.py train        # Train LoRA adapter
    python main.py eval         # Local evaluation
    python main.py package      # Package submission.zip
    python main.py submit       # Submit to Kaggle
    python main.py all          # Run full pipeline (download → train → package)
    python main.py new-exp NAME  # Create experiment branch + tracking file
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def cmd_download(args):
    from scripts.download_data import download_data

    download_data(args.data_dir)


def cmd_explore(args):
    from scripts.explore_data import explore

    explore(args.data_dir)


def cmd_train(args):
    from scripts.train import train

    train(args.config)


def cmd_eval(args):
    from scripts.evaluate import evaluate

    evaluate(args.config, args.adapter_path, args.num_samples)


def cmd_package(args):
    from scripts.package import package_submission

    package_submission(args.adapter_dir, args.output)


def cmd_submit(args):
    """Submit to Kaggle via CLI."""
    submission = args.submission
    if not Path(submission).exists():
        print(f"Submission file not found: {submission}")
        sys.exit(1)
    competition = "nvidia-nemotron-model-reasoning-challenge"
    message = args.message or "Automated submission"
    cmd = f'kaggle competitions submit -c {competition} -f "{submission}" -m "{message}"'
    print(f"Running: {cmd}")
    os.system(cmd)


def cmd_new_exp(args):
    """Create a new experiment branch and tracking file."""
    from scripts.new_experiment import main as new_exp_main

    sys.argv = ["new_experiment", args.name]
    if args.title:
        sys.argv += ["--title", args.title]
    if args.no_checkout:
        sys.argv.append("--no-checkout")
    new_exp_main()


def cmd_all(args):
    """Full pipeline: download → train → package."""
    print("=" * 60)
    print("STEP 1/3: Download Data")
    print("=" * 60)
    cmd_download(args)

    print("\n" + "=" * 60)
    print("STEP 2/3: Train LoRA Adapter")
    print("=" * 60)
    cmd_train(args)

    print("\n" + "=" * 60)
    print("STEP 3/3: Package Submission")
    print("=" * 60)
    cmd_package(args)
    print("\nDone! Submit with: python main.py submit")


def main():
    parser = argparse.ArgumentParser(description="Nemotron Reasoning Challenge Pipeline")
    sub = parser.add_subparsers(dest="command", help="Pipeline step to run")

    # download
    p = sub.add_parser("download", help="Download competition data")
    p.add_argument("--data-dir", default="data")

    # explore
    p = sub.add_parser("explore", help="Explore data")
    p.add_argument("--data-dir", default="data")

    # train
    p = sub.add_parser("train", help="Train LoRA adapter")
    p.add_argument("--config", default="configs/default.yaml")

    # eval
    p = sub.add_parser("eval", help="Local evaluation")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--num-samples", type=int, default=50)

    # package
    p = sub.add_parser("package", help="Package submission.zip")
    p.add_argument("--adapter-dir", default="outputs/lora_adapter")
    p.add_argument("--output", default="outputs/submission.zip")

    # submit
    p = sub.add_parser("submit", help="Submit to Kaggle")
    p.add_argument("--submission", default="outputs/submission.zip")
    p.add_argument("--message", default=None)

    # new-exp
    p = sub.add_parser("new-exp", help="Create experiment branch + tracking file")
    p.add_argument("name", help="Experiment name, e.g. sft-lr2e4")
    p.add_argument("--title", default=None, help="Human-readable title")
    p.add_argument("--no-checkout", action="store_true", help="Only create file")

    # all
    p = sub.add_parser("all", help="Full pipeline: download → train → package")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--adapter-dir", default="outputs/lora_adapter")
    p.add_argument("--output", default="outputs/submission.zip")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "download": cmd_download,
        "explore": cmd_explore,
        "train": cmd_train,
        "eval": cmd_eval,
        "package": cmd_package,
        "submit": cmd_submit,
        "new-exp": cmd_new_exp,
        "all": cmd_all,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
