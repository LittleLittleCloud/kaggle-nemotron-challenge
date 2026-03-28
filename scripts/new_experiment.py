"""Create a new experiment branch and its corresponding tracking file."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

TEMPLATE = """\
# Experiment: {title}
Date: {date}
Branch: exp/{name}
Config: configs/default.yaml
Tag:

## Hypothesis
-

## Changes
-

## Results
- Accuracy:
- LB Score:

## Notes
-
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create a new experiment branch + tracking file")
    parser.add_argument("name", help="Experiment name, e.g. sft-lr2e4 (branch will be exp/<name>)")
    parser.add_argument("--title", default=None, help="Human-readable title (defaults to name)")
    parser.add_argument(
        "--no-checkout", action="store_true", help="Only create file, don't switch branch"
    )
    args = parser.parse_args()

    name = args.name.removeprefix("exp/")
    branch = f"exp/{name}"
    title = args.title or name
    date = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date}_{name}.md"
    filepath = EXPERIMENTS_DIR / filename

    if filepath.exists():
        print(f"Error: {filepath} already exists")
        sys.exit(1)

    # Create and checkout branch
    if not args.no_checkout:
        result = subprocess.run(["git", "checkout", "-b", branch], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error creating branch: {result.stderr.strip()}")
            sys.exit(1)
        print(f"Created branch: {branch}")

    # Create experiment file
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath.write_text(TEMPLATE.format(title=title, date=date, name=name))
    print(f"Created: {filepath.relative_to(Path.cwd())}")

    # Auto-commit the experiment file
    if not args.no_checkout:
        subprocess.run(["git", "add", str(filepath)], check=True)
        subprocess.run(["git", "commit", "-m", f"exp: start {name}"], check=True)
        print(f"\nReady! Start experimenting on branch '{branch}'")


if __name__ == "__main__":
    main()
