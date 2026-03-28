"""Quick EDA on competition data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def explore(data_dir: str = "data"):
    data_path = Path(data_dir)
    train_file = data_path / "train.csv"
    test_file = data_path / "test.csv"

    if not train_file.exists():
        print("No data found. Run: python -m scripts.download_data")
        return

    # --- Train ---
    train = pd.read_csv(train_file)
    print("=" * 60)
    print("TRAIN SET")
    print("=" * 60)
    print(f"Shape: {train.shape}")
    print(f"Columns: {list(train.columns)}")
    print("\nAnswer value counts (top 20):")
    print(train["answer"].value_counts().head(20))
    print("\nPrompt length stats (chars):")
    print(train["prompt"].str.len().describe())
    print("\nAnswer length stats (chars):")
    print(train["answer"].astype(str).str.len().describe())
    print("\nSample prompt (first row):")
    print(train["prompt"].iloc[0][:500])
    print("...")
    print(f"\nSample answer: {train['answer'].iloc[0]}")

    # --- Test ---
    if test_file.exists():
        test = pd.read_csv(test_file)
        print("\n" + "=" * 60)
        print("TEST SET")
        print("=" * 60)
        print(f"Shape: {test.shape}")
        print(f"Columns: {list(test.columns)}")
        print("\nPrompt length stats (chars):")
        print(test["prompt"].str.len().describe())


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Explore competition data")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    explore(args.data_dir)


if __name__ == "__main__":
    main()
