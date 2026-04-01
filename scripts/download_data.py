"""Download competition data from Kaggle."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

sys_path_hack = str(Path(__file__).resolve().parent.parent)
import sys

if sys_path_hack not in sys.path:
    sys.path.insert(0, sys_path_hack)
from src.utils import load_dotenv

COMPETITION = "nvidia-nemotron-model-reasoning-challenge"


def download_data(data_dir: str = "data") -> Path:
    """Download and extract competition data.

    Loads credentials from .env, then KAGGLE_USERNAME/KAGGLE_KEY env vars,
    or ~/.kaggle/kaggle.json.
    """
    load_dotenv()
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_file = data_path / "train.csv"
    test_file = data_path / "test.csv"

    if train_file.exists() and test_file.exists():
        print(f"Data already exists in {data_path}")
        return data_path

    # Try kagglehub first, fallback to kaggle CLI
    try:
        import kagglehub

        path = kagglehub.competition_download(COMPETITION)
        print(f"Downloaded via kagglehub to: {path}")
        # Copy files to our data dir
        import shutil

        src = Path(path)
        for f in src.glob("*.csv"):
            shutil.copy2(f, data_path / f.name)
        return data_path
    except Exception as e:
        print(f"kagglehub failed ({e}), trying kaggle CLI...")

    # Fallback: kaggle CLI
    zip_path = data_path / f"{COMPETITION}.zip"
    os.system(f'kaggle competitions download -c {COMPETITION} -p "{data_path}"')

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_path)
        zip_path.unlink()
        print(f"Extracted data to {data_path}")
    else:
        raise FileNotFoundError(
            "Download failed. Ensure Kaggle credentials are configured.\n"
            "Set KAGGLE_USERNAME and KAGGLE_KEY env vars, or create ~/.kaggle/kaggle.json"
        )

    return data_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download competition data")
    parser.add_argument("--data-dir", default="data", help="Output directory")
    args = parser.parse_args()
    download_data(args.data_dir)


if __name__ == "__main__":
    main()
