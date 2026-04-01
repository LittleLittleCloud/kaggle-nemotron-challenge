# Offline Packages

The notebook installs offline wheels from `bigmiao/nemo-offline-packages`:
- `datasets`, `trl` — not pre-installed on Kaggle
- `nemo` — our library (`src/data.py`, `src/eval_utils.py`, `src/prompts.py`)

## Prerequisites

The project needs a `[build-system]` in `pyproject.toml` for `uv build` to work:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

Also need `pip` in the venv for downloading third-party wheels:
```bash
uv pip install pip
```

## Build Steps

```bash
# 1. Edit offline_packages/requirements.txt (third-party only, not nemo)
#    Example contents:
#      datasets==4.8.4
#      trl==1.0.0

# 2. Download third-party wheels for Kaggle (Linux x86_64, Python 3.12)
uv run python -m pip download --no-deps \
    -d offline_packages/ \
    datasets==4.8.4 trl==1.0.0 \
    --python-version 3.12 --platform manylinux2014_x86_64 --only-binary=:all:
# Note: download packages explicitly (not from requirements.txt) to skip nemo

# 3. Rebuild nemo wheel
uv build --wheel --out-dir offline_packages/

# 4. Verify
ls -lh offline_packages/*.whl
# Should see: datasets-*.whl, trl-*.whl, nemo-0.1.0-py3-none-any.whl

# 5. Upload to Kaggle dataset
export KAGGLE_API_TOKEN=<your_token>
uv run kaggle datasets version -p offline_packages/ -m "<description>"
```

## Notes
- `uv pip download` does not exist; use `uv run python -m pip download` instead
- The nemo package cannot be downloaded from PyPI — it's our own wheel, built via `uv build`
- `requirements.txt` lists the nemo entry for reference but pip download should skip it
- Kaggle kernel uses `pip install --no-index --no-deps` to install from this dataset
