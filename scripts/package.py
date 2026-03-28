"""Package LoRA adapter into submission.zip for Kaggle."""

from __future__ import annotations

import zipfile
from pathlib import Path

# Files required in a LoRA adapter submission
REQUIRED_FILES = ["adapter_config.json", "adapter_model.safetensors"]
OPTIONAL_FILES = ["adapter_model.bin"]


def package_submission(
    adapter_dir: str = "outputs/lora_adapter",
    output_path: str = "outputs/submission.zip",
):
    """Create submission.zip from LoRA adapter directory."""
    adapter_path = Path(adapter_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Verify adapter_config.json exists (required by competition)
    config_file = adapter_path / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found in {adapter_path}. "
            "Training may not have completed successfully."
        )

    # Find model weights
    safetensors = adapter_path / "adapter_model.safetensors"
    bin_file = adapter_path / "adapter_model.bin"
    if not safetensors.exists() and not bin_file.exists():
        raise FileNotFoundError(
            f"No adapter weights found in {adapter_path}. "
            "Expected adapter_model.safetensors or adapter_model.bin"
        )

    # Create zip
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        # Always include config
        zf.write(config_file, "adapter_config.json")
        print("  Added: adapter_config.json")

        # Include weights
        if safetensors.exists():
            zf.write(safetensors, "adapter_model.safetensors")
            print("  Added: adapter_model.safetensors")
        elif bin_file.exists():
            zf.write(bin_file, "adapter_model.bin")
            print("  Added: adapter_model.bin")

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"\nSubmission packaged: {output} ({size_mb:.1f} MB)")
    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Package submission")
    parser.add_argument("--adapter-dir", default="outputs/lora_adapter")
    parser.add_argument("--output", default="outputs/submission.zip")
    args = parser.parse_args()
    package_submission(args.adapter_dir, args.output)


if __name__ == "__main__":
    main()
