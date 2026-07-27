"""Merge a trained LoRA adapter into its unquantized base model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_utils import merge_peft_model


def main() -> None:
    """Merge adapter weights from the command line and print rank-zero output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML file to use instead of HYPERPARAMETERS_PATH/train.yaml.",
    )
    result = merge_peft_model(config_path=parser.parse_args().config)
    if result["rank"] == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
