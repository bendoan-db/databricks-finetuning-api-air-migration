#!/usr/bin/env python3
"""Copy one approved AIR template into a new workload directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


RECIPE_DIRECTORIES = {
    "axolotl_qlora": "axolotl_qlora",
    "axolotl_full_fsdp": "axolotl_full_fsdp",
    "axolotl_qlora_fsdp": "axolotl_qlora_fsdp",
}
REQUIRED_FILES = ("train.yaml", "train.py", "01_runner.py")


def _template_root(explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root.expanduser().resolve()

    candidates: list[Path] = []
    for start in (Path.cwd().resolve(), Path(__file__).resolve()):
        candidates.extend(parent / "air_templates" for parent in (start, *start.parents))

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate air_templates; pass --template-root explicitly"
    )


def _validate_source(source: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Template directory does not exist: {source}")

    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Template {source} is missing required files: {', '.join(missing)}"
        )

    unexpected = sorted(
        path.name for path in source.iterdir() if path.name not in REQUIRED_FILES
    )
    if unexpected:
        raise ValueError(
            f"Template {source} contains unexpected files: {', '.join(unexpected)}"
        )


def materialize(
    recipe: str, output_dir: Path, template_root: Path | None = None
) -> dict[str, object]:
    root = _template_root(template_root)
    source = root / RECIPE_DIRECTORIES[recipe]
    destination = output_dir.expanduser().resolve()

    _validate_source(source)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    for name in REQUIRED_FILES:
        shutil.copy2(source / name, destination / name)

    return {
        "recipe": recipe,
        "template_path": str(source),
        "output_path": str(destination),
        "files": list(REQUIRED_FILES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize an approved AIR template without modifying its source"
    )
    parser.add_argument("--recipe", choices=sorted(RECIPE_DIRECTORIES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template-root", type=Path)
    args = parser.parse_args()

    result = materialize(args.recipe, args.output_dir, args.template_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
