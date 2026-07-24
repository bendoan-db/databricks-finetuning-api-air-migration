#!/usr/bin/env python3
"""Copy one approved TRL AIR template into the canonical migration workload."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from pathlib import Path, PurePosixPath


RECIPE_DIRECTORIES = {
    "trl_lora": "trl_lora",
    "trl_lora_fsdp": "trl_lora_fsdp",
    "trl_full_fsdp": "trl_full_fsdp",
}
CORE_FILES = (
    "train.yaml",
    "train.py",
    "helper_utils.py",
    "training_utils.py",
    "01_runner.py",
    "02_register_uc.py",
)
RECIPE_FILES = {
    "trl_lora": (*CORE_FILES, "merge.py"),
    "trl_lora_fsdp": (*CORE_FILES, "merge.py"),
    "trl_full_fsdp": CORE_FILES,
}
DEFAULT_OUTPUT_DIR = Path("migrate/output/air_workload")
IGNORED_LOCAL_NAMES = {".DS_Store", ".ruff_cache", "__pycache__"}
SUPPORTED_ACCELERATORS = {
    "GPU_1xA10": {"num_accelerators": 1, "gpu_type": "A10"},
    "GPU_1xH100": {"num_accelerators": 1, "gpu_type": "H100"},
    "GPU_8xH100": {"num_accelerators": 8, "gpu_type": "H100"},
}


def _template_root(explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root.expanduser().resolve()

    candidates: list[Path] = []
    for start in (Path.cwd().resolve(), Path(__file__).resolve()):
        candidates.extend(
            parent / "air_templates" for parent in (start, *start.parents)
        )

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate air_templates; pass --template-root explicitly"
    )


def _validate_source(
    source: Path, recipe: str, required_files: tuple[str, ...]
) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Template directory does not exist: {source}")

    missing = [name for name in required_files if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Template {source} is missing required files: {', '.join(missing)}"
        )

    unexpected = sorted(
        path.name
        for path in source.iterdir()
        if path.name not in required_files and path.name not in IGNORED_LOCAL_NAMES
    )
    if unexpected:
        raise ValueError(
            f"Template {source} contains unexpected files: {', '.join(unexpected)}"
        )

    workload_text = (source / "train.yaml").read_text(encoding="utf-8")
    for field in ("local_model_cache_dir", "local_model_cache_copy_workers"):
        matches = re.findall(rf"^\s+{field}:\s*\S+", workload_text, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(
                f"Template {source} must define training_config.{field} exactly once"
            )

    training_path = source / "train.py"
    training_text = training_path.read_text(encoding="utf-8")
    training_tree = ast.parse(training_text, filename=str(training_path))
    training_functions = [
        node.name
        for node in training_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if training_functions != ["run_training", "main"]:
        raise ValueError(
            f"Template {source} train.py must contain only run_training and main; "
            f"found: {', '.join(training_functions)}"
        )
    if (
        "from helper_utils import" not in training_text
        or "from training_utils import" not in training_text
    ):
        raise ValueError(
            f"Template {source} train.py must import from helper_utils.py and "
            "training_utils.py"
        )

    helper_text = (source / "helper_utils.py").read_text(encoding="utf-8")
    training_utilities_text = (source / "training_utils.py").read_text(encoding="utf-8")
    required_staging_fragments = (
        "def _stage_model_references(",
        "fcntl.flock(",
        "local_files_only=",
        '"local_model_cache_dir"',
        '"local_model_cache_copy_workers"',
    )
    missing_staging = [
        fragment
        for fragment in required_staging_fragments
        if fragment not in training_utilities_text
    ]
    if missing_staging:
        raise ValueError(
            f"Template {source} is missing node-local staging behavior: "
            + ", ".join(missing_staging)
        )
    if "def load_training_config(" not in helper_text:
        raise ValueError(
            f"Template {source} helper_utils.py is missing load_training_config"
        )
    if "def register_trained_model(" not in training_utilities_text:
        raise ValueError(
            f"Template {source} training_utils.py is missing register_trained_model"
        )
    if recipe in {"trl_lora", "trl_lora_fsdp"}:
        if "def merge_peft_model(" not in training_utilities_text:
            raise ValueError(
                f"Template {source} training_utils.py is missing PEFT merge logic"
            )
        merge_text = (source / "merge.py").read_text(encoding="utf-8")
        if "from training_utils import merge_peft_model" not in merge_text:
            raise ValueError(
                f"Template {source} merge.py must call training_utils.merge_peft_model"
            )


def _load_compute_fields(config_path: Path) -> dict[str, str]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Migration config does not exist: {config_path}")

    lines = config_path.read_text(encoding="utf-8").splitlines()
    headers = [
        index
        for index, line in enumerate(lines)
        if re.fullmatch(r"compute:\s*(?:#.*)?", line)
    ]
    if len(headers) != 1:
        raise ValueError(
            "migrate/config.yaml must contain exactly one top-level block-style "
            "compute mapping"
        )

    fields: dict[str, str] = {}
    for line in lines[headers[0] + 1 :]:
        if line and not line[0].isspace() and not line.lstrip().startswith("#"):
            break
        match = re.fullmatch(
            r"\s+(num_accelerators|accelerator_type):\s*([^#]*?)\s*(?:#.*)?",
            line,
        )
        if match is None:
            continue
        name, raw_value = match.groups()
        if name in fields:
            raise ValueError(f"migrate/config.yaml.compute.{name} is duplicated")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        fields[name] = value
    return fields


def load_migration_experiment_path(config_path: Path) -> str:
    """Read and validate the authoritative workspace experiment path."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Migration config does not exist: {config_path}")

    lines = config_path.read_text(encoding="utf-8").splitlines()
    source_headers = [
        index
        for index, line in enumerate(lines)
        if re.fullmatch(r"source:\s*(?:#.*)?", line)
    ]
    if len(source_headers) != 1:
        raise ValueError(
            "migrate/config.yaml must contain exactly one top-level source mapping"
        )

    matches = []
    for line in lines[source_headers[0] + 1 :]:
        if line and not line[0].isspace() and not line.lstrip().startswith("#"):
            break
        match = re.fullmatch(
            r"\s+migration_experiment_path:\s*([^#]*?)\s*(?:#.*)?", line
        )
        if match is not None:
            matches.append(match.group(1).strip())
    if len(matches) != 1:
        raise ValueError(
            "migrate/config.yaml.source must contain exactly one non-empty "
            "migration_experiment_path"
        )

    value = matches[0]
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    path = PurePosixPath(value)
    if (
        not value
        or not path.is_absolute()
        or path == PurePosixPath("/")
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(
            "source.migration_experiment_path must be an absolute Databricks "
            "workspace path such as /Shared/fmt-migration"
        )
    return value


def load_requested_compute(config_path: Path) -> dict[str, object]:
    """Read and validate the authoritative AIR compute request."""
    compute = _load_compute_fields(config_path)
    raw_num_accelerators = compute.get("num_accelerators", "")
    if not re.fullmatch(r"[1-9]\d*", raw_num_accelerators):
        raise ValueError(
            "migrate/config.yaml.compute.num_accelerators must be a positive integer"
        )
    num_accelerators = int(raw_num_accelerators)

    accelerator_type = compute.get("accelerator_type")
    if accelerator_type is None or not accelerator_type.strip():
        raise ValueError(
            "migrate/config.yaml.compute.accelerator_type must be a non-empty string"
        )
    accelerator_type = accelerator_type.strip()

    specification = SUPPORTED_ACCELERATORS.get(accelerator_type)
    if specification is None:
        supported = ", ".join(sorted(SUPPORTED_ACCELERATORS))
        raise ValueError(
            f"Unsupported AIR accelerator_type {accelerator_type!r}; "
            f"supported values: {supported}"
        )
    expected_count = specification["num_accelerators"]
    if num_accelerators != expected_count:
        raise ValueError(
            f"compute.accelerator_type={accelerator_type!r} requires "
            f"num_accelerators={expected_count}, got {num_accelerators}"
        )

    return {
        "num_accelerators": num_accelerators,
        "accelerator_type": accelerator_type,
        "gpu_type": specification["gpu_type"],
    }


def _replace_once(text: str, pattern: str, replacement: str, field: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Could not uniquely update {field} in generated train.yaml")
    return updated


def apply_requested_compute(
    workload_path: Path, requested_compute: dict[str, object]
) -> None:
    """Keep AIR resources and torchrun world size synchronized."""
    num_accelerators = int(requested_compute["num_accelerators"])
    accelerator_type = str(requested_compute["accelerator_type"])
    text = workload_path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        r"^(\s*num_accelerators:\s*)\d+(\s*(?:#.*)?)$",
        rf"\g<1>{num_accelerators}\g<2>",
        "compute.num_accelerators",
    )
    text = _replace_once(
        text,
        r"^(\s*accelerator_type:\s*)\S+(\s*(?:#.*)?)$",
        rf"\g<1>{accelerator_type}\g<2>",
        "compute.accelerator_type",
    )
    text = _replace_once(
        text,
        r"(--nproc_per_node(?:=|\s+))\d+",
        rf"\g<1>{num_accelerators}",
        "command --nproc_per_node",
    )
    workload_path.write_text(text, encoding="utf-8")

    generated_compute = load_requested_compute(workload_path)
    if (
        generated_compute["num_accelerators"] != num_accelerators
        or generated_compute["accelerator_type"] != accelerator_type
    ):
        raise ValueError("Generated train.yaml does not preserve requested compute")
    if not re.search(
        rf"^command:.*--nproc_per_node(?:=|\s+){num_accelerators}(?:\s|$)",
        text,
        flags=re.MULTILINE,
    ):
        raise ValueError("Generated train.yaml torchrun world size is inconsistent")


def apply_migration_experiment_path(
    workload_path: Path, migration_experiment_path: str
) -> None:
    """Route AIR and application-created MLflow runs to one experiment."""
    yaml_value = "'" + migration_experiment_path.replace("'", "''") + "'"
    text = workload_path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        r"^(experiment_name:\s*).*$",
        rf"\g<1>{yaml_value}",
        "experiment_name",
    )
    text = _replace_once(
        text,
        r"^(\s+experiment_path:\s*).*$",
        rf"\g<1>{yaml_value}",
        "parameters.training_config.experiment_path",
    )
    workload_path.write_text(text, encoding="utf-8")


def materialize(
    recipe: str,
    output_dir: Path | None = None,
    template_root: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, object]:
    root = _template_root(template_root)
    source = root / RECIPE_DIRECTORIES[recipe]
    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else root.parent / DEFAULT_OUTPUT_DIR
    )
    migration_config = (
        config_path.expanduser().resolve()
        if config_path is not None
        else root.parent / "migrate" / "config.yaml"
    )
    requested_compute = load_requested_compute(migration_config)
    migration_experiment_path = load_migration_experiment_path(migration_config)
    required_files = RECIPE_FILES[recipe]

    _validate_source(source, recipe, required_files)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    for name in required_files:
        shutil.copy2(source / name, destination / name)
    apply_requested_compute(destination / "train.yaml", requested_compute)
    apply_migration_experiment_path(
        destination / "train.yaml", migration_experiment_path
    )

    return {
        "recipe": recipe,
        "template_path": str(source),
        "output_path": str(destination),
        "files": list(required_files),
        "migration_config": str(migration_config),
        "migration_experiment_path": migration_experiment_path,
        "compute": {
            "num_accelerators": requested_compute["num_accelerators"],
            "accelerator_type": requested_compute["accelerator_type"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize an approved AIR template without modifying its source"
    )
    parser.add_argument("--recipe", choices=sorted(RECIPE_DIRECTORIES), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination (default: <repository>/migrate/output/air_workload)",
    )
    parser.add_argument("--template-root", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="Migration config (default: <repository>/migrate/config.yaml)",
    )
    args = parser.parse_args()

    result = materialize(args.recipe, args.output_dir, args.template_root, args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
