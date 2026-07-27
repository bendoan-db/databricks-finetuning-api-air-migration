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
PEFT_RECIPES = frozenset({"trl_lora", "trl_lora_fsdp"})
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
    for obsolete_field in (
        "use_existing_weights",
        "existing_weights_volume_location",
    ):
        if re.search(rf"^\s+{obsolete_field}:\s*", workload_text, re.MULTILINE):
            raise ValueError(
                f"Template {source} still defines obsolete field {obsolete_field}"
            )
    if len(re.findall(r"^\s+HF_TOKEN:\s*\S+", workload_text, re.MULTILINE)) != 1:
        raise ValueError(
            f"Template {source} must define one replaceable HF_TOKEN secret"
        )
    for field in (
        "model_source",
        "requires_hf_token",
        "source_model_uri",
        "model_name",
        "tokenizer_path",
    ):
        matches = re.findall(rf"^\s+{field}:\s*\S+", workload_text, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(
                f"Template {source} must define training_config.{field} exactly once"
            )
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


def _required_string(mapping: dict[str, object], key: str, prefix: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{prefix}.{key} must be a non-empty string")
    return value.strip()


def _optional_string(mapping: dict[str, object], key: str) -> str:
    value = str(mapping.get(key) or "").strip()
    return "" if value.lower() in {"null", "~"} else value


def _required_boolean(mapping: dict[str, object], key: str, prefix: str) -> bool:
    """Parse a required YAML-style true or false scalar."""
    raw_value = mapping.get(key)
    if not isinstance(raw_value, str):
        raise ValueError(f"{prefix}.{key} must be true or false")
    normalized = raw_value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{prefix}.{key} must be true or false")
    return normalized == "true"


def _load_mapping_fields(
    config_path: Path, mapping_name: str, field_names: tuple[str, ...]
) -> dict[str, str]:
    lines = config_path.read_text(encoding="utf-8").splitlines()
    headers = [
        index
        for index, line in enumerate(lines)
        if re.fullmatch(rf"{re.escape(mapping_name)}:\s*(?:#.*)?", line)
    ]
    if len(headers) != 1:
        raise ValueError(
            f"migrate/config.yaml must contain exactly one top-level "
            f"{mapping_name} mapping"
        )

    field_pattern = "|".join(re.escape(name) for name in field_names)
    fields: dict[str, str] = {}
    for line in lines[headers[0] + 1 :]:
        if line and not line[0].isspace() and not line.lstrip().startswith("#"):
            break
        match = re.fullmatch(
            rf"\s+({field_pattern}):\s*([^#]*?)\s*(?:#.*)?", line
        )
        if match is None:
            continue
        name, value = match.groups()
        if name in fields:
            raise ValueError(
                f"migrate/config.yaml.{mapping_name}.{name} is duplicated"
            )
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        fields[name] = value
    return fields


def _volume_path(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or len(path.parts) < 5
        or path.parts[1] != "Volumes"
        or any(part in {"", ".", ".."} for part in path.parts[2:])
    ):
        raise ValueError(
            f"{label} must use "
            "/Volumes/<catalog>/<schema>/<volume>[/<checkpoint-path>]"
        )
    return str(path)


def _system_ai_model_uri(value: str) -> str:
    match = re.fullmatch(r"models:/([^/]+)/([1-9]\d*)", value)
    model_parts = match.group(1).split(".") if match is not None else []
    if (
        len(model_parts) != 3
        or model_parts[:2] != ["system", "ai"]
        or not all(model_parts)
    ):
        raise ValueError(
            "source.system_ai_model_uri must use "
            "models:/system.ai.<model>/<version>"
        )
    return value


def _huggingface_model_id(value: str) -> str:
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?",
        value,
    ):
        raise ValueError(
            "source.huggingface_model_id must be a Hugging Face repository ID "
            "such as meta-llama/Meta-Llama-3.1-8B-Instruct"
        )
    return value


def _hf_secret_reference(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", value):
        raise ValueError(
            "source.huggingface_token_secret must use <secret-scope>/<secret-key>"
        )
    return value


def load_migration_settings(config_path: Path) -> dict[str, object]:
    """Load the fields copied directly into a generated AIR workload."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Migration config does not exist: {config_path}")
    source = _load_mapping_fields(
        config_path,
        "source",
        (
            "catalog",
            "schema",
            "model",
            "version",
            "weights_volume_path",
            "system_ai_model_uri",
            "huggingface_model_id",
            "huggingface_token_secret",
            "migration_experiment_path",
            "peft_only",
        ),
    )
    catalog = _required_string(source, "catalog", "source")
    schema = _required_string(source, "schema", "source")
    model = _required_string(source, "model", "source")
    raw_version = source.get("version", "")
    if not re.fullmatch(r"[1-9]\d*", raw_version):
        raise ValueError("source.version must be a positive integer")
    version = int(raw_version)
    weights_volume_path = _optional_string(source, "weights_volume_path")
    system_ai_model_uri = _optional_string(source, "system_ai_model_uri")
    huggingface_model_id = _optional_string(source, "huggingface_model_id")
    huggingface_token_secret = _optional_string(source, "huggingface_token_secret")
    peft_only = _required_boolean(source, "peft_only", "source")

    if weights_volume_path:
        model_source = "volume"
        model_reference = _volume_path(
            weights_volume_path,
            "source.weights_volume_path",
        )
        selected_hf_secret = None
    elif system_ai_model_uri:
        model_source = "system_ai"
        model_reference = _system_ai_model_uri(system_ai_model_uri)
        selected_hf_secret = None
    else:
        model_source = "hugging_face"
        if not huggingface_model_id:
            raise ValueError(
                "source.huggingface_model_id is required when neither "
                "weights_volume_path nor system_ai_model_uri is populated"
            )
        model_reference = _huggingface_model_id(huggingface_model_id)
        selected_hf_secret = (
            _hf_secret_reference(huggingface_token_secret)
            if huggingface_token_secret
            else None
        )

    target = _load_mapping_fields(
        config_path, "target", ("catalog", "schema", "model", "volume")
    )
    target_values = {
        key: _optional_string(target, key)
        for key in ("catalog", "schema", "model")
    }
    populated = [bool(value) for value in target_values.values()]
    if any(populated) and not all(populated):
        raise ValueError(
            "target.catalog, target.schema, and target.model must be all blank "
            "or all populated"
        )
    if not any(populated):
        target_values = {"catalog": catalog, "schema": schema, "model": model}

    return {
        "source_model_uri": f"models:/{catalog}.{schema}.{model}/{version}",
        "model_source": model_source,
        "model_reference": model_reference,
        "requires_hf_token": selected_hf_secret is not None,
        "hf_token_secret": selected_hf_secret,
        "peft_only": peft_only,
        "registered_model_name": (
            f"{target_values['catalog']}.{target_values['schema']}."
            f"{target_values['model']}"
        ),
        "migration_experiment_path": load_migration_experiment_path(config_path),
        "compute": load_requested_compute(config_path),
    }


def validate_recipe_constraint(recipe: str, peft_only: bool) -> None:
    """Reject a full-weight recipe when configuration requires PEFT."""
    if peft_only and recipe not in PEFT_RECIPES:
        choices = ", ".join(sorted(PEFT_RECIPES))
        raise ValueError(
            "source.peft_only=true requires a PEFT recipe selected from model-size "
            f"evidence ({choices}); received {recipe!r}"
        )


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


def apply_migration_settings(
    workload_path: Path, settings: dict[str, object]
) -> None:
    """Pin the selected model source, lineage, and registration target."""
    text = workload_path.read_text(encoding="utf-8")
    fields = {
        "model_source": str(settings["model_source"]),
        "requires_hf_token": bool(settings["requires_hf_token"]),
        "source_model_uri": str(settings["source_model_uri"]),
        "model_name": str(settings["model_reference"]),
        "tokenizer_path": str(settings["model_reference"]),
        "registered_model_name": str(settings["registered_model_name"]),
    }
    for field, value in fields.items():
        text = _replace_once(
            text,
            rf"^(\s+{field}:\s*).*$",
            rf"\g<1>{json.dumps(value)}",
            f"parameters.training_config.{field}",
        )

    hf_token_secret = settings["hf_token_secret"]
    if hf_token_secret is None:
        text, count = re.subn(
            r"^[ \t]+HF_TOKEN:\s*.*(?:\n|$)",
            "",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError("Could not remove the generated HF_TOKEN secret")
        text = re.sub(
            r"^secrets:\s*\n(?=(?:\s*\n)*(?:\S|$))",
            "",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text = _replace_once(
            text,
            r"^(\s+HF_TOKEN:\s*).*$",
            rf"\g<1>{json.dumps(str(hf_token_secret))}",
            "secrets.HF_TOKEN",
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
    settings = load_migration_settings(migration_config)
    validate_recipe_constraint(recipe, bool(settings["peft_only"]))
    requested_compute = dict(settings["compute"])
    migration_experiment_path = str(settings["migration_experiment_path"])
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
    apply_migration_settings(destination / "train.yaml", settings)

    return {
        "recipe": recipe,
        "template_path": str(source),
        "output_path": str(destination),
        "files": list(required_files),
        "migration_config": str(migration_config),
        "migration_experiment_path": migration_experiment_path,
        "source_model_uri": settings["source_model_uri"],
        "model_source": settings["model_source"],
        "model_reference": settings["model_reference"],
        "requires_hf_token": settings["requires_hf_token"],
        "peft_only": settings["peft_only"],
        "registered_model_name": settings["registered_model_name"],
        "compute": {
            "num_accelerators": requested_compute["num_accelerators"],
            "accelerator_type": requested_compute["accelerator_type"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize an approved AIR template without modifying its source"
    )
    parser.add_argument(
        "--recipe",
        choices=sorted(RECIPE_DIRECTORIES),
        required=True,
        help="Planned recipe; source.peft_only=true permits only LoRA recipes",
    )
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
