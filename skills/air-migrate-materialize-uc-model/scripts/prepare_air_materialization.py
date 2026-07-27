#!/usr/bin/env python3
"""Create a self-contained AIR workload for UC model materialization."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_WORKLOAD_DIR = Path("migrate/output/air_materialization")
DEFAULT_LOCAL_STAGING_ROOT = Path("/tmp")
MATERIALIZER_NAME = "materialize_uc_model.py"
SOURCE_CONFIG_NAME = "migration-source.yaml"
WORKLOAD_NAME = "materialize.yaml"
SUPPORTED_ACCELERATORS = {
    "GPU_1xA10": 1,
    "GPU_1xH100": 1,
    "GPU_8xH100": 8,
}


class LiteralString(str):
    """Render a YAML string using block scalar syntax."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to prepare the AIR workload") from exc

    if not path.is_file():
        raise FileNotFoundError(f"Migration config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _required_string(mapping: dict[str, Any], key: str, prefix: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"{prefix}.{key} must be a non-empty string")
    return value


def _read_migration_experiment_path(source: dict[str, Any]) -> str:
    value = _required_string(source, "migration_experiment_path", "source")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
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


def _optional_volume_path(mapping: dict[str, Any], key: str, prefix: str) -> str | None:
    raw_value = mapping.get(key)
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ValueError(f"{prefix}.{key} must be blank or a string")
    value = raw_value.strip()
    if not value:
        return None

    path = PurePosixPath(value)
    parts = path.parts
    if (
        not path.is_absolute()
        or len(parts) < 5
        or parts[1] != "Volumes"
        or any(part in {"", ".", ".."} for part in parts[2:])
    ):
        raise ValueError(
            f"{prefix}.{key} must use "
            "/Volumes/<catalog>/<schema>/<volume>[/<checkpoint-path>]"
        )
    return str(path)


def _read_source(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    source = config.get("source")
    if not isinstance(source, dict):
        raise ValueError("migrate/config.yaml.source must be a mapping")

    catalog = _required_string(source, "catalog", "source")
    schema = _required_string(source, "schema", "source")
    model = _required_string(source, "model", "source")
    version = source.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ValueError("source.version must be a positive integer")
    use_existing_weights = source.get("use_existing_weights")
    if not isinstance(use_existing_weights, bool):
        raise ValueError("source.use_existing_weights must be true or false")
    run_full_migration = source.get("run_full_migration")
    if not isinstance(run_full_migration, bool):
        raise ValueError("source.run_full_migration must be true or false")
    existing_weights_volume_location = _optional_volume_path(
        source, "existing_weights_volume_location", "source"
    )
    if existing_weights_volume_location and not use_existing_weights:
        raise ValueError(
            "source.existing_weights_volume_location requires "
            "source.use_existing_weights=true"
        )

    normalized = {
        "catalog": catalog,
        "schema": schema,
        "model": model,
        "version": version,
        "use_existing_weights": use_existing_weights,
        "run_full_migration": run_full_migration,
        "existing_weights_volume_location": existing_weights_volume_location,
        "migration_experiment_path": _read_migration_experiment_path(source),
    }
    return normalized, f"models:/{catalog}.{schema}.{model}/{version}"


def _read_compute(config: dict[str, Any]) -> dict[str, Any]:
    compute = config.get("compute")
    if not isinstance(compute, dict):
        raise ValueError("migrate/config.yaml.compute must be a mapping")

    num_accelerators = compute.get("num_accelerators")
    if (
        isinstance(num_accelerators, bool)
        or not isinstance(num_accelerators, int)
        or num_accelerators <= 0
    ):
        raise ValueError("compute.num_accelerators must be a positive integer")
    accelerator_type = _required_string(compute, "accelerator_type", "compute")
    expected = SUPPORTED_ACCELERATORS.get(accelerator_type)
    if expected is None:
        supported = ", ".join(sorted(SUPPORTED_ACCELERATORS))
        raise ValueError(
            f"Unsupported compute.accelerator_type={accelerator_type!r}; "
            f"supported values: {supported}"
        )
    if num_accelerators != expected:
        raise ValueError(
            f"compute.accelerator_type={accelerator_type!r} requires "
            f"num_accelerators={expected}, got {num_accelerators}"
        )
    return {
        "num_accelerators": num_accelerators,
        "accelerator_type": accelerator_type,
    }


def _validate_model_uri(model_uri: str) -> str:
    normalized = model_uri.strip()
    match = re.fullmatch(r"models:/([^/]+)/([1-9][0-9]*)", normalized)
    if match is None or len(match.group(1).split(".")) != 3:
        raise ValueError(
            "--model-uri must use models:/<catalog>.<schema>.<model>/<version>"
        )
    return normalized


def _validate_volume_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if len(resolved.parts) < 6 or resolved.parts[1] != "Volumes":
        raise ValueError(
            f"{label} must use /Volumes/<catalog>/<schema>/<volume>/<path>"
        )
    return resolved


def _validate_local_staging_root(path: Path) -> Path:
    value = str(path).strip()
    normalized = PurePosixPath(value)
    if (
        not value
        or not normalized.is_absolute()
        or normalized == PurePosixPath("/")
        or any(part in {"", ".", ".."} for part in normalized.parts[1:])
        or normalized.parts[1] in {"Volumes", "dbfs", "Workspace"}
    ):
        raise ValueError(
            "--local-staging-root must use ephemeral node-local storage outside "
            "/Volumes, /dbfs, and /Workspace"
        )
    return Path(str(normalized))


def _prepare_workload_dir(path: Path) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Workload path is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Workload directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _command(
    *,
    purpose: str,
    output_dir: Path,
    metadata_output: Path,
    local_staging_root: Path,
    use_config_source: bool,
    model_uri: str | None,
    artifact_uri: str | None,
    checkpoint_subpath: str | None,
    tokenizer_subpath: str | None,
) -> str:
    arguments = [
        f'python "$CODE_SOURCE_PATH/{MATERIALIZER_NAME}"',
        f"--purpose {shlex.quote(purpose)}",
    ]
    if use_config_source:
        arguments.append(f'--config "$CODE_SOURCE_PATH/{SOURCE_CONFIG_NAME}"')
    else:
        arguments.append(f"--model-uri {shlex.quote(str(model_uri))}")
    arguments.extend(
        [
            f"--output-dir {shlex.quote(str(output_dir))}",
            f"--metadata-output {shlex.quote(str(metadata_output))}",
            f"--local-staging-root {shlex.quote(str(local_staging_root))}",
            "--require-volume",
        ]
    )
    for flag, value in (
        ("--artifact-uri", artifact_uri),
        ("--checkpoint-subpath", checkpoint_subpath),
        ("--tokenizer-subpath", tokenizer_subpath),
    ):
        if value:
            arguments.append(f"{flag} {shlex.quote(value)}")
    invocation = " \\\n    ".join(arguments)
    return (
        'if [[ "$NODE_RANK" == "0" ]]; then\n'
        f"  {invocation}\n"
        "else\n"
        '  echo "Skipping materialization on AIR node rank $NODE_RANK"\n'
        "fi"
    )


def prepare_air_materialization(
    *,
    config_path: Path,
    output_dir: Path,
    workload_dir: Path = DEFAULT_WORKLOAD_DIR,
    local_staging_root: Path = DEFAULT_LOCAL_STAGING_ROOT,
    purpose: str = "continue_training",
    model_uri: str | None = None,
    artifact_uri: str | None = None,
    checkpoint_subpath: str | None = None,
    tokenizer_subpath: str | None = None,
) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to prepare the AIR workload") from exc

    config_path = config_path.expanduser().resolve()
    config = _load_yaml(config_path)
    source, configured_model_uri = _read_source(config)
    migration_experiment_path = source["migration_experiment_path"]
    compute = _read_compute(config)
    selected_model_uri = (
        _validate_model_uri(model_uri) if model_uri else configured_model_uri
    )

    if (
        source["existing_weights_volume_location"]
        and selected_model_uri == configured_model_uri
    ):
        raise ValueError(
            "UC materialization is not required because "
            "source.existing_weights_volume_location is populated; validate and "
            "load that Volume checkpoint directly"
        )

    if purpose == "continue_training":
        if model_uri is not None:
            raise ValueError(
                "continue_training must resolve the source from migrate/config.yaml; "
                "omit --model-uri"
            )
        if not source["use_existing_weights"]:
            raise ValueError(
                "source.use_existing_weights is false; continued training must not "
                "materialize the configured legacy model"
            )
    elif purpose == "base_model_initialization" and model_uri is None:
        raise ValueError(
            "base_model_initialization requires the pinned --model-uri returned by "
            "resolve_training_source.py"
        )

    output_dir = _validate_volume_path(output_dir, "--output-dir")
    local_staging_root = _validate_local_staging_root(local_staging_root)
    metadata_output = output_dir / "materialization.json"
    workload_dir = _prepare_workload_dir(workload_dir)

    materializer_source = Path(__file__).resolve().with_name(MATERIALIZER_NAME)
    if not materializer_source.is_file():
        raise FileNotFoundError(
            f"Materializer script is missing: {materializer_source}"
        )
    shutil.copy2(materializer_source, workload_dir / MATERIALIZER_NAME)

    use_config_source = model_uri is None
    if use_config_source:
        (workload_dir / SOURCE_CONFIG_NAME).write_text(
            yaml.safe_dump({"source": source}, sort_keys=False),
            encoding="utf-8",
        )

    workload = {
        "experiment_name": migration_experiment_path,
        "environment": {
            "version": "5",
            "dependencies": ["mlflow>=3.6,<4", "pyyaml>=6.0"],
        },
        "compute": compute,
        "env_variables": {"MLFLOW_ENABLE_MULTIPART_DOWNLOAD": "false"},
        "code_source": {
            "type": "snapshot",
            "snapshot": {"root_path": "."},
        },
        "max_retries": 0,
        "command": LiteralString(
            _command(
                purpose=purpose,
                output_dir=output_dir,
                metadata_output=metadata_output,
                local_staging_root=local_staging_root,
                use_config_source=use_config_source,
                model_uri=selected_model_uri,
                artifact_uri=artifact_uri,
                checkpoint_subpath=checkpoint_subpath,
                tokenizer_subpath=tokenizer_subpath,
            )
        ),
    }
    workload_path = workload_dir / WORKLOAD_NAME

    class WorkloadDumper(yaml.SafeDumper):
        pass

    WorkloadDumper.add_representer(
        LiteralString,
        lambda dumper, value: dumper.represent_scalar(
            "tag:yaml.org,2002:str", value, style="|"
        ),
    )
    workload_path.write_text(
        yaml.dump(workload, Dumper=WorkloadDumper, sort_keys=False),
        encoding="utf-8",
    )

    files = [MATERIALIZER_NAME, WORKLOAD_NAME]
    if use_config_source:
        files.append(SOURCE_CONFIG_NAME)
    return {
        "status": "prepared",
        "engine": "databricks_air",
        "profile": "DEFAULT",
        "purpose": purpose,
        "run_full_migration": source["run_full_migration"],
        "migration_experiment_path": migration_experiment_path,
        "source_model_uri": selected_model_uri,
        "output_dir": str(output_dir),
        "metadata_output": str(metadata_output),
        "transfer_strategy": (
            "air_node_local_staging_then_sequential_verified_volume_copy"
        ),
        "local_staging_root": str(local_staging_root),
        "workload_dir": str(workload_dir),
        "workload_file": str(workload_path),
        "files": sorted(files),
        "compute": compute,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("migrate/config.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workload-dir", type=Path, default=DEFAULT_WORKLOAD_DIR)
    parser.add_argument(
        "--local-staging-root",
        type=Path,
        default=DEFAULT_LOCAL_STAGING_ROOT,
        help="Ephemeral AIR node-local directory used before sequential Volume copy",
    )
    parser.add_argument(
        "--purpose",
        choices=("base_model_initialization", "continue_training", "validation"),
        default="continue_training",
    )
    parser.add_argument("--model-uri")
    parser.add_argument("--artifact-uri")
    parser.add_argument("--checkpoint-subpath")
    parser.add_argument("--tokenizer-subpath")
    args = parser.parse_args()

    result = prepare_air_materialization(
        config_path=args.config,
        output_dir=args.output_dir,
        workload_dir=args.workload_dir,
        local_staging_root=args.local_staging_root,
        purpose=args.purpose,
        model_uri=args.model_uri,
        artifact_uri=args.artifact_uri,
        checkpoint_subpath=args.checkpoint_subpath,
        tokenizer_subpath=args.tokenizer_subpath,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
