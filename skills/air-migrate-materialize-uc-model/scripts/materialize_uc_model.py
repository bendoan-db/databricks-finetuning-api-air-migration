#!/usr/bin/env python3
"""Materialize portable Hugging Face weights from a UC registered model."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


MODEL_URI_PATTERN = re.compile(r"^models:/([^/]+)/([1-9][0-9]*)$")
TOKENIZER_ASSETS = {
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "vocab.txt",
}


@dataclass(frozen=True)
class ModelReference:
    name: str
    version: int

    @property
    def uri(self) -> str:
        return f"models:/{self.name}/{self.version}"


@dataclass(frozen=True)
class WeightInventory:
    format: str
    files: tuple[Path, ...]
    index_file: Path | None


def _required_string(mapping: dict[str, Any], key: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"source.{key} must be a non-empty string")
    return value


def reference_from_config(config_path: Path) -> ModelReference:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when using --config") from exc

    with config_path.expanduser().open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("source"), dict):
        raise ValueError(f"Expected a source mapping in {config_path}")

    source = payload["source"]
    catalog = _required_string(source, "catalog")
    schema = _required_string(source, "schema")
    model = _required_string(source, "model")
    version = source.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ValueError("source.version must be a positive integer")
    return ModelReference(f"{catalog}.{schema}.{model}", version)


def use_existing_weights_from_config(config_path: Path) -> bool:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when using --config") from exc

    with config_path.expanduser().open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("source"), dict):
        raise ValueError(f"Expected a source mapping in {config_path}")
    value = payload["source"].get("use_existing_weights")
    if not isinstance(value, bool):
        raise ValueError("source.use_existing_weights must be true or false")
    return value


def existing_weights_volume_from_config(config_path: Path) -> str | None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required when using --config") from exc

    with config_path.expanduser().open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("source"), dict):
        raise ValueError(f"Expected a source mapping in {config_path}")

    raw_value = payload["source"].get("existing_weights_volume_location")
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ValueError(
            "source.existing_weights_volume_location must be blank or a string"
        )
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
            "source.existing_weights_volume_location must use "
            "/Volumes/<catalog>/<schema>/<volume>[/<checkpoint-path>]"
        )
    return str(path)


def reference_from_uri(model_uri: str) -> ModelReference:
    match = MODEL_URI_PATTERN.fullmatch(model_uri.strip())
    if match is None:
        raise ValueError(
            "--model-uri must use models:/<catalog>.<schema>.<model>/<version>"
        )
    name, version_text = match.groups()
    if len(name.split(".")) != 3:
        raise ValueError("The registered model name must contain catalog.schema.model")
    return ModelReference(name, int(version_text))


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_subpath(root: Path, subpath: str, label: str) -> Path:
    candidate = (root / subpath).resolve()
    if not _is_within(candidate, root):
        raise ValueError(f"{label} must remain beneath the downloaded artifact")
    if not candidate.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {candidate}")
    return candidate


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _weight_inventory(model_path: Path) -> WeightInventory:
    indexes = (
        (model_path / "model.safetensors.index.json", "safetensors"),
        (model_path / "pytorch_model.bin.index.json", "pytorch_bin"),
    )
    for index_path, weight_format in indexes:
        if not index_path.is_file():
            continue
        payload = _load_json(index_path, "weight index")
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Weight index has no weight_map: {index_path}")
        names = sorted({str(name) for name in weight_map.values()})
        files: list[Path] = []
        for name in names:
            path = (model_path / name).resolve()
            if not _is_within(path, model_path):
                raise ValueError(f"Weight index references an unsafe path: {name}")
            if not path.is_file():
                raise FileNotFoundError(
                    f"Weight index references a missing shard: {path}"
                )
            files.append(path)
        return WeightInventory(weight_format, tuple(files), index_path)

    safetensors = sorted(
        path
        for path in model_path.glob("model*.safetensors")
        if path.name != "adapter_model.safetensors"
    )
    if safetensors:
        return WeightInventory("safetensors", tuple(safetensors), None)

    pytorch_bins = sorted(model_path.glob("pytorch_model*.bin"))
    if pytorch_bins:
        return WeightInventory("pytorch_bin", tuple(pytorch_bins), None)

    raise FileNotFoundError(f"No full Hugging Face model weights found in {model_path}")


def _validate_model_path(model_path: Path) -> WeightInventory:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Hugging Face config.json in {model_path}")
    config = _load_json(config_path, "model configuration")
    if not config.get("model_type") and not config.get("architectures"):
        raise ValueError(
            f"Model configuration identifies neither model_type nor architectures: {config_path}"
        )
    return _weight_inventory(model_path)


def _has_tokenizer(path: Path) -> bool:
    return (path / "tokenizer_config.json").is_file() and any(
        (path / name).is_file() for name in TOKENIZER_ASSETS
    )


def _model_score(path: Path, root: Path, inventory: WeightInventory) -> int:
    relative_depth = len(path.relative_to(root).parts)
    score = 100 if _has_tokenizer(path) else 0
    score += 20 if inventory.format == "safetensors" else 0
    score += 10 if path.name.lower() in {"model", "checkpoint", "weights"} else 0
    return score - relative_depth


def select_model_path(
    root: Path, checkpoint_subpath: str | None
) -> tuple[Path, WeightInventory]:
    if checkpoint_subpath:
        model_path = _resolve_subpath(root, checkpoint_subpath, "checkpoint subpath")
        return model_path, _validate_model_path(model_path)

    candidates: list[tuple[int, Path, WeightInventory]] = []
    for config_path in root.rglob("config.json"):
        model_path = config_path.parent.resolve()
        try:
            inventory = _validate_model_path(model_path)
        except (FileNotFoundError, ValueError):
            continue
        candidates.append(
            (_model_score(model_path, root, inventory), model_path, inventory)
        )

    if not candidates:
        raise FileNotFoundError(
            "The registered artifact contains no portable full Hugging Face checkpoint"
        )
    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        tied = ", ".join(
            str(item[1].relative_to(root))
            for item in candidates
            if item[0] == candidates[0][0]
        )
        raise ValueError(
            "Multiple checkpoint directories are equally plausible; pass "
            f"--checkpoint-subpath after reviewing them: {tied}"
        )
    _, model_path, inventory = candidates[0]
    return model_path, inventory


def _tokenizer_score(tokenizer_path: Path, model_path: Path) -> int:
    common_parts = 0
    for left, right in zip(tokenizer_path.parts, model_path.parts):
        if left != right:
            break
        common_parts += 1
    distance = len(tokenizer_path.parts) + len(model_path.parts) - 2 * common_parts
    name_bonus = 20 if tokenizer_path.name.lower() in {"tokenizer", "model"} else 0
    return name_bonus - distance


def select_tokenizer_path(
    root: Path, model_path: Path, tokenizer_subpath: str | None
) -> Path:
    if tokenizer_subpath:
        tokenizer_path = _resolve_subpath(root, tokenizer_subpath, "tokenizer subpath")
        if not _has_tokenizer(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer path is incomplete or unsupported: {tokenizer_path}"
            )
        return tokenizer_path

    if _has_tokenizer(model_path):
        return model_path

    candidates = sorted(
        {
            path.parent.resolve()
            for path in root.rglob("tokenizer_config.json")
            if _has_tokenizer(path.parent)
        },
        key=str,
    )
    if not candidates:
        raise FileNotFoundError(
            "The registered artifact contains no complete Hugging Face tokenizer"
        )
    ranked = sorted(
        ((_tokenizer_score(path, model_path), path) for path in candidates),
        key=lambda item: (-item[0], str(item[1])),
    )
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        tied = ", ".join(
            str(item[1].relative_to(root)) for item in ranked if item[0] == ranked[0][0]
        )
        raise ValueError(
            "Multiple tokenizer directories are equally plausible; pass "
            f"--tokenizer-subpath after reviewing them: {tied}"
        )
    return ranked[0][1]


def _prepare_destination(output_dir: Path, reuse_existing: bool) -> Path:
    destination = output_dir.expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {destination}")
    if reuse_existing:
        if not destination.is_dir() or not any(destination.iterdir()):
            raise ValueError("--reuse-existing requires a non-empty output directory")
        return destination
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _validate_volume_path(path: Path) -> None:
    parts = path.parts
    if len(parts) < 5 or parts[1] != "Volumes":
        raise ValueError(
            "--require-volume needs "
            "/Volumes/<catalog>/<schema>/<volume>[/<checkpoint-path>]"
        )


def _import_mlflow() -> Any:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required; install a current mlflow or mlflow-skinny[databricks] package"
        ) from exc
    return mlflow


def _status_text(status: Any) -> str | None:
    if status is None:
        return None
    value = str(status)
    return value.rsplit(".", 1)[-1].upper()


def materialize(
    reference: ModelReference,
    output_dir: Path,
    *,
    purpose: str = "continue_training",
    registry_uri: str = "databricks-uc",
    artifact_uri: str | None = None,
    checkpoint_subpath: str | None = None,
    tokenizer_subpath: str | None = None,
    reuse_existing: bool = False,
    require_volume: bool = False,
    verify_model_version: bool = True,
    mlflow_module: Any | None = None,
) -> dict[str, Any]:
    if purpose not in {
        "base_model_initialization",
        "continue_training",
        "validation",
    }:
        raise ValueError(
            "purpose must be base_model_initialization, continue_training, or validation"
        )
    if not verify_model_version and not reuse_existing:
        raise ValueError(
            "verify_model_version can be false only for existing checkpoint validation"
        )

    mlflow = None
    model_version = None
    status = None
    if verify_model_version:
        mlflow = mlflow_module or _import_mlflow()
        mlflow.set_registry_uri(registry_uri)
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version(
            name=reference.name, version=str(reference.version)
        )
        status = _status_text(getattr(model_version, "status", None))
        if status is not None and status != "READY":
            raise RuntimeError(
                f"Registered model version is not ready: {reference.uri} ({status})"
            )

    destination = _prepare_destination(output_dir, reuse_existing)
    if require_volume:
        _validate_volume_path(destination)

    if reuse_existing:
        downloaded_root = destination
    else:
        resolved_artifact_uri = artifact_uri or reference.uri
        if resolved_artifact_uri != reference.uri and not re.fullmatch(
            r"runs:/[^/]+/.+", resolved_artifact_uri
        ):
            raise ValueError(
                "--artifact-uri must be the selected models:/ URI or a runs:/<run_id>/<path> URI"
            )
        downloaded = mlflow.artifacts.download_artifacts(
            artifact_uri=resolved_artifact_uri, dst_path=str(destination)
        )
        downloaded_root = Path(downloaded).expanduser().resolve()
        if not downloaded_root.is_dir():
            raise FileNotFoundError(
                f"MLflow returned a missing artifact directory: {downloaded_root}"
            )
        if not _is_within(downloaded_root, destination):
            raise RuntimeError(
                "MLflow downloaded outside the requested destination; refusing the result"
            )

    model_path, inventory = select_model_path(downloaded_root, checkpoint_subpath)
    tokenizer_path = select_tokenizer_path(
        downloaded_root, model_path, tokenizer_subpath
    )
    tokenizer_files = sorted(
        path.name
        for path in tokenizer_path.iterdir()
        if path.is_file()
        and (path.name == "tokenizer_config.json" or path.name in TOKENIZER_ASSETS)
    )

    resolved_artifact_uri = artifact_uri or reference.uri
    result = {
        "status": "current",
        "purpose": purpose,
        "registry_uri": registry_uri,
        "source_model_uri": reference.uri,
        "artifact_uri": resolved_artifact_uri if verify_model_version else None,
        "acquisition": "provided_volume" if reuse_existing else "uc_download",
        "download_performed": not reuse_existing,
        "provided_volume_location": str(destination) if reuse_existing else None,
        "model_version_checked": verify_model_version,
        "source_run_id": (
            str(getattr(model_version, "run_id"))
            if getattr(model_version, "run_id", None) is not None
            else None
        ),
        "model_version_source": (
            str(getattr(model_version, "source"))
            if getattr(model_version, "source", None) is not None
            else None
        ),
        "model_version_status": status,
        "destination": str(destination),
        "checkpoint_root": str(downloaded_root),
        "downloaded_artifact_path": str(downloaded_root),
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "weight_format": inventory.format,
        "weight_index": str(inventory.index_file) if inventory.index_file else None,
        "weight_files": [
            {
                "name": str(path.relative_to(model_path)),
                "size_bytes": path.stat().st_size,
            }
            for path in inventory.files
        ],
        "tokenizer_files": tokenizer_files,
        "requires_hf_token": False,
    }
    if purpose == "continue_training":
        result["mode"] = "continue"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="Migration YAML containing source")
    source.add_argument(
        "--model-uri",
        help="Explicit models:/<catalog>.<schema>.<model>/<version> URI",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--purpose",
        choices=("base_model_initialization", "continue_training", "validation"),
        default="continue_training",
    )
    parser.add_argument("--registry-uri", default="databricks-uc")
    parser.add_argument(
        "--artifact-uri",
        help="Inspector-verified models:/ or runs:/ URI containing portable weights",
    )
    parser.add_argument("--checkpoint-subpath")
    parser.add_argument("--tokenizer-subpath")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--require-volume", action="store_true")
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optional new JSON file for the emitted materialization metadata",
    )
    args = parser.parse_args()

    reference = (
        reference_from_uri(args.model_uri)
        if args.model_uri
        else reference_from_config(args.config)
    )
    provided_volume = (
        existing_weights_volume_from_config(args.config) if args.config else None
    )
    if provided_volume:
        if not use_existing_weights_from_config(args.config):
            raise ValueError(
                "source.existing_weights_volume_location requires "
                "source.use_existing_weights=true"
            )
        if not args.reuse_existing:
            raise ValueError(
                "source.existing_weights_volume_location is populated; skip UC "
                "download and validate that checkpoint with --reuse-existing"
            )
        if args.output_dir.expanduser().resolve() != Path(provided_volume).resolve():
            raise ValueError(
                "--output-dir must equal source.existing_weights_volume_location "
                "when validating provided weights"
            )
        if args.artifact_uri:
            raise ValueError(
                "--artifact-uri cannot be used with provided Volume weights because "
                "no artifact download occurs"
            )
        if args.checkpoint_subpath or args.tokenizer_subpath:
            raise ValueError(
                "source.existing_weights_volume_location must point directly to the "
                "complete model and tokenizer checkpoint; do not pass subpaths"
            )
    if (
        args.config
        and args.purpose == "continue_training"
        and not use_existing_weights_from_config(args.config)
    ):
        raise ValueError(
            "source.use_existing_weights is false; do not initialize continued "
            "training from the configured legacy model"
        )
    if args.config and args.purpose == "base_model_initialization":
        raise ValueError(
            "base_model_initialization requires the pinned --model-uri returned by "
            "resolve_training_source.py"
        )
    result = materialize(
        reference,
        args.output_dir,
        purpose=args.purpose,
        registry_uri=args.registry_uri,
        artifact_uri=args.artifact_uri,
        checkpoint_subpath="." if provided_volume else args.checkpoint_subpath,
        tokenizer_subpath="." if provided_volume else args.tokenizer_subpath,
        reuse_existing=args.reuse_existing,
        require_volume=args.require_volume,
        verify_model_version=not bool(provided_volume),
    )
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.metadata_output:
        metadata_path = args.metadata_output.expanduser().resolve()
        if metadata_path.exists():
            raise FileExistsError(f"Metadata output already exists: {metadata_path}")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
