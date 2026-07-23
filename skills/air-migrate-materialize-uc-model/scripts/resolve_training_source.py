#!/usr/bin/env python3
"""Resolve the checkpoint source selected by migrate/config.yaml."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SYSTEM_AI_PREFIX = "system.ai."
MODEL_ID_TAG_KEYS = {
    "base_model",
    "base_model_id",
    "hugging_face_model_id",
    "hugging_face_repo_id",
    "model_id",
    "source_model",
}


@dataclass(frozen=True)
class MigrationSource:
    model_uri: str
    use_existing_weights: bool


def _required_string(mapping: dict[str, Any], key: str, prefix: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"{prefix}.{key} must be a non-empty string")
    return value


def read_migration_source(config_path: Path) -> MigrationSource:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read migrate/config.yaml") from exc

    with config_path.expanduser().open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("source"), dict):
        raise ValueError(f"Expected a source mapping in {config_path}")

    source = payload["source"]
    catalog = _required_string(source, "catalog", "source")
    schema = _required_string(source, "schema", "source")
    model = _required_string(source, "model", "source")
    version = source.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ValueError("source.version must be a positive integer")
    use_existing_weights = source.get("use_existing_weights")
    if not isinstance(use_existing_weights, bool):
        raise ValueError("source.use_existing_weights must be true or false")

    return MigrationSource(
        model_uri=f"models:/{catalog}.{schema}.{model}/{version}",
        use_existing_weights=use_existing_weights,
    )


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _tags(model: Any) -> dict[str, str]:
    raw = getattr(model, "tags", None)
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        return {
            str(getattr(tag, "key")): str(getattr(tag, "value"))
            for tag in raw
            if getattr(tag, "key", None) is not None
        }
    return {}


def _match_registered_model(model: Any, base_model_id: str) -> tuple[int, str] | None:
    name = str(getattr(model, "name", ""))
    if not name.startswith(SYSTEM_AI_PREFIX):
        return None

    expected = base_model_id.strip().lower()
    for key, value in _tags(model).items():
        if key.lower() in MODEL_ID_TAG_KEYS and value.strip().lower() == expected:
            return 100, f"tag:{key}"

    suffix = name[len(SYSTEM_AI_PREFIX) :]
    repository_name = base_model_id.rsplit("/", 1)[-1]
    if suffix.lower() == repository_name.lower():
        return 90, "exact_name_suffix"
    if _normalize(suffix) == _normalize(repository_name):
        return 80, "normalized_name_suffix"
    if _normalize(suffix) == _normalize(base_model_id):
        return 70, "normalized_full_model_id"
    return None


def _status_text(status: Any) -> str | None:
    if status is None:
        return None
    return str(status).rsplit(".", 1)[-1].upper()


def _latest_ready_version(client: Any, model_name: str) -> int:
    escaped_name = model_name.replace("'", "\\'")
    versions = client.search_model_versions(filter_string=f"name = '{escaped_name}'")
    ready_versions = []
    for version in versions:
        value = getattr(version, "version", None)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        status = _status_text(getattr(version, "status", None))
        if status in {None, "READY"}:
            ready_versions.append(parsed)
    if not ready_versions:
        raise RuntimeError(f"No READY model version exists for {model_name}")
    return max(ready_versions)


def resolve_training_source(
    migration_source: MigrationSource,
    base_model_id: str,
    *,
    registry_uri: str = "databricks-uc",
    mlflow_module: Any | None = None,
) -> dict[str, Any]:
    model_id = base_model_id.strip()
    if not model_id:
        raise ValueError("base_model_id must be a non-empty Hugging Face model ID")

    if migration_source.use_existing_weights:
        return {
            "use_existing_weights": True,
            "model_source": "existing_uc",
            "source_model_uri": migration_source.model_uri,
            "base_model_id": model_id,
            "requires_materialization": True,
            "requires_hf_token": False,
            "match_basis": "migrate.config.source",
        }

    if mlflow_module is None:
        try:
            import mlflow as mlflow_module
        except ImportError as exc:
            raise RuntimeError(
                "MLflow is required to check system.ai before using Hugging Face"
            ) from exc

    mlflow_module.set_registry_uri(registry_uri)
    client = mlflow_module.tracking.MlflowClient()
    try:
        registered_models = client.search_registered_models(
            filter_string="name LIKE 'system.ai.%'"
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to query system.ai; refusing to assume the base model is absent"
        ) from exc

    matches: list[tuple[int, str, str]] = []
    for model in registered_models:
        match = _match_registered_model(model, model_id)
        if match is not None:
            score, basis = match
            matches.append((score, str(model.name), basis))

    if not matches:
        return {
            "use_existing_weights": False,
            "model_source": "hugging_face",
            "source_model_uri": None,
            "base_model_id": model_id,
            "model_name": model_id,
            "tokenizer_path": None,
            "requires_materialization": False,
            "requires_hf_token": True,
            "match_basis": "no_system_ai_match",
        }

    matches.sort(key=lambda item: (-item[0], item[1]))
    best_score = matches[0][0]
    best = [item for item in matches if item[0] == best_score]
    if len(best) != 1:
        names = ", ".join(item[1] for item in best)
        raise ValueError(
            f"Multiple system.ai models match {model_id!r}; resolve explicitly: {names}"
        )

    _, model_name, match_basis = best[0]
    version = _latest_ready_version(client, model_name)
    return {
        "use_existing_weights": False,
        "model_source": "system_ai",
        "source_model_uri": f"models:/{model_name}/{version}",
        "base_model_id": model_id,
        "requires_materialization": True,
        "requires_hf_token": False,
        "match_basis": match_basis,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("migrate/config.yaml"))
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--registry-uri", default="databricks-uc")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    migration_source = read_migration_source(args.config)
    result = resolve_training_source(
        migration_source,
        args.base_model_id,
        registry_uri=args.registry_uri,
    )
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output_path = args.output.expanduser().resolve()
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite source resolution: {output_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
