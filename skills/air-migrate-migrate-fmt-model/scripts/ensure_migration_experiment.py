#!/usr/bin/env python3
"""Ensure the MLflow experiment configured for a migration exists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any


def read_migration_experiment_path(config_path: Path) -> str:
    """Read and validate source.migration_experiment_path."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read migrate/config.yaml") from exc

    resolved_path = config_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Migration config does not exist: {resolved_path}")
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {resolved_path}")

    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"Expected a source mapping in {resolved_path}")
    raw_value = source.get("migration_experiment_path")
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("source.migration_experiment_path must be a non-empty string")
    value = raw_value.strip()
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


def ensure_migration_experiment(
    config_path: Path,
    *,
    tracking_uri: str,
    mlflow_module: Any | None = None,
) -> dict[str, Any]:
    """Return the configured experiment, creating it when it is absent."""
    experiment_path = read_migration_experiment_path(config_path)
    if not tracking_uri.strip():
        raise ValueError("tracking_uri must be a non-empty string")

    if mlflow_module is None:
        try:
            import mlflow as mlflow_module
        except ImportError as exc:
            raise RuntimeError(
                "MLflow is required; install mlflow with Databricks support"
            ) from exc

    mlflow_module.set_tracking_uri(tracking_uri)
    client = mlflow_module.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_path)
    created = False
    if experiment is None:
        try:
            experiment_id = client.create_experiment(experiment_path)
        except Exception:
            # Another orchestrator may have created the same experiment after
            # the first lookup. Preserve genuine permission/service failures.
            experiment = client.get_experiment_by_name(experiment_path)
            if experiment is None:
                raise
        else:
            experiment = client.get_experiment(experiment_id)
            created = True

    lifecycle_stage = str(getattr(experiment, "lifecycle_stage", "active")).lower()
    if lifecycle_stage != "active":
        raise RuntimeError(
            f"Configured MLflow experiment is not active: {experiment_path} "
            f"({lifecycle_stage})"
        )

    return {
        "migration_experiment_path": experiment_path,
        "experiment_id": str(experiment.experiment_id),
        "created": created,
        "tracking_uri": tracking_uri,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("migrate/config.yaml"))
    parser.add_argument(
        "--profile",
        default="DEFAULT",
        help="Databricks CLI profile used when --tracking-uri is omitted.",
    )
    parser.add_argument("--tracking-uri")
    args = parser.parse_args()

    tracking_uri = args.tracking_uri
    if tracking_uri is None:
        profile = str(args.profile).strip()
        if not profile:
            raise ValueError("--profile must be non-empty")
        tracking_uri = f"databricks://{profile}"

    result = ensure_migration_experiment(
        args.config,
        tracking_uri=tracking_uri,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
