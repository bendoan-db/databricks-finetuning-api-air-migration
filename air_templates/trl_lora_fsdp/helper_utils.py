"""YAML loading, formatting, and configuration validation for LoRA with FSDP full sharding."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "train.yaml"
VOLUME_PREFIX = "/Volumes/"
MODEL_URI_PATTERN = re.compile(r"^models:/([^/]+)/([1-9][0-9]*)$")
SUPPORTED_STRATEGIES = {"no", "steps", "epoch"}
REQUIRED_FSDP_MODES = {"full_shard", "auto_wrap"}


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read and return a YAML mapping from ``path``."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve the training configuration path for CLI and AIR execution."""
    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    hyperparameters_path = os.environ.get("HYPERPARAMETERS_PATH")
    if hyperparameters_path:
        return Path(hyperparameters_path).expanduser().resolve()

    code_source_path = os.environ.get("CODE_SOURCE_PATH")
    if code_source_path:
        candidate = Path(code_source_path) / DEFAULT_CONFIG_PATH.name
        if candidate.is_file():
            return candidate.resolve()

    return DEFAULT_CONFIG_PATH


def _required_string(config: dict[str, Any], key: str) -> str:
    """Return a required nonblank string from the training configuration."""
    value = str(config.get(key, "")).strip()
    if not value:
        raise ValueError(f"training_config.{key} must be a non-empty string")
    return value


def _required_bool(config: dict[str, Any], key: str) -> bool:
    """Return a required Boolean from the training configuration."""
    value = config.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"training_config.{key} must be true or false")
    return value


def _positive_int(config: dict[str, Any], key: str) -> int:
    """Parse and return a strictly positive integer configuration value."""
    value = config.get(key)
    if isinstance(value, bool):
        raise ValueError(f"training_config.{key} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"training_config.{key} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"training_config.{key} must be greater than zero")
    return parsed


def _nonnegative_int(config: dict[str, Any], key: str) -> int:
    """Parse and return a nonnegative integer configuration value."""
    value = config.get(key)
    if isinstance(value, bool):
        raise ValueError(f"training_config.{key} must be a nonnegative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"training_config.{key} must be a nonnegative integer"
        ) from error
    if parsed < 0:
        raise ValueError(f"training_config.{key} must be zero or greater")
    return parsed


def _positive_float(config: dict[str, Any], key: str) -> float:
    """Parse and return a strictly positive floating-point value."""
    try:
        value = float(config.get(key))
    except (TypeError, ValueError) as error:
        raise ValueError(f"training_config.{key} must be a positive number") from error
    if value <= 0:
        raise ValueError(f"training_config.{key} must be greater than zero")
    return value


def _nonnegative_float(config: dict[str, Any], key: str) -> float:
    """Parse and return a nonnegative floating-point value."""
    try:
        value = float(config.get(key))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"training_config.{key} must be a nonnegative number"
        ) from error
    if value < 0:
        raise ValueError(f"training_config.{key} must be zero or greater")
    return value


def _validate_registered_model_name(config: dict[str, Any]) -> str:
    """Validate and return a three-level Unity Catalog model name."""
    name = _required_string(config, "registered_model_name")
    parts = name.split(".")
    if len(parts) != 3 or any(not part.strip() for part in parts):
        raise ValueError(
            "training_config.registered_model_name must use <catalog>.<schema>.<model>"
        )
    return name


def _validate_experiment_path(config: dict[str, Any]) -> str:
    """Validate and normalize the absolute MLflow workspace experiment path."""
    value = _required_string(config, "experiment_path")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path == PurePosixPath("/")
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(
            "training_config.experiment_path must be an absolute Databricks "
            "workspace path"
        )
    config["experiment_path"] = value
    return value


def _is_local_model_reference(value: str) -> bool:
    """Return whether a model reference is an absolute local filesystem path."""
    return Path(value).expanduser().is_absolute()


def _needs_hf_token(config: dict[str, Any]) -> bool:
    """Return whether the selected Hugging Face source requires a token."""
    return (
        config.get("model_source") == "hugging_face"
        and config.get("requires_hf_token") is True
    )


def _validate_model_source(config: dict[str, Any]) -> None:
    """Validate lineage and model references for the selected source mode."""
    source_model_uri = _required_string(config, "source_model_uri")
    match = MODEL_URI_PATTERN.fullmatch(source_model_uri)
    source_parts = match.group(1).split(".") if match is not None else []
    if len(source_parts) != 3 or not all(source_parts):
        raise ValueError(
            "training_config.source_model_uri must use "
            "models:/<catalog>.<schema>.<model>/<version>"
        )
    config["source_model_uri"] = source_model_uri

    model_source = _required_string(config, "model_source")
    if model_source not in {"volume", "system_ai", "hugging_face"}:
        raise ValueError(
            "training_config.model_source must be volume, system_ai, or "
            "hugging_face"
        )
    model_name = _required_string(config, "model_name")
    tokenizer_path = _required_string(config, "tokenizer_path")
    requires_hf_token = _required_bool(config, "requires_hf_token")
    if model_source != "hugging_face" and requires_hf_token:
        raise ValueError(
            "training_config.requires_hf_token may be true only for hugging_face"
        )

    if model_source == "volume":
        for key, value in (
            ("model_name", model_name),
            ("tokenizer_path", tokenizer_path),
        ):
            path = PurePosixPath(value)
            if (
                not path.is_absolute()
                or len(path.parts) < 5
                or path.parts[1] != "Volumes"
                or any(part in {"", ".", ".."} for part in path.parts[2:])
            ):
                raise ValueError(
                    f"training_config.{key} must use "
                    "/Volumes/<catalog>/<schema>/<volume>[/<checkpoint-path>]"
                )
            config[key] = str(path)
    elif model_source == "system_ai":
        system_match = MODEL_URI_PATTERN.fullmatch(model_name)
        system_parts = (
            system_match.group(1).split(".") if system_match is not None else []
        )
        if (
            len(system_parts) != 3
            or system_parts[:2] != ["system", "ai"]
            or not all(system_parts)
        ):
            raise ValueError(
                "model_source=system_ai requires model_name to use "
                "models:/system.ai.<model>/<version>"
            )
        if tokenizer_path != model_name:
            raise ValueError(
                "model_source=system_ai requires tokenizer_path to equal model_name"
            )
    else:
        for key, value in (
            ("model_name", model_name),
            ("tokenizer_path", tokenizer_path),
        ):
            if _is_local_model_reference(value) or value.startswith("models:/"):
                raise ValueError(
                    f"model_source=hugging_face requires {key} to be a remote "
                    "Hugging Face repository ID"
                )
            if any(character.isspace() for character in value):
                raise ValueError(
                    f"training_config.{key} must not contain whitespace"
                )


def _validate_local_model_cache(config: dict[str, Any]) -> None:
    """Validate and normalize ephemeral node-local cache settings."""
    cache_dir = Path(_required_string(config, "local_model_cache_dir")).expanduser()
    if not cache_dir.is_absolute():
        raise ValueError("training_config.local_model_cache_dir must be absolute")
    cache_dir = cache_dir.resolve()
    persistent_roots = (Path("/Volumes"), Path("/dbfs"), Path("/Workspace"))
    if any(cache_dir == root or root in cache_dir.parents for root in persistent_roots):
        raise ValueError(
            "training_config.local_model_cache_dir must use ephemeral node-local "
            "storage, not a Unity Catalog Volume, DBFS, or Workspace path"
        )
    if cache_dir == Path(cache_dir.anchor):
        raise ValueError(
            "training_config.local_model_cache_dir must not be a filesystem root"
        )
    config["local_model_cache_dir"] = str(cache_dir)

    copy_workers = _positive_int(config, "local_model_cache_copy_workers")
    if copy_workers > 32:
        raise ValueError(
            "training_config.local_model_cache_copy_workers must not exceed 32"
        )


def _validate_fsdp(config: dict[str, Any]) -> None:
    """Validate the required full-shard FSDP configuration."""
    fsdp = config.get("fsdp")
    if not isinstance(fsdp, list) or not all(
        isinstance(mode, str) and mode.strip() for mode in fsdp
    ):
        raise ValueError("training_config.fsdp must be a non-empty list of strings")
    if set(fsdp) != REQUIRED_FSDP_MODES:
        raise ValueError(
            "training_config.fsdp must contain exactly full_shard and auto_wrap"
        )

    fsdp_config = config.get("fsdp_config")
    if not isinstance(fsdp_config, dict):
        raise ValueError("training_config.fsdp_config must be a mapping")

    expected_strings = {
        "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        "backward_prefetch": "backward_pre",
        "state_dict_type": "FULL_STATE_DICT",
    }
    for key, expected in expected_strings.items():
        value = str(fsdp_config.get(key, "")).strip()
        if value != expected:
            raise ValueError(
                f"training_config.fsdp_config.{key} must be {expected!r}; got {value!r}"
            )

    for key in (
        "forward_prefetch",
        "limit_all_gathers",
        "use_orig_params",
        "sync_module_states",
        "cpu_ram_efficient_loading",
        "activation_checkpointing",
    ):
        if not isinstance(fsdp_config.get(key), bool):
            raise ValueError(f"training_config.fsdp_config.{key} must be true or false")

    if fsdp_config["forward_prefetch"]:
        raise ValueError("training_config.fsdp_config.forward_prefetch must be false")
    if not fsdp_config["limit_all_gathers"]:
        raise ValueError("training_config.fsdp_config.limit_all_gathers must be true")
    if fsdp_config["use_orig_params"]:
        raise ValueError("training_config.fsdp_config.use_orig_params must be false")
    if not fsdp_config["sync_module_states"]:
        raise ValueError("training_config.fsdp_config.sync_module_states must be true")
    if not fsdp_config["cpu_ram_efficient_loading"]:
        raise ValueError(
            "training_config.fsdp_config.cpu_ram_efficient_loading must be true"
        )
    if not fsdp_config["activation_checkpointing"]:
        raise ValueError(
            "training_config.fsdp_config.activation_checkpointing must be true"
        )


def load_training_config(
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Load and validate this template's training configuration."""
    resolved_path = _resolve_config_path(config_path)
    payload = _read_yaml(resolved_path)

    # AIR may write either the full workload mapping or only its parameters.
    parameters = payload.get("parameters", payload)
    if not isinstance(parameters, dict):
        raise ValueError(f"Expected parameters to be a mapping in {resolved_path}")
    config = parameters.get("training_config", parameters)
    if not isinstance(config, dict):
        raise ValueError(
            f"Expected parameters.training_config to be a mapping in {resolved_path}"
        )

    for key in (
        "model_name",
        "tokenizer_path",
        "model_source",
        "source_model_uri",
        "local_model_cache_dir",
        "experiment_path",
        "mlflow_run_name",
        "registered_model_name",
        "train_data_path",
        "eval_data_path",
        "output_dir",
        "merged_output_dir",
        "optimizer",
        "lr_scheduler",
        "eval_strategy",
        "save_strategy",
        "attn_implementation",
    ):
        _required_string(config, key)

    experiment_path = _validate_experiment_path(config)
    workload_experiment_name = payload.get("experiment_name")
    if (
        workload_experiment_name is not None
        and str(workload_experiment_name).strip() != experiment_path
    ):
        raise ValueError(
            "experiment_name and training_config.experiment_path must match"
        )
    _validate_registered_model_name(config)
    _validate_model_source(config)
    _validate_local_model_cache(config)

    for key in (
        "sequence_len",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "num_epochs",
        "logging_steps",
        "save_total_limit",
        "lora_r",
        "lora_alpha",
        "dataset_num_proc",
    ):
        _positive_int(config, key)
    _nonnegative_int(config, "dataloader_num_workers")
    _nonnegative_int(config, "seed")

    max_steps = config.get("max_steps")
    if max_steps is not None:
        _positive_int(config, "max_steps")
    for key in ("learning_rate", "max_grad_norm"):
        _positive_float(config, key)
    for key in ("warmup_ratio", "weight_decay", "lora_dropout"):
        _nonnegative_float(config, key)
    if float(config["warmup_ratio"]) > 1:
        raise ValueError("training_config.warmup_ratio must not exceed 1")
    if float(config["lora_dropout"]) >= 1:
        raise ValueError("training_config.lora_dropout must be less than 1")

    for key in (
        "packing",
        "assistant_only_loss",
        "gradient_checkpointing",
        "bf16",
        "tf32",
        "save_safetensors",
    ):
        _required_bool(config, key)
    if not config["assistant_only_loss"]:
        raise ValueError(
            "This template requires training_config.assistant_only_loss=true"
        )
    if not config["bf16"]:
        raise ValueError("Plain LoRA FSDP training requires training_config.bf16=true")
    if config["gradient_checkpointing"]:
        raise ValueError(
            "training_config.gradient_checkpointing must be false; use FSDP "
            "activation_checkpointing to avoid redundant all-gathers"
        )
    if not config["save_safetensors"]:
        raise ValueError("training_config.save_safetensors must be true")

    for key in ("eval_strategy", "save_strategy"):
        if config[key] not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"training_config.{key} must be one of {sorted(SUPPORTED_STRATEGIES)}"
            )
    if config["eval_strategy"] == "no":
        raise ValueError("This template requires evaluation during training")
    if config["attn_implementation"] not in {"eager", "sdpa", "flash_attention_2"}:
        raise ValueError(
            "training_config.attn_implementation must be eager, sdpa, or "
            "flash_attention_2"
        )

    train_path = _required_string(config, "train_data_path")
    eval_path = _required_string(config, "eval_data_path")
    output_dir = _required_string(config, "output_dir")
    merged_output_dir = _required_string(config, "merged_output_dir")
    for key, value in (
        ("train_data_path", train_path),
        ("eval_data_path", eval_path),
        ("output_dir", output_dir),
        ("merged_output_dir", merged_output_dir),
    ):
        if not value.startswith(VOLUME_PREFIX):
            raise ValueError(
                f"training_config.{key} must use a Unity Catalog Volume path "
                f"starting with {VOLUME_PREFIX!r}; got {value!r}"
            )
    if train_path == eval_path:
        raise ValueError("train_data_path and eval_data_path must be different")
    if Path(output_dir).resolve() == Path(merged_output_dir).resolve():
        raise ValueError("output_dir and merged_output_dir must be different")

    resume_from_checkpoint = config.get("resume_from_checkpoint")
    if resume_from_checkpoint is not None:
        resume_from_checkpoint = str(resume_from_checkpoint).strip()
        if not resume_from_checkpoint:
            raise ValueError(
                "training_config.resume_from_checkpoint must be null or non-empty"
            )
        config["resume_from_checkpoint"] = resume_from_checkpoint

    target_modules = config.get("lora_target_modules")
    if not isinstance(target_modules, list) or not target_modules:
        raise ValueError("training_config.lora_target_modules must be a non-empty list")
    if not all(isinstance(module, str) and module.strip() for module in target_modules):
        raise ValueError("Every lora_target_modules value must be a non-empty string")

    _validate_fsdp(config)

    return dict(config), resolved_path
