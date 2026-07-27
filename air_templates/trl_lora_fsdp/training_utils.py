"""Distributed training, artifact, merge, and registration helpers for LoRA with FSDP full sharding."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from helper_utils import (
    _is_local_model_reference,
    _needs_hf_token,
    _validate_registered_model_name,
    load_training_config,
)


LOCAL_CACHE_SCHEMA_VERSION = 1


def distributed_context() -> tuple[int, int, int]:
    """Return global rank, world size, and local rank for either launcher."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    try:
        from serverless_gpu import runtime

        return runtime.get_global_rank(), runtime.get_world_size(), local_rank
    except Exception:
        return (
            int(os.environ.get("RANK", "0")),
            int(os.environ.get("WORLD_SIZE", "1")),
            local_rank,
        )


def _directory_inventory(source: Path) -> list[tuple[Path, Path, int]]:
    """List model files with paths relative to the source and byte sizes."""
    inventory: list[tuple[Path, Path, int]] = []
    for current_root, directory_names, file_names in os.walk(source):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current_root)
        for file_name in file_names:
            source_file = current_path / file_name
            relative_path = source_file.relative_to(source)
            inventory.append((source_file, relative_path, source_file.stat().st_size))
    if not inventory:
        raise FileNotFoundError(f"Model directory contains no files: {source}")
    return inventory


def _read_local_cache_marker(
    marker_path: Path, source_reference: str
) -> dict[str, Any]:
    """Load and validate a completed node-local model cache marker."""
    try:
        with marker_path.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Invalid local model cache marker: {marker_path}"
        ) from error
    if not isinstance(marker, dict):
        raise RuntimeError(f"Invalid local model cache marker: {marker_path}")
    if marker.get("schema_version") != LOCAL_CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported local model cache marker: {marker_path}")
    if marker.get("source_path") != source_reference:
        raise RuntimeError(
            f"Local model cache source mismatch in {marker_path}: "
            f"{marker.get('source_path')!r} != {source_reference!r}"
        )
    return marker


def _stage_directory_to_local_cache(
    source_reference: str,
    cache_root_reference: str,
    copy_workers: int,
) -> tuple[str, dict[str, Any]]:
    """Copy a model directory into a process-safe node-local cache."""
    source = Path(source_reference).expanduser()
    cache_root = Path(cache_root_reference).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    cache_key = hashlib.sha256(
        f"{LOCAL_CACHE_SCHEMA_VERSION}\0{source}".encode("utf-8")
    ).hexdigest()[:24]
    destination = cache_root / cache_key
    marker_path = destination / ".air-local-cache.json"
    lock_path = cache_root / f".{cache_key}.lock"
    operation_started = time.monotonic()
    copied_by_process = False

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        lock_wait_seconds = time.monotonic() - operation_started
        if marker_path.is_file():
            marker = _read_local_cache_marker(marker_path, str(source))
        else:
            if destination.exists():
                raise RuntimeError(
                    "Local model cache destination exists without a completion "
                    f"marker: {destination}"
                )

            copy_started = time.monotonic()
            inventory = _directory_inventory(source)
            total_bytes = sum(size for _, _, size in inventory)
            free_bytes = shutil.disk_usage(cache_root).free
            reserve_bytes = max(1 << 30, total_bytes // 10)
            if free_bytes < total_bytes + reserve_bytes:
                raise OSError(
                    f"Insufficient node-local space under {cache_root}: need at "
                    f"least {total_bytes + reserve_bytes:,} bytes, found "
                    f"{free_bytes:,}"
                )

            partial = cache_root / f".{cache_key}.partial-{os.getpid()}"
            if partial.exists():
                raise RuntimeError(f"Stale local model cache staging path: {partial}")
            partial.mkdir(parents=False)

            def copy_file(item: tuple[Path, Path, int]) -> None:
                """Copy one inventoried file and verify its resulting size."""
                source_file, relative_path, expected_size = item
                destination_file = partial / relative_path
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, destination_file)
                observed_size = destination_file.stat().st_size
                if observed_size != expected_size:
                    raise OSError(
                        f"Local cache copy size mismatch for {relative_path}: "
                        f"expected {expected_size}, got {observed_size}"
                    )

            try:
                with ThreadPoolExecutor(max_workers=copy_workers) as executor:
                    list(executor.map(copy_file, inventory))
                copy_duration_seconds = time.monotonic() - copy_started
                marker = {
                    "schema_version": LOCAL_CACHE_SCHEMA_VERSION,
                    "source_path": str(source),
                    "file_count": len(inventory),
                    "total_bytes": total_bytes,
                    "copy_workers": copy_workers,
                    "copy_duration_seconds": copy_duration_seconds,
                }
                with (partial / marker_path.name).open("w", encoding="utf-8") as handle:
                    json.dump(marker, handle, indent=2, sort_keys=True)
                os.replace(partial, destination)
                copied_by_process = True
            except Exception:
                if partial.parent == cache_root and partial.name.startswith(
                    f".{cache_key}.partial-"
                ):
                    shutil.rmtree(partial, ignore_errors=True)
                raise
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    copy_duration_seconds = float(marker["copy_duration_seconds"])
    total_bytes = int(marker["total_bytes"])
    throughput_mib_per_second = (
        total_bytes / max(copy_duration_seconds, 1.0e-9) / (1024 * 1024)
    )
    return str(destination), {
        **marker,
        "destination": str(destination),
        "staged": True,
        "cache_hit": not copied_by_process,
        "lock_wait_seconds": lock_wait_seconds,
        "throughput_mib_per_second": throughput_mib_per_second,
    }


def _is_within(path: Path, root: Path) -> bool:
    """Return whether a path is contained within the given root."""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _select_portable_checkpoint(downloaded_root: Path) -> Path:
    """Find the sole portable model-and-tokenizer checkpoint in an artifact."""
    candidates: list[Path] = []
    tokenizer_names = {
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "vocab.json",
        "vocab.txt",
    }
    for config_path in sorted(downloaded_root.rglob("config.json"), key=str):
        candidate = config_path.parent
        has_weights = any(
            path.is_file()
            for pattern in ("model*.safetensors", "pytorch_model*.bin")
            for path in candidate.glob(pattern)
        )
        has_tokenizer = (candidate / "tokenizer_config.json").is_file() and any(
            (candidate / name).is_file() for name in tokenizer_names
        )
        if has_weights and has_tokenizer:
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            "The system.ai artifact does not contain a portable full model and "
            "tokenizer checkpoint"
        )
    if len(candidates) != 1:
        raise ValueError(
            "The system.ai artifact contains multiple portable checkpoints: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def _stage_system_ai_model(
    model_uri: str,
    cache_root_reference: str,
) -> tuple[str, dict[str, Any]]:
    """Download a system.ai model artifact into a node-local cache."""
    cache_root = Path(cache_root_reference).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        f"{LOCAL_CACHE_SCHEMA_VERSION}\0system_ai\0{model_uri}".encode("utf-8")
    ).hexdigest()[:24]
    destination = cache_root / cache_key
    marker_path = destination / ".air-local-cache.json"
    lock_path = cache_root / f".{cache_key}.lock"
    operation_started = time.monotonic()
    downloaded_by_process = False

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        lock_wait_seconds = time.monotonic() - operation_started
        if marker_path.is_file():
            marker = _read_local_cache_marker(marker_path, model_uri)
        else:
            if destination.exists():
                raise RuntimeError(
                    "Local system.ai cache exists without a completion marker: "
                    f"{destination}"
                )
            partial = cache_root / f".{cache_key}.partial-{os.getpid()}"
            if partial.exists():
                raise RuntimeError(f"Stale system.ai download path: {partial}")
            partial.mkdir(parents=False)
            download_started = time.monotonic()
            try:
                import mlflow

                mlflow.set_registry_uri("databricks-uc")
                downloaded = mlflow.artifacts.download_artifacts(
                    artifact_uri=model_uri,
                    dst_path=str(partial),
                )
                downloaded_root = Path(downloaded).expanduser().resolve()
                if not downloaded_root.is_dir() or not _is_within(
                    downloaded_root, partial.resolve()
                ):
                    raise RuntimeError(
                        "MLflow downloaded the system.ai artifact outside the "
                        "node-local staging directory"
                    )
                checkpoint = _select_portable_checkpoint(downloaded_root)
                inventory = _directory_inventory(checkpoint)
                total_bytes = sum(size for _, _, size in inventory)
                reserve_bytes = max(1 << 30, total_bytes // 10)
                if shutil.disk_usage(cache_root).free < reserve_bytes:
                    raise OSError(
                        f"Insufficient reserve under {cache_root} after system.ai "
                        f"download; need {reserve_bytes:,} free bytes"
                    )
                marker = {
                    "schema_version": LOCAL_CACHE_SCHEMA_VERSION,
                    "source_path": model_uri,
                    "acquisition": "system_ai_download",
                    "file_count": len(inventory),
                    "total_bytes": total_bytes,
                    "copy_workers": 0,
                    "copy_duration_seconds": time.monotonic() - download_started,
                }
                with (checkpoint / marker_path.name).open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(marker, handle, indent=2, sort_keys=True)
                os.replace(checkpoint, destination)
                downloaded_by_process = True
            finally:
                if partial.exists():
                    shutil.rmtree(partial, ignore_errors=True)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    duration_seconds = float(marker["copy_duration_seconds"])
    total_bytes = int(marker["total_bytes"])
    return str(destination), {
        **marker,
        "destination": str(destination),
        "staged": True,
        "cache_hit": not downloaded_by_process,
        "lock_wait_seconds": lock_wait_seconds,
        "throughput_mib_per_second": (
            total_bytes / max(duration_seconds, 1.0e-9) / (1024 * 1024)
        ),
    }


def _stage_model_references(
    config: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Resolve model and tokenizer references for the configured source type."""
    source_kind = str(config["model_source"])
    model_source = str(config["model_name"])
    tokenizer_source = str(config.get("tokenizer_path") or model_source)
    cache_root = str(config["local_model_cache_dir"])

    if source_kind == "volume":
        model_reference, model_staging = _stage_directory_to_local_cache(
            model_source,
            cache_root,
            int(config["local_model_cache_copy_workers"]),
        )
    elif source_kind == "system_ai":
        model_reference, model_staging = _stage_system_ai_model(
            model_source,
            cache_root,
        )
    else:
        model_reference = model_source
        model_staging = {
            "source_path": model_source,
            "destination": model_source,
            "acquisition": "hugging_face_download",
            "staged": False,
            "cache_hit": False,
        }

    tokenizer_reuses_model_cache = (
        Path(tokenizer_source).expanduser() == Path(model_source).expanduser()
        if _is_local_model_reference(tokenizer_source)
        and _is_local_model_reference(model_source)
        else tokenizer_source == model_source
    )
    if tokenizer_reuses_model_cache:
        tokenizer_reference = model_reference
        tokenizer_staging = model_staging
    elif source_kind == "volume":
        tokenizer_reference, tokenizer_staging = _stage_directory_to_local_cache(
            tokenizer_source,
            cache_root,
            int(config["local_model_cache_copy_workers"]),
        )
    else:
        tokenizer_reference = tokenizer_source
        tokenizer_staging = {
            "source_path": tokenizer_source,
            "destination": tokenizer_source,
            "acquisition": "hugging_face_download",
            "staged": False,
            "cache_hit": False,
        }

    return (
        model_reference,
        tokenizer_reference,
        {
            "model": model_staging,
            "tokenizer": tokenizer_staging,
            "tokenizer_reuses_model_cache": tokenizer_reuses_model_cache,
        },
    )


def _log_local_model_staging(staging: dict[str, Any]) -> None:
    """Log model and tokenizer staging metadata to the active MLflow run."""
    import mlflow

    model_staging = staging["model"]
    tokenizer_staging = staging["tokenizer"]
    mlflow.log_params(
        {
            "model_stage_source": model_staging["source_path"],
            "model_load_reference": model_staging["destination"],
            "model_staged_to_local": model_staging["staged"],
            "model_stage_rank_zero_cache_hit": model_staging["cache_hit"],
            "model_stage_file_count": model_staging.get("file_count", 0),
            "model_stage_total_bytes": model_staging.get("total_bytes", 0),
            "tokenizer_stage_source": tokenizer_staging["source_path"],
            "tokenizer_load_reference": tokenizer_staging["destination"],
            "tokenizer_staged_to_local": tokenizer_staging["staged"],
            "tokenizer_stage_rank_zero_cache_hit": tokenizer_staging["cache_hit"],
            "tokenizer_stage_file_count": tokenizer_staging.get("file_count", 0),
            "tokenizer_stage_total_bytes": tokenizer_staging.get("total_bytes", 0),
            "tokenizer_reuses_model_cache": staging["tokenizer_reuses_model_cache"],
        }
    )
    if model_staging["staged"]:
        mlflow.log_metrics(
            {
                "model_stage_copy_seconds": float(
                    model_staging["copy_duration_seconds"]
                ),
                "model_stage_lock_wait_seconds": float(
                    model_staging["lock_wait_seconds"]
                ),
                "model_stage_mib_per_second": float(
                    model_staging["throughput_mib_per_second"]
                ),
            }
        )
    if tokenizer_staging["staged"] and not staging["tokenizer_reuses_model_cache"]:
        mlflow.log_metrics(
            {
                "tokenizer_stage_copy_seconds": float(
                    tokenizer_staging["copy_duration_seconds"]
                ),
                "tokenizer_stage_lock_wait_seconds": float(
                    tokenizer_staging["lock_wait_seconds"]
                ),
                "tokenizer_stage_mib_per_second": float(
                    tokenizer_staging["throughput_mib_per_second"]
                ),
            }
        )


def _validate_runtime_inputs(config: dict[str, Any]) -> None:
    """Validate local inputs and prepare an empty or resumable output path."""
    for key in ("model_name", "tokenizer_path"):
        reference = config.get(key)
        if reference and _is_local_model_reference(str(reference)):
            path = Path(str(reference)).expanduser()
            if not path.is_dir():
                raise FileNotFoundError(f"{key} is not a readable directory: {path}")
    for key in ("train_data_path", "eval_data_path"):
        path = Path(config[key])
        if not path.is_file():
            raise FileNotFoundError(
                f"{key} does not exist or is not a file: {path}. "
                "Stage the JSONL data before starting training."
            )

    output_dir = Path(config["output_dir"])
    resume_from_checkpoint = config.get("resume_from_checkpoint")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"output_dir is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not resume_from_checkpoint:
            raise FileExistsError(
                f"output_dir must be empty unless resuming a checkpoint: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def _prepare_sft_dataset(dataset: Any, name: str, dataset_num_proc: int) -> Any:
    """Expand every assistant turn into a conversational completion example."""
    if "messages" not in dataset.column_names:
        raise ValueError(f"{name} dataset must contain a messages column")
    if len(dataset) == 0:
        raise ValueError(f"{name} dataset must not be empty")

    def expand_batch(
        batch: dict[str, list[Any]], indices: list[int]
    ) -> dict[str, list[Any]]:
        """Expand batched conversations into prompt-completion pairs."""
        prompts: list[list[dict[str, str]]] = []
        completions: list[list[dict[str, str]]] = []
        for row_index, messages in zip(indices, batch["messages"]):
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{name} row {row_index} messages must be non-empty")
            normalized: list[dict[str, str]] = []
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError(
                        f"{name} row {row_index} contains a non-mapping message"
                    )
                role = message.get("role")
                content = message.get("content")
                if role not in {"system", "user", "assistant"}:
                    raise ValueError(
                        f"{name} row {row_index} contains unsupported role {role!r}"
                    )
                if not isinstance(content, str) or not content:
                    raise ValueError(
                        f"{name} row {row_index} message content must be non-empty"
                    )
                normalized.append({"role": role, "content": content})

            assistant_turns = 0
            for message_index, message in enumerate(normalized):
                if message["role"] != "assistant":
                    continue
                if message_index == 0:
                    raise ValueError(
                        f"{name} row {row_index} starts with an assistant message"
                    )
                prompts.append(normalized[:message_index])
                completions.append([message])
                assistant_turns += 1
            if assistant_turns == 0:
                raise ValueError(f"{name} row {row_index} has no assistant response")
        return {"prompt": prompts, "completion": completions}

    prepared = dataset.map(
        expand_batch,
        batched=True,
        with_indices=True,
        remove_columns=dataset.column_names,
        num_proc=dataset_num_proc,
        desc=f"Preparing {name} assistant turns",
    )
    if len(prepared) == 0:
        raise ValueError(f"{name} dataset produced no assistant completions")
    return prepared


def _build_lora_config(config: dict[str, Any]) -> Any:
    """Build the PEFT LoRA configuration from normalized training settings."""
    from peft import LoraConfig

    return LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["lora_target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )


def _load_tokenizer(config: dict[str, Any], tokenizer_reference: str) -> Any:
    """Load and configure the tokenizer used by LoRA FSDP training."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_reference,
        trust_remote_code=False,
        local_files_only=_is_local_model_reference(tokenizer_reference),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define an EOS token or pad token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = int(config["sequence_len"])
    if not tokenizer.chat_template:
        raise ValueError(
            "The configured tokenizer must provide a chat template for messages data"
        )

    return tokenizer


def _build_sft_config(config: dict[str, Any], model_reference: str) -> Any:
    """Build TRL supervised fine-tuning arguments for LoRA with FSDP."""
    from trl import SFTConfig

    fsdp_config = dict(config["fsdp_config"])
    state_dict_type = str(fsdp_config.pop("state_dict_type"))
    # Transformers propagates most fsdp_config fields to Accelerate but does
    # not currently propagate state_dict_type. Set it before SFTTrainer creates
    # its Accelerator so adapter checkpoint saves gather a full state dict.
    os.environ["FSDP_STATE_DICT_TYPE"] = state_dict_type

    arguments: dict[str, Any] = {
        "output_dir": config["output_dir"],
        "model_init_kwargs": {
            "dtype": "bfloat16",
            "device_map": None,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
            "attn_implementation": config["attn_implementation"],
            "use_cache": False,
            "local_files_only": _is_local_model_reference(model_reference),
        },
        "max_length": int(config["sequence_len"]),
        "packing": bool(config["packing"]),
        # Conversational prompt/completion conversion makes every completion an
        # assistant turn without requiring tokenizer-specific generation masks.
        "completion_only_loss": bool(config["assistant_only_loss"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(config["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "num_train_epochs": float(config["num_epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "optim": config["optimizer"],
        "lr_scheduler_type": config["lr_scheduler"],
        "warmup_ratio": float(config["warmup_ratio"]),
        "weight_decay": float(config["weight_decay"]),
        "max_grad_norm": float(config["max_grad_norm"]),
        "logging_steps": int(config["logging_steps"]),
        "eval_strategy": config["eval_strategy"],
        "save_strategy": config["save_strategy"],
        "save_total_limit": int(config["save_total_limit"]),
        "save_safetensors": bool(config["save_safetensors"]),
        "bf16": bool(config["bf16"]),
        "tf32": bool(config["tf32"]),
        "gradient_checkpointing": bool(config["gradient_checkpointing"]),
        "fsdp": " ".join(config["fsdp"]),
        "fsdp_config": fsdp_config,
        "dataset_num_proc": int(config["dataset_num_proc"]),
        "dataloader_num_workers": int(config["dataloader_num_workers"]),
        "report_to": [],
        "run_name": config["mlflow_run_name"],
        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),
    }
    if config.get("max_steps") is not None:
        arguments["max_steps"] = int(config["max_steps"])
    return SFTConfig(**arguments)


def _start_or_reuse_mlflow_run(config: dict[str, Any]) -> tuple[Any, bool]:
    """Start or reuse a training run in the configured MLflow experiment."""
    import mlflow

    experiment = mlflow.set_experiment(config["experiment_path"])
    active_run = mlflow.active_run()
    owns_active_run = False
    if active_run is None:
        air_managed_run_id = os.environ.get("MLFLOW_RUN_ID", "").strip()
        if air_managed_run_id:
            active_run = mlflow.start_run(run_id=air_managed_run_id)
        else:
            active_run = mlflow.start_run(run_name=config["mlflow_run_name"])
            owns_active_run = True
    if str(active_run.info.experiment_id) != str(experiment.experiment_id):
        if owns_active_run:
            mlflow.end_run(status="FAILED")
        raise RuntimeError(
            "Active MLflow run belongs to experiment "
            f"{active_run.info.experiment_id}, expected {experiment.experiment_id} "
            f"for {config['experiment_path']}"
        )
    return active_run, owns_active_run


def _log_training_contract(
    config: dict[str, Any], world_size: int, train_rows: int, eval_rows: int
) -> None:
    """Record the effective LoRA FSDP training contract in MLflow."""
    import mlflow

    mlflow.set_tags(
        {
            "training_framework": "trl_sft_peft_fsdp",
            "training_scope": "lora_adapter",
            "model_source": config["model_source"],
            "source_model_uri": config["source_model_uri"],
        }
    )
    mlflow.log_params(
        {
            "base_model": config["model_name"],
            "migration_experiment_path": config["experiment_path"],
            "tokenizer": config.get("tokenizer_path") or config["model_name"],
            "local_model_cache_dir": config["local_model_cache_dir"],
            "local_model_cache_copy_workers": config["local_model_cache_copy_workers"],
            "train_data_path": config["train_data_path"],
            "eval_data_path": config["eval_data_path"],
            "train_rows": train_rows,
            "eval_rows": eval_rows,
            "world_size": world_size,
            "effective_global_batch_size": int(config["per_device_train_batch_size"])
            * world_size
            * int(config["gradient_accumulation_steps"]),
            "sequence_len": config["sequence_len"],
            "assistant_only_loss": config["assistant_only_loss"],
            "lora_r": config["lora_r"],
            "lora_alpha": config["lora_alpha"],
            "lora_dropout": config["lora_dropout"],
            "lora_target_modules": ",".join(config["lora_target_modules"]),
            "fsdp": ",".join(config["fsdp"]),
            "fsdp_transformer_layer_cls_to_wrap": config["fsdp_config"][
                "transformer_layer_cls_to_wrap"
            ],
            "fsdp_state_dict_type": config["fsdp_config"]["state_dict_type"],
            "fsdp_activation_checkpointing": config["fsdp_config"][
                "activation_checkpointing"
            ],
            "learning_rate": config["learning_rate"],
            "num_epochs": config["num_epochs"],
            "max_steps": config.get("max_steps") or "",
        }
    )


def merge_peft_model(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge the trained plain-LoRA adapter into its unquantized base model."""
    config, _ = load_training_config(config_path)
    if _needs_hf_token(config) and not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is required to reload the configured Hugging Face base model"
        )
    adapter_dir = Path(config["output_dir"]).resolve()
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_weights = next(
        (
            path
            for path in (
                adapter_dir / "adapter_model.safetensors",
                adapter_dir / "adapter_model.bin",
            )
            if path.is_file()
        ),
        None,
    )
    if not adapter_config.is_file() or adapter_weights is None:
        raise FileNotFoundError(
            "The final output must contain adapter_config.json and adapter weights"
        )

    rank, world_size, _ = distributed_context()
    if rank != 0:
        return {"rank": rank, "world_size": world_size, "merged_output_dir": None}

    model_reference, tokenizer_reference, staging = _stage_model_references(config)

    merged_output_dir = Path(config["merged_output_dir"]).resolve()
    if merged_output_dir.exists():
        if not merged_output_dir.is_dir():
            raise NotADirectoryError(
                f"merged_output_dir is not a directory: {merged_output_dir}"
            )
        if any(merged_output_dir.iterdir()):
            raise FileExistsError(
                f"merged_output_dir must be empty before merging: {merged_output_dir}"
            )
    merged_output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_reference,
        trust_remote_code=False,
        local_files_only=_is_local_model_reference(tokenizer_reference),
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_reference,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        attn_implementation="eager",
        local_files_only=_is_local_model_reference(model_reference),
    )
    if getattr(base_model, "is_quantized", False):
        raise ValueError("Plain-LoRA merging requires an unquantized base model")
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        is_trainable=False,
    )
    merged_model = peft_model.merge_and_unload(safe_merge=True, progressbar=True)
    merged_model.config.use_cache = True
    merged_model.save_pretrained(
        str(merged_output_dir),
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(str(merged_output_dir))

    return {
        "rank": rank,
        "world_size": world_size,
        "adapter_output_dir": str(adapter_dir),
        "merged_output_dir": str(merged_output_dir),
        "model_staging": staging["model"],
    }


def register_trained_model(
    mlflow_run_id: str,
    config_path: str | Path | None = None,
) -> dict[str, str]:
    """Log the merged full checkpoint and register it in Unity Catalog."""
    config, _ = load_training_config(config_path)
    run_id = str(mlflow_run_id or "").strip()
    if not run_id:
        raise ValueError("mlflow_run_id is required for model registration")

    merged_output_dir = Path(config["merged_output_dir"]).resolve()
    if not (merged_output_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"The merged output does not contain config.json: {merged_output_dir}"
        )
    weight_files = [
        path
        for pattern in ("model*.safetensors", "pytorch_model*.bin")
        for path in merged_output_dir.glob(pattern)
        if path.is_file()
    ]
    if not weight_files:
        raise FileNotFoundError(
            f"The merged output does not contain full model weights: "
            f"{merged_output_dir}"
        )
    if (merged_output_dir / "adapter_config.json").exists() or any(
        merged_output_dir.glob("adapter_model.*")
    ):
        raise ValueError("Merged output still contains PEFT adapter artifacts")

    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    experiment = mlflow.set_experiment(config["experiment_path"])
    training_run = mlflow.tracking.MlflowClient().get_run(run_id)
    if str(training_run.info.experiment_id) != str(experiment.experiment_id):
        raise ValueError(
            f"MLflow run {run_id} belongs to experiment "
            f"{training_run.info.experiment_id}, expected {experiment.experiment_id} "
            f"for {config['experiment_path']}"
        )
    registered_model_name = _validate_registered_model_name(config)
    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.transformers.log_model(
            transformers_model=str(merged_output_dir),
            task="text-generation",
            name="model",
            pip_requirements=[
                "mlflow>=3.6,<4",
                "torch",
                "transformers>=4.56,<5",
                "accelerate>=1.4,<2",
                "safetensors>=0.4",
            ],
            metadata={
                "artifact_type": "merged_peft_checkpoint",
                "training_framework": "trl_sft_peft_fsdp",
                "base_model": config["model_name"],
                "base_model_source": config["model_source"],
                "base_model_uri": config["source_model_uri"],
                "adapter_output_dir": config["output_dir"],
                "training_run_id": run_id,
            },
        )
    registered = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=registered_model_name,
        await_registration_for=600,
    )
    return {
        "name": str(registered.name),
        "version": str(registered.version),
        "model_uri": f"models:/{registered.name}/{registered.version}",
        "logged_model_uri": str(model_info.model_uri),
        "source_model_uri": config["source_model_uri"],
    }
