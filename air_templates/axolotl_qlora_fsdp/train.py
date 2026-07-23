"""Fine-tune Llama 3.1 70B Instruct with Axolotl QLoRA and FSDP.

The module supports both launchers included with this template:

* ``torchrun`` through ``air run --file train.yaml``.
* The ``@distributed`` function in ``01_runner.py``.

AI Runtime writes YAML ``parameters`` to ``HYPERPARAMETERS_PATH``. Outside an
AIR CLI run, this module reads the adjacent ``train.yaml`` instead.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "train.yaml"
VOLUME_PREFIX = "/Volumes/"
MODEL_URI_PATTERN = re.compile(r"^models:/([^/]+)/([1-9][0-9]*)$")
REQUIRED_FSDP_MODES = {"full_shard", "auto_wrap"}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _resolve_config_path(config_path: str | Path | None = None) -> Path:
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
    value = str(config.get(key, "")).strip()
    if not value:
        raise ValueError(f"training_config.{key} must be a non-empty string")
    return value


def _validate_registered_model_name(config: dict[str, Any]) -> str:
    name = _required_string(config, "registered_model_name")
    if len(name.split(".")) != 3 or any(not part.strip() for part in name.split(".")):
        raise ValueError(
            "training_config.registered_model_name must use "
            "<catalog>.<schema>.<model>"
        )
    return name


def _is_local_model_reference(value: str) -> bool:
    return Path(value).expanduser().is_absolute()


def _needs_hf_token(config: dict[str, Any]) -> bool:
    references = [str(config["model_name"])]
    if config.get("tokenizer_path"):
        references.append(str(config["tokenizer_path"]))
    return any(not _is_local_model_reference(value) for value in references)


def _validate_model_source(config: dict[str, Any]) -> None:
    use_existing_weights = config.get("use_existing_weights")
    if not isinstance(use_existing_weights, bool):
        raise ValueError("training_config.use_existing_weights must be true or false")

    model_source = _required_string(config, "model_source")
    if model_source not in {"existing_uc", "system_ai", "hugging_face"}:
        raise ValueError(
            "training_config.model_source must be existing_uc, system_ai, or hugging_face"
        )
    source_model_uri = config.get("source_model_uri")
    model_uri_match = None
    if source_model_uri is not None:
        source_model_uri = str(source_model_uri).strip()
        model_uri_match = MODEL_URI_PATTERN.fullmatch(source_model_uri)
        if model_uri_match is None or len(model_uri_match.group(1).split(".")) != 3:
            raise ValueError(
                "training_config.source_model_uri must be null or use "
                "models:/<catalog>.<schema>.<model>/<version>"
            )
        config["source_model_uri"] = source_model_uri

    references = [str(config["model_name"])]
    if config.get("tokenizer_path"):
        references.append(str(config["tokenizer_path"]))
    volume_references = all(value.startswith(VOLUME_PREFIX) for value in references)

    if use_existing_weights:
        if model_source != "existing_uc" or model_uri_match is None:
            raise ValueError(
                "use_existing_weights=true requires model_source=existing_uc and "
                "a versioned source_model_uri"
            )
        if not volume_references:
            raise ValueError(
                "Existing UC weights and tokenizer must be materialized under /Volumes"
            )
    elif model_source == "existing_uc":
        raise ValueError("model_source=existing_uc requires use_existing_weights=true")
    elif model_source == "system_ai":
        if model_uri_match is None or not model_uri_match.group(1).startswith("system.ai."):
            raise ValueError(
                "model_source=system_ai requires a versioned models:/system.ai.<model> URI"
            )
        if not volume_references:
            raise ValueError(
                "system.ai weights and tokenizer must be materialized under /Volumes"
            )
    else:
        if source_model_uri is not None:
            raise ValueError("model_source=hugging_face requires source_model_uri=null")
        if any(_is_local_model_reference(value) for value in references):
            raise ValueError(
                "model_source=hugging_face requires remote Hugging Face model references"
            )


def _positive_int(config: dict[str, Any], key: str) -> int:
    value = int(config.get(key, 0))
    if value <= 0:
        raise ValueError(f"training_config.{key} must be greater than zero")
    return value


def _positive_float(config: dict[str, Any], key: str) -> float:
    value = float(config.get(key, 0))
    if value <= 0:
        raise ValueError(f"training_config.{key} must be greater than zero")
    return value


def _nonnegative_float(config: dict[str, Any], key: str) -> float:
    value = float(config.get(key, -1))
    if value < 0:
        raise ValueError(f"training_config.{key} must be zero or greater")
    return value


def _required_bool(config: dict[str, Any], key: str) -> bool:
    value = config.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"training_config.{key} must be true or false")
    return value


def _validate_fsdp(config: dict[str, Any]) -> None:
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
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        "state_dict_type": "FULL_STATE_DICT",
        "final_state_dict_type": "FULL_STATE_DICT",
    }
    for key, expected in expected_strings.items():
        value = str(fsdp_config.get(key, "")).strip()
        if value != expected:
            raise ValueError(
                f"training_config.fsdp_config.{key} must be {expected!r}; got {value!r}"
            )

    for key in (
        "sync_module_states",
        "offload_params",
        "use_orig_params",
        "cpu_ram_efficient_loading",
    ):
        if not isinstance(fsdp_config.get(key), bool):
            raise ValueError(f"training_config.fsdp_config.{key} must be true or false")

    if not fsdp_config["sync_module_states"]:
        raise ValueError(
            "training_config.fsdp_config.sync_module_states must be true when "
            "cpu_ram_efficient_loading is enabled"
        )
    if not fsdp_config["cpu_ram_efficient_loading"]:
        raise ValueError(
            "training_config.fsdp_config.cpu_ram_efficient_loading must be true"
        )
    if fsdp_config["use_orig_params"]:
        raise ValueError(
            "training_config.fsdp_config.use_orig_params must be false for "
            "this PEFT FSDP recipe"
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
        "model_source",
        "experiment_path",
        "mlflow_run_name",
        "registered_model_name",
        "train_data_path",
        "eval_data_path",
        "output_dir",
        "merged_output_dir",
        "optimizer",
        "lr_scheduler",
        "attn_implementation",
    ):
        _required_string(config, key)

    _validate_registered_model_name(config)

    tokenizer_path = config.get("tokenizer_path")
    if tokenizer_path is not None:
        tokenizer_path = str(tokenizer_path).strip()
        if not tokenizer_path:
            raise ValueError("training_config.tokenizer_path must be null or non-empty")
        config["tokenizer_path"] = tokenizer_path

    _validate_model_source(config)

    for key in (
        "sequence_len",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "num_epochs",
        "logging_steps",
        "evals_per_epoch",
        "saves_per_epoch",
        "save_total_limit",
        "lora_r",
        "lora_alpha",
    ):
        _positive_int(config, key)

    _positive_float(config, "learning_rate")
    warmup_ratio = _nonnegative_float(config, "warmup_ratio")
    if warmup_ratio > 1:
        raise ValueError("training_config.warmup_ratio must not exceed 1")
    _nonnegative_float(config, "weight_decay")
    lora_dropout = _nonnegative_float(config, "lora_dropout")
    if lora_dropout >= 1:
        raise ValueError("training_config.lora_dropout must be less than 1")

    for key in (
        "load_in_8bit",
        "load_in_4bit",
        "sample_packing",
        "eval_sample_packing",
        "tf32",
        "gradient_checkpointing",
        "gradient_checkpointing_use_reentrant",
        "save_first_step",
        "save_safetensors",
    ):
        _required_bool(config, key)

    if config.get("adapter") != "qlora":
        raise ValueError("training_config.adapter must be 'qlora'")
    if config["load_in_8bit"] or not config["load_in_4bit"]:
        raise ValueError(
            "load_in_8bit must be false and load_in_4bit must be true for QLoRA"
        )
    if not config["gradient_checkpointing"]:
        raise ValueError("gradient_checkpointing must be true for this template")
    if not config["gradient_checkpointing_use_reentrant"]:
        raise ValueError(
            "gradient_checkpointing_use_reentrant must be true with the pinned "
            "Axolotl FSDP-QLoRA recipe"
        )
    if config.get("bf16") not in (True, False, "auto"):
        raise ValueError("training_config.bf16 must be true, false, or 'auto'")

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
                f"training_config.{key} must use a Unity Catalog volume path "
                f"starting with {VOLUME_PREFIX!r}; got {value!r}"
            )
    if train_path == eval_path:
        raise ValueError("train_data_path and eval_data_path must be different")
    if Path(output_dir).resolve() == Path(merged_output_dir).resolve():
        raise ValueError("output_dir and merged_output_dir must be different")

    target_modules = config.get("lora_target_modules")
    if not isinstance(target_modules, list) or not target_modules:
        raise ValueError("training_config.lora_target_modules must be a non-empty list")
    if not all(isinstance(module, str) and module.strip() for module in target_modules):
        raise ValueError("Every lora_target_modules value must be a non-empty string")

    max_steps = config.get("max_steps")
    if max_steps is not None and int(max_steps) <= 0:
        raise ValueError("training_config.max_steps must be null or greater than zero")

    _validate_fsdp(config)
    return dict(config), resolved_path


def build_axolotl_config(config: dict[str, Any]) -> dict[str, Any]:
    """Translate the workload parameters into an Axolotl configuration."""

    def chat_dataset(path: str) -> dict[str, Any]:
        return {
            "path": path,
            "ds_type": "json",
            "type": "chat_template",
            "field_messages": "messages",
            "roles_to_train": ["assistant"],
            "train_on_eos": "turn",
        }

    axolotl_config: dict[str, Any] = {
        "base_model": config["model_name"],
        "tokenizer_config": config.get("tokenizer_path") or config["model_name"],
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "trust_remote_code": False,
        "hf_use_auth_token": _needs_hf_token(config),
        "load_in_8bit": config["load_in_8bit"],
        "load_in_4bit": config["load_in_4bit"],
        "datasets": [chat_dataset(config["train_data_path"])],
        "test_datasets": [chat_dataset(config["eval_data_path"])],
        "val_set_size": 0.0,
        "chat_template": "tokenizer_default",
        "dataset_prepared_path": None,
        "output_dir": config["output_dir"],
        "adapter": config["adapter"],
        "lora_model_dir": None,
        "sequence_len": int(config["sequence_len"]),
        "sample_packing": config["sample_packing"],
        "eval_sample_packing": config["eval_sample_packing"],
        "lora_r": int(config["lora_r"]),
        "lora_alpha": int(config["lora_alpha"]),
        "lora_dropout": float(config["lora_dropout"]),
        "lora_target_modules": list(config["lora_target_modules"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "micro_batch_size": int(config["micro_batch_size"]),
        "num_epochs": int(config["num_epochs"]),
        "optimizer": config["optimizer"],
        "lr_scheduler": config["lr_scheduler"],
        "learning_rate": float(config["learning_rate"]),
        "bf16": config["bf16"],
        "tf32": config["tf32"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": {
            "use_reentrant": config["gradient_checkpointing_use_reentrant"]
        },
        "resume_from_checkpoint": config.get("resume_from_checkpoint"),
        "logging_steps": int(config["logging_steps"]),
        "warmup_ratio": float(config["warmup_ratio"]),
        "evals_per_epoch": int(config["evals_per_epoch"]),
        "saves_per_epoch": int(config["saves_per_epoch"]),
        "save_total_limit": int(config["save_total_limit"]),
        "save_first_step": config["save_first_step"],
        "save_safetensors": config["save_safetensors"],
        "weight_decay": float(config["weight_decay"]),
        "attn_implementation": config["attn_implementation"],
        "device_map": None,
        "ddp_find_unused_parameters": False,
        "fsdp": list(config["fsdp"]),
        "fsdp_config": dict(config["fsdp_config"]),
        "seed": int(config["seed"]),
        "special_tokens": {
            "pad_token": "<|finetune_right_pad_id|>",
            "eos_token": "<|eot_id|>",
        },
        "use_mlflow": True,
        "mlflow_tracking_uri": "databricks",
        "mlflow_run_name": config["mlflow_run_name"],
        "hf_mlflow_log_artifacts": False,
        "wandb_mode": "disabled",
        "wandb_project": None,
        "wandb_entity": None,
        "wandb_watch": None,
        "wandb_name": None,
        "wandb_log_model": None,
    }

    max_steps = config.get("max_steps")
    if max_steps is not None:
        axolotl_config["max_steps"] = int(max_steps)
    dataset_prepared_path = config.get("dataset_prepared_path")
    if dataset_prepared_path:
        axolotl_config["dataset_prepared_path"] = str(dataset_prepared_path)

    return axolotl_config


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


def _validate_runtime_inputs(config: dict[str, Any]) -> None:
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
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)


def run_training(
    config_path: str | Path | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Validate inputs, run distributed Axolotl training, and return metadata."""
    config, resolved_path = load_training_config(config_path)

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    if _needs_hf_token(config) and not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is required for the configured remote model or tokenizer. "
            "Configure train.yaml secrets for AIR CLI "
            "runs or the notebook secret widgets for interactive runs."
        )

    os.environ.setdefault("AXOLOTL_DO_NOT_TRACK", "1")
    os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "false")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _validate_runtime_inputs(config)
    rank, world_size, local_rank = distributed_context()

    import mlflow
    import torch
    from axolotl.cli.config import load_cfg
    from axolotl.common.datasets import load_datasets
    from axolotl.train import train
    from axolotl.utils import set_pytorch_cuda_alloc_conf
    from axolotl.utils.dict import DictDefault

    set_pytorch_cuda_alloc_conf()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(config["experiment_path"])

    raw_axolotl_config = build_axolotl_config(config)
    cfg = load_cfg(DictDefault(**raw_axolotl_config))
    dataset_meta = load_datasets(cfg=cfg)

    if rank == 0:
        print(f"Configuration: {resolved_path}")
        print(f"Model: {config['model_name']}")
        print(f"Model source: {config['model_source']}")
        print(f"Source model URI: {config.get('source_model_uri')}")
        print(f"Training data: {config['train_data_path']}")
        print(f"Evaluation data: {config['eval_data_path']}")
        print(f"Adapter output: {config['output_dir']}")
        print(f"Distributed world size: {world_size}")

    _, _, trainer = train(cfg=cfg, dataset_meta=dataset_meta)

    active_run = mlflow.last_active_run()
    run_id = active_run.info.run_id if active_run is not None else None
    return {
        "rank": rank,
        "world_size": world_size,
        "global_step": int(trainer.state.global_step),
        "mlflow_run_id": run_id if rank == 0 else None,
        "output_dir": config["output_dir"] if rank == 0 else None,
        "model_source": config["model_source"] if rank == 0 else None,
        "source_model_uri": config.get("source_model_uri") if rank == 0 else None,
    }


def merge_peft_model(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge the trained PEFT adapter into an unquantized base checkpoint."""
    config, _ = load_training_config(config_path)
    if _needs_hf_token(config) and not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is required to reload the configured base model or tokenizer"
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
            "The final output must contain adapter_config.json and "
            "adapter_model.safetensors or adapter_model.bin"
        )

    rank, world_size, _ = distributed_context()
    if rank != 0:
        return {
            "rank": rank,
            "world_size": world_size,
            "merged_output_dir": None,
        }

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

    tokenizer_reference = config.get("tokenizer_path") or config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_reference, trust_remote_code=False
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    if getattr(base_model, "is_quantized", False):
        raise ValueError("PEFT merging requires an unquantized base model")
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
    }


def register_trained_model(
    training_result: dict[str, Any],
    merge_result: dict[str, Any],
    config_path: str | Path | None = None,
) -> dict[str, str]:
    """Log the merged full checkpoint and register it in Unity Catalog."""
    config, _ = load_training_config(config_path)
    run_id = str(training_result.get("mlflow_run_id") or "").strip()
    if not run_id:
        raise ValueError("Training must return an MLflow run ID before registration")

    merged_output_dir = Path(
        str(merge_result.get("merged_output_dir") or "")
    ).resolve()
    if merged_output_dir != Path(config["merged_output_dir"]).resolve():
        raise ValueError("Merge result merged_output_dir does not match train.yaml")
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
            f"The merged output does not contain full model weights: {merged_output_dir}"
        )
    if (merged_output_dir / "adapter_config.json").exists() or any(
        merged_output_dir.glob("adapter_model.*")
    ):
        raise ValueError("Merged output still contains PEFT adapter artifacts")

    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    registered_model_name = _validate_registered_model_name(config)
    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.transformers.log_model(
            transformers_model=str(merged_output_dir),
            task="text-generation",
            name="model",
            pip_requirements=[
                "mlflow>=3.6,<4",
                "torch",
                "transformers",
                "accelerate",
                "safetensors",
            ],
            metadata={
                "artifact_type": "merged_peft_checkpoint",
                "base_model": config["model_name"],
                "base_model_source": config["model_source"],
                "base_model_uri": config.get("source_model_uri"),
                "use_existing_weights": config["use_existing_weights"],
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
        "source_model_uri": str(model_info.model_uri),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML file to use instead of HYPERPARAMETERS_PATH/train.yaml.",
    )
    args = parser.parse_args()
    result = run_training(config_path=args.config)
    if result["rank"] == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
