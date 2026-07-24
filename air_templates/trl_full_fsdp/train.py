"""Primary full-weight FSDP training entry point for Databricks AI Runtime.

Helpers live in ``helper_utils.py`` and ``training_utils.py`` so this module
contains only the training workflow used by AIR ``torchrun`` and the
``@distributed`` runner notebook.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from helper_utils import _needs_hf_token, load_training_config
from training_utils import (
    _build_sft_config,
    _load_tokenizer,
    _log_local_model_staging,
    _log_training_contract,
    _prepare_sft_dataset,
    _stage_model_references,
    _start_or_reuse_mlflow_run,
    _validate_runtime_inputs,
    distributed_context,
)


def run_training(
    config_path: str | Path | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Run full-weight TRL supervised fine-tuning with FSDP full sharding."""
    config, resolved_path = load_training_config(config_path)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    if _needs_hf_token(config) and not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is required for the configured remote model or tokenizer. "
            "Configure train.yaml secrets for AIR CLI runs or notebook widgets."
        )

    os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "false")
    os.environ.setdefault("MLFLOW_FLATTEN_PARAMS", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    _validate_runtime_inputs(config)
    rank, world_size, local_rank = distributed_context()

    import mlflow
    import torch
    from datasets import load_dataset
    from trl import SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("TRL full-weight FSDP training requires CUDA")
    torch.cuda.set_device(local_rank)

    model_reference, tokenizer_reference, staging = _stage_model_references(config)

    train_dataset = load_dataset(
        "json", data_files=config["train_data_path"], split="train"
    )
    eval_dataset = load_dataset(
        "json", data_files=config["eval_data_path"], split="train"
    )
    train_dataset = _prepare_sft_dataset(
        train_dataset, "training", int(config["dataset_num_proc"])
    )
    eval_dataset = _prepare_sft_dataset(
        eval_dataset, "evaluation", int(config["dataset_num_proc"])
    )

    sft_config = _build_sft_config(config, model_reference)
    tokenizer = _load_tokenizer(config, tokenizer_reference)
    trainer = SFTTrainer(
        # Passing the model reference lets SFTTrainer load after SFTConfig has
        # initialized distributed state and the FSDP rank-0 loading contract.
        model=model_reference,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    if getattr(trainer.model, "is_quantized", False):
        raise ValueError("Full-weight FSDP training forbids quantized models")
    trainable_parameters = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel() for parameter in trainer.model.parameters()
    )
    if total_parameters <= 0 or trainable_parameters != total_parameters:
        raise RuntimeError(
            "Full fine-tuning requires every model parameter to be trainable; got "
            f"{trainable_parameters:,} of {total_parameters:,}"
        )

    mlflow.set_tracking_uri("databricks")
    active_run = None
    owns_active_run = False
    if rank == 0:
        active_run, owns_active_run = _start_or_reuse_mlflow_run(config)
        _log_training_contract(
            config, world_size, len(train_dataset), len(eval_dataset)
        )
        _log_local_model_staging(staging)
        mlflow.log_params(
            {
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
            }
        )
        print(f"Configuration: {resolved_path}")
        print(f"Model: {config['model_name']}")
        print(f"Model load reference: {model_reference}")
        print(f"Model source: {config['model_source']}")
        print(f"Source model URI: {config.get('source_model_uri')}")
        print(f"Training data: {config['train_data_path']}")
        print(f"Evaluation data: {config['eval_data_path']}")
        print(f"Full-model output: {config['output_dir']}")
        print(f"Distributed world size: {world_size}")
        print(f"Trainable parameters: {trainable_parameters:,} / {total_parameters:,}")

    try:
        train_output = trainer.train(
            resume_from_checkpoint=config.get("resume_from_checkpoint")
        )
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        unwrapped_model.config.use_cache = True
        # FSDP FULL_STATE_DICT collection is collective. Every rank calls
        # save_model; Trainer restricts filesystem writes to rank 0.
        trainer.save_model(config["output_dir"])
        trainer.accelerator.wait_for_everyone()
        if rank == 0:
            tokenizer.save_pretrained(config["output_dir"])
            metrics = getattr(train_output, "metrics", {}) or {}
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"trainer_{name}", float(value))

            output_dir = Path(config["output_dir"])
            if not (output_dir / "config.json").is_file():
                raise FileNotFoundError("SFTTrainer did not save model config.json")
            weight_files = [
                path
                for pattern in ("model*.safetensors", "pytorch_model*.bin")
                for path in output_dir.glob(pattern)
                if path.is_file()
            ]
            if not weight_files:
                raise FileNotFoundError(
                    "SFTTrainer did not save gathered full model weights"
                )
            if (output_dir / "adapter_config.json").exists() or any(
                output_dir.glob("adapter_model.*")
            ):
                raise ValueError("Full-weight output contains PEFT adapter artifacts")
            mlflow.log_param("full_model_output_dir", config["output_dir"])

        run_id = active_run.info.run_id if active_run is not None else None
    except Exception:
        if active_run is not None and owns_active_run:
            mlflow.end_run(status="FAILED")
        raise
    else:
        if active_run is not None and owns_active_run:
            mlflow.end_run(status="FINISHED")

    return {
        "rank": rank,
        "world_size": world_size,
        "global_step": int(trainer.state.global_step),
        "mlflow_run_id": run_id if rank == 0 else None,
        "output_dir": config["output_dir"] if rank == 0 else None,
        "model_source": config["model_source"] if rank == 0 else None,
        "source_model_uri": config.get("source_model_uri") if rank == 0 else None,
        "migration_experiment_path": config["experiment_path"] if rank == 0 else None,
        "model_staging": staging["model"] if rank == 0 else None,
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
