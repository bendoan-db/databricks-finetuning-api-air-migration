#!/usr/bin/env python3
"""Compare assistant response-token accuracy for two portable HF checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


MODEL_URI_PATTERN = re.compile(r"^models:/([^/]+)/([1-9][0-9]*)$")


@dataclass(frozen=True)
class EvaluationRecord:
    line_number: int
    digest: str
    messages: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class EncodedRecord:
    line_number: int
    digest: str
    input_ids: tuple[int, ...]
    assistant_mask: tuple[bool, ...]
    truncated_tokens: int

    @property
    def scored_tokens(self) -> int:
        return sum(self.assistant_mask[1:])

    @property
    def tokenization_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(struct.pack(">Q", len(self.input_ids)))
        for token_id in self.input_ids:
            digest.update(struct.pack(">q", token_id))
        digest.update(bytes(self.assistant_mask))
        return digest.hexdigest()


def _required_string(mapping: dict[str, Any], key: str, prefix: str) -> str:
    value = str(mapping.get(key, "")).strip()
    if not value:
        raise ValueError(f"{prefix}.{key} must be a non-empty string")
    return value


def migration_config(config_path: Path) -> tuple[str, str]:
    """Return the configured legacy URI and migration experiment path."""
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

    experiment_path = _required_string(source, "migration_experiment_path", "source")
    workspace_path = PurePosixPath(experiment_path)
    if (
        not workspace_path.is_absolute()
        or workspace_path == PurePosixPath("/")
        or any(part in {"", ".", ".."} for part in workspace_path.parts[1:])
        or "\\" in experiment_path
        or any(ord(character) < 32 for character in experiment_path)
    ):
        raise ValueError(
            "source.migration_experiment_path must be an absolute Databricks "
            "workspace path such as /Shared/fmt-migration"
        )
    return f"models:/{catalog}.{schema}.{model}/{version}", experiment_path


def source_model_uri(config_path: Path) -> str:
    """Return the exact legacy model URI declared by the migration config."""
    return migration_config(config_path)[0]


def migration_experiment_path(config_path: Path) -> str:
    """Return the authoritative experiment declared by the migration config."""
    return migration_config(config_path)[1]


def _start_or_reuse_mlflow_run(
    experiment_path: str,
    *,
    mlflow_module: Any | None = None,
) -> tuple[Any, Any, bool]:
    """Bind token evaluation to the configured migration experiment."""
    if mlflow_module is None:
        try:
            import mlflow as mlflow_module
        except ImportError as exc:
            raise RuntimeError(
                "MLflow is required to log token-accuracy evaluation"
            ) from exc

    mlflow_module.set_tracking_uri("databricks")
    experiment = mlflow_module.set_experiment(experiment_path)
    active_run = mlflow_module.active_run()
    owns_active_run = False
    if active_run is None:
        air_managed_run_id = os.environ.get("MLFLOW_RUN_ID", "").strip()
        if air_managed_run_id:
            active_run = mlflow_module.start_run(run_id=air_managed_run_id)
        else:
            active_run = mlflow_module.start_run(
                run_name="air-migration-token-accuracy"
            )
            owns_active_run = True
    if str(active_run.info.experiment_id) != str(experiment.experiment_id):
        if owns_active_run:
            mlflow_module.end_run(status="FAILED")
        raise RuntimeError(
            "Active MLflow evaluation run belongs to experiment "
            f"{active_run.info.experiment_id}, expected {experiment.experiment_id} "
            f"for {experiment_path}"
        )
    return mlflow_module, active_run, owns_active_run


def validate_model_uri(model_uri: str, label: str) -> str:
    value = model_uri.strip()
    match = MODEL_URI_PATTERN.fullmatch(value)
    if match is None or len(match.group(1).split(".")) != 3:
        raise ValueError(
            f"{label} must use models:/<catalog>.<schema>.<model>/<version>"
        )
    return value


def validate_full_checkpoint(model_path: Path, label: str) -> Path:
    resolved = model_path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} model path is not a directory: {resolved}")
    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"{label} checkpoint has no config.json: {resolved}")

    full_weights = [
        path
        for pattern in (
            "model*.safetensors",
            "pytorch_model*.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
        for path in resolved.glob(pattern)
        if path.name != "adapter_model.safetensors"
    ]
    if not full_weights:
        if (resolved / "adapter_config.json").is_file() or any(
            resolved.glob("adapter_model.*")
        ):
            raise ValueError(
                f"{label} is adapter-only; merge PEFT weights into the base model first"
            )
        raise FileNotFoundError(
            f"{label} checkpoint has no full Hugging Face model weights: {resolved}"
        )
    return resolved


def validate_tokenizer_path(tokenizer_path: Path, label: str) -> Path:
    resolved = tokenizer_path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"{label} tokenizer path is not a directory: {resolved}"
        )
    if not (resolved / "tokenizer_config.json").is_file():
        raise FileNotFoundError(
            f"{label} tokenizer path has no tokenizer_config.json: {resolved}"
        )
    return resolved


def load_evaluation_records(path: Path) -> tuple[list[EvaluationRecord], str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Evaluation JSONL does not exist: {resolved}")
    raw_data = resolved.read_bytes()
    dataset_digest = hashlib.sha256(raw_data).hexdigest()
    records: list[EvaluationRecord] = []

    for line_number, raw_line in enumerate(raw_data.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON on evaluation line {line_number}") from exc
        if not isinstance(payload, dict) or not isinstance(
            payload.get("messages"), list
        ):
            raise ValueError(
                f"Evaluation line {line_number} must contain a messages array"
            )

        messages: list[dict[str, str]] = []
        assistant_count = 0
        for index, message in enumerate(payload["messages"]):
            if not isinstance(message, dict):
                raise ValueError(
                    f"Evaluation line {line_number} message {index} must be an object"
                )
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role.strip():
                raise ValueError(
                    f"Evaluation line {line_number} message {index} has no valid role"
                )
            if not isinstance(content, str):
                raise ValueError(
                    f"Evaluation line {line_number} message {index} content must be text"
                )
            normalized = {"role": role, "content": content}
            messages.append(normalized)
            assistant_count += role == "assistant"
        if not messages or assistant_count == 0:
            raise ValueError(
                f"Evaluation line {line_number} must contain an assistant response"
            )
        records.append(
            EvaluationRecord(
                line_number=line_number,
                digest=hashlib.sha256(raw_line).hexdigest(),
                messages=tuple(messages),
            )
        )

    if not records:
        raise ValueError("Evaluation JSONL contains no records")
    return records, dataset_digest


def _flat_token_ids(value: Any, label: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = value.get("input_ids")
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"Chat template returned invalid token IDs for {label}")
    return value


def _render_chat(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    add_generation_prompt: bool,
    template_date: str,
    label: str,
) -> list[int]:
    try:
        rendered = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            date_string=template_date,
        )
    except Exception as exc:
        raise ValueError(
            f"Unable to apply the chat template for {label}: {exc}"
        ) from exc
    return _flat_token_ids(rendered, label)


def encode_record(
    tokenizer: Any,
    record: EvaluationRecord,
    *,
    max_sequence_length: int,
    truncation: str,
    template_date: str,
) -> EncodedRecord:
    label = f"evaluation line {record.line_number}"
    full_ids = _render_chat(
        tokenizer,
        record.messages,
        add_generation_prompt=False,
        template_date=template_date,
        label=label,
    )
    if not full_ids:
        raise ValueError(f"Chat template emitted no tokens for {label}")
    assistant_mask = [False] * len(full_ids)

    for index, message in enumerate(record.messages):
        if message["role"] != "assistant":
            continue
        prefix_ids = _render_chat(
            tokenizer,
            record.messages[:index],
            add_generation_prompt=True,
            template_date=template_date,
            label=f"{label} assistant prefix {index}",
        )
        through_ids = _render_chat(
            tokenizer,
            record.messages[: index + 1],
            add_generation_prompt=False,
            template_date=template_date,
            label=f"{label} assistant turn {index}",
        )
        if through_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"Ambiguous assistant boundary on {label}: the generation prompt is "
                "not a stable prefix of the completed assistant turn"
            )
        if full_ids[: len(through_ids)] != through_ids:
            raise ValueError(
                f"Ambiguous assistant boundary on {label}: later messages rewrite "
                "the preceding serialized conversation"
            )
        if len(through_ids) <= len(prefix_ids):
            raise ValueError(f"Assistant turn emitted no scorable tokens on {label}")
        for position in range(len(prefix_ids), len(through_ids)):
            assistant_mask[position] = True

    truncated_tokens = max(0, len(full_ids) - max_sequence_length)
    if truncated_tokens:
        if truncation == "error":
            raise ValueError(
                f"{label} has {len(full_ids)} tokens, exceeding "
                f"--max-sequence-length={max_sequence_length}"
            )
        full_ids = full_ids[truncated_tokens:]
        assistant_mask = assistant_mask[truncated_tokens:]

    encoded = EncodedRecord(
        line_number=record.line_number,
        digest=record.digest,
        input_ids=tuple(full_ids),
        assistant_mask=tuple(assistant_mask),
        truncated_tokens=truncated_tokens,
    )
    if encoded.scored_tokens == 0:
        raise ValueError(f"No assistant response tokens remain on {label}")
    return encoded


def _stable_json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    vocab = sorted(
        (str(token), int(token_id)) for token, token_id in tokenizer.get_vocab().items()
    )
    chat_template = getattr(tokenizer, "chat_template", None)
    special_tokens = getattr(tokenizer, "special_tokens_map", {})
    return {
        "class": tokenizer.__class__.__name__,
        "vocabulary_size": len(vocab),
        "vocabulary_sha256": _stable_json_digest(vocab),
        "chat_template_sha256": _stable_json_digest(chat_template),
        "special_tokens_sha256": _stable_json_digest(special_tokens),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }


def _batched(
    values: Sequence[EncodedRecord], size: int
) -> Iterable[Sequence[EncodedRecord]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _torch_dtype(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _input_device(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except (AttributeError, StopIteration):
        return next(model.parameters()).device


def score_checkpoint(
    *,
    label: str,
    model_uri: str,
    model_path: Path,
    tokenizer_path: Path,
    records: Sequence[EvaluationRecord],
    max_sequence_length: int,
    truncation: str,
    template_date: str,
    batch_size: int,
    dtype: str,
    device: str,
) -> dict[str, Any]:
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "torch, transformers, accelerate, and safetensors are required for evaluation"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{label} tokenizer has no explicit chat template")
    encoded_records = [
        encode_record(
            tokenizer,
            record,
            max_sequence_length=max_sequence_length,
            truncation=truncation,
            template_date=template_date,
        )
        for record in records
    ]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError(f"{label} tokenizer has neither a pad nor EOS token")

    device_map: str | dict[str, str]
    device_map = "auto" if device == "auto" else {"": device}
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        device_map=device_map,
        torch_dtype=_torch_dtype(torch, dtype),
        low_cpu_mem_usage=True,
    )
    model.eval()
    input_device = _input_device(model)
    per_record: list[dict[str, Any]] = []
    correct_total = 0
    scored_total = 0

    try:
        for batch in _batched(encoded_records, batch_size):
            longest = max(len(item.input_ids) for item in batch)
            input_ids = torch.full(
                (len(batch), longest),
                int(pad_token_id),
                dtype=torch.long,
                device=input_device,
            )
            attention_mask = torch.zeros(
                (len(batch), longest), dtype=torch.long, device=input_device
            )
            assistant_mask = torch.zeros(
                (len(batch), longest), dtype=torch.bool, device=input_device
            )
            for row, item in enumerate(batch):
                length = len(item.input_ids)
                input_ids[row, :length] = torch.tensor(
                    item.input_ids, dtype=torch.long, device=input_device
                )
                attention_mask[row, :length] = 1
                assistant_mask[row, :length] = torch.tensor(
                    item.assistant_mask, dtype=torch.bool, device=input_device
                )

            with torch.inference_mode():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
            predictions = logits[:, :-1, :].argmax(dim=-1)
            targets = input_ids[:, 1:]
            score_mask = assistant_mask[:, 1:] & attention_mask[:, 1:].bool()
            matches = predictions.eq(targets) & score_mask

            for row, item in enumerate(batch):
                scored = int(score_mask[row].sum().item())
                correct = int(matches[row].sum().item())
                if scored <= 0:
                    raise RuntimeError(
                        f"No scored tokens reached inference for line {item.line_number}"
                    )
                correct_total += correct
                scored_total += scored
                per_record.append(
                    {
                        "line_number": item.line_number,
                        "record_sha256": item.digest,
                        "input_tokens": len(item.input_ids),
                        "truncated_tokens": item.truncated_tokens,
                        "correct_tokens": correct,
                        "scored_tokens": scored,
                        "accuracy": correct / scored,
                        "tokenization_sha256": item.tokenization_digest,
                    }
                )
            del logits, predictions, targets, score_mask, matches
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if scored_total <= 0:
        raise RuntimeError(f"{label} evaluation produced zero scored tokens")
    return {
        "model_uri": model_uri,
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer": tokenizer_fingerprint(tokenizer),
        "correct_tokens": correct_total,
        "scored_tokens": scored_total,
        "accuracy": correct_total / scored_total,
        "records": per_record,
        "software": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        },
    }


def compare_results(
    legacy: dict[str, Any],
    migrated: dict[str, Any],
    max_accuracy_regression: float | None,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    legacy_records = legacy["records"]
    migrated_records = migrated["records"]
    if len(legacy_records) != len(migrated_records):
        reasons.append("models were not evaluated on the same number of records")
    else:
        for old, new in zip(legacy_records, migrated_records):
            if old["record_sha256"] != new["record_sha256"]:
                reasons.append(
                    "record ordering or content differs between model evaluations"
                )
                break
            if old["tokenization_sha256"] != new["tokenization_sha256"]:
                reasons.append(
                    "token IDs or assistant masks differ between model tokenizers/templates"
                )
                break
    if (
        legacy["tokenizer"]["vocabulary_sha256"]
        != migrated["tokenizer"]["vocabulary_sha256"]
    ):
        reasons.append("tokenizer vocabularies differ")
    if (
        legacy["tokenizer"]["chat_template_sha256"]
        != migrated["tokenizer"]["chat_template_sha256"]
    ):
        reasons.append("chat templates differ")
    if (
        legacy["tokenizer"]["special_tokens_sha256"]
        != migrated["tokenizer"]["special_tokens_sha256"]
    ):
        reasons.append("special-token mappings differ")
    reasons = list(dict.fromkeys(reasons))

    old_accuracy = float(legacy["accuracy"])
    new_accuracy = float(migrated["accuracy"])
    absolute_delta = new_accuracy - old_accuracy
    relative_delta = None if old_accuracy == 0 else absolute_delta / old_accuracy
    directly_comparable = not reasons

    required_minimum = None
    if max_accuracy_regression is not None:
        required_minimum = old_accuracy - max_accuracy_regression
    if not directly_comparable or max_accuracy_regression is None:
        verdict = "inconclusive"
    elif new_accuracy + 1e-15 >= required_minimum:
        verdict = "pass"
    else:
        verdict = "fail"

    comparison = {
        "directly_comparable": directly_comparable,
        "comparability_reasons": reasons,
        "legacy_accuracy": old_accuracy,
        "migrated_accuracy": new_accuracy,
        "absolute_accuracy_delta": absolute_delta,
        "relative_accuracy_delta": relative_delta,
        "max_accuracy_regression": max_accuracy_regression,
        "required_minimum_migrated_accuracy": required_minimum,
        "verdict": verdict,
    }
    risks = []
    if reasons:
        risks.append(
            "Token accuracy is tokenizer-dependent; the observed accuracies are not a "
            "strict paired comparison until tokenizer and chat serialization match."
        )
    if max_accuracy_regression is None:
        risks.append(
            "No predeclared maximum accuracy regression was supplied; measurements do "
            "not constitute an acceptance verdict."
        )
    return comparison, risks


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 1")
    return parsed


def run_evaluation(
    args: argparse.Namespace,
    *,
    mlflow_module: Any | None = None,
) -> dict[str, Any]:
    """Evaluate both checkpoints and log the evidence to the migration run."""
    config_path = args.config.expanduser().resolve()
    configured_source_uri, experiment_path = migration_config(config_path)
    legacy_uri = validate_model_uri(configured_source_uri, "configured source")
    migrated_uri = validate_model_uri(args.migrated_model_uri, "--migrated-model-uri")
    if legacy_uri == migrated_uri:
        raise ValueError(
            "Legacy and migrated model URIs must identify different versions"
        )

    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output_path}")

    mlflow, active_run, owns_active_run = _start_or_reuse_mlflow_run(
        experiment_path,
        mlflow_module=mlflow_module,
    )
    try:
        legacy_model_path = validate_full_checkpoint(args.legacy_model_path, "legacy")
        legacy_tokenizer_path = validate_tokenizer_path(
            args.legacy_tokenizer_path or legacy_model_path, "legacy"
        )
        migrated_model_path = validate_full_checkpoint(
            args.migrated_model_path, "migrated"
        )
        migrated_tokenizer_path = validate_tokenizer_path(
            args.migrated_tokenizer_path or migrated_model_path, "migrated"
        )
        records, dataset_digest = load_evaluation_records(args.eval_data)

        mlflow.set_tags(
            {
                "migration_stage": "token_accuracy_evaluation",
                "metric_name": "assistant_response_token_accuracy",
                "legacy_model_uri": legacy_uri,
                "migrated_model_uri": migrated_uri,
            }
        )
        mlflow.log_params(
            {
                "migration_experiment_path": experiment_path,
                "evaluation_data_path": str(args.eval_data.expanduser().resolve()),
                "evaluation_data_sha256": dataset_digest,
                "evaluation_record_count": len(records),
                "max_sequence_length": args.max_sequence_length,
                "truncation": args.truncation,
                "template_date": args.template_date,
                "batch_size": args.batch_size,
                "dtype": args.dtype,
                "device": args.device,
                "max_accuracy_regression": (
                    args.max_accuracy_regression
                    if args.max_accuracy_regression is not None
                    else ""
                ),
            }
        )

        shared = {
            "records": records,
            "max_sequence_length": args.max_sequence_length,
            "truncation": args.truncation,
            "template_date": args.template_date,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "device": args.device,
        }
        legacy = score_checkpoint(
            label="legacy",
            model_uri=legacy_uri,
            model_path=legacy_model_path,
            tokenizer_path=legacy_tokenizer_path,
            **shared,
        )
        migrated = score_checkpoint(
            label="migrated",
            model_uri=migrated_uri,
            model_path=migrated_model_path,
            tokenizer_path=migrated_tokenizer_path,
            **shared,
        )
        comparison, risks = compare_results(
            legacy, migrated, args.max_accuracy_regression
        )

        result = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mlflow": {
                "experiment_path": experiment_path,
                "experiment_id": str(active_run.info.experiment_id),
                "run_id": str(active_run.info.run_id),
            },
            "metric": {
                "name": "assistant_response_token_accuracy",
                "definition": (
                    "teacher-forced correct argmax next-token predictions divided by "
                    "scored assistant-turn tokens"
                ),
                "direction": "higher_is_better",
                "assistant_turn_terminators_included": True,
                "prompt_tokens_excluded": True,
            },
            "inputs": {
                "config_path": str(config_path),
                "evaluation_data_path": str(args.eval_data.expanduser().resolve()),
                "evaluation_data_sha256": dataset_digest,
                "record_count": len(records),
            },
            "settings": {
                "max_sequence_length": args.max_sequence_length,
                "truncation": args.truncation,
                "template_date": args.template_date,
                "batch_size": args.batch_size,
                "dtype": args.dtype,
                "device": args.device,
                "deterministic_teacher_forcing": True,
            },
            "models": {"legacy": legacy, "migrated": migrated},
            "comparison": comparison,
            "risks": risks,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        metrics = {
            "legacy_assistant_response_token_accuracy": float(legacy["accuracy"]),
            "migrated_assistant_response_token_accuracy": float(migrated["accuracy"]),
            "assistant_response_token_accuracy_delta": float(
                comparison["absolute_accuracy_delta"]
            ),
            "legacy_scored_tokens": int(legacy["scored_tokens"]),
            "migrated_scored_tokens": int(migrated["scored_tokens"]),
        }
        if comparison["relative_accuracy_delta"] is not None:
            metrics["assistant_response_token_accuracy_relative_delta"] = float(
                comparison["relative_accuracy_delta"]
            )
        mlflow.log_metrics(metrics)
        mlflow.set_tag("token_accuracy_verdict", comparison["verdict"])
        mlflow.log_artifact(str(output_path), artifact_path="validation")
    except BaseException:
        if owns_active_run:
            mlflow.end_run(status="FAILED")
        raise
    else:
        if owns_active_run:
            mlflow.end_run(status="FINISHED")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("migrate/config.yaml"))
    parser.add_argument("--legacy-model-path", type=Path, required=True)
    parser.add_argument("--legacy-tokenizer-path", type=Path)
    parser.add_argument("--migrated-model-uri", required=True)
    parser.add_argument("--migrated-model-path", type=Path, required=True)
    parser.add_argument("--migrated-tokenizer-path", type=Path)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("migrate/output/token-accuracy-evaluation.json"),
    )
    parser.add_argument("--max-sequence-length", type=_positive_integer, default=4096)
    parser.add_argument("--batch-size", type=_positive_integer, default=1)
    parser.add_argument("--truncation", choices=("error", "left"), default="error")
    parser.add_argument(
        "--template-date",
        default="26 Jul 2024",
        help="Fixed date passed to chat templates for deterministic serialization",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--max-accuracy-regression",
        type=_fraction,
        help="Optional allowed absolute decrease, such as 0.01 for one point",
    )
    result = run_evaluation(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
