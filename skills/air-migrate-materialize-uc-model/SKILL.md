---
name: air-migrate-materialize-uc-model
description: "Resolve and materialize portable Hugging Face model weights and tokenizer files for AI Runtime training or validation. Applies source.use_existing_weights, selects the configured legacy UC version or checks system.ai before Hugging Face fallback, and stages exact UC model versions in a shared Volume. Use for continued training, base-model initialization, token-accuracy evaluation, or UC artifact validation."
---

# Materialize UC Model

Resolve the intended weight source, then download and validate any registered model artifact before using it as an Axolotl initialization checkpoint or validation input. For evaluation, use `purpose: validation` and do not change the migration mode.

Before training materialization, read [the model-source resolution contract](references/model-source-resolution.md) and run `scripts/resolve_training_source.py` with the recovered base-model ID. This applies `source.use_existing_weights` and checks `system.ai` before selecting Hugging Face.

## Prerequisites

For continued training, require:

- A current inspection manifest whose `artifacts.final_model` is portable full weights
- A plan with `mode: continue`
- `USE CATALOG`, `USE SCHEMA`, and model access for the source
- `WRITE VOLUME` on a persistent destination accessible to every AIR worker
- MLflow and PyYAML in the execution environment

For validation, instead require exact versioned source and target model URIs from the migration config and registration evidence, plus persistent destinations accessible to evaluation compute.

Reject adapter-only, serving-only, incomplete, or ambiguous artifacts. Do not merge adapters implicitly.

## Materialize

Run the bundled script on Databricks compute so `/Volumes/...` is mounted:

```bash
python3 skills/air-migrate-materialize-uc-model/scripts/materialize_uc_model.py \
  --config migrate/config.yaml \
  --output-dir /Volumes/<catalog>/<schema>/<volume>/source-model-v<version> \
  --require-volume
```

The script defaults to `--purpose continue_training`. For artifact or token-accuracy validation, materialize an exact version without implying continued training:

```bash
python3 skills/air-migrate-materialize-uc-model/scripts/materialize_uc_model.py \
  --purpose validation \
  --model-uri models:/<catalog>.<schema>.<model>/<version> \
  --output-dir /Volumes/<catalog>/<schema>/<volume>/validation/model-v<version> \
  --require-volume
```

For a base model resolved from `system.ai`, pass its pinned URI with `--purpose base_model_initialization` and `--model-uri`.

The script:

1. Resolves `models:/<catalog>.<schema>.<model>/<version>`.
2. Verifies that the registered version is ready.
3. Uses `mlflow.artifacts.download_artifacts` with the UC registry.
4. Locates a complete Hugging Face causal-language-model checkpoint and tokenizer.
5. Validates sharded weight indexes without loading weights into memory.
6. Emits JSON containing the model path, tokenizer path, source URI, run ID, format, and file inventory.

Use `--checkpoint-subpath` or `--tokenizer-subpath` only to resolve an explicitly reviewed ambiguity. Use `--reuse-existing` only to inspect a prior download after verifying its provenance. The destination is otherwise required to be empty.

By default, download the selected `models:/` URI. If inspection proves that the registered artifact is a serving package while portable weights live in the originating run, pass that observed URI with `--artifact-uri runs:/<run_id>/<path>`. Do not guess an artifact path or use an unrelated run.

Read [the portable checkpoint contract](references/portable-checkpoint-contract.md) when interpreting failures or updating the manifest.

## Update the migration manifest for continued training

Record the script result under `materialization`:

```yaml
materialization:
  status: current
  purpose: continue_training
  mode: continue
  source_model_uri: models:/catalog.schema.model/1
  artifact_uri: models:/catalog.schema.model/1
  destination: /Volumes/catalog/schema/volume/source-model-v1
  model_path: /Volumes/catalog/schema/volume/source-model-v1/model
  tokenizer_path: /Volumes/catalog/schema/volume/source-model-v1/model
  weight_format: safetensors
  source_run_id: abc123
```

Set `generation.status` and `validation.status` to `stale` when the materialized path or source version changes.

For validation materializations, keep their source/target URIs, paths, and file inventories in validation evidence instead of replacing the continue-mode `materialization` section.

## Handoff to generation

Require `air-migrate-generate-air-job` to set:

- `training_config.use_existing_weights` from `migrate/config.yaml`
- `training_config.model_source` and `training_config.source_model_uri` from the pinned resolution
- `training_config.model_name` to `materialization.model_path`
- `training_config.tokenizer_path` to `materialization.tokenizer_path`

Remove the AIR `HF_TOKEN` secret only when every model/tokenizer reference is local and the generated workload requires no other Hugging Face downloads. Preserve source URI and materialization paths in MLflow tags or migration metadata.

## Guardrails

- Do not represent continued training as reproduction or parity retraining.
- Do not point Axolotl directly at `models:/...`; materialize and validate first.
- Do not overwrite existing Volume content.
- Do not download large model weights merely for inspection; materialize only after a UC-backed training source is selected or model evaluation is authorized.
- Never emit credentials or secret values.
