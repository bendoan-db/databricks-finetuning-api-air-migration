---
name: air-migrate-materialize-uc-model
description: "Materialize portable Hugging Face model weights and tokenizer files from a Unity Catalog registered model version for continued training on Databricks AI Runtime. Use when a migration plan selects continue mode, an Axolotl workload must initialize from legacy fine-tuned weights instead of the original base model, or a coding assistant needs to validate and stage UC model artifacts in a shared Volume."
---

# Materialize UC Model

Download and validate a registered model artifact before using it as an Axolotl initialization checkpoint. Treat this as the bridge between `plan.mode: continue` and AIR job generation.

## Prerequisites

Require:

- A current inspection manifest whose `artifacts.final_model` is portable full weights
- A plan with `mode: continue`
- `USE CATALOG`, `USE SCHEMA`, and model access for the source
- `WRITE VOLUME` on a persistent destination accessible to every AIR worker
- MLflow and PyYAML in the execution environment

Reject adapter-only, serving-only, incomplete, or ambiguous artifacts. Do not merge adapters implicitly.

## Materialize

Run the bundled script on Databricks compute so `/Volumes/...` is mounted:

```bash
python3 skills/air-migrate-materialize-uc-model/scripts/materialize_uc_model.py \
  --config migrate/config.yaml \
  --output-dir /Volumes/<catalog>/<schema>/<volume>/source-model-v<version> \
  --require-volume
```

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

## Update the migration manifest

Record the script result under `materialization`:

```yaml
materialization:
  status: current
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

## Handoff to generation

Require `air-migrate-generate-air-job` to set:

- `training_config.model_name` to `materialization.model_path`
- `training_config.tokenizer_path` to `materialization.tokenizer_path`

Remove the AIR `HF_TOKEN` secret only when every model/tokenizer reference is local and the generated workload requires no other Hugging Face downloads. Preserve source URI and materialization paths in MLflow tags or migration metadata.

## Guardrails

- Do not represent continued training as reproduction or parity retraining.
- Do not point Axolotl directly at `models:/...`; materialize and validate first.
- Do not overwrite existing Volume content.
- Do not download large model weights merely for inspection; materialize only after continue mode is selected.
- Never emit credentials or secret values.
