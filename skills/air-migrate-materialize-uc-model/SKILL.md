---
name: air-migrate-materialize-uc-model
description: "Resolve portable Hugging Face model weights and tokenizers for Databricks AI Runtime, reusing source.existing_weights_volume_location when supplied or materializing exact Unity Catalog versions through AIR when it is blank. Applies source.use_existing_weights, checks system.ai before Hugging Face fallback, validates provided Volume checkpoints without loading tensors, and avoids OOM during large model handling. Use for continued training, base-model initialization, token-accuracy evaluation, UC artifact validation, or migration source resolution."
---

# Resolve UC Model Weights

Resolve the intended weight source locally. Reuse a configured Volume checkpoint without downloading it; otherwise perform every large UC artifact download and checkpoint materialization through AIR.

For training materialization, first read [the model-source resolution contract](references/model-source-resolution.md) and run `scripts/resolve_training_source.py` with the inspected base-model ID. This applies `source.use_existing_weights` and checks `system.ai` before selecting Hugging Face.

## Prerequisites

Require:

- A Databricks environment that can read configured Volumes; when materialization is required, an authenticated AIR CLI and profile named `DEFAULT`
- A valid authoritative `compute` block in `migrate/config.yaml`
- A required absolute `source.migration_experiment_path` in `migrate/config.yaml`, already ensured through the orchestrator helper before AIR submission
- Boolean `source.run_full_migration`; preserve it in source-resolution evidence and any minimal copied source config
- `source.existing_weights_volume_location` either blank or an absolute `/Volumes/<catalog>/<schema>/<volume>[/<checkpoint-path>]` string
- `USE CATALOG`, `USE SCHEMA`, and source model access when a UC artifact must be downloaded
- `READ VOLUME` on a supplied checkpoint, or `WRITE VOLUME` on an empty materialization destination
- PyYAML in the local environment that prepares the workload
- Current MLflow and PyYAML, supplied by the generated AI Runtime v5 environment

For continued training, also require a current inspection manifest whose `artifacts.final_model` is portable full weights and a plan with `mode: continue`. For validation, require exact source and target model versions from migration and registration evidence.

Reject adapter-only, serving-only, incomplete, or ambiguous artifacts. Do not merge adapters implicitly.

`source.run_full_migration: false` does not block input-weight preparation. Resolve provided weights or submit required source materialization through AIR, then hand off to generation and preflight validation. It does block later training, merge, registration, target materialization for evaluation, and model-quality evaluation. This skill must not reinterpret preparation-only mode as permission for those later stages.

## Reuse provided Volume weights

When `source.existing_weights_volume_location` is populated, require `source.use_existing_weights: true` and treat the configured directory as the complete Hugging Face model and tokenizer checkpoint. Set `requires_materialization: false`; do not generate or submit a UC-download AIR workload.

On Databricks compute where `/Volumes` is mounted, structurally validate the supplied directory without downloading, copying, loading, or saving tensor contents:

```bash
python3 skills/air-migrate-materialize-uc-model/scripts/materialize_uc_model.py \
  --config migrate/config.yaml \
  --output-dir /Volumes/<catalog>/<schema>/<volume>[/<checkpoint-path>] \
  --reuse-existing \
  --require-volume
```

Require the command's `acquisition: provided_volume`, `download_performed: false`, `model_version_checked: false`, exact configured destination, full-weight inventory, and complete tokenizer. This branch performs no MLflow/UC query. Record the returned `model_path` and `tokenizer_path`; both must equal the configured directory. If validation fails, stop instead of falling back to UC download. The path is user-supplied provenance for the configured UC version, not proof that its bytes came from that version.

Record the bypass before generation:

```yaml
materialization:
  status: not_required
  reason: existing_weights_volume_location
  source_resolution:
    model_source: existing_uc
    source_model_uri: models:/catalog.schema.model/1
    existing_weights_volume_location: /Volumes/catalog/schema/volume/checkpoint
    requires_materialization: false
  provided_weights:
    validation_status: current
    model_version_checked: false
    model_path: /Volumes/catalog/schema/volume/checkpoint
    tokenizer_path: /Volumes/catalog/schema/volume/checkpoint
    weight_format: safetensors
    inventory: []
```

## Prepare the AIR workload

Only when `source.existing_weights_volume_location` is blank, generate a self-contained workload for continued training from the configured legacy model:

```bash
python3 skills/air-migrate-materialize-uc-model/scripts/prepare_air_materialization.py \
  --config migrate/config.yaml \
  --output-dir /Volumes/<catalog>/<schema>/<volume>/source-model-v<version> \
  --workload-dir migrate/output/air_materialization
```

For artifact or token-accuracy validation, generate a distinct workload and pin the exact version explicitly:

```bash
python3 skills/air-migrate-materialize-uc-model/scripts/prepare_air_materialization.py \
  --config migrate/config.yaml \
  --purpose validation \
  --model-uri models:/<catalog>.<schema>.<model>/<version> \
  --output-dir /Volumes/<catalog>/<schema>/<volume>/validation/model-v<version> \
  --workload-dir migrate/output/air_materialization-model-v<version>
```

For a base model resolved from `system.ai`, use `--purpose base_model_initialization` with the pinned `--model-uri` returned by source resolution.

The preparer:

1. Reads and validates `source`, its `run_full_migration` gate and `migration_experiment_path`, plus authoritative compute from `migrate/config.yaml`.
2. Copies only the materializer and a minimal source config that preserves the gate into a new workload directory.
3. Generates `materialize.yaml` for AI Runtime v5 with current MLflow and PyYAML dependencies and sets `experiment_name` to the exact migration experiment path.
4. Runs one Python process on AIR node rank zero without `torchrun`.
5. Sets `max_retries: 0` because a partial nonempty destination requires explicit review.
6. Writes the inventory to `<output-dir>/materialization.json`.

Use `--artifact-uri`, `--checkpoint-subpath`, or `--tokenizer-subpath` on the preparer only after inspection establishes those exact values. Do not use the preparer for the configured source when the config supplies existing Volume weights; explicit target or `system.ai` model URIs may still be materialized when required. The preparer never accepts existing destination content.

## Submit and verify

Run from the generated workload directory so its `code_source.snapshot.root_path: .` includes only the generated files:

```bash
python3 skills/air-migrate-migrate-fmt-model/scripts/ensure_migration_experiment.py \
  --config migrate/config.yaml --profile DEFAULT
cd migrate/output/air_materialization
COPYFILE_DISABLE=1 air run --dry-run --file materialize.yaml -p DEFAULT
COPYFILE_DISABLE=1 air run --file materialize.yaml -p DEFAULT --watch
air get run <materialization-air-run-id> -p DEFAULT --json
```

AIR dry-run stages files but does not submit a Jobs run. Record the numeric AIR run ID from the live submission. Require terminal `SUCCESS`, then read the persistent inventory:

```bash
databricks fs cat dbfs:/Volumes/<catalog>/<schema>/<volume>/source-model-v<version>/materialization.json -p DEFAULT
```

Do not mark materialization current based only on a submitted run, partial files, or inventory printed before terminal success.

The AIR materializer resolves the exact registered version, verifies it is ready, downloads through MLflow with the UC registry, locates a complete Hugging Face causal-LM checkpoint and tokenizer, validates shard indexes without loading tensor contents, and emits paths, provenance, formats, sizes, and file inventory. Read [the portable checkpoint contract](references/portable-checkpoint-contract.md) when interpreting failures.

The materialized UC Volume remains the durable source of truth. Do not replace it with an AIR node-local path: downstream generated training modules copy Volume-backed model/tokenizer inputs into their ephemeral node-local cache once per node and load from that runtime copy. The materialization inventory supplies the source-byte evidence needed to plan local-cache capacity.

Prefer the registered `models:/catalog.schema.model/version` artifact. If inspection proves that it is a serving package while portable weights live in the originating run, pass the observed `runs:/<run_id>/<path>` as `--artifact-uri`. Do not guess an artifact path or use an unrelated run.

## Update the migration manifest

For the blank-path branch, only after AIR reports `SUCCESS` and the persisted inventory is structurally valid, record:

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
  inventory_path: /Volumes/catalog/schema/volume/source-model-v1/materialization.json
  execution:
    engine: databricks_air
    profile: DEFAULT
    air_run_id: "123456789"
    air_status: SUCCESS
    workload_file: migrate/output/air_materialization/materialize.yaml
    migration_experiment_path: /Shared/fmt-migration
    mlflow_experiment_id: "123456789"
    compute:
      num_accelerators: 8
      accelerator_type: GPU_8xH100
```

Set `generation.status` and `validation.status` to `stale` when the source version, materialized paths, artifact URI, inventory, or AIR execution changes. For validation materializations, preserve source and target run IDs, URIs, paths, and inventories in validation evidence rather than replacing the training materialization section.

## Handoff to generation

Require `air-migrate-generate-air-job` to set:

- `training_config.use_existing_weights` from `migrate/config.yaml`
- `training_config.existing_weights_volume_location` from pinned resolution
- `training_config.model_source` and `training_config.source_model_uri` from pinned resolution
- `training_config.model_name` and `training_config.tokenizer_path` to the validated provided Volume path when configured
- Otherwise, `training_config.model_name` to `materialization.model_path` and `training_config.tokenizer_path` to `materialization.tokenizer_path`
- `training_config.local_model_cache_dir` and `training_config.local_model_cache_copy_workers` from the selected template, with per-node capacity checked against the persistent inventory
- AIR `experiment_name` and `training_config.experiment_path` both set to the exact configured `source.migration_experiment_path`

Remove the AIR `HF_TOKEN` secret only when every model and tokenizer reference is local and no other Hugging Face download is required. Preserve source URI and materialization paths in MLflow tags or migration metadata.

## Guardrails

- Do not represent continued training as reproduction or parity retraining.
- Do not point Transformers or TRL directly at `models:/...`; resolve a Volume checkpoint and validate it first.
- Do not invoke `materialize_uc_model.py` outside the generated AIR workload for a download. The no-download `--reuse-existing` structural check is the only exception.
- Do not use `torchrun`; materialization is a single-process I/O task.
- Do not overwrite or automatically delete existing Volume or workload content.
- Do not load tensor contents during structural materialization validation; defer full model smoke loading to migration validation on suitable compute.
- Do not download weights merely for inspection. In the orchestrated workflow, both execution modes authorize required source materialization; target materialization for evaluation requires full-migration mode. For direct stage invocation, require equivalent authorization in the active request.
- Do not write durable materialization output, inventory, checkpoints, or registration artifacts into the downstream node-local input cache. It is ephemeral and may be absent in the next AIR job.
- Do not treat `run_full_migration: false` as a materialization prohibition or as authorization for any stage after preflight validation.
- Never emit credentials or secret values.
