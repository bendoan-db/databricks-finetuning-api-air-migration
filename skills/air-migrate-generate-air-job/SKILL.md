---
name: air-migrate-generate-air-job
description: "Generate the canonical Databricks AI Runtime migration workload at migrate/output/air_workload from the best-fitting approved TRL SFTTrainer template. Supports unquantized LoRA with DDP or FSDP and full-weight FSDP. Use when creating or refreshing model-specific train.yaml, thin train.py, helper_utils.py, training_utils.py, optional LoRA merge.py, 01_runner.py, and 02_register_uc.py modules for a legacy Foundation Model Fine-tuning API migration. Enforces template provenance, safe customization, configuration parity, explicit Unity Catalog registration, and static/AIR dry-run validation."
---

# Generate AI Runtime Job

Create the canonical runnable AIR training and registration modules at `migrate/output/air_workload` from the approved TRL templates. Do not synthesize workload files from scratch, and never run a source template directly as the migration.

## Prerequisites

Read `migrate/output/migration-manifest.yaml` and require `plan.status: current`, a selected `plan.recipe`, its exact `plan.template`, and recorded model/template compatibility evidence. If the plan is missing or stale, return to `air-migrate-plan-air-training`. If the plan uses `recipe: custom`, stop and report the template capability that must be added. If it uses `recipe: none`, skip this skill because the migration is a repackage operation. Never silently choose or change the recipe during generation.

Require `plan.input_model.use_existing_weights` to equal `source.use_existing_weights`. For `existing_weights_volume`, require the nonblank config path to equal `plan.input_model.existing_weights_volume_location`, `materialization.status: not_required` with the matching reason, current structural validation under `materialization.provided_weights`, and model/tokenizer paths equal to the configured directory. For `materialized_uc_model` or `materialized_system_ai`, require `materialization.status: current` from `air-migrate-materialize-uc-model`, plus absolute worker-accessible `model_path` and `tokenizer_path`. Reject `models:/` URIs and local driver-only paths. For `hugging_face`, require the recovered repository ID and a successful recorded `system.ai` lookup with no match.

Require boolean `source.run_full_migration` and a matching manifest `execution_policy`. Generation is allowed in both modes. In `preparation_only`, create the same complete runnable workload, perform static validation and AIR dry-runs when available, then hand back to the orchestrator without submitting training, merge, registration, or evaluation work.

Require `migrate/config.yaml.compute`, manifest `requested_compute`, and `plan.compute.requested` to match exactly. Require `plan.compute.resolved` to preserve the same values and its feasibility verdict to be compatible. Reject unsupported AIR types, count/type mismatches, and a plan that silently substituted template compute.

Require `migrate/config.yaml.source.migration_experiment_path`, manifest `migration_experiment.path`, and `plan.migration_experiment_path` to match exactly. Require current evidence that the experiment exists or was created. Reject a generated workload or historical run from another experiment.

Read [the template catalog](references/template-catalog.md) before materializing a workload.

## Materialize the selected template

Locate the repository root containing `air_templates/`. Run:

```bash
python3 skills/air-migrate-generate-air-job/scripts/materialize_air_template.py \
  --recipe <manifest-plan-recipe>
```

The default and required migration-flow destination is `migrate/output/air_workload`. The script reads `migrate/config.yaml`, validates its `compute` block and `source.migration_experiment_path`, copies both into the generated `train.yaml`, and synchronizes `torchrun --nproc_per_node`. Use `--config` only when testing an alternate migration config. Use `--template-root` only when the template directory is outside the detected repository root. The script refuses a nonempty destination and verifies that the source contains exactly the required workload files. Use `--output-dir` only for isolated utility tests, never to relocate the canonical migration workload.

The recipe mapping is exact:

- `trl_lora` -> `air_templates/trl_lora`
- `trl_lora_fsdp` -> `air_templates/trl_lora_fsdp`
- `trl_full_fsdp` -> `air_templates/trl_full_fsdp`

If no approved template fits the inspected model, stop and request a template extension. Do not use the nearest recipe.

## Customize the copy

Preserve the template's structure, launcher contract, distributed initialization, framework configuration construction, validations, MLflow behavior, and checkpoint semantics. Make the smallest necessary edits.

Customize:

- Model name/revision and tokenizer/chat template
- `use_existing_weights` copied unchanged from `migrate/config.yaml`
- `existing_weights_volume_location` copied for provenance when populated and null otherwise
- `model_source` mapped exactly as `existing_weights_volume -> existing_uc`, `materialized_uc_model -> existing_uc`, `materialized_system_ai -> system_ai`, or `hugging_face -> hugging_face`
- The pinned `source_model_uri`, or null only for Hugging Face
- For `existing_weights_volume`, set `model_name` and `tokenizer_path` to the validated configured directory; for either materialized source, use the materialization paths; for Hugging Face, use the recovered repository ID
- Train/eval UC Volume paths and data-format settings
- Adapter, merged-model, and checkpoint paths; experiment name; and MLflow run name
- `registered_model_name` as the resolved `<catalog>.<schema>.<model>` target
- Databricks secret references and environment variable names, never values
- `compute.num_accelerators` and `compute.accelerator_type` copied exactly from `migrate/config.yaml`, with the same count used by `torchrun --nproc_per_node`
- Runtime settings justified by the plan
- `experiment_name` and `training_config.experiment_path`, both copied exactly from `migrate/config.yaml.source.migration_experiment_path`
- `local_model_cache_dir` and `local_model_cache_copy_workers` from the selected template, customized only when the resolved AIR node-local storage or source-checkpoint size requires it
- Recovered or intentionally changed hyperparameters
- Runner and registration notebook titles, defaults, explanatory text, and configuration so both remain equivalent to `train.yaml`

Prefer YAML parameter changes. Keep `train.py` limited to `run_training` and its CLI entry point. Put YAML loading, value coercion, path/source validation, and configuration translation in `helper_utils.py`. Put distributed staging, dataset/model construction, trainer configuration, MLflow, merge, and registration operations in `training_utils.py`. Extend those helpers only when YAML cannot express the required behavior. Keep every workload module consistent.

Complete all customization in `migrate/output/air_workload`. The generated directory must retain the selected template's modular layout:

- `train.yaml`: canonical AIR workload and model-specific configuration
- `train.py`: thin model-specific TRL training entry point containing only `run_training` and `main`
- `helper_utils.py`: YAML loading, value coercion, formatting, path/source/FSDP validation, and configuration translation
- `training_utils.py`: distributed/runtime staging, dataset/model construction, trainer configuration, MLflow, checkpoint, merge, and registration operations
- `merge.py`: LoRA-only AIR entry point that calls `training_utils.merge_peft_model` without retraining
- `01_runner.py`: interactive training notebook; LoRA recipes also produce the merged full checkpoint here
- `02_register_uc.py`: dual-mode post-training notebook that reads `train.yaml`, accepts a `mlflow_run_id` widget interactively or `--mlflow-run-id` as an AIR Python command, resumes that run, and registers the portable full checkpoint in Unity Catalog

Keep registration dependencies in `train.yaml.environment.dependencies`; AIR executes the notebook source as Python and ignores its `%pip` cells. Guard `dbutils` calls in script mode. Keep each `# MAGIC %md` or `# MAGIC %pip` directive first in its command cell; place Python-only directives such as `# ruff: noqa` in a Python cell so workspace notebook parsing remains valid.

Preserve the template's node-local input-staging contract. `/Volumes/...` model and tokenizer paths are durable source/provenance locations, but Transformers must load their runtime copies from `local_model_cache_dir`. Keep one bounded parallel copy per source directory per AIR node, node-local rank coordination, source/file-size validation, an atomic completion marker, cache reuse, and `local_files_only=True` for staged loads. Keep staging active for PEFT merge-time base-model reloads. Remote Hugging Face IDs remain remote references and must not be rewritten to this cache.

Require `local_model_cache_dir` to be an absolute ephemeral node-local path, never a UC Volume, DBFS, workspace path, training output, adapter output, merged output, or checkpoint directory. Require `local_model_cache_copy_workers` in the template-supported range. Size every AIR node for one complete model/tokenizer source copy plus the template's safety reserve; separate tokenizer directories require additional capacity. The cache is disposable acceleration state, not a checkpoint or migration artifact. Preserve the template's MLflow staging provenance and timing/throughput metrics.

Do not edit the source template while customizing a migration.

Remove the `HF_TOKEN` AIR secret and notebook requirement for `existing_uc` and `system_ai` only when the model and tokenizer are validated local/Volume paths—provided or materialized—and no other Hugging Face download is required. Retain it for gated Hugging Face sources.

Do not remove recipe invariants:

- TRL LoRA DDP: `SFTTrainer` plus PEFT, an unquantized bf16 base replica per worker, DDP synchronization, assistant-only loss, no 4-bit or 8-bit loading, and merged full-model registration.
- TRL LoRA FSDP: `SFTTrainer` plus PEFT, an unquantized bf16 sharded base, PEFT-aware FSDP wrapping, rank-0-efficient loading, FSDP activation checkpointing, collective adapter save, assistant-only loss, and merged full-model registration.
- TRL full FSDP: `SFTTrainer` without PEFT or quantization, FSDP full sharding, rank-0-efficient loading, FSDP activation checkpointing, assistant-only loss, collective portable full-state save, and full-model registration.

## Validate before submission

1. Compile `train.py`, `helper_utils.py`, `training_utils.py`, `01_runner.py`, and `02_register_uc.py`, plus `merge.py` for LoRA recipes. Assert `train.py` defines only `run_training` and `main`, imports configuration functions from colocated `helper_utils.py`, and imports runtime/training functions from colocated `training_utils.py`.
2. Parse YAML and assert all required workload and training fields.
3. Exercise configuration loading/translation without downloading the model when possible.
4. Search for placeholder paths, hard-coded tokens, source-template experiment names, and inconsistent values across YAML and both notebooks. Confirm any `@distributed` launch derives its GPU count and A10/H100 type from generated `train.yaml` rather than a template constant.
5. Assert that every required generated module is under `migrate/output/air_workload` and traces back to `plan.template`: six core files for full FSDP and those files plus `merge.py` for LoRA.
6. Assert generated `compute` exactly equals `migrate/config.yaml.compute`, the accelerator type's embedded count agrees with `num_accelerators`, and the `torchrun` process count agrees with both.
7. Assert `experiment_name` and `training_config.experiment_path` are identical to the configured migration experiment. Ensure/create it through MLflow before any AIR dry-run or live submission, and verify any reused MLflow run belongs to its experiment ID.
8. Run the training dry-run from `migrate/output/air_workload` when the AIR CLI is available. Also dry-run the registration command without changing compute: `air run --dry-run --file train.yaml --override 'command=python $CODE_SOURCE_PATH/02_register_uc.py --mlflow-run-id DRY_RUN_ONLY'`. AIR dry-run skips Jobs submission but may still stage files in the workspace; record that side effect in preflight evidence.
9. Confirm output type matches the plan: full model for full FSDP; adapter training output plus a separate merged full checkpoint for every LoRA recipe.
10. Assert the YAML source tuple is internally consistent: `(use_existing_weights=true, populated existing_weights_volume_location, existing_uc, configured source URI, identical validated Volume paths)`, `(true, null location, existing_uc, configured URI, materialized paths)`, `(false, null location, system_ai, pinned system URI, materialized paths)`, or `(false, null location, hugging_face, null URI, remote repository ID)`.
11. For provided or materialized sources, verify every worker can read the model/tokenizer and that the selected framework uses those exact paths for model and tokenizer loading. For provided weights, also require both generated paths to equal `source.existing_weights_volume_location` and confirm no materialization workload was generated.
12. For every `/Volumes/...` model/tokenizer input, confirm the generated code retains the configured Volume path as provenance, stages it to the configured node-local cache once per node under a file lock, publishes only an atomically completed copy, and loads the staged path in local-only mode. Confirm cache-root validation, bounded copy workers, free-space reserve, per-file size checks, cache-hit behavior, and MLflow copy/lock/throughput evidence. Confirm Hugging Face references bypass this Volume prefetch.
13. Confirm no durable output field, resume path, adapter path, merged-model path, full checkpoint, MLflow registration input, or manifest artifact points under `local_model_cache_dir`. Confirm every AIR node has capacity for the source checkpoint and tokenizer plus reserve.
14. Confirm `02_register_uc.py` reads `registered_model_name` from YAML, supports both widget and AIR CLI input, guards notebook-only APIs, logs input-source provenance and the final artifact to the training MLflow run, and uses the `databricks-uc` registry. It must execute once rather than through `torchrun`. LoRA registration must consume only a safely merged full checkpoint; full-FSDP registration must exclude intermediate checkpoints and optimizer state.
15. For LoRA, confirm `merge.py` calls `training_utils.merge_peft_model`, can reuse the completed adapter without entering training, and returns the merged checkpoint path. Registration must reuse an already valid merged checkpoint rather than overwrite it.
16. If `source.run_full_migration` is false, stop after preflight regardless of prior run state or a runnable generated workload. If true, treat the config as explicit pipeline authorization for later execution; this generation skill still does not submit those live stages itself.

## Record provenance

Update `generation` in the manifest:

```yaml
generation:
  status: current
  recipe: trl_full_fsdp
  template_path: air_templates/trl_full_fsdp
  output_path: migrate/output/air_workload
  files: [train.yaml, train.py, helper_utils.py, training_utils.py, 01_runner.py, 02_register_uc.py]
  runnable: true
  run_from: migrate/output/air_workload
  compute:
    num_accelerators: 8
    accelerator_type: GPU_8xH100
  migration_experiment_path: /Shared/fmt-migration
  input_staging:
    volume_inputs_prefetched: true
    cache_dir: /tmp/air-model-cache
    copy_workers: 8
    cache_scope: node_ephemeral
    durable_outputs_under_cache: false
    capacity_check: pass
  customized_fields: []
  input_model:
    use_existing_weights: true
    source: materialized_uc_model
    source_model_uri: models:/catalog.schema.model/1
    existing_weights_volume_location: null
    model_path: /Volumes/catalog/schema/volume/source-model-v1/model
    tokenizer_path: /Volumes/catalog/schema/volume/source-model-v1/model
  validations: []
```

For preparation-only mode, set `validation.status: current`, `validation.scope: preflight`, and `validation.migration_complete: false` only after the preflight contract passes. A runnable workload means the files are ready to submit; it does not mean execution is authorized or the migration is complete.

Set `validation.status: stale` whenever generated workload content changes.

The generated workload—not `air_templates/` and not the migration manifest—is the executable migration artifact. In full-migration mode, train with `air run --file migrate/output/air_workload/train.yaml` or the generated `01_runner.py`. For AIR LoRA training, run `merge.py` next. Then submit `02_register_uc.py` through AIR with the training MLflow run ID when checkpoint size makes notebook compute unsafe, and verify both AIR `SUCCESS` and UC version `READY`. In preparation-only mode, stop after preflight.
