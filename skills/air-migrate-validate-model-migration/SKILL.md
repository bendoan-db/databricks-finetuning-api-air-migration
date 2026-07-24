---
name: air-migrate-validate-model-migration
description: "Validate a migration from the legacy Databricks Foundation Model Fine-tuning API to Databricks AI Runtime by checking workload configuration, executing authorized smoke or full runs, verifying artifacts and Unity Catalog registration, and comparing source and migrated models—including assistant response-token accuracy—on shared evaluation data. Use for preflight checks, parity testing, regression analysis, migration acceptance, or promotion decisions."
---

# Validate Model Migration

Produce an auditable verdict against criteria chosen before training. Separate workload validity, training health, artifact integrity, and model quality.

## Select validation scope

Require boolean `source.run_full_migration` and a matching manifest `execution_policy` before validating:

- When false, perform static preflight only. Do not submit training or merge commands, register a model, materialize a target model for evaluation, invoke `air-migrate-compare-token-accuracy`, or claim migration completion. Set manifest `execution.status: not_authorized` with reason `preparation_only`, write a scoped `preflight` verdict, and set `migration_complete: false`.
- When true, perform every applicable layer below and write a `final` verdict. Treat the config as pipeline authorization for smoke/full execution, registration, evaluation, and final validation.

Required source materialization belongs to preparation and is allowed in either scope. AIR dry-run is also allowed in preflight, but record that it stages workspace files even though it does not submit a Jobs run.

## Inputs

Require:

- `migrate/output/migration-manifest.yaml` with current plan and generation sections
- Generated AIR workload at `migrate/output/air_workload` and its dry-run evidence
- Source registered model and resolved target
- Shared evaluation dataset and acceptance criteria
- Current provided-Volume validation or materialization provenance when `plan.mode: continue`

If acceptance criteria are absent, report measurements without inventing a pass threshold.

## Validation layers

1. **Static preflight**
   - Require `train.yaml`, `train.py`, `helper_utils.py`, `training_utils.py`, `01_runner.py`, and `02_register_uc.py` under `migrate/output/air_workload`, plus `merge.py` for LoRA; compile every Python module, assert `train.py` defines only `run_training` and `main` and imports both helper modules, parse YAML, validate selected-template provenance and model compatibility, compare YAML and both notebook settings, inspect dependencies/secrets, and run AIR dry-run when available.
   - Reject a validation target under `air_templates/`; only the generated workload is runnable migration evidence.
   - Confirm dataset paths/schema, tokenizer/chat format, assistant-only masking, effective batch, output kind, checkpoint contract, and target location.
   - Confirm generated `train.yaml.compute` exactly matches `migrate/config.yaml.compute` and the manifest's requested, resolved, and generated compute. Confirm `num_accelerators`, the accelerator type's embedded count, `torchrun --nproc_per_node`, and the runner's YAML-derived `@distributed` launch are consistent.
   - Confirm the configured, manifest, planned, materialization, and generated experiment paths match exactly. Require generated `experiment_name` and `training_config.experiment_path` to equal `migrate/config.yaml.source.migration_experiment_path`, and require current create/existence evidence with its MLflow experiment ID before any AIR dry-run.
   - Confirm `use_existing_weights` and `existing_weights_volume_location` match the migration config and the generated source tuple is valid. For true with a populated path, verify both generated model/tokenizer paths equal that structurally validated directory and no materialization was claimed. For true with a blank path, verify the configured legacy URI and materialized files. For false, verify the pinned `system.ai` resolution or documented no-match Hugging Face fallback.
   - For every `/Volumes` model/tokenizer input, confirm `local_model_cache_dir` is an absolute ephemeral node-local path outside Volumes, DBFS, workspace files, and all durable output trees; confirm `local_model_cache_copy_workers` is supported and the per-node capacity plan covers the complete source plus reserve. Confirm ranks share one locked copy per node, only an atomically completed cache is reused, copied sizes are checked, staged Transformers loads are local-only, and remote Hugging Face IDs bypass the Volume prefetch.
   - Confirm the configured Volume paths remain source provenance while runtime load references point to the staged cache. Require MLflow parameters/metrics for source and runtime paths, cache hit, file/byte counts, lock wait, copy duration, and throughput. Reject any adapter, checkpoint, merged/full output, resume path, registration input, or manifest artifact under the cache root.
   - For `trl_lora`, confirm the base is unquantized, each worker holds one bf16 replica, only adapter parameters are trainable, and the merge produces a portable full checkpoint.
   - For `trl_lora_fsdp`, confirm the base is unquantized, only adapter parameters are trainable, PEFT-aware FSDP full sharding and rank-0-efficient loading are active, FSDP activation checkpointing replaces Trainer gradient checkpointing, and every rank participates in adapter saves before the safe merge.
   - For `trl_full_fsdp`, confirm PEFT and quantization are absent, FSDP full sharding and rank-0-efficient loading are active, FSDP activation checkpointing replaces Trainer gradient checkpointing, and every rank participates in the final `FULL_STATE_DICT` save.
   - Confirm `02_register_uc.py` reads `registered_model_name` from YAML, requires the training MLflow run ID, uses the UC registry, and registers only the portable full checkpoint. Require both the interactive `mlflow_run_id` widget and AIR `--mlflow-run-id` path, guarded `dbutils` calls, script-mode dependencies in `train.yaml`, and magic directives at the start of their notebook command cells.
   - For LoRA, confirm `merge.py` calls `training_utils.merge_peft_model` and materializes the full checkpoint without entering training. Dry-run the AIR registration command with the generated compute unchanged and one Python process rather than `torchrun`.
   - Do not treat completion of training, creation of a logged model, or creation of the UC model container as registration evidence.
2. **Execution health**
   - Run this layer only for `run_full_migration: true`. Run a bounded smoke workload first. Confirm distributed initialization, data loading, forward/backward pass, metric logging, checkpoint write/read, and clean worker completion.
   - Inspect logs for OOMs, NaNs, rank divergence, hidden retries, dropped configuration fields, and unexpectedly unused parameters.
   - Inspect first-load staging metrics on every node. Require exactly one completed source copy per node, cache reuse by other local ranks, no partial-cache load, and sufficient local disk. Treat a cache miss in a new AIR job as expected because node-local storage is ephemeral.
   - Record the numeric AIR run ID separately from the rank-zero training MLflow run ID. For large full checkpoints, run registration through AIR on the configured training compute and inspect host-memory and temporary-storage failures as registration-capacity failures, not training failures.
   - Resolve every materialization, training, merge, registration, smoke, and evaluation AIR/MLflow run and require its experiment ID to equal the configured migration experiment ID. Treat a mismatch as failure even when the run itself succeeded.
   - If registration fails after training succeeds, preserve the successful training run and merged checkpoint. Retry registration without retraining and retain all failed run IDs and partial logged-model evidence.
3. **Artifact integrity**
   - In preparation-only mode, limit this layer to the resolved input checkpoint and generated-workload contracts. Target checkpoint and registration checks are not run and must not be represented as passing.
   - In final scope, verify tokenizer/config files, training adapters, merged full-weight shards, checkpoint completeness, merge lineage for PEFT, loadability on a clean process, the registration AIR/notebook result, and target UC model lineage. Verify these durable artifacts remain readable after the AIR node and its local input cache are gone.
   - In final scope, require the registration AIR run to terminate with `SUCCESS` and the exact created UC model version to report `READY`. Confirm that version points to the intended training MLflow run and expected logged-model source URI.
   - For a configured existing-weights Volume, require current no-download structural validation, `READ VOLUME`, complete full-weight/tokenizer inventory, and exact path parity across config, manifest, and generated workload. Preserve the pinned UC URI as lineage but do not claim the bytes were downloaded from it.
   - For any UC-backed initialization that required materialization, require the AIR run to report `SUCCESS`; confirm its numeric run ID, configured compute, persistent input weight inventory, tokenizer path, pinned model URI/run ID, worker readability, and generated `base_model`/`tokenizer_config` values match the materialization record.
4. **Model comparison**
   - Run this layer only for `run_full_migration: true`; preparation-only mode has no migrated target to compare.
   - Evaluate source and migrated models on the same immutable examples with identical prompts, decoding parameters, and scoring code.
   - Use [`air-migrate-compare-token-accuracy`](../air-migrate-compare-token-accuracy/SKILL.md) to compute deterministic assistant response-token accuracy for both portable checkpoints. Resolve the legacy model version from `migrate/config.yaml`, and resolve the exact migrated version from registration evidence.
   - Require the evaluator to create or reuse an MLflow run in `source.migration_experiment_path`, reject any injected or active run from another experiment, and log its aggregate metrics and JSON evidence artifact to that run.
   - Require matching tokenizer/chat serialization for a directly comparable token-accuracy verdict. If tokenizations differ, record both measurements and mark this criterion `inconclusive`.
   - Compare loss/perplexity where meaningful, task metrics, structured-output validity, safety/format regressions, latency, and qualitative edge cases.
   - Use statistical uncertainty or repeated runs when sampling is involved.
5. **Promotion decision**
   - Evaluate every planned criterion. Return `pass`, `fail`, or `inconclusive`, with evidence and remediation. Do not collapse missing evidence into a pass.

Read [the validation contract](references/validation-contract.md) when creating the report.

## Parity interpretation

- TRL full FSDP can target full-weight behavioral and metric parity, but hidden legacy details normally prevent a byte-identical guarantee.
- LoRA updates only adapters and cannot satisfy full-weight-training parity. Its registered artifact is a merged full inference checkpoint; validate that the merge equals the configured base plus adapter and evaluate it as a behavioral replacement.
- Continued training must be compared as a new model, not represented as reproduction.

## Output

Write `migrate/output/migration-validation.yaml` and update the manifest validation section with its scope, verdict, and `migration_complete` value. `migration_complete` means accepted migration completion, not merely terminal execution: set it true only for a passing final-scope verdict. Record input materialization separately from training execution, including whether it was required, its AIR terminal status/run ID and inventory, or current provided-Volume validation. For full validation, preserve token-accuracy evidence at `migrate/output/token-accuracy-evaluation.json` (or a versioned equivalent) and include its MLflow experiment/run IDs alongside links or identifiers for training and registration AIR runs, the training MLflow run, logged-model source, artifacts, and registered model versions. Preserve failed-run evidence. For preflight validation, mark execution, registration, and evaluation as `not_run` because they are prohibited by the execution policy; a scoped pass means only that the generated workload is runnable.

Do not register an alias, redirect traffic, or remove the source model unless the active user request explicitly authorizes that separate promotion action.
