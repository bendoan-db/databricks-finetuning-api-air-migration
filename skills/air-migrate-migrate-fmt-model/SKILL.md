---
name: air-migrate-migrate-fmt-model
description: "Orchestrate a preparation-only or end-to-end migration of a Unity Catalog model produced by the legacy Databricks Foundation Model Fine-tuning API. Selects the best-fitting approved TRL air_templates recipe, creates the canonical model-specific AI Runtime training and registration modules under migrate/output/air_workload, enforces source.run_full_migration, and coordinates authorized execution, registration, and validation. Use when a user asks to assess, plan, generate, run, validate, or resume the migration."
---

# Migrate FMT Model

Coordinate the migration as a resumable, evidence-driven workflow. Delegate stage details to the sibling skills and keep `migrate/output/migration-manifest.yaml` as the shared contract.

## Required stage skills

Read and follow these sibling skill instructions in order when installed together:

1. [`air-migrate-inspect-fmt-model`](../air-migrate-inspect-fmt-model/SKILL.md)
2. [`air-migrate-plan-air-training`](../air-migrate-plan-air-training/SKILL.md)
3. [`air-migrate-materialize-uc-model`](../air-migrate-materialize-uc-model/SKILL.md)
4. [`air-migrate-generate-air-job`](../air-migrate-generate-air-job/SKILL.md)
5. [`air-migrate-compare-token-accuracy`](../air-migrate-compare-token-accuracy/SKILL.md)
6. [`air-migrate-validate-model-migration`](../air-migrate-validate-model-migration/SKILL.md)

If a stage skill is unavailable, stop before that stage and identify the missing package; do not improvise its guarded workflow.

In preparation-only mode, use the validation skill for preflight but do not invoke the token-accuracy skill. Its position in the full stage list does not override the execution gate.

## Workflow

1. **Resolve configuration**
   - Read `migrate/config.yaml`.
   - Require `source.migration_experiment_path` as a nonblank absolute Databricks workspace experiment path. Before any AIR dry-run or live submission, run `python3 skills/air-migrate-migrate-fmt-model/scripts/ensure_migration_experiment.py --config migrate/config.yaml --profile DEFAULT` and require/create the experiment through MLflow. Record its path and experiment ID; stop on permission or lifecycle errors.
   - Require `source.use_existing_weights` to be boolean and preserve it unchanged in every downstream stage.
   - Require `source.run_full_migration` to be boolean and preserve it unchanged. Derive `execution_policy` mechanically: false means `preparation_only` with a hard stop after preflight validation; true means `full_migration` and explicitly authorizes all post-preflight pipeline stages.
   - Normalize blank `source.existing_weights_volume_location` to null. If populated, require an absolute `/Volumes/...` path and `source.use_existing_weights: true`; preserve the exact path downstream.
   - Treat all-blank target fields as a new version at the source model name.
   - Reject partially populated target fields.
   - Require a valid `compute` mapping. Accept only `GPU_1xA10`/1, `GPU_1xH100`/1, or `GPU_8xH100`/8, and preserve the requested count and type unchanged in planning and generation.
2. **Inspect**
   - Recover UC/MLflow lineage, data, task semantics, hyperparameters, artifacts, permissions, and unknowns.
   - Gate: require a viable retrain, continue, or repackage starting point.
3. **Plan**
   - When `use_existing_weights` is true, select continued training from the configured Volume checkpoint when supplied, otherwise from the configured UC version after materialization. When false, select retraining from the recovered base model and resolve `system.ai` before falling back to Hugging Face.
   - Compare the inspected model and migration objective against every supported `air_templates/` recipe, then select the best-fitting template with recorded compatibility evidence and rationale.
   - Validate the requested compute against the selected model, context and batch, training method, checkpointing, and PEFT merge; never inherit template compute when it differs.
   - For any Volume-backed model/tokenizer source, require a node-local staging plan with per-node capacity for one source copy plus reserve. Treat the Volume path as durable lineage and the local cache as disposable runtime state.
   - Select objective, checkpoint/output contract, intentional divergences, and acceptance criteria.
   - Gate: require an explicit plan and one supported source template, `none` for repackaging, or a documented custom-template blocker.
4. **Resolve and materialize training weights**
   - Use `resolve_training_source.py` with the inspected base-model ID.
   - If `use_existing_weights` is true and `existing_weights_volume_location` is populated, structurally validate that exact model/tokenizer directory, record materialization as not required, and load it directly. Do not generate a UC materialization workload or silently fall back to downloading the model.
   - If `use_existing_weights` is true and the location is blank, generate an AIR materialization workload and stage the configured UC model and tokenizer in a shared UC Volume using `migrate/config.yaml.compute`.
   - If false and an exact `system.ai` model exists, pin and materialize that version through AIR. If no match exists, use the recovered Hugging Face model ID and record materialization as not required.
   - Run materialization as one rank-zero Python process without `torchrun`. Require AIR `SUCCESS`, its numeric run ID, the persistent structural inventory, and worker-readable paths before marking the stage current. Never perform the large download on local or standard notebook compute.
   - Gate: never pass a `models:/` URI directly to Transformers or TRL; require either current provided-Volume validation or current AIR materialization provenance and worker-readable paths.
5. **Generate from templates**
   - For retraining or continued training, use `air-migrate-generate-air-job` to copy exactly one approved `air_templates/` recipe to `migrate/output/air_workload` before customization.
   - Customize that generated copy for the inspected model while preserving the template's `train.yaml`, thin `train.py`, `helper_utils.py`, `training_utils.py`, optional LoRA `merge.py`, `01_runner.py`, and `02_register_uc.py` module format.
   - Copy `migrate/config.yaml.compute` exactly into generated `train.yaml`, synchronize `torchrun --nproc_per_node`, and keep the notebook's `@distributed` launch derived from that YAML.
   - Copy `source.migration_experiment_path` exactly into both generated `train.yaml.experiment_name` and `parameters.training_config.experiment_path`. All training, merge, registration, materialization, smoke, and evaluation AIR submissions use that same workload experiment; application-created MLflow runs must use it too.
   - Preserve `local_model_cache_dir`, `local_model_cache_copy_workers`, one-copy-per-node locking, atomic completion, local-only staged loading, and MLflow staging metrics from the selected template. Customize the cache only to an absolute ephemeral node-local path with sufficient capacity.
   - Keep all adapters, Trainer checkpoints, merged/full checkpoints, and registration artifacts on UC Volumes; never route durable outputs through the ephemeral input cache.
   - For repackaging, skip AIR workload generation and record `generation.status: skipped` with the reason.
   - Never write workload modules from scratch when an approved recipe is selected.
   - Gate: require every recipe-specific generated module, template provenance, model/template compatibility evidence, static validation, and AIR dry-run when available.
6. **Apply the execution gate**
   - If `run_full_migration` is false, use `air-migrate-validate-model-migration` for static preflight only. Record `execution.status: not_authorized` with reason `preparation_only`, plus `validation.status: current`, `scope: preflight`, a scoped verdict, and `migration_complete: false`, then stop explicitly.
   - Preparation-only mode may submit required source materialization and may use AIR `--dry-run`, which stages workspace files but does not submit a Jobs run. It must not submit training or merge commands, invoke registration, materialize the target for evaluation, or invoke `air-migrate-compare-token-accuracy`.
   - If `run_full_migration` is true, treat the config as explicit authorization to continue through training, required merge, registration, evaluation, and final validation. Do not require a second authorization prompt for those pipeline stages.
   - Gate: never infer authorization from generated code, earlier runs, or the active user request when the config says false.
7. **Execute and register**
   - Read and follow [the AIR execution and registration runbook](references/air-execution-registration.md).
   - Enter this stage only when `run_full_migration` is true.
   - Treat `migrate/output/air_workload` as the sole executable training migration. Never submit a source directory under `air_templates/`.
   - Train with generated `train.yaml` through the AIR CLI or with `01_runner.py`. Record the numeric AIR run ID separately from the rank-zero training MLflow run ID.
   - Before submission, ensure the configured experiment exists. Require the AIR-managed and rank-zero training runs to belong to its recorded experiment ID; reject an injected, active, or supplied run from another experiment instead of silently logging elsewhere.
   - Before interpreting model-load time, inspect the MLflow staging evidence. A first run on each node copies Volume-backed model/tokenizer files once; other ranks wait for and reuse the atomic local copy. A later job may copy again because AIR node-local storage is ephemeral.
   - For an AIR LoRA run, execute the generated `merge.py` action because AIR does not run the merge cell in `01_runner.py`. Require and inspect the merged portable full checkpoint before registration.
   - Register large checkpoints by submitting generated `02_register_uc.py` as the AIR Python command with `--mlflow-run-id <training-run-id>`. Preserve the generated `train.yaml.compute` block and do not use `torchrun` for this single registration process. Use the interactive widget path only when its compute has sufficient host memory and temporary storage.
   - Require the registration AIR run to terminate with `SUCCESS` and the exact new UC model version to reach `READY`. Creation of the UC model container or a pending version is not completion.
   - Preserve MLflow lineage, checkpoint evidence, output type, and the resolved target registration rule.
   - Do not overwrite an existing model version; UC registration creates a new version.
8. **Validate**
   - Enter final model validation only when `run_full_migration` is true. Do not reuse a preparation-only verdict as migration-completion evidence.
   - Compare source and target against the predeclared criteria on identical evaluation inputs.
   - Reuse the validated configured source Volume when supplied; otherwise materialize the exact registered source. Materialize the target full checkpoint, then use `air-migrate-compare-token-accuracy` to compare assistant response-token accuracy. The source URI must still come from `migrate/config.yaml`; the target must be the version created by this workflow. Require the evaluator's MLflow run and JSON evidence to identify the configured migration experiment.
   - Treat tokenizer or chat-serialization differences as an inconclusive token-accuracy comparison, not a pass.
   - Gate: set `migration_complete: true` only when final-scope validation passes. A failed or inconclusive verdict remains a completed pipeline attempt, not a completed accepted migration.
9. **Report**
   - For preparation-only mode, summarize source resolution/materialization, generated workload, preflight verdict, the explicit stop, and the manual next action without claiming a migrated target exists.
   - For full mode, summarize source and target URIs, selected mode/recipe/template, deliberate differences, runs, artifacts, final validation verdict, remaining risks, and exact next action.

## Template routing invariant

The planner owns recipe selection; the generator only materializes it:

- `trl_lora` -> `air_templates/trl_lora`
- `trl_lora_fsdp` -> `air_templates/trl_lora_fsdp`
- `trl_full_fsdp` -> `air_templates/trl_full_fsdp`

Use no fallback. For `recipe: none`, skip generation only when `mode: repackage`. For `recipe: custom`, pause generation until a suitable template is added and validated.

## Canonical runnable artifact

For every training migration, completion of planning and input-weight resolution must produce:

```text
migrate/output/air_workload/
├── train.yaml
├── train.py
├── helper_utils.py
├── training_utils.py
├── merge.py  # LoRA recipes only
├── 01_runner.py
└── 02_register_uc.py
```

This generated directory follows the selected template format and contains the model-specific source, data, compute, hyperparameters, output paths, and UC registration target. Run this directory to migrate the model. The repository-level `air_templates/` directories remain immutable sources and are never submitted directly.

`02_register_uc.py` must remain executable in both contexts: Databricks widgets for interactive use and `--mlflow-run-id` when AIR runs the source file as Python. Notebook `%pip` commands are ignored in AIR script mode, so `train.yaml.environment.dependencies` is authoritative for registration imports as well as training imports.

## Resume and invalidation

Reuse a completed stage only when its inputs and evidence are still current. Reinspect and replan when the source version, `use_existing_weights`, `existing_weights_volume_location`, `run_full_migration`, `migration_experiment_path`, requested compute, source run, config, permissions, data, or weight-source resolution changes. Revalidate provided weights when their path or inventory changes. Rematerialize only when a selected UC source requiring materialization, destination, artifact, tokenizer, AIR workload, successful run evidence, experiment path, or compute changes. Regenerate when the plan, requested compute, migration experiment, resolved input weights, authoritative template, node-local cache path, or cache copy-worker count changes. Do not treat a prior AIR job's ephemeral cache as resumable evidence. A gate-only change may reuse otherwise current materialization and generation evidence, but must invalidate execution and validation evidence. Rerun token accuracy and revalidate when generated code, evaluation data, model artifacts, tokenizer/chat templates, sequence-length policy, or acceptance criteria change.

## Safety and truthfulness

- Prefer read-only discovery before mutations.
- Never expose or copy secret values into code, manifests, or reports.
- Distinguish reproduction, continued training, and repackaging in every handoff.
- Enforce `run_full_migration: false` as a hard post-preflight stop even when runnable files or historical run IDs exist.
- Call LoRA adapter-based migrations, not full-weight parity.
- Surface unknown legacy-service behavior and unsupported architectures explicitly.
- Do not promote aliases, route serving traffic, delete artifacts, or retire the source model without separate explicit authorization.
