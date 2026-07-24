---
name: air-migrate-plan-air-training
description: "Turn a legacy Databricks fine-tuning migration manifest into an executable Databricks AI Runtime training plan. Use when choosing retrain, continue, or repackage semantics; selecting TRL SFTTrainer full fine-tuning or unquantized LoRA with DDP or FSDP; choosing compute, checkpointing, and hyperparameters; estimating fidelity and capacity; or deciding which approved TRL AIR template should generate the workload."
---

# Plan AI Runtime Training

Convert inspection evidence into an explicit training and validation contract. Update the manifest; do not generate workload files in this skill.

## Inputs

Read `migrate/output/migration-manifest.yaml`. If it is absent or stale, use `air-migrate-inspect-fmt-model` first. Require the source data, base-model identity, task formatting, and portable-artifact conclusion or record a blocking exception.

Require the manifest's migration experiment path and ID evidence to match `migrate/config.yaml.source.migration_experiment_path`. Preserve that path unchanged in every planned workload and MLflow run.

## Choose migration semantics

Read `source.use_existing_weights` from the manifest before selecting the mode:

- `true`: require `mode: continue`. Initialize from the validated `source.existing_weights_volume_location` when populated; otherwise materialize the exact configured UC version. State prominently that this further trains the legacy model and is not reproduction.
- `false`: require `mode: retrain`. Initialize from the recovered original base model, preferring an exact portable `system.ai` match and otherwise using Hugging Face.

Do not override this flag with a heuristic. Select the corresponding mode and state why:

- `retrain`: Start from the original base model and reproduce the legacy workflow. Default for migrating the training process and pursuing parity.
- `continue`: Start from portable legacy fine-tuned weights and train further. State prominently that this changes the model and is not reproduction.
- `repackage`: Register portable existing weights without training. State prominently that this is artifact migration, not an AI Runtime training migration.

Do not silently substitute `continue` when `use_existing_weights` is false and the original base model or dataset is unavailable.

## Preserve the execution gate

Require boolean `source.run_full_migration` and a matching top-level `execution_policy`. Derive the policy mechanically; do not broaden it based on the active stage or prior runs:

- `false`: set `execution_policy.mode: preparation_only` and `plan.execution_mode: preparation_only`, leave training, registration, and evaluation unauthorized, and set `execution_policy.stop_after: preflight_validation`. Planning may still require source materialization, generate a complete runnable workload, and define future acceptance criteria.
- `true`: set `execution_policy.mode: full_migration` and `plan.execution_mode: full_migration`, authorize training, registration, and evaluation, and set `execution_policy.stop_after: final_validation`.

Changing the flag invalidates execution and validation evidence. It does not change retrain-versus-continue semantics or whether required input weights must be materialized.

Use [the model-source resolution contract](../air-migrate-materialize-uc-model/references/model-source-resolution.md). For `continue`, require portable full Hugging Face weights and a tokenizer associated with the configured UC model. When `source.existing_weights_volume_location` is populated, require structural validation of that exact directory, set `input_model.source: existing_weights_volume`, and set `materialization.status: not_required` with reason `existing_weights_volume_location`. When it is blank, set `materialization.status: required` and materialize the UC source. For retraining, query `system.ai` for the recovered base model and use it only if an exact portable match exists; otherwise retain the Hugging Face ID. Set materialization required for `system.ai` and not required for Hugging Face. Do not point the training plan directly at a `models:/` URI.

## Select the recipe

Inspect the model before choosing a template. Compare its architecture, parameter count, context length, tokenizer requirements, weight source, requested full-weight versus PEFT semantics, and estimated per-worker memory against every entry in `air_templates/`. Use [the recipe decision guide](references/recipe-selection.md), then select the best-fitting supported recipe:

- `trl_lora`
- `trl_lora_fsdp`
- `trl_full_fsdp`
- `none`
- `custom`

Choose `trl_full_fsdp` when the objective requires updating every weight and the architecture has a validated FSDP transformer-layer wrap class. Choose `trl_lora` for unquantized bf16 PEFT when a full base-model replica fits on every worker; choose `trl_lora_fsdp` when the unquantized frozen base must be sharded or FSDP is explicitly required. If the migration requires quantized adapter training or another unsupported strategy, select `custom` and record the missing capability instead of mutating a TRL template.

Record the candidates considered, compatibility findings, selected template path, and selection rationale. Do not choose from the model name alone, and do not mutate one template into another training strategy.

Use `none` only with `mode: repackage`, because that path does not create a training job. If `custom` is required, record the missing capability. Do not route it to a merely similar template.

## Resolve requested compute

Treat `migrate/config.yaml.compute` and the manifest's `requested_compute` as authoritative. Require them to match exactly. Supported AIR resources are `GPU_1xA10` with one accelerator, `GPU_1xH100` with one accelerator, and `GPU_8xH100` with eight accelerators; reject any count/type mismatch.

Evaluate the requested resource against the inspected parameter count, precision, context length, batch size, activation budget, adapter or optimizer state, FSDP strategy, checkpoint consolidation, PEFT merge, node-local input staging, and MLflow registration packaging. For any `/Volumes` model/tokenizer source, require enough ephemeral disk on every AIR node for one complete staged source plus the template safety reserve; add separate tokenizer capacity when it is not the model directory. Registration can require host memory and temporary storage for multiple copies of the full checkpoint even when it performs no GPU computation. Record concrete feasibility checks for input staging, training, merge/consolidation, and registration. Do not silently replace an insufficient or unavailable resource with the template default: select a compatible plan, mark `recipe: custom`, or return a blocker that tells the user which compute or model-source value must change.

## Build the training contract

1. Map all observed legacy values before applying defaults.
2. Match task serialization, tokenizer revision, chat template, sequence length, truncation, packing, assistant-token loss masking, and special tokens.
3. Match optimizer, scheduler, learning rate, warmup, epochs or tokens, effective global batch size, precision, gradient clipping, and seed where known.
4. Derive micro-batch size and gradient accumulation from GPU capacity while preserving the effective batch size.
5. Copy the requested AIR accelerator count/type unchanged, then define runtime version, dependency pins, distributed settings, checkpoint cadence, resume behavior, final artifact format, MLflow experiment, UC Volume paths, and target model registration. Plan the registration notebook as a single-process AIR command on the same configured compute for large checkpoints; do not use the training `torchrun` launcher for registration.
6. For every Volume-backed model/tokenizer source, define `input_staging` with an absolute ephemeral node-local cache path, bounded parallel copy count, per-node source-size estimate, required reserve, and capacity evidence. Retain the Volume path as durable provenance and require runtime local-only loading from the staged copy. State that adapters, checkpoints, merged/full outputs, and registration inputs remain on UC Volumes.
7. Document every intentional divergence and its expected fidelity impact.
8. Define acceptance gates before generation: static checks, smoke training, metric thresholds, behavioral evaluation, artifact checks, and target registration checks. Include `assistant_response_token_accuracy` on the immutable evaluation JSONL, its maximum allowed absolute regression from the configured legacy model, and a requirement for equivalent tokenizer/chat serialization. If no defensible regression threshold exists, explicitly make this a measurement-only criterion.

## Output

Write or replace `plan` in the manifest with:

```yaml
plan:
  status: current
  execution_mode: full_migration
  mode: retrain
  objective: full_weight_parity
  framework: trl
  migration_experiment_path: /Shared/fmt-migration
  recipe: trl_full_fsdp
  template: air_templates/trl_full_fsdp
  template_selection:
    candidates_considered:
      - trl_lora
      - trl_lora_fsdp
      - trl_full_fsdp
    selected: trl_full_fsdp
    rationale: Full-weight migration objective and supported model architecture
    compatibility_checks: []
  input_model:
    use_existing_weights: false
    source: hugging_face  # existing_weights_volume | materialized_uc_model | materialized_system_ai | hugging_face
    source_model_uri: null
    existing_weights_volume_location: null
    model_path: meta-llama/Llama-3.1-8B-Instruct
    tokenizer_path: null
  compute:
    requested:
      num_accelerators: 8
      accelerator_type: GPU_8xH100
    resolved:
      num_accelerators: 8
      accelerator_type: GPU_8xH100
    runtime: "AI Runtime v5"
    feasibility:
      verdict: compatible
      evidence: []
  input_staging:
    strategy: volume_to_node_local
    cache_dir: /tmp/air-model-cache
    copy_workers: 8
    cache_scope: node_ephemeral
    estimated_source_bytes: null
    required_free_bytes: null
    capacity_verdict: compatible
    durable_outputs_under_cache: false
    evidence: []
  hyperparameters: {}
  data_contract: {}
  checkpoint_contract: {}
  output_contract:
    registration_execution: air
    registration_compute: same_as_training
    registration_processes: 1
    require_air_status: SUCCESS
    require_uc_version_status: READY
  intentional_divergences: []
  assumptions: []
  risks: []
  acceptance_criteria:
    token_accuracy:
      metric: assistant_response_token_accuracy
      evaluation_dataset_uri: /Volumes/catalog/schema/volume/eval.jsonl
      maximum_absolute_regression: 0.01
      require_equivalent_tokenization: true
```

For `mode: continue` with provided Volume weights, set `input_model.source: existing_weights_volume`, retain the configured source URI, copy `existing_weights_volume_location`, and set `model_path` and `tokenizer_path` to the structurally validated configured directory. For the blank-path branch, set `input_model.source: materialized_uc_model`, record the configured source URI and intended destination, and leave paths pending until materialization succeeds. For retraining from `system.ai`, use `materialized_system_ai` with its pinned URI and paths. For retraining from Hugging Face, use `hugging_face`, retain the recovered model ID, and set `materialization.status: not_required`.

The plan is not complete until it identifies one source template and fixes the generated workload destination as `migrate/output/air_workload`.

Set `materialization.status`, provided-weight validation, `generation.status`, and `validation.status` to `stale` when an existing plan materially changes their inputs, including any change between blank and populated `existing_weights_volume_location`. When only `run_full_migration` changes, retain current input-weight and generated-workload evidence only if their inputs are identical, but invalidate execution and validation evidence and rewrite `execution_policy` before proceeding.

## Guardrails

- Do not promise numerical weight equality when hidden legacy-service behavior cannot be recovered.
- Do not call LoRA full-weight training parity.
- Do not invent or override infrastructure capacity. Validate the selected model, sequence length, batch, precision, checkpoint strategy, and PEFT merge against the compute requested in `migrate/config.yaml`.
- Keep secrets as environment-variable or Databricks secret references.
- Require explicit rationale for CPU offload because it can materially reduce throughput.
