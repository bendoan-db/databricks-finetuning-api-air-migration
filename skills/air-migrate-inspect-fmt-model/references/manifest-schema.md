# Migration manifest contract

Use this shape for `migrate/output/migration-manifest.yaml`. Add evidence-specific fields when useful, but preserve these stable sections for downstream skills.

```yaml
schema_version: 1
generated_at: 2026-07-23T00:00:00Z
migration_experiment:
  path: /Shared/fmt-migration
  experiment_id: "456"
  status: current  # current | stale | inaccessible
  created: false
source:
  model_uri: models:/catalog.schema.model/1
  catalog: catalog
  schema: schema
  model: model
  version: 1
  migration_experiment_path: /Shared/fmt-migration
  use_existing_weights: true
  existing_weights_volume_location: null
  run_full_migration: false
  status: READY
  run_id: abc123
target:
  catalog: catalog
  schema: schema
  model: model
  resolution: source_default  # source_default | explicit
requested_compute:
  num_accelerators: 8
  accelerator_type: GPU_8xH100
execution_policy:
  mode: preparation_only  # preparation_only | full_migration
  training_authorized: false
  registration_authorized: false
  evaluation_authorized: false
  stop_after: preflight_validation  # preflight_validation | final_validation
lineage:
  mlflow_experiment_id: "123"
  run_id: abc123
  datasets:
    train:
      uri: /Volumes/catalog/schema/volume/train.jsonl
      format: jsonl
      schema: chat_messages
      provenance: observed
      confidence: high
    eval:
      uri: /Volumes/catalog/schema/volume/eval.jsonl
      format: jsonl
      schema: chat_messages
      provenance: observed
      confidence: high
training_contract:
  base_model:
    name: meta-llama/Llama-3.1-8B-Instruct
    revision: null
    provenance: observed
    confidence: high
  task: CHAT_COMPLETION
  chat_template: tokenizer_default
  sequence_length: 4096
  assistant_only_loss: true
  optimizer: null
  scheduler: null
  learning_rate: 5.0e-6
  epochs: 1
  max_steps: null
  micro_batch_size: null
  gradient_accumulation_steps: null
  effective_global_batch_size: null
  precision: bf16
  seed: null
artifacts:
  final_model:
    kind: full_weights  # full_weights | peft_adapter | serving_package | absent
    uri: runs:/abc123/model
    format: safetensors
    portable: true
  tokenizer:
    uri: runs:/abc123/model
    portable: true
  checkpoints:
    kind: fsdp_sharded
    uri: runs:/abc123/checkpoints
    resumable: unknown
permissions:
  source_readable: true
  existing_weights_volume_readable: null
  train_data_readable: true
  eval_data_readable: true
  target_writable: true
unknowns: []
risks: []
inspection:
  conclusion: reproducible  # reproducible | partially_reproducible | repackage_only | blocked
  evidence: []
plan:
  status: absent  # absent | current | stale
  execution_mode: null  # preparation_only | full_migration
  recipe: null
  template: null
  migration_experiment_path: /Shared/fmt-migration
  template_selection:
    candidates_considered: []
    selected: null
    rationale: null
    compatibility_checks: []
  input_model:
    source: null  # existing_weights_volume | materialized_uc_model | materialized_system_ai | hugging_face
    source_model_uri: null
    existing_weights_volume_location: null
    model_path: null
    tokenizer_path: null
  input_staging:
    strategy: null  # volume_to_node_local | direct_hugging_face
    cache_dir: null
    copy_workers: null
    cache_scope: null  # node_ephemeral when staging Volume inputs
    estimated_source_bytes: null
    required_free_bytes: null
    capacity_verdict: null  # compatible | incompatible | unknown
    durable_outputs_under_cache: false
    evidence: []
  compute:
    requested:
      num_accelerators: 8
      accelerator_type: GPU_8xH100
    resolved:
      num_accelerators: 8
      accelerator_type: GPU_8xH100
    runtime: null
    feasibility:
      verdict: null  # compatible | incompatible | unknown
      evidence: []
materialization:
  status: absent  # absent | required | current | stale | not_required
  source_resolution:
    model_source: null  # existing_uc | system_ai | hugging_face
    source_model_uri: null
    existing_weights_volume_location: null
    requires_materialization: null
    match_basis: null
  reason: null  # existing_weights_volume_location when status is not_required
  provided_weights:
    validation_status: null  # current | stale | invalid
    model_version_checked: null
    model_path: null
    tokenizer_path: null
    weight_format: null
    inventory: []
  inventory_path: null
  execution:
    engine: null  # databricks_air when materialization is required
    profile: null
    air_run_id: null
    air_status: null  # SUCCESS after required materialization completes
    workload_file: null
    migration_experiment_path: /Shared/fmt-migration
    mlflow_experiment_id: "456"
    compute:
      num_accelerators: null
      accelerator_type: null
generation:
  status: absent  # absent | current | stale
  output_path: migrate/output/air_workload
  files: [train.yaml, train.py, helper_utils.py, training_utils.py, 01_runner.py, 02_register_uc.py]
  runnable: false
  run_from: migrate/output/air_workload
  migration_experiment_path: /Shared/fmt-migration
  compute:
    num_accelerators: null
    accelerator_type: null
  input_staging:
    volume_inputs_prefetched: null
    cache_dir: null
    copy_workers: null
    cache_scope: null
    durable_outputs_under_cache: false
    capacity_check: null
execution:
  status: absent  # absent | not_authorized | running | current | failed | stale
  reason: null  # preparation_only when status is not_authorized
  training_air_run_id: null
  training_mlflow_run_id: null
  merge_air_run_id: null
  registration_air_run_id: null
  registered_model_version: null
validation:
  status: absent  # absent | current | stale
  scope: null  # preflight | final
  migration_complete: false
  report: null
  token_accuracy_evidence: null
  token_accuracy_mlflow_experiment_id: null
  token_accuracy_mlflow_run_id: null
  verdict: null  # pass | fail | inconclusive
```

Derive `execution_policy` mechanically from `source.run_full_migration`. For `false`, use `preparation_only`, set all authorization fields to false, set `execution.status: not_authorized` with reason `preparation_only`, and stop after preflight validation; required source materialization remains allowed. For `true`, use `full_migration`, set all authorization fields to true, and stop after final validation. A current preflight validation may have `scope: preflight` and `verdict: pass` while `migration_complete` remains false. `migration_complete` means the full workflow passed its acceptance criteria: set it true only for `scope: final` with `verdict: pass`; keep it false for preflight, failed, and inconclusive results even when all attempted stages reached terminal states.

For every inferred or ambiguous training value, use an object containing `value`, `provenance`, `confidence`, and `evidence` instead of a bare scalar. Never put access tokens, secret values, or downloaded weight contents in the manifest.
