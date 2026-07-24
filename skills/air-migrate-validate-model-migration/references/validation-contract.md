# Migration validation contract

Write a machine-readable report with this stable structure:

```yaml
schema_version: 1
scope: preflight  # preflight | final
migration_complete: false
execution_policy:
  run_full_migration: false
  mode: preparation_only
  stop_after: preflight_validation
source_model_uri: models:/catalog.schema.source/1
target_model_uri: null
plan_objective: full_weight_parity
migration_experiment:
  path: /Shared/fmt-migration
  experiment_id: "456"
  status: current
  created: false
workload:
  path: migrate/output/air_workload
  recipe: trl_full_fsdp
  files: [train.yaml, train.py, helper_utils.py, training_utils.py, 01_runner.py, 02_register_uc.py]
  template_provenance_valid: true
  experiment_name: /Shared/fmt-migration
  training_experiment_path: /Shared/fmt-migration
  experiment_consistent: true
  compute:
    requested:
      num_accelerators: 8
      accelerator_type: GPU_8xH100
    generated:
      num_accelerators: 8
      accelerator_type: GPU_8xH100
    torchrun_nproc_per_node: 8
    runner_gpu_type: H100
    consistent: true
  input_model:
    use_existing_weights: true
    model_source: existing_uc
    source_model_uri: models:/catalog.schema.source/1
    existing_weights_volume_location: null
    model_path: /Volumes/catalog/schema/volume/source-model-v1/model
  input_staging:
    strategy: volume_to_node_local
    cache_dir: /tmp/air-model-cache
    copy_workers: 8
    cache_scope: node_ephemeral
    source_bytes: 16000000000
    required_free_bytes: 17600000000
    capacity_verdict: pass
    one_copy_per_node: true
    atomic_completion: true
    local_files_only: true
    durable_outputs_under_cache: false
    mlflow_evidence:
      run_id: null
      cache_hit: null
      file_count: null
      total_bytes: null
      copy_seconds: null
      lock_wait_seconds: null
      throughput_mib_per_second: null
static_preflight:
  verdict: pass
  checks: []
input_materialization:
  required: true
  status: success  # not_required | success | failed | stale
  source_model_uri: models:/catalog.schema.source/1
  air_status: SUCCESS
  air_run_id: "123456789"
  inventory_path: /Volumes/catalog/schema/volume/source-model-v1/materialization.json
  provided_weights_validation: null
  checks: []
execution:
  authorized: false
  status: not_run  # not_run | success | failed
  reason: preparation_only
  smoke_run_id: null
  training_air_run_id: null
  training_mlflow_run_id: null
  merge_air_run_id: null
  registration_air_run_id: null
  verdict: not_run
  run_experiment_checks: []
  checks: []
artifacts:
  expected_kind: full_weights
  loadable: null
  tokenizer_complete: null
  checkpoint_resume_tested: null
  registration_notebook: migrate/output/air_workload/02_register_uc.py
  registered_model_version_created: false
  checks: []
registration:
  status: not_run  # not_run | success | failed
  reason: preparation_only
  execution_mode: null
  command: null
  compute_matches_training: null
  air_status: null
  registered_model_name: catalog.schema.target
  registered_model_version: null
  registered_model_status: null
  logged_model_source_uri: null
  training_mlflow_run_id: null
  checks: []
evaluation:
  status: not_run  # not_run | success | failed | inconclusive
  reason: preparation_only
  dataset_uri: /Volumes/catalog/schema/volume/eval.jsonl
  dataset_version: null
  dataset_sha256: null
  decoding_parameters: {}
  token_accuracy:
    metric: assistant_response_token_accuracy
    evidence: null
    mlflow_experiment_path: /Shared/fmt-migration
    mlflow_experiment_id: null
    mlflow_run_id: null
    directly_comparable: null
    legacy_accuracy: null
    migrated_accuracy: null
    absolute_accuracy_delta: null
    max_accuracy_regression: 0.01
    verdict: not_run
  metrics: []
  behavioral_cases: []
acceptance_criteria: []
unknowns: []
failures: []
verdict: pass  # pass | fail | inconclusive; scoped by scope
recommendation: null
```

For `run_full_migration: false`, use the preparation-only values shown above. A `scope: preflight` pass certifies only that input-weight resolution/materialization and the generated workload passed preflight; it must keep `migration_complete: false`, and `not_run` is the only valid status for training, merge, registration, and model evaluation.

In preflight, populate the input-staging configuration and capacity evidence, but leave runtime-only MLflow observations null. In final scope, copy the rank-zero staging evidence from the training run and retain per-node log/run references when the job spans multiple nodes. A Volume-backed source requires `strategy: volume_to_node_local`, one locked copy per node, atomic completion, and local-only Transformers loading. A remote Hugging Face source uses `strategy: direct_hugging_face` and may leave the template cache unused.

`cache_dir` is ephemeral acceleration state. It must not appear as an artifact, output, resume, merge, or registration path, and artifact-integrity validation must succeed independently after cache loss. Capacity must include each distinct staged source directory plus at least 1 GiB or 10 percent reserve, whichever is greater.

When the configured source reuses validated Volume weights, set `input_materialization.required: false`, `status: not_required`, and `provided_weights_validation: current`; leave AIR status, run ID, and inventory path null. When materialization is required, a preflight pass requires `status: success`, AIR `SUCCESS`, its numeric run ID, and the persisted inventory path as shown.

For `run_full_migration: true`, set `scope: final`, `migration_complete: true` only after every required stage passes, use `mode: full_migration` and `stop_after: final_validation`, and populate target, execution, artifact, registration, and evaluation evidence. Use `status: success` for each successful execution, registration, and evaluation section. A failed or inconclusive final verdict must keep `migration_complete: false`.

If a full migration fails before a target version exists, still write a final-scope failure report: keep `target_model_uri: null`, set the failing stage to `status: failed`, mark downstream stages `status: not_run` with reason `upstream_failure`, record available run/failure evidence, set top-level `verdict: fail`, and keep `migration_complete: false`. Use the same rule for a registration failure after successful training; preserve the training evidence and leave evaluation `not_run`.

Each check should name the criterion, expected value or range, observed value, evidence location, and individual verdict. Include run and artifact identifiers rather than pasting large logs. Keep AIR/Jobs run IDs distinct from MLflow run IDs. Redact credentials and secret values.

Resolve the experiment ID for every AIR-managed or application-created materialization, smoke, training, merge, registration, and evaluation run. Each must equal `migration_experiment.experiment_id`; a successful run in another experiment fails the affected stage. Preflight requires current existence/create evidence and exact parity among config, manifest, plan, generated `experiment_name`, and generated `training_config.experiment_path`.

In final scope, registration passes only when its AIR run is terminal `SUCCESS` and the exact UC model version is `READY`. A registered-model container without a ready version, a logged-model upload in progress, or a successful training run is insufficient evidence. Preserve failed registration runs and partial logged-model identifiers in `failures`.

Use fixed prompts and deterministic decoding for direct response comparisons when the task permits it. Record dataset versions or hashes so a later validation run uses identical inputs.

Token accuracy is teacher-forced and does not use decoding parameters. Copy its summary and MLflow experiment/run identifiers from the evidence produced by `air-migrate-compare-token-accuracy`; do not recompute or round values before evaluating its threshold. Require that evaluation run's experiment ID to match `migration_experiment.experiment_id`. A required token-accuracy criterion can pass only when its evidence says `directly_comparable: true` and its verdict is `pass`.
