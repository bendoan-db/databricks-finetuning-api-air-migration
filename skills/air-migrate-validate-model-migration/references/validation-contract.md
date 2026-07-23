# Migration validation contract

Write a machine-readable report with this stable structure:

```yaml
schema_version: 1
source_model_uri: models:/catalog.schema.source/1
target_model_uri: models:/catalog.schema.target/2
plan_objective: full_weight_parity
workload:
  path: migrate/output/air_workload
  recipe: axolotl_full_fsdp
  template_provenance_valid: true
  input_model:
    use_existing_weights: true
    model_source: existing_uc
    source_model_uri: models:/catalog.schema.source/1
    model_path: /Volumes/catalog/schema/volume/source-model-v1/model
static_preflight:
  verdict: pass
  checks: []
execution:
  authorized: true
  smoke_run_id: null
  training_run_id: null
  verdict: pass
  checks: []
artifacts:
  expected_kind: full_weights
  loadable: true
  tokenizer_complete: true
  checkpoint_resume_tested: false
  checks: []
evaluation:
  dataset_uri: /Volumes/catalog/schema/volume/eval.jsonl
  dataset_version: null
  dataset_sha256: null
  decoding_parameters: {}
  token_accuracy:
    metric: assistant_response_token_accuracy
    evidence: migrate/output/token-accuracy-evaluation.json
    directly_comparable: true
    legacy_accuracy: 0.90
    migrated_accuracy: 0.91
    absolute_accuracy_delta: 0.01
    max_accuracy_regression: 0.01
    verdict: pass
  metrics: []
  behavioral_cases: []
acceptance_criteria: []
unknowns: []
failures: []
verdict: inconclusive  # pass | fail | inconclusive
recommendation: null
```

Each check should name the criterion, expected value or range, observed value, evidence location, and individual verdict. Include run and artifact identifiers rather than pasting large logs. Redact credentials and secret values.

Use fixed prompts and deterministic decoding for direct response comparisons when the task permits it. Record dataset versions or hashes so a later validation run uses identical inputs.

Token accuracy is teacher-forced and does not use decoding parameters. Copy its summary from the evidence produced by `air-migrate-compare-token-accuracy`; do not recompute or round values before evaluating its threshold. A required token-accuracy criterion can pass only when its evidence says `directly_comparable: true` and its verdict is `pass`.
