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
  decoding_parameters: {}
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
