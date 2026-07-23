# Migration manifest contract

Use this shape for `migrate/output/migration-manifest.yaml`. Add evidence-specific fields when useful, but preserve these stable sections for downstream skills.

```yaml
schema_version: 1
generated_at: 2026-07-23T00:00:00Z
source:
  model_uri: models:/catalog.schema.model/1
  catalog: catalog
  schema: schema
  model: model
  version: 1
  use_existing_weights: true
  status: READY
  run_id: abc123
target:
  catalog: catalog
  schema: schema
  model: model
  resolution: source_default  # source_default | explicit
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
  input_model:
    source: null  # materialized_uc_model | materialized_system_ai | hugging_face
    source_model_uri: null
    model_path: null
    tokenizer_path: null
materialization:
  status: absent  # absent | required | current | stale | not_required
  source_resolution:
    model_source: null  # existing_uc | system_ai | hugging_face
    source_model_uri: null
    match_basis: null
generation:
  status: absent  # absent | current | stale
validation:
  status: absent  # absent | current | stale
  report: null
  token_accuracy_evidence: null
  verdict: null  # pass | fail | inconclusive
```

For every inferred or ambiguous training value, use an object containing `value`, `provenance`, `confidence`, and `evidence` instead of a bare scalar. Never put access tokens, secret values, or downloaded weight contents in the manifest.
