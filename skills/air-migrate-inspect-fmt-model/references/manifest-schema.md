# Migration manifest contract

Write `migrate/output/migration-manifest.yaml` with this minimum structure. Do not add materialization, execution, registration, or evaluation state.

```yaml
schema_version: 4
generated_at: 2026-07-27T00:00:00Z
source:
  catalog: catalog
  schema: schema
  model: legacy_model
  version: 1
  model_uri: models:/catalog.schema.legacy_model/1
  weights_volume_path: null
  system_ai_model_uri: null
  huggingface_model_id: meta-llama/Meta-Llama-3.1-8B-Instruct
  huggingface_token_secret: scope/key
  migration_experiment_path: /Shared/fmt-migration
  peft_only: true
  selected_model_source: hugging_face
  selected_model_reference: meta-llama/Meta-Llama-3.1-8B-Instruct
  requires_hf_token: true
target:
  catalog: catalog
  schema: schema
  model: migrated_model
  volume: null  # Reserved metadata; does not derive generated output paths
  registered_model_name: catalog.schema.migrated_model
requested_compute:
  num_accelerators: 8
  accelerator_type: GPU_8xH100
inspection:
  status: current
  source_model:
    parameter_count: 70000000000
    estimated_bf16_base_bytes: 140000000000
  source_run: {}
  datasets: {}
  training_contract: {}
  model_source_compatibility:
    status: current
    evidence: []
  permissions: []
  unknowns: []
  evidence: []
plan:
  status: current
  peft_only: true
  recipe: trl_lora_fsdp
  template: air_templates/trl_lora_fsdp
  candidates_considered: [trl_lora, trl_lora_fsdp]
  compatibility_checks: []
  model_size_selection:
    parameter_count: 70000000000
    estimated_bf16_base_bytes: 140000000000
    per_accelerator_memory_bytes: 80000000000
    estimated_non_model_overhead_bytes: 16000000000
    safety_margin_bytes: 8000000000
    full_base_replica_fits_per_worker: false
    evidence: ["Estimated bf16 base footprint exceeds one requested accelerator"]
  input_model:
    source: hugging_face
    model_reference: meta-llama/Meta-Llama-3.1-8B-Instruct
    tokenizer_reference: meta-llama/Meta-Llama-3.1-8B-Instruct
    requires_hf_token: true
  customization: {}
  assumptions: []
  risks: []
generation:
  status: current
  peft_only: true
  output_path: migrate/output/air_workload
  files: []
  template_path: air_templates/trl_lora_fsdp
  model_source: hugging_face
  customized_fields: []
  local_validations: []
  handoff_ready: true
```

Apply source precedence mechanically; do not discover system.ai automatically or fall through after a selected source fails validation. Keep the legacy UC `model_uri` separate from the selected training reference. Store secret references only, never values. When `source.peft_only` is true, require `plan.peft_only: true` and select only `trl_lora` or `trl_lora_fsdp` from recorded model-size and per-worker memory evidence. Carry provenance for observed and inferred values, and mark downstream sections `stale` after any input, PEFT constraint, or template change. Treat `target.volume` as planning metadata until output paths are explicitly customized.
