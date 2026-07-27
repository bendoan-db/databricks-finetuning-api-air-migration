---
name: air-migrate-inspect-fmt-model
description: "Inspect a legacy Databricks Foundation Model Fine-tuning model and determine whether generated AIR training should use a configured Unity Catalog Volume, an exact system.ai model, or Hugging Face. Record lineage, model-size evidence, the PEFT-only constraint, and template-selection inputs without downloading weights or executing workloads."
---

# Inspect FMT Model

Produce `migrate/output/migration-manifest.yaml` for planning and generation. This is read-only discovery.

## Workflow

1. Read `migrate/config.yaml`. Require source catalog, schema, model, positive version, absolute `migration_experiment_path`, and boolean `peft_only`.
2. Resolve the model source by strict precedence: nonblank `weights_volume_path`, then nonblank `system_ai_model_uri`, otherwise nonblank `huggingface_model_id`. Do not auto-discover system.ai or fall through when a selected source is invalid. Validate `/Volumes/...`, `models:/system.ai.<model>/<version>`, and Hugging Face repository syntax respectively.
3. Resolve all-blank target identity fields to the source identity. Reject partially populated target identities.
4. Verify read access to the source model version and follow its `run_id` into MLflow. Record parameters, metrics, datasets, tags, and artifact metadata without downloading large artifacts.
5. Recover task format, architecture, parameter count, estimated bf16 base-model bytes, tokenizer/chat template, sequence length, optimizer, batch behavior, precision, seed, distributed strategy, and data paths. This size evidence is required to choose replicated or sharded LoRA when `peft_only` is true.
6. For `volume`, structurally verify full model weights, `config.json`, and tokenizer assets without loading tensors. For `system_ai`, confirm the configured URI is an exact portable candidate when metadata is accessible. For `hugging_face`, record the repository and whether the configured secret reference indicates gated access.
7. Validate requested AIR compute and write the manifest using [the manifest contract](references/manifest-schema.md), copying `peft_only` unchanged.

## Guardrails

- Preserve the versioned legacy UC URI as lineage; it is distinct from the selected training input.
- Do not download, copy, or materialize weights during inspection.
- If remote compatibility cannot be established read-only, record an assumption or blocker instead of testing by download.
- Record unknown legacy behavior explicitly. Do not claim byte-identical reproducibility.
- Do not run AIR, training, registration, merge, smoke, or evaluation commands.

Finish when the source choice, PEFT constraint, model-size evidence, training/data contract, target, compute, unknowns, and template-selection inputs are recorded or a concrete blocker is identified.
