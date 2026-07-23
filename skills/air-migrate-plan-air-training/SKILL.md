---
name: air-migrate-plan-air-training
description: "Turn a legacy Databricks fine-tuning migration manifest into an executable Databricks AI Runtime training plan. Use when choosing retrain, continue, or repackage semantics; selecting full fine-tuning, QLoRA, FSDP, Axolotl, compute, checkpointing, and hyperparameters; estimating fidelity and capacity; or deciding which approved AIR template should generate the workload."
---

# Plan AI Runtime Training

Convert inspection evidence into an explicit training and validation contract. Update the manifest; do not generate workload files in this skill.

## Inputs

Read `migrate/output/migration-manifest.yaml`. If it is absent or stale, use `air-migrate-inspect-fmt-model` first. Require the source data, base-model identity, task formatting, and portable-artifact conclusion or record a blocking exception.

## Choose migration semantics

Select exactly one mode and state why:

- `retrain`: Start from the original base model and reproduce the legacy workflow. Default for migrating the training process and pursuing parity.
- `continue`: Start from portable legacy fine-tuned weights and train further. State prominently that this changes the model and is not reproduction.
- `repackage`: Register portable existing weights without training. State prominently that this is artifact migration, not an AI Runtime training migration.

Do not silently substitute `continue` when the original base model or dataset is unavailable.

For `continue`, require portable full Hugging Face weights and a tokenizer. Select a writable UC Volume destination and set `materialization.status: required`; `air-migrate-materialize-uc-model` must complete before generation. Do not point the training plan directly at a `models:/` URI.

## Select the recipe

Use [the recipe decision guide](references/recipe-selection.md), then set one of:

- `axolotl_full_fsdp`
- `axolotl_qlora`
- `axolotl_qlora_fsdp`
- `none`
- `custom`

Choose `axolotl_full_fsdp` for full-weight parity unless a documented incompatibility requires a custom framework. Choose a QLoRA recipe only when adapter-based behavioral replacement is acceptable. Use QLoRA plus FSDP when the quantized frozen base still requires sharding; model size alone is a heuristic, not proof.

Use `none` only with `mode: repackage`, because that path does not create a training job. If `custom` is required, record the missing capability. Do not route it to a merely similar template.

## Build the training contract

1. Map all observed legacy values before applying defaults.
2. Match task serialization, tokenizer revision, chat template, sequence length, truncation, packing, assistant-token loss masking, and special tokens.
3. Match optimizer, scheduler, learning rate, warmup, epochs or tokens, effective global batch size, precision, gradient clipping, and seed where known.
4. Derive micro-batch size and gradient accumulation from GPU capacity while preserving the effective batch size.
5. Define GPU count/type, runtime version, dependency pins, distributed settings, checkpoint cadence, resume behavior, final artifact format, MLflow experiment, UC Volume paths, and target model registration.
6. Document every intentional divergence and its expected fidelity impact.
7. Define acceptance gates before generation: static checks, smoke training, metric thresholds, behavioral evaluation, artifact checks, and target registration checks.

## Output

Write or replace `plan` in the manifest with:

```yaml
plan:
  status: current
  mode: retrain
  objective: full_weight_parity
  framework: axolotl
  recipe: axolotl_full_fsdp
  template: air_templates/axolotl_full_fsdp
  input_model:
    source: original_base_model  # original_base_model | materialized_uc_model
    model_path: meta-llama/Llama-3.1-8B-Instruct
    tokenizer_path: null
  compute:
    gpu_type: H100
    num_gpus: 8
    runtime: "AI Runtime v5"
  hyperparameters: {}
  data_contract: {}
  checkpoint_contract: {}
  output_contract: {}
  intentional_divergences: []
  assumptions: []
  risks: []
  acceptance_criteria: {}
```

For `mode: continue`, set `input_model.source: materialized_uc_model`, record the intended Volume destination, and leave `model_path` and `tokenizer_path` pending until materialization succeeds. For `retrain`, set `materialization.status: not_required`.

Set `materialization.status`, `generation.status`, and `validation.status` to `stale` when an existing plan materially changes their inputs.

## Guardrails

- Do not promise numerical weight equality when hidden legacy-service behavior cannot be recovered.
- Do not call QLoRA full-weight parity.
- Do not invent infrastructure capacity. Validate the selected model, sequence length, batch, quantization, and checkpoint strategy against available GPU and CPU memory.
- Keep secrets as environment-variable or Databricks secret references.
- Require explicit rationale for CPU offload because it can materially reduce throughput.
