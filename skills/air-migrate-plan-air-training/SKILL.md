---
name: air-migrate-plan-air-training
description: "Turn an inspected FMT migration into a generation-only AI Runtime plan. Enforce peft_only, select replicated or sharded LoRA from model-size evidence when required, and preserve the configured Volume, system.ai, or Hugging Face model source."
---

# Plan AI Runtime Training

Update the manifest with a concrete generation plan. Do not create files or execute workloads in this skill.

## Inputs

Require a current inspection manifest matching `migrate/config.yaml`. Preserve its mechanically selected source:

- `volume`: `weights_volume_path` is populated and overrides other candidates.
- `system_ai`: the Volume path is blank and an exact versioned `system_ai_model_uri` is populated.
- `hugging_face`: both prior fields are blank and `huggingface_model_id` is populated.

Do not introduce an agent-side materialization stage. The generated runtime handles node-local Volume copies, system.ai downloads, or Hugging Face downloads when the operator runs it.

Require boolean `source.peft_only` in both configuration and manifest and require the values to match. Treat it as an authoritative training-strategy constraint, not a preference.

## Select a template

Compare architecture, parameter count, estimated bf16 base footprint, context length, tokenizer, requested training semantics, model source, and compute against the eligible `air_templates/` entries. Read [the recipe guide](references/recipe-selection.md) and select exactly one:

- `trl_lora` for unquantized bf16 LoRA when one base replica fits per worker.
- `trl_lora_fsdp` for unquantized bf16 LoRA when the base must be sharded.
- `trl_full_fsdp` for full-weight training with a validated transformer-layer wrap class.
- `custom` when no approved template preserves the required semantics.

When `peft_only: true`, exclude `trl_full_fsdp`. Choose `trl_lora` when a complete unquantized bf16 base replica plus activations and PEFT overhead fit on every requested accelerator; otherwise choose `trl_lora_fsdp` when the architecture supports the required sharding. Base this choice on inspected parameter count and explicit per-worker memory evidence, not the model name. If that evidence is missing or neither LoRA recipe is compatible, stop with a concrete blocker or select `custom`; never fall back to full-weight training.

When `peft_only: false`, preserve the recovered training semantics and use the normal full-weight-versus-PEFT selection rules. Record all eligible candidates, compatibility evidence, selected template, model-size calculation, and rationale. Never silently change requested compute or model-source precedence.

## Build the generation contract

Record the versioned legacy UC lineage URI, `peft_only`, selected `model_source`, exact training reference, recovered data and hyperparameters, outputs, compute, local-cache settings, and target model. Include the inspected parameter count, estimated bf16 base bytes, per-worker memory budget, and whether a full base replica fits. Mark whether a Hugging Face token is required; store only `<scope>/<key>`, never a secret value.

Require `train.yaml`, `train.py`, `helper_utils.py`, `training_utils.py`, `01_runner.py`, and `02_register_uc.py`, plus `merge.py` for LoRA. Set `plan.status: current` and `generation.status: stale`.

## Guardrails

- For Volume input, plan capacity for one node-local copy plus reserve.
- For system.ai, plan capacity for one node-local MLflow artifact download plus reserve.
- For Hugging Face, preserve the repository ID and conditional token injection.
- With `peft_only: true`, permit only `trl_lora` or `trl_lora_fsdp` and require model-size evidence for the choice.
- Return `custom` when approved recipes do not fit.
- Do not define or authorize training, registration, evaluation, promotion, or migration-completion stages.
