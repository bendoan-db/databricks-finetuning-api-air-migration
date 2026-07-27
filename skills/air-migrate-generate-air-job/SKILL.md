---
name: air-migrate-generate-air-job
description: "Generate the canonical AI Runtime handoff at migrate/output/air_workload from an approved TRL template. Enforces peft_only recipe constraints and configured Volume, system.ai, or Hugging Face source precedence; then stops after local validation."
---

# Generate AI Runtime Workload

Create the model-specific handoff under `migrate/output/air_workload`. Never submit or execute it.

## Prerequisites

Require a current manifest plan with one supported recipe, compatibility evidence, matching compute, resolved target, and the same selected source and boolean `peft_only` as `migrate/config.yaml`. Source precedence is `weights_volume_path`, then `system_ai_model_uri`, then `huggingface_model_id`. Read [the template catalog](references/template-catalog.md).

When `peft_only` is true, require the plan recipe to be `trl_lora` or `trl_lora_fsdp` and require recorded model-size/per-worker-memory evidence for that choice. Reject `trl_full_fsdp`; do not change the planned recipe during generation. The materializer independently enforces this constraint before creating output files.

## Copy and customize

Run:

```bash
python3 skills/air-migrate-generate-air-job/scripts/materialize_air_template.py \
  --recipe <planned-recipe>
```

The destination must be empty. The script copies the exact template and pins compute, experiment, legacy source URI, registration target, and selected training input:

- `volume`: use the configured path for model and tokenizer; remove `HF_TOKEN`.
- `system_ai`: use the versioned `models:/system.ai...` URI; remove `HF_TOKEN`; download into node-local cache only when the operator runs training.
- `hugging_face`: use the repository ID; retain the configured `HF_TOKEN` secret only for gated access.

Customize only the generated copy for recovered data, serialization, hyperparameters, output paths, and model-specific FSDP or LoRA values. Preserve launcher, source staging, checkpoint, merge, and registration contracts.

## Validate locally and stop

1. Parse `train.yaml` and load its training configuration without loading the model.
2. Compile every Python file.
3. Confirm `train.py` defines only `run_training` and `main` and imports colocated helpers.
4. Confirm compute and `torchrun --nproc_per_node` agree.
5. Confirm `source_model_uri` is the legacy versioned URI and the generated source tuple matches configuration precedence.
6. Confirm `peft_only` matches configuration and manifest; when true, confirm the recipe is LoRA and matches the recorded model-size verdict.
7. Confirm Volume and system.ai inputs have no HF secret; gated Hugging Face has exactly the configured secret reference; public Hugging Face has none.
8. Confirm experiment, target, data, outputs, notebooks, required files, and template provenance are consistent and contain no secret values.

Do not run `air run --dry-run`; it stages external files and is outside this generation-only boundary. Do not execute training, merge, registration, smoke tests, evaluation, or promotion.

Update `generation` with template provenance, `peft_only`, source mode, files, customized fields, and local validations. Set `status: current` and `handoff_ready: true` only after all checks pass. This terminal success state does not mean a model was trained or registered.
