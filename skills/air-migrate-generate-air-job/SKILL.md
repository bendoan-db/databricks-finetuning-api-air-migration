---
name: air-migrate-generate-air-job
description: "Generate a Databricks AI Runtime fine-tuning workload from an approved migration plan by materializing and adapting the repository's Axolotl QLoRA, full-FSDP, or QLoRA-FSDP templates. Use when creating or refreshing train.yaml, train.py, and a Databricks runner notebook for a legacy Foundation Model Fine-tuning API migration. Enforces template provenance, safe customization, configuration parity, and static/AIR dry-run validation."
---

# Generate AI Runtime Job

Create AIR code from the approved templates. Do not synthesize the three workload files from scratch.

## Prerequisites

Read `migrate/output/migration-manifest.yaml` and require `plan.status: current`. If the plan is missing or stale, return to `air-migrate-plan-air-training`. If the plan uses `recipe: custom`, stop and report the template capability that must be added. If it uses `recipe: none`, skip this skill because the migration is a repackage operation. Never silently choose a recipe during generation.

Require `plan.input_model.use_existing_weights` to equal `source.use_existing_weights`. For `materialized_uc_model` or `materialized_system_ai`, additionally require `materialization.status: current` from `air-migrate-materialize-uc-model`, plus absolute worker-accessible `model_path` and `tokenizer_path`; reject `models:/` URIs and local driver-only paths. For `hugging_face`, require the recovered repository ID and a successful recorded `system.ai` lookup with no match.

Read [the template catalog](references/template-catalog.md) before materializing a workload.

## Materialize the selected template

Locate the repository root containing `air_templates/`. Run:

```bash
python3 skills/air-migrate-generate-air-job/scripts/materialize_air_template.py \
  --recipe <manifest-plan-recipe> \
  --output-dir <new-workload-directory>
```

Use `--template-root` only when the template directory is outside the detected repository root. The script refuses nonempty output directories and verifies that the source contains exactly the required workload files.

The recipe mapping is exact:

- `axolotl_qlora` -> `air_templates/axolotl_qlora`
- `axolotl_full_fsdp` -> `air_templates/axolotl_full_fsdp`
- `axolotl_qlora_fsdp` -> `air_templates/axolotl_qlora_fsdp`

If no approved template fits, stop and request a template extension. Do not use the nearest recipe.

## Customize the copy

Preserve the template's structure, launcher contract, distributed initialization, Axolotl configuration construction, validations, MLflow behavior, and checkpoint semantics. Make the smallest necessary edits.

Customize:

- Model name/revision and tokenizer/chat template
- `use_existing_weights` copied unchanged from `migrate/config.yaml`
- `model_source` mapped exactly as `materialized_uc_model -> existing_uc`, `materialized_system_ai -> system_ai`, or `hugging_face -> hugging_face`
- The pinned `source_model_uri`, or null only for Hugging Face
- For either materialized source, `model_name: <materialization.model_path>` and `tokenizer_path: <materialization.tokenizer_path>`; for Hugging Face, the recovered repository ID
- Train/eval UC Volume paths and data-format settings
- Adapter, merged-model, and checkpoint paths; experiment name; and MLflow run name
- `registered_model_name` as the resolved `<catalog>.<schema>.<model>` target
- Databricks secret references and environment variable names, never values
- GPU count/runtime settings justified by the plan
- Recovered or intentionally changed hyperparameters
- Runner title, defaults, explanatory text, and configuration so it remains equivalent to `train.yaml`

Prefer YAML parameter changes. Edit `train.py` only when the required behavior cannot be expressed through its existing parameters, and retain its safety validations. Keep `train.yaml`, `train.py`, and `01_runner.py` consistent.

Remove the `HF_TOKEN` AIR secret and notebook requirement for `existing_uc` and `system_ai` only when the model and tokenizer are both materialized local/Volume paths and no other Hugging Face download is required. Retain it for gated Hugging Face sources.

Do not remove recipe invariants:

- QLoRA: PEFT adapter, 4-bit loading, no accidental full-weight training.
- Full FSDP: no quantization or adapter, full sharding, portable full-model output.
- QLoRA-FSDP: PEFT plus 4-bit loading, FSDP-compatible placement/wrapping, CPU-efficient loading, and the template's checkpointing constraints.

## Validate before submission

1. Compile both Python files.
2. Parse YAML and assert all required workload and training fields.
3. Exercise configuration loading/translation without downloading the model when possible.
4. Search for placeholder paths, hard-coded tokens, source-template experiment names, and inconsistent values across YAML and runner.
5. Run `air run --dry-run --file train.yaml` from the generated directory when the AIR CLI is available.
6. Confirm output type matches the plan: full model for full FSDP; adapter training output plus a separate merged full checkpoint for QLoRA variants.
7. Assert the YAML source tuple is internally consistent: `(use_existing_weights=true, existing_uc, configured source URI, Volume paths)` or `(false, system_ai, pinned system URI, Volume paths)` or `(false, hugging_face, null URI, remote repository ID)`.
8. For materialized sources, verify every worker can read the model/tokenizer and that Axolotl uses them as `base_model` and `tokenizer_config`.
9. Confirm the runner reads `registered_model_name` from YAML, logs input-source provenance and the final artifact to the training run, and uses the `databricks-uc` registry. QLoRA runners must safely merge the adapter into an unquantized base model and register only the resulting full checkpoint. Full-FSDP registrations must exclude intermediate checkpoints and optimizer state.
10. Do not submit a training run or create a UC model version unless the active user request authorizes execution.

## Record provenance

Update `generation` in the manifest:

```yaml
generation:
  status: current
  recipe: axolotl_full_fsdp
  template_path: air_templates/axolotl_full_fsdp
  output_path: migrate/output/air_workload
  files: [train.yaml, train.py, 01_runner.py]
  customized_fields: []
  input_model:
    use_existing_weights: true
    source: materialized_uc_model
    source_model_uri: models:/catalog.schema.model/1
    model_path: /Volumes/catalog/schema/volume/source-model-v1/model
    tokenizer_path: /Volumes/catalog/schema/volume/source-model-v1/model
  validations: []
```

Set `validation.status: stale` whenever generated workload content changes.
