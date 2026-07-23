---
name: air-migrate-migrate-fmt-model
description: "Orchestrate an end-to-end migration of a Unity Catalog model produced by the legacy Databricks Foundation Model Fine-tuning API to a Databricks AI Runtime workflow. Use when a user asks to assess, plan, generate, run, validate, or resume the complete migration. Coordinates source inspection, strategy planning, template-based AIR code generation, training, target registration, and parity validation while preserving stage gates and evidence."
---

# Migrate FMT Model

Coordinate the migration as a resumable, evidence-driven workflow. Delegate stage details to the sibling skills and keep `migrate/output/migration-manifest.yaml` as the shared contract.

## Required stage skills

Read and follow these sibling skill instructions in order when installed together:

1. [`air-migrate-inspect-fmt-model`](../air-migrate-inspect-fmt-model/SKILL.md)
2. [`air-migrate-plan-air-training`](../air-migrate-plan-air-training/SKILL.md)
3. [`air-migrate-materialize-uc-model`](../air-migrate-materialize-uc-model/SKILL.md)
4. [`air-migrate-generate-air-job`](../air-migrate-generate-air-job/SKILL.md)
5. [`air-migrate-validate-model-migration`](../air-migrate-validate-model-migration/SKILL.md)

If a stage skill is unavailable, stop before that stage and identify the missing package; do not improvise its guarded workflow.

## Workflow

1. **Resolve configuration**
   - Read `migrate/config.yaml`.
   - Treat all-blank target fields as a new version at the source model name.
   - Reject partially populated target fields.
2. **Inspect**
   - Recover UC/MLflow lineage, data, task semantics, hyperparameters, artifacts, permissions, and unknowns.
   - Gate: require a viable retrain, continue, or repackage starting point.
3. **Plan**
   - Select migration semantics, objective, recipe, compute, checkpoint/output contract, intentional divergences, and acceptance criteria.
   - Gate: require an explicit plan and a supported training recipe, `none` for repackaging, or a documented custom-template blocker.
4. **Materialize existing weights when continuing**
   - For continue mode, use `air-migrate-materialize-uc-model` to download and structurally validate portable full weights and tokenizer files in a shared UC Volume.
   - For retraining and repackaging, record materialization as not required.
   - Gate: never pass a `models:/` URI directly to Axolotl; require current materialization provenance and worker-readable paths.
5. **Generate from templates**
   - For retraining or continued training, use `air-migrate-generate-air-job` to copy exactly one approved `air_templates/` recipe before customization.
   - For repackaging, skip AIR workload generation and record `generation.status: skipped` with the reason.
   - Never write `train.yaml`, `train.py`, or `01_runner.py` from scratch when an approved recipe is selected.
   - Gate: require template provenance, static validation, and AIR dry-run when available.
6. **Execute and register**
   - Submit smoke or training runs only to the extent authorized by the active request.
   - Preserve MLflow lineage, checkpoint evidence, output type, and the resolved target registration rule.
   - Do not overwrite an existing model version; UC registration creates a new version.
7. **Validate**
   - Compare source and target against the predeclared criteria on identical evaluation inputs.
   - Gate: do not claim completion until artifact integrity and the validation verdict are recorded.
8. **Report**
   - Summarize source and target URIs, selected mode/recipe/template, deliberate differences, runs, artifacts, validation verdict, remaining risks, and exact next action.

## Template routing invariant

The planner owns recipe selection; the generator only materializes it:

- `axolotl_qlora` -> `air_templates/axolotl_qlora`
- `axolotl_full_fsdp` -> `air_templates/axolotl_full_fsdp`
- `axolotl_qlora_fsdp` -> `air_templates/axolotl_qlora_fsdp`

Use no fallback. For `recipe: none`, skip generation only when `mode: repackage`. For `recipe: custom`, pause generation until a suitable template is added and validated.

## Resume and invalidation

Reuse a completed stage only when its inputs and evidence are still current. Reinspect when the source version, source run, config, permissions, or data changes. Replan when inspection changes. Rematerialize when a continue-mode source, destination, artifact, or tokenizer changes. Regenerate when the plan, materialization, or authoritative template changes. Revalidate when generated code, data, model artifacts, or acceptance criteria change.

## Safety and truthfulness

- Prefer read-only discovery before mutations.
- Never expose or copy secret values into code, manifests, or reports.
- Distinguish reproduction, continued training, and repackaging in every handoff.
- Call QLoRA an adapter-based migration, not full-weight parity.
- Surface unknown legacy-service behavior and unsupported architectures explicitly.
- Do not promote aliases, route serving traffic, delete artifacts, or retire the source model without separate explicit authorization.
