---
name: air-migrate-inspect-fmt-model
description: "Inspect a Unity Catalog model version created by the legacy Databricks Foundation Model Fine-tuning API and recover its MLflow lineage, datasets, task, base model, hyperparameters, checkpoints, tokenizer, permissions, and portable artifacts into a migration manifest. Use when assessing or beginning a migration from legacy foundation-model fine-tuning to Databricks AI Runtime, investigating whether a model can be reproduced, or refreshing migration metadata."
---

# Inspect FMT Model

Produce an evidence-backed `migration-manifest.yaml` that downstream planning, generation, and validation can consume without rediscovering the source run.

## Workflow

1. Locate the repository root and read `migrate/config.yaml`.
2. Require `source.catalog`, `source.schema`, `source.model`, positive integer `source.version`, and boolean `source.use_existing_weights`.
3. Resolve the target:
   - If every target field is blank, use the source catalog, schema, and model.
   - If every target field is populated, use the explicit target.
   - Reject a partially populated target.
4. Verify access to the source registered model version and record its status, URI, creation time, aliases, description, tags, and originating `run_id`.
5. Follow `run_id` into MLflow. Record run metadata, parameters, metrics, tags, dataset inputs, and artifact inventory. Inspect likely training, checkpoint, tokenizer, configuration, and model directories; do not download large weight files unless required.
6. Recover the original training contract:
   - Base model and revision
   - Task type and prompt/chat format
   - Train and evaluation locations and schemas
   - Tokenizer, chat template, sequence length, packing, truncation, and loss masking
   - Optimizer, scheduler, learning rate, warmup, epochs or token budget, batch sizes, gradient accumulation, precision, seed, and distributed strategy
   - Final full weights, adapter artifacts, and resumable checkpoint availability
   - When `use_existing_weights` is false, the exact Hugging Face base-model ID needed for `system.ai` lookup or remote download
7. Check read access to data and artifacts plus create/register permissions at the resolved target. Never persist credentials or secret values.
8. Assign each recovered value a provenance and confidence. Use `observed`, `derived`, `inferred`, or `unknown`; never convert an unknown into an unmarked guess.
9. Write `migrate/output/migration-manifest.yaml` using [the manifest contract](references/manifest-schema.md). Preserve previous planner and generator sections only when their inputs remain unchanged; otherwise mark them stale.
10. Report blockers, missing metadata, and the portable-artifact conclusion.

## Inspection rules

- Treat the UC model version as the entry point, not necessarily the training artifact itself.
- Treat `source.use_existing_weights` as authoritative initialization intent: true selects the configured UC weights; false selects the original base model.
- Prefer MLflow run inputs, artifacts, and logged parameters over model-card prose or filename inference.
- Record artifact metadata and checksums when practical. Avoid loading model weights merely to list them.
- Distinguish full Hugging Face weights, PEFT adapters, optimizer checkpoints, and serving-only packages.
- Treat a missing original dataset, base-model identity, or task-format contract as a reproducibility risk.
- Do not claim byte-for-byte recoverability when the legacy service hid data ordering, packing, optimizer, or distributed details.
- Make all read-only discovery possible before asking for additional access.

## Completion criteria

Finish only when the manifest identifies the resolved source and target, lineage evidence, data, artifacts, recovered training contract, permissions, unknowns, and a supported migration starting point or a concrete blocker.
