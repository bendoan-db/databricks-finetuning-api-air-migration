---
name: air-migrate-validate-model-migration
description: "Validate a migration from the legacy Databricks Foundation Model Fine-tuning API to Databricks AI Runtime by checking workload configuration, executing authorized smoke or full runs, verifying artifacts and Unity Catalog registration, and comparing source and migrated models on shared evaluation data. Use for preflight checks, parity testing, regression analysis, migration acceptance, or promotion decisions."
---

# Validate Model Migration

Produce an auditable verdict against criteria chosen before training. Separate workload validity, training health, artifact integrity, and model quality.

## Inputs

Require:

- `migrate/output/migration-manifest.yaml` with current plan and generation sections
- Generated AIR workload and its dry-run evidence
- Source registered model and resolved target
- Shared evaluation dataset and acceptance criteria
- Current materialization provenance when `plan.mode: continue`

If acceptance criteria are absent, report measurements without inventing a pass threshold.

## Validation layers

1. **Static preflight**
   - Compile Python, parse YAML, validate template provenance, compare YAML and notebook settings, inspect dependencies/secrets, and run AIR dry-run when available.
   - Confirm dataset paths/schema, tokenizer/chat format, assistant-only masking, effective batch, output kind, checkpoint contract, and target location.
2. **Execution health**
   - When execution is authorized, run a bounded smoke workload first. Confirm distributed initialization, data loading, forward/backward pass, metric logging, checkpoint write/read, and clean worker completion.
   - Inspect logs for OOMs, NaNs, rank divergence, hidden retries, dropped configuration fields, and unexpectedly unused parameters.
3. **Artifact integrity**
   - Verify tokenizer/config files, full-weight shards or adapters, checkpoint completeness, base-model dependency for PEFT, loadability on a clean process, and target UC model lineage.
   - For continued training, confirm the input weight inventory, tokenizer path, source UC model URI/run ID, worker readability, and generated `base_model`/`tokenizer_config` values match the materialization record.
4. **Model comparison**
   - Evaluate source and migrated models on the same immutable examples with identical prompts, decoding parameters, and scoring code.
   - Compare loss/perplexity where meaningful, task metrics, structured-output validity, safety/format regressions, latency, and qualitative edge cases.
   - Use statistical uncertainty or repeated runs when sampling is involved.
5. **Promotion decision**
   - Evaluate every planned criterion. Return `pass`, `fail`, or `inconclusive`, with evidence and remediation. Do not collapse missing evidence into a pass.

Read [the validation contract](references/validation-contract.md) when creating the report.

## Parity interpretation

- Full FSDP can target full-weight behavioral and metric parity, but hidden legacy details normally prevent a byte-identical guarantee.
- QLoRA produces adapters and cannot satisfy structural full-weight parity. Evaluate it as a behavioral replacement unless a separately validated merge is part of the plan.
- Continued training must be compared as a new model, not represented as reproduction.

## Output

Write `migrate/output/migration-validation.yaml`, update the manifest validation section with the report path and verdict, and include links or identifiers for MLflow runs, AIR jobs, artifacts, and registered model versions. Preserve failed-run evidence.

Do not register an alias, redirect traffic, or remove the source model unless the active user request explicitly authorizes that separate promotion action.
