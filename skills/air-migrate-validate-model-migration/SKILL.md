---
name: air-migrate-validate-model-migration
description: "Validate a migration from the legacy Databricks Foundation Model Fine-tuning API to Databricks AI Runtime by checking workload configuration, executing authorized smoke or full runs, verifying artifacts and Unity Catalog registration, and comparing source and migrated models—including assistant response-token accuracy—on shared evaluation data. Use for preflight checks, parity testing, regression analysis, migration acceptance, or promotion decisions."
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
   - Confirm `use_existing_weights` matches the migration config and the generated source tuple is valid. For true, verify the configured legacy URI and materialized files; for false, verify the pinned `system.ai` resolution or documented no-match Hugging Face fallback.
2. **Execution health**
   - When execution is authorized, run a bounded smoke workload first. Confirm distributed initialization, data loading, forward/backward pass, metric logging, checkpoint write/read, and clean worker completion.
   - Inspect logs for OOMs, NaNs, rank divergence, hidden retries, dropped configuration fields, and unexpectedly unused parameters.
3. **Artifact integrity**
   - Verify tokenizer/config files, training adapters, merged full-weight shards, checkpoint completeness, merge lineage for PEFT, loadability on a clean process, and target UC model lineage.
   - For either UC-backed initialization, confirm the input weight inventory, tokenizer path, pinned model URI/run ID, worker readability, and generated `base_model`/`tokenizer_config` values match the materialization record.
4. **Model comparison**
   - Evaluate source and migrated models on the same immutable examples with identical prompts, decoding parameters, and scoring code.
   - Use [`air-migrate-compare-token-accuracy`](../air-migrate-compare-token-accuracy/SKILL.md) to compute deterministic assistant response-token accuracy for both portable checkpoints. Resolve the legacy model version from `migrate/config.yaml`, and resolve the exact migrated version from registration evidence.
   - Require matching tokenizer/chat serialization for a directly comparable token-accuracy verdict. If tokenizations differ, record both measurements and mark this criterion `inconclusive`.
   - Compare loss/perplexity where meaningful, task metrics, structured-output validity, safety/format regressions, latency, and qualitative edge cases.
   - Use statistical uncertainty or repeated runs when sampling is involved.
5. **Promotion decision**
   - Evaluate every planned criterion. Return `pass`, `fail`, or `inconclusive`, with evidence and remediation. Do not collapse missing evidence into a pass.

Read [the validation contract](references/validation-contract.md) when creating the report.

## Parity interpretation

- Full FSDP can target full-weight behavioral and metric parity, but hidden legacy details normally prevent a byte-identical guarantee.
- QLoRA updates only adapters and cannot satisfy full-weight-training parity. Its registered artifact is a merged full inference checkpoint; validate that the merge equals the configured base plus adapter and evaluate it as a behavioral replacement.
- Continued training must be compared as a new model, not represented as reproduction.

## Output

Write `migrate/output/migration-validation.yaml`, preserve the token-accuracy evidence at `migrate/output/token-accuracy-evaluation.json` (or a versioned equivalent), update the manifest validation section with both paths and the verdict, and include links or identifiers for MLflow runs, AIR jobs, artifacts, and registered model versions. Preserve failed-run evidence.

Do not register an alias, redirect traffic, or remove the source model unless the active user request explicitly authorizes that separate promotion action.
