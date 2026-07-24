---
name: air-migrate-compare-token-accuracy
description: "Compare a newly migrated Databricks AI Runtime model with the legacy Unity Catalog model version declared in migrate/config.yaml by measuring deterministic assistant response-token accuracy on shared chat-completion evaluation data. Use when validating FMT-to-AIR model quality, checking a token-accuracy regression threshold, producing paired evaluation evidence, or investigating whether tokenizer and chat-template differences prevent a valid comparison."
---

# Compare Token Accuracy

Measure teacher-forced next-token accuracy for the legacy and migrated models on the same immutable evaluation records. Treat this as a model-quality stage inside `air-migrate-validate-model-migration`, not as a replacement for artifact, registration, or behavioral checks.

## Preconditions

Require:

- `migrate/config.yaml`, whose `source` identifies the exact legacy UC model version
- Its required `source.migration_experiment_path`, with current existence/create evidence; every legacy/target materialization AIR submission and token-accuracy MLflow run must use it
- `source.run_full_migration: true`; refuse model comparison in preparation-only mode
- A current migration manifest containing the exact migrated UC model version; never resolve an unpinned alias or guess the latest version
- One shared JSONL evaluation file containing text-only `messages` arrays and at least one assistant turn per record
- Portable full Hugging Face checkpoints and tokenizers for both registered versions
- A predeclared maximum absolute accuracy regression when a pass/fail verdict is required

Token accuracy requires causal-LM logits. Reject serving-only PyFunc artifacts and adapter-only PEFT artifacts. PEFT outputs must be merged into the unquantized base model before evaluation.

## Resolve both model versions

Resolve the source from the config and the target from completed registration evidence. When `source.existing_weights_volume_location` is populated, structurally validate and reuse that exact directory as both the legacy model and tokenizer path; do not materialize the legacy UC version. Materialize only the target through AIR.

If `source.run_full_migration` is false, stop before target materialization or GPU evaluation. The orchestrator must record preflight validation instead; a migrated target does not exist in this execution mode.

When the source path is blank, use `air-migrate-materialize-uc-model` to generate separate AIR workloads for both checkpoints:

```bash
python3 skills/air-migrate-materialize-uc-model/scripts/prepare_air_materialization.py \
  --purpose validation \
  --config migrate/config.yaml \
  --output-dir /Volumes/<catalog>/<schema>/<volume>/token-eval/legacy-v<version> \
  --workload-dir migrate/output/air_materialization-legacy-v<version>

python3 skills/air-migrate-materialize-uc-model/scripts/prepare_air_materialization.py \
  --purpose validation \
  --config migrate/config.yaml \
  --model-uri models:/<target.catalog>.<target.schema>.<target.model>/<target-version> \
  --output-dir /Volumes/<catalog>/<schema>/<volume>/token-eval/migrated-v<version> \
  --workload-dir migrate/output/air_materialization-migrated-v<version>

cd migrate/output/air_materialization-legacy-v<version>
COPYFILE_DISABLE=1 air run --dry-run --file materialize.yaml -p DEFAULT
COPYFILE_DISABLE=1 air run --file materialize.yaml -p DEFAULT --watch

cd ../air_materialization-migrated-v<version>
COPYFILE_DISABLE=1 air run --dry-run --file materialize.yaml -p DEFAULT
COPYFILE_DISABLE=1 air run --file materialize.yaml -p DEFAULT --watch
```

For the blank-path branch, require both AIR runs to report `SUCCESS` and record their numeric run IDs and inventories. For the provided-path branch, require current no-download structural validation and `READ VOLUME` evidence for the source, plus `SUCCESS` and inventory evidence for the target materialization. In an orchestrated migration, `source.run_full_migration: true` authorizes these operations; for direct stage invocation, require equivalent authorization in the active request. Reuse a prior path only when its configured location, model URI, and file inventory remain current.

## Run the comparison

Read [the metric and evidence contract](references/token-accuracy-contract.md), then run the bundled evaluator on GPU compute with enough memory to load each model sequentially:

```bash
python3 skills/air-migrate-compare-token-accuracy/scripts/compare_token_accuracy.py \
  --config migrate/config.yaml \
  --legacy-model-path /Volumes/<catalog>/<schema>/<volume>/token-eval/legacy-v1/model \
  --legacy-tokenizer-path /Volumes/<catalog>/<schema>/<volume>/token-eval/legacy-v1/model \
  --migrated-model-uri models:/<catalog>.<schema>.<model>/<new-version> \
  --migrated-model-path /Volumes/<catalog>/<schema>/<volume>/token-eval/migrated-v2/model \
  --migrated-tokenizer-path /Volumes/<catalog>/<schema>/<volume>/token-eval/migrated-v2/model \
  --eval-data /Volumes/<catalog>/<schema>/<volume>/data/eval.jsonl \
  --max-sequence-length 4096 \
  --max-accuracy-regression 0.01
```

Omit `--max-accuracy-regression` when the plan has no approved threshold. The output will be measurements with an `inconclusive` verdict. The default truncation policy is `error`; use `--truncation left` only when that evaluation policy was declared before scoring.

The script:

1. Resolves the legacy URI exclusively from `migrate/config.yaml`.
2. Calls `mlflow.set_experiment` for `source.migration_experiment_path`, creating the experiment when absent. It reuses an active or AIR-injected run only when its experiment ID matches; otherwise it fails before scoring.
3. Refuses adapter-only or incomplete local checkpoints.
4. Hashes the exact evaluation file and each record.
5. Uses each checkpoint's local tokenizer and chat template with a fixed template date.
6. Finds assistant token spans by comparing generation-prompt and completed-turn serialization.
7. Loads one model at a time and scores deterministic argmax next-token predictions.
8. Logs inputs, both accuracies, deltas, token counts, verdict, and the completed JSON evidence artifact to the migration MLflow run.
9. Reports token counts, weighted aggregate accuracies, absolute and relative deltas, tokenizer fingerprints, per-record evidence, and a verdict.

## Interpret comparability

Direct comparison requires matching vocabulary, special-token mapping, chat template, serialized token IDs, and assistant masks. If they differ, preserve both measured accuracies but return `inconclusive`: token accuracy is tokenizer-dependent and the values are not strictly paired.

Do not reinterpret a positive delta as proof of general behavioral superiority. Token accuracy measures agreement with reference responses under teacher forcing; retain task metrics, format/safety cases, and artifact checks in the broader validation report.

Do not implement this metric as an `mlflow.genai.evaluate()` output scorer: that interface evaluates generated outputs or traces and does not expose the causal-LM logits needed here. The validation workflow may log the completed JSON file as an MLflow artifact and log its aggregate values as metrics, while keeping this file as the authoritative token-weighted evidence.

## Record the result

The default evidence path is `migrate/output/token-accuracy-evaluation.json`. Refusing to overwrite prior evidence is intentional; choose a versioned output path for another run.

Preserve the evaluation MLflow experiment path, experiment ID, and run ID embedded in that evidence. Resolve the run during final validation and require its experiment ID to equal the configured migration experiment ID.

Add its summary and path to `migrate/output/migration-validation.yaml` under `evaluation.token_accuracy`, then update the migration manifest:

```yaml
validation:
  status: current
  scope: final
  migration_complete: true
  report: migrate/output/migration-validation.yaml
  token_accuracy_evidence: migrate/output/token-accuracy-evaluation.json
  verdict: pass  # pass | fail | inconclusive
```

The overall migration verdict cannot be `pass` when token accuracy is a required criterion and this comparison failed or is inconclusive.
