# Token-accuracy metric and evidence contract

## Metric semantics

`assistant_response_token_accuracy` is:

```text
correct argmax next-token predictions / scored assistant-turn tokens
```

Score the model with teacher forcing over complete chat records. Exclude system and user prompt tokens. Include tokens emitted by the chat template to terminate each assistant turn because those are part of the causal-LM training target. Weight the aggregate by token count; do not average per-record percentages.

The metric is deterministic for fixed weights, tokenizer files, chat template, template arguments, input records, sequence-length policy, and inference implementation. It does not run generation or use sampling parameters.

This is not an output-level MLflow GenAI scorer. It requires local checkpoint logits and aggregate token counts; `mlflow.genai.evaluate()` operates on generated outputs or traces. The evaluator must create or reuse a run in `source.migration_experiment_path`, log the resulting JSON as an artifact, and mirror the token-weighted aggregate accuracies as MLflow metrics. It must not replace weighted counts with an unweighted mean of per-record scorer values.

## Shared-input requirements

- Evaluate the exact same ordered JSONL bytes for both models and store the SHA-256 digest.
- Require a `messages` array with string `role` and `content` fields and at least one assistant turn.
- Pass the same fixed template date to both tokenizers so templates cannot inject the wall-clock date.
- Default to rejecting over-length records. If left truncation is part of the approved policy, record every truncated token count.
- Reject a record when chat serialization does not yield unambiguous assistant spans or no scored assistant token remains.

Each model uses the tokenizer packaged with its own registered checkpoint. A paired comparison is valid only when tokenizer vocabularies, special-token maps, chat templates, serialized token IDs, and assistant masks agree. Preserve measurements but mark the verdict `inconclusive` when any of these differ.

## Acceptance semantics

Let `legacy_accuracy` be `L`, `migrated_accuracy` be `M`, and the predeclared maximum absolute regression be `R`:

```text
pass when M >= L - R
fail when M < L - R
```

Report `M - L` as `absolute_accuracy_delta`. Report `(M - L) / L` as `relative_accuracy_delta` when `L` is nonzero. `R=0` requires the migrated result to meet or exceed the legacy result. Without a predeclared `R`, return `inconclusive` rather than inventing a threshold after seeing results.

## JSON evidence shape

The evaluator writes this stable top-level structure:

```json
{
  "schema_version": 1,
  "generated_at": "2026-07-23T00:00:00+00:00",
  "mlflow": {
    "experiment_path": "/Shared/fmt-migration",
    "experiment_id": "456",
    "run_id": "abc123"
  },
  "metric": {
    "name": "assistant_response_token_accuracy",
    "direction": "higher_is_better"
  },
  "inputs": {
    "config_path": "migrate/config.yaml",
    "evaluation_data_path": "/Volumes/catalog/schema/volume/eval.jsonl",
    "evaluation_data_sha256": "...",
    "record_count": 100
  },
  "settings": {},
  "models": {
    "legacy": {
      "model_uri": "models:/catalog.schema.model/1",
      "correct_tokens": 900,
      "scored_tokens": 1000,
      "accuracy": 0.9,
      "tokenizer": {},
      "records": []
    },
    "migrated": {
      "model_uri": "models:/catalog.schema.model/2",
      "correct_tokens": 910,
      "scored_tokens": 1000,
      "accuracy": 0.91,
      "tokenizer": {},
      "records": []
    }
  },
  "comparison": {
    "directly_comparable": true,
    "absolute_accuracy_delta": 0.01,
    "relative_accuracy_delta": 0.0111111111,
    "max_accuracy_regression": 0.01,
    "verdict": "pass"
  },
  "risks": []
}
```

Per-record evidence contains line number, record hash, input/scored/correct token counts, truncation count, accuracy, and tokenization hash. It intentionally omits prompts, responses, token IDs, and logits.

## Failure conditions

Stop without writing a result when:

- the config does not resolve an exact legacy UC version;
- the migrated URI is absent, unversioned, or identical to the source URI;
- either artifact lacks a portable full checkpoint or tokenizer;
- evaluation JSONL is invalid or contains no assistant response;
- a chat template is absent or assistant boundaries are ambiguous;
- a record exceeds the approved length policy or has no scorable tokens;
- inference fails or either model produces zero scored tokens.

Treat out-of-memory errors as execution failures. Change batch size or compute and rerun to a new evidence path; do not silently reduce the dataset.

When the evaluator owns its MLflow run, mark that run failed before propagating an error. When AIR or a parent context injected the active run, require its experiment ID to match and let that owner record terminal status.
