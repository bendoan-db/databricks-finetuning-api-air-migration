# Training weight-source resolution

Resolve the initialization checkpoint before generating an AIR workload. The boolean in `migrate/config.yaml` is authoritative:

| `source.use_existing_weights` | Resolution |
|---|---|
| `true` | Continue training from the exact configured legacy UC model version. Materialize its portable full weights and tokenizer. |
| `false` | Retrain from the recovered original base model. Prefer an exact `system.ai` registered-model match; otherwise use its Hugging Face model ID. |

Run:

```bash
python3 skills/air-migrate-materialize-uc-model/scripts/resolve_training_source.py \
  --config migrate/config.yaml \
  --base-model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output migrate/output/training-source.json
```

The base model ID must come from inspection evidence, not from a template default. The resolver uses the UC registry and returns one of these contracts:

```json
{
  "use_existing_weights": true,
  "model_source": "existing_uc",
  "source_model_uri": "models:/catalog.schema.model/2",
  "requires_materialization": true
}
```

```json
{
  "use_existing_weights": false,
  "model_source": "system_ai",
  "source_model_uri": "models:/system.ai.base_model/3",
  "requires_materialization": true
}
```

```json
{
  "use_existing_weights": false,
  "model_source": "hugging_face",
  "source_model_uri": null,
  "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "requires_materialization": false,
  "requires_hf_token": true
}
```

## `system.ai` matching

Search only registered models under `system.ai`. Accept an exact known model-ID tag match first, then an exact or normalized registered-model suffix match. Reject tied matches. Pin the highest READY version and preserve its exact URI.

If the `system.ai` query fails, stop instead of treating an access or service failure as proof that the model is absent. Fall back to Hugging Face only when the query succeeds and returns no match. If a matched `system.ai` artifact is not a portable Hugging Face checkpoint, stop and surface the incompatibility; do not silently change the resolved source after planning.

## Generated workload contract

Copy the resolution into every generated `train.yaml`:

- `use_existing_weights`
- `model_source`
- `source_model_uri`
- `model_name`
- `tokenizer_path`

For `existing_uc` and `system_ai`, `model_name` and `tokenizer_path` must be materialized `/Volumes/...` paths. Never pass `models:/...` directly to Axolotl. For `hugging_face`, use the recovered repository ID and retain the `HF_TOKEN` secret when the model is gated.
