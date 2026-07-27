# Model-source selection contract

The current migration is generation-only. Resolve one training input directly from `migrate/config.yaml`; do not call the retired materialization scripts.

## Precedence

| Configuration | Selected source | Generated model/tokenizer reference |
|---|---|---|
| Nonblank `source.weights_volume_path` | `volume` | Exact absolute `/Volumes/...` path |
| Blank Volume path and nonblank `source.system_ai_model_uri` | `system_ai` | Exact `models:/system.ai.<model>/<version>` URI |
| Both blank | `hugging_face` | Required `source.huggingface_model_id` |

Higher-precedence fields win even when lower-precedence fallbacks are also populated. Reject the configuration when no valid source remains.

`source_model_uri` always records the versioned legacy model (`models:/<catalog>.<schema>.<model>/<version>`) for lineage. It is distinct from `model_name` and `tokenizer_path`, which identify the selected training input.

## Generated fields

```yaml
parameters:
  training_config:
    model_source: system_ai  # volume | system_ai | hugging_face
    requires_hf_token: false
    source_model_uri: models:/catalog.schema.legacy_model/2
    model_name: models:/system.ai.base_model/3
    tokenizer_path: models:/system.ai.base_model/3
```

For gated Hugging Face input, set `requires_hf_token: true` and retain exactly the configured `secrets.HF_TOKEN: <scope>/<key>`. For public Hugging Face, Volume, and system.ai input, set it to false and remove the HF secret.

## Operator-time behavior

- `volume`: copy the checkpoint once per AIR node into the locked ephemeral cache and load locally.
- `system_ai`: use MLflow to download the exact artifact once per node into the same cache, select one portable checkpoint, and load locally.
- `hugging_face`: let Transformers download the repository; use `HF_TOKEN` only when configured.

These actions occur only when the receiving operator runs the generated workload. Inspection, planning, and generation never download, copy, train, register, or evaluate model weights.
