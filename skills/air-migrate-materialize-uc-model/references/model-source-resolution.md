# Training weight-source resolution

Resolve the initialization checkpoint before generating an AIR workload. The fields in `migrate/config.yaml` are authoritative:

Require boolean `source.run_full_migration` and copy it unchanged into every resolver result. Also require and preserve the absolute `source.migration_experiment_path`. It controls the MLflow experiment used by every generated AIR workload and does not change the weight-source decision below. The execution gate controls whether orchestration stops after preflight; it does not prevent required source materialization.

| `source.use_existing_weights` | `source.existing_weights_volume_location` | Resolution |
|---|---|---|
| `true` | Populated absolute `/Volumes/...` path | Continue training from the validated provided full checkpoint. Preserve the configured UC version as lineage; do not materialize it. |
| `true` | Blank | Continue training from the exact configured legacy UC version after AIR materialization. |
| `false` | Blank | Retrain from the recovered original base model. Prefer an exact `system.ai` registered-model match; otherwise use its Hugging Face model ID. |
| `false` | Populated | Invalid configuration; stop instead of ignoring the path. |

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
  "run_full_migration": false,
  "migration_experiment_path": "/Shared/fmt-migration",
  "model_source": "existing_uc",
  "source_model_uri": "models:/catalog.schema.model/2",
  "existing_weights_volume_location": "/Volumes/catalog/schema/volume/checkpoint",
  "model_name": "/Volumes/catalog/schema/volume/checkpoint",
  "tokenizer_path": "/Volumes/catalog/schema/volume/checkpoint",
  "requires_materialization": false,
  "requires_volume_validation": true
}
```

```json
{
  "use_existing_weights": true,
  "run_full_migration": false,
  "migration_experiment_path": "/Shared/fmt-migration",
  "model_source": "existing_uc",
  "source_model_uri": "models:/catalog.schema.model/2",
  "existing_weights_volume_location": null,
  "requires_materialization": true,
  "requires_volume_validation": false
}
```

```json
{
  "use_existing_weights": false,
  "run_full_migration": true,
  "migration_experiment_path": "/Shared/fmt-migration",
  "model_source": "system_ai",
  "source_model_uri": "models:/system.ai.base_model/3",
  "requires_materialization": true
}
```

```json
{
  "use_existing_weights": false,
  "run_full_migration": true,
  "migration_experiment_path": "/Shared/fmt-migration",
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

Preserve `run_full_migration` in source-resolution and orchestration provenance; the training entry point need not consume it. Copy these weight-source fields into every generated `train.yaml`:

- `use_existing_weights`
- `model_source`
- `source_model_uri`
- `existing_weights_volume_location`
- `model_name`
- `tokenizer_path`

Copy `source.migration_experiment_path` separately into both AIR `experiment_name` and `training_config.experiment_path`. Ensure/create the experiment before materialization or training submission and record its MLflow experiment ID.

For a supplied existing-weights path, structurally validate that exact directory and copy it into both `model_name` and `tokenizer_path`; do not submit a materialization job. For other `existing_uc` and `system_ai` sources, use `prepare_air_materialization.py` to stage the pinned model through AIR with the authoritative `migrate/config.yaml.compute` block. Require the materialization AIR run to finish with `SUCCESS` and preserve its numeric run ID and persistent JSON inventory before setting the resulting paths. Never pass `models:/...` directly to Transformers or TRL, and do not perform a large download on local or standard notebook compute. For `hugging_face`, use the recovered repository ID and retain the `HF_TOKEN` secret when the model is gated.

For every resolved `/Volumes/...` model or tokenizer path, preserve that path as durable provenance and require the generated training module to prefetch it into its configured ephemeral node-local cache before Transformers loads it. Size each node from the provided or materialized inventory. This prefetch does not change `model_source`, `source_model_uri`, or materialization status. Hugging Face repository IDs bypass the Volume-specific prefetch.
