# Portable checkpoint contract

A checkpoint is eligible for direct AI Runtime continued training only when all conditions below hold.

When `source.existing_weights_volume_location` is populated, the configured directory itself must satisfy this contract for both model and tokenizer. Validate it structurally without copying or loading tensors. Do not search another location or fall back to downloading the configured UC version.

## Model files

- `config.json` is valid JSON and identifies a model type or architecture.
- Full model weights exist as one of:
  - `model.safetensors`
  - `model-*.safetensors`, normally with `model.safetensors.index.json`
  - `pytorch_model.bin`
  - `pytorch_model-*.bin`, normally with `pytorch_model.bin.index.json`
- Every filename referenced by a sharded index exists beneath the same checkpoint directory.
- Adapter files such as `adapter_model.safetensors` do not count as full weights.

## Tokenizer files

- `tokenizer_config.json` exists.
- At least one self-contained tokenizer asset exists, such as `tokenizer.json`, `tokenizer.model`, `spiece.model`, `vocab.json`, or `vocab.txt`.
- The tokenizer directory may differ from the model directory; the AIR configuration must then set `tokenizer_path` explicitly.

## Unsupported artifacts

Stop and require an explicit conversion or template extension for:

- MLflow serving packages without portable Hugging Face weights
- PEFT adapters without a resolved base model and explicit merge/adapter-loading plan
- Optimizer-only or FSDP-only shards that cannot be consolidated as a Hugging Face checkpoint
- Multiple plausible checkpoints or tokenizers whose provenance cannot be distinguished
- Architectures incompatible with the selected AIR template

The AIR materialization step validates structure and provenance without loading tensor contents. Record its successful AIR run ID and persisted inventory. Perform a clean `AutoConfig`, tokenizer, and model smoke load later during migration validation on compute sized for the checkpoint.

Use the persisted inventory to estimate downstream node-local staging capacity. Every AIR node needs room for each distinct Volume-backed model/tokenizer directory plus at least 1 GiB or 10 percent reserve. This runtime cache is disposable; the portable checkpoint and inventory remain on the UC Volume.

Prefer the registered `models:/catalog.schema.model/version` artifact. Use a `runs:/run_id/path` fallback only when the inspection manifest records that exact portable artifact as lineage of the selected UC model version.
