# Portable checkpoint contract

Apply this structural contract to a configured `source.weights_volume_path` during read-only inspection and to a downloaded system.ai artifact at operator runtime. Do not load tensors merely to validate structure.

## Required model files

- A valid `config.json` identifying a model type or architecture.
- Full weights in `model*.safetensors` or `pytorch_model*.bin` files.
- Every shard named by a safetensors or PyTorch index file.
- No reliance on adapter-only files such as `adapter_model.safetensors`.

## Required tokenizer files

- `tokenizer_config.json`.
- At least one tokenizer asset: `tokenizer.json`, `tokenizer.model`, `spiece.model`, `sentencepiece.bpe.model`, `vocab.json`, or `vocab.txt`.

The current generated configurations use the same reference for model and tokenizer. Add explicit template support before accepting split directories or distinct repositories.

## Runtime staging

For Volume input, inventory files and sizes before copying. Require free node-local capacity for the complete checkpoint plus the greater of 1 GiB or 10 percent reserve. Copy under a per-source lock, verify copied file sizes, and publish only an atomically completed cache directory.

For system.ai input, download through MLflow into a partial node-local directory, require exactly one checkpoint satisfying this contract, write a completion marker, and atomically publish it to the cache. Preserve the exact versioned system.ai URI in the marker.

The cache is disposable. Durable adapters, checkpoints, merged weights, and registration inputs must remain on Unity Catalog Volumes.

## Unsupported artifacts

Stop or require a new template for serving-only MLflow packages, adapter-only sources without an explicit base model, optimizer/FSDP shards that cannot form a portable Transformers checkpoint, multiple plausible checkpoints, missing tokenizers, or incompatible architectures.
