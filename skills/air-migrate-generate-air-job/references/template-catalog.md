# Approved AIR template catalog

The repository-level `air_templates/` directory is authoritative. Each generated workload starts as an exact copy of one recipe.

| Recipe | Training behavior | Generated files |
|---|---|---|
| `trl_lora` | TRL bf16 LoRA with DDP and separate merge | Six core files plus `merge.py` |
| `trl_lora_fsdp` | TRL bf16 LoRA with PEFT-aware FSDP and separate merge | Six core files plus `merge.py` |
| `trl_full_fsdp` | Full-weight TRL with FSDP and portable full checkpoint | Six core files |

The six core files are `train.yaml`, `train.py`, `helper_utils.py`, `training_utils.py`, `01_runner.py`, and `02_register_uc.py`.

## PEFT-only invariant

`source.peft_only: true` restricts generation to `trl_lora` or `trl_lora_fsdp`. Planning chooses `trl_lora` when the inspected unquantized bf16 base replica and runtime overhead fit on every requested accelerator; otherwise it chooses `trl_lora_fsdp` when supported. The materializer rejects `trl_full_fsdp` for this configuration and never guesses or changes the planned recipe. With `peft_only: false`, recipe selection follows the recovered full-weight or PEFT training semantics.

## Model-source invariant

Generation selects exactly one source in order:

1. Populated `source.weights_volume_path` -> `model_source: volume`; model and tokenizer use that `/Volumes/...` path.
2. Otherwise populated `source.system_ai_model_uri` -> `model_source: system_ai`; both references use the exact versioned `models:/system.ai.<model>/<version>` URI.
3. Otherwise `source.huggingface_model_id` -> `model_source: hugging_face`; both references use the repository ID.

`source_model_uri` always records the versioned legacy UC model as lineage. It is not necessarily the training input. Only a gated Hugging Face selection retains `secrets.HF_TOKEN` and sets `requires_hf_token: true`.

Volume inputs are copied once per AIR node into an ephemeral cache. System.ai artifacts are downloaded through MLflow into that cache at operator runtime. Hugging Face references are downloaded by Transformers. These templates do no agent-side materialization.

## Other invariants

- Top-level `experiment_name` equals `training_config.experiment_path`.
- Generated compute matches `migrate/config.yaml`, including torchrun and notebook GPU settings.
- Generated recipe satisfies `source.peft_only` and traces to the planner's model-size evidence.
- Durable checkpoints, adapters, and merged outputs remain on UC Volumes.
- Registration reads its three-level target and final checkpoint from `train.yaml`.
- LoRA produces adapters during training and a portable full checkpoint after merge.

Changing framework, PEFT/FSDP strategy, launcher, or artifact type requires a new template. Migration performs local validation only and never runs AIR, training, merge, registration, or evaluation.
