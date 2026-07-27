# AIR recipe selection

| Objective and constraint | Recipe | Output | Key tradeoff |
|---|---|---|---|
| Full-weight training with a supported transformer wrap class | `trl_full_fsdp` | Gathered full model and tokenizer | Highest memory and checkpoint-consolidation cost |
| LoRA where one unquantized bf16 base replica fits each worker | `trl_lora` | Adapter plus merged full checkpoint | Simple DDP; every worker holds the base |
| LoRA where the unquantized base must be sharded | `trl_lora_fsdp` | Adapter plus merged full checkpoint | More complex PEFT-aware FSDP and collective saves |
| Unsupported architecture, modality, quantization, or framework | `custom` | New explicitly designed template | Generation pauses until the template exists |

## Decision order

1. Inventory every directory under `air_templates/`.
2. Read and preserve `source.peft_only` from configuration and inspection.
3. Reject recipes that cannot preserve architecture, tokenizer/chat format, objective, precision, distributed strategy, or output contract.
4. When `peft_only` is true, remove `trl_full_fsdp` from consideration. Estimate the unquantized bf16 base weights from the inspected parameter count, then add activations at the planned context/batch, adapter and optimizer state, temporary buffers, and safety margin. Choose `trl_lora` only when that replicated footprint fits on every requested accelerator; otherwise choose `trl_lora_fsdp` when its wrap strategy is supported.
5. When `peft_only` is false, choose `trl_full_fsdp` only when all model weights must be updated and the transformer layer is validated for FSDP wrapping; PEFT recipes remain eligible only when they preserve the requested semantics.
6. Validate the exact requested compute without substituting the template default.
7. Choose `custom` or return a blocker when no eligible approved recipe fits. A PEFT-only migration must never fall back to `trl_full_fsdp`.

Record the parameter count, estimated bf16 base bytes, per-accelerator memory budget, non-model overhead estimate, safety margin, and `full_base_replica_fits_per_worker` verdict. Model size selects between replicated and sharded LoRA; the model name alone is not evidence.

All recipes support the same source precedence. For a Volume, plan enough ephemeral disk for one copy plus at least 1 GiB or 10 percent reserve. For `system.ai`, plan the same capacity for the runtime MLflow download. Hugging Face is downloaded by Transformers and may require the configured token secret. Runtime acquisition changes I/O placement, not legacy UC lineage.

The selected template is copied to `migrate/output/air_workload` and customized there. Selection and generation do not run the workload.
