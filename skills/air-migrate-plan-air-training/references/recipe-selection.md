# AIR recipe selection

| Objective and constraint | Recipe | Output | Key tradeoff |
|---|---|---|---|
| Reproduce full-weight training with the closest supported practical parity | `trl_full_fsdp` | Gathered full model and tokenizer | Requires a validated FSDP transformer-layer wrap class; collective full-state save and MLflow registration can require substantial host memory and temporary storage |
| Unquantized PEFT where a bf16 base replica fits each worker | `trl_lora` | PEFT adapter plus merged full registration checkpoint | Straightforward `SFTTrainer`/DDP workflow, but every worker holds the complete base model and AIR CLI training requires the separate `merge.py` action |
| Unquantized PEFT where the bf16 base must be sharded | `trl_lora_fsdp` | PEFT adapter plus merged full registration checkpoint | PEFT-aware FSDP wrapping and collective adapter saves add complexity; AIR CLI training still requires the separate unquantized `merge.py` action |
| Repackage existing portable weights without training | `none` | Registered copy of the existing artifact | Not an AI Runtime training migration |
| Unsupported architecture, objective, data modality, quantized training need, or framework requirement | `custom` | Explicitly designed workload | No approved TRL template; requires extension before generation |

## Decision order

1. For `mode: repackage`, choose `none` and skip AIR code generation.
2. Inventory every directory in `air_templates/` and record all candidates considered.
3. Reject candidates whose architecture, tokenizer/chat format, training method, precision, distributed strategy, artifact contract, or requested compute cannot support the inspected model and migration objective.
4. Determine whether the objective requires updating every model weight. If yes, choose `trl_full_fsdp` when its FSDP wrap class is validated for the architecture and the requested compute can support full training and checkpoint consolidation.
5. If PEFT is acceptable, estimate bf16 base weights, activations at the chosen context length, adapter/optimizer state, temporary buffers, and safety margin.
6. Choose `trl_lora` when unquantized LoRA is desired and every worker can hold one complete bf16 base-model replica plus its activation budget.
7. Choose `trl_lora_fsdp` when unquantized LoRA is desired but the bf16 base must be sharded, or when FSDP is an explicit requirement and the architecture's transformer-layer wrap class is validated.
8. Choose `custom` when none of the TRL templates preserve the required semantics, including any quantized adapter-training requirement.

For all three training recipes, treat `/Volumes` model and tokenizer directories as durable sources rather than efficient random-read runtime filesystems. Plan one pre-training copy to ephemeral node-local storage per source directory per AIR node. Include the checkpoint inventory size, a reserve of at least 1 GiB or 10 percent, a bounded copy-worker count, and evidence that each node has sufficient local disk. This prefetch changes I/O placement only; it does not change the selected training recipe, source lineage, or durable output contract. Remote Hugging Face sources bypass this template-specific Volume prefetch.

The current templates provide validated starting points, not universal capacity guarantees:

- `trl_full_fsdp`: Llama 3.1 8B Instruct, full-weight TRL SFT with FSDP full sharding, 8xH100.
- `trl_lora_fsdp`: Llama 3.1 8B Instruct, unquantized bf16 LoRA with TRL and PEFT-aware FSDP full sharding, 8xH100.
- `trl_lora`: Llama 3.1 8B Instruct, unquantized bf16 LoRA with TRL `SFTTrainer` and DDP, 8xH100.

The template's reference GPU is not a default the planner may choose. Copy the exact request from `migrate/config.yaml` into `plan.compute.requested` and `plan.compute.resolved`, then record whether node-local source staging, the model, context/batch, checkpoint consolidation, any PEFT merge, and registration packaging fit that resource. Registration may temporarily materialize download and upload copies of the full checkpoint, so include host memory and local disk rather than GPU memory alone. Current AIR resource names are `GPU_1xA10`, `GPU_1xH100`, and `GPU_8xH100`, with counts 1, 1, and 8 respectively.

The selected template is a source artifact. Copy it to `migrate/output/air_workload`, customize that copy for the inspected model, and run only the generated copy.
