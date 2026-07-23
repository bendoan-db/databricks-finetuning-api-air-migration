# AIR recipe selection

| Objective and constraint | Recipe | Output | Key tradeoff |
|---|---|---|---|
| Reproduce full-weight training with the closest practical parity | `axolotl_full_fsdp` | Consolidated full model and tokenizer | Highest memory and compute cost |
| Cost-efficient PEFT where the 4-bit base fits per GPU | `axolotl_qlora` | PEFT adapter and tokenizer metadata | Not full-weight parity |
| PEFT for a very large model whose 4-bit base still needs sharding | `axolotl_qlora_fsdp` | PEFT adapter with a reproducible base dependency | More distributed complexity; checkpoint consolidation pressure |
| Repackage existing portable weights without training | `none` | Registered copy of the existing artifact | Not an AI Runtime training migration |
| Unsupported architecture, objective, data modality, or framework requirement | `custom` | Explicitly designed workload | No approved template; requires extension before generation |

## Decision order

1. For `mode: repackage`, choose `none` and skip AIR code generation.
2. Determine whether the objective requires updating every model weight. If yes, choose full fine-tuning, normally `axolotl_full_fsdp`.
3. If PEFT is acceptable, estimate quantized base weights, activations at the chosen context length, adapter/optimizer state, temporary buffers, and safety margin.
4. Choose plain `axolotl_qlora` when each worker can hold the needed base-model replica and activation budget.
5. Choose `axolotl_qlora_fsdp` when sharding the frozen quantized base is necessary or explicitly requested and the architecture is supported.
6. Choose `custom` when none of the templates preserve the required semantics.

The current templates provide validated starting points, not universal capacity guarantees:

- `axolotl_qlora`: Llama 3.1 8B Instruct, 4-bit QLoRA, 8xH100.
- `axolotl_full_fsdp`: Llama 3.1 8B Instruct, full fine-tuning, 8xH100.
- `axolotl_qlora_fsdp`: Llama 3.1 70B Instruct, 4-bit QLoRA with FSDP and CPU-efficient loading, 8xH100.
