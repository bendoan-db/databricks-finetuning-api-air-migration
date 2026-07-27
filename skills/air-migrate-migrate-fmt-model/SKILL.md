---
name: air-migrate-migrate-fmt-model
description: "Orchestrate a generation-only FMT-to-AI Runtime migration. Apply peft_only recipe selection, resolve configured Volume, system.ai, or Hugging Face weights, generate the AIR handoff, validate it locally, and stop before execution."
---

# Migrate FMT Model

Coordinate a generation-only workflow with `migrate/output/migration-manifest.yaml` as the shared contract.

## Required stage skills

Read and follow these sibling skills in order:

1. [`air-migrate-inspect-fmt-model`](../air-migrate-inspect-fmt-model/SKILL.md)
2. [`air-migrate-plan-air-training`](../air-migrate-plan-air-training/SKILL.md)
3. [`air-migrate-generate-air-job`](../air-migrate-generate-air-job/SKILL.md)

Materialization, live validation, and token-accuracy skills are not part of this workflow.

## Workflow

1. **Resolve configuration.** Require a versioned source UC model, an absolute migration experiment path, boolean `source.peft_only`, a complete or all-blank target identity, and supported compute. Select the model input in this order:
   1. Nonblank `source.weights_volume_path` -> `volume`.
   2. Otherwise nonblank `source.system_ai_model_uri` -> `system_ai`.
   3. Otherwise `source.huggingface_model_id` -> `hugging_face`.
2. **Inspect.** Recover read-only UC/MLflow lineage, data, training settings, architecture, parameter count, estimated bf16 base size, permissions, and unknowns. Structurally inspect a selected Volume checkpoint. For `system.ai` or Hugging Face, validate identity and compatibility without downloading weights.
3. **Plan.** Preserve `peft_only`. When true, select `trl_lora` if a replicated bf16 base plus overhead fits per requested accelerator, otherwise select `trl_lora_fsdp` when supported; never select full FSDP. When false, preserve recovered full-weight or PEFT semantics. Record model-size evidence, requested compute, selected source, data, outputs, and runtime staging behavior.
4. **Generate.** Copy exactly one approved template into `migrate/output/air_workload`, customize the copy, and run local static checks.
5. **Stop and report.** Return the generated files, source mode, recipe, validations, assumptions, and operator handoff. State explicitly that training, merge, registration, and evaluation were not run.

## Terminal boundary

Never invoke Databricks or AIR CLI commands, MLflow mutations, training code, `merge.py`, `02_register_uc.py`, evaluation code, or promotion actions. Do not perform AIR dry-runs. A user or operator owns all execution after handoff.

The operator-time templates copy Volume inputs to node-local storage, download `system.ai` artifacts to that cache, or let Transformers download from Hugging Face. These runtime paths are not agent-side materialization.

## Canonical handoff

```text
migrate/output/air_workload/
├── train.yaml
├── train.py
├── helper_utils.py
├── training_utils.py
├── merge.py              # LoRA recipes only
├── 01_runner.py
└── 02_register_uc.py
```

`train.yaml` is the AIR config; `01_runner.py` is the training notebook; the Python modules implement training and optional merge support; and `02_register_uc.py` is the registration notebook. Repository templates remain immutable.

Regenerate whenever `peft_only`, source selection, recovered training behavior, recipe, compute, experiment, target, data, outputs, or template content changes.
