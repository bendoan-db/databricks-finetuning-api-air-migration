---
name: air-migrate-materialize-uc-model
description: "Retired compatibility skill for the former agent-side UC model-materialization workflow. Do not use for the current generation-only FMT-to-AIR migration; generated templates load a configured Volume or acquire system.ai or Hugging Face weights only when an operator runs them."
---

# Retired: UC Model Materialization

The current migration selects `source.weights_volume_path`, otherwise `source.system_ai_model_uri`, otherwise `source.huggingface_model_id`. The migration agent never downloads or materializes weights. Generated runtime code later copies Volume inputs, downloads an exact system.ai artifact, or lets Transformers fetch Hugging Face weights when an operator starts training.

Scripts in this directory are retained only for historical compatibility and maintenance of older branches. Do not invoke them from `air-migrate-migrate-fmt-model`, planning, generation, validation, or operator handoff. The references document the current source-selection and portable-checkpoint contracts without reactivating materialization.

Use `air-migrate-inspect-fmt-model` to record source compatibility and `air-migrate-generate-air-job` to pin the selected source into the generated workload.
