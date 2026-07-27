---
name: air-migrate-compare-token-accuracy
description: "Retired compatibility skill for the former post-training token-accuracy workflow. Do not use in the current generation-only FMT-to-AIR migration."
---

# Retired: Token-Accuracy Comparison

The current migration ends after creation and local static validation of `migrate/output/air_workload`. It does not train, register, materialize evaluation checkpoints, or compare model quality.

The evaluator and reference contract in this directory are retained only for historical compatibility. Do not invoke them from the migration orchestrator or treat their output as part of the generated handoff. Any evaluation after training is owned by the user or downstream operator.
