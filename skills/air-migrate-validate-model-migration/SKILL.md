---
name: air-migrate-validate-model-migration
description: "Retired compatibility skill for the former live migration-validation workflow. Current migrations perform local generated-file checks inside air-migrate-generate-air-job and stop before AIR execution."
---

# Retired: Live Migration Validation

Local validation of the AIR config, Python modules, notebooks, compute parity, Volume source, experiment path, and registration target now belongs to `air-migrate-generate-air-job`.

Do not submit AIR dry-runs, smoke training, merge, registration, materialization, evaluation, or promotion from this skill. Its reference now describes only the local handoff checks recorded as `generation.handoff_ready`; the orchestrator never reports model migration completion or a post-training validation verdict.
