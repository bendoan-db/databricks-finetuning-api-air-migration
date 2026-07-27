# Repository Guidelines

## Project Structure & Module Organization

`air_templates/` contains the canonical Databricks AI Runtime workloads for LoRA, LoRA with FSDP, and full-weight FSDP. Keep each recipe's AIR config, Python helpers, training notebook, and registration notebook synchronized. `skills/` contains the staged `air-migrate-*` workflows; each skill keeps instructions in `SKILL.md`, supporting contracts in `references/`, and utilities in `scripts/`. Migration inputs live in `migrate/config.yaml`; generated artifacts under `migrate/output/` are ignored by Git.

## Build, Test, and Development Commands

This repository has no package build step. Use these checks from the repository root:

```bash
python3 -m compileall -q air_templates skills
ruff check air_templates skills
python3 skills/air-migrate-generate-air-job/scripts/materialize_air_template.py \
  --recipe trl_lora_fsdp
```

The generator validates and copies a template into `migrate/output/air_workload`; its destination must be empty. It pins the selected Volume, `system.ai`, or Hugging Face input plus the legacy source URI, compute, experiment, and registration target. Do not run AIR, training, registration, or evaluation as part of repository validation.

## Coding Style & Naming Conventions

Use four spaces in Python and two spaces in YAML. Follow existing Python patterns: type annotations, `pathlib.Path`, concise docstrings, `snake_case` functions and variables, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Format skill directories as `air-migrate-<action>` and keep their entry file named `SKILL.md`. In Databricks source notebooks, preserve `# COMMAND ----------` boundaries and place `# MAGIC` directives first in their cells.

## Testing Guidelines

No automated test framework or coverage threshold is currently configured. Run compile and Ruff checks for every Python change, then exercise the generator with Volume, `system.ai`, gated Hugging Face, and public Hugging Face configurations. Test both LoRA recipes with `peft_only: true`, confirm full FSDP is rejected, and test full FSDP with `peft_only: false`. Template changes require local YAML/configuration loading checks; AIR dry-runs are outside this generation-only workflow. If adding unit tests, place pytest-style files under `tests/` and name them `test_<module>.py`; keep external Databricks calls mocked.

## Commit & Pull Request Guidelines

Recent commits use short, lowercase imperative subjects, such as `add support for post migration eval comparison`. Keep commits focused and avoid committing generated migration outputs, credentials, checkpoints, or local Databricks state. Pull requests should explain the migration behavior changed, identify affected templates or skills, list validation commands and results, link the relevant issue, and include logs or screenshots when notebook or AIR behavior changes.

## Security & Configuration

Use `<scope>/<key>` Databricks secret references for gated Hugging Face tokens; never commit token values. Do not commit `.env`, `.databrickscfg`, `.databricks-resources.json`, private keys, checkpoints, or populated customer-specific configuration.
