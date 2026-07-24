# AIR execution and registration runbook

Use this runbook only after the generated workload passes static validation and `migrate/config.yaml` has `source.run_full_migration: true`. That value is the pipeline's explicit authorization for training, merge, registration, evaluation, and final validation. If it is false, do not run any live command in this runbook; AIR dry-run commands may be used during preflight but still stage workspace files. Run the experiment helper from the repository root, then run AIR commands from `migrate/output/air_workload`, never from `air_templates/`.

Before the first dry-run or live AIR command, ensure the configured experiment exists:

```bash
python3 skills/air-migrate-migrate-fmt-model/scripts/ensure_migration_experiment.py \
  --config migrate/config.yaml --profile DEFAULT
```

Require the returned path to equal `migrate/config.yaml.source.migration_experiment_path`. Record its experiment ID and whether it was created. Generated `train.yaml.experiment_name` and `training_config.experiment_path` must both equal that path. Repeat this check after a config/profile change; do not submit into a fallback experiment.

```bash
cd migrate/output/air_workload
```

## 1. Train

```bash
COPYFILE_DISABLE=1 air run --dry-run --file train.yaml -p DEFAULT
COPYFILE_DISABLE=1 air run --file train.yaml -p DEFAULT --watch
```

Record separately:

- The numeric AIR/Databricks Jobs run ID used to monitor infrastructure execution.
- The rank-zero MLflow run ID printed by `train.py`; pass this ID to registration.
- The configured MLflow experiment ID; verify both the AIR-managed run and rank-zero training run belong to it.

AIR dry-run skips Jobs submission but still stages configuration and snapshot files in the user's workspace. `COPYFILE_DISABLE=1` prevents macOS AppleDouble files from entering the snapshot; extended-attribute warnings may still be harmless.

## 2. Materialize a portable full checkpoint

Full-FSDP training already writes portable full weights. AIR LoRA training runs `train.py` only and does not execute the merge cell in `01_runner.py`, so require a separate merge:

```bash
COPYFILE_DISABLE=1 air run --file train.yaml -p DEFAULT --watch \
  --override 'command=python $CODE_SOURCE_PATH/merge.py'
```

Require `training_config.output_dir` to contain the adapter and `training_config.merged_output_dir` to be distinct and empty before the first merge. A retry may reuse an already valid merged checkpoint; never overwrite or delete a nonempty destination merely to make a retry pass.

Verify the merged directory contains `config.json`, tokenizer files, a safetensor index when sharded, and full `model*.safetensors` or `pytorch_model*.bin` weights. Reject directories containing `adapter_config.json` or `adapter_model.*` as registration inputs.

## 3. Register through AIR

Large checkpoints can exhaust standard serverless notebook memory because MLflow may materialize temporary download, packaging, and upload copies. Submit the dual-mode registration notebook as the AIR Python command and keep `train.yaml.compute` unchanged unless the user explicitly changes the migration config:

```bash
COPYFILE_DISABLE=1 air run --dry-run --file train.yaml -p DEFAULT \
  --override 'command=python $CODE_SOURCE_PATH/02_register_uc.py --mlflow-run-id DRY_RUN_ONLY'

COPYFILE_DISABLE=1 air run --file train.yaml -p DEFAULT --watch \
  --override 'command=python $CODE_SOURCE_PATH/02_register_uc.py --mlflow-run-id <training-mlflow-run-id>'
```

Run one Python process; do not use `torchrun` for registration. In AIR script mode, notebook `%pip` lines are comments, so every import needed by `02_register_uc.py` must be present in `train.yaml.environment.dependencies`. The source file must guard `dbutils` calls and accept `--mlflow-run-id`; its interactive path must retain the `mlflow_run_id` widget.

For source notebooks, place `# MAGIC %md` or `# MAGIC %pip` first in its command cell. A Python directive such as `# ruff: noqa` before the opening magic can make Databricks parse Markdown as Python.

## 4. Verify completion

Do not infer success from artifact progress, creation of the registered-model container, or the message that a new version is being created. Require both terminal evidence:

```bash
air get run <registration-air-run-id> -p DEFAULT --json
databricks model-versions get <catalog.schema.model> <version> -p DEFAULT -o json
```

The AIR run must report `SUCCESS`; the exact UC model version must report `READY`, point to the intended training MLflow run ID, and use the expected logged-model source URI. Record the registered model version, its source URI, the training MLflow run ID, and the registration AIR run ID.

Resolve every training, merge, and registration run's experiment and require it to match the configured migration experiment ID. A successful run in another experiment is invalid migration evidence.

## Retry rules

- Do not retrain after a registration-only OOM or notebook parsing failure. Reuse the successful training MLflow run and validated full checkpoint.
- Treat standard serverless `ModuleNotFoundError: serverless_gpu` as evidence that a notebook-only launcher was used in the wrong execution context. Use the AIR script path or require a pre-merged checkpoint.
- Preserve failed run IDs and error summaries. Registration retries can leave failed logged-model records or create the UC model container without a ready version.
- Do not remove partial artifacts, model versions, or registered-model containers without explicit authorization.
