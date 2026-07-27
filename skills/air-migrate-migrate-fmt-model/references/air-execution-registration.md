# Operator handoff: execution and registration

This runbook is for the human or system that receives `migrate/output/air_workload`. The migration skills must not execute any command in this file.

Before execution, review `train.yaml`, ensure the MLflow experiment and UC destinations exist, and confirm the selected model source, data, and output paths are accessible from AIR. For gated Hugging Face input, verify the configured `HF_TOKEN` secret reference.

Typical operator flow:

```bash
cd migrate/output/air_workload
COPYFILE_DISABLE=1 air run --dry-run --file train.yaml -p DEFAULT
COPYFILE_DISABLE=1 air run --file train.yaml -p DEFAULT --watch
```

LoRA AIR runs produce an adapter. The operator must run the generated merge entry point before registration:

```bash
COPYFILE_DISABLE=1 air run --file train.yaml -p DEFAULT --watch \
  --override 'command=python $CODE_SOURCE_PATH/merge.py'
```

The alternative notebook flow is `01_runner.py`, which trains and, for LoRA, merges. After either flow, open `02_register_uc.py`, provide the training MLflow run ID, and run it on compute with enough memory and temporary storage to package the final checkpoint.

Training, merge, registration, model evaluation, verification of UC readiness, and promotion are all operator-owned post-handoff responsibilities. They are not evidence produced by the migration skills.
