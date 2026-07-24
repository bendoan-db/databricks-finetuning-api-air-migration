# Foundation Model Fine-tuning to AI Runtime migration

This project demonstrates how to migrate a model trained with the legacy Databricks Foundation Model Fine-tuning API to a reproducible Databricks AI Runtime workload. It contains:

- An optional legacy fine-tuning demo using the IBM TabFormer dataset
- AI Runtime templates for TRL `SFTTrainer` LoRA with DDP or FSDP and full-weight FSDP
- Coding-assistant skills that inspect, plan, materialize, generate, and validate a migration

## Repository layout

| Directory | Purpose |
|---|---|
| `example_setup/` | Optional demo that creates a TabFormer source model with the legacy fine-tuning API |
| `air_templates/` | Approved TRL workload templates for Databricks AI Runtime |
| `migrate/` | Identifies the source UC model version, target UC model, and generated migration outputs |
| `skills/` | Portable `air-migrate-*` skills for coding assistants |

## How To Run

If you already have a Unity Catalog model produced by the legacy Foundation Model Fine-tuning API:

1. Update [`migrate/config.yaml`](migrate/config.yaml) with `source.migration_experiment_path`, the source model, `use_existing_weights`, optional `existing_weights_volume_location`, required `run_full_migration` execution gate, target location, and AIR `compute`. Set `use_existing_weights` to `true` to continue from existing weights, or `false` to retrain from the original base model (`system.ai` when available, otherwise Hugging Face). With `true`, populate the Volume path to load an already staged checkpoint directly; leave it blank to materialize the configured UC version through AIR.
2. Install or expose the folders under `skills/` to your coding assistant.
3. Invoke the migration orchestrator:

   ```text
   Use $air-migrate-migrate-fmt-model to migrate the model configured in migrate/config.yaml.
   ```

4. Review the migration manifest's model/template compatibility evidence, selected AIR template, and validation criteria. The migration flow copies and customizes that template into `migrate/output/air_workload` using `train.yaml`, a thin `train.py`, configuration helpers in `helper_utils.py`, training/runtime helpers in `training_utils.py`, `01_runner.py`, and `02_register_uc.py`; LoRA recipes also include `merge.py`.
5. Let `source.run_full_migration` control the post-preflight boundary. With `true`, the orchestrator runs training, any required LoRA merge, registration, model evaluation, and final validation. With `false`, it resolves or materializes the input weights, generates the workload, records preflight validation, and stops without submitting training, merge, registration, or model-evaluation work. Required source materialization still runs through AIR in either mode.

   To run or resume a generated workload manually, use only `migrate/output/air_workload` after its static checks pass. Do not run a source directory under `air_templates/`. Use either launch path:

   - **AIR CLI:** Authenticate the Databricks CLI, confirm the `HF_TOKEN` secret reference in `train.yaml`, and run from the generated workload directory:

     ```bash
     python3 skills/air-migrate-migrate-fmt-model/scripts/ensure_migration_experiment.py \
       --config migrate/config.yaml --profile DEFAULT
     cd migrate/output/air_workload
     COPYFILE_DISABLE=1 air run --dry-run --file train.yaml -p DEFAULT
     COPYFILE_DISABLE=1 air run --file train.yaml -p DEFAULT --watch
     ```

     AIR dry-run validates the payload and skips Jobs submission, but it still stages launch files and the code snapshot in the user's workspace. The live AIR command runs `train.py` and does not register a model. Record both the numeric AIR run ID and the rank-zero training MLflow run ID from the output.

     If the configured model or tokenizer is under `/Volumes`, the training process automatically stages it once per AIR node into the ephemeral cache configured by `local_model_cache_dir`. Other ranks on that node wait for and reuse the completed local copy. Ensure every node has local capacity for the full source checkpoint, any separate tokenizer directory, and the safety reserve described below.

     An AIR LoRA training run writes an adapter but does not execute the merge cell in `01_runner.py`. Materialize the portable full checkpoint before registration:

     ```bash
     COPYFILE_DISABLE=1 air run --file train.yaml -p DEFAULT --watch \
       --override 'command=python $CODE_SOURCE_PATH/merge.py'
     ```

     The merge reuses the trained adapter and does not start another training run. Its destination, `training_config.merged_output_dir`, must be distinct from and empty before the first merge.

   - **Runner notebook:** Set `training_config.registered_model_name` in `train.yaml` to the target `<catalog>.<schema>.<model>`. For LoRA recipes, also set `training_config.merged_output_dir` to an empty UC Volume directory. Import every generated file, including `helper_utils.py`, `training_utils.py`, and LoRA `merge.py`, into the same Databricks workspace folder. Open `01_runner.py` as a Databricks source notebook, attach it to Serverless GPU using the configured compute and AI Runtime version (the templates default to 8xH100 and AI v5), configure the Hugging Face secret widgets, and select **Run all**. The notebook launches training through `@distributed`; LoRA variants also merge the adapter into the unquantized base model. Record the printed training MLflow run ID.

   Registration is intentionally separate from training. For small checkpoints, open `02_register_uc.py`, enter the training MLflow run ID, and run the notebook. For large checkpoints, submit the same notebook source through AIR so MLflow packaging has the training workload's host memory and temporary storage:

   ```bash
   COPYFILE_DISABLE=1 air run --file train.yaml -p DEFAULT --watch \
     --override 'command=python $CODE_SOURCE_PATH/02_register_uc.py --mlflow-run-id <training-mlflow-run-id>'
   ```

   Do not prefix this command with `torchrun`: registration must execute once. Leaving the compute block unchanged submits registration with the same AIR compute as training. The notebook supports widgets in Databricks and `--mlflow-run-id` in AIR script mode; its Python dependencies must therefore also be listed in `train.yaml`, because notebook `%pip` cells are comments when AIR executes the source file.

   A 16 GB checkpoint can require materially more than 16 GB of host memory and temporary disk because MLflow may download, package, and upload full model copies. Standard serverless notebook compute can run out of memory even when training succeeded. A failed registration attempt does not require retraining: retain the training MLflow run and merged checkpoint, then retry registration on suitable AIR compute. Do not delete partially created logged-model artifacts or UC objects without explicit authorization.

   Do not treat creation of the UC model container as completion. Require the registration AIR run to finish with `SUCCESS` and the exact new UC model version to report `READY`:

   ```bash
   air get run <registration-air-run-id> -p DEFAULT --json
   databricks model-versions get <catalog.schema.model> <version> -p DEFAULT -o json
   ```

6. In full-migration mode, validate the registered model. The validation workflow reuses the configured existing-weights Volume for the legacy model when populated; otherwise it materializes the exact legacy UC version. It materializes the newly registered version, then compares assistant response-token accuracy on the same hashed evaluation JSONL. The evaluator creates or reuses an MLflow run only in `source.migration_experiment_path` and logs its inputs, aggregate metrics, verdict, and JSON evidence there. A pass/fail result requires a predeclared maximum regression and equivalent tokenizer/chat serialization. Preparation-only mode records a scoped preflight verdict and does not claim that the migration is complete.

If you do not already have a legacy fine-tuned model, first complete the optional demo below to create one, then follow the same migration steps.

## Optional demo: create a legacy source model

Skip this section if you already have a Unity Catalog model produced by the legacy Foundation Model Fine-tuning API. Continue directly to [Configure a migration](#configure-a-migration).

All setup notebooks read [`example_setup/config.yaml`](example_setup/config.yaml). Update its Unity Catalog destinations and model registration target, then run these notebooks in order on Databricks:

1. [`01_load_dataset.ipynb`](example_setup/01_load_dataset.ipynb) downloads TabFormer and creates the Delta table.
2. [`02_stage_data.ipynb`](example_setup/02_stage_data.ipynb) creates deterministic train and evaluation JSONL files in a UC Volume.
3. [`03_finetune_model.ipynb`](example_setup/03_finetune_model.ipynb) submits the legacy `CHAT_COMPLETION` fine-tuning run.

The staged JSONL records use OpenAI-style `messages` arrays and can be consumed directly by the AIR templates.

## Configure a migration

Edit [`migrate/config.yaml`](migrate/config.yaml) with the source Unity Catalog model version:

```yaml
source:
  catalog: my_catalog
  schema: my_schema
  model: my_model
  version: 1
  migration_experiment_path: /Shared/ft-api-migration
  use_existing_weights: true
  existing_weights_volume_location:
  run_full_migration: false

target:
  catalog:
  schema:
  model:

compute:
  num_accelerators: 8
  accelerator_type: GPU_8xH100
```

Leave all target fields blank to register the migrated model as a new version of the source model. Otherwise, populate all three target fields. Partially populated targets are invalid.

`source.migration_experiment_path` is required and must be an absolute Databricks workspace experiment path such as `/Shared/ft-api-migration` or `/Users/name@example.com/ft-api-migration`. Before any AIR dry-run or live submission, the migration flow checks for that MLflow experiment and creates it when absent. The generator copies the exact path into both the AIR workload's top-level `experiment_name` and `training_config.experiment_path`. Materialization, training, merge, registration, smoke, and evaluation AIR runs must all resolve to the same experiment ID; a run in another experiment is rejected rather than used as migration evidence.

`existing_weights_volume_location` must be blank or an absolute `/Volumes/<catalog>/<schema>/<volume>[/<checkpoint-path>]` directory containing a complete portable Hugging Face model and tokenizer. It is valid only with `use_existing_weights: true`. When populated, the migration validates and loads that directory directly, preserves the configured UC version as lineage, and does not materialize or copy the UC model. When blank, `use_existing_weights: true` materializes the exact source version through AIR. `use_existing_weights: false` requires the field to be blank, recovers the original base-model ID, checks for an exact portable model under `system.ai`, and falls back to Hugging Face only when no match exists. Every generated `train.yaml` records the selected source and rejects inconsistent URI/path combinations.

`run_full_migration` is a required boolean pipeline gate. Set it to `false` for preparation only: the orchestrator resolves or AIR-materializes required source weights, generates `migrate/output/air_workload`, performs static and optional AIR dry-run preflight checks, records a preflight-scoped verdict, and stops. AIR dry-run does not submit a Jobs run but does stage files in the workspace. Preparation-only mode never submits training or merge commands, registers a model, materializes a target for evaluation, or runs model-quality evaluation. Set it to `true` to explicitly authorize the orchestrator to continue after preflight through training, merge when required, AIR registration, token-accuracy and other evaluation, and final validation. `migration_complete` becomes true only for a passing final-scope verdict; preflight, failed, and inconclusive results keep it false. Changing this flag invalidates later execution and validation evidence.

`compute` is authoritative for job submission. Supported AIR resource shapes are `GPU_1xA10` with `num_accelerators: 1`, `GPU_1xH100` with `1`, and `GPU_8xH100` with `8`. The planner verifies that the requested resource can run the selected model and recipe. The generator copies it into `migrate/output/air_workload/train.yaml`, synchronizes `torchrun --nproc_per_node`, and the runner notebook derives its `@distributed` GPU count/type from that same YAML. It never silently falls back to a template's reference compute.

### Fast loading for Volume-backed model weights

UC Volumes provide durable, governed model storage, but model initialization performs many large and small file reads that are slower through the Volume filesystem than from the AIR node's local disk. All three templates therefore keep the `/Volumes/...` path as the source of truth and automatically prefetch that directory into ephemeral node-local storage before Transformers loads it.

The behavior is controlled in `training_config`:

```yaml
local_model_cache_dir: /tmp/air-model-cache
local_model_cache_copy_workers: 8
```

`local_model_cache_dir` must be an absolute node-local path, not a UC Volume, DBFS, workspace path, checkpoint directory, or model output directory. `local_model_cache_copy_workers` controls bounded parallel copying and must be between 1 and 32. Ranks on the same node use a file lock so only one process copies each model/tokenizer source. Files are size-checked, the completed directory is published atomically, and Transformers loads the staged path with local-only mode. LoRA merge-time base-model reloads follow the same path. Hugging Face repository IDs bypass this Volume-specific prefetch.

Each AIR node needs free local disk for every distinct staged source directory plus a reserve of at least 1 GiB or 10 percent of the staged bytes, whichever is larger. The cache can disappear between AIR jobs and is never migration evidence. Training checkpoints, adapters, merged/full checkpoints, and registration artifacts continue to use their configured UC Volume paths. MLflow records the durable source path, runtime load path, cache hit, copied bytes/files, lock wait, copy duration, and throughput so slow starts can be diagnosed without changing lineage.

## AI Runtime templates

Each source template contains `train.yaml`, a thin training-only `train.py`, YAML/configuration helpers in `helper_utils.py`, training/runtime/registration helpers in `training_utils.py`, a training notebook named `01_runner.py`, and a post-training registration notebook named `02_register_uc.py`. LoRA templates also contain `merge.py` for the separate AIR merge action. The migration planner selects the best fit for the inspected model, and the generator creates a model-specific copy at `migrate/output/air_workload`. The same TRL training entry point supports AIR CLI submission with `torchrun` and notebook execution with `@distributed`. The registration notebook is dual-mode: widgets when run interactively and CLI arguments when AIR executes it as Python. Unity Catalog registration remains an explicit second step.

| Recipe | Template | Intended use | Default reference workload |
|---|---|---|---|
| `trl_lora` | [`air_templates/trl_lora`](air_templates/trl_lora/train.yaml) | Unquantized bf16 LoRA with TRL `SFTTrainer` and DDP | Llama 3.1 8B Instruct, 8xH100 |
| `trl_lora_fsdp` | [`air_templates/trl_lora_fsdp`](air_templates/trl_lora_fsdp/train.yaml) | Unquantized bf16 LoRA with TRL and PEFT-aware FSDP full sharding | Llama 3.1 8B Instruct, 8xH100 |
| `trl_full_fsdp` | [`air_templates/trl_full_fsdp`](air_templates/trl_full_fsdp/train.yaml) | Full-weight TRL `SFTTrainer` with FSDP full sharding | Llama 3.1 8B Instruct, 8xH100 |

The source templates are not runnable migration outputs. The generator updates the selected copy's source tuple (`use_existing_weights`, `existing_weights_volume_location`, `model_source`, `source_model_uri`, `model_name`, and `tokenizer_path`), configured compute, UC Volume paths, experiment/output locations, and any `HF_TOKEN` Databricks secret reference. Existing UC sources use the validated configured Volume path when supplied or AIR-materialized paths when blank; `system.ai` sources use materialized paths. A Hugging Face token must have access to the selected gated remote model.

Validate and submit the generated workload:

```bash
cd migrate/output/air_workload
COPYFILE_DISABLE=1 air run --dry-run --file train.yaml -p DEFAULT
COPYFILE_DISABLE=1 air run --file train.yaml -p DEFAULT --watch
```

## Migration skills

Install or expose the folders under `skills/` to your coding assistant. Invoke the orchestrator for the complete workflow or use a stage skill directly.

| Skill | Capability | Primary output |
|---|---|---|
| [`air-migrate-inspect-fmt-model`](skills/air-migrate-inspect-fmt-model/SKILL.md) | Resolves the UC model version and recovers MLflow lineage, datasets, base model, task semantics, hyperparameters, checkpoints, artifacts, permissions, and unknowns | `migrate/output/migration-manifest.yaml` |
| [`air-migrate-plan-air-training`](skills/air-migrate-plan-air-training/SKILL.md) | Chooses retrain, continue, or repackage semantics; TRL full fine-tuning or unquantized LoRA; DDP/FSDP requirements; compute; checkpointing; and acceptance criteria | Training plan in the migration manifest |
| [`air-migrate-materialize-uc-model`](skills/air-migrate-materialize-uc-model/SKILL.md) | Reuses validated configured Volume weights or resolves and materializes exact UC versions through AIR when required | Pinned source resolution and worker-readable Volume paths |
| [`air-migrate-generate-air-job`](skills/air-migrate-generate-air-job/SKILL.md) | Materializes the best-fitting approved AIR template, safely customizes it, maintains YAML/notebook parity, and performs static and AIR dry-run checks | `migrate/output/air_workload` |
| [`air-migrate-compare-token-accuracy`](skills/air-migrate-compare-token-accuracy/SKILL.md) | Compares deterministic assistant response-token accuracy for the configured legacy model and newly registered migrated model, including tokenizer comparability and regression gating | `migrate/output/token-accuracy-evaluation.json` |
| [`air-migrate-validate-model-migration`](skills/air-migrate-validate-model-migration/SKILL.md) | Validates workload execution, artifacts, UC registration, shared-dataset metrics, behavioral parity, and promotion criteria | `migrate/output/migration-validation.yaml` |
| [`air-migrate-migrate-fmt-model`](skills/air-migrate-migrate-fmt-model/SKILL.md) | Orchestrates inspection, planning, template generation, and config-authorized execution/registration/validation with resumable stage gates | Prepared workload or end-to-end migration |

Example invocation:

```text
Use $air-migrate-migrate-fmt-model to migrate the model configured in migrate/config.yaml.
```

### Template-driven generation

The planner compares every approved template against the inspected model's architecture, size, context length, training semantics, and memory requirements, then records the best-fitting recipe and rationale. The generator must copy that exact template to `migrate/output/air_workload` before customization. It does not silently choose a similar recipe or generate the AIR files from scratch.

You can also materialize a planned template directly:

```bash
python3 skills/air-migrate-generate-air-job/scripts/materialize_air_template.py \
  --recipe trl_full_fsdp
```

The canonical destination `migrate/output/air_workload` must be empty. The materializer reads `migrate/config.yaml` automatically and applies its compute block while copying the template. The generated copy may change model/data paths, secret references, compute, hyperparameters, output locations, and experiment names, while preserving the selected recipe's TRL, adapter, FSDP, launcher, merge, and registration semantics. This generated copy is what you run to migrate the model.

## Migration semantics

- **Retrain (`use_existing_weights: false`):** Start from the original base model and reproduce the legacy training workflow. Prefer an exact portable `system.ai` model; otherwise download the recovered Hugging Face model.
- **Continue (`use_existing_weights: true`):** Load the validated `existing_weights_volume_location` directly when populated; otherwise materialize the legacy fine-tuned weights and tokenizer from the exact configured UC version. Train further from those worker-readable paths. This creates a different model and is not reproduction.
- **Repackage:** Register portable existing weights without training. This is artifact migration, not an AI Runtime training migration.

LoRA training updates PEFT adapters and should be evaluated as a behavioral replacement even though its training workflow merges the adapter into a full inference checkpoint for registration. Full fine-tuning with FSDP can target full-weight behavioral and metric parity, but hidden legacy-service details generally prevent a byte-identical guarantee.
