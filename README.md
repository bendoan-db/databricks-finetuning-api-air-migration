# Foundation Model Fine-tuning to AI Runtime migration

This project demonstrates how to migrate a model trained with the legacy Databricks Foundation Model Fine-tuning API to a reproducible Databricks AI Runtime workload. It contains:

- An optional legacy fine-tuning demo using the IBM TabFormer dataset
- Axolotl AI Runtime templates for QLoRA, full fine-tuning with FSDP, and QLoRA with FSDP
- Coding-assistant skills that inspect, plan, materialize, generate, and validate a migration

## Repository layout

| Directory | Purpose |
|---|---|
| `example_setup/` | Optional demo that creates a TabFormer source model with the legacy fine-tuning API |
| `air_templates/` | Approved Axolotl workload templates for Databricks AI Runtime |
| `migrate/` | Identifies the source UC model version, target UC model, and generated migration outputs |
| `skills/` | Portable `air-migrate-*` skills for coding assistants |

## How To Run

If you already have a Unity Catalog model produced by the legacy Foundation Model Fine-tuning API:

1. Update [`migrate/config.yaml`](migrate/config.yaml) with the source model, `use_existing_weights`, and optional target location. Set the flag to `true` to continue from the configured UC model version, or `false` to retrain from its original base model (`system.ai` when available, otherwise Hugging Face).
2. Install or expose the folders under `skills/` to your coding assistant.
3. Invoke the migration orchestrator:

   ```text
   Use $air-migrate-migrate-fmt-model to migrate the model configured in migrate/config.yaml.
   ```

4. Review the generated migration manifest, selected AIR recipe, workload, and validation criteria—including the allowed token-accuracy regression—before authorizing training.
5. Run the generated workload after its static checks pass. Use either launch path:

   - **AIR CLI:** Authenticate the Databricks CLI, confirm the `HF_TOKEN` secret reference in `train.yaml`, and run from the generated workload directory:

     ```bash
     cd migrate/output/air_workload
     air run --dry-run --file train.yaml
     air run --file train.yaml --watch
     ```

   - **Runner notebook:** Set `training_config.registered_model_name` in `train.yaml` to the target `<catalog>.<schema>.<model>`. For PEFT recipes, also set `training_config.merged_output_dir` to an empty UC Volume directory. Import `01_runner.py`, `train.py`, and `train.yaml` into the same Databricks workspace folder, open `01_runner.py` as a Databricks source notebook, attach it to Serverless GPU using the configured compute and AI Runtime version (the templates default to 8xH100 and AI v5), configure the Hugging Face secret widgets, and select **Run all**. The notebook launches training through `@distributed`; PEFT variants then merge their adapter into the unquantized base model, and every recipe registers a portable full checkpoint as a new Unity Catalog model version.

6. Validate the registered model. The validation workflow materializes the exact legacy version from `migrate/config.yaml` and the newly registered version, then compares their assistant response-token accuracy on the same hashed evaluation JSONL. A pass/fail result requires a predeclared maximum regression and equivalent tokenizer/chat serialization.

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
  use_existing_weights: true

target:
  catalog:
  schema:
  model:
```

Leave all target fields blank to register the migrated model as a new version of the source model. Otherwise, populate all three target fields. Partially populated targets are invalid.

`use_existing_weights: true` materializes the exact source version and uses those files as the Axolotl base checkpoint. `false` recovers the original base-model ID, checks for an exact portable model under `system.ai`, and falls back to the Hugging Face repository only when no match exists. Every generated `train.yaml` records the selected source and rejects inconsistent URI/path combinations.

## AI Runtime templates

Each template contains `train.yaml`, `train.py`, and a Databricks source notebook named `01_runner.py`. The same training entry point supports AIR CLI submission with `torchrun` and notebook execution with `@distributed`.

| Recipe | Template | Intended use | Default reference workload |
|---|---|---|---|
| `axolotl_qlora` | [`air_templates/axolotl_qlora`](air_templates/axolotl_qlora/train.yaml) | Cost-efficient 4-bit PEFT without FSDP | Llama 3.1 8B Instruct, 8xH100 |
| `axolotl_full_fsdp` | [`air_templates/axolotl_full_fsdp`](air_templates/axolotl_full_fsdp/train.yaml) | Full-weight fine-tuning and closest practical parity | Llama 3.1 8B Instruct, 8xH100 |
| `axolotl_qlora_fsdp` | [`air_templates/axolotl_qlora_fsdp`](air_templates/axolotl_qlora_fsdp/train.yaml) | PEFT when the quantized base model still requires sharding | Llama 3.1 70B Instruct, 8xH100 |

Before running a template, update its source tuple (`use_existing_weights`, `model_source`, `source_model_uri`, `model_name`, and `tokenizer_path`), UC Volume paths, experiment/output locations, and any `HF_TOKEN` Databricks secret reference. Existing UC and `system.ai` sources use materialized Volume paths; a Hugging Face token must have access to the selected gated remote model.

Validate and submit from the workload directory:

```bash
air run --dry-run --file train.yaml
air run --file train.yaml --watch
```

## Migration skills

Install or expose the folders under `skills/` to your coding assistant. Invoke the orchestrator for the complete workflow or use a stage skill directly.

| Skill | Capability | Primary output |
|---|---|---|
| [`air-migrate-inspect-fmt-model`](skills/air-migrate-inspect-fmt-model/SKILL.md) | Resolves the UC model version and recovers MLflow lineage, datasets, base model, task semantics, hyperparameters, checkpoints, artifacts, permissions, and unknowns | `migrate/output/migration-manifest.yaml` |
| [`air-migrate-plan-air-training`](skills/air-migrate-plan-air-training/SKILL.md) | Chooses retrain, continue, or repackage semantics; full fine-tuning or QLoRA; FSDP requirements; compute; checkpointing; and acceptance criteria | Training plan in the migration manifest |
| [`air-migrate-materialize-uc-model`](skills/air-migrate-materialize-uc-model/SKILL.md) | Applies `use_existing_weights`, resolves `system.ai` versus Hugging Face, and materializes portable weights from exact UC versions when required | Pinned source resolution and worker-readable Volume paths |
| [`air-migrate-generate-air-job`](skills/air-migrate-generate-air-job/SKILL.md) | Materializes one approved AIR template, safely customizes it, maintains YAML/notebook parity, and performs static and AIR dry-run checks | Runnable AIR workload |
| [`air-migrate-compare-token-accuracy`](skills/air-migrate-compare-token-accuracy/SKILL.md) | Compares deterministic assistant response-token accuracy for the configured legacy model and newly registered migrated model, including tokenizer comparability and regression gating | `migrate/output/token-accuracy-evaluation.json` |
| [`air-migrate-validate-model-migration`](skills/air-migrate-validate-model-migration/SKILL.md) | Validates workload execution, artifacts, UC registration, shared-dataset metrics, behavioral parity, and promotion criteria | `migrate/output/migration-validation.yaml` |
| [`air-migrate-migrate-fmt-model`](skills/air-migrate-migrate-fmt-model/SKILL.md) | Orchestrates inspection, planning, template generation, authorized execution/registration, and validation with resumable stage gates | End-to-end migration |

Example invocation:

```text
Use $air-migrate-migrate-fmt-model to migrate the model configured in migrate/config.yaml.
```

### Template-driven generation

The planner selects the recipe; the generator must copy that exact template before customization. It does not silently choose a similar recipe or generate the AIR files from scratch.

You can also materialize a planned template directly:

```bash
python3 skills/air-migrate-generate-air-job/scripts/materialize_air_template.py \
  --recipe axolotl_full_fsdp \
  --output-dir migrate/output/air_workload
```

The destination must be empty. The generated copy may change model/data paths, secret references, compute, hyperparameters, output locations, and experiment names, while preserving the selected recipe's adapter, quantization, FSDP, launcher, and artifact semantics.

## Migration semantics

- **Retrain (`use_existing_weights: false`):** Start from the original base model and reproduce the legacy training workflow. Prefer an exact portable `system.ai` model; otherwise download the recovered Hugging Face model.
- **Continue (`use_existing_weights: true`):** Materialize the legacy fine-tuned weights and tokenizer from the exact configured UC version, then train further from those worker-readable paths. This creates a different model and is not reproduction.
- **Repackage:** Register portable existing weights without training. This is artifact migration, not an AI Runtime training migration.

QLoRA training updates PEFT adapters and should be evaluated as a behavioral replacement even though the runner merges those adapters into full inference checkpoints for registration. Full fine-tuning with FSDP can target full-weight behavioral and metric parity, but hidden legacy-service details generally prevent a byte-identical guarantee.
