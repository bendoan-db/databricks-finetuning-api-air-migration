# Approved AIR template catalog

The repository-level `air_templates/` directory is authoritative. Always copy one complete template before editing it.

| Recipe | Source directory | Training behavior | Reference workload |
|---|---|---|---|
| `axolotl_qlora` | `air_templates/axolotl_qlora` | 4-bit QLoRA without FSDP; merged full-model registration | Llama 3.1 8B Instruct on 8xH100 |
| `axolotl_full_fsdp` | `air_templates/axolotl_full_fsdp` | Unquantized full fine-tuning with FSDP full sharding | Llama 3.1 8B Instruct on 8xH100 |
| `axolotl_qlora_fsdp` | `air_templates/axolotl_qlora_fsdp` | 4-bit QLoRA with FSDP and CPU-efficient loading; merged full-model registration | Llama 3.1 70B Instruct on 8xH100 |

Every template must contain:

- `train.yaml`: AIR compute, dependencies, secrets, launcher, and workload parameters.
- `train.py`: shared Axolotl training entry point plus the template-specific MLflow registration helper.
- `01_runner.py`: Databricks source notebook using `@distributed`, then registering the final artifact to the YAML-configured UC model.

Every `train.yaml` must set `training_config.registered_model_name` to a three-level Unity Catalog name. PEFT templates must also set a distinct UC Volume `merged_output_dir`. Their runner reloads the unquantized base model, applies the adapter with a safe merge, writes portable full weights, and registers only that merged checkpoint. The full-FSDP template registers its portable Transformers checkpoint without intermediate checkpoint directories or optimizer state. All registration paths resume the training MLflow run.

Every template also carries a validated input-source tuple: `use_existing_weights`, `model_source`, `source_model_uri`, `model_name`, and `tokenizer_path`. `existing_uc` and `system_ai` require pinned UC URIs plus materialized Volume paths; `hugging_face` requires a null UC URI plus remote model references. Generated workloads must log this provenance with the registered output.

## Allowed adaptation surface

Change configuration values that describe the migrated workload: identity, source revision, data, secrets, compute, output locations, hyperparameters, experiment naming, registered-model target, and documented task-format behavior. Keep YAML and notebook configuration synchronized.

Changing the recipe's adapter/quantization/FSDP training mode, distributed launcher, core checkpoint semantics, or registration artifact type is a new template variant. Extend and validate the template catalog instead of mutating an incompatible copy.

## No-fallback rule

Template selection comes from `plan.recipe`. Never infer the recipe from the model name during generation or fall back from FSDP to a single-device recipe. A merged QLoRA registration is a full inference checkpoint, but the training method remains PEFT and must not be described as full-weight fine-tuning.
