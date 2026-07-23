# Approved AIR template catalog

The repository-level `air_templates/` directory is authoritative. Always copy one complete template before editing it.

| Recipe | Source directory | Training behavior | Reference workload |
|---|---|---|---|
| `axolotl_qlora` | `air_templates/axolotl_qlora` | 4-bit QLoRA without FSDP | Llama 3.1 8B Instruct on 8xH100 |
| `axolotl_full_fsdp` | `air_templates/axolotl_full_fsdp` | Unquantized full fine-tuning with FSDP full sharding | Llama 3.1 8B Instruct on 8xH100 |
| `axolotl_qlora_fsdp` | `air_templates/axolotl_qlora_fsdp` | 4-bit QLoRA with FSDP and CPU-efficient loading | Llama 3.1 70B Instruct on 8xH100 |

Every template must contain:

- `train.yaml`: AIR compute, dependencies, secrets, launcher, and workload parameters.
- `train.py`: shared Axolotl training entry point plus the template-specific MLflow registration helper.
- `01_runner.py`: Databricks source notebook using `@distributed`, then registering the final artifact to the YAML-configured UC model.

Every `train.yaml` must set `training_config.registered_model_name` to a three-level Unity Catalog name. The runner resumes the training MLflow run before logging the model. QLoRA templates register only the final adapter config and weights with base-model/tokenizer lineage; the full-FSDP template registers the portable Transformers checkpoint without intermediate checkpoint directories or optimizer state.

## Allowed adaptation surface

Change configuration values that describe the migrated workload: identity, source revision, data, secrets, compute, output locations, hyperparameters, experiment naming, registered-model target, and documented task-format behavior. Keep YAML and notebook configuration synchronized.

Changing the recipe's adapter/quantization/FSDP mode, distributed launcher, core checkpoint semantics, or output artifact type is a new template variant. Extend and validate the template catalog instead of mutating an incompatible copy.

## No-fallback rule

Template selection comes from `plan.recipe`. Never infer the recipe from the model name during generation, fall back from FSDP to a single-device recipe, or call a QLoRA adapter a full-weight migration.
