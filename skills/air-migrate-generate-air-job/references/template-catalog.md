# Approved AIR template catalog

The repository-level `air_templates/` directory is authoritative. Compare all candidates against the inspected model, then copy one complete best-fitting template to `migrate/output/air_workload` before editing it.

| Recipe | Source directory | Training behavior | Reference workload |
|---|---|---|---|
| `trl_lora` | `air_templates/trl_lora` | TRL `SFTTrainer`, unquantized bf16 LoRA with DDP; merged full-model registration | Llama 3.1 8B Instruct on 8xH100 |
| `trl_lora_fsdp` | `air_templates/trl_lora_fsdp` | TRL `SFTTrainer`, unquantized bf16 LoRA with PEFT-aware FSDP full sharding; merged full-model registration | Llama 3.1 8B Instruct on 8xH100 |
| `trl_full_fsdp` | `air_templates/trl_full_fsdp` | TRL `SFTTrainer`, unquantized full-weight SFT with FSDP full sharding and a gathered portable checkpoint | Llama 3.1 8B Instruct on 8xH100 |

Every template must contain:

- `train.yaml`: AIR compute, dependencies, secrets, launcher, and workload parameters.
- `train.py`: thin framework-specific entry point containing only `run_training` and `main`.
- `helper_utils.py`: YAML loading, value coercion, formatting, path/source/FSDP validation, and configuration translation.
- `training_utils.py`: distributed/runtime staging, dataset/model construction, trainer configuration, MLflow, checkpoint, merge, and registration operations.
- `merge.py`: required for LoRA templates only; invokes `training_utils.merge_peft_model` for AIR runs that produced an adapter without executing `01_runner.py`.
- `01_runner.py`: Databricks source notebook using `@distributed` for training; LoRA variants also merge the adapter into a portable full checkpoint.
- `02_register_uc.py`: dual-mode post-training source notebook that reads the target from `train.yaml`, uses a widget interactively or `--mlflow-run-id` under AIR, resumes the selected training MLflow run, and creates the UC model version.

Keep helper placement consistent across all recipes. `train.py` must not define configuration, staging, dataset/model-construction, MLflow, merge, or registration helpers. It imports configuration functions from colocated `helper_utils.py` and runtime/training functions from colocated `training_utils.py`; notebooks and post-training entry points use the same split.

Every template's top-level `experiment_name` and `training_config.experiment_path` must be the same absolute workspace path. Generation replaces both with the exact required `migrate/config.yaml.source.migration_experiment_path`. Template training creates the experiment through `mlflow.set_experiment` when launched interactively, reuses an AIR-managed run only when its experiment ID matches, and rejects registration against a training run from another experiment. The orchestrator ensures the experiment exists before every AIR dry-run or live submission.

Every `train.yaml` must set `training_config.registered_model_name` to a three-level Unity Catalog name. PEFT templates must also set a distinct UC Volume `merged_output_dir`. Their training workflow reloads the unquantized base model, applies the adapter with a safe merge, and writes portable full weights; `02_register_uc.py` registers only that merged checkpoint. An AIR LoRA training command does not run the merge cell in `01_runner.py`, so the generated module must run `merge.py` before registration. The TRL LoRA DDP recipe must reject quantized base models and keep one base replica per worker. The TRL LoRA-FSDP recipe must also reject quantization, configure the PEFT-aware FSDP wrapping policy, load the base after FSDP activates rank-0-efficient loading, use FSDP activation checkpointing, and call adapter saves collectively on every rank. The TRL full-FSDP recipe must omit PEFT, use the same loading/checkpointing constraints, and call the final full-state save collectively on every rank. Full-FSDP registration must exclude intermediate checkpoint directories and optimizer state. All registration notebooks resume the training MLflow run identified by the user.

The registration source must be valid both as a Databricks notebook and as a normal Python file. Guard `dbutils` calls, keep magic directives first in their command cells, accept `--mlflow-run-id`, and list every script-mode dependency in `train.yaml`. Run registration as one AIR Python process on the configured compute; never use the training `torchrun` launcher for it. Plan sufficient host memory and temporary storage for MLflow to materialize more than one copy of the portable checkpoint.

Every generated template carries a validated input-source tuple: `use_existing_weights`, `existing_weights_volume_location`, `model_source`, `source_model_uri`, `model_name`, and `tokenizer_path`. `existing_uc` requires the pinned source URI plus either the validated configured location or AIR-materialized Volume paths. `system_ai` requires a pinned UC URI plus materialized paths; `hugging_face` requires a null UC URI plus remote references. Generated workloads must log this provenance with the registered output.

Every template also carries the same input-staging contract. A model or tokenizer directory under `/Volumes` remains the durable governed source but is copied to `training_config.local_model_cache_dir` before Transformers loads it. The cache is absolute, node-local, ephemeral, and outside `/Volumes`, DBFS, the workspace filesystem, and every durable output directory. `training_config.local_model_cache_copy_workers` bounds parallel copies. Ranks sharing a node coordinate with a file lock, reuse an atomically marked complete copy, verify copied file sizes, and load the local path with network access disabled. A separate tokenizer source is staged independently; an identical source reuses the model cache. LoRA merge-time base reloads use the same behavior. Remote Hugging Face references bypass this Volume-specific prefetch.

Plan local disk on every AIR node for one full source checkpoint plus tokenizer and safety reserve; the template enforces at least 1 GiB or 10 percent of staged bytes, whichever is greater. This space is independent of the durable UC Volume space required for adapters, Trainer checkpoints, merged models, and full checkpoints. Do not use the cache for outputs or registration artifacts. Preserve MLflow parameters and metrics for source/load paths, cache hits, copied bytes/files, lock wait, copy duration, and throughput.

Template compute is only a reference default. Generation must replace it with `migrate/config.yaml.compute`, using one of AIR's supported resource shapes: `GPU_1xA10`/1, `GPU_1xH100`/1, or `GPU_8xH100`/8. The generated `torchrun --nproc_per_node` count and notebook `@distributed` count/type must derive from that same block. Planning must reject a requested shape that cannot run the selected model and recipe; generation never falls back to the template's 8xH100 default.

Template selection must record architecture and tokenizer compatibility, model size and context length, training semantics, GPU/host-memory fit, node-local input-cache capacity, and output-artifact compatibility. The reference workload is evidence about the starting point, not a model-name routing rule.

## Allowed adaptation surface

Change configuration values that describe the migrated workload: identity, source revision, data, secrets, requested compute, output locations, hyperparameters, the required migration experiment path, registered-model target, and documented task-format behavior. Keep YAML, `torchrun`, and notebook configuration synchronized. Never invent a recipe-specific fallback experiment.

Changing the recipe's framework, adapter/quantization/FSDP training mode, distributed launcher, core checkpoint semantics, or registration artifact type is a new template variant. Extend and validate the template catalog instead of mutating an incompatible copy.

The customized copy under `migrate/output/air_workload` is the runnable migration artifact. Never submit a directory under `air_templates/` directly.

## No-fallback rule

Template selection comes from `plan.recipe`. Never infer the recipe from the model name during generation or fall back from FSDP to a single-device recipe. A merged LoRA registration is a full inference checkpoint, but the training method remains PEFT and must not be described as full-weight fine-tuning.
