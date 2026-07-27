# Databricks notebook source
# ruff: noqa: E402,F821
# MAGIC %md
# MAGIC # Fine-tune Llama 3.1 8B Instruct with TRL SFTTrainer and LoRA
# MAGIC
# MAGIC This notebook runs the same `train.py`, utility modules, and `train.yaml`
# MAGIC used by the AI Runtime CLI. Attach it to **Serverless GPU** with the accelerator in
# MAGIC `train.yaml` and use the configured AI environment. The default data
# MAGIC paths consume the JSONL files produced by `example_setup/02_stage_data`.

# COMMAND ----------

# MAGIC %pip install "trl==0.27.1" "peft>=0.17,<0.19" "transformers>=4.56,<5"
# MAGIC %pip install "datasets>=3.0,<5" "accelerate>=1.4,<2" "mlflow>=3.6,<4"
# MAGIC %pip install "pyyaml>=6.0" "safetensors>=0.4" "hf-transfer==0.1.9"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
from pathlib import Path


def find_project_dir() -> Path:
    """Locate the generated AIR project directory in local or notebook contexts."""
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd())

    try:
        context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        notebook_path = context.notebookPath().get()
    except Exception:
        pass
    else:
        candidates.append(
            Path("/Workspace") / notebook_path.lstrip("/").rsplit("/", 1)[0]
        )

    for candidate in dict.fromkeys(candidates):
        if all(
            (candidate / name).is_file()
            for name in (
                "train.py",
                "train.yaml",
                "helper_utils.py",
                "training_utils.py",
            )
        ):
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find train.py, train.yaml, helper_utils.py, and "
        f"training_utils.py; searched: {searched}"
    )


PROJECT_DIR = find_project_dir()
CONFIG_PATH = PROJECT_DIR / "train.yaml"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from helper_utils import _needs_hf_token, load_training_config


training_config, _ = load_training_config(CONFIG_PATH)
print(f"Configuration: {CONFIG_PATH}")
print(f"Model source: {training_config['model_source']}")
print(f"Configured model: {training_config['model_name']}")
print(f"Source model URI: {training_config.get('source_model_uri')}")
print(f"Migration experiment: {training_config['experiment_path']}")
print(
    f"Tokenizer: {training_config.get('tokenizer_path') or training_config['model_name']}"
)
print(f"Node-local model cache: {training_config['local_model_cache_dir']}")
print(f"Cache copy workers: {training_config['local_model_cache_copy_workers']}")
print(f"Training data: {training_config['train_data_path']}")
print(f"Evaluation data: {training_config['eval_data_path']}")
print(f"Adapter output: {training_config['output_dir']}")
print(f"Merged output: {training_config['merged_output_dir']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure gated Hugging Face access
# MAGIC
# MAGIC This step is skipped for Volume, `system.ai`, and public Hugging Face
# MAGIC inputs. For a gated Hugging Face repository, the generated workload's
# MAGIC `HF_TOKEN` secret reference is fetched without exposing the token.

# COMMAND ----------

import yaml


HF_TOKEN = os.environ.get("HF_TOKEN")
if _needs_hf_token(training_config) and not HF_TOKEN:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        workload = yaml.safe_load(handle)
    secret_reference = str((workload.get("secrets") or {}).get("HF_TOKEN", ""))
    if "/" not in secret_reference:
        raise ValueError(
            "A gated Hugging Face source requires secrets.HF_TOKEN as <scope>/<key>"
        )
    default_scope, default_key = secret_reference.split("/", 1)
    dbutils.widgets.text("hf_secret_scope", default_scope, "HF secret scope")
    dbutils.widgets.text("hf_secret_key", default_key, "HF secret key")
    HF_TOKEN = dbutils.secrets.get(
        scope=dbutils.widgets.get("hf_secret_scope").strip(),
        key=dbutils.widgets.get("hf_secret_key").strip(),
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Launch distributed TRL SFT training
# MAGIC
# MAGIC Each worker keeps one unquantized bf16 base-model replica. Hugging Face
# MAGIC Accelerate wraps the trainable LoRA model with DDP, and `SFTTrainer`
# MAGIC shards the dataset across workers. Each assistant turn is trained as a
# MAGIC conversational completion, so prompt tokens remain masked from loss.
# MAGIC Volume inputs are copied and system.ai artifacts are downloaded only
# MAGIC once per node into the locked cache configured in `train.yaml`.
# MAGIC Hugging Face inputs are downloaded by Transformers.

# COMMAND ----------

import yaml
from serverless_gpu.compute import GPUType
from serverless_gpu.launcher import distributed


with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    workload = yaml.safe_load(handle)

supported_air_accelerators = {
    "GPU_1xA10": (1, "A10"),
    "GPU_1xH100": (1, "H100"),
    "GPU_8xH100": (8, "H100"),
}
compute = workload.get("compute")
if not isinstance(compute, dict):
    raise ValueError("train.yaml compute must be a mapping")
num_gpus = compute.get("num_accelerators")
accelerator_type = compute.get("accelerator_type")
if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus <= 0:
    raise ValueError("compute.num_accelerators must be a positive integer")
accelerator_specification = supported_air_accelerators.get(accelerator_type)
if accelerator_specification is None:
    supported = ", ".join(sorted(supported_air_accelerators))
    raise ValueError(
        f"Unsupported compute.accelerator_type={accelerator_type!r}; "
        f"supported values: {supported}"
    )
expected_gpus, gpu_type_name = accelerator_specification
if num_gpus != expected_gpus:
    raise ValueError(
        f"compute.accelerator_type={accelerator_type!r} requires "
        f"num_accelerators={expected_gpus}, got {num_gpus}"
    )
gpu_type = getattr(GPUType, gpu_type_name)


@distributed(gpus=num_gpus, gpu_type=gpu_type)
def run_training_job(config_path: str, hf_token: str | None):
    """Run plain-LoRA training on one AIR distributed worker."""
    import os
    import sys
    from pathlib import Path

    project_dir = Path(config_path).resolve().parent
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    from train import run_training

    return run_training(config_path=config_path, hf_token=hf_token)


distributed_results = run_training_job.distributed(str(CONFIG_PATH), HF_TOKEN)
rank_zero_result = next(
    (result for result in distributed_results if result and result.get("rank") == 0),
    None,
)
if rank_zero_result is None:
    raise RuntimeError(f"Training returned no rank-zero result: {distributed_results}")

print(f"MLflow run ID: {rank_zero_result['mlflow_run_id']}")
print(f"Adapter output: {rank_zero_result['output_dir']}")
rank_zero_result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge the LoRA adapter into the base model
# MAGIC
# MAGIC The merge reloads the unquantized base model, applies the trained
# MAGIC adapter with PEFT's safe merge, and writes portable sharded
# MAGIC safetensors to `training_config.merged_output_dir`. The destination
# MAGIC must be an empty UC Volume directory. Volume and system.ai base weights
# MAGIC use the same node-local staging path before the merge loads them.

# COMMAND ----------


@distributed(gpus=num_gpus, gpu_type=gpu_type)
def run_merge_job(config_path: str, hf_token: str | None):
    """Merge plain-LoRA adapter weights on one AIR distributed worker."""
    import os
    import sys
    from pathlib import Path

    project_dir = Path(config_path).resolve().parent
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    from training_utils import merge_peft_model

    return merge_peft_model(config_path=config_path)


merge_results = run_merge_job.distributed(str(CONFIG_PATH), HF_TOKEN)
rank_zero_merge = next(
    (result for result in merge_results if result and result.get("rank") == 0),
    None,
)
if rank_zero_merge is None:
    raise RuntimeError(f"Merge returned no rank-zero result: {merge_results}")

print(f"Merged checkpoint: {rank_zero_merge['merged_output_dir']}")
rank_zero_merge

# COMMAND ----------

# MAGIC %md
# MAGIC The adapter directory contains the plain bf16 LoRA training output and
# MAGIC resumable Trainer checkpoints. The separate merged directory contains
# MAGIC full model weights with the adapter applied. Run `02_register_uc` with
# MAGIC the MLflow run ID printed above to register those weights in Unity
# MAGIC Catalog. Model weights come from the selected Volume, `system.ai`, or
# MAGIC Hugging Face source. Submit the same training workload from a terminal with:
# MAGIC
# MAGIC ```bash
# MAGIC air run --file train.yaml --watch
# MAGIC ```
