# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-tune Llama 3.1 8B Instruct with Axolotl QLoRA
# MAGIC
# MAGIC This notebook runs the same `train.py` and `train.yaml` used by the AI
# MAGIC Runtime CLI. Attach it to **Serverless GPU**, select **8xH100**, and use
# MAGIC the **AI v5** environment. The default data paths consume the JSONL files
# MAGIC produced by `example_setup/02_stage_data`.

# COMMAND ----------

# MAGIC %pip install --no-build-isolation "axolotl[flash-attn]==0.13.1"
# MAGIC %pip install "trl==0.27.1" "torchao==0.16.0" "mlflow>=3.6,<4" "pyyaml>=6.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
from pathlib import Path


def find_project_dir() -> Path:
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
        if (candidate / "train.py").is_file() and (candidate / "train.yaml").is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find train.py and train.yaml; searched: {searched}")


PROJECT_DIR = find_project_dir()
CONFIG_PATH = PROJECT_DIR / "train.yaml"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from train import load_training_config, register_trained_model


training_config, _ = load_training_config(CONFIG_PATH)
print(f"Configuration: {CONFIG_PATH}")
print(f"Model: {training_config['model_name']}")
print(f"Tokenizer: {training_config.get('tokenizer_path') or training_config['model_name']}")
print(f"Training data: {training_config['train_data_path']}")
print(f"Evaluation data: {training_config['eval_data_path']}")
print(f"Adapter output: {training_config['output_dir']}")
print(f"UC model target: {training_config['registered_model_name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Hugging Face access
# MAGIC
# MAGIC Remote gated models require a Hugging Face token. A model and tokenizer
# MAGIC materialized to absolute UC Volume paths do not. If `HF_TOKEN` is
# MAGIC already defined in the notebook environment, it is used instead.

# COMMAND ----------

dbutils.widgets.text("hf_secret_scope", "your-secret-scope", "HF secret scope")
dbutils.widgets.text("hf_secret_key", "hf_token", "HF secret key")

HF_TOKEN = os.environ.get("HF_TOKEN")
model_references = [training_config["model_name"]]
if training_config.get("tokenizer_path"):
    model_references.append(training_config["tokenizer_path"])
requires_hf_token = any(
    not Path(str(reference)).expanduser().is_absolute()
    for reference in model_references
)
if requires_hf_token and not HF_TOKEN:
    secret_scope = dbutils.widgets.get("hf_secret_scope").strip()
    secret_key = dbutils.widgets.get("hf_secret_key").strip()
    if not secret_scope or secret_scope == "your-secret-scope" or not secret_key:
        raise ValueError(
            "Set hf_secret_scope/hf_secret_key to a Databricks secret containing "
            "a Hugging Face token, or provide HF_TOKEN in the environment."
        )
    HF_TOKEN = dbutils.secrets.get(scope=secret_scope, key=secret_key)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Launch distributed training

# COMMAND ----------

import yaml
from serverless_gpu.compute import GPUType
from serverless_gpu.launcher import distributed


with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    workload = yaml.safe_load(handle)

num_gpus = int(workload["compute"]["num_accelerators"])
accelerator_type = workload["compute"]["accelerator_type"]
if num_gpus != 8 or accelerator_type != "GPU_8xH100":
    raise ValueError(
        "This runner expects compute.num_accelerators=8 and "
        "compute.accelerator_type=GPU_8xH100"
    )


@distributed(gpus=num_gpus, gpu_type=GPUType.H100)
def run_training_job(config_path: str, hf_token: str | None):
    import os
    import sys
    from pathlib import Path

    project_dir = Path(config_path).resolve().parent
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    from train import run_training

    return run_training(config_path=config_path)


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
# MAGIC ## Register the model in Unity Catalog
# MAGIC
# MAGIC The target is read from `training_config.registered_model_name` in
# MAGIC `train.yaml`. The runner logs the final adapter files to the training
# MAGIC MLflow run and creates a new UC model version. The caller needs
# MAGIC `USE CATALOG`, `USE SCHEMA`, and permission to create or update the
# MAGIC configured registered model.

# COMMAND ----------

registered_model = register_trained_model(
    training_result=rank_zero_result,
    config_path=CONFIG_PATH,
)
print(f"Registered model: {registered_model['model_uri']}")
registered_model

# COMMAND ----------

# MAGIC %md
# MAGIC The resulting directory contains the PEFT/QLoRA adapter and Axolotl
# MAGIC checkpoints, while Unity Catalog contains an MLflow PEFT model version
# MAGIC that lazily loads the configured base model and adapter. For remote
# MAGIC gated models, edit the `HF_TOKEN` secret reference in `train.yaml`;
# MAGIC materialized local model/tokenizer paths do not require it. Submit the
# MAGIC same training workload from a terminal with:
# MAGIC
# MAGIC ```bash
# MAGIC air run --file train.yaml --watch
# MAGIC ```
