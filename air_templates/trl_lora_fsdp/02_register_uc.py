# Databricks notebook source
# ruff: noqa: E402,F821
# MAGIC %md
# MAGIC # Register the TRL LoRA/FSDP model in Unity Catalog
# MAGIC
# MAGIC Run this notebook after `01_runner` has trained and merged the LoRA
# MAGIC adapter. Enter the training MLflow run ID in the widget. The merged
# MAGIC checkpoint path and Unity Catalog model target are read from
# MAGIC `train.yaml`.

# COMMAND ----------

# MAGIC %pip install "mlflow>=3.6,<4" "transformers>=4.56,<5"
# MAGIC %pip install "accelerate>=1.4,<2" "pyyaml>=6.0" "safetensors>=0.4"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

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

from helper_utils import load_training_config
from training_utils import register_trained_model


training_config, _ = load_training_config(CONFIG_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select the training run
# MAGIC
# MAGIC Use the MLflow run ID printed by `01_runner` or the AIR training job.
# MAGIC This is not the numeric Databricks Jobs run ID.

# COMMAND ----------

dbutils.widgets.text("mlflow_run_id", "", "MLflow run ID")

mlflow_run_id = dbutils.widgets.get("mlflow_run_id").strip()
if not mlflow_run_id:
    raise ValueError("Set the mlflow_run_id widget before registering the model")

import mlflow


mlflow.set_tracking_uri("databricks")
training_run = mlflow.tracking.MlflowClient().get_run(mlflow_run_id)

print(f"MLflow run ID: {training_run.info.run_id}")
print(f"MLflow run status: {training_run.info.status}")
print(f"Migration experiment: {training_config['experiment_path']}")
print(f"Merged checkpoint: {training_config['merged_output_dir']}")
print(f"UC model target: {training_config['registered_model_name']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the merged model
# MAGIC
# MAGIC The caller needs `USE CATALOG`, `USE SCHEMA`, and permission to create
# MAGIC or update the configured registered model. This logs the merged full
# MAGIC checkpoint to the selected MLflow run and creates a new UC model version.

# COMMAND ----------

registered_model = register_trained_model(
    mlflow_run_id=mlflow_run_id,
    config_path=CONFIG_PATH,
)
print(f"Registered model: {registered_model['model_uri']}")
registered_model
