# Databricks notebook source
# MAGIC %md
# MAGIC # Register the TRL LoRA model in Unity Catalog
# MAGIC
# MAGIC Run this notebook after `01_runner` or an AIR job has trained the LoRA
# MAGIC adapter. Enter the training MLflow run ID in the widget, or submit this
# MAGIC source through AIR with `--mlflow-run-id`. If needed, the interactive
# MAGIC notebook merges the adapter into the base model before registration.
# MAGIC For AIR registration, run `merge.py` first. Paths and the
# MAGIC Unity Catalog model target are read from `train.yaml`.

# COMMAND ----------

# MAGIC %pip install "mlflow>=3.6,<4" "peft>=0.17,<0.19" "transformers>=4.56,<5"
# MAGIC %pip install "accelerate>=1.4,<2" "pyyaml>=6.0" "safetensors>=0.4" "hf-transfer==0.1.9"

# COMMAND ----------

# ruff: noqa: E402,F821
if "dbutils" in globals():
    dbutils.library.restartPython()

# COMMAND ----------

import argparse
import json
import os
import sys
from pathlib import Path


IS_DATABRICKS_NOTEBOOK = "dbutils" in globals()


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
# MAGIC Large checkpoints should use the AIR CLI path so registration has the
# MAGIC training workload's host memory and temporary storage.

# COMMAND ----------

if IS_DATABRICKS_NOTEBOOK:
    dbutils.widgets.text("mlflow_run_id", "", "MLflow run ID")
    dbutils.widgets.text("hf_secret_scope", "your-secret-scope", "HF secret scope")
    dbutils.widgets.text("hf_secret_key", "hf_token", "HF secret key")
    mlflow_run_id = dbutils.widgets.get("mlflow_run_id").strip()
    hf_secret_scope = dbutils.widgets.get("hf_secret_scope").strip()
    hf_secret_key = dbutils.widgets.get("hf_secret_key").strip()
else:
    parser = argparse.ArgumentParser(
        description="Register a trained TRL LoRA checkpoint in Unity Catalog."
    )
    parser.add_argument("--mlflow-run-id", required=True)
    parser.add_argument("--hf-secret-scope", default="your-secret-scope")
    parser.add_argument("--hf-secret-key", default="hf_token")
    args = parser.parse_args()
    mlflow_run_id = args.mlflow_run_id.strip()
    hf_secret_scope = args.hf_secret_scope.strip()
    hf_secret_key = args.hf_secret_key.strip()

if not mlflow_run_id:
    raise ValueError(
        "Set the mlflow_run_id widget or pass --mlflow-run-id before registration"
    )

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
# MAGIC ## Materialize the merged checkpoint
# MAGIC
# MAGIC If `merged_output_dir` already contains a portable full checkpoint it
# MAGIC is reused. Otherwise the notebook launches the same GPU merge used by
# MAGIC `01_runner`. Volume-backed base weights are staged into the node-local
# MAGIC cache configured in `train.yaml` before the merge loads them. Remote
# MAGIC Hugging Face sources require a token.

# COMMAND ----------

merged_output_dir = Path(training_config["merged_output_dir"]).resolve()
merged_weights = [
    path
    for pattern in ("model*.safetensors", "pytorch_model*.bin")
    for path in merged_output_dir.glob(pattern)
    if path.is_file()
]
merged_checkpoint_ready = (
    (merged_output_dir / "config.json").is_file()
    and bool(merged_weights)
    and not (merged_output_dir / "adapter_config.json").exists()
    and not any(merged_output_dir.glob("adapter_model.*"))
)

if merged_checkpoint_ready:
    print(f"Reusing merged checkpoint: {merged_output_dir}")
else:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    model_references = [training_config["model_name"]]
    if training_config.get("tokenizer_path"):
        model_references.append(training_config["tokenizer_path"])
    requires_hf_token = any(
        not Path(str(reference)).expanduser().is_absolute()
        for reference in model_references
    )
    if requires_hf_token and not HF_TOKEN:
        if not IS_DATABRICKS_NOTEBOOK:
            raise ValueError(
                "Set HF_TOKEN in the AIR environment before materializing a "
                "merged checkpoint from a remote Hugging Face model."
            )
        if (
            not hf_secret_scope
            or hf_secret_scope == "your-secret-scope"
            or not hf_secret_key
        ):
            raise ValueError(
                "Set hf_secret_scope/hf_secret_key to a Databricks secret "
                "containing a Hugging Face token, or provide HF_TOKEN."
            )
        HF_TOKEN = dbutils.secrets.get(
            scope=hf_secret_scope,
            key=hf_secret_key,
        )

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
    def run_merge_job(config_path: str, hf_token: str | None):
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
if not IS_DATABRICKS_NOTEBOOK:
    print(json.dumps(registered_model, indent=2))
registered_model
