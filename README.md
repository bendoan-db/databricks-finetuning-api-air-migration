# Foundation Model Fine-tuning to AI Runtime migration

This repository converts a model trained with the legacy Databricks Foundation Model Fine-tuning API into a ready-to-run Databricks AI Runtime handoff.

The migration is generation-only: it inspects and plans, generates workload files, validates them locally, and stops. Training, LoRA merge, registration, evaluation, and promotion belong to the receiving operator.

## Repository layout

| Directory | Purpose |
|---|---|
| `example_setup/` | Optional legacy fine-tuning demo notebooks |
| `air_templates/` | Approved TRL LoRA, LoRA/FSDP, and full-FSDP templates |
| `migrate/` | Migration configuration and ignored generated output |
| `skills/` | Inspection, planning, generation, and orchestration skills |

## Configure model loading

Edit `migrate/config.yaml`. Model input is selected mechanically by precedence:

| Priority | Field | Runtime behavior |
|---|---|---|
| 1 | `weights_volume_path` | Copy the existing checkpoint once per AIR node and load from node-local cache. |
| 2 | `system_ai_model_uri` | Download the exact versioned artifact through MLflow into the node-local cache. |
| 3 | `huggingface_model_id` | Download the repository through Transformers. |

Only nonblank fields participate. A populated Volume always wins; system.ai is used only without a Volume; Hugging Face is required only when neither prior field is selected.

```yaml
source:
  catalog: my_catalog
  schema: my_schema
  model: legacy_model
  version: 1
  weights_volume_path:
  system_ai_model_uri: models:/system.ai.meta_llama_v3_1_8b_instruct/1
  huggingface_model_id: meta-llama/Meta-Llama-3.1-8B-Instruct
  huggingface_token_secret: my-scope/hf-token
  migration_experiment_path: /Shared/ft-api-migration
  peft_only: true

target:
  catalog:
  schema:
  model:
  volume:

compute:
  num_accelerators: 8
  accelerator_type: GPU_8xH100
```

`system_ai_model_uri` must use `models:/system.ai.<model>/<version>` and is never auto-discovered. `huggingface_token_secret` is optional for public repositories and uses `<scope>/<key>`, never a token value. Set `peft_only: true` to require a LoRA workload: the planner selects `trl_lora` when the inspected bf16 base model and overhead fit on each requested accelerator, otherwise `trl_lora_fsdp`. It never falls back to full-weight training. With `false`, selection preserves the recovered full-weight or PEFT semantics. Leave all three target identity fields blank to reuse the source model name, or populate all three. `target.volume` is reserved metadata; it does not automatically rewrite data, checkpoint, or merge paths. Supported compute shapes are `GPU_1xA10`/1, `GPU_1xH100`/1, and `GPU_8xH100`/8.

## Run the migration

Configure `migrate/config.yaml` first. Both launch paths use the same orchestrator and stop after generating and locally validating the handoff.

### From an IDE

Prerequisites:

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and the [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install). On macOS, the CLI can be installed with Homebrew:

   ```bash
   brew tap databricks/tap
   brew install databricks
   ```

2. Authenticate the CLI to the target workspace and verify the profile. Replace the host and profile when not using `DEFAULT`:

   ```bash
   databricks auth login --host https://<workspace-host> --profile DEFAULT
   databricks current-user me --profile DEFAULT
   ```

3. From this repository root, install the [Databricks AI Dev Kit](https://github.com/databricks-solutions/ai-dev-kit). Its installer configures supported IDE agents at project scope and uses the `DEFAULT` profile unless `--profile` is supplied:

   ```bash
   bash <(curl -sL https://raw.githubusercontent.com/databricks-solutions/ai-dev-kit/main/install.sh) \
     --profile DEFAULT
   ```

   Follow the installer prompts, then restart or reload the IDE so its skills and Databricks tools are discovered. Windows users should follow the PowerShell installation instructions in the AI Dev Kit repository.

To run:

1. Open the repository root in the configured agent-enabled IDE or terminal coding assistant.
2. Confirm the agent can discover this repository's `skills/` directory. The selected CLI profile needs read access to the source UC model and MLflow metadata.
3. Submit:

```text
Use $air-migrate-migrate-fmt-model to generate the AIR workload configured in migrate/config.yaml.
```

4. Review the changes under `migrate/output/`, especially the manifest and `air_workload/train.yaml`. The agent must not submit AIR, training, merge, registration, or evaluation workloads.

### From Genie Code

1. Start a Genie Code session with this repository root attached as code context in a workspace that can read the configured UC model and MLflow metadata.
2. Ensure Genie Code can discover the repository `skills/` directory, then submit the following instruction:

```text
Read and follow skills/air-migrate-migrate-fmt-model/SKILL.md to generate the AIR workload configured in migrate/config.yaml. Stop after local validation; do not run AIR, training, merge, registration, or evaluation.
```

3. Review the generated manifest and workload files in the session diff before accepting or committing them.

### What the orchestrator does

The orchestrator:

1. Inspects source lineage, training behavior, data, and selected model input without downloading weights.
2. Applies `peft_only`, compares eligible recipes using inspected model-size and requested-compute evidence, and creates a generation plan.
3. Copies and customizes the selected template under `migrate/output/air_workload`, runs local checks, and stops.

It does not authenticate to AIR, mutate MLflow, submit a dry-run, train, merge, register, or evaluate.

## Generated handoff

```text
migrate/output/air_workload/
├── train.yaml
├── train.py
├── helper_utils.py
├── training_utils.py
├── merge.py              # LoRA recipes only
├── 01_runner.py
└── 02_register_uc.py
```

`train.yaml` is the AIR configuration, `01_runner.py` is the training notebook, the Python modules implement training and optional merge support, and `02_register_uc.py` is the registration notebook.

`source_model_uri` in the generated YAML always preserves the legacy UC model lineage. `model_source`, `model_name`, and `tokenizer_path` identify the actual training input. At operator runtime, Volume and system.ai inputs are staged once per AIR node into an ephemeral local cache. Hugging Face inputs are fetched by Transformers, with `HF_TOKEN` injected only when configured. Durable checkpoints, adapters, and merged models remain on UC Volumes.

## Approved templates

| Recipe | Intended use |
|---|---|
| `trl_lora` | Unquantized bf16 LoRA with DDP; selected for PEFT when one base replica fits per accelerator |
| `trl_lora_fsdp` | Unquantized bf16 LoRA with PEFT-aware FSDP; selected for PEFT when the base must be sharded |
| `trl_full_fsdp` | Full-weight TRL training with FSDP; forbidden when `peft_only: true` |

For isolated generator testing, use an empty destination:

```bash
python3 skills/air-migrate-generate-air-job/scripts/materialize_air_template.py \
  --recipe trl_lora_fsdp \
  --output-dir /tmp/air-workload-test
```

The generator refuses a nonempty destination. Validate changes locally with:

```bash
python3 -m compileall -q air_templates skills
ruff check air_templates skills
git diff --check
```

Exercise both LoRA recipes with `peft_only: true`; verify full FSDP is rejected for that setting and accepted with `peft_only: false`. Cover Volume, system.ai, gated Hugging Face, and public Hugging Face configurations. Do not use AIR dry-run as repository validation because it stages workspace files externally.

After review, the operator may run the AIR config or training notebook and then the registration notebook. See `skills/air-migrate-migrate-fmt-model/references/air-execution-registration.md`.

## Optional legacy demo

`example_setup/` can create a legacy fine-tuned source model. Its `fine_tuning.register_to` value is independent of the sample values in `migrate/config.yaml`. After running the demo, copy the exact registered catalog, schema, model name, and created version into the migration config; do not assume the checked-in examples already match.
