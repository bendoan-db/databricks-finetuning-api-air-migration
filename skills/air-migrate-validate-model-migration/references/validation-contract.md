# Retired validation contract

The current migration does not perform live model validation. It ends after generating and locally checking `migrate/output/air_workload`; training, registration, evaluation, and migration-completion verdicts are operator-owned.

Record local results under the manifest's `generation` section:

```yaml
generation:
  status: current
  output_path: migrate/output/air_workload
  template_path: air_templates/trl_full_fsdp
  model_source: hugging_face
  files:
    - train.yaml
    - train.py
    - helper_utils.py
    - training_utils.py
    - 01_runner.py
    - 02_register_uc.py
  customized_fields: []
  local_validations:
    - check: python_compile
      verdict: pass
    - check: yaml_configuration
      verdict: pass
    - check: source_precedence
      verdict: pass
    - check: compute_consistency
      verdict: pass
    - check: template_provenance
      verdict: pass
  handoff_ready: true
```

Require these local checks:

- Every recipe-specific file exists and traces to the selected template.
- Python compiles and `train.py` remains a thin entry point.
- Generated compute, torchrun process count, and notebook GPU settings agree.
- Legacy `source_model_uri`, selected model source, experiment, and target match configuration.
- Volume and system.ai sources contain no HF secret.
- Gated Hugging Face uses exactly the configured `<scope>/<key>` reference; public Hugging Face contains no secret.
- Data and durable output paths use intended UC Volumes and no output points into the ephemeral input cache.

Do not add execution, registration, evaluation, or `migration_complete` evidence. Do not invoke AIR dry-run or any external workload while producing this local handoff verdict.
