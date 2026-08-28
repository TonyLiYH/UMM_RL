# Task acceptance contracts

Files in this directory encode the machine-checkable conditions that a remote
task must satisfy before it can be submitted as `awaiting_review`.

Each `<TASK_ID>.acceptance.yaml` may declare:

- the authoritative base reference that must be an ancestor of the task branch;
- required files or glob patterns;
- commands that must return zero;
- JSON metrics and comparison thresholds;
- phrases forbidden when a trigger condition holds.

Run:

```bash
bash scripts/validate_task_submission.sh T155
```

from a clean committed task branch after setting the task state to
`awaiting_review`. A passing command is necessary for submission but does not
replace local scientific review.

Model tasks should also generate a storage preflight before expensive
execution:

```bash
HF_HOME=/local/ssd/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/model_storage_preflight.py \
  --path /local/ssd/hf-cache \
  --minimum-free-bytes 50000000000 \
  --output configs/admission/<model>/storage-preflight.json
```

External artifacts referenced by a run manifest can be checked on the machine
that owns them with:

```bash
.venv/bin/python scripts/verify_manifest_artifacts.py \
  --manifest runs/<run-id>/manifest.json \
  --output configs/<scope>/artifact-verification.json
```
