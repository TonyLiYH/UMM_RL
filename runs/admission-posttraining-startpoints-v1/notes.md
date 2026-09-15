# Run notes — admission-posttraining-startpoints-v1 (T250)

## What this run is

A CPU/source-only audit comparing three candidate starting checkpoints for
future joint post-training (T270): SenseNova-U1-8B-MoT-SFT, the T210-accepted
Show-o2-1.5B, and UniDDT. No GPU command was executed, no model weights were
downloaded, and no persistent parameter update occurred. All evidence for
stage labels and training/resume interfaces comes from official public
documentation (GitHub READMEs, arXiv papers via ar5iv HTML, Hugging Face Hub
API responses) and direct reads of official source files
(`checkpoint_manager.py`, `train_stage_one.py`, `main.py`), fetched read-only
over HTTPS. Full narrative detail lives in the `reports/T250/*.md` files
referenced below — this file is a pointer/summary, not a duplicate.

## Where the evidence lives

- Stage labels + citations: `reports/T250/checkpoint-stage-evidence.md`
- Trainer/optimizer/resume static audit: `reports/T250/training-interface-audit.md`
- Weighted decision matrix + license hard gate: `reports/T250/decision-matrix.md`
- Outcome narrative: `reports/T250/result-summary.md`
- Claim-by-claim evidence table: `reports/T250/claim-check.md`
- Honest open items / gaps: `reports/T250/failure-ledger.md`
- Structured per-candidate records: `configs/admission/posttraining-startpoints/candidates.yaml`

## Why no GPU smoke was run

Execution stage 6 of the task file permits a minimal load/resume-interface
smoke "only where static evidence is insufficient." For all three candidates,
the resume-restoration-granularity question (the dimension most plausibly
needing a live run) was resolved by reading the actual training/checkpoint
source code directly:

- SenseNova-U1: `sensenovalm/checkpoint/checkpoint_manager.py`'s
  `try_load_internevo_ckpt` explicitly restores model + optimizer + scheduler
  + step/dataloader counters.
- Show-o2: `train_stage_one.py`'s resume block explicitly restores weights
  only, with a fresh optimizer/scheduler instantiated post-resume.
- UniDDT: `main.py` wires `ckpt_path` into PyTorch Lightning's
  `Trainer.fit`, whose documented default is full-state resume (with an
  explicitly flagged caveat that UniDDT's own trainer-subclass overrides were
  not independently verified).

Given this, a GPU smoke would have added confirmation but no new information
that the source code did not already provide, and would have consumed part
of the 4-GPU-hour budget for no incremental audit value. GPU-hours consumed:
0 (of 4 allowed).

## Environment used for this run

CPU-only. Python 3.11.6, `comppareto` package installed editable from this
worktree. `.venv` in the worktree root is a symlink to a locally-built
virtualenv (`/tmp/T250-venv`) because this worktree's filesystem (a
`fuse.ceph-fuse` network mount) exhibited severe latency and intermittent
`rm`/install failures for direct in-place venv creation; building the venv on
local `xfs` storage and symlinking it in avoided that bottleneck while still
exercising the exact `.venv/bin/python` invocation the acceptance contract's
commands require. The storage preflight itself (see
`configs/admission/posttraining-startpoints/storage-preflight.json`) was run
against `/tmp/T250-storage-preflight`, a genuine local `xfs` filesystem,
correctly classified `filesystem_class=local`, status `pass`.

## Commands actually run

```
.venv/bin/python -m comppareto.repo_state.cli --root .        # task_tree=pass tasks=32; run_manifests=pass; research_state=pass
.venv/bin/python -m pytest -q                                  # 156 passed
.venv/bin/python -m compileall -q src tests                     # pass
.venv/bin/python scripts/model_storage_preflight.py --path /tmp/T250-storage-preflight --minimum-free-bytes 5000000000 --output configs/admission/posttraining-startpoints/storage-preflight.json   # status=pass
.venv/bin/python scripts/verify_manifest_artifacts.py --manifest runs/admission-posttraining-startpoints-v1/manifest.json --output configs/admission/posttraining-startpoints/artifact-verification.json   # checked=9 passed=9 failed=0
bash scripts/validate_task_submission.sh T250
```

## Result

Pass. Primary recommendation: SenseNova-U1-8B-MoT-SFT. Fallback:
Show-o2-1.5B. UniDDT excluded at the license hard gate (no `LICENSE` file, no
Hugging Face `license` tag found). See `reports/T250/decision-matrix.md` for
the full weighted scoring and `reports/T250/failure-ledger.md` for every
honestly-documented open item.
