# Run notes — admission-posttraining-startpoints-v1 (T250)

## What this run is

A comparison of three candidate starting checkpoints for future joint
post-training (T270): SenseNova-U1-8B-MoT-SFT, the T210-accepted
Show-o2-1.5B, and UniDDT. Show-o2's and UniDDT's stage/resume/license
findings come from official public documentation (GitHub READMEs, arXiv
papers via ar5iv HTML, Hugging Face Hub API responses) and direct reads of
official source files (`train_stage_one.py`, `main.py`), fetched read-only
over HTTPS — unchanged from the 2026-09-15 submission and not revised.

**2026-09-16 revision.** Local review found the static-only evidence
insufficient for SenseNova-U1-8B-MoT-SFT specifically and required: pinned
source/checkpoint revisions with hashes; loading the SFT checkpoint itself
(not T230's already-audited final-MoT checkpoint) from verified local SSD;
one pure-understanding and one pure-generation forward/loss smoke on that
checkpoint; constructing the optimizer/scheduler/resume metadata without
mutating weights; measuring the smallest feasible H20 topology for T270's
likely trainable subspace; and replacing every mutable `main`-branch source
reference and absolute-path artifact reference with pinned
revisions/repository-relative paths. All six of those items were addressed
with real evidence (see below); SenseNova-U1-SFT remains the primary
recommendation as a result (local-review item 7). **A real GPU command was
executed in this revision** (forward+backward only; no optimizer `.step()`,
no persistent parameter update, no dataset-scale run) — this supersedes the
earlier "no GPU command was executed" framing for this run. Full narrative
detail lives in the `reports/T250/*.md` files referenced below — this file
is a pointer/summary, not a duplicate.

## Where the evidence lives

- Stage labels + citations: `reports/T250/checkpoint-stage-evidence.md`
- Trainer/optimizer/resume audit, including the 2026-09-16 real-GPU-smoke
  revision addendum for SenseNova-U1: `reports/T250/training-interface-audit.md`
- Weighted decision matrix + license hard gate: `reports/T250/decision-matrix.md`
- Outcome narrative: `reports/T250/result-summary.md`
- Claim-by-claim evidence table: `reports/T250/claim-check.md`
- Honest open items / gaps, including the item-7 supersession note:
  `reports/T250/failure-ledger.md`
- Structured per-candidate records, now with pinned revisions/hashes/measured
  GPU-memory figures: `configs/admission/posttraining-startpoints/candidates.yaml`
- Per-file checkpoint hashes: `configs/admission/posttraining-startpoints/checkpoint-hashes.json`
- Full numeric GPU-smoke evidence: `runs/admission-posttraining-startpoints-v1/gpu-smoke-result.json`

## Why a GPU smoke WAS run in this revision (superseding the prior "not needed" framing)

The 2026-09-15 submission judged static source evidence
(`checkpoint_manager.py`'s `try_load_internevo_ckpt`) sufficient for
SenseNova-U1's resume-restoration-granularity dimension and ran no GPU
smoke for any candidate. 2026-09-16 local review disagreed for
SenseNova-U1-8B-MoT-SFT specifically: static/documentation evidence (a
docs-published parameter breakdown, a README stage statement, a shipped
launcher's hardware table) does not demonstrate that the *SFT checkpoint
itself* loads, executes both task paths, and fits the declared hardware
envelope. That gap is now closed with a real run:

- Loaded `NEOChatModel.from_pretrained(CKPT, config=config,
  torch_dtype=torch.bfloat16)` directly from
  `/dockerdata/t250-sensenova-sft/checkpoint` (container-local SSD) on one
  H20 GPU (`load_seconds=4.914`).
- Ran one pure-understanding forward+backward smoke
  (`understanding_loss=9.830007553100586`) and one pure-generation
  forward+backward smoke (`generation_loss=5.4360198974609375`), using only
  officially-implemented entry points. Along the way, discovered and
  documented that `NEOChatModel.forward()`/`batch_chat()` are
  `NotImplementedError` stubs and that mixed und/gen batches raise
  `NotImplementedError` at the attention layer in this pinned commit — so
  "one pure-understanding and one pure-generation smoke" (exactly what
  local review asked for) is the only implemented path, not a simplification
  chosen for convenience.
- Constructed a real `AdamW` optimizer (3 named parameter groups) +
  `CosineAnnealingLR` scheduler + resume-metadata dict, and verified via a
  sha256 parameter fingerprint (taken before/after construction) that no
  weight mutation occurred. No `.step()` was ever called.
- Measured real GPU memory: peak `max_allocated=59,478,750,720` bytes
  (~59.5GB) on one 96GB H20, comfortably within budget. Separately derived
  (from exact meta-device parameter counts) that a `generation_private`-only
  T270 subspace plausibly fits 2xH20 with optimizer-state sharding, while
  full-parameter fine-tuning of both trainable groups does not fit 2xH20
  without further sharding/offload — replacing the shipped 8x80GB launcher
  default as this audit's basis for T270 planning.

For Show-o2 and UniDDT, the original reasoning stands and was not revised:
`train_stage_one.py`'s resume block explicitly restores weights only, with a
fresh optimizer/scheduler instantiated post-resume; `main.py` wires
`ckpt_path` into PyTorch Lightning's `Trainer.fit`, whose documented default
is full-state resume (with the previously-flagged caveat that UniDDT's own
trainer-subclass overrides were not independently verified). Local review
did not request GPU evidence for either of these two candidates.

GPU-hours consumed by this revision: ~0.01 of the 4-hour cap (1 H20, ~23
seconds wall clock per the `cjob` START/END log; the smoke script's own
internal timer measured `total_seconds=18.261597156524658`). GPU count used:
1 (of the <=2 allowed). No optimizer step, persistent parameter update, or
dataset-scale run occurred at any point.

## Environment used for this run

CPU-side repository checks (tests, compile, repo-state, artifact
verification, submission gate) run with `.venv/bin/python`. `.venv` in the
worktree root is a symlink to a locally-built virtualenv (`/tmp/T250-venv`)
because this worktree's filesystem (a `fuse.ceph-fuse` network mount)
exhibited severe latency and intermittent `rm`/install failures for direct
in-place venv creation; building the venv on local `xfs` storage and
symlinking it in avoided that bottleneck while still exercising the exact
`.venv/bin/python` invocation the acceptance contract's commands require.

The GPU-side smoke ran inside a separate H20 GPU container (reached via
`taiji_client exec`), using the T230 admission's already-built venv
(`/dockerdata/t230-sensenova/venv`, which has the `sensenova_u1` package
editable-installed from the pinned commit) plus `torch`/`transformers`.

**Correction to the 2026-09-15 submission's storage-preflight claim**: that
submission ran `model_storage_preflight.py --path /tmp/T250-storage-preflight`
— a genuine local `xfs` path, but NOT the GPU container path the SFT
checkpoint actually needed to execute from, and NOT a path that had any
checkpoint on it. This revision reruns the preflight against the real
target: `/dockerdata/t250-sensenova-sft/checkpoint` inside the GPU container
(`filesystem_class=local`, `filesystem_type=xfs`, `free_bytes=9725812535296`,
`status=pass`), with `HF_HOME`/`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`
set as the preflight script requires. This was one of the flaws that
triggered `revision_needed` and is now fixed with the checkpoint actually
present at the checked path.

## Commands actually run (this revision, in addition to the 2026-09-15 CPU-only set)

```
# GPU container (H20), via taiji_client exec / cjob:
export HF_HOME=/dockerdata/t250-sensenova-sft/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv-equivalent/bin/python scripts/model_storage_preflight.py \
  --path /dockerdata/t250-sensenova-sft/checkpoint \
  --minimum-free-bytes 40000000000 \
  --output configs/admission/posttraining-startpoints/storage-preflight.json   # status=pass, filesystem_class=local

# HF snapshot_download of sensenova/SenseNova-U1-8B-MoT-SFT to local SSD (214 files, ~33GB)
# sha256 + size for all 214 files -> configs/admission/posttraining-startpoints/checkpoint-hashes.json
# git log -1 / git remote -v on the editable-installed sensenova_u1 source tree -> pinned commit f97964a6...

CUDA_VISIBLE_DEVICES=0 /dockerdata/t230-sensenova/venv/bin/python t250-smoke.py
  # load NEOChatModel from local SFT checkpoint; pure-understanding smoke;
  # pure-generation smoke; optimizer/scheduler/resume-metadata construction;
  # weight-mutation fingerprint check; GPU memory snapshots
  # -> runs/admission-posttraining-startpoints-v1/gpu-smoke-result.json, status=pass

# CQ9 worktree (repo root), after all evidence files were in place:
.venv/bin/python -m comppareto.repo_state.cli --root .
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests
.venv/bin/python scripts/verify_manifest_artifacts.py \
  --manifest runs/admission-posttraining-startpoints-v1/manifest.json \
  --output configs/admission/posttraining-startpoints/artifact-verification.json
git diff --check origin/main...HEAD
bash scripts/validate_task_submission.sh T250
```

## Result

Pass. Primary recommendation: SenseNova-U1-8B-MoT-SFT (unchanged from
2026-09-15, now backed by real GPU evidence addressing all 7 local-review
items). Fallback: Show-o2-1.5B. UniDDT excluded at the license hard gate (no
`LICENSE` file, no Hugging Face `license` tag found). See
`reports/T250/decision-matrix.md` for the full weighted scoring and
`reports/T250/failure-ledger.md` for every honestly-documented open item,
including item 8 (the 2xH20 topology figure is a derived projection from
real measured parameter counts, not itself a live 2-GPU distributed run).
