---
id: T210
title: Show-o2 checkpoint, training-interface, and evaluation admission
parent: T200
status: awaiting_review
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T210-showo2-admission
depends_on: []
blocks: [T215, T300]
allowed_paths: ["tasks/T210-showo2-admission.md", "configs/admission/showo2/", "runs/admission-showo2-*/", "reports/T210/", "src/comppareto/adapters/showo2/", "tests/adapters/"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-28
---

# T210: Show-o2 admission

## Research claim

Show-o2 is the first executable pilot only if its public checkpoint, shared/private blocks, and both task paths are reproducible.

## Objective

Audit official code/revision/license, map trainable blocks, and reproduce one understanding and one generation smoke/evaluation path.

## Dependencies and inputs

Official Show-o repository and public 1.5B checkpoint; no method implementation is required.

## Allowed changes

Admission adapter, tests, configs, run metadata, report, and this task file.

## Frozen protocol

Use official sources; record every external component and do not begin joint post-training.

## Execution stages

Read-only code audit; environment smoke; understanding path; generation path; parameter-block registry draft.

## Pass/fail gate

Both paths execute, checkpoint/revisions are pinned, licenses are recorded, and eligible shared/private blocks are auditable.

## First report

Before GPU execution, return repository commit, license, checkpoint identifier/size, required VRAM, commands, and missing assets.

## Required deliverables

Admission manifest, smoke outputs, block map, environment record, result summary, and failure ledger.

## Artifact and provenance requirements

Record official URL/commit, checkpoint hash, external tokenizer/VAE revisions, config hash, hardware, and output hashes.

## Failure and retry rules

Do not use unofficial fixes or a different checkpoint without local authorization.

## Automated submission gate

Before downloading or loading large model assets, generate and retain a storage
preflight using:

```bash
HF_HOME=<local-ssd-cache> \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/model_storage_preflight.py \
  --path <local-ssd-cache> \
  --minimum-free-bytes <required-bytes> \
  --output configs/admission/showo2/storage-preflight.json
```

Before setting `awaiting_review`, run:

```bash
bash scripts/validate_task_submission.sh T210
```

The machine-readable contract is `tasks/contracts/T210.acceptance.yaml`. It
requires a local-filesystem preflight, remote artifact hash verification,
schema-valid run evidence, successful MMU/T2I exit codes, and the complete
repository test suite.

## Successor opening

Accepted T210 opens T215 and contributes to T300. T300 retains its other
declared dependencies.

## Review history

- 2026-08-26 — Authorized for audit and smoke only; no training authorized.
- 2026-08-27 — Remote executor created branch `agent/T210-showo2-admission` from `origin/main`, set status to `running`, and published the first report (`reports/T210/first-report.md`) before any GPU execution.
- 2026-08-27 — Remote executor completed environment + checkpoint smoke on H20-FoldUMM, fixing an upstream unpinned `torch`/`torchvision` version clobber (`reports/T210/environment-checkpoint-smoke.md`).
- 2026-08-27 — Remote executor completed both task-path smokes (`inference_mmu.py`, `inference_t2i.py`, both exit 0 on GPU with the pinned checkpoint), fixing an upstream unpinned `wandb`/`protobuf` incompatibility, and recorded a previously-undocumented external component (`CompVis/stable-diffusion-safety-checker`) (`reports/T210/task-path-smoke.md`).
- 2026-08-27 — Remote executor published the parameter-block registry draft, enumerating `named_parameters()` on the loaded checkpoint and correcting two boundary omissions from the stage-1 provisional reading (`reports/T210/parameter-block-registry.md`, `configs/admission/showo2/parameter-block-registry.yaml`).
- 2026-08-27 — Remote executor submitted `reports/T210/{result-summary,claim-check,failure-ledger}.md`, flagging 2 unresolved license-status open items (Wan2.1 VAE, safety checker) for local-reviewer decision, and set status to `awaiting_review`. No training was started or authorized; successor opening (T215/T300) is left to local review per `reports/README.md`.
- 2026-08-28 — Local review of `ce29888` requested revision. The model assets
  and Hugging Face cache must be migrated from shared storage to GPU-container
  local SSD, hashes reverified, and cold-process/warm-process loading measured.
  The branch must also add a schema-valid admission manifest, raw hash-addressed
  smoke evidence, measured VRAM/GPU-hours, a frozen repaired environment, and
  resolution or formal containment of the two license/provenance open items.
  Full review: `reports/T210/local-review.md`.
- 2026-08-28 — Remote executor addressed all R1-R7 revisions. R1: migrated the
  four required components plus HF cache metadata to `/dockerdata/t210-showo2/`
  local SSD (~15GB, hash-verified identical to the shared-storage originals,
  which remain as provenance), reconfigured `HF_HOME`/offline-mode variables,
  and proved zero shared-storage/network fallback across all R2 logs. R2: three
  SSD-sourced reruns (`mmu_cold1`, `mmu_cold2`, `t2i_fresh1`) via a new external
  timing/memory harness (`configs/admission/showo2/timing_wrapper.py`, no
  Show-o2 source touched) measured ~8.7-9.4s model load (~150x faster than the
  prior ~26min shared-storage load), full phase timings, and bit-identical
  output hashes across the two cold understanding runs; same-process warm
  inference was assessed and documented as not attempted (no loop entry point
  in the official scripts; a custom warm-loop driver was judged too risky to
  the audited code path). R3: committed a schema-valid
  `runs/admission-showo2-2026-08-28/manifest.json` (`run_kind=formal`,
  `dirty=false`) plus a run note, referencing 14 hash/byte-addressed artifacts.
  R4: raw stdout/stderr/timing/wandb evidence for all three reruns preserved
  durably outside Git. R5: measured peak VRAM for both paths (MMU ~13.87GiB
  alloc/~38.48GiB reserved; T2I ~12.36GiB alloc/~12.69GiB reserved, replacing
  the prior estimate) and reported GPU wall-clock/footprint in
  `reports/T210/r2-r5-ssd-rerun.md`. R6: resolved the Wan2.1 VAE source revision
  and Apache-2.0 license via a live HF Hub query; the safety-checker's license
  remains genuinely unspecified upstream and is now formally constrained as an
  optional, display-only dependency with a documented safety-checker-free
  evaluation path (`reports/T210/failure-ledger.md`). R7: froze the repaired
  environment's required pins in
  `configs/admission/showo2/environment-lock.md`. Repository validator passes
  (`run_manifests=pass manifests=2`); 29/30 tests pass — the one failure is a
  pre-existing, out-of-scope `tests/repo_state/test_cli.py` assertion that
  hardcodes a manifest count of 1, now stale because a second formal manifest
  legitimately exists; flagged in
  `runs/admission-showo2-2026-08-28/notes.md` rather than edited, since
  `tests/repo_state/` is outside this task's `allowed_paths`. Status set to
  `awaiting_review`.
- 2026-08-28 — Second local review of `b05b439` confirmed the SSD loading
  improvement and narrowed the remaining work to evidence consistency:
  merge current `origin/main`, reconcile stale summaries, record provenance
  and SSD execution URIs separately, qualify or strengthen no-fallback
  evidence, complete resource fields, and remotely reverify all external
  artifacts. Complete R8–R13 in `reports/T210/local-review.md`.
