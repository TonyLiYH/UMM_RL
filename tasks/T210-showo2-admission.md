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
allowed_paths: ["tasks/T210-showo2-admission.md", "configs/admission/showo2/", "runs/admission-showo2/", "reports/T210/", "src/comppareto/adapters/showo2/", "tests/adapters/"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-27
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
