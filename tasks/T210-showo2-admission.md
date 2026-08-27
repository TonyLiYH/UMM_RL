---
id: T210
title: Show-o2 checkpoint, training-interface, and evaluation admission
parent: T200
status: running
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T210-showo2-admission
depends_on: []
blocks: [T215, T300]
allowed_paths: ["tasks/T210-showo2-admission.md", "configs/admission/showo2/", "runs/admission-showo2/", "reports/T210/", "src/comppareto/adapters/showo2/", "tests/adapters/"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
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
