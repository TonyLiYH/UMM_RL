---
id: T200
title: Public-model admission programme
parent: T000
status: running
priority: P0
owner: local-research-agent
reviewer: user
branch: feature/model-admission
depends_on: []
blocks: [T300, T500]
allowed_paths: ["tasks/T2*.md", "docs/surveys/", "configs/admission/", "runs/admission-*/", "reports/T200/"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T200: Public-model admission

## Research claim

Model comparisons are credible only when public checkpoints, trainable blocks, evaluation paths, licenses, and missing assets are audited before method results are visible.

## Objective

Admit Show-o2 first, then UniDDT, SenseNova-U1, and UniAR under a common evidence contract.

## Dependencies and inputs

Official repositories, checkpoints, papers, licenses, and the model admission rules in `docs/plans/research-plan.md`.

## Allowed changes

Admission tasks, read-only model adapters, smoke configurations, and admission reports.

## Frozen protocol

Admission cannot depend on CompPareto performance; failed models remain in the record and follow the preregistered replacement order.

## Execution stages

Run T210; use its format for T220–T240; publish a parent comparison table.

## Pass/fail gate

At least Show-o2 has an accepted checkpoint/evaluation/training-interface audit before T300.

## First report

List official revisions, licenses, checkpoint sizes, required external components, and expected smoke resources.

## Required deliverables

One admission manifest and report per model plus a parent-level availability matrix.

## Artifact and provenance requirements

Checkpoint and repository revisions, hashes, commands, environment, and smoke outputs are mandatory.

## Failure and retry rules

Do not substitute unofficial forks without local authorization.

## Successor opening

Accepted T210 plus T100 unblocks T300; other accepted admissions feed T500.

## Review history

- 2026-08-26 — Local planner opened the model-admission programme.

