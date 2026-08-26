---
id: T230
title: SenseNova-U1 native pixel/MoT admission
parent: T200
status: planned
priority: P1
owner: unassigned
reviewer: local-research-agent
branch: agent/T230-sensenova-u1-admission
depends_on: [T210]
blocks: [T500]
allowed_paths: ["tasks/T230-sensenova-u1-admission.md", "configs/admission/sensenova-u1/", "runs/admission-sensenova-u1/", "reports/T230/", "src/comppareto/adapters/sensenova_u1/", "tests/adapters/"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T230: SenseNova-U1 admission

## Research claim

SenseNova-U1 tests whether compensation remains useful in a more native pixel-space and mixture-of-Transformers architecture.

## Objective

Audit the released U1 full-parameter path; U1.5 remains excluded until its announced training pipeline is public.

## Dependencies and inputs

Accepted T210 format and official SenseNova-U1 sources.

## Allowed changes

Admission adapter, tests, configs, run metadata, report, and this task file.

## Frozen protocol

Do not substitute U1.5 preview or unreleased training code for the admitted U1 path.

## Execution stages

Source/license audit, understanding smoke, generation smoke, trainable-block map, report.

## Pass/fail gate

Released U1 code exposes a reproducible full-parameter path and both task evaluations.

## First report

Return exact official release, code/checkpoint revisions, missing components, and GPU estimate.

## Required deliverables

Admission manifest, smokes, block map, report, and failure ledger.

## Artifact and provenance requirements

Record model family/version explicitly so U1 and U1.5 evidence cannot mix.

## Failure and retry rules

An unreleased pipeline is a blocker, not permission to recreate it.

## Successor opening

Acceptance contributes to T500.

## Review history

- 2026-08-26 — Planned; T210 format not yet accepted.

