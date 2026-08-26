---
id: T220
title: UniDDT deep-sharing admission
parent: T200
status: planned
priority: P1
owner: unassigned
reviewer: local-research-agent
branch: agent/T220-uniddt-admission
depends_on: [T210]
blocks: [T500]
allowed_paths: ["tasks/T220-uniddt-admission.md", "configs/admission/uniddt/", "runs/admission-uniddt/", "reports/T220/", "src/comppareto/adapters/uniddt/", "tests/adapters/"]
source_revision: "dab902f90dedf500751ae852ceaeda5e1012f6ff"
created_at: 2026-08-26
updated_at: 2026-08-26
---

# T220: UniDDT admission

## Research claim

UniDDT provides a deep shared semantic path with a private diffusion decoder suitable for cross-architecture validation.

## Objective

Audit and reproduce the public UniDDT understanding and generation paths under the common admission contract.

## Dependencies and inputs

Accepted T210 format and official UniDDT sources.

## Allowed changes

Admission adapter, tests, configs, run metadata, report, and this task file.

## Frozen protocol

Use the official architecture/checkpoint and record all FLUX/Qwen external dependencies.

## Execution stages

Source audit, environment smoke, dual-path reproduction, block map, admission report.

## Pass/fail gate

Both paths are reproducible and the NoisyViT/LLM/decoder split is programmatically exposed.

## First report

Return official revisions, checkpoint size/hash plan, dependencies, commands, and expected GPU resources.

## Required deliverables

Admission manifest, block map, smokes, report, and failure ledger.

## Artifact and provenance requirements

Use the common admission fields and external component revisions.

## Failure and retry rules

No alternate fork or reimplemented model without local authorization.

## Successor opening

Acceptance contributes to T500.

## Review history

- 2026-08-26 — Planned; T210 format not yet accepted.

