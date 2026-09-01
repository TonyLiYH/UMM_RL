---
id: T220
title: UniDDT deep-sharing admission
parent: T200
status: ready
priority: P1
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T220-uniddt-admission
depends_on: [T210]
blocks: [T500]
allowed_paths: ["tasks/T220-uniddt-admission.md", "configs/admission/uniddt/", "runs/admission-uniddt-v1/", "reports/T220/", "src/comppareto/adapters/uniddt/", "tests/adapters/uniddt/"]
source_revision: "217d183473a14ad48852205ea3f2746301915729"
created_at: 2026-08-26
updated_at: 2026-09-01
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

Commit and push the first report before downloading large assets or starting
GPU work. Proceed only if official sources are available, storage preflight
passes, and the plan stays within the resource envelope.

## Required deliverables

Admission manifest, block map, smokes, report, and failure ledger.

## Artifact and provenance requirements

Use the common admission fields and external component revisions.

## Failure and retry rules

No alternate fork or reimplemented model without local authorization.

## Resource envelope

- one H20 GPU by default, at most two when the official path requires it;
- at most ten H20-equivalent GPU-hours;
- admission smoke only; no fine-tuning or method comparison;
- weights and caches must execute from verified local SSD.

## Automated submission gate

Before setting `awaiting_review`, run:

```bash
bash scripts/validate_task_submission.sh T220
```

## Successor opening

Acceptance contributes to T500.

## Review history

- 2026-08-26 — Planned; T210 format not yet accepted.
- 2026-09-01 — T210 accepted; T220 authorized as an independent deep-sharing
  admission task.
