---
id: T240
title: UniAR homogeneous-objective boundary-control admission
parent: T200
status: ready
priority: P1
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T240-uniar-admission
depends_on: [T210]
blocks: [T500]
allowed_paths: ["tasks/T240-uniar-admission.md", "configs/admission/uniar/", "runs/admission-uniar-v1/", "reports/T240/", "src/comppareto/adapters/uniar/", "tests/adapters/uniar/"]
source_revision: "217d183473a14ad48852205ea3f2746301915729"
created_at: 2026-08-26
updated_at: 2026-09-01
---

# T240: UniAR admission

## Research claim

UniAR is a boundary control where understanding and visual-token generation share an autoregressive objective, so heterogeneity-driven gains may shrink.

## Objective

Audit public AR/GRPO paths and the unreleased visual-decoder-training limitation under the common admission contract.

## Dependencies and inputs

Accepted T210 format and official UniAR sources.

## Allowed changes

Admission adapter, tests, configs, run metadata, report, and this task file.

## Frozen protocol

Do not claim visual-decoder training support that the official repository does not release.

## Execution stages

Source audit, understanding/generation smoke, AR trainable-block map, missing-code report.

## Pass/fail gate

The admitted scope is executable and sufficient for the declared boundary-control comparison.

## First report

Return official revisions, available training entry points, unavailable components, and resource estimate.

Commit and push the first report before downloading large assets or starting
GPU work. Proceed only with the officially released scope, a passing storage
preflight, and the declared resource envelope.

## Required deliverables

Admission manifest, smokes, scope map, report, and failure ledger.

## Artifact and provenance requirements

Record the visual tokenizer, decoder, and AR checkpoint revisions separately.

## Failure and retry rules

Missing official decoder training remains a documented limitation.

## Resource envelope

- one H20 GPU by default;
- at most eight H20-equivalent GPU-hours;
- admission smoke and boundary-control scope audit only;
- do not recreate or claim unreleased decoder training;
- weights and caches must execute from verified local SSD.

## Automated submission gate

Before setting `awaiting_review`, run:

```bash
bash scripts/validate_task_submission.sh T240
```

## Successor opening

Acceptance contributes to T500.

## Review history

- 2026-08-26 — Planned; T210 format not yet accepted.
- 2026-09-01 — T210 accepted; T240 authorized as an independent
  homogeneous-objective boundary-control admission task.
