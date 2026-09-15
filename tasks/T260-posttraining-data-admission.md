---
id: T260
title: Joint post-training dataset admission and frozen manifests
parent: T000
status: ready
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T260-posttraining-data-admission
depends_on: []
blocks: [T270, T300, T400]
allowed_paths: ["tasks/T260-posttraining-data-admission.md", "configs/data/posttraining-v1/", "runs/data-admission-posttraining-v1/", "reports/T260/", "src/comppareto/data/", "tests/data/"]
source_revision: "45c54ba403a2b5c95985a49206243449436717c8"
created_at: 2026-09-15
updated_at: 2026-09-15
---

# T260: Joint post-training dataset admission and frozen manifests

## Research claim

The method requires enough data for actual post-training while retaining a
paired semantic subset that isolates optimization geometry from task
distribution differences.

## Objective

Audit and prepare deterministic manifests for paired image-caption
understanding/generation, visual instruction understanding, diverse
text-to-image generation, and independent diagnostic/evaluation splits.

Initial candidates are COCO captions/images, LLaVA-Instruct-150K, and a
license-compatible deterministic JourneyDB subset. Propose alternatives when
access, license, provenance, or storage makes a candidate inadmissible.

## Dataset roles

- **D1 paired semantic core:** use the same image-caption universe for
  image-to-text and text-to-image.
- **D2 understanding extension:** image-question/instruction-answer records.
- **D3 generation extension:** diverse prompt-image pairs.
- **D4 diagnostic/meta:** image/group-disjoint records for response adaptation
  and evaluation.

## Required audit dimensions

For every source record:

- canonical source and revision/date;
- terms for images, annotations, prompts, and generated images;
- redistribution restrictions;
- download method and size;
- schema, media availability, and corrupt-record policy;
- counts before and after filtering;
- safety and personal-data considerations;
- decontamination against declared evaluations;
- split unit and near-duplicate policy;
- mapping into both task formats;
- expected tokens, images, and storage.

## Frozen pilot manifests

Produce deterministic manifests without committing large media:

1. `diagnostic`: 512--2,048 paired examples where available;
2. `pilot_train`: at least 100,000 usable task records across D1--D3;
3. `pilot_validation`: group-disjoint validation records;
4. `pilot_meta`: disjoint adaptation/meta records;
5. `evaluation_only`: held-out benchmark references never used for training.

The executor must not choose examples after observing model gradients or
method outcomes.

## Sampling and accounting proposal

Propose before model outcomes are available:

- initial D1/D2/D3 mixture;
- task batch construction;
- image/token/optimizer-step accounting;
- paired versus independent-task controls;
- equal-data and equal-compute comparisons;
- failed/corrupt example ledger policy.

## Execution stages

1. Commit and push a source/license/storage first report.
2. Resolve official metadata and terms.
3. Implement deterministic manifest builders and stable sample IDs.
4. Materialize metadata and a small verification sample only.
5. Validate schema, availability, split disjointness, and duplicate isolation.
6. Produce storage/download plans for the pilot subset.
7. Emit admission decisions and a primary/fallback mixture.

## Pass/fail gate

Every admitted source must have traceable provenance and recorded terms; all
splits must be group-disjoint; the paired core must map the same semantic
sample in both task directions; evaluation-only records must never enter
training; and the pilot must meet 100,000 usable task records or provide an
evidence-backed reduced target. Unknown or incompatible terms block the
affected source.

## First report

Before downloading more than 20 GB, report sources, terms, declared sizes,
pilot storage, manifest schema, stable IDs, split/decontamination procedure,
commands, and storage/network requirements.

## Required deliverables

- `reports/T260/first-report.md`
- `reports/T260/source-license-audit.md`
- `reports/T260/split-and-decontamination.md`
- `reports/T260/mixture-and-accounting.md`
- `reports/T260/result-summary.md`
- `reports/T260/claim-check.md`
- `reports/T260/failure-ledger.md`
- source and split manifests under `configs/data/posttraining-v1/`
- manifest builders and tests under the allowed source/test paths
- `runs/data-admission-posttraining-v1/manifest.json`
- `runs/data-admission-posttraining-v1/metrics.json`
- `runs/data-admission-posttraining-v1/notes.md`

## Resource envelope

- CPU/network/storage task; zero GPU hours;
- no download above 20 GB before the first report;
- no complete multi-terabyte download;
- no model inference, gradients, or training.

## Automated submission gate

```bash
bash scripts/validate_task_submission.sh T260
```

## Successor opening

Accepted T260 supplies frozen data inputs for T270 and later D0/E1 tasks.

