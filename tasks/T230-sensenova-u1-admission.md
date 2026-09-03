---
id: T230
title: SenseNova-U1 native pixel/MoT admission
parent: T200
status: awaiting_review
priority: P1
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T230-sensenova-u1-admission
depends_on: [T210]
blocks: [T500]
allowed_paths: ["tasks/T230-sensenova-u1-admission.md", "configs/admission/sensenova-u1/", "runs/admission-sensenova-u1-v1/", "reports/T230/", "src/comppareto/adapters/sensenova_u1/", "tests/adapters/sensenova_u1/"]
source_revision: "217d183b30995db4ac82158259f45800e57e2eb1"
created_at: 2026-08-26
updated_at: 2026-09-03
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

Commit and push the first report before downloading large assets or starting
GPU work. Proceed only with the released U1 path, a passing storage preflight,
and the declared resource envelope.

## Required deliverables

Admission manifest, smokes, block map, report, and failure ledger.

## Artifact and provenance requirements

Record model family/version explicitly so U1 and U1.5 evidence cannot mix.

## Failure and retry rules

An unreleased pipeline is a blocker, not permission to recreate it.

## Resource envelope

- one H20 GPU by default, at most two when required by the official path;
- at most twelve H20-equivalent GPU-hours;
- admission smoke and static routed-overlap audit only;
- no U1.5 preview substitution and no recreated missing training pipeline;
- weights and caches must execute from verified local SSD.

## Automated submission gate

Before setting `awaiting_review`, run:

```bash
bash scripts/validate_task_submission.sh T230
```

## Successor opening

Acceptance contributes to T500.

## Review history

- 2026-08-26 — Planned; T210 format not yet accepted.
- 2026-09-01 — T210 accepted; T230 authorized as an independent native
  pixel/MoT admission and static-overlap stress audit.
- 2026-09-03 — Remote executor created branch `agent/T230-sensenova-u1-admission`
  from `origin/main` (`4e34878abbb03e11bd722af40788e5b0fdb87a66`) and set
  status to `running`. **Data-integrity correction**: this file's
  `source_revision` field as found on `origin/main` was
  `217d183473a14ad48852205ea3f2746301915729`, which does not resolve to any
  git object in this repository (confirmed via `git cat-file -e` across every
  fetched ref). This is the identical malformed value already found and
  corrected in T215/T220/T240's frontmatter (see T215's 2026-09-03
  review-history entry for the full investigation). The same real commit,
  `217d183b30995db4ac82158259f45800e57e2eb1` ("merge: accept Show-o2 admission
  evidence"), shares the same 7-character abbreviation but differs in the
  remaining 33 hex characters. Corrected `source_revision` to the verified
  real commit per the same user authorization (2026-09-03) covering all four
  tasks sharing this defect (T215/T220/T230/T240); this is the only
  frontmatter field changed.
- 2026-09-03 — Remote executor completed the formal admission run
  (`runs/admission-sensenova-u1-v1/`, `manifest.json` status `pass`, 15
  artifacts / 15 passed / 0 failed) and both required deliverables sets
  (`configs/admission/sensenova-u1/`, `reports/T230/`). Both task-path smokes
  (`examples/vqa/inference.py`, `examples/t2i/inference.py`) exited 0 from
  local SSD on H20-FoldUMM GPU0; parameter-block registry accounts for all
  17,552,340,992 params with `unassigned_trainable_parameters: 0`; the
  routed-overlap audit recorded one static assumption violation (upstream
  `NotImplementedError` on the mixed understanding/generation forward path,
  issue #207), explicitly distinguished from the out-of-scope A3B variant's
  learned MoE-gate mechanism. Total measured GPU wall-clock 162s (~0.045
  GPU-hours, well under the 12-hour cap). No U1.5 asset was downloaded or
  exercised at any stage. `bash scripts/validate_task_submission.sh T230`
  passes (task-tree, run-manifests, research-state, full test suite,
  compileall all green) once status is set to `awaiting_review` below.
  Setting status to `awaiting_review` for local review.
