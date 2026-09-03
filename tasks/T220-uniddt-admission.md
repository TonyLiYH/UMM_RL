---
id: T220
title: UniDDT deep-sharing admission
parent: T200
status: awaiting_review
priority: P1
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T220-uniddt-admission
depends_on: [T210]
blocks: [T500]
allowed_paths: ["tasks/T220-uniddt-admission.md", "configs/admission/uniddt/", "runs/admission-uniddt-v1/", "reports/T220/", "src/comppareto/adapters/uniddt/", "tests/adapters/uniddt/"]
source_revision: "217d183b30995db4ac82158259f45800e57e2eb1"
created_at: 2026-08-26
updated_at: 2026-09-03
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
- 2026-09-03 — Remote executor created branch `agent/T220-uniddt-admission`
  from `origin/main` (`4e34878abbb03e11bd722af40788e5b0fdb87a66`) and set
  status to `running`. **Data-integrity correction**: this file's
  `source_revision` field as found on `origin/main` was
  `217d183473a14ad48852205ea3f2746301915729`, which does not resolve to any
  git object in this repository (confirmed via `git cat-file -e` across every
  fetched ref). This is the identical malformed value already found and
  corrected in T215's frontmatter (see that task's 2026-09-03 review-history
  entry for the full investigation). The same real commit,
  `217d183b30995db4ac82158259f45800e57e2eb1` ("merge: accept Show-o2 admission
  evidence"), shares the same 7-character abbreviation but differs in the
  remaining 33 hex characters. Corrected `source_revision` to the verified
  real commit per the same user authorization (2026-09-03) covering all four
  tasks sharing this defect (T215/T220/T230/T240); this is the only
  frontmatter field changed.
- 2026-09-03 — Remote executor completed stages 2-5: built a dedicated venv
  from the official `requirements.txt` (one `einops` gap-fill; see
  `configs/admission/uniddt/environment-lock.md`); downloaded and hash-verified
  `vlm_uniddt_512.ckpt` plus the FLUX VAE and Qwen3-VL-4B-Instruct
  tokenizer/config, canonicalized to CQ7, remotely reverified
  (`configs/admission/uniddt/artifact-verification.json`, 9/9 pass, 0 failed);
  ran a dual-path smoke via an external harness driving `app_uniddt.py`'s own
  `Pipeline` class (understanding `exit_code: 0`, generation `exit_code: 0`,
  134.27s total GPU wall-clock, ≈0.0373 GPU-hours, well inside the 10-hour
  envelope); enumerated the checkpoint-loaded `DDT2` module's
  `named_parameters()` into a weight-level shared/frozen-I/O/generation-private
  block registry (`configs/admission/uniddt/parameter-block-registry.yaml`,
  `unassigned_trainable_parameters: 0`). Published
  `reports/T220/{result-summary.md,claim-check.md,failure-ledger.md}` and the
  formal run record `runs/admission-uniddt-v1/{manifest.json,metrics.json,notes.md}`
  (`status: pass`). Three license open items flagged in `first-report.md`
  (no `MCG-NJU/UniDDT` repository license, no checkpoint license, ambiguous
  FLUX VAE license) were **not** resolved further this stage and remain
  explicit, recorded, non-blocking open items for local-reviewer decision
  before any onward use of this admission's evidence beyond
  audit/reproduction — see `reports/T220/failure-ledger.md`. Ran
  `bash scripts/validate_task_submission.sh T220` (repository-state,
  full-tests [156 passed], compile, whitespace all pass) and set status to
  `awaiting_review`.
