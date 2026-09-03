---
id: T240
title: UniAR homogeneous-objective boundary-control admission
parent: T200
status: awaiting_review
priority: P1
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T240-uniar-admission
depends_on: [T210]
blocks: [T500]
allowed_paths: ["tasks/T240-uniar-admission.md", "configs/admission/uniar/", "runs/admission-uniar-v1/", "reports/T240/", "src/comppareto/adapters/uniar/", "tests/adapters/uniar/"]
source_revision: "217d183b30995db4ac82158259f45800e57e2eb1"
created_at: 2026-08-26
updated_at: 2026-09-03
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
- 2026-09-03 — Remote executor created branch `agent/T240-uniar-admission`
  from `origin/main` (`4e34878abbb03e11bd722af40788e5b0fdb87a66`) and set
  status to `running`. **Data-integrity correction**: this file's
  `source_revision` field as found on `origin/main` was
  `217d183473a14ad48852205ea3f2746301915729`, which does not resolve to any
  git object in this repository (confirmed via `git cat-file -e` across every
  fetched ref). This is the identical malformed value already found and
  corrected in T215/T220's frontmatter (see T215's 2026-09-03 review-history
  entry for the full investigation). The same real commit,
  `217d183b30995db4ac82158259f45800e57e2eb1` ("merge: accept Show-o2 admission
  evidence"), shares the same 7-character abbreviation but differs in the
  remaining 33 hex characters. Corrected `source_revision` to the verified
  real commit per the same user authorization (2026-09-03) covering all four
  tasks sharing this defect (T215/T220/T230/T240); this is the only
  frontmatter field changed.
- 2026-09-03 — Remote executor completed the formal admission run
  `runs/admission-uniar-v1/` (`manifest.json` status `pass`, 28/28 artifacts
  reverified, 0 failed) after: storage preflight (`status: pass`,
  `filesystem_class: local`); a fresh H20-FoldUMM venv build with four
  documented upstream environment defects fixed (torch silently downgraded by
  transitive installs x2, `flash-attn` ABI staleness after each torch
  reinstall, a broken system `xformers`/`triton` leak via
  `--system-site-packages`, and mixed CUDA12/CUDA13 `nvidia-*` packages
  breaking cuDNN init); a ~43.7GiB checkpoint download to local SSD with
  redundant `sd3_pipeline` fp16-duplicate shards excluded and every retained
  file sha256-verified; both official smoke entry points
  (`inference/chat.py`, `inference/generate.py`) run verbatim to completion
  (`exit_code=0` each), the generation smoke producing a real, hash-verified
  PNG; and a read-only AR trainable-block enumeration
  (`configs/admission/uniar/parameter-block-registry.yaml`) showing
  `shared_llm_backbone`/`understanding_private`/`generation_private` sum
  exactly to the checkpoint's 9,627,074,032 total parameters. The
  distinctive finding this task exists to establish — that visual-decoder
  (SD3 pixel-decoder) training code is confirmed absent from the official
  repository (its own sole README TODO item) — is documented via
  `scope.unreleased_decoder_training_recorded=true` in
  `runs/admission-uniar-v1/metrics.json`, with no attempt made to
  reimplement, patch around, or claim support for it, per the frozen
  protocol. Total GPU usage: ~0.69 GPU-hours on 1x H20, within the 8-hour
  cap. `bash scripts/validate_task_submission.sh T240` passes every check
  (task tree, run manifests, research state, full pytest suite, compile,
  metrics assertions, whitespace); status set to `awaiting_review` for
  local-reviewer decision — remote executor does not self-accept.
