---
id: T250
title: Post-training starting-checkpoint selection audit
parent: T200
status: awaiting_review
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T250-posttraining-startpoint-audit
depends_on: [T210]
blocks: [T270, T300, T400]
allowed_paths: ["tasks/T250-posttraining-startpoint-audit.md", "configs/admission/posttraining-startpoints/", "runs/admission-posttraining-startpoints-v1/", "reports/T250/"]
source_revision: "45c54ba403a2b5c95985a49206243449436717c8"
created_at: 2026-09-15
updated_at: 2026-09-15
---

# T250: Post-training starting-checkpoint selection audit

## Research claim

The primary experiment should start from a reproducible checkpoint that already
has usable understanding and generation capabilities but has not undergone the
target joint preference/RL or compensation-aware post-training stage.

## Objective

Compare SenseNova-U1-8B-MoT-SFT, the admitted Show-o2-1.5B checkpoint, and
UniDDT as candidate starting points. Determine each checkpoint's actual
training stage, availability of an earlier usable checkpoint, official
training/resume support, shared/private/routed parameter ownership, and
feasibility of interleaved understanding and generation batches.

## Decision rule

Prefer the earliest reproducible checkpoint that:

- executes both target task paths;
- exposes official trainable parameters and checkpoint-resume support;
- precedes the target preference/RL stage;
- has adequate headroom for supervised or preference post-training;
- has auditable shared/private/routed ownership;
- permits the intended research use under recorded licenses;
- fits the declared H20 resource envelope.

"Never post-trained" is not a hard requirement. A checkpoint unable to perform
both tasks is not useful merely because it is earlier. The desired operating
point is normally SFT-capable and pre-preference/RL.

## Required audit dimensions

For each candidate record:

- official repository, paper/model card, source revision, checkpoint hash, and
  license;
- evidence-backed stage label: base/pretraining, SFT, preference/RL, distilled,
  merged, or unknown;
- official trainer, optimizer, resume, configuration, and data interfaces;
- whether resume restores weights, optimizer, scheduler, scaler, and counters;
- understanding and generation path availability;
- shared/private/routed parameter classification;
- minimum and recommended GPU memory;
- sequential-task-batch support at one frozen shared version;
- missing code, data, tokenizer, VAE, reward, or decoder dependencies;
- expected headroom and contamination risks.

## Execution stages

1. Commit and push a source/checkpoint inventory before GPU work.
2. Audit primary official documentation and source.
3. Verify stage labels; do not infer SFT solely from filenames.
4. Inspect official training and resume paths statically.
5. Reuse accepted admission evidence only when still applicable.
6. Run minimal load/resume-interface smokes only where static evidence is
   insufficient.
7. Produce a weighted decision matrix with one primary and one fallback.

## Pass/fail gate

The task passes when every candidate has an evidence-backed or explicitly
unknown stage label, inference admission is separated from training readiness,
and at least one candidate is classified as a reproducible starting point—or
the report demonstrates that none is currently admissible. No persistent
parameter update is allowed.

## First report

Before downloading new large assets or running GPU commands, report candidate
IDs, expected stage evidence, resume-interface inspection, reusable assets,
downloads, memory, runtime, and exact commands.

## Required deliverables

- `reports/T250/first-report.md`
- `reports/T250/checkpoint-stage-evidence.md`
- `reports/T250/training-interface-audit.md`
- `reports/T250/decision-matrix.md`
- `reports/T250/result-summary.md`
- `reports/T250/claim-check.md`
- `reports/T250/failure-ledger.md`
- `configs/admission/posttraining-startpoints/candidates.yaml`
- `configs/admission/posttraining-startpoints/storage-preflight.json`
- `configs/admission/posttraining-startpoints/artifact-verification.json`
- `runs/admission-posttraining-startpoints-v1/manifest.json`
- `runs/admission-posttraining-startpoints-v1/metrics.json`
- `runs/admission-posttraining-startpoints-v1/notes.md`

## Resource envelope

- CPU/source audit first;
- at most two H20 GPUs;
- at most four H20-equivalent GPU-hours;
- no optimizer step, persistent update, or dataset-scale run;
- checkpoints execute from verified local SSD.

## Automated submission gate

```bash
bash scripts/validate_task_submission.sh T250
```

## Successor opening

Accepted T250 selects the candidate for T270. It does not authorize training.

## Review history

- 2026-09-15 — Remote executor confirmed the existing worktree/branch
  (`agent/T250-posttraining-startpoint-audit`, base `d260b5a`, clean tree),
  verified `source_revision` as a real ancestor, set status to `running`, and
  began the CPU/source documentation audit (no GPU work, no large downloads)
  before publishing the first report.
- 2026-09-15 — Remote executor completed the CPU/source audit of all three
  candidates (SenseNova-U1-8B-MoT-SFT, Show-o2-1.5B, UniDDT), published all
  13 required deliverables, ran `pytest` (156 passed), `compileall`,
  `comppareto.repo_state.cli` (pass), and
  `scripts/verify_manifest_artifacts.py` (9/9 artifacts verified), committed
  and pushed. Primary recommendation: SenseNova-U1-8B-MoT-SFT. Fallback:
  Show-o2-1.5B. UniDDT excluded at the license hard gate (no LICENSE file,
  no Hugging Face license tag found). 0 GPU-hours consumed (of 4 allowed).
  Set status to `awaiting_review` for local review.

