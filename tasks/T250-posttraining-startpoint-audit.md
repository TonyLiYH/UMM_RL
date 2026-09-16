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
updated_at: 2026-09-16
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

## Local review requirements — 2026-09-16

The static audit supports SenseNova-U1-SFT as a provisional candidate but does not yet establish a reproducible starting checkpoint. Revise before acceptance:

1. Pin the exact SenseNova-U1 source and SFT checkpoint revisions; record checkpoint file hashes and measured size.
2. Load `sensenova/SenseNova-U1-8B-MoT-SFT` itself from verified local SSD. T230's final-MoT checkpoint smoke is not evidence for the SFT checkpoint.
3. Execute one pure-understanding and one pure-generation forward/loss smoke on the SFT checkpoint. No optimizer step is required.
4. Construct the official model, selected trainable groups, optimizer, scheduler, and resume metadata without mutating weights.
5. Measure the smallest feasible H20 topology for T270's intended trainable subspace. The published 8x80GB default is not evidence that the planned smoke fits the current envelope.
6. Replace mutable `main` source references with pinned revisions. Use portable repository-relative paths for Git-tracked evidence; current absolute CQ9 worktree artifact paths cannot be independently verified elsewhere.
7. Keep SenseNova-U1-SFT as primary only if these checks pass; otherwise promote Show-o2 as the executable fallback and state the limitation.

No persistent training is authorized by this revision.

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
- 2026-09-16 — Local review set `revision_needed`: the recommendation is static-only, the SenseNova source is unpinned, the SFT checkpoint is neither hashed nor loaded, and hardware feasibility for T270 is unmeasured.
- 2026-09-16 — Remote executor resumed the branch, set status to `running`, and began addressing all 7 revision items: pin `sensenova/SenseNova-U1-8B-MoT-SFT`@`846ff1352e3a4e900d064740cddfc163b115646f` on GitHub source pinned to the already-audited T230 commit `f97964a6e54b0abf92aa2db849af4e942bb2ff08`; read `modeling_qwen3.py`/`modeling_neo_chat.py` in full to design real understanding/generation forward-loss smokes (found and will document: `NEOChatModel.forward` and the mixed und/gen attention path both raise `NotImplementedError` in the released source — only `forward_und`-only and `forward_gen`-only batches are implemented, which matches exactly the "one pure-understanding and one pure-generation" smoke shape required); proceeding to connect to H20-FoldUMM, download and hash the SFT checkpoint to container-local SSD, run the two smokes plus optimizer/scheduler/resume construction without any `.step()`, measure real GPU memory footprint for the T270 trainable subspace, and regenerate all deliverables with repository-relative paths before re-requesting review.
- 2026-09-16 — Remote executor completed all 7 revision items with real evidence and re-submitted for review.
  (1) Pinned SenseNova-U1 source to commit `f97964a6e54b0abf92aa2db849af4e942bb2ff08` (2026-09-02, clean tree,
  verified via `git log -1`/`git remote -v` on the editable-installed source tree) and the
  `sensenova/SenseNova-U1-8B-MoT-SFT` checkpoint to Hugging Face revision `846ff1352e3a4e900d064740cddfc163b115646f`;
  hashed and sized all 214 checkpoint files (35,217,355,798 bytes total; sha256 per file in the new
  `configs/admission/posttraining-startpoints/checkpoint-hashes.json`), including confirming three 16-byte
  shards are benign empty-tensor shards (cross-checked against `model.safetensors.index.json`'s `weight_map`),
  not corrupted downloads.
  (2) Loaded `NEOChatModel.from_pretrained` directly from the SFT checkpoint on container-local SSD
  (`filesystem_class=local`, `filesystem_type=xfs`, real `storage-preflight.json` rerun against the actual
  checkpoint path — the 2026-09-15 preflight had checked an unrelated empty path, now corrected and noted in
  `runs/admission-posttraining-startpoints-v1/notes.md`), on one H20 GPU.
  (3) Ran one pure-understanding forward+backward smoke (`understanding_loss=9.830007553100586`, gradient
  confined to `shared_backbone`) and one pure-generation forward+backward smoke
  (`generation_loss=5.4360198974609375`, gradient confined to `generation_private`); no `.step()` was called.
  Along the way confirmed `NEOChatModel.forward()`/`batch_chat()` are `NotImplementedError` stubs and mixed
  understanding+generation batches raise `NotImplementedError` in this pinned commit, so pure-task-path batches
  are the only implemented shape, not a convenience simplification.
  (4) Constructed a real `AdamW` optimizer (3 parameter groups) + `CosineAnnealingLR` scheduler + resume-metadata
  dict over the loaded model and verified via a sha256 parameter fingerprint (identical before/after:
  `79efdf9257c805710d0ccd92f4fed56512f6fd5e05db5cbba6965a4116b646b9`) that construction did not mutate weights.
  (5) Measured real single-H20 peak memory for the load+forward+backward smoke (`max_allocated=59,478,750,720`
  bytes, ~59.5GB of 96GB) and, from exact meta-device parameter counts
  (`shared_backbone=9,348,413,952`, `generation_private=8,186,358,272`), derived that a `generation_private`-only
  T270 trainable subspace plausibly fits 2xH20 with optimizer-state sharding, while full-parameter fine-tuning of
  both trainable groups does not fit 2xH20 without further sharding/offload — recorded as a derived/hedged
  projection (not itself a live 2-GPU run) in `reports/T250/failure-ledger.md` item 8 and `claim-check.md`.
  (6) Replaced every `main`-branch source reference with the pinned commit/revision above and rewrote
  `runs/admission-posttraining-startpoints-v1/manifest.json`'s `artifacts[].canonical_uri` to portable
  repository-relative paths (e.g. `reports/T250/first-report.md`), verified by re-running
  `scripts/verify_manifest_artifacts.py` from the repository root: 11/11 artifacts pass, 0 failed.
  (7) All six checks passed, so SenseNova-U1-8B-MoT-SFT remains the primary recommendation; Show-o2-1.5B remains
  the fallback; UniDDT remains excluded at the license gate. GPU resources consumed: ~0.01 GPU-hours on 1 H20
  (of the 4-hour/2-GPU envelope; ~23s wall clock per `cjob` START/END log, `total_seconds=18.261597156524658`
  measured internally by the smoke script). No optimizer step, persistent parameter update, or dataset-scale run
  occurred. Re-ran `pytest` (201 passed — the suite grew since the 2026-09-15 entry due to the
  `revision_needed` merge sync from `origin/main`), `compileall`, `comppareto.repo_state.cli` (pass), and
  `scripts/verify_manifest_artifacts.py` (11/11 pass) fresh after all edits. Set status to `awaiting_review`.
