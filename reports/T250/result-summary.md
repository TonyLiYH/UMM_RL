# T250 — Result summary

## Outcome

**Pass.** Every candidate has an evidence-backed or explicitly-hedged stage
label; inference admission (Show-o2, via T210) is explicitly separated from
training readiness (independently audited here); at least one candidate
(in fact two) is classified as a reproducible starting point. No persistent
parameter update was performed — 0 optimizer steps, 0 dataset-scale runs.
The 2026-09-16 revision executed a real (forward+backward only, no
`.step()`) GPU smoke on the pinned SenseNova-U1-8B-MoT-SFT checkpoint,
consuming **~0.01 GPU-hours on 1 H20** against the 4-GPU-hour/2-GPU cap.

## Selection

- **Primary: SenseNova-U1-8B-MoT-SFT.** Evidence-backed SFT stage label
  (official README states explicitly that SFT precedes Multi-Expert RL/OPD;
  arXiv 2605.12500 names Stage 5 as explicit Flow-GRPO reinforcement learning
  with named reward models). Full-state resume (model + optimizer + scheduler
  + step/dataloader counters) confirmed by direct code inspection of
  `try_load_internevo_ckpt`. Exact, published, reproducible shared/private
  parameter breakdown (1.245B shared / 8.121B understanding / 8.186B
  generation of 17.552B total), independently reconciled against a
  zero-cost meta-device parameter count in this revision. Apache-2.0
  (+ attributed MIT-derived files). Native five-task-type training with
  per-task loss bucketing. **2026-09-16 revision, addressing all 7
  local-review items:** source pinned to
  `github.com/OpenSenseNova/SenseNova-U1@f97964a6e54b0abf92aa2db849af4e942bb2ff08`;
  checkpoint pinned to HF revision `846ff1352e3a4e900d064740cddfc163b115646f`,
  downloaded to and executed from verified local SSD (214 files hashed,
  35,217,355,798 bytes); the SFT checkpoint itself (not T230's checkpoint) was
  loaded directly and exercised with one pure-understanding
  (`understanding_loss=9.830`) and one pure-generation
  (`generation_loss=5.436`) forward+backward smoke, each confining gradients
  to its own owned parameter group; a real optimizer+scheduler+resume-metadata
  construction was verified (sha256 fingerprint match) to not mutate weights;
  and the smallest-feasible-H20-topology was measured (peak ~59.5GB on one
  H20) and derived for T270's likely `generation_private`-only subspace
  (plausibly fits 2xH20 with optimizer-state sharding). All artifact
  `canonical_uri` references now use repository-relative paths. No
  limitation was found that requires promoting Show-o2 — SenseNova-U1-SFT
  remains primary.
- **Fallback: Show-o2-1.5B.** Reused T210's accepted admission. The only
  candidate with **GPU-executed** functional evidence of both task paths
  (T210's `inference_mmu.py`/`inference_t2i.py` smokes). Apache-2.0. Honest
  limitations disclosed: (a) official resume restores weights only, not
  optimizer/scheduler state; (b) stage label is moderate-confidence
  (inferred, not an explicit checkpoint-card statement).
- **Excluded: UniDDT.** No `LICENSE`/`LICENSE.md` file in the GitHub repo
  (HTTP 404 confirmed) and no `license:` tag on the Hugging Face Hub API —
  fails the decision rule's licensing requirement independent of its
  otherwise-competitive architecture and resume support.

## What this task does and does not authorize

This task selects a candidate for **T270's future consideration**. Per the
task file's "Successor opening": acceptance of T250 does **not** authorize
training. No optimizer step, persistent weight update, or dataset-scale run
was performed, and none is authorized by this report. The GPU smoke executed
in the 2026-09-16 revision was forward+backward only, explicitly to satisfy
local-review item 3 ("no optimizer step is required").

## Evidence trail

- `reports/T250/checkpoint-stage-evidence.md` — per-candidate stage labels
  with direct quotes/citations; Candidate A's label is now also backed by a
  real load+forward+backward on the pinned SFT checkpoint (2026-09-16
  addendum).
- `reports/T250/training-interface-audit.md` — per-candidate static
  trainer/optimizer/resume audit, plus Candidate A's real GPU smoke,
  optimizer/scheduler-construction, and GPU-memory-measurement evidence
  (2026-09-16 revision addendum).
- `reports/T250/decision-matrix.md` — weighted scoring and the license
  hard-gate exclusion, updated scores for Candidate A's now-GPU-verified
  task-path and H20-envelope dimensions.
- `reports/T250/failure-ledger.md` — every genuine open item/gap found,
  documented honestly rather than glossed over.
- `configs/admission/posttraining-startpoints/candidates.yaml` — structured
  per-candidate records covering all required audit dimensions, now with
  pinned source/checkpoint revisions, measured parameter counts, and
  measured/derived GPU-memory figures for Candidate A.
- `configs/admission/posttraining-startpoints/checkpoint-hashes.json` — per-
  file sha256 + byte size for all 214 files of the downloaded SFT checkpoint.
- `configs/admission/posttraining-startpoints/storage-preflight.json` — local
  SSD (`xfs`, filesystem_class=local) storage preflight against the real
  container-local checkpoint path, status=pass.
- `configs/admission/posttraining-startpoints/artifact-verification.json` —
  produced by `scripts/verify_manifest_artifacts.py` against
  `runs/admission-posttraining-startpoints-v1/manifest.json`.
- `runs/admission-posttraining-startpoints-v1/gpu-smoke-result.json` — full
  numeric evidence for the real GPU smoke (losses, gradient-isolation
  booleans, memory snapshots, weight-mutation fingerprint check).
- `runs/admission-posttraining-startpoints-v1/{manifest.json,metrics.json,notes.md}` —
  formal run manifest (status=pass), metrics (gpu_hours=0.01,
  gpu_count_used=1, persistent_updates=0, stage_labels_complete=true,
  training_readiness_separated_from_inference=true,
  primary_recommendation_recorded=true), and narrative notes describing the
  real GPU work performed.
