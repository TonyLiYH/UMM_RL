# T250 — Result summary

## Outcome

**Pass.** Every candidate has an evidence-backed or explicitly-hedged stage
label; inference admission (Show-o2, via T210) is explicitly separated from
training readiness (independently audited here); at least one candidate
(in fact two) is classified as a reproducible starting point. No persistent
parameter update was performed — 0 GPU-hours consumed against a 4-hour cap,
0 optimizer steps, 0 dataset-scale runs.

## Selection

- **Primary: SenseNova-U1-8B-MoT-SFT.** Evidence-backed SFT stage label
  (official README states explicitly that SFT precedes Multi-Expert RL/OPD;
  arXiv 2605.12500 names Stage 5 as explicit Flow-GRPO reinforcement learning
  with named reward models). Full-state resume (model + optimizer + scheduler
  + step/dataloader counters) confirmed by direct code inspection of
  `try_load_internevo_ckpt`. Exact, published, reproducible shared/private
  parameter breakdown (1.245B shared / 8.121B understanding / 8.186B
  generation of 17.552B total). Apache-2.0 (+ attributed MIT-derived files).
  Native five-task-type training with per-task loss bucketing.
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
training. No training was run, and none is authorized by this report.

## Evidence trail

- `reports/T250/checkpoint-stage-evidence.md` — per-candidate stage labels
  with direct quotes/citations.
- `reports/T250/training-interface-audit.md` — per-candidate static
  trainer/optimizer/resume audit.
- `reports/T250/decision-matrix.md` — weighted scoring and the license
  hard-gate exclusion.
- `reports/T250/failure-ledger.md` — every genuine open item/gap found,
  documented honestly rather than glossed over.
- `configs/admission/posttraining-startpoints/candidates.yaml` — structured
  per-candidate records covering all required audit dimensions.
- `configs/admission/posttraining-startpoints/storage-preflight.json` — local
  SSD (`xfs`, filesystem_class=local) storage preflight, status=pass.
- `configs/admission/posttraining-startpoints/artifact-verification.json` —
  produced by `scripts/verify_manifest_artifacts.py` against
  `runs/admission-posttraining-startpoints-v1/manifest.json`.
- `runs/admission-posttraining-startpoints-v1/{manifest.json,metrics.json,notes.md}` —
  formal run manifest (status=pass), metrics (gpu_hours=0,
  persistent_updates=0, stage_labels_complete=true,
  training_readiness_separated_from_inference=true,
  primary_recommendation_recorded=true), and narrative notes.
