# T230 claim check

Each row maps a claim or pass/fail-gate bullet from `tasks/T230-sensenova-u1-admission.md` to the
evidence that supports or refutes it.

## Research claim

> (from the task's frozen protocol and objective) SenseNova-U1's released full-parameter native
> pixel/MoT path is admissible only if its released code exposes a reproducible full-parameter
> path and both task evaluations, with U1 (not the unreleased U1.5 training pipeline) as the
> audited target.

Supported. The public `sensenova/SenseNova-U1-8B-MoT` checkpoint downloads, hash-verifies, and
loads via the official model-loading path (`configs/admission/sensenova-u1/artifact-verification.json`,
15/15 pass); its shared and private parameter blocks are enumerated exactly from the loaded
model's `named_parameters()` (`configs/admission/sensenova-u1/parameter-block-registry.yaml`); and
both the understanding path (`examples/vqa/inference.py`) and generation path
(`examples/t2i/inference.py`) run to completion on GPU producing verifiable, on-topic real-model
output (`runs/admission-sensenova-u1-v1/metrics.json`). U1.5's unreleased training pipeline was
never substituted or exercised (`first-report.md`'s Project Status / news-entry evidence). No
claim of training-readiness or performance is made — only reproducibility of the public artifact
and its two inference paths.

## Objective

> "Audit the released U1 full-parameter path; U1.5 remains excluded until its announced training
> pipeline is public."

| Sub-objective | Status | Evidence |
|---|---|---|
| Source/license audit | **Done** | `reports/T230/first-report.md` (commit `f97964a6e54b0abf92aa2db849af4e942bb2ff08`, Apache-2.0 repo + checkpoint license, read-only) |
| U1-vs-U1.5 scope confirmation | **Done** | `first-report.md`'s "U1 vs. U1.5 scope confirmation" section — U1 checkpoints/report released, U1.5 training pipeline explicitly "in preparation," not substituted |
| Understanding smoke | **Done** | `runs/admission-sensenova-u1-v1/metrics.json`'s `smoke.understanding`, `exit_code: 0`, direct evidence |
| Generation smoke | **Done** | `runs/admission-sensenova-u1-v1/metrics.json`'s `smoke.generation`, `exit_code: 0`, direct evidence |
| Trainable-block map | **Done** | `configs/admission/sensenova-u1/parameter-block-registry.yaml` — exact weight-level enumeration (`shared`, `understanding_transformer`, `generation_transformer`), `unassigned_trainable_parameters: 0` |
| Static routed-overlap audit | **Done** | `runs/admission-sensenova-u1-v1/metrics.json`'s `routed_overlap` block — MoT boolean-mask mechanism documented, distinguished from A3B's MoE gate, one static assumption violation recorded (`static_assumption_violations_recorded: true`) |
| Report | **Done** | `reports/T230/first-report.md`, `reports/T230/result-summary.md`, `reports/T230/claim-check.md`, `reports/T230/failure-ledger.md` |

## Pass/fail gate

> "Released U1 code exposes a reproducible full-parameter path and both task evaluations."

| Gate bullet | Status | Evidence |
|---|---|---|
| Both paths execute | **Pass** | `runs/admission-sensenova-u1-v1/metrics.json`, both scripts exit 0 with real output (VQA caption text, T2I generated image) |
| Full-parameter path (not a reduced/quantized variant) | **Pass** | `configs/admission/sensenova-u1/parameter-block-registry.yaml` records 17,552,340,992 total = trainable params, matching the checkpoint's documented full-parameter size (`docs/parameter_breakdown.md`'s example figures reproduced exactly on the actual downloaded weights) |
| Checkpoint/revisions are pinned | **Pass** | `result-summary.md` checkpoint table; HF sha `bfa9b436503cb8aed4f2bc60e3236710cc77468d`, repo commit `f97964a6e54b0abf92aa2db849af4e942bb2ff08` |
| Licenses are recorded | **Pass** | Apache-2.0 recorded for both the repository and the primary checkpoint; no non-Apache third-party dependency identified |
| Eligible shared/private blocks are auditable | **Pass** | `parameter-block-registry.yaml`; every parameter accounted for across three disjoint blocks, zero catch-all/unassigned |

## Frozen-protocol constraints

| Constraint | Status | Evidence |
|---|---|---|
| Do not substitute U1.5 preview or unreleased training code for the admitted U1 path | **Honored** | `first-report.md`'s scope-confirmation section; no U1.5 asset, checkpoint, or code path was downloaded or exercised at any stage (confirmed via checkpoint identifiers table in `result-summary.md`) |
| Use official sources | **Honored** | All code (SenseNova-U1 repo at its pinned commit), all configs, all checkpoints (via official model download) are official; adaptation was limited to CLI arguments the scripts themselves expose (`--attn_backend sdpa`) |
| Record every external component | **Honored** | `first-report.md` + `result-summary.md` record the primary checkpoint plus the out-of-scope SFT and A3B variants for completeness; no additional third-party dependency (VAE/safety-checker-style) was found bundled outside the `sensenova_u1` package |
| Resource envelope: one H20 GPU default, ≤12 GPU-hours, admission smoke + static routed-overlap audit only, no U1.5 substitution, no recreated training pipeline, weights/caches execute from verified local SSD | **Honored** | Single GPU0 used throughout; total measured GPU wall-clock 162s (≈0.045 GPU-hours, well under the 12-hour cap); no training was run; `configs/admission/sensenova-u1/storage-preflight.json` confirms `filesystem_class: local` for the execution path |
| An unreleased pipeline is a blocker, not permission to recreate it | **Honored** | The `training/` subtree (derived from InternEvo) is recorded as out of scope and was not exercised; the mixed und/gen forward path's `NotImplementedError` (issue #207) was recorded as a documented limitation, not worked around or reimplemented |

## Required deliverables checklist

| Deliverable | Path |
|---|---|
| Admission manifest | `first-report.md` (checkpoint/revision/license table) + `result-summary.md` (cumulative) + `runs/admission-sensenova-u1-v1/manifest.json` (formal, schema-valid, `status: pass`) |
| Smokes | `runs/admission-sensenova-u1-v1/metrics.json` (`smoke.understanding`/`smoke.generation`, both `exit_code: 0`) |
| Block map | `configs/admission/sensenova-u1/parameter-block-registry.yaml` |
| Routed-overlap audit | `runs/admission-sensenova-u1-v1/metrics.json`'s `routed_overlap` block |
| Environment record | `configs/admission/sensenova-u1/environment-lock.md` |
| Storage preflight | `configs/admission/sensenova-u1/storage-preflight.json` (`status: pass`, `filesystem_class: local`) |
| Remote artifact reverification | `configs/admission/sensenova-u1/artifact-verification.json` (15/15 pass, 0 failed) |
| Result summary | `reports/T230/result-summary.md` |
| Failure ledger | `reports/T230/failure-ledger.md` |

## Conclusion

**Supports gate.** Every pass/fail-gate bullet is satisfied. The formal admission run
(`runs/admission-sensenova-u1-v1/manifest.json`, `status: pass`) and its remote reverification
(`artifact-verification.json`, 0 failed) confirm every declared artifact's existence, byte size,
and SHA-256, consistent with `result-summary.md`. No U1.5 substitution occurred at any stage.
