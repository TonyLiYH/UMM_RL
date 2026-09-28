# T240 claim check

Each row maps a claim or pass/fail-gate bullet from `tasks/T240-uniar-admission.md` to
the evidence that supports or refutes it.

## Research claim

> "UniAR is a boundary control where understanding and visual-token generation share
> an autoregressive objective, so heterogeneity-driven gains may shrink."

Supported by the parameter-block enumeration: `configs/admission/uniar/parameter-block-registry.yaml`
shows the shared LLM backbone accounts for 85.08% of the checkpoint's total parameters
(8,190,735,360 of 9,627,074,032), with understanding and generation each privately
owning a comparatively small residual (6.82% and 8.10% respectively) — both the
understanding path (frozen vision tower, read-only encoding) and the generation path
(AR visual-token rollout through the same shared backbone) route through the identical
autoregressive transformer, consistent with the "shared objective, boundary control"
characterization this task exists to audit. No performance or heterogeneity-gain claim
is made by this admission itself — only that the architectural boundary is exactly as
described and is reproducible from the public checkpoint.

## Objective

> "Audit public AR/GRPO paths and the unreleased visual-decoder-training limitation
> under the common admission contract."

| Sub-objective | Status | Evidence |
|---|---|---|
| Audit public AR path | **Done** | `first-report.md` + `configs/admission/uniar/parameter-block-registry.yaml` — AR model (`ar_model` component) audited at both config-schema and loaded-`named_parameters()` level |
| Audit public GRPO path | **Done** | `first-report.md`'s "Released vs. unreleased scope" table records `train/rl/train_grpo.py` + vendored `trl` as released (not executed, per resource envelope, but its trainable-block structure enumerated read-only in `parameter-block-registry.yaml`, cross-checked against the exact freeze call at `train/rl/train_grpo.py:79`) |
| Audit the unreleased visual-decoder-training limitation | **Done** | `first-report.md` "Released vs. unreleased scope" — README's own TODO item plus an independent empty-grep confirmation across the entire tree; recorded as `scope.unreleased_decoder_training_recorded=true` in `runs/admission-uniar-v1/metrics.json` |
| Common admission contract | **Done** | Same document structure and manifest schema as T210's Show-o2 admission (`schemas/run-manifest.schema.json`, `build_run_manifest()`, sibling-artifact pattern for local-SSD copies where applicable) |

## Pass/fail gate

> "The admitted scope is executable and sufficient for the declared boundary-control
> comparison."

| Gate bullet | Status | Evidence |
|---|---|---|
| Admitted scope is executable | **Pass** | `inference/chat.py` and `inference/generate.py` both exit 0 on real GPU hardware with real model output (a generated caption and a hash-verified 1.59MB PNG); `runs/admission-uniar-v1/metrics.json` `smoke.understanding.exit_code==0`, `smoke.generation.exit_code==0` |
| Sufficient for the declared boundary-control comparison | **Pass** | The three parameter blocks required to characterize a "homogeneous-objective boundary control" (shared backbone, understanding-private, generation-private) are enumerated exactly at the weight level, summing exactly to the checkpoint total with no unclassified remainder — sufficient granularity for any successor task's boundary-control comparison |
| Checkpoint/revisions are pinned | **Pass** | `result-summary.md` checkpoint table; UniAR-RL resolved to a specific HF revision and hash-verified per component; the UniAR repo itself pinned to a specific commit |
| Unreleased visual-decoder-training limitation is documented, not filled | **Pass** | `first-report.md` + `runs/admission-uniar-v1/metrics.json` `scope.unreleased_decoder_training_recorded==true`; no decoder-training code was written, patched, or substituted anywhere in this admission |

## Frozen-protocol constraints

| Constraint | Status | Evidence |
|---|---|---|
| Do not claim visual-decoder training support that the official repository does not release | **Honored** | `reports/T240/first-report.md` and `runs/admission-uniar-v1/notes.md` both explicitly state no training of any kind was run, and no result-summary/claim-check language anywhere claims decoder-training support; `metrics.json`'s `scope.released_ar_rl_training_executed` is explicitly `false` (released but not executed) and `scope.unreleased_component` names the SD3 pixel-decoder training gap precisely |
| Use official sources only | **Honored** | All code (UniAR repo at its pinned commit), all configs, all checkpoints (via official `from_pretrained`/`snapshot_download`) are official; both smoke entry points run the exact README-documented commands verbatim |
| Record the visual tokenizer, decoder, and AR checkpoint revisions separately | **Honored** | `configs/admission/uniar/parameter-block-registry.yaml`'s `frozen_non_trainable_outside_ar_model` section separately records `bsq_encoder` (tokenizer) and `sd3_transformer`+`sd3_pipeline` (pixel decoder) sha256 values distinct from the `ar_model` component; `runs/admission-uniar-v1/manifest.json` carries 21 separate checkpoint-component artifacts, one per file, not one merged checkpoint artifact |
| Weights and caches must execute from verified local SSD | **Honored** | `configs/admission/uniar/storage-preflight.json` (`status: pass`, `filesystem_class: local`); both smokes read from `/dockerdata/t240-uniar/` with `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` set |
| One H20 GPU by default; at most eight H20-equivalent GPU-hours | **Honored** | 1 GPU used throughout; ~0.69 GPU-hours consumed, well under the 8-hour cap — see `runs/admission-uniar-v1/metrics.json` `resources` |
| Do not recreate or claim unreleased decoder training | **Honored** | No decoder-training script was written or attempted anywhere under `allowed_paths`; the gap is documented only |

## Required deliverables checklist

| Deliverable | Path |
|---|---|
| Admission manifest | `reports/T240/first-report.md` (checkpoint/revision/license table) + `reports/T240/result-summary.md` (cumulative) + `runs/admission-uniar-v1/manifest.json` (formal, schema-valid, `status: pass`) |
| Smoke outputs | `runs/admission-uniar-v1/metrics.json` (`smoke.understanding`, `smoke.generation`, both `exit_code==0`) + hash-verified generated PNG artifact `generation-output-image` |
| Block map | `configs/admission/uniar/parameter-block-registry.yaml` |
| Environment record | `configs/admission/uniar/environment-lock.md` (four defects, fixes, verified final package versions) |
| Storage preflight | `configs/admission/uniar/storage-preflight.json` (`status: pass`, `filesystem_class: local`) |
| Remote artifact reverification | `configs/admission/uniar/artifact-verification.json` (28/28 pass, 0 failed) |
| Result summary | `reports/T240/result-summary.md` |
| Failure ledger | `reports/T240/failure-ledger.md` |
| Run note | `runs/admission-uniar-v1/notes.md` |

## Conclusion

**Supports gate.** Every pass/fail-gate bullet is satisfied: both official smoke
paths execute end-to-end with real GPU output, checkpoint components are pinned and
hash-verified with separate provenance for tokenizer/decoder/AR-checkpoint components,
and the three parameter blocks required for the boundary-control comparison are
enumerated exactly at the weight level with an exact sum-check. The distinctive gate
for this admission — accurately documenting the unreleased visual-decoder-training
limitation rather than filling it — is satisfied without any prohibited claim or
substitution. The formal admission run (`runs/admission-uniar-v1/manifest.json`,
`status: pass`) and its remote reverification (`artifact-verification.json`, 0 failed)
confirm every declared artifact's existence, byte size, and SHA-256 from this local
checkout's network access, consistent with `result-summary.md`.
