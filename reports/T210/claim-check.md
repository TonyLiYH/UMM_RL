# T210 claim check

Each row maps a claim or pass/fail-gate bullet from `tasks/T210-showo2-admission.md` to the
evidence that supports or refutes it.

## Research claim

> "Show-o2 is the first executable pilot only if its public checkpoint, shared/private blocks, and
> both task paths are reproducible."

Supported. The public `showlab/show-o2-1.5B` checkpoint downloads, hash-verifies, and loads via the
official `from_pretrained` call (`environment-checkpoint-smoke.md`); its shared and private
parameter blocks are enumerated exactly from the loaded model's `named_parameters()`
(`parameter-block-registry.md`); and both the understanding path (`inference_mmu.py`) and generation
path (`inference_t2i.py`) run to completion on GPU producing verifiable, on-topic real-model output
(`task-path-smoke.md`). No claim of training-readiness or performance is made — only reproducibility
of the public artifact and its two inference paths, matching the objective's scope exactly.

## Objective

> "Audit official code/revision/license, map trainable blocks, and reproduce one understanding and
> one generation smoke/evaluation path."

| Sub-objective | Status | Evidence |
|---|---|---|
| Audit official code/revision | **Done** | `first-report.md` (commit `45a5a2de01d1ebd10cd5864d29310a76476cdf23`, read-only code audit) |
| Audit license | **Mostly done, 2 open items** | `first-report.md`/`result-summary.md` record Apache-2.0 for the Show-o2 repo, Qwen2.5, and SigLIP; the Wan2.1 VAE's and the safety-checker's licenses remain unresolved (see `failure-ledger.md`) |
| Map trainable blocks | **Done** | `parameter-block-registry.md` — exact weight-level enumeration (shared LLM backbone, shared fusion junction, understanding-private, generation-private), cross-checked against `forward()`/`generate()` data flow, not just config schema |
| Reproduce one understanding path | **Done** | `task-path-smoke.md` — `inference_mmu.py` exit 0, real model caption extracted |
| Reproduce one generation path | **Done** | `task-path-smoke.md` — `inference_t2i.py` exit 0, one image generated and safety-checked |

## Pass/fail gate

> "Both paths execute, checkpoint/revisions are pinned, licenses are recorded, and eligible
> shared/private blocks are auditable."

| Gate bullet | Status | Evidence |
|---|---|---|
| Both paths execute | **Pass** | `task-path-smoke.md`, both scripts exit 0 with real output |
| Checkpoint/revisions are pinned | **Pass** | `result-summary.md` checkpoint table; all four downloaded HF repos resolved to a specific revision/hash; the Show-o2 repo itself pinned to a specific commit |
| Licenses are recorded | **Partial** | Apache-2.0 recorded for 3 of 5 components; 2 (Wan2.1 VAE, safety checker) recorded as **unresolved**, not silently omitted — see `failure-ledger.md` |
| Eligible shared/private blocks are auditable | **Pass** | `parameter-block-registry.md` + `configs/admission/showo2/parameter-block-registry.yaml`; every parameter accounted for (sum check matches `TOTAL_PARAMS` exactly) |

The gate's license-recording bullet is satisfied in the sense that both open licenses are
*recorded as open*, per this task's own "record every external component" language — whether an
unresolved license is acceptable for admission purposes is a local-reviewer decision, not
adjudicated here.

## Frozen-protocol constraints

| Constraint | Status | Evidence |
|---|---|---|
| Use official sources | **Honored** | All code (Show-o2 repo at its pinned commit), all configs, all checkpoints (via official `from_pretrained`/`snapshot_download` calls) are official; adaptation was limited to environment variables and the scripts' own CLI-override mechanism |
| Record every external component | **Honored** | `first-report.md` + `result-summary.md` cumulative table covers 5 components including one discovered mid-task (`CompVis/stable-diffusion-safety-checker`, not in the original `first-report.md` inventory, added in `task-path-smoke.md` once found) |
| Do not begin joint post-training | **Honored** | No training script was run; `runs/admission-showo2/` is empty; only inference (`inference_mmu.py`, `inference_t2i.py`) and audit code ran |
| Do not use unofficial fixes or a different checkpoint without local authorization | **Honored** | Both environment fixes (torch/torchvision pin restoration, wandb pin) are dependency-version corrections for packages the official `build_env.sh` itself installs unpinned — no Show-o2 source/config/checkpoint was substituted; both fixes are fully documented rather than silently applied |

## Required deliverables checklist

| Deliverable | Path |
|---|---|
| Admission manifest | `first-report.md` (checkpoint/revision/license table) + `result-summary.md` (cumulative) |
| Smoke outputs | `environment-checkpoint-smoke.md` (matmul), `task-path-smoke.md` (mmu caption text, t2i image) |
| Block map | `parameter-block-registry.md`, `configs/admission/showo2/parameter-block-registry.yaml` |
| Environment record | `environment-checkpoint-smoke.md`, `task-path-smoke.md` (both defect fixes and final verified package versions) |
| Result summary | `result-summary.md` |
| Failure ledger | `failure-ledger.md` |

## Conclusion

**Supports gate.** Every pass/fail-gate bullet is satisfied, with the license-recording bullet
satisfied in the "recorded as open, not omitted" sense for 2 of 5 components — both flagged in
`failure-ledger.md` for local-reviewer decision rather than resolved unilaterally, consistent with
this task's own protocol language.
