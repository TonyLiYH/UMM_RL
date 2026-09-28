---
id: T215
title: Show-o2 finite-response diagnostic feasibility
parent: T200
status: blocked
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T215-showo2-finite-response-feasibility
depends_on: [T210]
blocks: [T300, T310]
allowed_paths: ["tasks/T215-showo2-finite-response-feasibility.md", "configs/feasibility/showo2/", "runs/feasibility-showo2-v1/", "reports/T215/", "src/comppareto/adapters/showo2/", "tests/adapters/showo2/"]
source_revision: "217d183b30995db4ac82158259f45800e57e2eb1"
created_at: 2026-08-27
updated_at: 2026-09-28
---

# T215: Show-o2 finite-response diagnostic feasibility

## Research claim

Show-o2 can support a reversible, optimizer-state-aware finite-response
diagnostic on a selected shared/private parameter subspace before formal D0
experiments are authorized.

## Objective

Implement and validate parameter/optimizer-state snapshot and restore, raw and
commit-response gradients, rerun-response finite unrolling, directional
finite-difference checks, and resource accounting for one understanding and one
generation path.

## Dependencies and inputs

Accepted T210 admission evidence, its pinned official source/checkpoint, and the
protocol in `docs/plans/showo2-first-attempt.md`.

## Allowed changes

Show-o2 diagnostic adapter code and tests, feasibility configs and runs, T215
reports, and this task file only.

## Frozen protocol

Use the accepted official Show-o2 revision and checkpoint. Freeze tokenizer,
VAE, and unrelated backbone blocks. Start with one selected shared block and
one private block per task. \(K=1\) is mandatory; \(K=3\) is conditional on the
\(K=1\) memory and correctness gate. Every diagnostic transition is rolled
back. No persistent joint post-training is authorized.

## Execution stages

1. Publish the proposed subspace, state inventory, batch manifest, finite
   difference directions, tolerances, and measured resource estimate.
2. Implement deterministic parameter, optimizer, RNG, and data-order
   snapshot/restore.
3. Compute raw and commit-response stop-gradients.
4. Compute the rerun-response finite-unroll hypergradient for \(K=1\).
5. Compare automatic differentiation with central finite differences.
6. Compare parameter-only and complete optimizer-state differentiation.
7. Run \(K=3\) only after the \(K=1\) gate passes.
8. Emit feasibility, resource, and failure reports.

## Pass/fail gate

At least one declared shared/private subspace must:

- restore persistent floating state within the declared dtype tolerance and
  restore counters, RNG, and data-order state exactly;
- produce separate raw, commit-response, and rerun-response measurements;
- match the directional finite-difference reference to relative error at most
  \(10^{-3}\) when the reference magnitude exceeds \(10^{-8}\), or absolute
  error at most \(10^{-6}\) near zero;
- record complete peak-memory, wall-clock, FLOPs or gradient-evaluation, and
  extra-data accounting;
- remain inside the two-GPU, eight-H20-equivalent-GPU-hour default envelope.

Unsupported transitions, OOMs, state-restoration mismatches, silent detachments,
or unstable finite differences fail the affected configuration and remain in
the ledger.

## First report

Before GPU execution, return the selected module paths and parameter counts,
optimizer-state tensors and counters, rerun/commit pseudocode, batch and seed
manifest, finite-difference directions, expected memory, expected runtime,
required assets, and exact commands.

Commit and push the first report before GPU execution. The executor may proceed
without another approval only when the selected subspace and resource estimate
remain inside this task's frozen protocol and the storage preflight passes.

## Required deliverables

Adapter code, snapshot/restore and gradient tests, resolved configurations,
run manifests, raw/commit/rerun comparison table, finite-difference residuals,
resource table, result summary, and failure ledger.

## Artifact and provenance requirements

Every row records the accepted Show-o2 source/checkpoint, task path, block IDs,
parameter and optimizer-state hashes, response protocol, \(K\), batches, seeds,
dtype, hardware, source revision, config hash, output hashes, and rollback
result.

## Failure and retry rules

Do not expand the trainable subspace, response horizon, GPU count, or budget
after observing diagnostic values. Infrastructure retries reuse the same
configuration and seed. Numerical or differentiation failure counts as a
result unless a task-wide preregistered rule applies.

## Resource envelope

- at most two H20 GPUs;
- at most eight H20-equivalent GPU-hours;
- \(K=1\) must pass before \(K=3\);
- no full-backbone unroll and no persistent parameter update;
- all model assets and caches execute from the accepted local-SSD layout.

## Automated submission gate

Before setting `awaiting_review`, run:

```bash
bash scripts/validate_task_submission.sh T215
```

## Successor opening

Accepted T215 contributes to T310 and T300. T300 still requires all of its
other declared dependencies.

## Review history

- 2026-08-27 — Planned after the Show-o2 first-attempt design; T210 is not yet accepted.
- 2026-09-01 — T210 accepted with recorded limitations; T215 authorized for
  reversible diagnostic execution. No persistent training is authorized.
- 2026-09-03 — Remote executor created branch
  `agent/T215-showo2-finite-response-feasibility` from `origin/main`
  (`4e34878abbb03e11bd722af40788e5b0fdb87a66`) and set status to `running`.
  **Data-integrity correction**: this file's `source_revision` field as found on
  `origin/main` was `217d183473a14ad48852205ea3f2746301915729`, which does not
  resolve to any git object in this repository (confirmed via
  `git cat-file -e` and `git rev-list --objects --all` across every fetched
  ref: `main`, `agent/T155-exact-oracle`, `agent/T210-showo2-admission`). The
  same malformed value was found identically in T220/T230/T240's frontmatter.
  A real commit, `217d183b30995db4ac82158259f45800e57e2eb1` ("merge: accept
  Show-o2 admission evidence"), shares the same 7-character abbreviation but
  differs in the remaining 33 hex characters — consistent with a
  truncate-then-refill transcription defect at task-authoring time. Corrected
  `source_revision` to the verified real commit
  `217d183b30995db4ac82158259f45800e57e2eb1` per explicit user authorization
  (2026-09-03) after presenting this exact finding; this is the only
  frontmatter field changed. `scripts/validate_task_submission.sh T215`'s
  `revision_is_ancestor` check would otherwise hard-fail
  (`git merge-base --is-ancestor 217d183473a14ad48852205ea3f2746301915729 HEAD`
  errors with "Not a valid commit name") independent of any work performed on
  this branch.
- 2026-09-03 — Remote executor ran the real K=1 diagnostic against the accepted checkpoint for
  both task paths (`runs/feasibility-showo2-v1/`). Two genuine infrastructure defects were found
  and fixed as permitted infrastructure retries (commits `cd18c3b`, `22892b1`): the default SDPA
  backend lacked double-backward support for `create_graph=True`; the finite-difference
  Rademacher direction tensors were built on the wrong device. After both fixes, the diagnostic
  completed without a Python exception but **the K=1 gate did not pass for either task path**:
  MMU's rerun-response analytic gradient is `NaN` (confirmed root cause via an isolated repro on
  the same container/torch build — AdamW's `eps`-outside-`sqrt` denominator differentiated through
  an exactly-zero `grad_p` coordinate at optimizer step 1, an intrinsic property of the frozen
  AdamW formula, not an infrastructure bug); T2I's analytic gradient is finite but misses the
  declared finite-difference tolerance by 43%-183% relative error on 3 of 4 fixed directions
  (reported as a genuine numerical-accuracy result). Per the task's explicit rule, K=3 was **not**
  run for either task path. `persistent_updates: 0` and every actual parameter/gradient tensor
  restored exactly in all runs (`data_max_abs_diff: 0.0`); the combined rollback check nonetheless
  reports `snapshot_restore.failed: 4` due to an RNG-exact-restore sub-check artifact (see
  `reports/T215/failure-ledger.md`, `ROLLBACK-RNG`), unrelated to any persistent parameter
  mutation. Full numeric results: `reports/T215/result-summary.md`; claim-by-claim gate
  determination: `reports/T215/claim-check.md`; complete failure/root-cause ledger:
  `reports/T215/failure-ledger.md`. Cumulative GPU usage this session (including 3 earlier
  infrastructure-debugging attempts): ~0.0323 GPU-hours on 1 H20 GPU, against the 8-hour/2-GPU
  cap. Committed and pushed: `b2ad196` (results/reports), `37826a5` (removed an out-of-scope
  `src/comppareto/adapters/__init__.py` file that `validate_task_submission.sh` flagged as an
  unauthorized changed path outside this task's `allowed_paths`; confirmed unnecessary via direct
  import test and the full showo2 test suite).
  **Setting status to `blocked` rather than `awaiting_review`**: `tasks/contracts/T215.acceptance.yaml`
  (outside this task's `allowed_paths`, so not modifiable by this executor) requires
  `runs/feasibility-showo2-v1/manifest.json:status == "pass"` and
  `metrics.json:{snapshot_restore.failed, finite_difference.failed} == 0` unconditionally, with no
  branch for an honestly-reported, gated K=1 failure. The task's own pass/fail gate language
  explicitly anticipates this outcome ("Unsupported transitions... or unstable finite differences
  fail the affected configuration and remain in the ledger"), and the diagnostic's actual, correct
  behavior when K=1 fails is to report the failure and skip K=3 — not to retry with a different
  seed/subspace/tolerance to force a pass, which the failure/retry rules explicitly forbid. This is
  therefore a genuine, out-of-scope blocker (a static acceptance-contract metric that cannot
  represent a valid negative diagnostic result) rather than something this run can resolve; a
  local-research-agent/task-owning decision is needed on whether `T215.acceptance.yaml` should be
  revised to accept a documented K=1 failure as a valid submission, or whether T215 should be
  formally closed as a negative feasibility result via some other path. No fabricated or rounded
  numbers were used to force a pass.
- 2026-09-28 — Remote executor resumed T215 for a second, independent verification pass, per
  `origin/main`'s `tasks/README.md` re-listing T215 as `ready` (this task file's own front matter
  was still `blocked` from the prior attempt — that is the state resumed from, not a fresh restart;
  no local-review acceptance of the prior `blocked` outcome had occurred). Merged 71 unrelated
  `origin/main` commits into this branch with zero conflicts in T215's `allowed_paths` (merge commit
  `2eab9ba`). No adapter/protocol/config file was modified and no new GPU run was executed this
  round — the objective was to determine, independently, whether commit `e33d76c`'s `blocked`
  verdict reflects a genuine, fixable implementation defect (in which case fix and rerun for real)
  or an irreducible negative finding under the frozen protocol (in which case strengthen the
  evidence trail). Findings, each re-derived directly from the code and the already-recorded
  `runs/feasibility-showo2-v1/metrics.json`, not merely re-read from the prior report:
  1. **MMU-NAN reconfirmed as a true mathematical singularity, not a coincidental fp32 rounding
     artifact.** Re-derived `protocols.py::adamw_step`'s bias-correction identity by hand: at
     `step=1`, `bias_correction2 == 1 - ADAMW_BETA2`, and since `exp_avg_sq` starts at zero,
     `exp_avg_sq_1 == (1-ADAMW_BETA2) * grad_p**2` exactly, so `exp_avg_sq_1/bias_correction2 ==
     grad_p**2` bit-for-bit and `denom = sqrt(grad_p**2) + eps = |grad_p| + eps`. `d(sqrt(x))/dx`
     at `x=0` is a genuine `0/0` under the chain rule when `grad_p` is exactly zero, independent of
     floating-point precision — the MMU NTP loss's `IGNORE_INDEX=-100` masking structurally zeroes
     the gradient contribution of most target positions across the ~21.7M-parameter
     `fusion_proj`+`und_trans.layers[0]` block, making an exact-zero coordinate a structural
     near-certainty, not an underflow coincidence a higher-precision dtype would avoid. Checked
     whether moving `eps` inside the `sqrt` (a known differentiable-optimizer stabilization
     convention) would fix this: `reports/T215/first-report.md` section 4 declared the
     exact-`torch.optim.AdamW`-semantics formula (`eps` outside `sqrt`) in the pre-registered first
     report, published before any GPU execution — relocating `eps` now, after observing the NaN,
     would be a post-hoc protocol-formula change with no preregistered rule authorizing it. Declined
     to apply it. **MMU-NAN stands as previously recorded.**
  2. **T2I-FDMISS: identified and quantitatively confirmed the actual noise mechanism (fp32
     catastrophic cancellation in the central-difference numerator), strengthening rather than
     overturning the prior finding.** Estimated cancellation-error floor:
     `machine_eps(fp32) * max(|loss_plus|,|loss_minus|) / (2*eps) ≈ 1.19e-7 * 0.1 / 1.16e-4 ≈ 1e-4`
     — matches the observed `|analytic_value - fd_value|` gaps on the 3 Rademacher directions of the
     `disjoint_k1` T2I variant (`1.18e-4`, `5.6e-5`, `1.2e-5`) to within an order of magnitude, and
     explains why only the 4th ("natural", gradient-aligned) direction — whose `fd_value` (`0.147`)
     is 3-4 orders of magnitude larger than the Rademacher directions' (`~1e-4`) — comes anywhere
     close to tolerance (0.6-1.0% relative error): its true directional derivative is large enough to
     swamp this fp32 floor, while the small-magnitude directions are not. This sharpens, and does not
     contradict, the prior ledger's qualitative "higher-order curvature / cancellation noise"
     description. Considered recomputing the FD reference in fp64 to remove this artifact: plausible
     in principle, but declined — `reports/T215/first-report.md` section 1 explicitly commits to
     fp32 for the diagnostic ("the diagnostic's finite-difference check ... need[s] fp32 numerical
     precision"), published before any GPU execution; switching to fp64 now, having observed the
     fp32 result fail, is exactly the post-hoc eps/precision substitution the failure ledger already
     identifies as prohibited. A fp64 FD variant would need to be proposed and preregistered as a new,
     distinct diagnostic configuration, not retrofitted onto this K=1 gate's recorded result. **T2I-FDMISS
     stands as previously recorded, now with a quantitatively confirmed root-cause mechanism.**
  3. **Contract-vs-finding mismatch reconfirmed independently.** Re-ran
     `bash scripts/validate_task_submission.sh T215` from a clean, post-merge tree: the full local
     test suite (341 tests, up from 190 due to the merged-in unrelated tasks' own tests) passes
     cleanly; the submission gate fails on exactly the same 4 conditions as before (task status not
     `awaiting_review`; `manifest.json:status` `fail`≠`pass`; `metrics.json:snapshot_restore.failed`
     `4`≠`0`; `metrics.json:finite_difference.failed` `4`≠`0`). Verbatim output:
     ```
     task_tree=pass tasks=43
     run_manifests=pass manifests=12
     research_state=pass
     341 passed in 188.63s (0:03:08)
     submission_validation=fail task=T215
     task status must be awaiting_review for submission; found blocked
     runs/feasibility-showo2-v1/manifest.json:status: expected == 'pass', found 'fail'
     runs/feasibility-showo2-v1/metrics.json:snapshot_restore.failed: expected == 0, found 4
     runs/feasibility-showo2-v1/metrics.json:finite_difference.failed: expected == 0, found 4
     ```
  **Conclusion**: the prior `blocked` verdict is confirmed correct and irreducible under the frozen
  protocol as declared in the pre-registered first report. No fixable defect was found in the
  diagnostic implementation itself — both the AdamW singularity and the FD-noise mechanism were
  independently re-derived mathematically this round, not merely re-read. The two K=1 failures
  (MMU's exact differentiation singularity at a structurally-zero-gradient coordinate; T2I's
  fp32-cancellation-dominated FD mismatch on 3 of 4 directions) are genuine, now doubly root-caused
  negative results, and `tasks/contracts/T215.acceptance.yaml`'s unconditional pass-only metric
  structure has no branch to represent them. **Status remains `blocked`.** This remains a
  local-research-agent/task-owning decision point (revise the acceptance contract to accept a
  documented, gated K=1 failure as a valid closed submission, or formally close T215 as a negative
  feasibility result via another mechanism) — not something further remote GPU execution within
  this task's current authorization can resolve. No file outside this task's `allowed_paths` was
  modified; no new fabricated or rounded numbers were used.
