---
id: T215
title: Show-o2 finite-response diagnostic feasibility
parent: T200
status: running
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T215-showo2-finite-response-feasibility
depends_on: [T210]
blocks: [T300, T310]
allowed_paths: ["tasks/T215-showo2-finite-response-feasibility.md", "configs/feasibility/showo2/", "runs/feasibility-showo2-v1/", "reports/T215/", "src/comppareto/adapters/showo2/", "tests/adapters/showo2/"]
source_revision: "217d183b30995db4ac82158259f45800e57e2eb1"
created_at: 2026-08-27
updated_at: 2026-09-03
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
