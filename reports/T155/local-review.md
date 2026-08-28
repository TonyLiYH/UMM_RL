# T155 local review — 2026-08-28

Decision: `revision_needed`.

Reviewed branch: `origin/agent/T155-exact-oracle` at
`e1cb03446340744f8a59ea88fe149e5481263645`.

## Verified progress

- Exact SGD and augmented-state momentum response implementations are present.
- Structured graph families, curvature, coupling, gradient geometry, and noise
  controls are implemented.
- Independent local run completed with 81 passing tests and one repository
  state failure.
- The reported 288-case sweep is reproducible from the committed configuration;
  286 cases pass and the two retained failures occur in deliberately unstable
  momentum cases.

## Required revisions

### R1 — valid repository run manifest

`runs/oracle-20260827-baseline/manifest.json` is currently a JSON array.
The repository validator requires every `*manifest.json` to be a JSON object
conforming to `schemas/run-manifest.schema.json`; consequently the branch fails
`python -m comppareto.repo_state.cli --root .` and the full test suite.

Create a schema-valid run manifest object and move the per-case array to a
separate result file such as `case-records.json`. The manifest must include
task ID, source and execution revisions, clean/dirty state, configuration hash,
environment, status, result files, artifact references, hashes, sizes, and
retry information.

### R2 — persist the promised detailed reference artifacts

The reports claim that six selected cases persist complete trajectories,
Jacobians, sensitivities, and exact quadratic objects. No task record in the
committed case array contains a `detail` field because `case_record()` does not
serialize `TaskCaseResult.detail`.

Persist and hash the selected detailed references required by T160, including:

- state trajectories;
- momentum buffers where applicable;
- per-step \(J_k,B_k\);
- sensitivity trajectories;
- exact local gradient and \(Q_i^K\);
- selector and case identifiers.

Add tests that load these artifacts and independently verify their shapes,
hashes, and final values.

### R3 — implement the declared high-accuracy Pareto reference

The T155 research claim includes high-accuracy Pareto references, but the
current implementation evaluates random probe directions only. Add an
independent exact or high-accuracy common-descent/Pareto reference over the
lifted task gradients, with KKT or projection residuals and boundary tests for
disjoint, partial, and full overlap.

### R4 — correct the differentiation terminology

The reference implementation is an independently hand-coded reverse-mode
derivative over the literal unroll, not an external automatic-differentiation
engine. Rename the claim and reports accordingly, or obtain local authorization
for an additional dependency and provide a genuinely independent AD reference.

### R5 — resolve the two unstable loss-change checks without relaxing the gate

Keep the original float64 failures. Add a higher-precision or
conditioning-aware reference evaluation to determine whether the discrepancies
are pure cancellation. Report absolute forward error, relative error,
conditioning quantities, and the higher-precision result. Do not change the
frozen tolerance without a separate local decision.

### R6 — remove the out-of-scope change

Revert the T155 modification to `CHANGELOG.md`; it is outside the task's
declared `allowed_paths`.

## Re-review gate

Before resubmission:

- repository state validator passes;
- complete test suite passes;
- `git diff --check` passes;
- every changed path is allowed;
- the regenerated run is tied to a clean execution revision;
- T155 remains `awaiting_review` only after all revised evidence is pushed.
