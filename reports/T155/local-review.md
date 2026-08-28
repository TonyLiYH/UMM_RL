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

## Second review — 2026-08-28

Reviewed branch: `origin/agent/T155-exact-oracle` at
`0ba8bf8bc221c104f29821c7963efcf06fd1a22f`.

Decision remains `revision_needed`.

The manifest envelope, detailed-reference serialization, Pareto implementation,
terminology correction, and allowed-path cleanup are materially improved.
However, the following issues remain blocking.

### R7 — replace the failed extended-precision check with a genuinely independent reference

Independent local execution produced:

```text
127 passed, 3 failed
```

Two failures are:

- `test_longdouble_pipeline_is_self_consistent_at_extended_precision[41-4]`;
- `test_longdouble_pipeline_is_self_consistent_at_extended_precision[287-1]`.

For case 41, the long-double relative error is slightly larger than the
float64 error. For case 287 it is materially larger. The current result
therefore does not establish that the original discrepancy is pure float64
cancellation.

Required revision:

1. Construct an independent 80–100 decimal-digit reference for the two fixed
   cases. Do not generate the complete computation in float64 and then merely
   cast the results to a wider dtype.
2. Reconstruct all affine state transitions, sensitivities, quadratic terms,
   and direct losses from exactly represented saved inputs. Python
   `decimal.Decimal.from_float` plus explicit matrix operations is acceptable
   and requires no new dependency; another approach requires local
   authorization.
3. Persist for both cases:
   - exact and direct loss changes;
   - absolute and relative discrepancy;
   - the two cancelling terms;
   - baseline loss magnitude;
   - condition number and trajectory amplification;
   - comparison among float64, platform long-double, and the independent
     high-precision result.
4. Make the high-precision tests pass. If the discrepancy persists at genuine
   high precision, classify it as a formula/implementation mismatch rather
   than numerical cancellation and correct the underlying implementation.

Do not weaken or remove the existing float64 failures.

### R8 — make the independent Pareto cross-check accurate and gate-visible

The active-set solver now produces small KKT residuals in most cases, but the
Frank–Wolfe cross-check is not a high-accuracy reference across the sweep.
Local audit found:

- normalized combined-gradient disagreement up to approximately 7%;
- large absolute Frank–Wolfe KKT residuals;
- cases with poor relative KKT residual.

Required revision:

1. Add a truly independent constrained-QP reference, preferably SciPy
   `minimize`/SLSQP or another already-declared solver, with simplex feasibility
   and scale-normalized KKT diagnostics.
2. Retain Frank–Wolfe as an optional iterative diagnostic if useful, but do not
   describe it as a successful high-accuracy cross-check where it has not
   converged.
3. Define preregistered scale-aware thresholds for:
   - simplex feasibility;
   - nonnegative weights;
   - stationarity/KKT residual;
   - objective gap;
   - combined-gradient discrepancy.
4. Include the independent Pareto checks in each case's `all_passed` result and
   in the formal run Gate. A stored `pareto_reference` without a passing
   independent check is insufficient.
5. Regenerate all run artifacts and report every failing Pareto case.

### R9 — merge the latest authoritative main and rerun all checks

The submitted branch does not contain the current `origin/main` revision.
Merge the latest authoritative main before regeneration. This also incorporates
the local repository-test repair described below.

Resubmission requires:

- repository validator passes;
- complete test suite passes with zero failures;
- high-precision and Pareto checks are represented in run status;
- all reports are consistent with the regenerated evidence.
