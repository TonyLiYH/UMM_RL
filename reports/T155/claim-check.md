# T155 claim check

Each row maps a claim or pass/fail-gate bullet from `tasks/T155-exact-finite-response-oracle.md` to the evidence that supports or refutes it.

## Research claim

> "A small structured shared/private multi-task system can provide exact finite-response trajectories, hypergradients, realized loss changes, and high-accuracy Pareto references without claiming to simulate the full behavior of a unified multimodal model."

Supported. `src/comppareto/oracle/` implements the trajectories (`sgd.py`, `momentum.py`), hypergradients (`hypergradient.py`), and realized loss changes (`crosscheck.py::exact_loss_change`) exactly in closed form for the linear-quadratic system defined in `docs/theory/oracle-spec.md`. All three are independently cross-checked (state vs. unroll, hypergradient vs. independently implemented reverse-mode differentiation, loss-change vs. direct evaluation) and pass at machine precision in 286/288 generated cases. No claim of UMM realism is made anywhere in `docs/theory/oracle-spec.md` or the code; section 1 of the spec explicitly frames this as a mechanism-validation benchmark, matching the frozen protocol's own language. No automatic-differentiation library is used anywhere in this implementation (none is available under `pyproject.toml`); every reverse-mode gradient is a hand-coded backward pass over the literal unroll (see §7a of the spec).

## Pass/fail gate bullets

| Gate bullet | Status | Evidence |
|---|---|---|
| Analytic finite-response state matches independently unrolled state, rel <=1e-10 | **Pass**, all 288 cases | `tests/oracle/test_sgd.py`, `tests/oracle/test_momentum.py` (unit level); `manifest.json` `state` check field, every case, typical error ~1e-15/1e-16 |
| Analytic hypergradient matches independently implemented reverse-mode differentiation, rel <=1e-9 (or abs <=1e-11 near zero) | **Pass**, all 288 cases | `tests/oracle/test_sgd.py::test_sgd_hypergradient_analytic_matches_reverse_mode` and `tests/oracle/test_momentum.py::test_momentum_hypergradient_analytic_matches_reverse_mode`; `manifest.json` `hypergradient` check field, every case |
| Central finite-difference directional derivatives match within the preregistered step-size envelope | **Pass**, all required steps in h in [1e-6,1e-2], all cases | `tests/oracle/test_crosscheck_tolerances.py`; `manifest.json` `finite_difference` check field |
| Direct rerun-response loss changes match the oracle evaluation | **Pass on 286/288 cases; fails on case_index 41 and 287** | `manifest.json` `loss_change` check field; see `failure-ledger.md` |
| Selectors satisfy the single-lift and no-duplicate-coordinate contract | **Pass**, all 288 cases | `src/comppareto/oracle/selectors.py::validate_selector` (raises `SelectorError` on violation, never raised across the sweep); `tests/oracle/test_selectors.py` |
| All failed or unstable seeds remain in the ledger | **Pass** | `runs/oracle-20260827-baseline/failure_ledger.json` contains both failing cases; `manifest.json` retains all 288 records including the 2 with `all_passed: false`; no seed removed or hidden |
| Full-overlap and disjoint regimes behave as declared boundary controls | **Pass** | `configs/oracle/baseline.yaml` realizes `full_overlap` at `(num_tasks=2, num_blocks=4, block_width=1)` (maximal per-task overlap given m=2) and `disjoint` at `(num_tasks=8, num_blocks=64, block_width=2)` (zero shared blocks by construction); `src/comppareto/oracle/generation.py`'s family-specific incidence builder is asserted against each family's defining property in `tests/oracle/test_generation.py::test_build_incidence_matches_family_property`, run for all 6 families including these two |

## Frozen-protocol constraints

| Constraint | Status | Evidence |
|---|---|---|
| Do not remove hard seeds | **Honored** | `seeds.py` derives seeds deterministically from `config_seed`; no seed exclusion logic exists anywhere in `sweep.py` or `case.py` |
| Do not tune generation ranges after inspecting method performance | **Honored** | `configs/oracle/baseline.yaml` was written and committed as the resolved range selection before the full 288-case sweep was run; the smoke test that preceded the full sweep used a shrunk case count with the same ranges, not narrowed ranges |
| Do not describe the oracle as a realistic UMM simulator | **Honored** | No such claim appears in `docs/theory/oracle-spec.md`, `src/comppareto/oracle/`, `runs/oracle-20260827-baseline/notes.md`, or this report |
| Numerical tolerance changes require local review | **Honored** | The 1e-9 `loss_change` tolerance was not changed despite 2 observed failures; both are left for local-reviewer decision (see `failure-ledger.md`) |
| Infrastructure retries preserve the same configuration and seed | **N/A this run** | No infrastructure retry occurred; the sweep ran to completion on the first attempt with no process failure |

## Deliverables checklist (task file "Required deliverables")

| Deliverable | Path |
|---|---|
| Oracle specification | `docs/theory/oracle-spec.md` |
| Typed implementation | `src/comppareto/oracle/*.py` |
| Analytic and numerical references | `src/comppareto/oracle/crosscheck.py`, `hypergradient.py`, `sgd.py`/`momentum.py` (independently implemented reverse-mode differentiation path), `pareto.py` (independent common-descent/Pareto reference), `highprecision.py` (extended-precision recheck) |
| Unit/property tests | `tests/oracle/` (100 tests) |
| Resolved configurations | `configs/oracle/baseline.yaml` |
| Deterministic run manifest | `runs/oracle-20260827-baseline/manifest.json` (schema envelope), `case-records.json` (per-case array) |
| Per-seed result table | `runs/oracle-20260827-baseline/case-records.json` (one record per case_index/seed) |
| Boundary and failure cases | full-overlap/disjoint families (see above); `runs/oracle-20260827-baseline/failure_ledger.json` |
| Summary report | this `reports/T155/` directory |
| Failure ledger | `runs/oracle-20260827-baseline/failure_ledger.json`, `reports/T155/failure-ledger.md` |

## Conclusion

**Supports gate.** Every pass/fail bullet is satisfied except the direct-loss-change-match bullet, which fails on exactly 2 of 288 cases for a documented, non-code-bug reason, with both cases correctly retained in the ledger per the "all failed or unstable seeds remain in the ledger" bullet (which is itself satisfied). This is flagged for local-reviewer decision rather than resolved unilaterally, per "numerical tolerance changes require local review."
