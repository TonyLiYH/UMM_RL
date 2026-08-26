# Progress

## Current phase

**Phase T1b + model admission preparation.** No real-model training has started.

Authoritative task status: [tasks/README.md](tasks/README.md).

Current authorized remote entries:

- T110 random overlap families;
- T120 independent KKT/direct reference;
- T130 indefinite-curvature trust-region tests;
- T210 Show-o2 admission audit and smoke.

These tasks do not authorize joint post-training. T300 remains closed until T100 and T210 are accepted.

## Decision gates

| Gate | Question | Pass condition | Status |
|---|---|---|---|
| T0 | Is the formulation technically coherent and distinguishable from prior work? | Adversarial review reaches `almost` or `ready`, with unresolved assumptions named | Passed with open items: 6/10 almost |
| T1a | Are the deterministic algebraic contracts executable? | Exact elimination, selector, attainable-gain, rescaling, certificate, and rejection smoke tests pass | Passed 2026-08-24 |
| T1b | Do solver/approximations match independent synthetic references? | Random overlap families, direct KKT, trust acceptance, CG/unroll/diagonal/low-rank error tests pass | In progress |
| D0 | Do compensation-aware diagnostics predict realized changes? | Better held-out calibration than the strongest raw first-/second-order Taylor baseline; target Spearman at least `0.5` | Not started |
| E0 | Can the public checkpoint/evaluation path be reproduced? | Key public metrics within a preregistered tolerance and no hidden training dependency | Not started |
| E1 | Does CompPareto beat strong baselines on Show-o2? | Positive worst-task gain with 95% CI and higher hypervolume than tuned scalarization | Not started |
| E2 | Does the effect transfer across architectures? | Positive worst-task result on at least two model families | Not started |
| E3 | Does it survive heterogeneous preference/RL updates? | Non-inferior stability with positive worst-task gain | Not started |

Numerical thresholds in D0/E0/E1 will be frozen after T1b and a variance-only dry run, before method comparisons.

## T1a minimum executable gate

The repository must provide the following before T0 can move beyond `In progress`:

| Deliverable | Path | Check |
|---|---|---|
| Synthetic quadratic object and exact private elimination | `src/comppareto/quadratic.py` | Direct and Schur-minimized values agree |
| Partial-overlap block lift | `src/comppareto/quadratic.py` | Shape and single-lift tests |
| Two-task common-descent solver | `src/comppareto/quadratic.py` | Directional derivatives satisfy the certificate |
| Deterministic T1 manifest | `src/comppareto/synthetic.py` | Records tolerance and all pass/fail checks |
| Frozen synthetic configuration | `configs/t1_synthetic.json` | Seed and tolerance committed |
| Numerical tests | `tests/` | Rescaling, certificate, exact elimination, indefinite rejection |
| One-command gate | `scripts/run_t1.sh` | Runs tests and emits manifest |

Command: `bash scripts/run_t1.sh`. A nonzero exit blocks T1a.

Latest evidence: `runs/t1_synthetic/t1_manifest.json`. This validates deterministic algebraic contracts only, not the complete T1b solver gate or any real-model hypothesis.

## Immediate next actions

1. Execute T110, T120, and T130 under the task contracts.
2. Execute T210 as an admission audit/smoke without joint training.
3. Open T140/T150 only after their declared prerequisites are accepted.
4. Accept T100 and T210 before changing T300 to `ready`.
