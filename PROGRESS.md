# Progress

## Current phase

**Phase T1b + model admission preparation.** No real-model training has started.

The 2026-08-27 theory/data/model audit is captured in [the research-state handoff](docs/handoffs/2026-08-27-research-state.md). It keeps the current execution Gates unchanged while recording the next recommended theory target: a data-aware, finite-response robust Pareto certificate with explicit inner, curvature, sampling, and distribution-shift errors.

The follow-up mathematical audit is recorded in [the theory breakthrough audit](docs/theory/2026-08-27-theory-breakthrough-audit.md). It narrows the preferred contribution to a finite-horizon optimizer-state posterior error bound, graph-localized simultaneous descent certificate, and certificate-width-optimal sample/compute allocation. It also separates rerun-response finite unrolling from commit-response stop-gradient semantics.

Authoritative task status: [tasks/README.md](tasks/README.md).

Current authorized remote entries:

- T110 random overlap families;
- T120 independent KKT/direct reference;
- T130 indefinite-curvature trust-region tests;
- T155 exact finite-response oracle benchmark — accepted 2026-09-01;
- T210 Show-o2 admission audit and smoke — accepted with recorded limitations 2026-09-01;
- T215 Show-o2 reversible finite-response diagnostics — ready;
- T220 UniDDT admission — ready;
- T230 SenseNova-U1 admission and routed-overlap audit — ready;
- T240 UniAR boundary-control admission — ready.

These tasks do not authorize joint post-training. T300 remains closed until
T100, T170, T210, and T215 are accepted.

The first Show-o2 execution plan is recorded in
[`docs/plans/showo2-first-attempt.md`](docs/plans/showo2-first-attempt.md).
T215 is now authorized for reversible finite-response diagnostics. T220–T240
are authorized as independent model-admission tasks. None authorizes persistent
joint training.

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

1. Continue T110, T120, and T130 under their task contracts.
2. Execute T215 on the accepted Show-o2 revision under the reversible,
   no-persistent-update protocol.
3. Execute T220, T230, and T240 independently under the common admission
   contract and local-SSD preflight.
4. Open T140/T150 only after their declared prerequisites are accepted.
5. Accept T100, T170, and T215 before changing T300 to `ready`.
6. Complete the planned T160 finite-response posterior-certificate and T170
   graph-localized robust-certificate contracts after their dependencies are
   accepted; neither task is authorized yet.
7. Freeze data-source, capability, split, and decontamination manifests before
   D0; do not attribute a data-scheduling gain to update geometry.
