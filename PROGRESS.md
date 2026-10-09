# Progress

## Current phase

**Parameter ownership and plasticity is the first-paper mainline.** No complete
real-model joint training run or stable ownership map has been accepted. The
measurement protocol is [the ownership plan](docs/plans/2026-10-09-umm-parameter-ownership.md);
task states are controlled by [tasks/README.md](tasks/README.md) and the task
files.

- T780 is `ready` for CPU-only protocol freeze: Janus-Pro group definitions,
  U/G evaluators and splits, controls, uncertainty rules, and a pilot budget.
- T711 remains `revision_needed`. The submitted CoRL branch contains reports
  and a QA-dispatch patch, but it has not supplied committed A–E implementation,
  executable tests, and complete evidence for semantic admission. Its report
  marks some checks confirmed despite failed or incomplete probes. The
  fixed-anchor finite-response check has moved to conditional T756.
- T712, T725, T781, T740, T782, and T783 are planned gates. T781 is the first
  parameter-intervention pilot; no ownership-guided training is authorized.
- T720 is accepted reusable Janus-Pro-R1 engineering evidence. T750 is accepted
  only for mock instrumentation. T250/T260 retain accepted model/data audits.
- T710 is stopped as a separate smoke task; its semantic defects are carried
  forward to T711. Older finite-response theory, Show-o2 experiments, D0/E1–E3,
  and CoRL negotiator tasks are P2 conditional work, not first-paper gates.

## Mainline decision sequence

1. Finish T780 without GPU execution. Freeze complete shared groups, U/G
   evaluation definitions, disjoint calibration and held-out IDs, zero-gate
   restoration tests, practical-effect thresholds, random/conventional
   controls, and resource limits.
2. Repair T711 checks A–E. A valid admission needs causal image scoring,
   declared behavior/scoring semantics, mixed QA and generation rewards,
   padding and reduction tests, actual parameter updates, and full-state
   resume evidence. A `confirmed` JSON field alone is insufficient.
3. After local acceptance and a frozen resource manifest, execute T712, then
   T725. Establish U-only/G-only controls through T740 before training claims.
4. Open T781 only after T780 and T711 pass. Advance to T782/T783 only if the
   ownership map is stable across splits and predicts nontrivial U/G variation
   under the frozen controls.
5. Use T784 only after held-out first-model evidence supports the mechanism.
   Open optimizer-response work only after a measured interference pattern
   survives simpler restricted-update controls and a separate budget gate.

## Conditional optimizer decision gates

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

## Historical T1a executable gate

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

## Current authorized work

T780 permits planning and source inspection without GPU use. T711 permits a
bounded repair submission under its revised A–E contract; it does not permit
full training. All new GPU tasks remain `planned` until local review freezes
budgets, first reports, dependencies, and acceptance criteria.
