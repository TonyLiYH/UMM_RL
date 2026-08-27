# Changelog

## 2026-08-27

- Added a research-state handoff for subsequent local and remote agents.
- Recorded the data-aware finite-response robust Pareto certificate as an open theory target.
- Corrected the loss-scale-invariance proof so attainable-gain scaling follows from its definition under a fixed trust set.
- Added verified mathematical prior work and the exact-oracle benchmark rationale.
- Preserved the existing T110/T120/T130/T210 execution authorization and all downstream Gates.
- [exp][T155] Implemented the exact finite-response oracle core (`src/comppareto/oracle/`: selectors, tasks, noise, sgd, momentum, hypergradient, crosscheck, generation, seeds, stability, case, manifest, sweep) matching `docs/theory/oracle-spec.md` section 1-11. Added 49 unit tests (`tests/oracle/`), all passing.
- [exp][T155] Ran the resolved section-8 baseline sweep (`configs/oracle/baseline.yaml`, seed `20260827155`) at `runs/oracle-20260827-baseline/`: 288 cases (6 families x 2 optimizers x 4 horizons x 2 stability regimes x 3 seeds), 286 passed (99.3%), 2 failed. Elapsed 9.26s single-core; manifest 2.7MB; both within the section-10 CPU/memory/size estimate, no throttling needed. The 6-case detailed-trajectory subset achieves full family/optimizer/horizon/regime coverage.
- [FAIL][T155] 2/288 cases (case_index 41: disjoint/momentum/K=5/unstable; case_index 287: random_sparse/momentum/K=10/unstable) fail the `loss_change` tolerance (1e-9) at 2.7e-9 and 1.3e-8 respectively, while `state`/`hypergradient` pass at 1e-15/1e-16 and all required finite-difference steps pass. Root cause: deliberately-unstable spectral radius (~1.64-1.67) compounded over K=5-10 amplifies floating-point cancellation in the exact quadratic loss-change identity. Per the frozen protocol, tolerance was not relaxed and both seeds remain in the failure ledger (`runs/oracle-20260827-baseline/failure_ledger.json`) for local-reviewer decision. Details in `runs/oracle-20260827-baseline/notes.md`.
- [CORRECTION][T155] The `docs/theory/oracle-spec.md` section 4 sensitivity closed form had the matrix multiplication order reversed (`H_phix(I-M^K)C^-1` instead of `(I-M^K)C^-1 H_phix` — both factors commute with `M^K` individually but `H_phix` does not commute with `C^-1` in general); fixed before implementation, caught by the closed-form-vs-unroll cross-check while writing `src/comppareto/oracle/sgd.py`.

This file records factual repository and experiment history. Research interpretation belongs in `PROGRESS.md`; individual task status belongs in `tasks/`.

## 2026-08-26

- Initialized the UMM_RL local-planner / remote-GPU-executor collaboration design.
- Added the authoritative task tree, task-state contract, result-report structure, and worklog structure.
- Added JSON Schemas and Python validation for task graphs and formal run provenance.
- Added a combined repository-state CLI, pull-request contract, and GitHub Actions validation workflow.
- Verified 24 task nodes, one committed legacy T1a manifest, and 27 local tests.
- Narrowed CI whitespace enforcement to real line-end/tab issues; harmless extra blank lines at end of file no longer fail a push.
- Preserved the existing CompPareto theory, T1a implementation, tests, and review evidence.
