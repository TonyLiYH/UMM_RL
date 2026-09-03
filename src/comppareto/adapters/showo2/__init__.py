"""Show-o2 finite-response diagnostic adapter (T215).

This package implements the reversible snapshot/restore, raw/commit/rerun
gradient protocols, and central finite-difference reference described in
``reports/T215/first-report.md`` and the frozen protocol
``docs/plans/showo2-first-attempt.md``.

Submodules:

- :mod:`comppareto.adapters.showo2.state` -- deterministic snapshot/restore
  of parameters, AdamW optimizer state, and RNG state. Pure PyTorch, no GPU
  or real model required.
- :mod:`comppareto.adapters.showo2.protocols` -- raw / commit-response
  (stop-gradient) / rerun-response (exact finite unroll) gradient
  computation, generic over any differentiable ``loss_fn``. Pure PyTorch.
- :mod:`comppareto.adapters.showo2.finite_diff` -- central finite-difference
  reference (4 directions) and T215's own tolerance gate. Pure PyTorch.
- :mod:`comppareto.adapters.showo2.model_io` -- import-guarded loading of the
  real official Show-o2 model and construction of the MMU/T2I losses. Only
  imported when the CLI actually runs against the real checkpoint; not
  imported by the unit tests in ``tests/adapters/showo2/``.
- :mod:`comppareto.adapters.showo2.run_feasibility` -- CLI entry point that
  orchestrates the full K=1 (then conditional K=3) diagnostic for both task
  paths and writes the run's manifest/metrics/notes artifacts.
"""
