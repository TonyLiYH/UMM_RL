# T1a Synthetic Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement each behavior test-first.

**Goal:** Deliver a one-command algebraic smoke test of the corrected Schur complement, legal block selection, trust-region attainable gain, conditional negotiation rescaling, deterministic common descent, and invalid private-curvature rejection.

**Architecture:** A small NumPy package defines validated quadratic task objects and pure numerical functions. A CLI runs deterministic cases and emits a JSON manifest. Pytest tests the mathematical contracts independently of the CLI.

**Tech Stack:** Python 3.11+, NumPy, pytest, standard-library JSON/argparse.

---

## File map

- `pyproject.toml`: install and test configuration.
- `src/comppareto/quadratic.py`: quadratic task validation, private response, Schur complement, block lift, and common-descent direction.
- `src/comppareto/synthetic.py`: deterministic T1 suite and JSON output.
- `tests/test_quadratic.py`: unit and numerical property tests.
- `tests/test_synthetic_cli.py`: executable manifest smoke test.
- `configs/t1_synthetic.json`: frozen T1 tolerances and seed.
- `scripts/run_t1.sh`: one-command local gate.

## TDD sequence

1. Write failing tests for exact private elimination and direct minimization equality.
2. Implement the minimum validated quadratic object and Schur functions.
3. Write failing tests for local-to-global block lifting and shape errors.
4. Implement single-lift behavior.
5. Write failing tests for deterministic common descent and Pareto-stationary zero case.
6. Implement the two-task convex-hull projection solver.
7. Write failing tests for conditional loss rescaling and indefinite private curvature rejection.
8. Implement validation and the normalized retained-gain helper.
9. Write a failing CLI smoke test requiring a JSON manifest with all T1 checks.
10. Implement the deterministic CLI and one-command shell gate.

## Acceptance

`bash scripts/run_t1.sh` must run all tests and create a manifest in a caller-specified temporary output directory. This passes T1a only. Random overlap families, direct KKT comparison, overall-indefinite trust-region acceptance, and CG/unroll/diagonal/low-rank error curves remain the separate T1b gate in `research-plan.md`.
