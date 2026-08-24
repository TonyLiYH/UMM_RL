# AGENTS — CompPareto repository rules

This repository is the source of truth for the CompPareto research project: theory, experiment plans, implementation, configurations, and factual run evidence belong here.

## Before editing

1. Read `PROJECT.md`, `PROGRESS.md`, and the relevant document under `docs/`.
2. Run `git status` and preserve unrelated user changes.
3. Distinguish established results, theorem targets, hypotheses, and implementation proposals.

## Research integrity

- Do not describe a theorem target as proven until assumptions and proof are complete.
- Do not report an empirical gain without a committed run configuration and result source.
- Keep tuned scalarization as a strong baseline with the same search budget as CompPareto.
- Report all preregistered seeds and capability slices; do not select only favorable checkpoints.
- A failed diagnostic or experiment gate must be recorded, not silently bypassed.

## Code and experiment rules

- Import future code as `comppareto`.
- Use Python 3.11+, type annotations, pytest, and configuration-driven experiments.
- Model adapters must expose shared and private parameter blocks explicitly.
- GPU-required commands must fail if CUDA is unavailable; never silently fall back to CPU.
- Use environment variables for dataset, model, output, and cache roots.
- Formal run metadata belongs in `runs/<run_id>/`; large artifacts stay outside Git.

## Completion checks

For documentation changes, run link/path checks and `git diff --check`. For code changes, additionally run unit tests, numerical gradient checks, and Python compilation.

