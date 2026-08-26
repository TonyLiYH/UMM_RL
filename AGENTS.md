# AGENTS — CompPareto repository rules

This repository is the source of truth for the CompPareto / UMM_RL research project: theory, task control, experiment plans, implementation, configurations, and factual run evidence belong here.

## Before editing

1. Read `PROJECT.md`, `PROGRESS.md`, `tasks/README.md`, and the active task file.
2. Run `git status` and preserve unrelated user changes.
3. Distinguish established results, theorem targets, hypotheses, and implementation proposals.
4. Confirm the current branch and the task's `source_revision`, dependencies, and `allowed_paths`.

## Authority model

### Local planning and review

- Creates tasks, dependencies, budgets, Gates, and successor authorization.
- Owns research claims, `PROJECT.md`, `PROGRESS.md`, and decision records.
- Reviews remote branches and is the only side allowed to set `accepted`, `revision_needed`, or `stopped`.
- Pushes the authoritative `main`.

### Remote execution

- Starts only from a task marked `ready`.
- Uses the exact `agent/<task-id>-<slug>` branch declared by the task.
- May modify only declared `allowed_paths`.
- May set its branch task state to `running`, `awaiting_review`, or `blocked`.
- Must report failures, retries, artifact locations, hashes, costs, and exact blockers.
- Must not push directly to `main`, alter frozen protocols, open successor tasks, or mark work `accepted`.

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
- Formal task status belongs in `tasks/<task-id>-<slug>.md`; `PROGRESS.md` is only a milestone summary.
- Remote GPU tasks require the task's first report before expensive execution.
- Large artifacts require canonical references, hashes, measured sizes, producer run IDs, and source revisions.

## Completion checks

Run the repository validator, unit tests, Python compilation, shell syntax checks, local-link checks, and `git diff --check`. A remote result remains `awaiting_review` until the local review side independently verifies it.
