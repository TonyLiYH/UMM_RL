# CompPareto

CompPareto is a research project on **compensation-aware Pareto post-training for unified multimodal models**. The first validation scope is joint post-training for image understanding and image generation; no pretraining is planned.

The central question is not how to add two losses. It is:

> How should shared parameters be updated when tasks use heterogeneous objectives and optimizers, touch only partially overlapping parameter blocks, and retain private components that can compensate for a shared update?

## Status

The repository is in the theory-and-experiment-design stage. It contains no claim of empirical improvement yet. The current method name is provisional.

- Research charter: [PROJECT.md](PROJECT.md)
- Current status and gates: [PROGRESS.md](PROGRESS.md)
- Mathematical formulation: [docs/theory/formulation.md](docs/theory/formulation.md)
- Related-work and novelty audit: [docs/surveys/related-work.md](docs/surveys/related-work.md)
- Staged experiment plan: [docs/plans/research-plan.md](docs/plans/research-plan.md)
- Experiment matrix: [docs/plans/experiment-matrix.md](docs/plans/experiment-matrix.md)
- Repository layout rationale: [docs/decisions/0002-repository-layout.md](docs/decisions/0002-repository-layout.md)
- Adversarial review record: [review-stage/AUTO_REVIEW.md](review-stage/AUTO_REVIEW.md)

## Research hypothesis

Raw shared gradients can overstate or mischaracterize task conflict because they ignore the response of task-private components. CompPareto instead optimizes a post-compensation value function and uses a conditionally loss-scale-invariant retained-gain objective to negotiate a shared update.

The project is falsified in its current form if compensation-aware quantities do not predict realized joint loss changes better than raw-gradient diagnostics, or if a carefully tuned scalarization matches the proposed method across the preregistered model suite.

## Collaboration model

This repository keeps the complete project under version control:

- research notes, theory, surveys, plans, and decisions;
- the authoritative local/remote Agent task tree;
- executable code, tests, scripts, and configurations;
- lightweight run provenance, result reports, and review evidence.

Large datasets, model weights, checkpoints, generated media, and raw logs remain in remote storage and are referenced by canonical path/URI, hash, size, producing run, and source revision.

The local side publishes authorized tasks on `main`. A remote GPU executor works on the declared `agent/<task-id>-<slug>` branch, submits its task as `awaiting_review`, and pushes the branch. The local review side verifies the code and evidence, then returns `revision_needed`, records a blocker, or merges and marks the task `accepted`.

Remote executors never push directly to `main` and never mark their own result `accepted`.

## Repository layout

```text
CompPareto/
├── tasks/               # Authoritative task tree and task contracts
├── worklog/             # Process, observations, unresolved questions
├── reports/             # Result summaries, claim checks, failure ledgers
├── configs/             # Reproducible experiment configurations
├── docs/
│   ├── decisions/       # Durable scope and design decisions
│   ├── plans/           # Research and experiment plans
│   ├── surveys/         # Literature and novelty audits
│   └── theory/          # Definitions, theorem targets, derivations
├── experiments/         # Experiment manifests and launch documentation
├── review-stage/        # Raw adversarial reviews and repair log
├── runs/                # Sanitized metadata for formal runs
├── src/comppareto/      # Implementation and repository validators
└── tests/               # Numerical, repository, and adapter tests
```

The layout follows the public-facing organization of research repositories such as [Show-o](https://github.com/showlab/Show-o), [UniAR](https://github.com/ShareLab-SII/UniAR), [SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1), and [Janus](https://github.com/deepseek-ai/Janus), while keeping theory, plans, decisions, and review evidence explicit during the pre-experiment stage.

## Reproducibility policy

- Every formal run will have a committed sanitized configuration and run note under `runs/<run_id>/`.
- Checkpoints, datasets, generated media, secrets, and machine-specific paths are never committed.
- Baseline tuning budgets and stopping rules are fixed before inspecting final test metrics.
- Negative results and failed gates remain in `PROGRESS.md` and the run ledger.
- A license is intentionally not selected before publication and organizational IP requirements are decided.

## Agent entry points

- [Authoritative task tree](tasks/README.md)
- [Project progress](PROGRESS.md)
- [Factual changelog](CHANGELOG.md)
- [Worklog](worklog/README.md)
- [Result-report contract](reports/README.md)
- [Agent collaboration design](docs/superpowers/specs/2026-08-26-agent-task-tree-collaboration-design.md)

Remote executors start only from tasks marked `ready`, use the exact branch named by the task, and return the required first report before expensive GPU work.

Validate repository control state with:

```bash
.venv/bin/python -m comppareto.repo_state.cli --root .
```

## T1a synthetic smoke gate

After installing the package with test dependencies, run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
bash scripts/run_t1.sh
```

This validates only deterministic algebraic contracts. It does not pass the full T1b solver/approximation gate and does not constitute evidence that CompPareto improves a unified multimodal model.
