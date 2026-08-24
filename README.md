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

## Planned repository layout

```text
CompPareto/
├── configs/             # Reproducible experiment configurations
├── docs/
│   ├── decisions/       # Durable scope and design decisions
│   ├── plans/           # Research and experiment plans
│   ├── surveys/         # Literature and novelty audits
│   └── theory/          # Definitions, theorem targets, derivations
├── experiments/         # Experiment manifests and launch documentation
├── review-stage/        # Raw adversarial reviews and repair log
├── runs/                # Sanitized metadata for formal runs
├── src/comppareto/      # Future implementation
└── tests/               # Future unit and numerical tests
```

The layout follows the public-facing organization of research repositories such as [Show-o](https://github.com/showlab/Show-o), [UniAR](https://github.com/ShareLab-SII/UniAR), [SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1), and [Janus](https://github.com/deepseek-ai/Janus), while keeping theory, plans, decisions, and review evidence explicit during the pre-experiment stage.

## Reproducibility policy

- Every formal run will have a committed sanitized configuration and run note under `runs/<run_id>/`.
- Checkpoints, datasets, generated media, secrets, and machine-specific paths are never committed.
- Baseline tuning budgets and stopping rules are fixed before inspecting final test metrics.
- Negative results and failed gates remain in `PROGRESS.md` and the run ledger.
- A license is intentionally not selected before publication and organizational IP requirements are decided.

## T1a synthetic smoke gate

After installing the package with test dependencies, run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
bash scripts/run_t1.sh
```

This validates only deterministic algebraic contracts. It does not pass the full T1b solver/approximation gate and does not constitute evidence that CompPareto improves a unified multimodal model.
