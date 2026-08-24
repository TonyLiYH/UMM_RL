# Decision 0002: use a research-first standalone repository

- Date: 2026-08-24
- Status: accepted

## Decision

CompPareto is maintained as a standalone Git repository. Theory, literature, decisions, plans, future implementation, and run metadata live in one auditable project tree, but large artifacts remain external.

## References considered

- [Show-o](https://github.com/showlab/Show-o) exposes configurations, distributed launch configuration, model code, training, validation assets, and a public roadmap.
- [UniAR](https://github.com/ShareLab-SII/UniAR) separates training, evaluation conversion, documentation, inference, and reward-server guidance.
- [SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1) uses a package-oriented `src/` layout with examples and documentation suitable for multiple model capabilities.
- [Janus](https://github.com/deepseek-ai/Janus) keeps a compact installable model package plus demos and inference entry points.

## Local adaptation

Research decisions and review evidence are first-class because the project is not yet an implementation release. Therefore `docs/theory`, `docs/surveys`, `docs/plans`, `docs/decisions`, `PROGRESS.md`, and `review-stage/` are present before training code.

## Consequences

- The top-level README stays public-facing and does not become a lab notebook.
- Plans remain separate from factual progress and run evidence.
- `src/comppareto/`, `configs/`, `experiments/`, and `tests/` are reserved with explicit responsibility documents rather than placeholder code.
- No remote is created or pushed without explicit authorization.

