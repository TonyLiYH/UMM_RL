# UMM_RL Agent Task Tree and Collaboration Design

- Date: 2026-08-26
- Status: approved concept, implementation pending written-spec review
- Local repository: `CompPareto`
- GitHub repository: `git@github.com:alexlovecoding/UMM_RL.git`
- Python package: `comppareto`

## 1. Purpose

This repository is the shared control plane and implementation repository for the unified multimodal post-training project.

It keeps four kinds of versioned knowledge together:

1. research knowledge: problem statements, literature, theory, decisions, and experiment plans;
2. task control: the authoritative task tree, dependencies, gates, assignments, and review status;
3. executable assets: code, tests, scripts, and portable configurations;
4. lightweight evidence: run manifests, result summaries, failure ledgers, and review records.

Large datasets, model weights, checkpoints, generated images/videos, raw logs, and other expensive artifacts remain in remote storage. The repository records their canonical URI or path, hash, size, producing run, and relevant revision.

## 2. Collaboration model

The repository uses a local-planner / remote-executor / local-reviewer workflow.

### Local planning and review side

The local side owns:

- research claims and scope;
- task creation, decomposition, priority, and dependency structure;
- frozen model, dataset, metric, budget, and success criteria;
- authorization to start a GPU task;
- result and code review;
- Gate Pass/Fail decisions;
- opening successor tasks;
- the final `accepted` and `stopped` states.

### Remote GPU execution side

The remote side may:

- execute an authorized task on an `agent/<task-id>-<slug>` branch;
- modify files within the task's declared path scope;
- implement code, tests, scripts, and configurations;
- run CPU/GPU experiments;
- commit lightweight run metadata and result reports;
- report exact blockers and failed experiments;
- move its branch copy of a task to `running`, `awaiting_review`, or `blocked`.

The remote side may not:

- alter the research claim, model family, data split, metric, budget, or threshold without a new local decision;
- weaken a Gate;
- omit failed runs or use an unregistered replacement experiment;
- mark its own task `accepted`;
- open a dependent GPU task before the local review side accepts its prerequisite;
- push directly to `main`.

## 3. Source-of-truth hierarchy

The repository uses one source of truth per concern:

| Concern | Authoritative location |
|---|---|
| Research problem, hypotheses, scope | `PROJECT.md` |
| Project-level milestone interpretation | `PROGRESS.md` |
| Individual task state and acceptance | `tasks/<task-id>-<slug>.md` |
| Task relationships and current entry points | `tasks/README.md` |
| Frozen experiment design | `docs/plans/` |
| Durable research decisions | `docs/decisions/` |
| Factual code/experiment history | `CHANGELOG.md` |
| Per-run configuration and provenance | `runs/<run-id>/manifest.json` |
| Result interpretation submitted for review | `reports/<task-id>/` |
| Adversarial or formal review evidence | `review-stage/` |
| Execution process and observations | `worklog/YYYY-MM/` |

GitHub Issues and Projects may later mirror active tasks for notification and visualization, but they are not authoritative.

## 4. Repository layout

The implementation preserves the existing `docs/` history and adds the missing collaboration layers.

```text
UMM_RL/
├── README.md
├── PROJECT.md
├── PROGRESS.md
├── CHANGELOG.md
├── AGENTS.md
│
├── tasks/
│   ├── README.md
│   ├── T000-root-research.md
│   ├── T100-t1b-validation.md
│   ├── T110-overlap-family.md
│   ├── T120-independent-kkt-reference.md
│   ├── T130-indefinite-trust-region.md
│   ├── T140-approximation-error.md
│   ├── T150-negotiation-audit.md
│   ├── T200-model-admission.md
│   ├── T210-showo2-admission.md
│   ├── T220-uniddt-admission.md
│   ├── T230-sensenova-u1-admission.md
│   ├── T240-uniar-admission.md
│   ├── T300-d0-conflict-diagnostics.md
│   ├── T310-parameter-block-registry.md
│   ├── T320-hypergradient-cache.md
│   ├── T330-predictor-comparison.md
│   ├── T340-calibration-audit.md
│   ├── T400-e1-showo2-pilot.md
│   ├── T410-budget-search-freeze.md
│   ├── T420-strong-baseline-wave.md
│   ├── T430-comppareto-wave.md
│   ├── T440-confirmatory-evaluation.md
│   ├── T500-e2-architecture-transfer.md
│   ├── T600-e3-heterogeneous-posttraining.md
│   └── archive/
│
├── docs/
│   ├── decisions/
│   ├── plans/
│   ├── surveys/
│   ├── theory/
│   └── superpowers/
│
├── worklog/
│   ├── README.md
│   └── YYYY-MM/
│
├── reports/
│   ├── README.md
│   └── <task-id>/
│       ├── result-summary.md
│       ├── claim-check.md
│       └── failure-ledger.md
│
├── configs/
├── experiments/
├── scripts/
├── src/comppareto/
├── tests/
│
├── runs/
│   └── <run-id>/
│       ├── manifest.json
│       ├── resolved-config.yaml
│       └── notes.md
│
├── review-stage/
└── .github/
    ├── pull_request_template.md
    └── workflows/
        └── validate-research-state.yml
```

## 5. Initial research task tree

The initial tree reflects the accepted review outcome and prevents real-model training from bypassing T1b.

```text
T000  CompPareto / UMM_RL unified multimodal post-training research [root]
├── T100  T1b independent solver and approximation validation
│   ├── T110  Random disjoint / partial / full overlap families
│   ├── T120  Independent KKT and direct-solver reference
│   ├── T130  Overall-indefinite curvature and trust-region rejection
│   ├── T140  CG / unroll / diagonal / low-rank error curves
│   └── T150  Negotiation feasibility, KKT, and reference audit
├── T200  Public model admission and adapter audit
│   ├── T210  Show-o2 admission
│   ├── T220  UniDDT admission
│   ├── T230  SenseNova-U1 admission
│   └── T240  UniAR boundary-control admission
├── T300  D0 compensation-aware conflict diagnostics
│   ├── T310  Shared/private block registry
│   ├── T320  Identical-A_i^K hypergradient cache
│   ├── T330  Raw Taylor and compensated predictors
│   └── T340  Held-out calibration and certificate audit
├── T400  E1 Show-o2 controlled pilot
│   ├── T410  Budget and search freeze
│   ├── T420  Strong baseline wave
│   ├── T430  CompPareto estimator/negotiation wave
│   └── T440  Confirmatory seeds and capability slices
├── T500  E2 cross-architecture validation
└── T600  E3 DPO / OPD plus Flow-DPO / GRPO validation
```

Dependency rules:

- T200 may perform read-only admission audits while T100 is running.
- T300 implementation starts only after T100 is accepted and at least T210 is accepted.
- T400 starts only after T300 is accepted.
- T500 and T600 start only after T400 is accepted.
- A failed Gate changes the affected branch to `stopped` and prevents automatic opening of descendants.

## 6. Task-file contract

Each task uses machine-readable YAML front matter followed by the complete research and execution contract.

```yaml
---
id: T110
title: Random overlap quadratic families
parent: T100
status: ready
priority: P0
owner: remote-gpu-agent
reviewer: local-research-agent
branch: agent/T110-overlap-family
depends_on: []
blocks: [T120, T140]
allowed_paths:
  - src/comppareto/
  - tests/
  - configs/
  - runs/t1b-*/
  - reports/T110/
source_revision: "<main revision published with the task>"
created_at: 2026-08-26
updated_at: 2026-08-26
---
```

The body must contain:

1. the research claim served by the task;
2. the exact problem and current evidence;
3. inputs and dependencies;
4. allowed and forbidden changes;
5. frozen configurations, data, models, budgets, and revisions;
6. ordered execution stages;
7. Pass/Fail and stopping conditions;
8. required first report before expensive GPU work;
9. required deliverables;
10. artifact and provenance requirements;
11. failure and retry rules;
12. successor-opening conditions;
13. review history.

## 7. Task states and authority

Allowed states:

| State | Meaning | Who may set it |
|---|---|---|
| `planned` | Defined but dependency or authorization is incomplete | Local |
| `ready` | Authorized for execution | Local |
| `running` | Executor has started the declared work | Remote |
| `awaiting_review` | Required evidence has been pushed | Remote |
| `revision_needed` | Review found a concrete repair requirement | Local |
| `blocked` | External condition prevents meaningful progress | Remote or local |
| `accepted` | Code and result evidence passed review | Local |
| `stopped` | Gate failed or route was deliberately terminated | Local |

State transition rules:

```text
planned -> ready -> running -> awaiting_review -> accepted
                         │             │
                         └-> blocked   └-> revision_needed -> running

ready/running/awaiting_review -> stopped  only by local review
```

No task is `accepted` without:

- a reviewed task branch or equivalent commit;
- passing required tests;
- a result report;
- run provenance for every empirical claim;
- explicit local reviewer sign-off.

## 8. Branch and handoff protocol

### Publishing work

1. Local side creates or updates the task on `main`.
2. Local side records the task's `source_revision`.
3. Local side pushes `main`.
4. Remote side fetches `main` and creates the exact declared task branch.

### Remote execution

1. Confirm task status is `ready` and dependencies are accepted.
2. Confirm local checkout matches `source_revision`.
3. Change the branch copy of the task to `running`.
4. Produce the required first report before expensive execution.
5. Implement and run only the authorized scope.
6. Record failed attempts as well as successful runs.
7. Change the task to `awaiting_review`.
8. Push the task branch and report branch name, commit, runs, artifacts, costs, and blockers.

### Local review

1. Fetch the task branch without trusting its summary.
2. Validate path scope, code, tests, configs, manifests, and claim/result correspondence.
3. Reproduce cheap checks locally and inspect remote evidence.
4. Set `revision_needed` with exact findings, or accept the result.
5. Merge accepted work into `main`.
6. Record the acceptance decision and open eligible successors in a local follow-up commit.

Direct pushes to `main` from remote executors are outside the protocol.

## 9. Run and artifact contract

Every formal run uses `runs/<run-id>/`.

The manifest records:

- run ID and task ID;
- source and execution commit;
- dirty-state flag;
- model/checkpoint identifiers and hashes;
- dataset manifests and revisions;
- resolved configuration hash;
- hardware and software environment;
- command or launcher;
- start/end timestamps;
- sample, token, update, FLOP, GPU-hour, and wall-clock accounting as applicable;
- exit status;
- metric output references;
- artifact references;
- retry parent and failure reason.

Large artifact references record:

```yaml
artifact_id: T430-seed0-step1000
kind: checkpoint
canonical_uri: "storage://umm-rl/checkpoints/T430/seed0/step1000"
sha256: "6f7f7096d8f346a1619adad15cdbb014ac44b1f875e6c638bca689bfd71600fe"
bytes: 2147483648
producer_run: "T430-showo2-seed0-20260826"
source_revision: "0123456789abcdef0123456789abcdef01234567"
retention: "keep until project close or superseding decision"
```

The values above are illustrative schema examples. A committed artifact record uses the measured URI, hash, size, run ID, and source revision; zero is not a valid completed artifact size.

## 10. Result submission and checking

Each task report directory contains:

- `result-summary.md`: factual results, tables, costs, and failures;
- `claim-check.md`: each research claim mapped to supporting runs and metrics;
- `failure-ledger.md`: failed runs, retries, exclusions, and unresolved anomalies.

Remote reports may conclude:

- evidence supports the preregistered Gate;
- evidence fails the Gate;
- evidence is inconclusive;
- execution is blocked.

Only the local review side converts that report into a project decision in `PROGRESS.md` or `docs/decisions/`.

## 11. Automatic validation

The implementation introduces two validators and one CI workflow.

### Task-tree validator

It checks:

- unique task IDs and filenames;
- valid status values;
- exactly one root;
- existing parent and dependency nodes;
- no dependency cycles or orphan tasks;
- branch and owner fields for `ready` or later tasks;
- accepted dependencies before a task becomes `ready`;
- review evidence before `accepted`;
- no task marked accepted by a remote-only result commit.

### Run-manifest validator

It checks:

- required provenance fields;
- valid configuration and source hashes;
- no dirty formal run unless explicitly marked diagnostic;
- referenced result files;
- artifact hashes and nonzero measured sizes;
- retry chains and failure reasons;
- consistency between task ID, branch, source revision, and report.

### CI

Every push and pull request runs:

- task-tree validation;
- run-manifest validation;
- Markdown local-link validation;
- Python tests;
- Python compilation;
- shell syntax checks;
- `git diff --check`.

## 12. Failure and concurrency handling

- A remote agent that cannot satisfy a prerequisite sets `blocked` with the exact check and evidence; it does not invent a workaround outside scope.
- Numerical divergence counts as a failed run unless the task predefines a universal retry rule.
- Infrastructure failure may be retried only under the task's retry budget.
- Concurrent tasks must have disjoint allowed paths or explicitly named shared files and a merge order.
- If `main` advances in a way that changes the task contract, the local side publishes a new `source_revision`; the executor does not silently rebase into a changed experiment.
- Conflicting results are both retained and escalated to a review or decision task.

## 13. Implementation boundary

The first implementation phase will:

1. configure the `origin` remote;
2. add `CHANGELOG.md`, `tasks/`, `worklog/`, `reports/`, and `.github/`;
3. create the initial task tree above;
4. add task and run schemas/validators;
5. add CI and pull-request guidance;
6. update README, AGENTS, and PROGRESS navigation;
7. verify the repository locally;
8. commit and push `main` to `git@github.com:alexlovecoding/UMM_RL.git`.

It will not create GitHub Issues/Projects, upload large artifacts, run GPU experiments, rename the Python package, or change the accepted research formulation.
