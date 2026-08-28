# Result reports

Remote executors submit task-scoped evidence under `reports/<task-id>/`.

Each report directory contains:

- `result-summary.md`: factual metrics, costs, run IDs, and failures;
- `claim-check.md`: each claim mapped to supporting runs and metrics;
- `failure-ledger.md`: failed attempts, retries, exclusions, and unresolved anomalies.

Remote reports may conclude `supports gate`, `fails gate`, `inconclusive`, or `blocked`. Only the local review side updates `PROGRESS.md`, accepts a task, stops a route, or opens successor tasks.

## Submission validation

Every task with `tasks/contracts/<task-id>.acceptance.yaml` must pass:

```bash
bash scripts/validate_task_submission.sh <task-id>
```

before the executor sets `awaiting_review`. Task-local test success is not a
substitute for this command. If the formal run status is `fail`, the final
summary must not say `supports gate` or `passes gate`.

`result-summary.md` is the current authoritative summary and is rewritten when
evidence changes. Historical observations belong in the task review history or
run notes. `failure-ledger.md` retains history but labels every item as open,
resolved, accepted limitation, or blocked.
