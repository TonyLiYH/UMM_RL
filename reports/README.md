# Result reports

Remote executors submit task-scoped evidence under `reports/<task-id>/`.

Each report directory contains:

- `result-summary.md`: factual metrics, costs, run IDs, and failures;
- `claim-check.md`: each claim mapped to supporting runs and metrics;
- `failure-ledger.md`: failed attempts, retries, exclusions, and unresolved anomalies.

Remote reports may conclude `supports gate`, `fails gate`, `inconclusive`, or `blocked`. Only the local review side updates `PROGRESS.md`, accepts a task, stops a route, or opens successor tasks.

