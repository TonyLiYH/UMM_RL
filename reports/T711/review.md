# T711 hard semantic review

## Disposition

`revision_needed`

Reviewed branch: `origin/agent/T711-corl-semantic-correctness-admission`

Reviewed head: `0dd0d32babd93aff4e99cf2b20f93abc775b8a1f`

Actual fork revision: `e001548976ede8c8dd9805e5602fc2ab9ac6274b`

Review date: 2026-09-28

## Scope and provenance

The branch has one commit after its merge base and changes only
`tasks/T711-corl-semantic-correctness-admission.md` (184 added lines). It does
not contain changes under the required implementation, test, configuration,
run, or report paths. The submitted task file records source revision
`1fcb9e9964c678a330f32514f211828d551dc48c`, which is not a Git object in the
review repository. `git merge-base origin/main <reviewed-head>` identifies the
actual fork revision above.

The branch therefore supplies a specification, not the artifacts required to
support the research claim or satisfy the acceptance contract.

## Semantic findings

### 1. Causal image scoring is not implemented or tested

No adapter change or test demonstrates that image token log probabilities are
scored from pre image logits at matching causal positions. There is no
synthetic alignment test that would fail under a one token shift.

Required repair:

- implement pre image causal scoring in `src/comppareto/adapters/corl/` or the
  admitted vendored path;
- add a deterministic token by token alignment test in `tests/adapters/corl/`;
- record the executed command and result in the run evidence.

### 2. Old and reference policy semantics are absent

No code or evidence defines rollout time old log probabilities, a frozen
reference policy, or explicit reference free mode. There is no identity
smoke showing ratio one and KL zero before an optimizer update.

Required repair:

- capture old log probabilities with the rollout sample and keep them fixed
  through optimization of that sample;
- define and freeze a reference snapshot, or declare and test reference free
  operation;
- add an executable pre update identity smoke with tolerances and raw values.

### 3. Reward dispatch and reduction are not validated

No submitted implementation proves that understanding rows, generation rows,
and mixed batches receive the intended reward functions. No grouped reduction
test covers uneven valid token counts or mixed modality batches.

Required repair:

- implement explicit task and modality dispatch;
- add tests for understanding only, generation only, and mixed batches;
- add grouped reduction tests with uneven masks and assert the intended
  per group normalization.

### 4. Generation parameter ownership is not established

The branch does not enumerate the optimized generation parameters and has no
gradient/update evidence for that set. It therefore cannot show that the
generation loss reaches the intended image path.

Required repair:

- emit a deterministic owned parameter manifest with names, shapes, and counts;
- prove the intended generation subset has finite nonzero gradients and a
  realized parameter update in a bounded smoke;
- distinguish frozen parameters from disconnected parameters in the report.

### 5. Counterfactual anchors and leakage guards are missing

There is no anchor manifest, no anchor provenance, and no overlap audit against
train, validation, or test examples. Fixed counterfactual anchors are therefore
neither frozen nor protected from evaluation leakage.

Required repair:

- add an immutable anchor manifest with stable identifiers and content hashes;
- document selection provenance without using evaluation outcomes;
- run exact and near duplicate checks against all evaluation partitions and
  publish their machine readable results.

### 6. Deterministic evidence bundle is absent

The required task specific paths `configs/corl/admission/`,
`runs/corl-admission-v1/`, `reports/T711/`, and `tests/adapters/corl/` are absent
from the reviewed branch. There is no immutable environment record, command
log, raw metric artifact, or summary tying results to a revision.

Required repair:

- add a pinned admission config and immutable environment metadata;
- store commands, exit codes, raw metrics, and test outputs under
  `runs/corl-admission-v1/`;
- write the final evidence summary under `reports/T711/`, citing exact artifact
  paths and the reviewed commit;
- ensure a clean checkout can reproduce the admission commands.

## Validation result

`bash scripts/validate_task_submission.sh T711` completed the repository test
suite with 307 passing tests, then rejected the submission. It reported all
five artifacts referenced by the inherited run manifest as missing, all five
required T711 report/config patterns as missing, and all six semantic metric
checks as unreadable because their task-specific evidence is absent. The
validator also confirms that a task in `revision_needed` is not eligible for
submission.

## Downstream gate

T711 is not admitted. T712, T730, T740, and T755 remain closed wherever T711 is
a prerequisite. Persistent or claim bearing CoRL training must not use this
submission as semantic evidence.
