from __future__ import annotations

from pathlib import Path

import pytest

from comppareto.repo_state.acceptance import load_acceptance_contract


SCHEMA = Path("schemas/task-acceptance.schema.json")


def test_loads_valid_acceptance_contract(tmp_path: Path) -> None:
    path = tmp_path / "T999.acceptance.yaml"
    path.write_text(
        """
schema_version: 1
task_id: T999
base_ref: origin/main
required_files:
  - reports/T999/result-summary.md
commands:
  - name: tests
    command: python -m pytest -q
metrics:
  - file: runs/example/summary.json
    path: passed_cases
    op: ">="
    value: 10
forbidden_claims:
  - file: reports/T999/result-summary.md
    when:
      file: runs/example/manifest.json
      path: status
      equals: fail
    phrases: [supports gate]
"""
    )

    contract = load_acceptance_contract(path, SCHEMA)

    assert contract.task_id == "T999"
    assert contract.required_files == ("reports/T999/result-summary.md",)
    assert contract.metrics[0].operator == ">="


def test_rejects_metric_with_unknown_operator(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
schema_version: 1
task_id: T999
base_ref: origin/main
required_files: []
commands: []
metrics:
  - file: result.json
    path: score
    op: approximately
    value: 1
forbidden_claims: []
"""
    )

    with pytest.raises(ValueError, match="invalid acceptance contract"):
        load_acceptance_contract(path, SCHEMA)
