from __future__ import annotations

from pathlib import Path

from comppareto.repo_state.acceptance import load_acceptance_contract


def test_repository_acceptance_contracts_are_valid() -> None:
    schema = Path("schemas/task-acceptance.schema.json")
    contracts = sorted(Path("tasks/contracts").glob("T*.acceptance.yaml"))

    assert {path.stem.split(".")[0] for path in contracts} >= {"T155", "T210"}
    loaded = [load_acceptance_contract(path, schema) for path in contracts]
    assert [contract.task_id for contract in loaded] == sorted(
        contract.task_id for contract in loaded
    )
