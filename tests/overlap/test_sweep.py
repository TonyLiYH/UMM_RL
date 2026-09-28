from __future__ import annotations

import yaml

from comppareto.overlap.regimes import REGIMES
from comppareto.overlap.sweep import enumerate_case_keys, run_sweep

CONFIG_PATH = "configs/t1b/overlap-family.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_enumerate_case_keys_matches_seeds_per_regime() -> None:
    config = _load_config()
    keys = enumerate_case_keys(config)
    assert len(keys) == len(REGIMES) * config["seeds_per_regime"]
    assert len(set(keys)) == len(keys)
    for regime in REGIMES:
        assert sum(1 for r, _ in keys if r == regime) == config["seeds_per_regime"]


def test_resolved_config_meets_frozen_protocol_minimum() -> None:
    config = _load_config()
    assert config["seeds_per_regime"] >= 100
    assert config["global_dim_floor"] == 2
    assert config["global_dim_floor"] + config["global_dim_cycle"] == 32


def test_run_sweep_on_resolved_config_passes_with_zero_failures() -> None:
    config = _load_config()
    outcome = run_sweep(config, source_revision="0" * 40)
    summary = outcome["summary"]
    assert summary["total_cases"] == len(REGIMES) * config["seeds_per_regime"]
    assert summary["failed_cases"] == 0
    assert outcome["failures"] == []
    for regime in REGIMES:
        per_regime = summary["per_regime"][regime]
        assert per_regime["total"] >= 100
        assert per_regime["failed"] == 0
        assert min(per_regime["global_dims_covered"]) == 2
        assert max(per_regime["global_dims_covered"]) == 32
