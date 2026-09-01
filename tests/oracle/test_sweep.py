from __future__ import annotations

import yaml

from comppareto.oracle.sweep import COVERAGE_ATTRS, enumerate_cases, select_detailed_subset

CONFIG_PATH = "configs/oracle/baseline.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_enumerate_cases_matches_section10_grid_size() -> None:
    config = _load_config()
    cases = enumerate_cases(config)
    families = len(config["families"])
    optimizers = len(config["optimizers"])
    horizons = len(config["horizons"])
    regimes = len(config["stability_regimes"])
    seeds = config["seeds_per_cell"]
    assert len(cases) == families * optimizers * horizons * regimes * seeds
    assert len({c.case_index for c in cases}) == len(cases)


def test_enumerate_cases_private_dims_match_num_tasks() -> None:
    config = _load_config()
    for case in enumerate_cases(config):
        assert len(case.private_dims) == case.num_tasks


def test_select_detailed_subset_covers_every_attribute_value() -> None:
    config = _load_config()
    cases = enumerate_cases(config)
    budget = config["detailed_subset_size"]
    chosen, covered = select_detailed_subset(cases, budget)
    assert len(chosen) <= budget
    for attr in COVERAGE_ATTRS:
        all_values = {getattr(c, attr) for c in cases}
        assert covered[attr] == all_values
