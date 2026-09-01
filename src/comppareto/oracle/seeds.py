"""Hierarchical seed policy: one config seed spawns independent streams per axis.

Matches ``docs/theory/oracle-spec.md`` section 8 ("Seed policy"): changing
one axis (e.g. adding a noise-model variant) must not perturb any other
axis's realized values for a fixed top-level seed, and every case must be
exactly reproducible from ``(config_seed, case_index)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AXES: tuple[str, ...] = (
    "graph_structure",
    "curvature",
    "coupling",
    "gradient",
    "noise",
    "meta_batch",
    "probe_direction",
)


@dataclass(frozen=True)
class CaseSeeds:
    """One independent ``numpy.random.Generator`` per generation axis for a single case."""

    graph_structure: np.random.Generator
    curvature: np.random.Generator
    coupling: np.random.Generator
    gradient: np.random.Generator
    noise: np.random.Generator
    meta_batch: np.random.Generator
    probe_direction: np.random.Generator


def case_seeds(config_seed: int, case_index: int) -> CaseSeeds:
    """Deterministic per-case, per-axis generators from ``(config_seed, case_index)``.

    The top-level sequence is spawned once per case index (so cases are
    mutually independent and reproducible in isolation), then each case's
    sequence is spawned once per axis in a fixed order.
    """

    root = np.random.SeedSequence(config_seed)
    case_sequences = root.spawn(case_index + 1)
    axis_sequences = case_sequences[case_index].spawn(len(AXES))
    generators = {axis: np.random.default_rng(seq) for axis, seq in zip(AXES, axis_sequences)}
    return CaseSeeds(**generators)
