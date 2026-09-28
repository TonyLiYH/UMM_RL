"""Seeded selector-support generation for disjoint, partial, and full overlap.

Deliberately independent of ``comppareto.oracle.selectors``' block-layout
machinery: that module's ``build_incidence`` requires ``num_blocks`` in
``[4, 64]``, which excludes T110's required global-dimension floor of 2
(``tasks/T110-overlap-family.md``: "dimensions 2-32"). Blocks here always
have width 1, so a "support" is directly a set of global coordinate indices.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

OverlapRegime = Literal["disjoint", "partial", "full"]
REGIMES: tuple[OverlapRegime, ...] = ("disjoint", "partial", "full")


class RegimeError(ValueError):
    """Raised when a generated support family violates its regime contract."""


def sample_num_tasks(
    rng: np.random.Generator, global_dim: int, *, max_tasks: int = 5
) -> int:
    """Sample a task count in ``[2, min(max_tasks, global_dim)]``.

    Capped at ``global_dim`` so the disjoint regime can always give every
    task at least one private coordinate.
    """

    upper = min(max_tasks, global_dim)
    if upper < 2:
        raise RegimeError("global_dim must be at least 2 to host >=2 tasks")
    return int(rng.integers(2, upper + 1))


def sample_supports(
    rng: np.random.Generator,
    regime: OverlapRegime,
    num_tasks: int,
    global_dim: int,
) -> tuple[tuple[int, ...], ...]:
    """Return one ordered tuple of global column indices per task.

    Order encodes each task's local coordinate ordering: row ``r`` of that
    task's selector maps to global column ``support[r]``. The *set* of a
    support (order-independent) is what the overlap-regime contract in
    :func:`_validate_regime` is defined over.
    """

    if global_dim < 2:
        raise RegimeError("global_dim must be at least 2")
    if num_tasks < 2:
        raise RegimeError("num_tasks must be at least 2")

    if regime == "full":
        supports = tuple(
            tuple(int(x) for x in rng.permutation(global_dim)) for _ in range(num_tasks)
        )
    elif regime == "disjoint":
        if num_tasks > global_dim:
            raise RegimeError("disjoint regime needs num_tasks <= global_dim")
        supports = _sample_disjoint_supports(rng, num_tasks, global_dim)
    elif regime == "partial":
        supports = _sample_partial_supports(rng, num_tasks, global_dim)
    else:  # pragma: no cover - exhaustive Literal
        raise RegimeError(f"unknown regime {regime!r}")

    _validate_regime(regime, supports, global_dim)
    return supports


def _sample_disjoint_supports(
    rng: np.random.Generator, num_tasks: int, global_dim: int
) -> tuple[tuple[int, ...], ...]:
    perm = rng.permutation(global_dim)
    if num_tasks > 1:
        cut_points = sorted(
            int(x)
            for x in rng.choice(np.arange(1, global_dim), size=num_tasks - 1, replace=False)
        )
    else:
        cut_points = []
    groups = np.split(perm, cut_points)
    return tuple(tuple(sorted(int(x) for x in group)) for group in groups)


def _sample_partial_supports(
    rng: np.random.Generator, num_tasks: int, global_dim: int
) -> tuple[tuple[int, ...], ...]:
    """Construct genuine partial overlap by forcing a shared hub coordinate
    and, on task 0, forcing a coordinate that is deliberately excluded.

    This guarantees, by construction rather than by chance, that every case
    realizes both a shared coordinate (distinguishing it from ``disjoint``)
    and at least one non-full task (distinguishing it from ``full``), for
    every ``global_dim >= 2``.
    """

    all_coords = list(range(global_dim))
    hub = int(rng.choice(all_coords))
    excluded_for_task0 = next((c for c in all_coords if c != hub), None)

    supports: list[tuple[int, ...]] = []
    for task_index in range(num_tasks):
        pool = [c for c in all_coords if c != hub]
        if task_index == 0 and excluded_for_task0 is not None:
            pool = [c for c in pool if c != excluded_for_task0]
        extra_size = int(rng.integers(0, len(pool) + 1)) if pool else 0
        extra = (
            [int(x) for x in rng.choice(pool, size=extra_size, replace=False)]
            if extra_size
            else []
        )
        support_set = sorted({hub, *extra})
        order = [int(x) for x in rng.permutation(support_set)]
        supports.append(tuple(order))
    return tuple(supports)


def _validate_regime(
    regime: OverlapRegime,
    supports: tuple[tuple[int, ...], ...],
    global_dim: int,
) -> None:
    full_set = set(range(global_dim))
    sets = []
    for support in supports:
        support_set = set(support)
        if len(support_set) != len(support):
            raise RegimeError("support must not repeat a coordinate within one task")
        if not support_set:
            raise RegimeError("support must be nonempty")
        if not support_set.issubset(full_set):
            raise RegimeError("support references a coordinate outside global_dim")
        sets.append(support_set)

    pair_overlaps = [
        len(sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))
    ]

    if regime == "disjoint":
        if any(overlap > 0 for overlap in pair_overlaps):
            raise RegimeError("disjoint regime must not share any coordinate across tasks")
    elif regime == "full":
        if not all(s == full_set for s in sets):
            raise RegimeError("full regime must select every global coordinate for every task")
    elif regime == "partial":
        if not any(overlap > 0 for overlap in pair_overlaps):
            raise RegimeError("partial regime must realize at least one shared coordinate")
        if all(s == full_set for s in sets):
            raise RegimeError("partial regime must not degenerate into full overlap")
