"""Deterministic split assignment shared across D1-D3 training-pool sources.

``evaluation_only`` is intentionally *not* produced by this module: it is
sourced directly from each dataset's own official held-out partition
(for example COCO ``val2017``, which never appears in COCO ``train2017``)
so that evaluation/training disjointness is guaranteed by upstream dataset
construction rather than by a hash function this project controls. This
module only partitions the *training-pool* universe (COCO train2017 image
ids, and independently DiffusionDB image names) into
``diagnostic`` / ``pilot_train`` / ``pilot_validation`` / ``pilot_meta``.

The partition is a pure function of a group key (never of any model
gradient, loss, or training outcome) so it can be safely computed before
any model exists, per the task's research-integrity requirement.
"""

from __future__ import annotations

from .ids import group_bucket

#: Fraction of the training-pool bucket space assigned to each split.
#: Order matters: boundaries are cumulative and evaluated in this order.
SPLIT_FRACTIONS: tuple[tuple[str, float], ...] = (
    ("diagnostic", 0.02),
    ("pilot_validation", 0.05),
    ("pilot_meta", 0.03),
    ("pilot_train", 0.90),
)

NUM_BUCKETS = 10_000


def _boundaries() -> tuple[tuple[str, int, int], ...]:
    total_fraction = sum(fraction for _, fraction in SPLIT_FRACTIONS)
    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError(f"SPLIT_FRACTIONS must sum to 1.0, got {total_fraction}")
    boundaries = []
    lower = 0
    for name, fraction in SPLIT_FRACTIONS:
        upper = lower + round(fraction * NUM_BUCKETS)
        boundaries.append((name, lower, upper))
        lower = upper
    # Ensure the last boundary consumes any rounding remainder exactly.
    last_name, last_lower, _ = boundaries[-1]
    boundaries[-1] = (last_name, last_lower, NUM_BUCKETS)
    return tuple(boundaries)


_BOUNDARIES = _boundaries()


def assign_split(group_key: str) -> str:
    """Return the deterministic training-pool split name for ``group_key``."""
    bucket = group_bucket(group_key, num_buckets=NUM_BUCKETS)
    for name, lower, upper in _BOUNDARIES:
        if lower <= bucket < upper:
            return name
    raise AssertionError("bucket fell outside every declared split boundary")


def split_names() -> tuple[str, ...]:
    return tuple(name for name, _ in SPLIT_FRACTIONS)
