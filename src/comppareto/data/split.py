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
#:
#: ``diagnostic``'s fraction (0.0063, i.e. 63 of 10,000 buckets) is
#: deliberately much smaller than the other splits' fractions: the task
#: file (`tasks/T260-posttraining-data-admission.md`, "Frozen pilot
#: manifests") requires `diagnostic` to contain "512--2,048 paired examples
#: where available" -- an *absolute* record-count target, not a fraction of
#: the (much larger) training pool. A flat 2% share was tried first and
#: measured (against the real, complete downloaded sources) to produce
#: 6,350 diagnostic records -- more than 3x over the 2,048 ceiling, because
#: the training-pool group-key universe is large (D1/D2 share ~118k COCO
#: image ids; D3 had its own ~31k-row group-key space after filtering,
#: under the prior row-level-only subsample design). 63 buckets was chosen
#: because, at that time, it was the largest bucket count that kept the
#: real measured diagnostic record count (2,037) within the declared
#: [512, 2048] range (64 buckets measured 2,068 -- just over the ceiling).
#: See `reports/T260/failure-ledger.md` for the full original measurement
#: record. The buckets freed by shrinking diagnostic are absorbed by
#: `pilot_train` (via the "absorb any rounding remainder" rule below), not
#: silently dropped.
#:
#: **Re-measured 2026-09-16** after DiffusionDB's archive-fan-out redesign
#: (`src/comppareto/data/diffusiondb.py`) shrank D3's training-pool row
#: count from 31,485 to 19,842: the *same* 63-bucket share, unchanged,
#: now measures `diagnostic = 1,828` real records (737 COCO + 980 LLaVA +
#: 111 DiffusionDB) against the real, complete rebuilt sources -- still
#: comfortably inside [512, 2048], so no further re-tuning of
#: `SPLIT_FRACTIONS` was needed this round. See
#: `reports/T260/mixture-and-accounting.md` Sec. 2 for the full updated
#: table.
SPLIT_FRACTIONS: tuple[tuple[str, float], ...] = (
    ("diagnostic", 0.0063),
    ("pilot_validation", 0.05),
    ("pilot_meta", 0.03),
    ("pilot_train", 0.9137),
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
