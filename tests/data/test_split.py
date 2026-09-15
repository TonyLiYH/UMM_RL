from __future__ import annotations

from collections import Counter

from comppareto.data.split import SPLIT_FRACTIONS, assign_split, split_names


def test_assign_split_is_deterministic() -> None:
    for key in ("1", "42", "some-image.jpg", "diffusiondb-part-000001"):
        assert assign_split(key) == assign_split(key)


def test_assign_split_only_returns_known_split_names() -> None:
    names = set(split_names())
    assert names == {name for name, _ in SPLIT_FRACTIONS}
    for key in range(500):
        assert assign_split(str(key)) in names


def test_evaluation_only_is_never_assigned_by_hashing() -> None:
    # evaluation_only is sourced from an official held-out partition, never
    # from this hash-based assignment -- assert the split table agrees.
    assert "evaluation_only" not in split_names()


def test_split_distribution_roughly_matches_declared_fractions() -> None:
    counts = Counter(assign_split(str(i)) for i in range(20_000))
    total = sum(counts.values())
    for name, fraction in SPLIT_FRACTIONS:
        observed = counts[name] / total
        assert abs(observed - fraction) < 0.02, (name, observed, fraction)
