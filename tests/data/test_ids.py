from __future__ import annotations

import pytest

from comppareto.data.ids import group_bucket, stable_id


def test_stable_id_is_deterministic() -> None:
    assert stable_id("d1-coco-caption", "42:7") == stable_id("d1-coco-caption", "42:7")


def test_stable_id_changes_with_namespace() -> None:
    assert stable_id("ns-a", "key") != stable_id("ns-b", "key")


def test_stable_id_changes_with_key() -> None:
    assert stable_id("ns", "key-a") != stable_id("ns", "key-b")


def test_stable_id_prefixed_with_namespace() -> None:
    assert stable_id("d2-llava-instruct", "000000033471").startswith("d2-llava-instruct-")


@pytest.mark.parametrize("namespace,native_key", [("", "key"), ("ns", "")])
def test_stable_id_rejects_empty_inputs(namespace: str, native_key: str) -> None:
    with pytest.raises(ValueError):
        stable_id(namespace, native_key)


def test_group_bucket_is_deterministic_and_bounded() -> None:
    bucket = group_bucket("some-group-key", num_buckets=10_000)
    assert bucket == group_bucket("some-group-key", num_buckets=10_000)
    assert 0 <= bucket < 10_000


def test_group_bucket_rejects_non_positive_buckets() -> None:
    with pytest.raises(ValueError):
        group_bucket("key", num_buckets=0)
