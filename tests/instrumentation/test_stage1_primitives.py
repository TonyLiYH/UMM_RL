"""Stage 1: deterministic toy tests for the framework-neutral primitives.

Covers the pass/fail gate's "no silent device fallback" requirement and the
basic tensor-level statistics every other module builds on.
"""

from __future__ import annotations

import numpy as np
import pytest

from comppareto.instrumentation import stats
from comppareto.instrumentation.device import DeviceUnavailableError, resolve_device


def test_resolve_device_accepts_cpu() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("CPU") == "cpu"
    assert resolve_device("  cpu  ") == "cpu"


def test_resolve_device_rejects_unsupported_without_fallback() -> None:
    with pytest.raises(DeviceUnavailableError):
        resolve_device("cuda:0")
    with pytest.raises(DeviceUnavailableError):
        resolve_device("tpu")


def test_l2_norm_matches_known_value() -> None:
    assert stats.l2_norm(np.array([3.0, 4.0], dtype=np.float32)) == pytest.approx(5.0)
    assert stats.l2_norm(np.zeros(0, dtype=np.float32)) == 0.0


def test_cosine_orthogonal_and_parallel() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert stats.cosine(a, b) == pytest.approx(0.0)
    assert stats.cosine(a, a) == pytest.approx(1.0)
    assert stats.cosine(a, -a) == pytest.approx(-1.0)


def test_cosine_none_on_zero_vector() -> None:
    zero = np.zeros(3, dtype=np.float32)
    nonzero = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert stats.cosine(zero, nonzero) is None
    assert stats.cosine(nonzero, zero) is None


def test_norm_ratio_and_none_on_zero_whole() -> None:
    part = np.array([1.0, 0.0], dtype=np.float32)
    whole = np.array([2.0, 0.0], dtype=np.float32)
    assert stats.norm_ratio(part, whole) == pytest.approx(0.5)
    assert stats.norm_ratio(part, np.zeros(2, dtype=np.float32)) is None


def test_near_zero_rate() -> None:
    array = np.array([0.0, 1e-10, 1.0, -1e-12], dtype=np.float32)
    assert stats.near_zero_rate(array, epsilon=1e-8) == pytest.approx(0.75)


def test_is_finite_detects_nan_and_inf() -> None:
    assert stats.is_finite(np.array([1.0, 2.0], dtype=np.float32))
    assert not stats.is_finite(np.array([1.0, np.nan], dtype=np.float32))
    assert not stats.is_finite(np.array([np.inf, 0.0], dtype=np.float32))


def test_gram_matrix_is_symmetric_and_correct() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 1.0], dtype=np.float32)
    gram = stats.gram_matrix([a, b])
    assert gram[0][0] == pytest.approx(1.0)
    assert gram[1][1] == pytest.approx(2.0)
    assert gram[0][1] == pytest.approx(gram[1][0])
    assert gram[0][1] == pytest.approx(1.0)


def test_directional_derivative() -> None:
    gradient = np.array([1.0, 2.0], dtype=np.float32)
    direction = np.array([3.0, -1.0], dtype=np.float32)
    assert stats.directional_derivative(gradient, direction) == pytest.approx(1.0)
