"""Array-level statistics shared by the recorder, blocks, and adapters.

All functions are pure and side-effect free: they read arrays and return
plain Python floats/bools (never mutate their inputs), which is part of the
transparency contract this task must prove (instrumentation must not alter
gradients or optimizer results).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def flatten(arrays: Iterable[np.ndarray]) -> np.ndarray:
    """Concatenate a sequence of arrays into one 1-D vector, preserving order."""
    parts = [np.asarray(a).reshape(-1) for a in arrays]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def is_finite(array: np.ndarray) -> bool:
    if array.size == 0:
        return True
    return bool(np.isfinite(array).all())


def l2_norm(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(array.reshape(-1), ord=2))


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cosine similarity of two flat vectors, or ``None`` if either is zero."""
    if a.size == 0 or b.size == 0:
        return None
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    dot = float(np.dot(a, b))
    value = dot / (norm_a * norm_b)
    # Guard against floating-point excursions just outside [-1, 1].
    return max(-1.0, min(1.0, value))


def norm_ratio(part: np.ndarray, whole: np.ndarray) -> float | None:
    """``||part|| / ||whole||``, or ``None`` if ``whole`` has zero norm."""
    denom = l2_norm(whole)
    if denom == 0.0:
        return None
    return l2_norm(part) / denom


def near_zero_rate(array: np.ndarray, *, epsilon: float) -> float:
    """Fraction of entries with absolute value strictly below ``epsilon``."""
    if array.size == 0:
        return 0.0
    return float(np.mean(np.abs(array) < epsilon))


def directional_derivative(gradient: np.ndarray, direction: np.ndarray) -> float:
    """``<gradient, direction>`` -- the first-order rate of loss change along
    ``direction``. Callers pass the realized update direction to measure how
    much the actual step moved along (or against) the raw gradient.
    """
    if gradient.size == 0:
        return 0.0
    return float(np.dot(gradient, direction))


def gram_matrix(vectors: Sequence[np.ndarray]) -> list[list[float]]:
    """Pairwise inner-product (Gram) matrix of a list of flat vectors.

    This is exactly the input MGDA's min-norm-point-in-the-convex-hull QP
    needs: it depends on the task gradients only through their pairwise dot
    products.
    """
    n = len(vectors)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            value = float(np.dot(vectors[i], vectors[j]))
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix
