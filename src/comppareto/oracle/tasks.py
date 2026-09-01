"""Task-local exact quadratic model: train loss, meta loss, private curvature.

Matches ``docs/theory/oracle-spec.md`` section 2. Deliberately independent of
``src/comppareto/quadratic.py`` (outside this task's allowed paths) even
though the private-curvature contract mirrors ``QuadraticTask``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class OracleCurvatureError(ValueError):
    """Raised when the regularized private curvature is not positive definite."""


def _array(value: NDArray[np.floating], *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class OracleTask:
    """A task-local regularized quadratic with independent train/meta linear terms.

    ``h_xx``, ``h_xphi``, ``h_phiphi`` are shared between train and meta
    (curvature is a property of the local model, not of which batch
    evaluates it). ``(a_train, b_train)`` and ``(a_meta, b_meta)`` are drawn
    independently to represent disjoint train/meta batches.
    """

    h_xx: FloatArray
    h_xphi: FloatArray
    h_phiphi: FloatArray
    mu: float
    phi0: FloatArray
    a_train: FloatArray
    b_train: FloatArray
    a_meta: FloatArray
    b_meta: FloatArray
    selector: FloatArray

    def __post_init__(self) -> None:
        h_xx = _array(self.h_xx, name="h_xx")
        h_xphi = _array(self.h_xphi, name="h_xphi")
        h_phiphi = _array(self.h_phiphi, name="h_phiphi")
        phi0 = _array(self.phi0, name="phi0")
        a_train = _array(self.a_train, name="a_train")
        b_train = _array(self.b_train, name="b_train")
        a_meta = _array(self.a_meta, name="a_meta")
        b_meta = _array(self.b_meta, name="b_meta")
        selector = _array(self.selector, name="selector")

        if h_xx.ndim != 2 or h_xx.shape[0] != h_xx.shape[1]:
            raise ValueError("h_xx must be a square matrix")
        p_i = h_xx.shape[0]
        if h_xphi.ndim != 2 or h_xphi.shape[0] != p_i:
            raise ValueError("h_xphi rows must match the task-local shared dimension")
        d_i = h_xphi.shape[1]
        if h_phiphi.shape != (d_i, d_i):
            raise ValueError("h_phiphi shape must match the private dimension")
        if not np.allclose(h_xx, h_xx.T):
            raise ValueError("h_xx must be symmetric")
        if not np.allclose(h_phiphi, h_phiphi.T):
            raise ValueError("h_phiphi must be symmetric")
        if self.mu <= 0 or not np.isfinite(self.mu):
            raise ValueError("mu must be finite and strictly positive")
        for name, vec, dim in (
            ("phi0", phi0, d_i),
            ("b_train", b_train, d_i),
            ("b_meta", b_meta, d_i),
        ):
            if vec.shape != (dim,):
                raise ValueError(f"{name} must have shape ({dim},)")
        for name, vec in (("a_train", a_train), ("a_meta", a_meta)):
            if vec.shape != (p_i,):
                raise ValueError(f"{name} must have shape ({p_i},)")
        if selector.ndim != 2 or selector.shape[0] != p_i:
            raise ValueError("selector rows must match the task-local shared dimension")
        if not np.all((selector == 0.0) | (selector == 1.0)):
            raise ValueError("selector must be binary")
        if not np.all(np.sum(selector, axis=1) == 1.0):
            raise ValueError("each selector row must select exactly one global coordinate")
        if not np.all(np.sum(selector, axis=0) <= 1.0):
            raise ValueError("selector must not select a global coordinate more than once")

        private_curvature = h_phiphi + self.mu * np.eye(d_i)
        if d_i and np.min(np.linalg.eigvalsh(private_curvature)) <= 0:
            raise OracleCurvatureError("regularized private curvature must be positive definite")

        object.__setattr__(self, "h_xx", h_xx)
        object.__setattr__(self, "h_xphi", h_xphi)
        object.__setattr__(self, "h_phiphi", h_phiphi)
        object.__setattr__(self, "phi0", phi0)
        object.__setattr__(self, "a_train", a_train)
        object.__setattr__(self, "b_train", b_train)
        object.__setattr__(self, "a_meta", a_meta)
        object.__setattr__(self, "b_meta", b_meta)
        object.__setattr__(self, "selector", selector)

    @property
    def shared_dim(self) -> int:
        return self.h_xx.shape[0]

    @property
    def private_dim(self) -> int:
        return self.h_phiphi.shape[0]

    @property
    def h_phix(self) -> FloatArray:
        return self.h_xphi.T

    @property
    def private_curvature(self) -> FloatArray:
        """``C_i = H_phiphi + mu I``, positive definite by construction."""

        return self.h_phiphi + self.mu * np.eye(self.private_dim)

    @property
    def private_linear(self) -> FloatArray:
        """``c_i = b_train - mu * phi0``."""

        return self.b_train - self.mu * self.phi0

    def private_affine_term(self, x_i: FloatArray) -> FloatArray:
        """``H_phix x_i + c_i``: the x_i-dependent constant in the private-gradient recursion."""

        return self.h_phix @ x_i + self.private_linear

    def train_loss(self, x_i: FloatArray, phi_i: FloatArray) -> float:
        return float(
            self.a_train @ x_i
            + 0.5 * x_i @ self.h_xx @ x_i
            + x_i @ self.h_xphi @ phi_i
            + self.b_train @ phi_i
            + 0.5 * phi_i @ self.h_phiphi @ phi_i
        )

    def regularized_train_loss(self, x_i: FloatArray, phi_i: FloatArray) -> float:
        """``J_i(x_i, phi_i)``: train loss plus the private proximal term."""

        return self.train_loss(x_i, phi_i) + 0.5 * self.mu * float(
            np.dot(phi_i - self.phi0, phi_i - self.phi0)
        )

    def meta_loss(self, x_i: FloatArray, phi_i: FloatArray) -> float:
        return float(
            self.a_meta @ x_i
            + 0.5 * x_i @ self.h_xx @ x_i
            + x_i @ self.h_xphi @ phi_i
            + self.b_meta @ phi_i
            + 0.5 * phi_i @ self.h_phiphi @ phi_i
        )

    def train_private_gradient(self, x_i: FloatArray, phi_i: FloatArray) -> FloatArray:
        """``grad_phi J_i = H_phix x_i + C_i phi_i + c_i``."""

        return self.h_phix @ x_i + self.private_curvature @ phi_i + self.private_linear

    def meta_gradient_x(self, x_i: FloatArray, phi_i: FloatArray) -> FloatArray:
        return self.a_meta + self.h_xx @ x_i + self.h_xphi @ phi_i

    def meta_gradient_phi(self, x_i: FloatArray, phi_i: FloatArray) -> FloatArray:
        return self.b_meta + self.h_phix @ x_i + self.h_phiphi @ phi_i
