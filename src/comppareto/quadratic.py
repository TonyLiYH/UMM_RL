"""Exact numerical contracts for compensation-aware quadratic tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]


class CurvatureError(ValueError):
    """Raised when the regularized private quadratic has no unique minimum."""


def _array(value: NDArray[np.floating], *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class QuadraticTask:
    """A task-local regularized quadratic with a global block selector."""

    local_gradient: FloatArray
    h_xx: FloatArray
    h_xphi: FloatArray
    h_phiphi: FloatArray
    mu: float
    selector: FloatArray

    def __post_init__(self) -> None:
        local_gradient = _array(self.local_gradient, name="local_gradient")
        h_xx = _array(self.h_xx, name="h_xx")
        h_xphi = _array(self.h_xphi, name="h_xphi")
        h_phiphi = _array(self.h_phiphi, name="h_phiphi")
        selector = _array(self.selector, name="selector")

        if local_gradient.ndim != 1:
            raise ValueError("local_gradient must be one-dimensional")
        local_dim = local_gradient.size
        if h_xx.shape != (local_dim, local_dim):
            raise ValueError("h_xx shape must match the local shared dimension")
        if h_xphi.ndim != 2 or h_xphi.shape[0] != local_dim:
            raise ValueError("h_xphi rows must match the local shared dimension")
        private_dim = h_xphi.shape[1]
        if h_phiphi.shape != (private_dim, private_dim):
            raise ValueError("h_phiphi shape must match the private dimension")
        if selector.ndim != 2 or selector.shape[0] != local_dim:
            raise ValueError("selector rows must match the local shared dimension")
        if not np.all((selector == 0.0) | (selector == 1.0)):
            raise ValueError("selector must be binary")
        if not np.all(np.sum(selector, axis=1) == 1.0):
            raise ValueError("each selector row must select exactly one global coordinate")
        if not np.all(np.sum(selector, axis=0) <= 1.0):
            raise ValueError("selector must not select a global coordinate more than once")
        if self.mu < 0 or not np.isfinite(self.mu):
            raise ValueError("mu must be finite and non-negative")
        if not np.allclose(h_xx, h_xx.T):
            raise ValueError("h_xx must be symmetric")
        if not np.allclose(h_phiphi, h_phiphi.T):
            raise ValueError("h_phiphi must be symmetric")

        private_curvature = h_phiphi + self.mu * np.eye(private_dim)
        if private_dim and np.min(np.linalg.eigvalsh(private_curvature)) <= 0:
            raise CurvatureError("regularized private curvature must be positive definite")

        object.__setattr__(self, "local_gradient", local_gradient)
        object.__setattr__(self, "h_xx", h_xx)
        object.__setattr__(self, "h_xphi", h_xphi)
        object.__setattr__(self, "h_phiphi", h_phiphi)
        object.__setattr__(self, "selector", selector)

    @property
    def private_curvature(self) -> FloatArray:
        private_dim = self.h_phiphi.shape[0]
        return self.h_phiphi + self.mu * np.eye(private_dim)

    def _local_step(self, global_step: NDArray[np.floating]) -> FloatArray:
        step = _array(global_step, name="global_step")
        if step.shape != (self.selector.shape[1],):
            raise ValueError("global_step shape must match selector columns")
        return self.selector @ step

    def private_response(self, global_step: NDArray[np.floating]) -> FloatArray:
        local_step = self._local_step(global_step)
        return -np.linalg.solve(self.private_curvature, self.h_xphi.T @ local_step)

    def schur(self) -> FloatArray:
        correction = self.h_xphi @ np.linalg.solve(
            self.private_curvature, self.h_xphi.T
        )
        return self.h_xx - correction

    def direct_change(
        self,
        global_step: NDArray[np.floating],
        private_step: NDArray[np.floating],
    ) -> float:
        local_step = self._local_step(global_step)
        private = _array(private_step, name="private_step")
        if private.shape != (self.h_phiphi.shape[0],):
            raise ValueError("private_step shape must match private curvature")
        value = (
            self.local_gradient @ local_step
            + 0.5 * local_step @ self.h_xx @ local_step
            + local_step @ self.h_xphi @ private
            + 0.5 * private @ self.private_curvature @ private
        )
        return float(value)

    def compensated_change(self, global_step: NDArray[np.floating]) -> float:
        local_step = self._local_step(global_step)
        value = self.local_gradient @ local_step + 0.5 * local_step @ self.schur() @ local_step
        return float(value)

    def lifted_gradient(self) -> FloatArray:
        return self.selector.T @ self.local_gradient

    def scaled(self, scale: float) -> "QuadraticTask":
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError("scale must be finite and positive")
        return QuadraticTask(
            local_gradient=scale * self.local_gradient,
            h_xx=scale * self.h_xx,
            h_xphi=scale * self.h_xphi,
            h_phiphi=scale * self.h_phiphi,
            mu=scale * self.mu,
            selector=self.selector.copy(),
        )


def retained_gain(change: float, attainable: float, epsilon: float) -> float:
    """Return the normalized retained improvement for a local change."""

    if attainable < 0 or epsilon <= 0:
        raise ValueError("attainable must be non-negative and epsilon must be positive")
    return -float(change) / (float(attainable) + float(epsilon))


@dataclass(frozen=True)
class CommonDescentResult:
    direction: FloatArray
    combined_gradient: FloatArray
    first_weight: float
    margin: float
    stationary: bool


@dataclass(frozen=True)
class TrustRegionResult:
    step: FloatArray
    objective_change: float
    attainable_gain: float
    boundary_active: bool


@dataclass(frozen=True)
class NegotiationResult:
    step: FloatArray
    tau: float
    attainable_gains: FloatArray
    retained_gains: FloatArray


def common_descent_two(
    first: NDArray[np.floating],
    second: NDArray[np.floating],
    *,
    metric: NDArray[np.floating],
    tolerance: float = 1e-12,
) -> CommonDescentResult:
    """Project zero onto the metric convex segment of two global gradients."""

    g1 = _array(first, name="first")
    g2 = _array(second, name="second")
    m = _array(metric, name="metric")
    if g1.ndim != 1 or g2.shape != g1.shape:
        raise ValueError("gradients must be one-dimensional with matching shapes")
    if m.shape != (g1.size, g1.size) or not np.allclose(m, m.T):
        raise ValueError("metric must be a symmetric square matrix matching gradients")
    if np.min(np.linalg.eigvalsh(m)) <= 0:
        raise ValueError("metric must be positive definite")

    metric_inverse = np.linalg.inv(m)
    delta = g1 - g2
    denominator = float(delta @ metric_inverse @ delta)
    if denominator <= tolerance:
        first_weight = 0.0
    else:
        first_weight = float(
            np.clip(-(delta @ metric_inverse @ g2) / denominator, 0.0, 1.0)
        )
    combined = first_weight * g1 + (1.0 - first_weight) * g2
    margin = float(combined @ metric_inverse @ combined)
    stationary = margin <= tolerance
    direction = np.zeros_like(combined) if stationary else -(metric_inverse @ combined)
    return CommonDescentResult(
        direction=direction,
        combined_gradient=combined,
        first_weight=first_weight,
        margin=margin,
        stationary=stationary,
    )


def _validate_metric(metric: NDArray[np.floating], dimension: int) -> FloatArray:
    result = _array(metric, name="metric")
    if result.shape != (dimension, dimension) or not np.allclose(result, result.T):
        raise ValueError("metric must be a symmetric square matrix matching the dimension")
    if np.min(np.linalg.eigvalsh(result)) <= 0:
        raise ValueError("metric must be positive definite")
    return result


def trust_region_optimum(
    *,
    gradient: NDArray[np.floating],
    hessian: NDArray[np.floating],
    metric: NDArray[np.floating],
    radius: float,
    tolerance: float = 1e-12,
) -> TrustRegionResult:
    """Minimize a convex quadratic inside an ellipsoidal trust region."""

    g = _array(gradient, name="gradient")
    h = _array(hessian, name="hessian")
    if g.ndim != 1:
        raise ValueError("gradient must be one-dimensional")
    if h.shape != (g.size, g.size) or not np.allclose(h, h.T):
        raise ValueError("hessian must be symmetric and match the gradient")
    if np.min(np.linalg.eigvalsh(h)) < -tolerance:
        raise CurvatureError("trust-region helper requires a positive-semidefinite hessian")
    m = _validate_metric(metric, g.size)
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("radius must be finite and positive")

    chol = np.linalg.cholesky(m)
    transformed_gradient = np.linalg.solve(chol, g)
    left_solved = np.linalg.solve(chol, h)
    transformed_hessian = np.linalg.solve(chol, left_solved.T).T
    transformed_hessian = 0.5 * (transformed_hessian + transformed_hessian.T)

    candidate = -np.linalg.pinv(transformed_hessian, rcond=tolerance) @ transformed_gradient
    residual = transformed_hessian @ candidate + transformed_gradient
    boundary_active = not (
        np.linalg.norm(residual) <= tolerance
        and np.linalg.norm(candidate) <= radius + tolerance
    )

    if boundary_active:
        def boundary_norm(multiplier: float) -> float:
            shifted = transformed_hessian + multiplier * np.eye(g.size)
            return float(
                np.linalg.norm(np.linalg.solve(shifted, -transformed_gradient))
            )

        lower = 0.0
        upper = 1.0
        while boundary_norm(upper) > radius:
            upper *= 2.0
        for _ in range(200):
            middle = 0.5 * (lower + upper)
            if boundary_norm(middle) > radius:
                lower = middle
            else:
                upper = middle
        candidate = np.linalg.solve(
            transformed_hessian + upper * np.eye(g.size), -transformed_gradient
        )

    step = np.linalg.solve(chol.T, candidate)
    change = float(g @ step + 0.5 * step @ h @ step)
    return TrustRegionResult(
        step=step,
        objective_change=change,
        attainable_gain=max(0.0, -change),
        boundary_active=boundary_active,
    )


def negotiate_retained_gain(
    tasks: list[QuadraticTask],
    *,
    metric: NDArray[np.floating],
    radius: float,
    epsilons: list[float],
    penalty: float = 0.0,
) -> NegotiationResult:
    """Solve the small convex max-min retained-gain diagnostic with SLSQP."""

    if len(tasks) < 2 or len(epsilons) != len(tasks):
        raise ValueError("provide at least two tasks and one epsilon per task")
    global_dim = tasks[0].selector.shape[1]
    if any(task.selector.shape[1] != global_dim for task in tasks):
        raise ValueError("all tasks must share one global coordinate space")
    if any(epsilon <= 0 for epsilon in epsilons):
        raise ValueError("epsilons must be positive")
    m = _validate_metric(metric, global_dim)

    gradients = [task.lifted_gradient() for task in tasks]
    hessians = [task.selector.T @ task.schur() @ task.selector for task in tasks]
    attainable = np.array(
        [
            trust_region_optimum(
                gradient=gradient,
                hessian=hessian,
                metric=m,
                radius=radius,
            ).attainable_gain
            for gradient, hessian in zip(gradients, hessians, strict=True)
        ]
    )

    def local_change(task_index: int, step: FloatArray) -> float:
        return float(
            gradients[task_index] @ step
            + 0.5 * step @ hessians[task_index] @ step
        )

    def objective(vector: FloatArray) -> float:
        step = vector[:-1]
        tau = vector[-1]
        return float(-tau + 0.5 * penalty * step @ m @ step)

    constraints: list[dict[str, object]] = [
        {
            "type": "ineq",
            "fun": lambda vector: radius**2 - vector[:-1] @ m @ vector[:-1],
        }
    ]
    for index, epsilon in enumerate(epsilons):
        denominator = float(attainable[index] + epsilon)
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda vector, idx=index, denom=denominator: (
                    -local_change(idx, vector[:-1]) / denom - vector[-1]
                ),
            }
        )

    initial = np.zeros(global_dim + 1)
    solved = minimize(
        objective,
        initial,
        method="SLSQP",
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not solved.success:
        raise RuntimeError(f"retained-gain negotiation failed: {solved.message}")
    step = np.asarray(solved.x[:-1], dtype=np.float64)
    retained = np.array(
        [
            -local_change(index, step) / (attainable[index] + epsilons[index])
            for index in range(len(tasks))
        ]
    )
    return NegotiationResult(
        step=step,
        tau=float(solved.x[-1]),
        attainable_gains=attainable,
        retained_gains=retained,
    )
