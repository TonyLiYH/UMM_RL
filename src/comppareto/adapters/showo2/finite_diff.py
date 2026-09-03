"""Central finite-difference reference and T215's tolerance gate.

Per ``reports/T215/first-report.md`` section 5:

- **Directions**: 4 fixed unit vectors in ``theta_s`` space -- 3
  Rademacher-random (seeds 42/43/44) and 1 "natural" direction (the raw
  gradient, normalized).
- **Step size**: ``eps = max(1e-3 * (||theta_s|| / sqrt(numel(theta_s))),
  1e-6)`` (scale-adaptive, with a fixed floor).
- **Estimator**: central difference using the rerun-response protocol
  (non-differentiable, only the scalar loss is needed).
- **Gates** (verbatim from the frozen protocol / task file's pass/fail
  gate): relative error <= 1e-3 when the FD reference magnitude exceeds
  1e-8; absolute error <= 1e-6 near zero.

These tolerance constants are T215's OWN, distinct from
``comppareto.oracle.crosscheck``'s ``NEAR_ZERO_NORM=1e-10`` /
``NEAR_ZERO_ABS_TOL=1e-11`` / ``FD_REL_TOL=1e-6`` (those are T155's tighter
analytic-domain tolerances for a different, closed-form synthetic task
family and are NOT reused here). Only the near-zero-vs-relative-error
branching *pattern* in ``crosscheck.compare`` is used as a design reference,
per the task instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

# T215's own frozen tolerance constants (first-report.md section 5 /
# docs/plans/showo2-first-attempt.md section 5 "initial numerical gates").
NEAR_ZERO_THRESHOLD = 1e-8
REL_TOL = 1e-3
ABS_TOL = 1e-6

RademacherSeeds: tuple[int, ...] = (42, 43, 44)


@dataclass(frozen=True)
class FdDirectionResult:
    """One direction's finite-difference comparison against an analytic
    directional derivative.
    """

    label: str
    eps: float
    fd_value: float
    analytic_value: float
    reference_magnitude: float
    error: float
    mode: str  # "relative" | "absolute"
    passed: bool
    loss_plus: float
    loss_minus: float


def scale_adaptive_eps(theta_s: torch.Tensor) -> float:
    """``eps = max(1e-3 * (||theta_s|| / sqrt(numel(theta_s))), 1e-6)``."""

    numel = theta_s.numel()
    if numel == 0:
        return 1e-6
    norm = float(theta_s.detach().norm())
    candidate = 1e-3 * (norm / (numel ** 0.5))
    return max(candidate, 1e-6)


def rademacher_direction(shape: tuple[int, ...], seed: int, dtype: torch.dtype) -> torch.Tensor:
    """A fixed +-1 Rademacher direction, normalized to unit L2 norm, seeded
    independently of the ambient RNG stream (uses a scratch ``Generator`` so
    this does not perturb any other diagnostic RNG state).
    """

    gen = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, shape, generator=gen, dtype=torch.int64)
    direction = torch.where(bits == 0, torch.tensor(-1.0), torch.tensor(1.0)).to(dtype)
    norm = direction.norm()
    return direction / norm if float(norm) > 0 else direction


def natural_direction(raw_grad: torch.Tensor) -> torch.Tensor:
    """The raw gradient, normalized -- the 4th, gradient-aligned direction."""

    norm = raw_grad.detach().norm()
    if float(norm) == 0.0:
        raise ValueError("raw_grad has zero norm; cannot normalize the natural FD direction")
    return raw_grad.detach() / norm


def build_directions(theta_s: torch.Tensor, raw_grad: torch.Tensor) -> dict[str, torch.Tensor]:
    """The 4 fixed FD directions: 3 Rademacher (seeds 42/43/44) + 1 natural."""

    directions: dict[str, torch.Tensor] = {}
    for seed in RademacherSeeds:
        directions[f"rademacher_seed_{seed}"] = rademacher_direction(
            tuple(theta_s.shape), seed, theta_s.dtype
        )
    directions["natural_raw_grad"] = natural_direction(raw_grad)
    return directions


def gate(reference_magnitude: float, error_value: float) -> tuple[str, float, bool]:
    """T215's own near-zero-vs-relative-error gate.

    Returns ``(mode, tolerance, passed)``. ``error_value`` is interpreted as
    an absolute difference when ``mode == "absolute"`` and as a relative
    error (already divided by ``reference_magnitude``) when
    ``mode == "relative"`` -- callers should pass the appropriately
    pre-computed quantity via :func:`compare_directional`.
    """

    if reference_magnitude > NEAR_ZERO_THRESHOLD:
        return "relative", REL_TOL, error_value <= REL_TOL
    return "absolute", ABS_TOL, error_value <= ABS_TOL


def compare_directional(
    label: str,
    eps: float,
    analytic_value: float,
    loss_plus: float,
    loss_minus: float,
) -> FdDirectionResult:
    fd_value = (loss_plus - loss_minus) / (2.0 * eps)
    reference_magnitude = abs(fd_value)
    if reference_magnitude > NEAR_ZERO_THRESHOLD:
        error = abs(analytic_value - fd_value) / reference_magnitude
        mode, tol, passed = "relative", REL_TOL, error <= REL_TOL
    else:
        error = abs(analytic_value - fd_value)
        mode, tol, passed = "absolute", ABS_TOL, error <= ABS_TOL
    return FdDirectionResult(
        label=label,
        eps=eps,
        fd_value=fd_value,
        analytic_value=analytic_value,
        reference_magnitude=reference_magnitude,
        error=error,
        mode=mode,
        passed=passed,
        loss_plus=loss_plus,
        loss_minus=loss_minus,
    )


def run_finite_difference_check(
    theta_s: torch.Tensor,
    raw_grad: torch.Tensor,
    analytic_grad: torch.Tensor,
    loss_at: Callable[[torch.Tensor], float],
) -> list[FdDirectionResult]:
    """Run the full 4-direction central finite-difference check.

    ``loss_at(theta_s_perturbed) -> float`` must be the caller-supplied,
    non-differentiable rerun-response meta loss evaluator (e.g.
    ``protocols.rerun_loss_only`` partially applied over the fixed
    adapt/meta batches, opt state, and lr) evaluated at a perturbed
    ``theta_s``. ``analytic_grad`` is the rerun-response gradient from
    ``protocols.compute_rerun_response`` (or, for the labeled
    expected-to-diverge reference, the commit-response gradient).
    """

    eps = scale_adaptive_eps(theta_s)
    directions = build_directions(theta_s, raw_grad)
    results: list[FdDirectionResult] = []
    for label, direction in directions.items():
        loss_plus = loss_at(theta_s + eps * direction)
        loss_minus = loss_at(theta_s - eps * direction)
        analytic_value = float((analytic_grad.detach() * direction).sum())
        results.append(compare_directional(label, eps, analytic_value, loss_plus, loss_minus))
    return results
