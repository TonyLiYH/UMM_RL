"""CompPareto numerical research utilities."""

from .quadratic import (
    CommonDescentResult,
    CurvatureError,
    NegotiationResult,
    QuadraticTask,
    TrustRegionResult,
    common_descent_two,
    negotiate_retained_gain,
    retained_gain,
    trust_region_optimum,
)

__all__ = [
    "CommonDescentResult",
    "CurvatureError",
    "NegotiationResult",
    "QuadraticTask",
    "TrustRegionResult",
    "common_descent_two",
    "negotiate_retained_gain",
    "retained_gain",
    "trust_region_optimum",
]
