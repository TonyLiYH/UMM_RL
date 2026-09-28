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
from .trust_region_counterexamples import Counterexample, build_counterexample_suite
from .trust_region_guard import (
    SchurCurvature,
    StepAcceptanceRecord,
    evaluate_step,
    frozen_measured_change,
    naive_predicted_only_accept,
    project_out_negative_curvature,
    schur_curvature,
)

__all__ = [
    "CommonDescentResult",
    "Counterexample",
    "CurvatureError",
    "NegotiationResult",
    "QuadraticTask",
    "SchurCurvature",
    "StepAcceptanceRecord",
    "TrustRegionResult",
    "build_counterexample_suite",
    "common_descent_two",
    "evaluate_step",
    "frozen_measured_change",
    "naive_predicted_only_accept",
    "negotiate_retained_gain",
    "project_out_negative_curvature",
    "retained_gain",
    "schur_curvature",
    "trust_region_optimum",
]
