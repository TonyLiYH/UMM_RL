"""Run the deterministic T1 synthetic validation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import scipy

from .quadratic import (
    CurvatureError,
    QuadraticTask,
    common_descent_two,
    negotiate_retained_gain,
    retained_gain,
    trust_region_optimum,
)


def _task() -> QuadraticTask:
    return QuadraticTask(
        local_gradient=np.array([0.4, -0.2]),
        h_xx=np.array([[3.0, 0.4], [0.4, 2.0]]),
        h_xphi=np.array([[0.5], [-0.3]]),
        h_phiphi=np.array([[1.2]]),
        mu=0.8,
        selector=np.eye(2),
    )


def run_suite(config: dict[str, Any]) -> dict[str, Any]:
    tolerance = float(config["absolute_tolerance"])
    task = _task()
    step = np.array([0.1, -0.2])
    private = task.private_response(step)
    exact_error = abs(
        task.compensated_change(step) - task.direct_change(step, private)
    )

    base_change = task.compensated_change(step)
    base_schur = task.schur()
    base_retained = retained_gain(base_change, 0.7, 1e-6)
    rescaling_errors: list[float] = []
    for scale in config["rescaling_factors"]:
        scaled = task.scaled(float(scale))
        rescaling_errors.append(
            float(np.max(np.abs(scaled.schur() - scale * base_schur)))
        )
        rescaling_errors.append(
            abs(
                retained_gain(scale * base_change, scale * 0.7, scale * 1e-6)
                - base_retained
            )
        )

    common = common_descent_two(
        np.array([1.0, 0.0]), np.array([0.0, 1.0]), metric=np.eye(2)
    )
    directional_max = max(
        float(np.array([1.0, 0.0]) @ common.direction),
        float(np.array([0.0, 1.0]) @ common.direction),
    )

    indefinite_rejected = False
    try:
        QuadraticTask(
            local_gradient=np.array([1.0]),
            h_xx=np.eye(1),
            h_xphi=np.ones((1, 1)),
            h_phiphi=np.array([[-2.0]]),
            mu=0.5,
            selector=np.eye(1),
        )
    except CurvatureError:
        indefinite_rejected = True

    selector_rejected = False
    try:
        QuadraticTask(
            local_gradient=np.array([1.0]),
            h_xx=np.eye(1),
            h_xphi=np.zeros((1, 1)),
            h_phiphi=np.eye(1),
            mu=0.5,
            selector=np.array([[0.5, 0.5]]),
        )
    except ValueError:
        selector_rejected = True

    attainable = trust_region_optimum(
        gradient=np.array([-2.0, 0.0]),
        hessian=np.eye(2),
        metric=np.eye(2),
        radius=0.5,
    )

    first = QuadraticTask(
        local_gradient=np.array([-2.0, 0.0]),
        h_xx=np.eye(2),
        h_xphi=np.zeros((2, 1)),
        h_phiphi=np.eye(1),
        mu=0.5,
        selector=np.eye(2),
    )
    second = QuadraticTask(
        local_gradient=np.array([0.0, -1.0]),
        h_xx=np.eye(2),
        h_xphi=np.zeros((2, 1)),
        h_phiphi=np.eye(1),
        mu=0.5,
        selector=np.eye(2),
    )
    negotiated = negotiate_retained_gain(
        [first, second], metric=np.eye(2), radius=0.5, epsilons=[1e-8, 2e-8]
    )
    negotiated_scaled = negotiate_retained_gain(
        [first.scaled(1e3), second.scaled(1e-3)],
        metric=np.eye(2),
        radius=0.5,
        epsilons=[1e-5, 2e-11],
    )
    negotiation_error = max(
        float(np.max(np.abs(negotiated.step - negotiated_scaled.step))),
        abs(negotiated.tau - negotiated_scaled.tau),
    )

    max_rescaling_error = max(rescaling_errors)
    checks = {
        "exact_elimination": {
            "passed": exact_error <= tolerance,
            "error": exact_error,
            "tolerance": tolerance,
        },
        "conditional_rescaling": {
            "passed": max_rescaling_error <= tolerance,
            "error": max_rescaling_error,
            "tolerance": tolerance,
        },
        "common_descent": {
            "passed": (not common.stationary) and directional_max < 0,
            "maximum_directional_derivative": directional_max,
            "margin": common.margin,
        },
        "indefinite_rejection": {"passed": indefinite_rejected},
        "selector_validation": {"passed": selector_rejected},
        "attainable_gain": {
            "passed": abs(attainable.attainable_gain - 0.875) <= tolerance,
            "value": attainable.attainable_gain,
            "expected": 0.875,
            "tolerance": tolerance,
        },
        "negotiation_rescaling": {
            "passed": negotiation_error <= 1e-7,
            "solution_error": negotiation_error,
            "tolerance": 1e-7,
        },
    }
    return {
        "status": "pass" if all(item["passed"] for item in checks.values()) else "fail",
        "checks": checks,
    }


def _git_metadata() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else "unborn",
        "dirty": bool(status.stdout.strip()) or status.returncode != 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    manifest = run_suite(config)
    manifest["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    manifest["environment"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    manifest["git"] = _git_metadata()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
