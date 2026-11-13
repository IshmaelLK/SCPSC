"""Saved-trajectory swing postprocessing; independent of planning and geometry.

PASS means one IVP completed with finite states and positive checked tension.
Errors are reported without an added consistency or convergence acceptance limit.
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import trapezoid

from .dynamics import DynamicsParams
from .swing_physics import (
    SwingIntegrationError,
    SwingSettings,
    integrate_swing,
    tension_per_mass,
)
from .swing_trajectory import SwingTrajectoryInput, output_grid

SWING_SCHEMA = "SCPSC_SWING_CHECK_V1"


def swing_metrics(times, q, nu, tension):
    """Output-grid estimates, with time-weighted trapezoidal RMS, in SI units."""
    theta = np.arctan2(np.linalg.norm(q[:, :2], axis=1), q[:, 2])
    rate = np.linalg.norm(nu, axis=1)
    return {
        "peak_theta_sampled_rad": float(theta.max()),
        "rms_theta_time_weighted_rad": float(
            np.sqrt(trapezoid(theta**2, times) / (times[-1] - times[0]))
        ),
        "peak_swing_rate_sampled_radps": float(rate.max()),
        "initial_theta_rad": float(theta[0]),
        "final_theta_rad": float(theta[-1]),
        "initial_swing_rate_radps": float(rate[0]),
        "final_swing_rate_radps": float(rate[-1]),
        "minimum_tension_per_mass_sampled_mps2": float(np.min(tension)),
    }


def forward_arrays(reference, integrated, params, times=None):
    q, nu = integrated["q"], integrated["nu"]
    pL = reference["pQ"] - params.ell * q
    q_error = np.linalg.norm(q - reference["q"], axis=1)
    p_error = np.linalg.norm(pL - reference["pL"], axis=1)
    tension = tension_per_mass(q, nu, reference["aQ"], params)
    if not np.isfinite(tension).all() or np.any(tension <= 0):
        index = int(np.flatnonzero(~np.isfinite(tension) | (tension <= 0))[0])
        location = float(times[index]) if times is not None else 0.0
        raise SwingIntegrationError(
            "Nonpositive/nonfinite tension on the output grid",
            location,
            "TAUT_BRANCH_FAILED",
        )
    return {
        "qsim": q,
        "nu": nu,
        "pLsim": pL,
        "theta_rad": np.arctan2(np.linalg.norm(q[:, :2], axis=1), q[:, 2]),
        "omega_s_radps": np.cross(q, nu),
        "swing_rate_radps": np.linalg.norm(nu, axis=1),
        "tension_per_mass_mps2": tension,
        "position_error_m": p_error,
        "direction_error": q_error,
    }


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _metrics(times, reference, forward):
    metrics = {}
    for name, q, nu, tension in (
        (
            "reference",
            reference["q"],
            reference["q1"],
            reference["tension_per_mass_mps2"],
        ),
        ("forward", forward["qsim"], forward["nu"], forward["tension_per_mass_mps2"]),
    ):
        metrics[name] = swing_metrics(times, q, nu, tension)
    return metrics


def _evaluate(input_curve, settings, output):
    times = output_grid(input_curve.breaks, settings.output_step_s)
    reference = input_curve.reference(times)
    np.savez_compressed(output / "swing_reference.npz", time=times, **reference)
    integrated = integrate_swing(
        input_curve.acceleration_piece,
        input_curve.breaks,
        times,
        input_curve.params,
        settings,
    )
    forward = forward_arrays(reference, integrated, input_curve.params, times)
    np.savez_compressed(output / "swing_forward.npz", time=times, **forward)
    metrics = _metrics(times, reference, forward)
    with (output / "swing_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["state"] + list(metrics["reference"])
        )
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow(dict(state=name, **values))
    maxima = {
        key: float(forward[key].max())
        for key in (
            "position_error_m",
            "direction_error",
        )
    }
    return {
        "passed": True,
        "status": "SWING_PASS",
        "checks": {
            "integration_success": True,
            "finite_state": True,
            "positive_tension_sampled": True,
        },
        "maxima_sampled": maxima,
        "metrics_sampled": metrics,
        "integration": integrated["diagnostics"],
        "initial_reference": {
            "q": reference["q"][0].tolist(),
            "qdot_radps": reference["q1"][0].tolist(),
            "direction_residual": float(
                np.linalg.norm(reference["q"][0] - [0.0, 0.0, 1.0])
            ),
            "rate_residual_radps": float(np.linalg.norm(reference["q1"][0])),
        },
        "output_grid": {
            "nodes": len(times),
            "maximum_spacing_s": float(np.diff(times).max()),
            "stored_breaks_retained": True,
            "finite_sampling_estimates": True,
        },
        "tension_scope": "positive at the single IVP RHS evaluations and output nodes; not continuous certification",
    }


def validate_saved_swing(run_dir, output_dir, settings=None):
    """Write new swing files from trajectory.npz and effective_config.json only.

    Existing output directories are rejected. Invalid inputs produce a failed
    swing_check.json and never trigger planning, retiming or a parameter change.
    """
    begin = perf_counter()
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": SWING_SCHEMA,
        "validator_version": "2",
        "passed": False,
        "status": "SWING_NOT_COMPLETED",
        "source_run_directory": str(run_dir.resolve()),
        "trajectory_file": str((run_dir / "trajectory.npz").resolve()),
        "source_geometry_rebuilt": False,
        "continuous_safety_recertified": False,
        "transport_evaluated": False,
        "optimizer_calls": 0,
        "trajectory_modified": False,
        "acceptance_scope": "single independent IVP completion, finite states and positive sampled taut-cable tension",
        "initial_condition": "qsim=e3, nu=0",
        "projection": False,
        "tolerance_source": "adaptive integrator accuracy only; no posterior error acceptance thresholds",
    }
    try:
        settings = (
            SwingSettings(**settings)
            if isinstance(settings, dict)
            else (settings or SwingSettings())
        )
        config = json.loads(
            (run_dir / "effective_config.json").read_text(encoding="utf-8")
        )
        physical = config["dynamics"]
        params = DynamicsParams(
            ell=float(physical["ell"]),
            g=float(physical["g"]),
            specific_force_guard=float(physical.get("specific_force_guard", 1e-12)),
        )
        report.update(settings=asdict(settings), physical_parameters=asdict(params))
        curve = SwingTrajectoryInput.load(run_dir, params)
        report.update(curve.identity)
        report["physical_parameter_source"] = (
            "effective_config.json; trajectory coefficients contain no independent ell/g"
        )
        report.update(_evaluate(curve, settings, output_dir))
    except SwingIntegrationError as error:
        report.update(
            status=error.status, error=str(error), failure_time_s=error.time_s
        )
    except (ValueError, TypeError, KeyError, OSError) as error:
        report.update(
            status="SWING_INPUT_FAILED",
            error_type=type(error).__name__,
            error=str(error),
        )
    report["seconds"] = perf_counter() - begin
    _write_json(output_dir / "swing_check.json", report)
    return report
