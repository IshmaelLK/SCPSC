"""Certify the QP curve, then serialize sampled diagnostics on request."""
import time
from pathlib import Path

import numpy as np
import torch

from .dynamics import recover
from .envelope import planning_envelope_mode
from .polynomial import PolynomialTrajectory
from .result_io import save_json


def accept_same_curve(qp, reference, endpoint, env, params, config, output):
    """Certify the unchanged QP curve on its registered common partition."""
    from .specific_force import certify_spline
    from .uniform_bspline import UniformBSpline

    if not qp["numerical_pass"]:
        raise ValueError("QP_NOT_ACCEPTED: cannot accept an unsolved candidate")
    output = Path(output)
    begin = time.perf_counter()
    trajectory = PolynomialTrajectory(qp["coefficients"], qp["durations"], endpoint.head, endpoint.tail, "fixed")
    version = dict(
        trajectory_version="SCPSC_TRAJECTORY_LAYER_V1",
        source_geometry_version=env["source"].source_geometry_version,
        source_geometry_id=env["source"].source_geometry_id,
    )
    np.savez_compressed(
        output / "trajectory.npz",
        coefficients=trajectory.coeffs,
        durations=trajectory.durations,
        head=trajectory.head,
        tail=trajectory.tail,
        control_points=qp["controls"],
        local_control_points=qp["local_controls"],
        control_origin=qp["diagnostics"]["origin"],
        knots=qp["knots"],
        breaks=qp["breaks"],
        corridor_indices=qp["corridor_indices"],
        geometry_mode="fixed",
        trajectory_kind="exact_uniform_bspline_power_conversion",
        **version,
    )
    spline = UniformBSpline(float(reference["times"][-1]), len(qp["controls"]))
    certificate = certify_spline(
        spline,
        qp["local_controls"],
        qp["breaks"],
        g=params.g,
        gamma=config["limits"]["gamma"],
        theta=config["limits"]["theta"],
        **version,
    )
    save_json(output / "specific_force_certificate.json", certificate)
    diagnostics = qp["diagnostics"]
    mode = planning_envelope_mode(config["ubsqp"])
    planning_margin = float(diagnostics["planning_corridor_margin_lower_bound_m"])
    system_margin = float(diagnostics["fixed_envelope_corridor_margin_lower_bound_m"])
    tolerance = float(config["ubsqp"].get("inequality_tolerance", 1e-6))
    planning_pass = bool(np.isfinite(planning_margin) and planning_margin >= -tolerance)
    system_pass = bool(np.isfinite(system_margin) and system_margin >= -tolerance)
    geometry_pass = planning_pass if mode == "bare_point" else system_pass
    passed = bool(certificate["passed"] and geometry_pass)
    status = (
        ("POINT_TRAJECTORY_CERTIFIED" if mode == "bare_point" else "TRAJECTORY_CERTIFIED")
        if passed
        else (certificate["status"] if not certificate["passed"] else "CORRIDOR_NOT_CERTIFIED")
    )
    report = dict(
        status=status,
        passed=passed,
        specific_force_status=certificate["status"],
        specific_force_passed=certificate["passed"],
        planning_envelope=mode,
        planning_geometry_certificate_pass=planning_pass,
        wholebody_geometry_certificate_pass=system_pass,
        planning_corridor_margin_lower_bound_m=planning_margin,
        fixed_envelope_corridor_margin_lower_bound_m=system_margin,
        load_speed_certificate_upper_mps=diagnostics["load_speed_certificate_upper_mps"],
        flight_time_s=float(reference["times"][-1]),
        snap_squared_integral=diagnostics["snap_integral"],
        acceptance_scope="original linear QP rows and continuous specific-force control-point certificate",
        qp_calls=1,
        trajectory_modified=False,
        adaptive_refinement=False,
        sampled_acceptance=False,
        **version,
    )
    report["seconds"] = time.perf_counter() - begin
    save_json(output / "backend_output.json", report)
    return trajectory, report


def write_trajectory_diagnostics(trajectory, qp, reference, params, config, output, report):
    """Write sampled output after the first certified planning result is timed."""
    output = Path(output)
    begin = time.perf_counter()
    version = {key: report[key] for key in ("trajectory_version", "source_geometry_version", "source_geometry_id")}
    # Samples are output diagnostics only, and cannot rescue a failed certificate.
    if report["specific_force_passed"]:
        step = float(config["validation"]["time_step"])
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("OUTPUT_TIME_STEP must be finite and positive")
        grid = np.unique(np.r_[np.arange(0.0, reference["times"][-1], step), qp["breaks"]])
        with torch.no_grad():
            recovered = recover(torch.as_tensor(trajectory.sample(grid, 5)), params)
        states = {k: v.detach().cpu().numpy() for k, v in recovered.items()}
        np.savez_compressed(
            output / "planning_reference.npz",
            time=grid,
            coefficients=trajectory.coeffs,
            durations=trajectory.durations,
            geometry_mode="fixed",
            **states,
            **version,
        )
        report["maximum_load_speed_sampled_mps"] = float(np.linalg.norm(states["vL"], axis=1).max())
        report["sample_count"] = len(grid)
    report["output_statistics_seconds"] = time.perf_counter() - begin
    save_json(output / "backend_output.json", report)
    return report["output_statistics_seconds"]
