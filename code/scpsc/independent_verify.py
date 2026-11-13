"""Independent trajectory-layer verification from saved original coefficients.

No saved acceptance flag, optimizer, sampled rope angle or legacy rigid-body
force/attitude model participates in acceptance.
"""

from pathlib import Path
from time import perf_counter
import json

import numpy as np

from .polynomial import PolynomialTrajectory
from .envelope import (
    center_halfspaces,
    planning_envelope_mode,
    planning_envelope_parameters,
    fixed_support,
    effective_radii,
)
from .specific_force import TRAJECTORY_VERSION, certify_power, power_to_bernstein
from .velocity_constraints import from_config


def _source_identity(directory, config, saved):
    """Rebuild the currently configured Hop and check the saved input identity."""
    from .scene_inputs import load_scene

    env = load_scene(config)
    source = env["source"]
    current = dict(source_geometry_version=source.source_geometry_version, source_geometry_id=source.source_geometry_id)
    for key, value in current.items():
        if saved[key] != value:
            raise ValueError("SOURCE_GEOMETRY_VERSION_MISMATCH: " + key)
    return current, np.stack((env["start"], env["goal"]))


def _load_saved_curve(directory):
    with np.load(directory / "trajectory.npz", allow_pickle=False) as z:
        tr = PolynomialTrajectory(z["coefficients"], z["durations"], z["head"], z["tail"], "fixed")
        indices = z["corridor_indices"].copy()
        breaks = z["breaks"].copy()
        knots = z["knots"].copy()
        identity = {
            key: str(z[key].item()) for key in ("trajectory_version", "source_geometry_version", "source_geometry_id")
        }
    if identity["trajectory_version"] != TRAJECTORY_VERSION:
        raise ValueError("TRAJECTORY_VERSION_MISMATCH: historical curves require explicit new verification")
    if (
        breaks.shape != (len(tr.durations) + 1,)
        or not np.isfinite(breaks).all()
        or breaks[0] != 0.0
        or np.any(np.diff(breaks) <= 0.0)
        or not np.array_equal(np.diff(breaks), tr.durations)
    ):
        raise ValueError("SAVED_COMMON_PARTITION: positive exact breaks/durations required")
    with np.load(directory / "reference.npz", allow_pickle=False) as reference:
        switches = reference["times"].copy()
    if (
        switches.ndim != 1
        or len(switches) < 2
        or not np.isfinite(switches).all()
        or switches[0] != 0.0
        or switches[-1] != breaks[-1]
        or np.any(np.diff(switches) <= 0.0)
    ):
        raise ValueError("SAVED_REFERENCE: invalid complete reference times")
    if (
        knots.ndim != 1
        or not np.isfinite(knots).all()
        or np.any(np.diff(knots) < 0.0)
        or knots[0] != 0.0
        or knots[-1] != breaks[-1]
    ):
        raise ValueError("SAVED_KNOTS: invalid knot partition")
    common = np.unique(np.r_[knots, switches])
    if not np.all(np.isin(common, breaks)):
        raise ValueError("SAVED_COMMON_PARTITION: a knot or corridor switch is missing")
    expected_indices = np.searchsorted(switches, breaks[:-1], side="right") - 1
    if indices.shape != (len(tr.durations),) or not np.array_equal(indices, expected_indices):
        raise ValueError("SAVED_CORRIDOR_ASSOCIATION: interval uses the wrong corridor")
    return tr, indices, breaks, identity, switches


def _geometry_and_speed(tr, indices, corridors, source_identity, ell, lim, model, mode):
    minimum_geometry = np.inf
    minimum_system_geometry = np.inf
    speed_violation = 0.0
    speed_upper = 0.0
    for coeff, dt, corridor_index in zip(tr.coeffs, tr.durations, indices):
        corridor = corridors[int(corridor_index)]
        proof = corridor.get("proof") or {}
        for key, value in source_identity.items():
            recorded = proof.get(key)
            if recorded != value:
                raise ValueError("SAVED_CORRIDOR_SOURCE_MISMATCH: " + key)
        normals, center_bounds = center_halfspaces(corridor, ell, lim, mode)
        lengths = np.linalg.norm(normals, axis=1)
        if (
            normals.ndim != 2
            or normals.shape[1] != 3
            or center_bounds.shape != (len(normals),)
            or not np.isfinite(normals).all()
            or not np.isfinite(center_bounds).all()
            or np.any(lengths <= 0.0)
        ):
            raise ValueError("SAVED_CORRIDOR: finite nonzero normals and finite center bounds required")
        # Add the absolute position only after the algebraic basis conversion.
        # Derivative extraction below never sees the world coordinate offset.
        position = power_to_bernstein(coeff, dt)
        # Section 3.3, unnumbered margin: minimum signed face distance over Bernstein controls.
        minimum_geometry = min(minimum_geometry, float(np.min((center_bounds - position @ normals.T) / lengths)))
        rhoL, rhoQ = effective_radii(lim)
        true_support = fixed_support(normals, ell, rhoL, rhoQ, lim["theta"])
        planning_support = fixed_support(normals, *planning_envelope_parameters(ell, lim, mode))
        minimum_system_geometry = min(
            minimum_system_geometry,
            float(np.min((center_bounds - position @ normals.T - true_support + planning_support) / lengths)),
        )
        velocity = power_to_bernstein(np.arange(1, 8)[:, None] * coeff[1:], dt)
        speed_violation = max(speed_violation, float(np.max(np.abs(velocity @ model.normals.T) - model.bounds)))
        speed_upper = max(speed_upper, float(np.linalg.norm(velocity, axis=1).max()))
    return minimum_geometry, minimum_system_geometry, speed_violation, speed_upper


def verify_saved(directory, config):
    begin = perf_counter()
    directory = Path(directory)
    tr, indices, breaks, identity, switches = _load_saved_curve(directory)
    source_identity, configured_endpoints = _source_identity(directory, config, identity)
    corridors = json.loads((directory / "corridors.json").read_text(encoding="utf-8"))
    if len(corridors) != len(switches) - 1:
        raise ValueError("SAVED_CORRIDORS: one corridor per reference interval is required")
    lim = config["limits"]
    ell = float(config["dynamics"]["ell"])
    g = float(config["dynamics"]["g"])
    model = from_config(config["ubsqp"], lim["vL"])
    mode = planning_envelope_mode(config["ubsqp"])
    minimum_geometry, minimum_system_geometry, speed_violation, speed_upper = _geometry_and_speed(
        tr, indices, corridors, source_identity, ell, lim, model, mode
    )
    endpoint = np.stack((tr.evaluate(0, 0.0, 5), tr.evaluate(len(tr.durations) - 1, tr.durations[-1], 5)))
    targets = np.stack((tr.head.T, tr.tail.T))
    if np.any(targets[:, 1:] != 0.0):
        raise ValueError("SAVED_ENDPOINTS: the trajectory-layer model requires stationary PVAJ targets")
    if not np.array_equal(targets[:, 0], configured_endpoints):
        raise ValueError("SAVED_ENDPOINTS: targets differ from the currently configured task endpoints")
    equation_error = max(float(np.max(np.abs(endpoint[:, :4] - targets))), float(np.max(np.abs(endpoint[:, 4:6, :2]))))
    seam_error = 0.0
    for i in range(len(tr.durations) - 1):
        left = tr.evaluate(i, tr.durations[i], 6)
        right = tr.evaluate(i + 1, 0.0, 6)
        seam_error = max(seam_error, float(np.max(np.abs(left - right))))
    force = certify_power(
        tr.coeffs, tr.durations, breaks=breaks, g=g, gamma=lim["gamma"], theta=lim["theta"], **identity
    )
    eq_tolerance = float(config["ubsqp"].get("equality_tolerance", 1e-6))
    ineq_tolerance = float(config["ubsqp"].get("inequality_tolerance", 1e-6))
    endpoint_pass = bool(np.isfinite(equation_error) and equation_error <= eq_tolerance)
    continuity_pass = bool(np.isfinite(seam_error) and seam_error <= eq_tolerance)
    geometry_pass = bool(np.isfinite(minimum_geometry) and minimum_geometry >= -ineq_tolerance)
    system_geometry_pass = bool(np.isfinite(minimum_system_geometry) and minimum_system_geometry >= -ineq_tolerance)
    speed_pass = bool(np.isfinite(speed_violation) and speed_violation <= ineq_tolerance)
    accepted_geometry = geometry_pass if mode == "bare_point" else system_geometry_pass
    passed = bool(endpoint_pass and continuity_pass and accepted_geometry and speed_pass and force["passed"])
    nodes, weights = np.polynomial.legendre.leggauss(32)
    length = sum(
        dt / 2 * np.dot(weights, np.linalg.norm(tr.evaluate(i, (nodes + 1) * dt / 2, 1)[:, 1], axis=1))
        for i, dt in enumerate(tr.durations)
    )
    # Paper Eqs. (6) and (20) apply at stationary endpoints. These diagnostics
    # retain vertical d4/d5 freedom and contain no mass or yaw dependence.
    body_acceleration = endpoint[:, 2].copy()
    body_jerk = endpoint[:, 3].copy()
    body_acceleration[:, :2] += ell / g * endpoint[:, 4, :2]
    body_jerk[:, :2] += ell / g * endpoint[:, 5, :2]
    return dict(
        status="INDEPENDENT_PASS" if passed else "INDEPENDENT_FAIL",
        passed=passed,
        model="SCPSC_TRAJECTORY_LAYER",
        planning_envelope=mode,
        optimizer_calls=0,
        coefficient_bernstein_reconstruction=True,
        source_geometry_rebuilt=True,
        **identity,
        endpoint_equations_pass=endpoint_pass,
        endpoint_equation_error_SI=equation_error,
        endpoint_rows=32,
        vertical_high_order_endpoints_free=True,
        recovered_endpoint_acceleration_maximum_SI=float(np.max(np.abs(body_acceleration))),
        recovered_endpoint_jerk_maximum_SI=float(np.max(np.abs(body_jerk))),
        continuity_pass=continuity_pass,
        continuity_through_order=6,
        continuity_maximum_error_SI=seam_error,
        planning_geometry_certificate_pass=geometry_pass,
        wholebody_geometry_certificate_pass=system_geometry_pass,
        fixed_envelope_corridor_minimum_m=minimum_system_geometry,
        center_corridor_minimum_m=minimum_geometry,
        load_speed_continuous_pass=speed_pass,
        velocity_projection_violation_mps=speed_violation,
        load_speed_continuous_upper_mps=speed_upper,
        specific_force_certificate=force,
        specific_force_continuous_pass=force["passed"],
        acceptance_scope="saved-power endpoint, C6 continuity, center-corridor, load-speed and specific-force Bernstein checks",
        samples_used_for_acceptance=0,
        flight_time_s=float(breaks[-1]),
        length_m=float(length),
        length_quadrature_order=32,
        snap_integral=float(tr.snap_cost()),
        seconds=perf_counter() - begin,
    )
