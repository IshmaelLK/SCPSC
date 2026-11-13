"""Physical-time QP data and original-coordinate constraint/objective assembly."""
from dataclasses import dataclass

import numpy as np

from .envelope import (
    effective_radii, fixed_support, normalized_halfspaces,
    planning_envelope_mode, planning_envelope_parameters, center_halfspaces,
)
from .uniform_bspline import UniformBSpline
from .velocity_constraints import VelocityConstraint, from_config


@dataclass
class QPInputs:
    """Validated physical-time reference; arrays are borrowed within this solve."""

    reference_times: np.ndarray
    reference_points: np.ndarray
    spline: UniformBSpline
    origin: np.ndarray
    endpoint_pvaj: np.ndarray
    ell: float


@dataclass
class QPConstraints:
    """Original-coordinate rows in unchanged assembly order."""

    Aeq: np.ndarray
    beq: np.ndarray
    labels: list[str]
    family: list[str]
    breaks: np.ndarray
    corridor_indices: np.ndarray
    corridor_physical_offsets: list[float]
    A: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    row_scale: np.ndarray
    rank: int
    singular: np.ndarray
    eq_consistency: float
    velocity_model: VelocityConstraint
    use_system_envelope: bool
    planning_mode: str


@dataclass
class QPObjective:
    """Fixed-time objective and its report-only factors."""

    H: np.ndarray
    f: np.ndarray
    objective_scale: float
    eigenvalues: np.ndarray
    reference_weights: np.ndarray
    B: np.ndarray
    snap: np.ndarray
    alpha_s: float
    alpha_r: float
    snap_ref: float
    length_ref: float


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _vector_rows(basis_row):
    return np.kron(np.asarray(basis_row)[None, :], np.eye(3))


def _scalar_row(basis_row, normal):
    return np.outer(basis_row, normal).reshape(-1)


def endpoint_equations(spline, endpoint_pvaj, origin):
    """24 PVAJ rows and 8 horizontal fourth/fifth derivative rows.

    The vertical fourth/fifth derivatives remain free. No equilibrium Jacobian,
    finite-difference force model, or redundant sixth-order endpoint row is used.
    """
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    labels: list[str] = []
    for endpoint, time in enumerate((0.0, spline.total_time)):
        for order in range(4):  # Eq. (6), prescribed endpoint position and stationary v, a, j.
            rows.extend(_vector_rows(spline.matrix(time, order)))
            target = np.asarray(endpoint_pvaj[endpoint, order], float).copy()
            if order == 0:
                target -= origin
            rhs.extend(target)
            labels.extend(f"endpoint_{endpoint}_d{order}_{axis}" for axis in "xyz")
        for order in (4, 5):  # Eq. (20), horizontal components only.
            rows.extend(_vector_rows(spline.matrix(time, order))[:2])
            rhs.extend((0.0, 0.0))
            labels.extend(f"endpoint_{endpoint}_horizontal_d{order}_{axis}" for axis in "xy")
    return np.asarray(rows), np.asarray(rhs), labels


def validate_inputs(reference_times, reference_points, corridors, endpoint_data, params, qp_config, endpoint_pvaj):
    reference_times, reference_points = _array(reference_times), _array(reference_points)
    if (
        reference_times.ndim != 1
        or len(reference_times) < 2
        or reference_times[0] != 0.0
        or np.any(np.diff(reference_times) <= 0.0)
        or reference_points.shape != (len(reference_times), 3)
        or not np.isfinite(reference_times).all()
        or not np.isfinite(reference_points).all()
    ):
        raise ValueError("reference requires finite [R,3] points and positive times starting at zero")
    if len(corridors) != len(reference_times) - 1 or len(endpoint_data) != 2:
        raise ValueError("one original corridor per reference edge and two endpoints required")
    spline = UniformBSpline(reference_times[-1], qp_config.get("n_control", 16))
    origin = reference_points[0].copy()
    if endpoint_pvaj is None:
        endpoint_pvaj = np.zeros((2, 4, 3))
        endpoint_pvaj[:, 0] = reference_points[[0, -1]]
    endpoint_pvaj = _array(endpoint_pvaj)
    if endpoint_pvaj.shape != (2, 4, 3) or not np.isfinite(endpoint_pvaj).all():
        raise ValueError("endpoint_pvaj must be finite [2,4,3]")
    # The original steady-state high-order equations below assume V=A=J=0.
    if np.any(endpoint_pvaj[:, 1:] != 0.0):
        raise ValueError("this endpoint compatibility formula requires stationary PVAJ endpoints")
    ell = params["ell"] if isinstance(params, dict) else params.ell
    return QPInputs(
        reference_times,
        reference_points,
        spline,
        origin,
        endpoint_pvaj,
        ell,
    )


def assemble_constraints(inputs, corridors, limits, qp_config, approach):
    reference_times = inputs.reference_times
    spline = inputs.spline
    origin = inputs.origin
    endpoint_pvaj = inputs.endpoint_pvaj
    ell = inputs.ell
    Aeq, beq, labels = endpoint_equations(spline, endpoint_pvaj, origin)
    rows, lower_rows, upper_rows = list(Aeq), list(beq), list(beq)
    family = ["endpoint"] * len(rows)
    extra_breaks = []
    if approach is not None:
        first, last = float(approach["start_time"]), float(approach["end_time"])
        if not 0.0 <= first < last <= spline.total_time or float(approach["v_down"]) <= 0.0:
            raise ValueError("approach must have a positive in-domain duration and descent bound")
        extra_breaks = [first, last]
        if approach.get("direction_xy") is not None:
            approach_direction = _array(approach["direction_xy"])
            if approach_direction.shape != (2,) or abs(np.linalg.norm(approach_direction) - 1.0) > 1e-10:
                raise ValueError("approach direction_xy must be a fixed horizontal unit vector")
            beta = float(approach["beta_max_rad"])
            if not 0.0 < beta < np.pi / 2.0:
                raise ValueError("approach beta_max_rad must lie in (0, pi/2)")
    breaks = np.unique(np.r_[spline.breaks, reference_times, extra_breaks])
    # Every reference switch is retained in breaks.  The left endpoint gives
    # the exact half-open association, even when a one-ulp span's midpoint
    # rounds onto its right switch.
    corridor_indices = np.searchsorted(reference_times, breaks[:-1], side="right") - 1
    rhoL, rhoQ = effective_radii(limits)
    planning_mode = planning_envelope_mode(qp_config)
    use_system_envelope = planning_mode == "fixed_system"
    planning_parameters = planning_envelope_parameters(ell, limits, planning_mode)
    halfspaces = []
    corridor_physical_offsets = []
    for corridor in corridors:
        normals, bounds = normalized_halfspaces(*center_halfspaces(corridor, ell, limits, planning_mode))
        true_support = fixed_support(normals, ell, rhoL, rhoQ, limits["theta"])
        planning_support = fixed_support(normals, *planning_parameters)
        halfspaces.append((normals, bounds - normals @ origin, true_support - planning_support))
    velocity_model = from_config(qp_config, limits["vL"])
    velocity_normals, velocity_bounds = velocity_model.normals, velocity_model.bounds
    for piece, (start, end, corridor_index) in enumerate(zip(breaks[:-1], breaks[1:], corridor_indices)):
        R = spline.bezier_matrix(start, end)
        velocity = spline.derivative_bezier_matrix(start, end, 1)
        normals, bounds, physical_offsets = halfspaces[int(corridor_index)]
        for k, basis in enumerate(R):  # Eq. (21), eight position controls inside the segment corridor.
            for face, (normal, bound) in enumerate(zip(normals, bounds)):
                rows.append(_scalar_row(basis, normal))
                lower_rows.append(-np.inf)
                upper_rows.append(bound)
                labels.append(f"corridor_{piece}_bezier_{k}_face_{face}")
                family.append("corridor")
                corridor_physical_offsets.append(physical_offsets[face])
        for k, basis in enumerate(velocity):  # Eq. (21), bilateral velocity-face constraints on seven controls.
            for direction, (normal, bound) in enumerate(zip(velocity_normals, velocity_bounds)):
                rows.append(_scalar_row(basis, normal))
                lower_rows.append(-bound)
                upper_rows.append(bound)
                labels.append(f"velocity_{piece}_bezier_{k}_{direction}")
                family.append("velocity")
            if approach is not None and start >= first and end <= last:
                rows.append(_scalar_row(basis, [0.0, 0.0, 1.0]))
                lower_rows.append(-float(approach["v_down"]))
                upper_rows.append(0.0)
                labels.append(f"approach_descent_{piece}_{k}")
                family.append("approach")
                if approach.get("direction_xy") is not None:
                    horizontal = np.r_[approach_direction, 0.0]
                    rows.append(_scalar_row(basis, horizontal))
                    lower_rows.append(0.0)
                    upper_rows.append(np.inf)
                    labels.append(f"approach_forward_{piece}_{k}")
                    family.append("approach")
                    normal = -np.tan(beta) * horizontal + np.array([0.0, 0.0, -1.0])
                    rows.append(_scalar_row(basis, normal))
                    lower_rows.append(-np.inf)
                    upper_rows.append(0.0)
                    labels.append(f"approach_slope_{piece}_{k}")
                    family.append("approach")
    A, lower, upper = np.asarray(rows), np.asarray(lower_rows), np.asarray(upper_rows)
    if not np.isfinite(A).all():
        raise ValueError("nonfinite QP matrix")
    row_norm = np.linalg.norm(A, axis=1)
    if np.any(row_norm <= 0.0):
        raise ValueError("zero QP row requires explicit mathematical diagnosis")
    row_scale = 1.0 / row_norm
    A_scaled = A * row_scale[:, None]
    lo_scaled, _up_scaled = lower * row_scale, upper * row_scale
    eq_scaled = A_scaled[: len(Aeq)]
    singular = np.linalg.svd(eq_scaled, compute_uv=False)
    rank = int(np.linalg.matrix_rank(eq_scaled))
    equality_fit = np.linalg.lstsq(eq_scaled, lo_scaled[: len(Aeq)], rcond=None)[0]
    eq_consistency = float(np.max(np.abs(eq_scaled @ equality_fit - lo_scaled[: len(Aeq)])))

    return QPConstraints(
        Aeq,
        beq,
        labels,
        family,
        breaks,
        corridor_indices,
        corridor_physical_offsets,
        A,
        lower,
        upper,
        row_scale,
        rank,
        singular,
        eq_consistency,
        velocity_model,
        use_system_envelope,
        planning_mode,
    )


def assemble_objective(inputs, limits, qp_config):
    reference_times = inputs.reference_times
    reference_points = inputs.reference_points
    spline = inputs.spline
    origin = inputs.origin
    alpha_s = float(qp_config.get("alpha_s", limits["lambda_s"]))
    alpha_r = float(qp_config.get("alpha_r", 1.0))
    snap_ref = float(qp_config.get("snap_ref", limits["snap_ref"]))
    length_ref = float(qp_config.get("length_ref", limits.get("Lref", 10.0)))
    if min(alpha_s, alpha_r) < 0.0 or min(snap_ref, length_ref) <= 0.0:
        raise ValueError("QP objective requires nonnegative weights and positive scales")
    if "reference_weights" in qp_config:
        reference_weights = _array(qp_config["reference_weights"])
    else:
        dt = np.diff(reference_times)
        reference_weights = np.r_[dt[0] / 2.0, (dt[:-1] + dt[1:]) / 2.0, dt[-1] / 2.0] / spline.total_time
    if (
        reference_weights.shape != reference_times.shape
        or np.any(reference_weights < 0.0)
        or not np.isfinite(reference_weights).all()
    ):
        raise ValueError("reference_weights must be finite nonnegative weights per reference point")
    B, snap = spline.matrix(reference_times), spline.snap_matrix()
    tracking = B.T @ (reference_weights[:, None] * B)
    # Eq. (7), snap energy plus weighted reference error in OSQP quadratic form.
    scalar_hessian = 2.0 * (alpha_s / snap_ref**2 * snap + alpha_r / length_ref**2 * tracking)
    H = np.kron(scalar_hessian, np.eye(3))
    f = (-2.0 * alpha_r / length_ref**2 * B.T @ (reference_weights[:, None] * (reference_points - origin))).reshape(-1)
    # Multiplication by a positive scalar preserves the QP minimizer.  No
    # extra regularizer is inserted to conceal a poorly assembled Hessian.
    objective_scale = 1.0 / max(float(np.max(np.abs(H))), float(np.max(np.abs(f))), 1e-30)
    _H_scaled, _f_scaled = H * objective_scale, f * objective_scale
    eigenvalues = np.linalg.eigvalsh(scalar_hessian)
    if eigenvalues[0] < -1e-10 * max(1.0, np.max(np.abs(eigenvalues))):
        raise ValueError("QP Hessian is not positive semidefinite within rounding tolerance")
    return QPObjective(
        H,
        f,
        objective_scale,
        eigenvalues,
        reference_weights,
        B,
        snap,
        alpha_s,
        alpha_r,
        snap_ref,
        length_ref,
    )


def assembled_matrices(inputs, constraints, objective):
    reference_times = inputs.reference_times
    reference_points = inputs.reference_points
    spline = inputs.spline
    origin = inputs.origin
    A = constraints.A
    lower = constraints.lower
    upper = constraints.upper
    Aeq = constraints.Aeq
    beq = constraints.beq
    row_scale = constraints.row_scale
    breaks = constraints.breaks
    corridor_indices = constraints.corridor_indices
    labels = constraints.labels
    use_system_envelope = constraints.use_system_envelope
    corridor_physical_offsets = constraints.corridor_physical_offsets
    family = constraints.family
    velocity_model = constraints.velocity_model
    H = objective.H
    f = objective.f
    objective_scale = objective.objective_scale
    reference_weights = objective.reference_weights
    matrices = dict(
        H=H,
        f=f,
        A=A,
        l=lower,
        u=upper,
        Aeq=Aeq,
        beq=beq,
        row_scale=row_scale,
        objective_scale=objective_scale,
        origin=origin,
        reference_times=reference_times,
        reference_points=reference_points,
        reference_weights=reference_weights,
        knots=spline.knots,
        breaks=breaks,
        corridor_indices=corridor_indices,
        constraint_labels=np.asarray(labels),
        use_system_envelope=use_system_envelope,
        corridor_physical_offsets=np.asarray(corridor_physical_offsets),
        constraint_families=np.asarray(family),
        **velocity_model.npz_fields(),
    )
    return matrices
