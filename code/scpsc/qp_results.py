"""Convert the unchanged QP candidate and prepare numerical result diagnostics."""
import numpy as np
import scipy


def initial_diagnostics(
    inputs,
    constraints,
    objective,
    reduced,
    raw,
    settings,
    qp_config,
    osqp,
    assembly_seconds,
    preprocess_seconds,
    setup_seconds,
    solve_seconds,
    candidate_exists,
):
    origin = inputs.origin
    spline = inputs.spline
    Aeq = constraints.Aeq
    rank = constraints.rank
    singular = constraints.singular
    eq_consistency = constraints.eq_consistency
    velocity_model = constraints.velocity_model
    use_system_envelope = constraints.use_system_envelope
    planning_mode = constraints.planning_mode
    A = constraints.A
    breaks = constraints.breaks
    eigenvalues = objective.eigenvalues
    objective_scale = objective.objective_scale
    alpha_s = objective.alpha_s
    alpha_r = objective.alpha_r
    snap_ref = objective.snap_ref
    length_ref = objective.length_ref
    raw_status = {
        name: value for name, value in vars(raw.info).items() if not name.startswith("_") and not callable(value)
    }
    diagnostics = dict(
        osqp=raw_status,
        settings=settings,
        wall_clock_limit_enabled=False,
        dependencies=dict(numpy=np.__version__, scipy=scipy.__version__, osqp=osqp.__version__),
        endpoint_rows=len(Aeq),
        endpoint_rank=rank,
        endpoint_scaled_singular_values=singular.tolist(),
        endpoint_scaled_consistency_residual=eq_consistency,
        hessian_minimum_eigenvalue=float(eigenvalues[0]),
        objective_scale=objective_scale,
        origin=origin.tolist(),
        velocity_constraint=velocity_model.metadata(),
        use_system_envelope=use_system_envelope,
        planning_geometry_mode=planning_mode,
        alpha_s=alpha_s,
        alpha_r=alpha_r,
        snap_ref=snap_ref,
        length_ref=length_ref,
        reference_weight_rule="explicit"
        if "reference_weights" in qp_config
        else "normalized trapezoid in fixed reference time",
        assembly_seconds=assembly_seconds,
        preprocess_seconds=preprocess_seconds,
        setup_seconds=setup_seconds,
        solve_seconds=solve_seconds,
        primal_check_seconds=0.0,
        curve_conversion_seconds=0.0,
        report_statistics_seconds=None,
        variable_count=3 * spline.n_control,
        constraint_count=len(A),
        piece_count=len(breaks) - 1,
        reduced_variable_count=len(reduced["q"]),
        original_inequality_rows=len(A) - len(Aeq),
        retained_inequality_rows=reduced["retained_inequality_rows"],
        constraint_rows_dropped=0,
        augmented_endpoint_rows=len(reduced["Eaug"]),
        augmented_endpoint_rank=len(reduced["Eaug"]),
        augmented_endpoint_scaled_singular_values=reduced["equality_singular_values"],
        whiten_identity_error=reduced["whiten_identity_error"],
        terminal_face=reduced["face"],
        constant_rows=reduced["constant_rows"],
        constant_original_rows=reduced["constant_original_rows"],
        near_constant_rows=reduced["near_constant_rows"],
        solver_coordinates="reduced z; original x=xbase+M z",
        infeasibility_certificate_space="reduced problem only",
        original_coordinate_candidate_available=bool(candidate_exists),
    )
    return diagnostics


def convert_candidate(inputs, constraints, x):
    spline = inputs.spline
    origin = inputs.origin
    breaks = constraints.breaks
    local_controls = x.reshape(spline.n_control, 3).copy()
    controls = local_controls + origin
    # Convert in local coordinates before adding the world origin only to
    # the constant term; this avoids subtracting large map coordinates.
    coefficients = spline.power_coefficients(x.reshape(spline.n_control, 3), breaks)
    coefficients[:, 0] += origin
    return local_controls, controls, coefficients


def objective_statistics(inputs, objective, x):
    spline = inputs.spline
    reference_points = inputs.reference_points
    origin = inputs.origin
    B = objective.B
    snap = objective.snap
    reference_weights = objective.reference_weights
    alpha_s = objective.alpha_s
    alpha_r = objective.alpha_r
    snap_ref = objective.snap_ref
    length_ref = objective.length_ref
    tracking_value = B @ x.reshape(spline.n_control, 3) - (reference_points - origin)
    snap_integral = float(np.sum(x.reshape(spline.n_control, 3) * (snap @ x.reshape(spline.n_control, 3))))
    tracking_cost = float(np.sum(reference_weights[:, None] * tracking_value**2))
    reported_objective = alpha_s / snap_ref**2 * snap_integral + alpha_r / length_ref**2 * tracking_cost
    return snap_integral, tracking_cost, reported_objective


def candidate_diagnostics(constraints, checked, snap_integral, tracking_cost, reported_objective):
    Aeq = constraints.Aeq
    labels = constraints.labels
    family = constraints.family
    upper = constraints.upper
    corridor_physical_offsets = constraints.corridor_physical_offsets
    velocity_bounds = constraints.velocity_model.bounds
    value = checked.value
    violation = checked.violation
    eq_residual = checked.eq_residual
    eq_max = checked.eq_max
    ineq_max = checked.ineq_max
    equality_tolerance = checked.equality_tolerance
    inequality_tolerance = checked.inequality_tolerance
    return dict(
        equality_maximum_absolute_SI=eq_max,
        inequality_maximum_violation_SI=ineq_max,
        endpoint_residuals_SI=dict(zip(labels[: len(Aeq)], eq_residual.tolist())),
        family_maximum_violation_SI={
            name: float(np.max(violation[np.asarray(family) == name])) for name in sorted(set(family))
        },
        equality_tolerance=equality_tolerance,
        inequality_tolerance=inequality_tolerance,
        planning_corridor_margin_lower_bound_m=float(
            np.min(upper[np.asarray(family) == "corridor"] - value[np.asarray(family) == "corridor"])
        ),
        fixed_envelope_corridor_margin_lower_bound_m=float(
            np.min(
                upper[np.asarray(family) == "corridor"]
                - value[np.asarray(family) == "corridor"]
                - np.asarray(corridor_physical_offsets)
            )
        ),
        load_speed_certificate_upper_mps=float(
            np.linalg.norm(
                value[np.asarray(family) == "velocity"].reshape(-1, len(velocity_bounds))[:, :3], axis=1
            ).max()
        ),
        snap_integral=snap_integral,
        reference_squared_error=tracking_cost,
        qp_objective=reported_objective,
    )
