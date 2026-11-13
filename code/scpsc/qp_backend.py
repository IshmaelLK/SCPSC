"""Build, solve and certify the fixed-time degree-seven QP exactly once.

This module owns solve ordering and timing; model assembly, the OSQP call,
original-row acceptance and result conversion have separate implementations.
QP numerical passage is not transportation success.
"""
from time import perf_counter

import numpy as np

from .qp_face_reduction import reduce_qp
from .qp_model import validate_inputs, assemble_constraints, assemble_objective, assembled_matrices
from .qp_solver import solver_settings, solve_reduced, check_original_candidate, result_status
from .qp_results import initial_diagnostics, convert_candidate, objective_statistics, candidate_diagnostics
from .qp_result_io import json_value, save_assembled_matrices, save_qp_result


def solve_fixed_time_qp(
    reference_times,
    reference_points,
    corridors,
    endpoint_data,
    params,
    limits,
    qp_config,
    approach=None,
    output_dir=None,
    endpoint_pvaj=None,
):
    """Build and solve once, returning the candidate and explicit QP status.

    reference_times start at zero; corridor j applies to its j-th positive
    interval.  endpoint_data contains the two original equilibrium outputs.
    approach uses start_time/end_time/v_down and optionally direction_xy and
    beta_max_rad.  All choices are fixed before this call.
    """
    try:
        import osqp
    except ImportError as error:
        raise RuntimeError("OSQP_DEPENDENCY_UNAVAILABLE: install osqp==1.1.3") from error
    started = perf_counter()
    inputs = validate_inputs(
        reference_times, reference_points, corridors, endpoint_data, params, qp_config, endpoint_pvaj
    )
    constraints = assemble_constraints(inputs, corridors, limits, qp_config, approach)
    objective = assemble_objective(inputs, limits, qp_config)
    settings = solver_settings(qp_config)
    matrices = assembled_matrices(inputs, constraints, objective)
    assembly_seconds = perf_counter() - started
    if output_dir is not None:
        # Preserve the assembled problem before preprocessing can fail.
        save_assembled_matrices(output_dir, matrices)
    preprocess_started = perf_counter()
    reduced = reduce_qp(matrices, endpoint_data, approach)
    preprocess_seconds = perf_counter() - preprocess_started
    raw, setup_seconds, solve_seconds = solve_reduced(reduced, settings, osqp)
    status_val = int(raw.info.status_val)
    # Infeasibility certificates are not coordinates. Limited primal iterates
    # can be preserved only as explicitly failed curves.
    candidate_status = status_val in (1, 2, 7, 8)
    candidate_exists = candidate_status and raw.x is not None and np.isfinite(raw.x).all()
    diagnostics = initial_diagnostics(
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
    )
    controls = local_controls = coefficients = None
    x, y = np.empty(0), np.empty(0)
    numerical_pass = False
    if candidate_exists:
        check_started = perf_counter()
        checked = check_original_candidate(raw, reduced, constraints, qp_config, status_val)
        x, numerical_pass = checked.x, checked.numerical_pass
        diagnostics["primal_check_seconds"] = perf_counter() - check_started
        diagnostics["original_row_evaluations"] = 1
        diagnostics["dual_lift_performed"] = False
        conversion_started = perf_counter()
        local_controls, controls, coefficients = convert_candidate(inputs, constraints, x)
        diagnostics["curve_conversion_seconds"] = perf_counter() - conversion_started
        # Report-only evaluation stays outside the original-row/curve timings.
        statistics_started = perf_counter()
        snap_integral, tracking_cost, reported_objective = objective_statistics(inputs, objective, x)
        diagnostics["report_statistics_seconds"] = perf_counter() - statistics_started
        diagnostics.update(
            candidate_diagnostics(constraints, checked, snap_integral, tracking_cost, reported_objective)
        )
    status = result_status(numerical_pass, status_val)
    diagnostics.update(status=status, numerical_pass=bool(numerical_pass), total_seconds=perf_counter() - started)
    diagnostics = json_value(diagnostics)
    if output_dir is not None:
        save_qp_result(output_dir, matrices, reduced, raw, x, y, diagnostics)
    return dict(
        status=status,
        numerical_pass=bool(numerical_pass),
        controls=controls,
        local_controls=local_controls,
        knots=inputs.spline.knots,
        breaks=constraints.breaks,
        coefficients=coefficients,
        durations=np.diff(constraints.breaks),
        corridor_indices=constraints.corridor_indices,
        diagnostics=diagnostics,
    )
