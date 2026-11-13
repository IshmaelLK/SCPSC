"""One reduced OSQP solve and acceptance against the original QP rows."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse


@dataclass
class PrimalCheck:
    """One original-coordinate residual evaluation of the OSQP candidate."""

    x: np.ndarray
    value: np.ndarray
    violation: np.ndarray
    eq_residual: np.ndarray
    eq_max: float
    ineq_max: float
    equality_tolerance: float
    inequality_tolerance: float
    numerical_pass: bool


def solver_settings(qp_config):
    iteration_limit = qp_config.get("max_iter", 100000)
    if qp_config.get("time_limit") is not None:
        import warnings

        warnings.warn("ubsqp.time_limit is retired and ignored; QP has no wall-clock limit", FutureWarning)
    if iteration_limit is None:
        raise ValueError("max_iter must retain a positive finite iteration limit")
    # Native OSQP 1.1.3 retains IEEE +inf (verified by a setup/solve test).
    # JSON diagnostics represent this setting as null, with an explicit flag.
    settings = dict(
        verbose=False,
        eps_abs=float(qp_config.get("eps_abs", 1e-8)),
        eps_rel=float(qp_config.get("eps_rel", 1e-8)),
        max_iter=int(iteration_limit),
        time_limit=float("inf"),
        polishing=True,
        scaled_termination=False,
        adaptive_rho=True,
        check_termination=25,
    )
    return settings


def solve_reduced(reduced, settings, osqp):
    solver = osqp.OSQP()
    setup_started = perf_counter()
    solver.setup(
        P=sparse.csc_matrix(np.triu(reduced["P"])),
        q=reduced["q"],
        A=sparse.csc_matrix(reduced["A"]),
        l=reduced["l"],
        u=reduced["u"],
        **settings,
    )
    setup_seconds = perf_counter() - setup_started
    solve_started = perf_counter()
    raw = solver.solve(raise_error=False)
    solve_seconds = perf_counter() - solve_started
    return raw, setup_seconds, solve_seconds


def check_original_candidate(raw, reduced, constraints, qp_config, status_val):
    A = constraints.A
    lower = constraints.lower
    upper = constraints.upper
    Aeq = constraints.Aeq
    beq = constraints.beq
    x = reduced["xbase"] + reduced["M"] @ np.asarray(raw.x)
    # This is the sole original-row pass, including equalities dropped by
    # affine elimination. Trust OSQP for its dual convergence status; do not
    # reconstruct multipliers and audit the same KKT equations a second time.
    value = A @ x
    violation = np.maximum(np.maximum(lower - value, value - upper), 0.0)
    eq_residual = value[: len(Aeq)] - beq
    eq_max = float(np.max(np.abs(eq_residual)))
    ineq_max = float(np.max(violation[len(Aeq) :]))
    equality_tolerance = float(qp_config.get("equality_tolerance", 1e-6))
    inequality_tolerance = float(qp_config.get("inequality_tolerance", 1e-6))
    numerical_pass = (
        status_val in (1, 2)
        and np.isfinite(x).all()
        and np.isfinite(value).all()
        and eq_max <= equality_tolerance
        and ineq_max <= inequality_tolerance
    )
    return PrimalCheck(
        x,
        value,
        violation,
        eq_residual,
        eq_max,
        ineq_max,
        equality_tolerance,
        inequality_tolerance,
        numerical_pass,
    )


def result_status(numerical_pass, status_val):
    if numerical_pass:
        status = "QP_NUMERICAL_PASS"
    elif status_val in (3, 4):
        status = "QP_PRIMAL_INFEASIBLE"
    elif status_val in (5, 6):
        status = "QP_DUAL_INFEASIBLE"
    elif status_val == 7:
        status = "QP_ITERATION_LIMIT"
    elif status_val == 8:
        status = "QP_EXTERNAL_TIME_LIMIT"
    else:
        status = "QP_NUMERICAL_FAIL"
    return status
