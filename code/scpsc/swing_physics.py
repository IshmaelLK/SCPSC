"""Independent taut-rope IVP from Eq. (24), driven by the prescribed vehicle acceleration.

All integration settings below are fixed engineering choices, not paper limits.
The integrator performs no projection, damping, clipping or state reset.
"""

import itertools
from dataclasses import asdict, dataclass
from typing import TypedDict

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import DynamicsParams

DEFAULT_DYNAMICS = DynamicsParams()


class IntegrationDiagnostics(TypedDict, total=False):
    """Numerical diagnostics written by the independent forward integrator."""

    nfev: int
    integrated_intervals: int
    method: str
    rtol: float
    atol: list[float]
    projection: bool
    initial_state: list[float]
    time_coordinate: str


class SwingIntegrationError(RuntimeError):
    """A forward IVP failed, with its physical time retained."""

    def __init__(self, message, time_s, status="SWING_INTEGRATION_FAILED"):
        super().__init__(message)
        self.time_s = float(time_s)
        self.status = status


@dataclass(frozen=True)
class SwingSettings:
    """Adaptive integration accuracy and output sampling, not acceptance limits."""

    method: str = "DOP853"
    rtol: float = 1e-8
    atol_q: float = 1e-10
    atol_nu_radps: float = 1e-10
    output_step_s: float = 0.05

    def __post_init__(self):
        numeric = [value for key, value in asdict(self).items() if key != "method"]
        if not np.isfinite(numeric).all() or min(numeric) <= 0:
            raise ValueError("Swing settings require finite positive values")
        if self.method != "DOP853":
            raise ValueError("Swing method must be DOP853")


def tension_per_mass(q, nu, acceleration, params):
    """Axial tension per load mass along the integrated motion, in m/s²."""
    force = np.asarray(acceleration) + np.array([0.0, 0.0, params.g])
    return np.sum(q * force, axis=-1) + params.ell * np.sum(nu * nu, axis=-1)


def swing_rhs(time_s, state, acceleration, params):
    """Evaluate the first-order form for the current rope state and prescribed aQ."""
    state = np.asarray(state, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    if state.shape != (6,) or acceleration.shape != (3,):
        raise ValueError("Swing RHS requires state (6,) and acceleration (3,)")
    if not np.isfinite(state).all() or not np.isfinite(acceleration).all():
        raise SwingIntegrationError(
            "Nonfinite forward state or prescribed acceleration", time_s
        )
    q, nu = state[:3], state[3:]
    tension = float(tension_per_mass(q, nu, acceleration, params))
    if not np.isfinite(tension) or tension <= 0:
        raise SwingIntegrationError(
            "Nonpositive/nonfinite tension per mass: " + str(tension),
            time_s,
            "TAUT_BRANCH_FAILED",
        )
    force = acceleration + np.array([0.0, 0.0, params.g])
    nu_dot = (force - q * np.dot(q, force)) / params.ell - np.dot(nu, nu) * q  # Eq. (24), nu = dq/dt.
    return np.r_[nu, nu_dot]


def integrate_swing(
    acceleration_piece,
    breaks,
    times,
    params=DEFAULT_DYNAMICS,
    settings=None,
    initial_state=None,
):
    """Integrate each exact stored interval, carrying the final state unchanged.

    acceleration_piece(index, local_time_s) evaluates exact coefficients. Each
    interval uses normalized integration time in [0,1], so one-ULP physical
    intervals are retained without SciPy's absolute-time minimum-step failure.
    initial_state exists for separately labeled mathematical fixtures only; the
    saved-run validator always uses the paper's e3,0 initial condition.
    """
    settings = settings or SwingSettings()
    breaks, times = np.asarray(breaks, dtype=float), np.asarray(times, dtype=float)
    if (
        breaks.ndim != 1
        or len(breaks) < 2
        or not np.isfinite(breaks).all()
        or breaks[0] != 0
        or np.any(np.diff(breaks) <= 0)
    ):
        raise ValueError(
            "Swing integration requires exact, finite, strictly increasing breaks from zero"
        )
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or times[0] != 0
        or times[-1] != breaks[-1]
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "Swing output times must strictly increase and include both endpoints"
        )
    state = np.array(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] if initial_state is None else initial_state,
        dtype=float,
    )
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("Swing initial state must be a finite six-vector")
    atol = np.r_[np.full(3, settings.atol_q), np.full(3, settings.atol_nu_radps)]
    sampled = np.empty((len(times), 6))
    sampled[0] = state
    diagnostics: IntegrationDiagnostics = {
        "nfev": 0,
        "integrated_intervals": 0,
    }
    for piece, (left, right) in enumerate(itertools.pairwise(breaks)):
        duration = right - left

        def rhs(
            local_fraction,
            current,
            interval_duration=duration,
            interval_start=left,
            interval_index=piece,
        ):
            local = local_fraction * interval_duration
            physical = interval_start + local
            acceleration = acceleration_piece(interval_index, local)
            derivative = swing_rhs(physical, current, acceleration, params)
            return derivative * interval_duration

        solution = solve_ivp(
            rhs,
            (0.0, 1.0),
            state,
            method=settings.method,
            rtol=settings.rtol,
            atol=atol,
            dense_output=True,
        )
        if not solution.success:
            raise SwingIntegrationError(
                solution.message, left + solution.t[-1] * duration
            )
        mask = (times > left) & (times <= right)
        sampled[mask] = solution.sol((times[mask] - left) / duration).T
        state = solution.y[:, -1].copy()
        diagnostics["nfev"] += solution.nfev
        diagnostics["integrated_intervals"] += 1
    if not np.isfinite(sampled).all():
        index = int(np.flatnonzero(~np.isfinite(sampled).all(axis=1))[0])
        raise SwingIntegrationError("Nonfinite forward output state", times[index])
    diagnostics.update(
        {
            "method": settings.method,
            "rtol": settings.rtol,
            "atol": atol.tolist(),
            "projection": False,
            "initial_state": sampled[0].tolist(),
            "time_coordinate": "normalized independently on every stored interval",
        }
    )
    return {"q": sampled[:, :3], "nu": sampled[:, 3:], "diagnostics": diagnostics}
