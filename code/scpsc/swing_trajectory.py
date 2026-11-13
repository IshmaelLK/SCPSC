"""Exact saved coefficient input for swing verification; no planner or geometry IO."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dynamics import DynamicsParams, RecoveryDomainError, recover
from .polynomial import PolynomialTrajectory
from .specific_force import TRAJECTORY_VERSION


@dataclass
class SwingTrajectoryInput:
    trajectory: PolynomialTrajectory
    breaks: np.ndarray
    params: DynamicsParams
    identity: dict

    @classmethod
    def load(cls, run_dir, params):
        with np.load(Path(run_dir) / "trajectory.npz", allow_pickle=False) as saved:
            trajectory = PolynomialTrajectory(
                saved["coefficients"],
                saved["durations"],
                saved["head"],
                saved["tail"],
                "fixed",
            )
            breaks = saved["breaks"].copy()
            identity = {
                key: str(saved[key].item())
                for key in (
                    "trajectory_version",
                    "source_geometry_version",
                    "source_geometry_id",
                )
            }
        if identity["trajectory_version"] != TRAJECTORY_VERSION:
            raise ValueError(
                "SWING_TRAJECTORY_VERSION_MISMATCH: expected " + TRAJECTORY_VERSION
            )
        if (
            breaks.shape != (len(trajectory.durations) + 1,)
            or not np.isfinite(breaks).all()
            or breaks[0] != 0.0
            or np.any(np.diff(breaks) <= 0.0)
            or not np.array_equal(np.diff(breaks), trajectory.durations)
        ):
            raise ValueError(
                "SWING_COMMON_PARTITION: exact positive breaks/durations required"
            )
        return cls(trajectory, breaks, params, identity)

    def state(self, times):
        """Physical-time derivatives; saved breaks, never reconstructed cumulative sums."""
        times = np.asarray(times, dtype=float)
        if (
            not np.isfinite(times).all()
            or np.any(times < 0)
            or np.any(times > self.breaks[-1])
        ):
            raise ValueError("Swing reference time outside saved partition")
        pieces = np.minimum(
            np.searchsorted(self.breaks[1:], times, side="left"), len(self.breaks) - 2
        )
        result = np.empty(times.shape + (6, 3))
        for piece in np.unique(pieces):
            mask = pieces == piece
            local = np.where(
                times[mask] == self.breaks[piece + 1],
                self.trajectory.durations[piece],
                times[mask] - self.breaks[piece],
            )
            result[mask] = self.trajectory.evaluate(int(piece), local, 5)
        return result

    def reference(self, times):
        return {
            key: value.detach().cpu().numpy()
            for key, value in recover(self.state(times), self.params).items()
        }

    def acceleration_piece(self, piece, local_time_s):
        """Recover only aQ=aL+ell*q'', using u=||u||q through order two."""
        state = self.trajectory.evaluate(piece, local_time_s, 4)
        u = state[2] + np.array([0.0, 0.0, self.params.g])
        jerk, snap = state[3], state[4]
        magnitude = np.linalg.norm(u)
        if not np.isfinite(state[2:]).all() or not np.isfinite(magnitude):
            raise RecoveryDomainError(
                "RECOVERY_DOMAIN_ERROR: nonfinite payload derivatives"
            )
        if magnitude <= self.params.specific_force_guard:
            raise RecoveryDomainError(
                "RECOVERY_DOMAIN_ERROR: specific-force normalization below guard"
            )
        q = u / magnitude
        magnitude1 = np.dot(q, jerk)
        q1 = (jerk - magnitude1 * q) / magnitude
        magnitude2 = (
            np.dot(jerk, jerk) + np.dot(u, snap) - magnitude1**2
        ) / magnitude
        q2 = (snap - 2 * magnitude1 * q1 - magnitude2 * q) / magnitude
        return state[2] + self.params.ell * q2


def output_grid(breaks, step):
    """Every stored break is retained exactly, including one-ULP neighbors."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Swing output step must be finite positive seconds")
    return np.unique(np.r_[np.arange(0.0, breaks[-1], step), breaks])
