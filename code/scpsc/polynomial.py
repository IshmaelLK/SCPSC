"""Actual polynomial coefficients for QP output; no optimizer or validator."""
from dataclasses import dataclass
import math
import numpy as np

def derivative_basis(t, order=0):
    t = np.asarray(t, dtype=float)
    out = np.zeros(t.shape + (8,))
    for k in range(order, 8):
        out[..., k] = math.factorial(k) / math.factorial(k-order) * t**(k-order)
    return out


@dataclass
class PolynomialTrajectory:
    """Actual ascending physical-time coefficients, never a MINCO reconstruction."""
    coeffs: np.ndarray
    durations: np.ndarray
    head: np.ndarray
    tail: np.ndarray
    geometry_mode: str = 'configuration'

    def __post_init__(self):
        self.coeffs = np.asarray(self.coeffs, dtype=float).copy()
        self.durations = np.asarray(self.durations, dtype=float).copy()
        self.head = np.asarray(self.head, dtype=float).copy()
        self.tail = np.asarray(self.tail, dtype=float).copy()
        if (self.coeffs.shape != (len(self.durations), 8, 3) or len(self.durations) == 0
                or self.head.shape != (3, 4) or self.tail.shape != (3, 4)
                or np.any(self.durations <= 0)
                or not all(np.isfinite(a).all() for a in (self.coeffs, self.durations, self.head, self.tail))):
            raise ValueError('CLASSIC_TRAJECTORY: invalid finite coefficients, durations or PVAJ dimensions')
        self.points = np.array([self.evaluate(i, self.durations[i], 0)[0]
                                for i in range(len(self.durations)-1)]).reshape(-1, 3)

    def evaluate(self, piece, local_time, max_order=6):
        if not 0 <= piece < len(self.durations) or not 0 <= max_order <= 7:
            raise ValueError('CLASSIC_TRAJECTORY: invalid piece or derivative order')
        t = np.asarray(local_time, dtype=float)
        if np.any(t < 0) or np.any(t > self.durations[piece]):
            raise ValueError('CLASSIC_TRAJECTORY: query outside the actual piece')
        return np.stack([derivative_basis(t, k) @ self.coeffs[piece]
                         for k in range(max_order+1)], axis=-2)

    def sample(self, times, max_order=6):
        t = np.asarray(times, dtype=float)
        ends = np.cumsum(self.durations)
        if np.any(t < 0) or np.any(t > ends[-1]):
            raise ValueError('CLASSIC_TRAJECTORY: query outside the actual curve')
        pieces = np.minimum(np.searchsorted(ends, t, side='left'), len(ends)-1)
        starts = np.r_[0., ends[:-1]]
        result = np.empty(t.shape + (max_order+1, 3))
        for i in range(len(ends)):
            mask = pieces == i
            # A stored cumulative seam is exactly the local end by definition;
            # subtracting its rounded cumulative start can exceed dt by one ulp.
            local = np.where(t[mask] == ends[i], self.durations[i], t[mask]-starts[i])
            result[mask] = self.evaluate(i, local, max_order)
        return result

    def snap_cost(self):
        nodes, weights = np.polynomial.legendre.leggauss(4)
        return float(sum(dt/2 * np.dot(weights, np.sum(
            self.evaluate(i, (nodes+1)*dt/2, 4)[:, 4]**2, axis=1))
            for i, dt in enumerate(self.durations)))

