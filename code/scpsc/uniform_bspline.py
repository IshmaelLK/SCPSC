"""Degree-seven clamped uniform B-splines and exact polynomial restrictions.

All derivative matrices use physical seconds.  Bezier extraction and power
conversion are algebraic changes of basis; neither samples nor refits a curve.
"""
from math import comb, factorial

import numpy as np
from scipy.interpolate import BSpline


class UniformBSpline:
    degree = 7

    def __init__(self, total_time, n_control=16):
        self.total_time = float(total_time)
        self.n_control = int(n_control)
        if self.n_control != n_control or self.n_control < 8:
            raise ValueError('n_control must be an integer >= 8')
        if not np.isfinite(self.total_time) or self.total_time <= 0:
            raise ValueError('total_time must be finite and positive')
        self.breaks = np.linspace(0., self.total_time, self.n_control - 6)  # Eq. (18), h = T / (n_control - 7).
        self.knots = np.r_[np.zeros(8), self.breaks[1:-1],
                           np.full(8, self.total_time)]
        self.basis = BSpline(self.knots, np.eye(self.n_control), 7,
                             extrapolate=False, axis=0)

    def matrix(self, times, derivative=0):
        """B^(derivative)(t), ending in n_control; no time extrapolation."""
        times = np.asarray(times, dtype=float)
        if (not isinstance(derivative, (int, np.integer)) or
                not 0 <= derivative <= 7):
            raise ValueError('derivative must be an integer in [0, 7]')
        if (not np.isfinite(times).all() or np.any(times < 0.) or
                np.any(times > self.total_time)):
            raise ValueError('spline evaluation lies outside [0, total_time]')
        return self.basis(times, nu=derivative)  # Eq. (18), physical-time basis derivative.

    def power_matrix(self, start, end):
        """Ascending physical local-time powers on one polynomial subinterval.

        The row k is B^(k)(start)/k!, so p(t)=sum a[k]*(t-start)^k.
        A request that crosses a simple knot is not one polynomial.
        """
        start, end = float(start), float(end)
        if not 0. <= start < end <= self.total_time:
            raise ValueError('power restriction requires a positive in-domain interval')
        if np.any((self.breaks > start) & (self.breaks < end)):
            raise ValueError('power restriction crosses a B-spline knot')
        return np.stack([self.matrix(start, k) / factorial(k) for k in range(8)])

    def bezier_matrix(self, start, end):
        """Eight Bezier rows R with p_Bezier=R @ control_points.

        Convert the exact Taylor polynomial on u=(t-start)/(end-start):
        u^k=sum_{j=k}^7 comb(j,k)/comb(7,k) Bernstein_j^7(u).
        """
        power = self.power_matrix(start, end)
        power = power * (float(end) - float(start)) ** np.arange(8)[:, None]
        change = np.zeros((8, 8))
        for j in range(8):
            for k in range(j + 1):
                change[j, k] = comb(j, k) / comb(7, k)
        return change @ power

    def derivative_bezier_matrix(self, start, end, derivative=1):
        """Convex-hull derivative controls, with the physical-time factor."""
        if (not isinstance(derivative, (int, np.integer)) or
                not 0 <= derivative <= 7):
            raise ValueError('derivative must be an integer in [0, 7]')
        # Extract the differentiated polynomial directly.  This is identical
        # to repeated degree/duration differences of the position controls,
        # but avoids cancellation if two common-refinement breaks are close.
        power = self.power_matrix(start, end)
        degree = 7 - derivative
        duration = float(end) - float(start)
        power = np.stack([power[k + derivative] * factorial(k + derivative) /
                          factorial(k) * duration**k for k in range(degree + 1)])
        change = np.zeros((degree + 1, degree + 1))
        for j in range(degree + 1):
            for k in range(j + 1):
                change[j, k] = comb(j, k) / comb(degree, k)
        return change @ power

    def power_coefficients(self, controls, breaks=None):
        """Return [piece, power=0..7, xyz] for the unchanged curve."""
        controls = np.asarray(controls, dtype=float)
        if controls.shape != (self.n_control, 3) or not np.isfinite(controls).all():
            raise ValueError('controls must be finite [n_control, 3]')
        breaks = self.breaks if breaks is None else np.asarray(breaks, dtype=float)
        if (breaks.ndim != 1 or breaks[0] != 0. or breaks[-1] != self.total_time
                or np.any(np.diff(breaks) <= 0.)):
            raise ValueError('breaks must partition the complete spline domain')
        return np.stack([self.power_matrix(a, b) @ controls
                         for a, b in zip(breaks[:-1], breaks[1:])])

    def snap_matrix(self):
        """Integral B^(4).T B^(4) by four-point Gauss on each actual span.

        Snap is cubic on each degree-seven polynomial span.  Its squared norm
        has degree six, so this quadrature is exact in exact arithmetic.
        """
        nodes, weights = np.polynomial.legendre.leggauss(4)
        result = np.zeros((self.n_control, self.n_control))
        for start, end in zip(self.breaks[:-1], self.breaks[1:]):
            times = start + (nodes + 1.) * (end - start) / 2.
            derivative = self.matrix(times, 4)
            result += derivative.T @ (derivative * (weights * (end - start) / 2.)[:, None])
        return (result + result.T) / 2.
