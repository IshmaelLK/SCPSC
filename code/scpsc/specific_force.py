"""Post-QP Bernstein certificate for the trajectory-layer specific-force cone.

No sampling, curve changes, interval subdivision or solver calls occur here.
The floating-point guard is a scale-dependent arithmetic/basis-change guard,
not a negative feasibility tolerance. Margins inside it are not certified.
"""
from math import comb, factorial
from time import perf_counter

import numpy as np


TRAJECTORY_VERSION = 'SCPSC_TRAJECTORY_LAYER_V1'


def common_partition(spline, switches):
    """Keep every distinct binary64 knot/switch, including one-ulp intervals."""
    switches = np.asarray(switches, dtype=float)
    if (switches.ndim != 1 or not np.isfinite(switches).all() or
            np.any(switches < 0.) or np.any(switches > spline.total_time)):
        raise ValueError('specific-force switches must be finite and in the spline domain')
    return np.unique(np.r_[0., spline.breaks, switches, spline.total_time])


def power_to_bernstein(coeff, duration):
    """Algebraic conversion of ascending, physical-local-time coefficients."""
    coeff = np.asarray(coeff, dtype=float)
    duration = float(duration)
    if (coeff.ndim != 2 or coeff.shape[1] != 3 or len(coeff) == 0 or
            not np.isfinite(coeff).all() or not np.isfinite(duration) or duration <= 0.):
        raise ValueError('Bernstein extraction requires finite coefficients and positive seconds')
    degree = len(coeff) - 1
    change = np.zeros((degree + 1, degree + 1))
    for j in range(degree + 1):
        for k in range(j + 1):
            change[j, k] = comb(j, k) / comb(degree, k)
    return change @ (coeff * duration**np.arange(degree + 1)[:, None])


def _parameters(g, gamma, theta):
    g, gamma, theta = float(g), float(gamma), float(theta)
    if not np.isfinite(g) or g <= 0.:
        raise ValueError('specific-force g must be finite and positive')
    if not np.isfinite(gamma) or not 0. < gamma < 1.:
        raise ValueError('specific-force gamma must lie strictly between zero and one')
    if not np.isfinite(theta) or not 0. <= theta < np.pi / 2.:
        raise ValueError('specific-force theta must lie in [0, pi/2)')
    return g, gamma, theta


def _report(acceleration, arithmetic_scale, breaks, *, g, gamma, theta,
            trajectory_version, source_geometry_version, source_geometry_id,
            extraction, started):
    g, gamma, theta = _parameters(g, gamma, theta)
    acceleration = np.asarray(acceleration, dtype=float)
    arithmetic_scale = np.asarray(arithmetic_scale, dtype=float)
    if (acceleration.shape != (len(breaks) - 1, 6, 3) or
            arithmetic_scale.shape != acceleration.shape or
            not np.isfinite(acceleration).all() or not np.isfinite(arithmetic_scale).all()):
        raise ValueError('nonfinite or invalid five-degree acceleration control points')
    controls = acceleration.copy()
    controls[:, :, 2] += g  # Eq. (22), add gravity to the six acceleration controls.
    tangent = np.tan(theta)
    # Eqs. (5), (23): nonnegative vertical and cone margins on every control suffice for the full interval.
    mz = controls[:, :, 2] - gamma*g
    mc = tangent*controls[:, :, 2] - np.linalg.norm(controls[:, :, :2], axis=2)
    # Guard dot products, basis conversion, gravity addition, norm and
    # subtraction. It grows with cancellation, instead of forgiving a fixed
    # negative margin. This is a numerical guard, not interval arithmetic.
    eps = np.finfo(float).eps
    error_u = 128.*eps*arithmetic_scale
    error_u[:, :, 2] += 4.*eps*(np.abs(acceleration[:, :, 2]) + g)
    guard_z = error_u[:, :, 2] + 4.*eps*(np.abs(controls[:, :, 2]) + gamma*g)
    guard_c = (abs(tangent)*error_u[:, :, 2] + np.linalg.norm(error_u[:, :, :2], axis=2) +
               8.*eps*(abs(tangent)*np.abs(controls[:, :, 2]) +
                       np.linalg.norm(controls[:, :, :2], axis=2)))
    if not all(np.isfinite(x).all() for x in (controls, mz, mc, guard_z, guard_c)):
        raise ValueError('nonfinite specific-force margins or arithmetic guards')
    nonnegative = (mz >= 0.) & (mc >= 0.)
    resolved = (mz > guard_z) & (mc > guard_c)
    # At theta=0 an identically zero horizontal acceleration is exact in
    # both extraction paths: every contributing horizontal term is zero.
    exact_axis = (theta == 0.) & np.all(arithmetic_scale[:, :, :2] == 0., axis=2)
    resolved |= exact_axis & (mz > guard_z) & (mc == 0.)
    passed = bool(np.all(nonnegative & resolved))
    uncertain = bool(not passed and np.all((mz >= -guard_z) & (mc >= -guard_c)))
    index_z = tuple(int(i) for i in np.unravel_index(np.argmin(mz), mz.shape))
    index_c = tuple(int(i) for i in np.unravel_index(np.argmin(mc), mc.shape))
    return dict(status=('SPECIFIC_FORCE_CERTIFIED' if passed else
                        'SPECIFIC_FORCE_NUMERICALLY_UNCERTAIN' if uncertain else
                        'SPECIFIC_FORCE_NOT_CERTIFIED'),
        passed=passed, gamma=gamma, theta_rad=theta, g_mps2=g,
        trajectory_version=trajectory_version,
        source_geometry_version=source_geometry_version, source_geometry_id=source_geometry_id,
        extraction=extraction, derivative_degree=5, controls_per_interval=6,
        intervals=np.column_stack((breaks[:-1], breaks[1:])).tolist(),
        specific_force_controls_mps2=controls.tolist(),
        vertical_margins_mps2=mz.tolist(), cone_margins_mps2=mc.tolist(),
        minimum_vertical_margin_mps2=float(mz[index_z]),
        minimum_cone_margin_mps2=float(mc[index_c]),
        eta_z=float(np.min(mz/g)), eta_theta=float(np.min(mc/g)),
        worst_vertical=dict(interval=index_z[0], control=index_z[1],
            endpoints_s=breaks[index_z[0]:index_z[0]+2].tolist()),
        worst_cone=dict(interval=index_c[0], control=index_c[1],
            endpoints_s=breaks[index_c[0]:index_c[0]+2].tolist()),
        vertical_roundoff_guard_mps2=guard_z.tolist(), cone_roundoff_guard_mps2=guard_c.tolist(),
        numerical_policy=('nonnegative signed margins and strictly above scale-dependent '
            'binary64 roundoff guards; unresolved boundary values are not certified; '
            'exact zero horizontal polynomial is allowed at theta=0; not interval arithmetic'),
        negative_margin_tolerance=0., adaptive_subdivision=False, samples_used=0,
        seconds=perf_counter()-started)


def certify_spline(spline, controls, breaks, *, g, gamma, theta,
                   trajectory_version=TRAJECTORY_VERSION,
                   source_geometry_version=None, source_geometry_id=None):
    """Extract actual-second acceleration controls directly from the spline."""
    started = perf_counter()
    _parameters(g, gamma, theta)
    controls = np.asarray(controls, dtype=float)
    if controls.shape != (spline.n_control, 3) or not np.isfinite(controls).all():
        raise ValueError('specific-force spline controls must be finite [n_control,3]')
    partition = common_partition(spline, breaks)
    # Translation vanishes under differentiation; removing it first prevents
    # world-coordinate offsets from contaminating derivative matrix products.
    local = controls - controls[0]
    acceleration, scale = [], []
    for a, b in zip(partition[:-1], partition[1:]):
        matrix = spline.derivative_bezier_matrix(a, b, 2)  # Eq. (22), equivalent direct derivative extraction.
        acceleration.append(matrix @ local)
        # Include cancellation within the Taylor-to-Bernstein basis change,
        # not just the final matrix-vector product.
        derivative_power = np.stack([spline.matrix(a, k+2)/factorial(k)
                                     for k in range(6)])
        absolute_terms = np.abs(derivative_power) @ np.abs(local)
        scale.append(power_to_bernstein(absolute_terms, b-a))
    return _report(acceleration, scale, partition, g=g, gamma=gamma, theta=theta,
        trajectory_version=trajectory_version, source_geometry_version=source_geometry_version,
        source_geometry_id=source_geometry_id, extraction='direct spline derivative Bernstein matrix',
        started=started)


def certify_power(coefficients, durations, *, breaks=None, g, gamma, theta,
                  trajectory_version=TRAJECTORY_VERSION,
                  source_geometry_version=None, source_geometry_id=None):
    """Recompute from saved original polynomial coefficients, never a pass flag.

    Each polynomial must already be restricted to the complete common
    partition. Explicit breaks avoid losing a short interval by cumsum.
    """
    started = perf_counter()
    _parameters(g, gamma, theta)
    coefficients = np.asarray(coefficients, dtype=float)
    durations = np.asarray(durations, dtype=float)
    if (durations.ndim != 1 or len(durations) == 0 or
            coefficients.shape != (len(durations), 8, 3) or
            not np.isfinite(coefficients).all() or not np.isfinite(durations).all() or
            np.any(durations <= 0.)):
        raise ValueError('specific-force powers require finite [piece,8,3] and positive durations')
    partition = (np.r_[0., np.cumsum(durations)] if breaks is None else
                 np.asarray(breaks, dtype=float))
    if (partition.shape != (len(durations)+1,) or not np.isfinite(partition).all() or
            partition[0] != 0. or np.any(np.diff(partition) <= 0.) or
            (breaks is not None and not np.array_equal(np.diff(partition), durations))):
        raise ValueError('saved common partition must retain all positive original intervals')
    acceleration, scale = [], []
    for coeff, dt in zip(coefficients, durations):
        derivative = (np.arange(2, 8)*np.arange(1, 7))[:, None]*coeff[2:]
        acceleration.append(power_to_bernstein(derivative, dt))
        scale.append(power_to_bernstein(np.abs(derivative), dt))
    return _report(acceleration, scale, partition, g=g, gamma=gamma, theta=theta,
        trajectory_version=trajectory_version, source_geometry_version=source_geometry_version,
        source_geometry_id=source_geometry_id, extraction='independent saved power differentiation and Bernstein conversion',
        started=started)

