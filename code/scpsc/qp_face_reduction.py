"""Conditional terminal-face reduction for the frozen SCPSC degree-7 QP.

No solver is called here. The objective, total time, original inequalities and
endpoint conditions are preserved. The terminal closure is enabled only after
checking its geometric condition and its identities against the ORIGINAL rows.
Every inequality remains in the reduced problem, including near-constant rows.
"""
from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline


def _maxabs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x), initial=0.0))


def affine_chart(E: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    den = np.linalg.norm(E, axis=1)
    if np.any(den <= 0):
        raise ValueError('Zero equality row must be diagnosed, not silently removed.')
    U, s, V = linalg.svd(E / den[:, None], full_matrices=True)
    tol = max(E.shape) * np.finfo(float).eps * s[0]
    rank = int(np.sum(s > tol))
    if rank != len(E):
        raise ValueError('This candidate requires independent equalities; do not truncate rank.')
    p = V[:rank].T @ ((U[:, :rank].T @ (b / den)) / s[:rank])
    N = V[rank:].T
    if _maxabs(E @ p - b) > 1e-8 or _maxabs(E @ N) > 1e-8:
        raise ValueError('Equality chart failed its original-unit residual check.')
    return p, N, s


def terminal_face(m: Mapping[str, np.ndarray], endpoint_data: list[dict[str, Any]],
                  approach: Mapping[str, Any]) -> dict[str, Any]:
    """Prove two implied scalar equalities from terminal Bernstein cone rows.

    PVAJ(T) is stationary. E_T.T p4 = 0 and E_T.T p5=0
    are the original conditions. When the terminal cone excludes both +/-q_T,
    the final velocity Bernstein controls force p4(T)=p5(T)=0.
    Only q_T.T p4 = q_T.T p5 = 0 are new independent rows.
    """
    E, b = np.asarray(m['Aeq']), np.asarray(m['beq'])
    A = np.asarray(m['A']); T = float(m['reference_times'][-1])
    base = dict(enabled=False, reason='No applicable terminal directional cone.')
    if not approach or approach.get('direction_xy') is None:
        return base
    if float(approach['end_time']) != T or not 0 <= float(approach['start_time']) < T:
        return base
    beta = float(approach['beta_max_rad'])
    d = np.r_[np.asarray(approach['direction_xy'], dtype=float), 0.]
    if not 0 < beta < np.pi / 2 or abs(np.linalg.norm(d) - 1) > 1e-10:
        raise ValueError('Invalid original approach cone.')
    et = np.asarray(endpoint_data[1]['E'], dtype=float)
    q = np.cross(et[:, 0], et[:, 1]); q /= np.linalg.norm(q)
    if q[2] < 0: q = -q
    if _maxabs(et.T @ et - np.eye(2)) > 1e-10:
        raise ValueError('Endpoint tangent basis is not orthonormal.')
    # For q_z>0, +q violates descent. -q violates the slope inequality when
    # q_z+tan(beta)*(d.q)>0. These strict margins avoid a near-parallel case.
    cone_margin = q[2] + np.tan(beta) * float(d @ q)
    if min(q[2], cone_margin) <= 1e-8:
        return dict(enabled=False, reason='Cone-line exclusion condition not established.')
    ncontrol = E.shape[1] // 3
    basis = BSpline(m['knots'], np.eye(ncontrol), 7, extrapolate=False, axis=0)
    g = np.stack([np.outer(basis(T, nu=j), q).reshape(-1) for j in (4, 5)])
    labels = list(map(str, m['constraint_labels']))
    # Check all terminal VAJ zero rows rather than assuming their targets.
    for order in (1, 2, 3):
        for axis, name in enumerate('xyz'):
            i = labels.index(f'endpoint_1_d{order}_{name}')
            expected = np.outer(basis(T, nu=order), np.eye(3)[axis]).reshape(-1)
            if m['l'][i] != 0 or m['u'][i] != 0:
                return dict(enabled=False, reason='Stationary terminal PVAJ premise is absent.')
            if _maxabs(A[i] - expected) > 1e-10:
                raise ValueError('Terminal PVAJ row disagrees with the spline derivative.')
    piece = len(m['breaks']) - 2
    if float(m['breaks'][-2]) < float(approach['start_time']):
        return dict(enabled=False, reason='The last complete piece is not in the cone interval.')
    names = [f'approach_descent_{piece}_3', f'approach_slope_{piece}_3',
             f'approach_descent_{piece}_2', f'approach_slope_{piece}_2']
    ids = np.array([labels.index(label) for label in names], dtype=int)
    if not np.all(m['u'][ids] == 0):
        raise ValueError('Required terminal cone upper bounds are not exactly zero.')
    p0, N0, _ = affine_chart(E, b)
    G0 = g @ N0
    # For each original cone row, identify a = c4*g4+c5*g5 (modulo E x=b).
    # This checks the actual saved matrix, not just an abstract model.
    coefficients = linalg.lstsq(G0.T, (A[ids] @ N0).T)[0].T
    error = _maxabs(A[ids] @ N0 - coefficients @ G0)
    affine_error = _maxabs(A[ids] @ p0 - coefficients @ (g @ p0))
    if error > 1e-10 or affine_error > 1e-8:
        raise ValueError('Terminal face identity fails against original source rows.')
    if _maxabs(coefficients[:2, 1]) > 1e-10:
        raise ValueError('The snap-exposing rows still depend on the fifth derivative.')
    if not (coefficients[0, 0] < -1e-8 and coefficients[1, 0] > 1e-8
            and coefficients[2, 1] > 1e-8 and coefficients[3, 1] < -1e-8):
        raise ValueError('Opposite-sign implication coefficients are not safely resolved.')
    return dict(enabled=True, reason='Original cone rows imply two scalar terminal equalities.',
                extra_rows=g, q=q, cone_margin=float(cone_margin), source_rows=ids,
                row_coefficients=coefficients, identity_error=error,
                affine_identity_error=affine_error)


def reduce_qp(m: Mapping[str, np.ndarray], endpoint_data: list[dict[str, Any]],
              approach: Mapping[str, Any]) -> dict[str, Any]:
    H, f, A = [np.asarray(m[k], dtype=float) for k in ('H', 'f', 'A')]
    E, b = np.asarray(m['Aeq']), np.asarray(m['beq']); ne = len(E)
    face = terminal_face(m, endpoint_data, approach)
    Ea = np.vstack((E, face['extra_rows'])) if face['enabled'] else E.copy()
    ba = np.r_[b, 0., 0.] if face['enabled'] else b.copy()
    p, N, singular = affine_chart(Ea, ba)
    Hr = N.T @ H @ N; Hr = (Hr + Hr.T) / 2
    # A coordinate transformation, NOT an added regularizer or changed weight.
    L = linalg.cholesky(Hr, lower=True)
    W = linalg.solve_triangular(L.T, np.eye(len(Hr)), lower=False)
    xbase = p + N @ linalg.solve(Hr, -N.T @ (H @ p + f), assume_a='pos')
    M = N @ W
    Ar = A[ne:] @ M
    lr, ur = m['l'][ne:] - A[ne:] @ xbase, m['u'][ne:] - A[ne:] @ xbase
    norms = np.linalg.norm(Ar, axis=1)
    # Prove candidate constant rows belong to the augmented equality row span.
    # A small projection alone is not the proof. Keep every row, but represent
    # a proved constant exactly and record only the authorized roundoff repair.
    # PVAJ fixes the first/last four position Bezier controls and three
    # velocity controls. The proved terminal face fixes two more at T.
    # Do not canonicalize an arbitrary small row elsewhere in the curve.
    ids = []
    last_piece = len(m['breaks']) - 2
    terminal_order = 5 if face['enabled'] else 3
    for i, label in enumerate(m['constraint_labels'][ne:]):
        fields = str(label).split('_')
        if fields[0] in ('corridor', 'velocity'):
            piece, control = int(fields[1]), int(fields[3])
            position = fields[0] == 'corridor'
        elif fields[0] == 'approach':
            piece, control, position = int(fields[2]), int(fields[3]), False
        else:
            continue
        fixed = (piece == 0 and control <= (3 if position else 2)) or (
            piece == last_piece and control >= 7 - terminal_order)
        if fixed:
            if norms[i] > 1e-11:
                raise ValueError(f'Endpoint constant-row projection failed: {label}.')
            ids.append(i)
    ids = np.asarray(ids, dtype=int)
    den = np.linalg.norm(Ea, axis=1)
    weights = linalg.lstsq((Ea / den[:, None]).T, A[ne + ids].T)[0].T / den
    residual = np.max(np.abs(A[ne + ids] - weights @ Ea), axis=1, initial=0.)
    tolerance = 128 * np.finfo(float).eps * np.maximum(1., np.linalg.norm(A[ne + ids], axis=1))
    if np.any(residual > tolerance):
        raise ValueError('A near-zero projected row lacks an equality-span proof.')
    constant_rows = []
    for j, i in enumerate(ids):
        lower_before, upper_before = float(lr[i]), float(ur[i])
        lower_repair = max(0., lower_before)
        upper_repair = max(0., -upper_before)
        if max(lower_repair, upper_repair) > 1e-9:
            raise ValueError(f'Constant row conflict: original row {ne+i} '
                             f'({m["constraint_labels"][ne+i]}), '
                             f'lower={lower_before}, upper={upper_before}; limit=1e-9.')
        Ar[i] = 0.
        lr[i], ur[i] = min(lower_before, 0.), max(upper_before, 0.)
        constant_rows.append(dict(original_row=int(ne+i), reduced_row=int(i),
            label=str(m['constraint_labels'][ne+i]), projected_norm_before=float(norms[i]),
            proof='Endpoint Bezier controls fixed by PVAJ and, when enabled, the terminal face',
            equality_span_residual=float(residual[j]), equality_span_tolerance=float(tolerance[j]),
            constant_from_equalities=float(weights[j] @ ba),
            lower_before=lower_before, upper_before=upper_before,
            lower_after=float(lr[i]), upper_after=float(ur[i]),
            lower_relaxation_SI=lower_repair, upper_relaxation_SI=upper_repair))
    # Constant rows stay present with denominator one; nonconstant rows retain
    # their original positive normalization and unchanged shifted bounds.
    row_den = np.where(norms > 1e-11, norms, 1.)
    As, ls, us = Ar / row_den[:, None], lr / row_den, ur / row_den
    return dict(P=np.eye(M.shape[1]), q=np.zeros(M.shape[1]), A=As, l=ls, u=us,
                xbase=xbase, M=M, Eaug=Ea, baug=ba, p=p, N=N, face=face,
                row_den=row_den, near_constant_rows=np.flatnonzero(norms <= 1e-11) + ne,
                constant_rows=constant_rows, constant_row_weights=weights,
                constant_original_rows=ids + ne,
                equality_singular_values=singular, original_equality_rows=ne,
                original_rows=len(A), retained_inequality_rows=len(As),
                whiten_identity_error=_maxabs(M.T @ H @ M - np.eye(M.shape[1])))


