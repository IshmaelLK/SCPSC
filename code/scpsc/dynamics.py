"""Mass-independent taut-cable reference geometry, paper Eqs. (3) and (19).

Input contains physical payload derivatives p^(0..5), in SI units.
Continuous acceptance belongs to specific_force.py, never sampled recovery.
"""
from dataclasses import dataclass
from math import comb

import numpy as np
import torch


class RecoveryDomainError(ValueError):
    pass

@dataclass(frozen=True)
class DynamicsParams:
    ell: float = 10.
    g: float = 9.81
    specific_force_guard: float = 1e-12

    def __post_init__(self):
        values = (self.ell, self.g, self.specific_force_guard)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Finite positive rope length, gravity and normalization guard required')


DEFAULT_DYNAMICS = DynamicsParams()

def _dot(a, b):
    return (a * b).sum(dim=-1)

def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise RecoveryDomainError(f'RECOVERY_DOMAIN_ERROR: nonfinite {name}')

def normalization_derivatives(u, guard):
    """Differentiate u=||u|| q through order three, without mass or attitude."""
    magnitude = [torch.linalg.vector_norm(u[0], dim=-1)]
    _finite(magnitude[0], 'specific-force norm')
    if bool((magnitude[0] <= guard).any()):
        raise RecoveryDomainError('RECOVERY_DOMAIN_ERROR: specific-force normalization below guard')
    q = [u[0] / magnitude[0][..., None]]  # Eq. (3), q = u / ||u||.
    for r in range(1, len(u)):
        numerator = sum(comb(r, a) * _dot(u[a], u[r-a]) for a in range(r+1))
        numerator -= sum(comb(r, a) * magnitude[a] * magnitude[r-a] for a in range(1, r))
        magnitude.append(numerator / (2 * magnitude[0]))
        q.append((u[r] - sum(comb(r, a) * magnitude[a][..., None] * q[r-a]
                            for a in range(1, r+1))) / magnitude[0][..., None])
    return magnitude, q

def recover(state, params=DEFAULT_DYNAMICS):
    """Recover rope direction and both bodies' translational derivatives."""
    if isinstance(params, dict):
        params = DynamicsParams(**params)
    state = torch.as_tensor(state, dtype=torch.float64)
    if state.shape[-2:] != (6, 3):
        raise ValueError('Expected physical payload derivatives [...,6,3]')
    _finite(state, 'payload derivatives')
    u = [state[..., 2, :] + state.new_tensor([0., 0., params.g])]
    u.extend(state[..., r+2, :] for r in range(1, 4))
    magnitude, q = normalization_derivatives(u, params.specific_force_guard)
    vehicle = [state[..., r, :] + params.ell*q[r] for r in range(4)]  # Eq. (3) and its time derivatives.
    out = {'pL': state[..., 0, :], 'vL': state[..., 1, :],
               'aL': state[..., 2, :], 'jL': state[..., 3, :],
               'pQ': vehicle[0], 'vQ': vehicle[1], 'aQ': vehicle[2], 'jQ': vehicle[3]}
    # Eq. (19) for the recovered rope direction and its first derivative.
    out.update(theta_rad=torch.atan2(torch.linalg.vector_norm(q[0][..., :2], dim=-1), q[0][..., 2]),
               omega_s_radps=torch.linalg.cross(q[0], q[1], dim=-1),
               swing_rate_radps=torch.linalg.vector_norm(q[1], dim=-1),
               tension_per_mass_mps2=magnitude[0])
    for label, values in (('u', u), ('q', q)):
        for r, value in enumerate(values):
            out[label + (str(r) if r else '')] = value
    for key, value in out.items():
        _finite(value, key)
    return out

def endpoint_equilibrium(positions, params=DEFAULT_DYNAMICS):
    """Stationary reference with q=e3; no force balance or yaw input."""
    positions = torch.as_tensor(positions, dtype=torch.float64)
    if positions.shape[-1:] != (3,):
        raise ValueError('Endpoint shape must end in 3')
    state = torch.zeros((*positions.shape[:-1], 6, 3), dtype=positions.dtype)
    state[..., 0, :] = positions
    out = recover(state, params)
    out['E'] = torch.eye(3, dtype=positions.dtype)[:, :2].expand(*positions.shape[:-1], 3, 2)
    return out
