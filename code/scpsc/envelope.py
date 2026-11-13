"""System convex-envelope support and the declared fixed-cap comparison.

The optimizer keeps smooth endpoint halfspaces in configuration mode.  Fixed
mode uses the exact support over the allowed rope-direction cap, without
changing the recovered rope direction or any physical constraint.
"""
import numpy as np


def geometry_mode(config):
    mode=config['optimization'].get('geometry_mode','configuration')
    if mode not in ('configuration','fixed'):
        raise ValueError('geometry_mode must be configuration or fixed')
    return mode


def normal_lengths(normals):
    normals=np.asarray(normals,dtype=float)
    if normals.shape[-1:]!=(3,) or not np.isfinite(normals).all():
        raise ValueError('Envelope normals must be finite three-vectors')
    lengths=np.linalg.norm(normals,axis=-1)
    if np.any(lengths<=0) or not np.isfinite(lengths).all():
        raise ValueError('Envelope requires nonzero finite face normals')
    return lengths


def normalized_halfspaces(normals,bounds):
    """Unit face rows, preserving the stored baseline's numerical unit rows.

    A three-vector normalized in binary64 need not have a recomputed norm of
    exactly one. Within four machine epsilons retain its stored n and b so
    the original A0 rho-b arithmetic stays bitwise unchanged. Other positive
    scales are divided out. This is a representation convention, not a
    change to a physical acceptance tolerance or solver parameter.
    """
    normals=np.asarray(normals,dtype=float);bounds=np.asarray(bounds,dtype=float)
    lengths=normal_lengths(normals)
    scales=np.where(np.abs(lengths-1.)<=4*np.finfo(float).eps,1.,lengths)
    return normals/scales[...,None],bounds/scales


def cap_extrema(normals,theta):
    """Min/max of unit-normal dot q on ||q||=1, q_z>=cos(theta)."""
    if not np.isfinite(theta) or not 0<=theta<=np.pi:
        raise ValueError('Rope cap angle must lie in [0, pi]')
    normals=np.asarray(normals,dtype=float)
    lengths=normal_lengths(normals)
    gamma=np.arccos(np.clip(normals[...,2]/lengths,-1.,1.))  # Eq. (9), angle between n and e3.
    return np.cos(np.minimum(np.pi,gamma+theta)),np.cos(np.maximum(0.,gamma-theta))


def effective_radii(limits):
    radii = np.asarray([limits['rL'], limits['rQ'], limits['clearance']], dtype=float)
    if not np.isfinite(radii).all() or min(radii[:2]) <= 0. or radii[2] < 0.:
        raise ValueError('Positive object radii and nonnegative common clearance required')
    return radii[0]+radii[2], radii[1]+radii[2]


def planning_envelope_mode(qp_config):
    """Select planning geometry without changing the true physical model."""
    mode=qp_config.get('planning_envelope')
    if mode is None:
        return 'fixed_system' if qp_config.get('use_system_envelope',True) else 'point'
    if mode not in ('fixed_system','load_only','point','bare_point'):
        raise ValueError('Unknown planning_envelope: '+str(mode))
    if mode not in ('point','bare_point') and not qp_config.get('use_system_envelope',True):
        raise ValueError('planning_envelope conflicts with legacy use_system_envelope=false')
    return mode


def planning_envelope_parameters(ell,limits,mode):
    """Return (ell, rhoL, rhoQ, theta) for the selected planning support."""
    if mode=='fixed_system':
        rhoL,rhoQ=effective_radii(limits)
        return ell,rhoL,rhoQ,limits['theta']
    if mode=='load_only':
        return 0.,limits['rL']+limits['clearance'],0.,0.
    if mode in ('point','bare_point'):
        return 0.,0.,0.,0.
    raise ValueError('Unknown planning_envelope: '+str(mode))


def configuration_support(normals,q,ell,rhoL,rhoQ):
    normals=np.asarray(normals,dtype=float);lengths=normal_lengths(normals)
    # Eq. (4) in payload-centered coordinates, using the supplied ball radii.
    return np.maximum(rhoL*lengths,ell*(normals@np.asarray(q))+rhoQ*lengths)


def fixed_support(normals,ell,rhoL,rhoQ,theta):
    """h_fix(n)=||n|| max(rhoL,ell U_n+rhoQ), with clearance applied once."""
    _,upper=cap_extrema(normals,theta)
    return normal_lengths(normals)*np.maximum(rhoL,ell*upper+rhoQ)  # Eq. (9), support of the hull in Eq. (8).


def active_endpoint_faces(normals,ell,rhoL,rhoQ,theta,prune=True):
    """Original whole-cap dominance, retaining both rows near a zero bound."""
    lower,upper=cap_extrema(normals,theta);faces=[]
    for j,(lo,hi) in enumerate(zip(lower,upper)):
        if prune and ell*hi+rhoQ-rhoL < -1e-10:
            faces.append((j,'L'))
        elif prune and ell*lo+rhoQ-rhoL > 1e-10:
            faces.append((j,'Q'))
        else:faces.extend(((j,'L'),(j,'Q')))
    return faces


def original_halfspaces(corridor):
    return (corridor.normals if corridor.original_normals is None else corridor.original_normals,
            corridor.bounds if corridor.original_bounds is None else corridor.original_bounds)


def center_halfspaces(corridor, ell, limits, planning_envelope='fixed_system'):
    """Return load-center rows; the declared envelope is applied exactly once."""
    if isinstance(corridor, dict):
        normals, bounds = corridor.get('original_normals'), corridor.get('original_bounds')
        normals = corridor['normals'] if normals is None else normals
        bounds = corridor['bounds'] if bounds is None else bounds
        proof = corridor.get('proof', {})
    else:
        normals, bounds = original_halfspaces(corridor)
        proof = corridor.proof or {}
    normals, bounds = np.asarray(normals, float), np.asarray(bounds, float)
    lengths = normal_lengths(normals)
    if bounds.shape != lengths.shape or not np.isfinite(bounds).all():
        raise ValueError('One finite bound per corridor normal required')
    space = proof.get('halfspace_semantics', 'RAW_WORKSPACE')
    if space in ('LOAD_CENTER', 'QP_SUPPORT_ENCODED_CENTER'):
        recorded = np.asarray(proof.get('center_envelope_parameters', []), float)
        actual = np.asarray(planning_envelope_parameters(ell, limits, 'fixed_system'), float)
        if recorded.shape != (4,) or not np.array_equal(recorded, actual):
            raise ValueError('CENTER_ENVELOPE_MISMATCH: corridor belongs to a different or unspecified fixed envelope')
    if space == 'LOAD_CENTER':
        if not proof.get('center_space_source'):
            raise ValueError('LOAD_CENTER corridor requires its construction source')
        return normals, bounds.copy()
    if space not in ('RAW_WORKSPACE', 'QP_SUPPORT_ENCODED_CENTER'):
        raise ValueError('Unknown corridor halfspace space: ' + str(space))
    # Explicitly decode the historical support-encoded format on read.
    # New center-space constructors store their actual center bounds.
    support = fixed_support(normals, *planning_envelope_parameters(ell, limits, planning_envelope))
    return normals, bounds - support  # Eq. (12) for the raw workspace halfspaces.


def fixed_seed_margins(corridor, points, ell, limits, use_system_envelope=True, planning_envelope=None):
    """Signed metres against the planning support (zero in the point ablation).

    This is an admission diagnostic of a polyline, not a dynamical certificate.
    """
    mode = planning_envelope or ('fixed_system' if use_system_envelope else 'point')
    normals, bounds = center_halfspaces(corridor, ell, limits, mode)
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=float)
    return (bounds - points @ normals.T) / np.linalg.norm(normals, axis=1)

