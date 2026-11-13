"""Shared velocity geometry for clocks and the linear Bezier QP certificate.

The current model is an inscribed 26-face polytope. Legacy configurations
without an explicit selector retain their original box (no silent migration).
The time gauge excludes kappa; callers multiply by kappa exactly once.
"""
from dataclasses import dataclass
import numpy as np

POLY26_FACTOR = 1.0 / np.sqrt(9.0 - 2.0*np.sqrt(2.0) - 2.0*np.sqrt(6.0))

def poly26_normals():
    rows = list(np.eye(3))
    for i in range(3):
        for j in range(i + 1, 3):
            for sign in (1., -1.):
                row = np.zeros(3); row[i] = 1.; row[j] = sign
                rows.append(row / np.sqrt(2.))
    for sy in (1., -1.):
        for sz in (1., -1.):
            rows.append(np.array([1., sy, sz]) / np.sqrt(3.))
    return np.asarray(rows)

@dataclass(frozen=True)
class VelocityConstraint:
    kind: str
    normals: np.ndarray
    bounds: np.ndarray
    circumradius: float

    def gauge(self, displacement):
        d = np.asarray(displacement, dtype=float)
        if d.ndim < 1 or d.shape[-1] != 3 or not np.isfinite(d).all():
            raise ValueError('Velocity gauge requires finite (...,3) displacement')
        return np.max(np.abs(d @ self.normals.T) / self.bounds, axis=-1)

    def metadata(self):
        return dict(model=self.kind, unit_normals=self.normals.tolist(),
                    symmetric_bounds_mps=self.bounds.tolist(),
                    circumradius_mps=float(self.circumradius),
                    inradius_mps=float(self.bounds.min()),
                    bilateral_rows_per_control=int(len(self.bounds)),
                    single_sided_faces=int(2*len(self.bounds)))

    def npz_fields(self):
        result = dict(velocity_model=np.asarray(self.kind),
                      velocity_normals=self.normals,
                      velocity_symmetric_bounds_mps=self.bounds,
                      velocity_circumradius_mps=np.asarray(self.circumradius))
        if self.kind == 'box':
            result['velocity_box_mps'] = self.bounds
        return result


def box_constraint(bounds):
    b = np.asarray(bounds, dtype=float)
    if b.shape != (3,) or not np.isfinite(b).all() or np.any(b <= 0.):
        raise ValueError('Velocity box requires three finite positive bounds')
    return VelocityConstraint('box', np.eye(3), b.copy(), float(np.linalg.norm(b)))


def coerce_velocity(value):
    return value if isinstance(value, VelocityConstraint) else box_constraint(value)


def from_config(settings, load_speed_limit=None):
    kind = settings.get('velocity_constraint', 'box')
    if load_speed_limit is not None:
        load_speed_limit = float(load_speed_limit)
        if not np.isfinite(load_speed_limit) or load_speed_limit <= 0.:
            raise ValueError('Original load speed limit must be finite and positive')
    if kind == 'polytope26':
        if 'velocity_component_bounds' in settings or 'velocity_box' in settings:
            raise ValueError('POLY26_CONFLICT: remove legacy velocity box fields')
        vmax = float(settings.get('velocity_max_mps', load_speed_limit or np.nan))
        if not np.isfinite(vmax) or vmax <= 0.:
            raise ValueError('polytope26 requires finite positive velocity_max_mps')
        if load_speed_limit is not None and abs(vmax-load_speed_limit) > 1e-12*max(1.,load_speed_limit):
            raise ValueError('POLY26_SPEED_MISMATCH: configured ball must equal physical vL')
        model = VelocityConstraint(kind, poly26_normals(),
                                   np.full(13, POLY26_FACTOR*vmax), vmax)
    elif kind == 'box':
        default = None if load_speed_limit is None else np.full(3, load_speed_limit/np.sqrt(3.))
        model = box_constraint(settings.get('velocity_component_bounds', default))
        if load_speed_limit is not None and model.circumradius > load_speed_limit*(1.+1e-12):
            raise ValueError('Velocity box must fit the original load speed ball')
    else:
        raise ValueError('Unknown velocity constraint: '+str(kind))
    return model


def apply_poly26(config):
    """Set a complete application config; preserve all physical limits/kappa."""
    vmax = float(config['limits']['vL'])
    qp = config['ubsqp']
    qp.pop('velocity_component_bounds', None); qp.pop('velocity_box', None)
    qp.update(velocity_constraint='polytope26', velocity_max_mps=vmax)
    from_config(qp, vmax)
    return config
