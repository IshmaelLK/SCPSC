"""Shared options and failure types for terrain separating-plane construction."""
from dataclasses import dataclass
import warnings

import numpy as np


class SeedPlaneFailure(ValueError):
    def __init__(self, proof):
        self.proof=proof
        super().__init__('SEED_PLANE_NOT_CONSTRUCTED: '+proof['message'])


class SeedPlaneSolverError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlaneOptions:
    plane_selection_mode: str = 'normal_margin'
    envelope_approximation: str = 'exact'
    vertex_filter: str = 'upper_hull'
    max_iter: int = 200
    time_limit_s: float | None = None
    tol: float = 1e-9
    certificate_eps_m: float = 1e-8
    seed_tolerance_m: float = 1e-7

    def __post_init__(self):
        if self.plane_selection_mode not in ('normal_margin','vertical_margin','point_then_shrink','horizontal_max'):
            raise ValueError('plane_selection_mode must be normal_margin, vertical_margin, point_then_shrink, or horizontal_max')
        if self.envelope_approximation not in ('exact','axis_box'):
            raise ValueError('envelope_approximation must be exact or axis_box')
        if self.envelope_approximation=='axis_box' and self.plane_selection_mode!='normal_margin':
            raise ValueError('axis_box requires fixed_system and normal_margin')
        if self.vertex_filter not in ('all','hull','upper_hull','constraint_generation'):
            raise ValueError('vertex_filter must be all, hull, upper_hull, or constraint_generation')
        if self.vertex_filter=='constraint_generation' and (
                self.plane_selection_mode!='normal_margin' or self.envelope_approximation!='exact'):
            raise ValueError('constraint_generation requires normal_margin and the exact envelope')
        if (self.max_iter is None or self.max_iter<1 or
                not np.isfinite([self.tol,self.certificate_eps_m,self.seed_tolerance_m]).all()):
            raise ValueError('Invalid finite plane solver settings')
        if self.time_limit_s is not None:
            warnings.warn('plane_solver.time_limit_s is deprecated and ignored', FutureWarning)
            object.__setattr__(self, 'time_limit_s', None)
        if self.tol<=0 or min(self.certificate_eps_m,self.seed_tolerance_m)<0:
            raise ValueError('Invalid positive time/tolerance or certificate slack')
