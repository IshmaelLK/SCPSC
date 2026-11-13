"""Sampled body/cable geometry against the original terrain, without corridors.

The current surface is the fixed common-refinement Hop supplied by load_scene. Signed
surface distance is positive above the terrain and negative inside its solid;
its magnitude is Euclidean distance to the top mesh, never a vertical gap.
Radius minus signed distance is a clearance violation depth, NOT the minimum
translation needed to separate a complete body/capsule from nonconvex terrain.
Time is sampled.  At each time the entire cable centre segment is evaluated.
"""
from functools import lru_cache
import heapq
from pathlib import Path
import time

import numpy as np

from .geometry import Terrain, point_triangle_distances
from .canopy_source import OperationSource


class SourceCoverageUnknown(ValueError):
    pass


def _closest_point(p, tri):
    a, b, c = tri
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    projected = p-normal*(np.dot(p-a, normal)/np.dot(normal, normal))
    gram = np.array([[ab@ab, ab@ac], [ab@ac, ac@ac]])
    u, v = np.linalg.solve(gram, np.array([(projected-a)@ab, (projected-a)@ac]))
    candidates = []
    if u >= 0 and v >= 0 and u+v <= 1:
        candidates.append(projected)
    for x, y in ((a, b), (b, c), (c, a)):
        edge = y-x
        candidates.append(x+np.clip((p-x)@edge/(edge@edge), 0., 1.)*edge)
    return min(candidates, key=lambda q: np.dot(p-q, p-q))


def _point_mesh(p, triangles):
    distances = point_triangle_distances(p, triangles)
    k = int(np.argmin(distances))
    return float(distances[k]), _closest_point(p, triangles[k]), triangles[k]


def _segment_mesh(a, b, triangles, endpoints):
    """Exact closest pair: endpoint/facet, edge/edge, or facet intersection."""
    q = min(endpoints, key=lambda q: abs(q['signed_distance_m']))
    best = [abs(q['signed_distance_m']), q['position'], q['surface_position'], q['triangle']]
    d = b-a
    dd = d@d
    if dd == 0:
        return best
    ids = np.arange(len(triangles))

    def consider(p, surface, triangle_ids):
        nonlocal best
        if not len(triangle_ids):
            return
        distances = np.linalg.norm(p-surface, axis=-1)
        k = int(np.argmin(distances))
        if distances[k] < best[0]:
            best = [float(distances[k]), p[k], surface[k], triangles[triangle_ids[k]]]

    ta, tb, tc = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = tb-ta, tc-ta
    normals = np.cross(ab, ac)
    denom = normals@d
    s = np.divide(np.einsum('ij,ij->i', ta-a, normals), denom,
                  out=np.full(len(triangles), np.inf), where=denom != 0)
    good = (s >= 0)&(s <= 1)
    chosen = ids[good]
    if len(chosen):
        p = a+s[good, None]*d
        w = p-ta[good]
        e, f = ab[good], ac[good]
        ee, ef, ff = np.sum(e*e, axis=1), np.sum(e*f, axis=1), np.sum(f*f, axis=1)
        ew, fw = np.sum(e*w, axis=1), np.sum(f*w, axis=1)
        det = ee*ff-ef*ef
        u, v = (ff*ew-ef*fw)/det, (ee*fw-ef*ew)/det
        inside = (u >= 0)&(v >= 0)&(u+v <= 1)
        if np.any(inside):
            k = int(np.flatnonzero(inside)[0])
            return [0., p[k], p[k].copy(), triangles[chosen[k]]]
    for c, e1 in ((ta, tb), (tb, tc), (tc, ta)):
        e, w = e1-c, a-c
        ee = np.sum(e*e, axis=1)
        de, dw, ew = e@d, w@d, np.sum(e*w, axis=1)
        # Triangle vertices projected onto the entire cable segment.
        for vertex in (c, e1):
            u = np.clip((vertex-a)@d/dd, 0., 1.)
            consider(a+u[:, None]*d, vertex, ids)
        det = dd*ee-de*de
        nonparallel = det > 0
        u = np.divide(de*ew-ee*dw, det, out=np.zeros(len(det)), where=nonparallel)
        v = np.divide(dd*ew-de*dw, det, out=np.zeros(len(det)), where=nonparallel)
        interior = nonparallel&(u >= 0)&(u <= 1)&(v >= 0)&(v <= 1)
        consider(a+u[interior, None]*d, c[interior]+v[interior, None]*e[interior], ids[interior])
    return best


class SourceSurface:
    """Local mesh queries; every closer triangle lies in the searched XY box."""
    def __init__(self, terrain):
        # Do not rebuild a raster: that would discard Hseal and input ridges.
        self.terrain = terrain if isinstance(terrain,OperationSource) else Terrain(
            terrain.x,terrain.y,terrain.dtm,np.maximum(terrain.dtm,terrain.dsm))
        self.bounds = self.terrain.bounds
        self._cached_triangles = lru_cache(maxsize=32)(self._triangles_cells)

    def _triangles_cells(self, cells):
        try:
            return self.terrain._triangles_cells(cells)
        except ValueError as exc:
            if 'NoData' in str(exc):
                raise SourceCoverageUnknown(str(exc)) from exc
            raise

    def triangles(self, lo, hi):
        return self._cached_triangles(self.terrain._cell_range(
            np.maximum(lo, self.bounds[0]), np.minimum(hi, self.bounds[1])))

    def require_coverage(self, a, b, radius):
        if (np.any(np.minimum(a[:2], b[:2])-radius < self.bounds[0]) or
                np.any(np.maximum(a[:2], b[:2])+radius > self.bounds[1])):
            raise SourceCoverageUnknown('Object plus requested clearance leaves known source XY coverage')

    def point(self, p):
        p = np.asarray(p, float)
        self.require_coverage(p, p, 0.)
        try:
            height = float(self.terrain.height(p))
        except ValueError as exc:
            if 'NoData' in str(exc):
                raise SourceCoverageUnknown(str(exc)) from exc
            raise
        # The vertical surface witness gives an upper bound even below ground.
        upper = abs(p[2]-height)
        local = self.triangles(p[:2], p[:2])
        upper = min(upper, _point_mesh(p, local)[0])
        pad = 64*np.finfo(float).eps*max(1., float(np.max(abs(p))), upper)
        tris = self.triangles(p[:2]-upper-pad, p[:2]+upper+pad)
        distance, nearest, triangle = _point_mesh(p, tris)
        signed = distance if p[2] >= height else -distance
        return dict(signed_distance_m=float(signed), position=p.copy(),
                    surface_position=nearest, triangle=triangle)

    def segment(self, a, b, tolerance_m=1e-4, endpoints=None):
        """Minimum signed distance on the COMPLETE segment, with an enclosure.

        Outside: exact segment/triangle closest pair. Inside: global interval
        bounds from 1-Lipschitz signed distance and convex triangle distances.
        The returned witness attains the upper bound; lower bound covers every
        unsampled cable point. Floating-point bounds are not exact arithmetic.
        """
        a, b = np.asarray(a, float), np.asarray(b, float)
        if not np.isfinite(tolerance_m) or tolerance_m <= 0:
            raise ValueError('Positive finite cable-distance tolerance required')
        qa, qb = endpoints if endpoints is not None else (self.point(a), self.point(b))
        upper = min(abs(qa['signed_distance_m']), abs(qb['signed_distance_m']))
        pad = 64*np.finfo(float).eps*max(1., float(np.max(abs(np.r_[a, b]))), upper)
        tris = self.triangles(np.minimum(a[:2], b[:2])-upper-pad,
                              np.maximum(a[:2], b[:2])+upper+pad)
        distance, position, nearest, tri = _segment_mesh(a, b, tris, (qa, qb))
        if qa['signed_distance_m'] >= 0 and qb['signed_distance_m'] >= 0 and distance > pad:
            return dict(signed_distance_m=distance, lower_bound_m=distance,
                        position=position, surface_position=nearest, triangle=tri,
                        distance_error_bound_m=0., segment_evaluations=0)
        length = float(np.linalg.norm(b-a))
        best = min((qa, qb), key=lambda q: q['signed_distance_m'])
        if length == 0:
            return dict(best, lower_bound_m=best['signed_distance_m'],
                        distance_error_bound_m=0., segment_evaluations=0)

        def bound(left, right, qleft, qright):
            lipschitz = (qleft['signed_distance_m']+qright['signed_distance_m']-
                         length*(right-left))/2
            # Distance to any convex triangle is convex along this interval.
            candidates = np.array([qleft['triangle'], qright['triangle']])
            dl = point_triangle_distances(qleft['position'], candidates)
            dr = point_triangle_distances(qright['position'], candidates)
            triangle_bound = -float(np.min(np.maximum(dl, dr)))
            return max(lipschitz, triangle_bound)-pad

        serial = 0
        queue = [(bound(0., 1., qa, qb), serial, 0., 1., qa, qb)]
        evaluations = 0
        while best['signed_distance_m']-queue[0][0] > tolerance_m:
            _, _, left, right, qleft, qright = heapq.heappop(queue)
            middle = (left+right)/2
            qmiddle = self.point(a+middle*(b-a))
            evaluations += 1
            if qmiddle['signed_distance_m'] < best['signed_distance_m']:
                best = qmiddle
            for lo, hi, qlo, qhi in ((left, middle, qleft, qmiddle), (middle, right, qmiddle, qright)):
                serial += 1
                heapq.heappush(queue, (bound(lo, hi, qlo, qhi), serial, lo, hi, qlo, qhi))
        lower = min(queue[0][0], best['signed_distance_m'])
        return dict(best, lower_bound_m=float(lower),
                    distance_error_bound_m=float(best['signed_distance_m']-lower),
                    segment_evaluations=evaluations)


def evaluate_states(terrain, times, pL, pQ, limits, *, rope_tolerance_m=1e-4):
    """Evaluate supplied true centres; retain each object's own physical radius."""
    started = time.perf_counter()
    times, pL, pQ = np.asarray(times, float), np.asarray(pL, float), np.asarray(pQ, float)
    if (times.ndim != 1 or len(times) == 0 or pL.shape != (len(times), 3) or pQ.shape != pL.shape
            or np.any(np.diff(times) <= 0) or not all(np.isfinite(x).all() for x in (times, pL, pQ))):
        raise ValueError('Finite increasing times and matching N by 3 body positions required')
    clearance = float(limits['clearance'])
    # This sampled diagnostic capsule is contained in the two-ball convex hull.
    # The formal geometry certificate uses the complete fixed system envelope.
    radii = dict(load=float(limits['rL']), vehicle=float(limits['rQ']),
                 rope=float(min(limits['rL'],limits['rQ'])))
    if not np.isfinite([clearance, *radii.values()]).all() or min(clearance, *radii.values()) < 0:
        raise ValueError('Finite nonnegative physical radii and clearance required')
    surface = terrain if isinstance(terrain, SourceSurface) else SourceSurface(terrain)
    summaries = {name: dict(radius_m=r, requested_clearance_m=clearance,
        evaluated_samples=0, unknown_samples=0, collision_samples=0, clearance_violation_samples=0,
        collision_unresolved_samples=0, clearance_unresolved_samples=0,
        minimum_signed_surface_distance_m=None, minimum_signed_surface_distance_lower_bound_m=None,
        extrema_scope='successfully evaluated time samples only; unknown samples have no distance bound',
        worst=None, unknown_reason=None) for name, r in radii.items()}
    for t, load, vehicle in zip(times, pL, pQ):
        centres = {}
        for name, a, b in (('load', load, load), ('vehicle', vehicle, vehicle), ('rope', load, vehicle)):
            row, radius = summaries[name], radii[name]
            try:
                surface.require_coverage(a, b, radius+clearance)
                if name != 'rope':
                    result = surface.point(a)
                    centres[name] = result
                    lower = result['signed_distance_m']
                else:
                    endpoints = (centres['load'], centres['vehicle']) if len(centres) == 2 else None
                    result = surface.segment(a, b, rope_tolerance_m, endpoints)
                    lower = result['lower_bound_m']
            except SourceCoverageUnknown as exc:
                row['unknown_samples'] += 1
                row['unknown_reason'] = str(exc)
                continue
            upper = result['signed_distance_m']
            row['evaluated_samples'] += 1
            row['collision_samples'] += int(upper < radius)
            row['clearance_violation_samples'] += int(upper < radius+clearance)
            row['collision_unresolved_samples'] += int(lower < radius <= upper)
            row['clearance_unresolved_samples'] += int(lower < radius+clearance <= upper)
            previous = row['minimum_signed_surface_distance_lower_bound_m']
            row['minimum_signed_surface_distance_lower_bound_m'] = lower if previous is None else min(previous, lower)
            previous = row['minimum_signed_surface_distance_m']
            if previous is None or upper < previous:
                row['minimum_signed_surface_distance_m'] = upper
                row['worst'] = dict(time_s=float(t), object_position_m=result['position'].tolist(),
                    nearest_surface_position_m=result['surface_position'].tolist(),
                    load_center_m=load.tolist(), vehicle_center_m=vehicle.tolist(),
                    signed_surface_distance_m=float(upper),
                    distance_error_bound_m=float(upper-lower))
                if name == 'rope':
                    delta = vehicle-load
                    row['worst']['rope_fraction'] = float((result['position']-load)@delta/(delta@delta)) if delta@delta else 0.
    for name, row in summaries.items():
        hi, lo, radius = row['minimum_signed_surface_distance_m'], row['minimum_signed_surface_distance_lower_bound_m'], radii[name]
        if hi is not None:
            row.update(minimum_physical_clearance_m=float(hi-radius),
                       minimum_safety_clearance_m=float(hi-radius-clearance),
                       physical_clearance_lower_bound_m=float(lo-radius),
                       safety_clearance_lower_bound_m=float(lo-radius-clearance),
                       maximum_collision_depth_m=float(max(0., radius-hi)),
                       maximum_collision_depth_upper_bound_m=float(max(0., radius-lo)),
                       maximum_clearance_violation_depth_m=float(max(0., radius+clearance-hi)),
                       maximum_clearance_violation_depth_upper_bound_m=float(max(0., radius+clearance-lo)))
        row['status'] = ('SAMPLED_COLLISION' if row['collision_samples'] else
                         'SAMPLED_CLEARANCE_VIOLATION' if row['clearance_violation_samples'] else
                         'UNKNOWN' if row['unknown_samples'] or row['clearance_unresolved_samples'] else 'SAMPLED_CLEAR')
    return dict(status=('SAMPLED_COLLISION' if any(r['collision_samples'] for r in summaries.values()) else
                        'SAMPLED_CLEARANCE_VIOLATION' if any(r['clearance_violation_samples'] for r in summaries.values()) else
                        'UNKNOWN' if any(r['status'] == 'UNKNOWN' for r in summaries.values()) else 'SAMPLED_CLEAR'),
        objects=summaries, sample_count=len(times), maximum_time_step_s=float(np.diff(times).max()) if len(times) > 1 else 0.,
        source_geometry='fixed common-refinement Hop; no corridor halfspaces' if isinstance(surface.terrain,OperationSource) else 'legacy H0 diagnostic fixture; no canopy closure claim',
        source_geometry_version=getattr(surface.terrain,'source_geometry_version','LEGACY_H0_TEST_ONLY'),
        source_geometry_id=getattr(surface.terrain,'source_geometry_id',None),
        geometry_scope='body spheres and endpoint-ball-covered straight cable capsule at sampled times; no continuous-time certificate',
        depth_definition='max(0, radius - minimum centre/centreline signed Euclidean distance to source top mesh); safety adds clearance once; not minimum separating translation or deepest material-point distance of a body/capsule',
        rope_distance_tolerance_m=float(rope_tolerance_m), seconds=time.perf_counter()-started)


def evaluate_operation_domain(times, pL, pQ, limits, domain):
    """Exact sphere/capsule support against six AABB faces at supplied times.

    A straight segment attains every axis extremum at an endpoint, so these
    margins cover the entire cable capsule, without spatial cable sampling.
    Negative physical margin means domain violation; safety subtracts delta
    once. These are face-containment margins, not a separating translation.
    """
    times, pL, pQ = np.asarray(times, float), np.asarray(pL, float), np.asarray(pQ, float)
    domain = np.asarray(domain, float)
    if (times.ndim != 1 or len(times) == 0 or pL.shape != (len(times), 3) or pQ.shape != pL.shape
            or np.any(np.diff(times) <= 0) or not all(np.isfinite(x).all() for x in (times, pL, pQ))):
        raise ValueError('Finite increasing times and matching N by 3 body positions required')
    if domain.shape != (2, 3) or not np.isfinite(domain).all() or np.any(domain[1] <= domain[0]):
        raise ValueError('Finite 2 by 3 operation domain with strictly positive extents required')
    clearance = float(limits['clearance'])
    radii = dict(load=float(limits['rL']), vehicle=float(limits['rQ']),
                 rope=float(min(limits['rL'],limits['rQ'])))
    if not np.isfinite([clearance, *radii.values()]).all() or min(clearance, *radii.values()) < 0:
        raise ValueError('Finite nonnegative physical radii and clearance required')
    faces = ('x_min', 'y_min', 'z_min', 'x_max', 'y_max', 'z_max')
    summaries = {}
    for name, lower, upper in (('load', pL, pL), ('vehicle', pQ, pQ),
                               ('rope', np.minimum(pL, pQ), np.maximum(pL, pQ))):
        radius = radii[name]
        margins = np.c_[lower-domain[0]-radius, domain[1]-upper-radius]
        sample, face = np.unravel_index(np.argmin(margins), margins.shape)
        margin = float(margins[sample, face])
        axis, direction = face % 3, -1. if face < 3 else 1.
        if name == 'rope':
            ends = np.array([pL[sample], pQ[sample]])
            endpoint = int(np.argmax(direction*ends[:, axis]))
            position = ends[endpoint]
        else:
            position = (pL if name == 'load' else pQ)[sample]
        physical_boundary, safety_boundary = position.copy(), position.copy()
        physical_boundary[axis] += direction*radius
        safety_boundary[axis] += direction*(radius+clearance)
        face_position = physical_boundary.copy()
        face_position[axis] = domain[int(face >= 3), axis]
        by_sample = margins.min(axis=1)
        physical_violations = int(np.count_nonzero(by_sample < 0.))
        safety_violations = int(np.count_nonzero(by_sample-clearance < 0.))
        worst = dict(time_s=float(times[sample]), face=faces[face],
            object_position_m=position.tolist(), physical_boundary_point_m=physical_boundary.tolist(),
            safety_boundary_point_m=safety_boundary.tolist(), domain_face_position_m=face_position.tolist(),
            load_center_m=pL[sample].tolist(), vehicle_center_m=pQ[sample].tolist())
        if name == 'rope':
            worst['rope_fraction'] = float(endpoint)
        summaries[name] = dict(radius_m=radius, requested_clearance_m=clearance,
            minimum_physical_clearance_m=margin, minimum_safety_clearance_m=margin-clearance,
            physical_violation_samples=physical_violations, clearance_violation_samples=safety_violations,
            worst=worst, status=('SAMPLED_DOMAIN_VIOLATION' if physical_violations else
                'SAMPLED_CLEARANCE_VIOLATION' if safety_violations else 'SAMPLED_CLEAR'))
    status = ('SAMPLED_DOMAIN_VIOLATION' if any(r['physical_violation_samples'] for r in summaries.values()) else
              'SAMPLED_CLEARANCE_VIOLATION' if any(r['clearance_violation_samples'] for r in summaries.values()) else
              'SAMPLED_CLEAR')
    return dict(status=status, objects=summaries, sample_count=len(times), domain_m=domain.tolist(),
        maximum_time_step_s=float(np.diff(times).max()) if len(times) > 1 else 0.,
        geometry_scope='true body spheres and complete straight cable capsule against six operation-domain faces at sampled times; no continuous-time certificate',
        margin_definition='minimum signed inward face distance of the sphere/capsule; negative magnitude is the largest single-face normal excess, not Euclidean distance to the box or minimum return translation; safety subtracts clearance once; strict negative margins count as violations')


def evaluate_saved(directory, config, *, time_step=None, use_saved_states=False, rope_tolerance_m=1e-4):
    """Read only. Default: exact saved curve, 0.095 s grid plus every seam."""
    directory = Path(directory)
    from .scene_inputs import load_scene
    terrain = load_scene(config)['source']
    if use_saved_states:
        with np.load(directory/'planning_reference.npz', allow_pickle=False) as data:
            times, pL, pQ = data['time'].copy(), data['pL'].copy(), data['pQ'].copy()
        state_source = 'planning_reference.npz saved recovered states'
    else:
        from .dynamics import recover
        from .polynomial import PolynomialTrajectory
        step = .095 if time_step is None else float(time_step)
        if not np.isfinite(step) or step <= 0:
            raise ValueError('Positive finite temporal sampling step required')
        with np.load(directory/'trajectory.npz', allow_pickle=False) as data:
            trajectory = PolynomialTrajectory(data['coefficients'], data['durations'], data['head'], data['tail'], 'fixed')
        seams = np.r_[0., np.cumsum(trajectory.durations)]
        times = np.unique(np.r_[np.arange(0., seams[-1], step), seams])
        recovered = recover(trajectory.sample(times, 5), config['dynamics'])
        pL, pQ = [recovered[key].detach().cpu().numpy() for key in ('pL', 'pQ')]
        state_source = 'trajectory.npz exact polynomial and original no-aero recovery'
    report = evaluate_states(terrain, times, pL, pQ, config['limits'], rope_tolerance_m=rope_tolerance_m)
    report['state_source'] = state_source
    domain = evaluate_operation_domain(times, pL, pQ, config['limits'], config['mechanism_scene']['operation_domain'])
    report['operation_domain'] = domain
    statuses = (report['status'], domain['status'])
    report['true_system_geometry_status'] = next((status for status in (
        'SAMPLED_COLLISION', 'SAMPLED_DOMAIN_VIOLATION', 'SAMPLED_CLEARANCE_VIOLATION', 'UNKNOWN')
        if status in statuses), 'SAMPLED_CLEAR')
    return report
