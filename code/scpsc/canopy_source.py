"""Given canopy closure on a common refinement (paper Eq. 1).

This module never estimates a closure surface.  The supplied Hseal is a
piecewise affine height field.  Triangle intersections preserve both input
meshes; taking nodal maxima on each intersection makes Hop cover both affine
inputs.  Missing coverage remains unknown, including missing Hseal coverage.
"""
import hashlib
import time
from fractions import Fraction
from dataclasses import dataclass
import numpy as np
from .geometry import Terrain, clip_polygon, point_triangle_distances

SOURCE_GEOMETRY_VERSION = 'SCPSC_HOP_COMMON_REFINEMENT_V1'


def _raster_arrays(x,y,z):
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    if (z.shape != (len(y), len(x)) or len(x) < 2 or len(y) < 2 or
            not np.isfinite(x).all() or not np.isfinite(y).all() or
            np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)):
        raise ValueError('INVALID_MAP: raster dimensions or coordinate ordering')
    return x,y,z


def raster_triangles(x, y, z):
    """The existing lower-left to upper-right diagonal, including NoData."""
    x,y,z=_raster_arrays(x,y,z)
    xx, yy = np.meshgrid(x, y)
    v = np.stack((xx, yy, z), -1)
    a, b = v[:-1, :-1].reshape(-1, 3), v[:-1, 1:].reshape(-1, 3)
    c, d = v[1:, 1:].reshape(-1, 3), v[1:, :-1].reshape(-1, 3)
    return np.concatenate((np.stack((a, b, c), 1), np.stack((a, c, d), 1)))


@dataclass(frozen=True)
class RasterSeal:
    """Given nodal values, keeping the raster topology without expansion."""
    x: np.ndarray
    y: np.ndarray
    hseal: np.ndarray


def load_seal(path):
    """Explicit NPZ: conforming triangles[N,3,3], or x/y/hseal raster."""
    with np.load(path, allow_pickle=False) as data:
        if 'triangles' in data:
            return np.asarray(data['triangles'], float).copy()
        if not {'x', 'y', 'hseal'} <= set(data.files):
            raise ValueError('HSEAL_FORMAT_REQUIRED: triangles or x/y/hseal arrays')
        return RasterSeal(*_raster_arrays(data['x'],data['y'],data['hseal']))


def _areas(triangles):
    e, f = triangles[:, 1, :2]-triangles[:, 0, :2], triangles[:, 2, :2]-triangles[:, 0, :2]
    return (e[:, 0]*f[:, 1]-e[:, 1]*f[:, 0])/2


def _orientation_sign(a,b,c):
    """2D orientation; uncertain float determinants use exact input floats."""
    if np.array_equal(c,a) or np.array_equal(c,b):return 0
    ab,ac=b-a,c-a
    first,second=ab[0]*ac[1],ab[1]*ac[0]
    determinant=first-second
    if abs(determinant)>8*np.finfo(float).eps*(abs(first)+abs(second)):
        return 1 if determinant>0 else -1
    ax,ay,bx,by,cx,cy=map(lambda value:Fraction(float(value)),(*a,*b,*c))
    exact=(bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
    return 1 if exact>0 else -1 if exact<0 else 0


def _require_disjoint_faces(triangles, xy, node_ids, name):
    """Reject positive-area XY overlap before any coverage-area sum is used."""
    if len(np.unique(np.sort(node_ids,axis=1),axis=0)) != len(triangles):
        raise ValueError(name+': overlapping faces (duplicate XY triangle)')
    lower,upper=triangles[...,:2].min(1),triangles[...,:2].max(1)
    active=np.empty(0,dtype=int)
    # Sweep only simultaneous X extents, then filter strict Y overlap. Shared
    # boundaries need no area test. A raster does not trigger all-pairs work.
    for i in np.argsort(lower[:,0],kind='stable'):
        active=active[upper[active,0]>lower[i,0]]
        candidates=active[(upper[active,1]>lower[i,1])&(lower[active,1]<upper[i,1])]
        for j in candidates:
            shared=np.intersect1d(node_ids[i],node_ids[j])
            if len(shared)==2:
                a,b=xy[shared]
                c=xy[np.setdiff1d(node_ids[i],shared)[0]]
                d=xy[np.setdiff1d(node_ids[j],shared)[0]]
                if _orientation_sign(a,b,c)*_orientation_sign(a,b,d)<0:continue
            # Two convex triangle interiors are disjoint iff some edge axis
            # separates them. Exact signs at contact avoid mistaking a tiny
            # roundoff polygon near a common vertex for positive-area overlap.
            first,second=triangles[i,:,:2],triangles[j,:,:2]
            separated=False
            for face,other in ((first,second),(second,first)):
                for a,b in zip(face,np.roll(face,-1,axis=0)):
                    if all(_orientation_sign(a,b,c)<=0 for c in other):
                        separated=True;break
                if separated:break
            if not separated:
                raise ValueError(name+': overlapping faces with positive XY area')
        active=np.append(active,i)


def _mesh(triangles, name):
    t = np.asarray(triangles, float).copy()
    if t.ndim != 3 or t.shape[1:] != (3, 3) or not len(t) or not np.isfinite(t[..., :2]).all():
        raise ValueError(name+': finite XY triangles[N,3,3] required')
    areas = _areas(t)
    if np.any(areas == 0):
        raise ValueError(name+': degenerate XY triangle')
    negative = areas < 0
    t[negative] = t[negative][:, [0, 2, 1]]
    # A shared mesh node must describe one height (unknown nodes stay unknown).
    xy, ids = np.unique(t[..., :2].reshape(-1, 2), axis=0, return_inverse=True)
    z = t[..., 2].ravel()
    lower, upper = np.full(len(xy), np.inf), np.full(len(xy), -np.inf)
    finite = np.isfinite(z)
    np.minimum.at(lower, ids[finite], z[finite]); np.maximum.at(upper, ids[finite], z[finite])
    known = np.isfinite(lower)
    if np.any(upper[known]-lower[known] > 32*np.finfo(float).eps*np.maximum(1., abs(upper[known]))):
        raise ValueError(name+': inconsistent shared-node heights')
    # A conforming mesh is required. Even a continuous T-junction input can
    # make nodal max interpolate different Hop heights on opposite edge sides.
    node_ids=ids.reshape(-1,3)
    _require_disjoint_faces(t,xy,node_ids,name)
    edges=np.sort(np.concatenate((node_ids[:,[0,1]],node_ids[:,[1,2]],node_ids[:,[2,0]])),axis=1)
    edges,counts=np.unique(edges,axis=0,return_counts=True)
    boundary=edges[counts==1]; boundary_nodes=np.unique(boundary)
    for a,b in boundary:
        vector=xy[b]-xy[a];offset=xy[boundary_nodes]-xy[a]
        fraction=offset@vector/(vector@vector)
        cross=vector[0]*offset[:,1]-vector[1]*offset[:,0]
        scale=np.maximum(1.,np.abs(vector[0]*offset[:,1])+np.abs(vector[1]*offset[:,0]))
        on=(fraction>0)&(fraction<1)&(np.abs(cross)<=32*np.finfo(float).eps*scale)
        nodes=boundary_nodes[on]
        nodes=np.array([node for node in nodes if _orientation_sign(xy[a],xy[b],xy[node])==0],dtype=int)
        if not len(nodes):continue
        if known[a] and known[b]:
            along=(xy[nodes]-xy[a])@vector/(vector@vector)
            expected=lower[a]+along*(lower[b]-lower[a])
            selected=known[nodes]
            if np.any(np.abs(lower[nodes[selected]]-expected[selected])>64*np.finfo(float).eps*np.maximum(1.,abs(expected[selected]))):
                raise ValueError(name+': discontinuous heights at a subdivided shared edge')
        raise ValueError(name.upper()+'_NONCONFORMING_MESH: a vertex lies inside another triangle edge; supply an explicit common triangulation')
    return t


def _height_on(triangle, xy):
    weights = np.linalg.solve((triangle[1:, :2]-triangle[0, :2]).T, (xy-triangle[0, :2]).T).T
    return triangle[0, 2]+weights@(triangle[1:, 2]-triangle[0, 2])


def common_refinement(h0, hseal):
    h0, hseal = _mesh(h0, 'H0'), _mesh(hseal, 'Hseal')
    if h0.shape == hseal.shape and np.array_equal(h0[..., :2], hseal[..., :2]):
        hop = h0.copy(); hop[..., 2] = np.maximum(h0[..., 2], hseal[..., 2])
        return hop
    lower, upper = hseal[..., :2].min(1), hseal[..., :2].max(1)
    result = []
    for original in h0:
        lo, hi = original[:, :2].min(0), original[:, :2].max(0)
        candidates = hseal[np.all(upper >= lo, axis=1)&np.all(lower <= hi, axis=1)]
        covered = 0.
        original_area = float(_areas(original[None])[0])
        for seal in candidates:
            # Clip XY only: unknown input heights must not contaminate XY edges.
            polygon = original.copy(); polygon[:, 2] = 0.
            for a, b in zip(seal, np.roll(seal, -1, axis=0)):
                edge = b[:2]-a[:2]
                normal = np.array([edge[1], -edge[0]])
                polygon = clip_polygon(polygon, normal, normal@a[:2])
                if len(polygon) < 3:
                    break
            if len(polygon) < 3:
                continue
            polygon[:, 2] = np.maximum(_height_on(original, polygon[:, :2]), _height_on(seal, polygon[:, :2]))
            for k in range(1, len(polygon)-1):
                triangle = polygon[[0, k, k+1]]
                area = float(_areas(triangle[None])[0])
                if area > 0:
                    result.append(triangle); covered += area
        area_guard = 256*np.finfo(float).eps*max(1., original_area)
        if covered > original_area+area_guard:
            raise ValueError('HSEAL_INVALID_MESH: overlapping faces in common refinement')
        if covered < original_area-area_guard:
            # Preserve the missing region conservatively as an unknown source
            # triangle; no uncovered part is silently replaced by H0.
            unknown = original.copy(); unknown[:, 2] = np.nan
            result.append(unknown)
    if not result:
        raise ValueError('HSEAL_NO_COVERAGE: no common source faces')
    return np.asarray(result)


class OperationSource(Terrain):
    """Immutable triangular Hop; the Terrain interface never resamples it."""
    def __init__(self, h0, hseal, bounds, query_mode='scan'):
        self._surface = np.ascontiguousarray(common_refinement(h0, hseal), dtype=np.float64)  # Eq. (1), common mesh for H_op.
        self.bounds = np.asarray(bounds, float).copy()
        self._lower, self._upper = self._surface[..., :2].min(1), self._surface[..., :2].max(1)
        self.source_geometry_version = SOURCE_GEOMETRY_VERSION
        # Explicitly requested geometry-cache identity; not a file manifest.
        self.source_geometry_id = hashlib.sha256(self.source_geometry_version.encode()+self._surface.tobytes()+self.bounds.tobytes()).hexdigest()
        for array in (self._surface, self.bounds, self._lower, self._upper):
            array.flags.writeable = False
        self.set_query_mode(query_mode)
        self.reset_query_statistics()

    @classmethod
    def from_inputs(cls, original, seal, query_mode='scan'):
        if isinstance(seal,RasterSeal):
            x,y,z=_raster_arrays(seal.x,seal.y,seal.hseal)
            if np.array_equal(original.x,x) and np.array_equal(original.y,y):
                return _RasterOperationSource(original,z,query_mode)
            seal=raster_triangles(x,y,z)
        h0 = raster_triangles(original.x, original.y, np.maximum(original.dtm, original.dsm))
        return cls(h0, seal, original.bounds, query_mode)

    def set_query_mode(self, mode):
        if mode not in ('scan', 'quadtree'):
            raise ValueError('Unknown terrain query mode: '+str(mode))
        # A raster quadtree cannot represent the common refinement. Keep the
        # requested setting in statistics and report the actual triangle scan.
        self.query_mode = mode

    def reset_query_statistics(self):
        self._query_stats = {name: {'queries': 0, 'seconds': 0., 'node_visits': 0,
            'leaf_visits': 0, 'triangle_visits': 0} for name in ('max_height', 'distance')}

    def query_statistics(self):
        return dict(query_mode='common_refinement_triangle_scan', requested_query_mode=self.query_mode,
            build_seconds=0., build_count=0, node_count=0, leaf_count=0, index_memory_bytes_estimate=0,
            source_array_bytes=self._surface.nbytes, queries={k: dict(v) for k, v in self._query_stats.items()},
            query_seconds=sum(v['seconds'] for v in self._query_stats.values()),
            build_and_query_seconds=sum(v['seconds'] for v in self._query_stats.values()),
            source_geometry_version=self.source_geometry_version, source_geometry_id=self.source_geometry_id)

    def _cell_range(self, lo, hi):
        return tuple(np.r_[lo, hi].astype(float))

    def _triangles_cells(self, window):
        return self.triangles(np.asarray(window[:2]), np.asarray(window[2:]))

    def triangles(self, lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if np.any(lo < self.bounds[0]) or np.any(hi > self.bounds[1]) or np.any(lo > hi):
            raise ValueError('INVALID_MAP: query outside coverage')
        mask = np.all(self._upper >= lo, axis=1)&np.all(self._lower <= hi, axis=1)
        selected = self._surface[mask]
        if not len(selected) or not np.isfinite(selected).all():
            raise ValueError('INVALID_MAP: NoData triangle in fixed Hop')
        return selected

    def max_height(self, lo, hi):
        stats = self._query_stats['max_height']; stats['queries'] += 1
        started = time.perf_counter()
        try:
            triangles = self.triangles(lo, hi); stats['triangle_visits'] += len(triangles)
            return float(triangles[..., 2].max())
        finally:
            stats['seconds'] += time.perf_counter()-started

    def height(self, p):
        p = np.asarray(p, float)
        if p.shape[-1] not in (2, 3) or not np.isfinite(p).all():
            raise ValueError('INVALID_MAP: finite XY or XYZ query required')
        result = []
        for xy in p[..., :2].reshape(-1, 2):
            candidates = self.triangles(xy, xy)
            values = []
            for triangle in candidates:
                weights = np.linalg.solve((triangle[1:, :2]-triangle[0, :2]).T, xy-triangle[0, :2])
                if min(weights) >= -32*np.finfo(float).eps and weights.sum() <= 1+32*np.finfo(float).eps:
                    values.append(float(_height_on(triangle, xy[None])[0]))
            if not values:
                raise ValueError('INVALID_MAP: NoData query in fixed Hop')
            result.append(max(values))
        return np.asarray(result).reshape(p.shape[:-1])

    def distance(self, p):
        p = np.asarray(p, float); stats = self._query_stats['distance']; stats['queries'] += 1
        started = time.perf_counter()
        try:
            height = float(self.height(p))
            if p[2] <= height:
                return p[2]-height
            radius = p[2]-height
            triangles = self.triangles(np.maximum(p[:2]-radius, self.bounds[0]), np.minimum(p[:2]+radius, self.bounds[1]))
            stats['triangle_visits'] += len(triangles)
            return float(point_triangle_distances(p, triangles).min())
        finally:
            stats['seconds'] += time.perf_counter()-started


class _RasterOperationSource(OperationSource):
    """Exact same-grid Hop with local cell indexing, not a resampled surface.

Both source triangulations have identical coordinates and the same diagonal,
so their common refinement is precisely the original grid.  Queries return
the same complete closed-cell triangles, including boundary-touching cells.
"""
    def __init__(self,original,hseal,query_mode):
        self._x,self._y=original.x,original.y
        self._dsm=np.maximum(np.maximum(original.dtm,original.dsm),hseal)  # Eq. (1), nodal H_op on the shared grid.
        self._dtm=self._dsm
        self.bounds=np.asarray(original.bounds,float).copy()
        self.source_geometry_version=SOURCE_GEOMETRY_VERSION
        # Preserve the existing source-identity byte sequence without retaining
        # the millions of explicit triangles. This is the existing cache key.
        identity=hashlib.sha256(self.source_geometry_version.encode())
        for family in (0,1):
            for j in range(0,len(self.y)-1,32):
                last=min(j+32,len(self.y)-1)
                triangles=raster_triangles(self.x,self.y[j:last+1],self.dsm[j:last+1])
                count=len(triangles)//2
                identity.update(triangles[family*count:(family+1)*count].tobytes())
        identity.update(self.bounds.tobytes())
        self.source_geometry_id=identity.hexdigest()
        for array in (self.x,self.y,self.dsm,self.bounds):array.flags.writeable=False
        self.set_query_mode(query_mode)
        self.reset_query_statistics()

    def _cell_range(self,lo,hi):
        lo,hi=np.asarray(lo,float),np.asarray(hi,float)
        if np.any(lo<self.bounds[0]) or np.any(hi>self.bounds[1]) or np.any(lo>hi):
            raise ValueError('INVALID_MAP: query outside coverage')
        return Terrain._cell_range(self,lo,hi)

    def raster_window(self, lo, hi):
        """Expose immutable node arrays and inclusive cell-corner bounds."""
        return self.x, self.y, self.dsm, self._cell_range(lo, hi)

    def _triangles_cells(self,cells):
        return Terrain._triangles_cells(self,cells)

    def triangles(self,lo,hi):
        return self._triangles_cells(self._cell_range(lo,hi))

    def max_height(self,lo,hi):
        stats=self._query_stats['max_height'];stats['queries']+=1
        started=time.perf_counter()
        try:
            i0,i1,j0,j1=self._cell_range(lo,hi)
            values=self.dsm[j0:j1+1,i0:i1+1]
            if not values.size or not np.isfinite(values).all():
                raise ValueError('INVALID_MAP: NoData triangle in fixed Hop')
            return float(values.max())
        finally:stats['seconds']+=time.perf_counter()-started

    def query_statistics(self):
        return dict(query_mode='same_grid_local_cell_index',requested_query_mode=self.query_mode,
            build_seconds=0.,build_count=0,node_count=0,leaf_count=0,index_memory_bytes_estimate=0,
            source_array_bytes=self.x.nbytes+self.y.nbytes+self.dsm.nbytes,
            source_triangle_count=2*(len(self.x)-1)*(len(self.y)-1),
            queries={k:dict(v) for k,v in self._query_stats.items()},
            query_seconds=sum(v['seconds'] for v in self._query_stats.values()),
            build_and_query_seconds=sum(v['seconds'] for v in self._query_stats.values()),
            source_geometry_version=self.source_geometry_version,source_geometry_id=self.source_geometry_id)
