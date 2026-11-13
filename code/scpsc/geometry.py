"""Height-field triangle queries and corridor geometry for the obstacle model in Eq. (1)."""
from dataclasses import dataclass
import heapq
import sys
import time
import numpy as np
from .envelope import active_endpoint_faces,normalized_halfspaces


def clip_polygon(poly, normal, bound):
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ga, gb = a[:2] @ normal - bound, b[:2] @ normal - bound
        if ga <= 0:
            out.append(a)
        if (ga < 0 < gb) or (gb < 0 < ga):
            out.append(a + ga / (ga - gb) * (b - a))
    return np.asarray(out).reshape(-1, 3)


def point_triangle_distances(p, tris):
    """Exact Euclidean distances to closed triangles, independently of corridors."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    ab, ac, ap = b-a, c-a, p-a
    n = np.cross(ab, ac)
    nn = np.einsum('ij,ij->i', n, n)
    proj = ap - (np.einsum('ij,ij->i', ap, n)/nn)[:, None]*n
    d00 = np.einsum('ij,ij->i', ab, ab)
    d01 = np.einsum('ij,ij->i', ab, ac)
    d11 = np.einsum('ij,ij->i', ac, ac)
    d20 = np.einsum('ij,ij->i', proj, ab)
    d21 = np.einsum('ij,ij->i', proj, ac)
    den = d00*d11-d01*d01
    u, v = (d11*d20-d01*d21)/den, (d00*d21-d01*d20)/den
    dsq = np.where((u >= 0)&(v >= 0)&(u+v <= 1),
                    np.einsum('ij,ij->i', ap,n)**2/nn, np.inf)
    for e0, e1 in ((a,b),(b,c),(c,a)):
        e=e1-e0
        t=np.clip(np.einsum('ij,ij->i',p-e0,e)/np.einsum('ij,ij->i',e,e),0,1)
        r=p-e0-t[:,None]*e
        dsq=np.minimum(dsq,np.einsum('ij,ij->i',r,r))
    return np.sqrt(dsq)


@dataclass(slots=True)
class _TerrainQuadNode:
    # Half-open ORIGINAL cell indices also identify the original two triangles.
    i0: int
    i1: int
    j0: int
    j1: int
    zmin: float
    zmax: float
    valid: bool
    children: tuple


class _TerrainQuadTree:
    def __init__(self, terrain, leaf_cells):
        self.terrain=terrain
        z=terrain.dsm
        corners=(z[:-1,:-1],z[:-1,1:],z[1:,:-1],z[1:,1:])
        self.cell_min=np.minimum.reduce(corners)
        self.cell_max=np.maximum.reduce(corners)
        self.cell_valid=np.logical_and.reduce([np.isfinite(v) for v in corners])
        valid_x=np.isfinite(terrain.x[:-1])&np.isfinite(terrain.x[1:])
        valid_y=np.isfinite(terrain.y[:-1])&np.isfinite(terrain.y[1:])
        self.cell_valid&=valid_y[:,None]&valid_x[None,:]
        self.node_count=self.leaf_count=0
        self.memory_bytes=sum(sys.getsizeof(a) for a in (self.cell_min,self.cell_max,self.cell_valid))

        def build(i0,i1,j0,j1):
            children=()
            if (i1-i0)*(j1-j0)>leaf_cells:
                ii=(i0,(i0+i1)//2,i1) if i1-i0>1 else (i0,i1)
                jj=(j0,(j0+j1)//2,j1) if j1-j0>1 else (j0,j1)
                children=tuple(build(a,b,c,d) for c,d in zip(jj,jj[1:])
                               for a,b in zip(ii,ii[1:]))
                low=float(np.minimum.reduce([c.zmin for c in children]))
                high=float(np.maximum.reduce([c.zmax for c in children]))
                valid=all(c.valid for c in children)
            else:
                low=float(self.cell_min[j0:j1,i0:i1].min())
                high=float(self.cell_max[j0:j1,i0:i1].max())
                valid=bool(self.cell_valid[j0:j1,i0:i1].all())
                self.leaf_count+=1
            node=_TerrainQuadNode(i0,i1,j0,j1,low,high,valid,children)
            self.node_count+=1
            self.memory_bytes+=sys.getsizeof(node)+sys.getsizeof(children)
            self.memory_bytes+=sum(sys.getsizeof(v) for v in (i0,i1,j0,j1,low,high))
            return node

        self.root=build(0,len(terrain.x)-1,0,len(terrain.y)-1)

    @staticmethod
    def overlap(node,cells):
        i0,i1,j0,j1=cells
        return max(i0,node.i0),min(i1,node.i1),max(j0,node.j0),min(j1,node.j1)

    def maximum(self,cells,stats):
        best=-np.inf
        stack=[self.root]
        while stack:
            node=stack.pop();stats['node_visits']+=1
            i0,i1,j0,j1=self.overlap(node,cells)
            if i0>=i1 or j0>=j1:continue
            if not node.children:stats['leaf_visits']+=1
            if (i0,i1,j0,j1)==(node.i0,node.i1,node.j0,node.j1):
                if not node.valid:raise ValueError('INVALID_MAP: NoData triangle')
                best=max(best,node.zmax)
            elif node.children:stack.extend(node.children)
            else:
                if not self.cell_valid[j0:j1,i0:i1].all():
                    raise ValueError('INVALID_MAP: NoData triangle')
                best=max(best,float(self.cell_max[j0:j1,i0:i1].max()))
        return float(best)

    def nearest(self,p,cells,stats):
        # Validate ALL original candidate cells before distance pruning. An
        # unrelated invalid sibling is not part of a partial-window query.
        self.maximum(cells,stats)
        terrain=self.terrain

        def lower_bound(node):
            if not node.valid:return 0.  # Unknown box: descend, never prune.
            lo=np.array([terrain.x[node.i0],terrain.y[node.j0],node.zmin])
            hi=np.array([terrain.x[node.i1],terrain.y[node.j1],node.zmax])
            return float(np.linalg.norm(np.maximum(np.maximum(lo-p,p-hi),0.)))

        queue=[(lower_bound(self.root),0,self.root)]
        serial=0;best=np.inf
        while queue:
            lower,_,node=heapq.heappop(queue);stats['node_visits']+=1
            # Keep uncertain/equidistant boxes. The padding only prevents
            # pruning; it never changes the exact triangle result/tolerance.
            pad=64*np.finfo(float).eps*max(1.,float(np.max(np.abs(p))),
                abs(node.zmin),abs(node.zmax),abs(best) if np.isfinite(best) else 1.)
            if np.isfinite(lower) and np.isfinite(best) and lower>best+pad:continue
            i0,i1,j0,j1=self.overlap(node,cells)
            if i0>=i1 or j0>=j1:continue
            if node.children:
                for child in node.children:
                    a,b,c,d=self.overlap(child,cells)
                    if a<b and c<d:
                        serial+=1
                        heapq.heappush(queue,(lower_bound(child),serial,child))
            else:
                stats['leaf_visits']+=1
                triangles=terrain._triangles_cells((i0,i1,j0,j1))
                stats['triangle_visits']+=len(triangles)
                best=float(np.minimum(best,point_triangle_distances(p,triangles).min()))
        return float(best)


class Terrain:
    def __init__(self, x, y, dtm, dsm, query_mode='scan', quadtree_leaf_cells=64):
        self.quadtree_leaf_cells=int(quadtree_leaf_cells)
        if self.quadtree_leaf_cells<1:raise ValueError('quadtree_leaf_cells must be positive')
        self._quadtree=None
        self.reset_query_statistics()
        self.replace_map(x,y,dtm,dsm)
        self.set_query_mode(query_mode)

    @property
    def x(self):return self._x

    @property
    def y(self):return self._y

    @property
    def dtm(self):return self._dtm

    @property
    def dsm(self):return self._dsm

    def replace_map(self,x,y,dtm,dsm):
        """Install an immutable source map and explicitly invalidate its index."""
        self._quadtree=None
        self._x,self._y=np.array(x,copy=True),np.array(y,copy=True)
        self._dtm,self._dsm=np.array(dtm,copy=True),np.array(dsm,copy=True)
        if self.dsm.shape != (len(y),len(x)) or self.dtm.shape != self.dsm.shape:
            raise ValueError('INVALID_MAP: raster dimensions')
        if not (np.all(np.diff(x)>0) and np.all(np.diff(y)>0)):
            raise ValueError('INVALID_MAP: coordinate ordering')
        self.bounds=np.array([[x[0],y[0]],[x[-1],y[-1]]])
        for a in (self.x,self.y,self.dtm,self.dsm,self.bounds):a.flags.writeable=False

    def set_query_mode(self,mode):
        if mode not in ('scan','quadtree'):raise ValueError('Unknown terrain query mode: '+str(mode))
        self.query_mode=mode

    def reset_query_statistics(self):
        self._query_stats={name:{'queries':0,'seconds':0.,'node_visits':0,
                                'leaf_visits':0,'triangle_visits':0}
                           for name in ('max_height','distance')}
        self._build_seconds=0.;self._build_count=0

    def query_statistics(self):
        tree=self._quadtree
        return {'query_mode':self.query_mode,'quadtree_leaf_cells':self.quadtree_leaf_cells,
                'build_seconds':self._build_seconds,'build_count':self._build_count,
                'node_count':tree.node_count if tree else 0,
                'leaf_count':tree.leaf_count if tree else 0,
                'index_memory_bytes_estimate':tree.memory_bytes if tree else 0,
                'memory_scope':'index arrays and Python node/scalar storage; excludes source map and transient query arrays',
                'source_array_bytes':sum(a.nbytes for a in (self.x,self.y,self.dtm,self.dsm)),
                'queries':{name:dict(value) for name,value in self._query_stats.items()},
                'query_seconds':sum(s['seconds'] for s in self._query_stats.values()),
                'build_and_query_seconds':self._build_seconds+sum(s['seconds'] for s in self._query_stats.values())}

    def _index(self):
        if self._quadtree is None:
            started=time.perf_counter()
            self._quadtree=_TerrainQuadTree(self,self.quadtree_leaf_cells)
            self._build_seconds+=time.perf_counter()-started;self._build_count+=1
        return self._quadtree

    @classmethod
    def load(cls,path,query_mode='scan',quadtree_leaf_cells=64):
        d=np.load(path)
        return cls(d['x'],d['y'],d['dtm'],d['dsm'],query_mode,quadtree_leaf_cells)

    def height(self,p):
        p=np.asarray(p); xy=p[...,:2]
        if np.any(xy<self.bounds[0]) or np.any(xy>self.bounds[1]):
            raise ValueError('INVALID_MAP: query outside coverage')
        ix=np.minimum(np.searchsorted(self.x,xy[...,0],side='right')-1,len(self.x)-2)
        iy=np.minimum(np.searchsorted(self.y,xy[...,1],side='right')-1,len(self.y)-2)
        u=(xy[...,0]-self.x[ix])/(self.x[ix+1]-self.x[ix])
        v=(xy[...,1]-self.y[iy])/(self.y[iy+1]-self.y[iy])
        z00=self.dsm[iy,ix];z10=self.dsm[iy,ix+1]
        z01=self.dsm[iy+1,ix];z11=self.dsm[iy+1,ix+1]
        z=np.where(v<=u,z00+(z10-z00)*u+(z11-z10)*v,
                   z00+(z11-z01)*u+(z01-z00)*v)
        if not np.isfinite(z).all(): raise ValueError('INVALID_MAP: NoData query')
        return z

    def _cell_range(self,lo,hi):
        i0=max(0,np.searchsorted(self.x,lo[0])-1);i1=min(len(self.x)-1,np.searchsorted(self.x,hi[0],side='right'))
        j0=max(0,np.searchsorted(self.y,lo[1])-1);j1=min(len(self.y)-1,np.searchsorted(self.y,hi[1],side='right'))
        return i0,i1,j0,j1

    def triangles(self,lo,hi):
        return self._triangles_cells(self._cell_range(lo,hi))

    def _triangles_cells(self,cells):
        i0,i1,j0,j1=cells
        xx,yy=np.meshgrid(self.x[i0:i1+1],self.y[j0:j1+1])
        vertices=np.stack((xx,yy,self.dsm[j0:j1+1,i0:i1+1]),-1)
        a=vertices[:-1,:-1].reshape(-1,3); b=vertices[:-1,1:].reshape(-1,3)
        c=vertices[1:,1:].reshape(-1,3); d=vertices[1:,:-1].reshape(-1,3)
        ts=np.concatenate((np.stack((a,b,c),1),np.stack((a,c,d),1)))
        if not np.isfinite(ts).all(): raise ValueError('INVALID_MAP: NoData triangle')
        return ts

    def max_height(self,lo,hi):
        """Same complete-cell maximum/rejection as triangles(lo,hi)[:,:,2].max()."""
        cells=self._cell_range(lo,hi)
        i0,i1,j0,j1=cells
        # The original empty-window reduction raises rather than returning -inf.
        if i0>=i1 or j0>=j1:return float(self.triangles(lo,hi)[:,:,2].max())
        tree=self._index() if self.query_mode=='quadtree' else None
        stats=self._query_stats['max_height'];stats['queries']+=1
        started=time.perf_counter()
        try:
            if tree is not None:return tree.maximum(cells,stats)
            ts=self._triangles_cells(cells);stats['triangle_visits']+=len(ts)
            return float(ts[:,:,2].max())
        finally:stats['seconds']+=time.perf_counter()-started

    def envelope(self,lo,hi,eta=0.):
        if np.any(lo<self.bounds[0]) or np.any(hi>self.bounds[1]):
            raise ValueError('CORRIDOR_INCOMPATIBLE: window lacks complete map coverage')
        ts=self.triangles(lo,hi)
        inside=np.all((ts[...,:2]>=lo)&(ts[...,:2]<=hi),axis=(1,2))
        vertices=[ts[inside].reshape(-1,3)]
        for t in ts[~inside]:
            poly=t
            for n,b in (([1,0],hi[0]),([-1,0],-lo[0]),([0,1],hi[1]),([0,-1],-lo[1])):
                if not len(poly):break
                poly=clip_polygon(poly,np.array(n),b)
            if len(poly): vertices.append(poly)
        v=np.concatenate(vertices)
        if len(v)<3:raise ValueError('CORRIDOR_INCOMPATIBLE: no clipped triangles')
        center=v[:,:2].mean(axis=0)
        slope=np.linalg.lstsq(np.c_[v[:,:2]-center,np.ones(len(v))],v[:,2],rcond=None)[0][:2]
        beta=np.max(v[:,2]+eta-v[:,:2]@slope)
        return slope,beta,{'clipped_vertices':len(v),'boundary_triangles':int((~inside).sum()),
                          'envelope_gap_min':float(np.min(v[:,:2]@slope+beta-v[:,2]-eta))}

    def distance(self,p):
        """Signed above-surface distance to source mesh; unknown coverage rejected."""
        tree=self._index() if self.query_mode=='quadtree' else None
        stats=self._query_stats['distance'];stats['queries']+=1
        started=time.perf_counter()
        try:return self._distance(p,tree,stats)
        finally:stats['seconds']+=time.perf_counter()-started

    def _distance(self,p,tree,stats):
        p=np.asarray(p)
        height=float(self.height(p))
        if p[2]<=height:return p[2]-height
        def nearest(lo,hi):
            if tree is not None:return tree.nearest(p,self._cell_range(lo,hi),stats)
            triangles=self.triangles(lo,hi);stats['triangle_visits']+=len(triangles)
            return float(np.min(point_triangle_distances(p,triangles)))
        upper=p[2]-height
        # A nearby exact triangle distance tightens the search radius; it remains
        # an upper bound, so every possibly closer triangle is still included.
        if upper>15. and max(np.diff(self.x[:2])[0],np.diff(self.y[:2])[0])<5.:
            upper=min(upper,nearest(np.maximum(p[:2]-15.,self.bounds[0]),np.minimum(p[:2]+15.,self.bounds[1])))
        lo=np.maximum(p[:2]-upper,self.bounds[0]);hi=np.minimum(p[:2]+upper,self.bounds[1])
        return nearest(lo,hi)

    def cable_clearance(self,a,b,radius,spacing=.5):
        """1-Lipschitz lower bound over the ENTIRE straight segment at this instant."""
        return self.cable_clearance_bounds(a,b,radius,spacing)[0]

    def cable_clearance_bounds(self,a,b,radius,spacing=.5):
        """Lower enclosure and sampled upper bound on minimum segment clearance."""
        length=np.linalg.norm(b-a); count=max(1,int(np.ceil(length/spacing)))
        vals=[self.distance(a+(b-a)*s) for s in np.linspace(0,1,count+1)]
        return float(min(vals)-length/(2*count)-radius),float(min(vals)-radius)

    def fast_clearance(self,p,radius):
        # A disk-containing square maximum is a conservative front-end screen.
        p=np.asarray(p)
        if np.any(p[:2]-radius<self.bounds[0]) or np.any(p[:2]+radius>self.bounds[1]):return -np.inf
        return p[2]-self.max_height(p[:2]-radius,p[:2]+radius)-radius


@dataclass
class Corridor:
    normals: np.ndarray
    bounds: np.ndarray
    proof: dict
    original_normals: np.ndarray = None
    original_bounds: np.ndarray = None

    def residual(self,p,radius):
        normals=self.normals if self.original_normals is None else self.original_normals
        bounds=self.bounds if self.original_bounds is None else self.original_bounds
        normals,bounds=normalized_halfspaces(normals,bounds)
        return np.asarray(p)@normals.T+radius-bounds

    def active_faces(self,ell,rhoL,rhoQ,theta,prune=True):
        return active_endpoint_faces(self.normals,ell,rhoL,rhoQ,theta,prune=prune)


def remove_duplicates(normals,bounds):
    # Keep each exact direction in first-seen order; retain the strictest
    # row (including its original signed zeros) when its bound improves.
    kept = {}
    for n, b in zip(normals, bounds):
        key = tuple(n)
        old = kept.get(key)
        if old is None or b < old[1]:
            kept[key] = (n, b)
    return np.asarray([n for n, _ in kept.values()]), np.asarray([b for _, b in kept.values()])



class CorridorIncompatible(ValueError):
    """A local initialization failure, including its actual segment and face."""
    def __init__(self,segment,face,reason,diagnostics=None,attempts=None):
        self.segment=int(segment);self.face=None if face is None else int(face)
        self.diagnostics=diagnostics or {};self.attempts=attempts or []
        self.split_segment=self.segment
        self.reason=reason
        super().__init__(f'CORRIDOR_INCOMPATIBLE: segment {self.segment}, face {self.face}, {reason}')

    def to_dict(self):
        return {'segment':self.segment,'face':self.face,'reason':self.reason,
                'diagnostics':self.diagnostics,'attempts':self.attempts,
                'split_segment':self.split_segment}


def seed_plane_interval(slope,c_lo,points,radii):
    """For this fixed slope, no intercept can work if c_lo exceeds c_hi."""
    norm=float(np.sqrt(1+np.dot(slope,slope)))
    c_hi=float(np.min(points[:,2]-points[:,:2]@slope-radii*norm))
    return {'slope':np.asarray(slope).tolist(),'c_lo':float(c_lo),'c_hi':c_hi,
            'intercept_gap_m':c_hi-float(c_lo),
            'normalized_seed_margin_m':(c_hi-float(c_lo))/norm,
            'fixed_slope_seed_feasible':bool(c_lo<=c_hi),
            'scope':'specified inflated knot spheres only; not the entire polynomial segment'}


