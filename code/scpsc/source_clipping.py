"""Window clipping shared by single-plane construction and AABB sources."""
from functools import lru_cache
from pathlib import Path
import ctypes
import os

import numpy as np

from .geometry import clip_polygon
from .piecewise_geometry import _source_raster_window


def clipped_source_vertices_python(terrain,lo,hi,eta=0.):
    """Original implementation retained for numerical equivalence checks."""
    lo,hi=np.asarray(lo,float),np.asarray(hi,float)
    if lo.shape!=(2,) or hi.shape!=(2,) or np.any(lo>=hi):raise ValueError('Invalid XY source window')
    if np.any(lo<terrain.bounds[0]) or np.any(hi>terrain.bounds[1]):
        raise ValueError('CORRIDOR_INCOMPATIBLE: source window outside known terrain')
    triangles=terrain.triangles(lo,hi)
    if not np.isfinite(triangles).all():raise ValueError('INVALID_MAP: NoData triangle')
    inside=np.all((triangles[...,:2]>=lo)&(triangles[...,:2]<=hi),axis=(1,2))
    parts=[triangles[inside].reshape(-1,3)] if np.any(inside) else []
    for tri in triangles[~inside]:
        p=tri
        for n,b in (([1,0],hi[0]),([-1,0],-lo[0]),([0,1],hi[1]),([0,-1],-lo[1])):
            if len(p):p=clip_polygon(p,np.asarray(n),b)
        if len(p):parts.append(p)
    if not parts:raise ValueError('CORRIDOR_INCOMPATIBLE: no source coverage')
    V=np.unique(np.concatenate(parts),axis=0);V[:,2]+=eta
    return V,dict(source_triangle_count=len(triangles),source_vertex_count=len(V),
                  source_geometry='clipped piecewise-affine canopy')


@lru_cache(maxsize=1)
def _native_source_clip(raster=False):
    root=Path(__file__).resolve().parents[2]
    path=root/'native/build'/('source_clip.dll' if os.name=='nt' else 'source_clip.so')
    if not path.is_file():
        raise RuntimeError('SOURCE_CLIP_DEPENDENCY_UNAVAILABLE: run python scripts/build_source_clip.py')
    library=ctypes.CDLL(str(path))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    index=ctypes.c_int64
    if raster:
        function=library.scpsc_clip_raster_vertices
        function.argtypes=[array,array,array,index,index,index,index,index,index,array,array,array,index]
    else:
        function=library.scpsc_clip_triangles
        function.argtypes=[array,index,array,array,array,index]
    function.restype=ctypes.c_int64
    return function


def clipped_source_vertices(terrain,lo,hi,eta=0.):
    """C++ clipping of the same source triangles, before any hull reduction."""
    lo,hi=np.asarray(lo,float),np.asarray(hi,float)
    if lo.shape!=(2,) or hi.shape!=(2,) or np.any(lo>=hi):raise ValueError('Invalid XY source window')
    if np.any(lo<terrain.bounds[0]) or np.any(hi>terrain.bounds[1]):
        raise ValueError('CORRIDOR_INCOMPATIBLE: source window outside known terrain')
    raster=_source_raster_window(terrain,lo,hi)
    if raster is not None:
        x,y,height,(i0,i1,j0,j1)=raster
        triangle_count=int(2*(i1-i0)*(j1-j0))
        output=np.empty((7*triangle_count,3),dtype=np.float64)
        count=_native_source_clip(True)(x,y,height,len(x),len(y),i0,i1,j0,j1,
            np.ascontiguousarray(lo),np.ascontiguousarray(hi),output,len(output))
        if count==-2:raise ValueError('INVALID_MAP: NoData triangle')
    else:
        triangles=np.ascontiguousarray(terrain.triangles(lo,hi),dtype=np.float64)
        if not np.isfinite(triangles).all():raise ValueError('INVALID_MAP: NoData triangle')
        triangle_count=len(triangles)
        output=np.empty((7*triangle_count,3),dtype=np.float64)
        count=_native_source_clip()(triangles,triangle_count,np.ascontiguousarray(lo),
                                    np.ascontiguousarray(hi),output,len(output))
    if count<0:raise RuntimeError(f'SOURCE_CLIP_FAILED: native error {count}')
    if count==0:raise ValueError('CORRIDOR_INCOMPATIBLE: no source coverage')
    V=output[:count].copy();V[:,2]+=eta
    return V,dict(source_triangle_count=triangle_count,source_vertex_count=len(V),
                  source_geometry='clipped piecewise-affine canopy')
