"""Complete source fragments, window clipping, and payload-distance brackets."""
from dataclasses import dataclass

import numpy as np

from .piecewise_native import piece_geometry_library


@dataclass(frozen=True)
class SourcePieces:
    vertices: np.ndarray
    offsets: np.ndarray
    source_ids: np.ndarray
    input_triangle_count: int

    def __len__(self):
        return len(self.offsets) - 1

    def piece(self, i):
        return self.vertices[self.offsets[i] : self.offsets[i + 1]]

    def validate(self):
        v, o = self.vertices, self.offsets
        if (
            v.ndim != 2
            or v.shape[1] != 3
            or not np.isfinite(v).all()
            or o.ndim != 1
            or len(o) < 2
            or o.dtype.kind not in "iu"
            or o[0] != 0
            or o[-1] != len(v)
            or np.any(np.diff(o) < 1)
            or np.any(np.diff(o) > 7)
            or self.source_ids.shape != (len(self),)
        ):
            raise ValueError("PIECE_GEOMETRY: finite complete clipped fragments required")


def _source_raster_window(terrain, lo, hi):
    # Algorithm 1, ClipWindow. Alternate surface triangulations keep their own query.
    raster_window = getattr(terrain, "raster_window", None)
    if raster_window is not None:
        x, y, height, cells = raster_window(lo, hi)
        if (x.ndim == y.ndim == 1 and height.shape == (len(y), len(x))
                and all(a.dtype == np.float64 and a.flags.c_contiguous for a in (x, y, height))):
            return x, y, height, cells
    return None


def _window_triangles(terrain, lo, hi):
    # Retained for the triangle-order comparison benchmark.
    raster = _source_raster_window(terrain, lo, hi)
    if raster is not None:
        x, y, height, (i0, i1, j0, j1) = raster
        count = 2 * (i1 - i0) * (j1 - j0)
        triangles = np.empty((count, 3, 3), dtype=np.float64)
        status = piece_geometry_library().scpsc_raster_window(
            x, y, height, len(x), len(y), i0, i1, j0, j1, triangles, count)
        if status == -2:
            raise ValueError("INVALID_MAP: NoData triangle")
        if status:
            raise ValueError("INVALID_MAP: invalid raster window")
        return triangles
    triangles = np.ascontiguousarray(terrain.triangles(lo, hi), dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
        raise ValueError("INVALID_MAP: unknown or invalid source triangle")
    return triangles


def clipped_source_pieces(terrain, lo, hi, eta=0.0):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if (
        lo.shape != (2,)
        or hi.shape != (2,)
        or not np.isfinite([lo, hi]).all()
        or np.any(lo >= hi)
        or not np.isfinite(eta)
        or eta < 0.0
    ):
        raise ValueError("Invalid XY source window or vertical error")
    if np.any(lo < terrain.bounds[0]) or np.any(hi > terrain.bounds[1]):
        raise ValueError("CORRIDOR_INCOMPATIBLE: source window outside known terrain")
    raster = _source_raster_window(terrain, lo, hi)
    if raster is not None:
        x, y, height, (i0, i1, j0, j1) = raster
        triangle_count = int(2 * (i1 - i0) * (j1 - j0))
    else:
        triangles = _window_triangles(terrain, lo, hi)
        triangle_count = len(triangles)
    vertices: np.ndarray = np.empty((7 * triangle_count, 3), dtype=np.float64)
    offsets = np.empty(triangle_count + 1, dtype=np.int64)
    ids = np.empty(triangle_count, dtype=np.int64)
    clip_args = (np.ascontiguousarray(lo), np.ascontiguousarray(hi),
                 vertices, len(vertices), offsets, ids)
    if raster is not None:
        count = piece_geometry_library().scpsc_clip_raster_pieces(
            x, y, height, len(x), len(y), i0, i1, j0, j1, *clip_args)
        if count == -2:
            raise ValueError("INVALID_MAP: NoData triangle")
        if count == -1:
            raise ValueError("INVALID_MAP: invalid raster window")
    else:
        count = piece_geometry_library().scpsc_clip_pieces(triangles, triangle_count, *clip_args)
    if count < 0:
        raise RuntimeError("PIECE_CLIP_FAILED: native code " + str(count))
    if count == 0:
        raise ValueError("CORRIDOR_INCOMPATIBLE: no source coverage")
    vertices = vertices[: offsets[count]].copy()
    vertices[:, 2] += eta
    out = SourcePieces(vertices, offsets[: count + 1].copy(), ids[:count].copy(), triangle_count)
    out.validate()
    return out


def payload_distance_brackets(pieces, seeds, tolerance=1e-7, max_iter=96):
    """Batch segment-to-downward-prism distances; the ball is not yet subtracted."""
    pieces.validate()
    s = np.ascontiguousarray(seeds, dtype=np.float64)
    if s.shape != (2, 3) or not np.isfinite(s).all():
        raise ValueError("Expected two finite seed endpoints")
    out = np.empty((len(pieces), 7), dtype=np.float64)
    status = piece_geometry_library().scpsc_payload_brackets(
        np.ascontiguousarray(pieces.vertices),
        len(pieces.vertices),
        np.ascontiguousarray(pieces.offsets, dtype=np.int64),
        len(pieces),
        s,
        float(tolerance),
        int(max_iter),
        out,
    )
    if status != 0 or not np.isfinite(out).all():
        raise RuntimeError("PIECE_DISTANCE_FAILED: native code " + str(status))
    if np.any(out[:, 3] < 0.0) or np.any(out[:, 4] < out[:, 3] - tolerance):
        raise RuntimeError("PIECE_DISTANCE_FAILED: inconsistent geometric bounds")
    return out


def _guard(*arrays):
    return 256.0 * np.finfo(float).eps * (1.0 + sum(float(np.max(np.abs(a))) for a in arrays if np.size(a)))
