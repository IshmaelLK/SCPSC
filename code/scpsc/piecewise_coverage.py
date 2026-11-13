"""Independent whole-fragment coverage checks for returned world halfspaces."""
import numpy as np

from .piecewise_native import scpsc_module, scpsc_native_enabled


def _verify_piece_coverage_python(pieces, normals, bounds, owners, *, eps_m=0.0):
    """Independent whole-fragment check against returned WORLD halfspaces."""
    pieces.validate()
    A, b, owner = np.asarray(normals), np.asarray(bounds), np.asarray(owners)
    if (
        A.ndim != 2
        or A.shape[1] != 3
        or b.shape != (len(A),)
        or not np.isfinite(A).all()
        or not np.isfinite(b).all()
        or owner.shape != (len(pieces),)
        or owner.dtype.kind not in "iu"
        or np.any(owner < 0)
        or np.any(owner >= len(A))
        or np.any(A[:, 2] > 0.0)
        or np.any(np.linalg.norm(A, axis=1) <= 0.0)
    ):
        raise ValueError("INVALID_PIECE_COVERAGE_WITNESS")
    # Choose the common origin before projection; world-bound rounding is kept.
    origin = pieces.vertices.mean(axis=0).astype(np.longdouble)
    Al = A.astype(np.longdouble)
    bl = b.astype(np.longdouble) - Al @ origin
    v = pieces.vertices.astype(np.longdouble) - origin
    vertex_owner = np.repeat(owner, np.diff(pieces.offsets))
    values = np.sum(v * Al[vertex_owner], axis=1) - bl[vertex_owner]
    minimum = float(np.min(values / np.linalg.norm(A, axis=1)[vertex_owner]))
    return dict(
        passed=bool(minimum >= eps_m),
        minimum_source_separation_m=minimum,
        covered_piece_count=len(pieces),
        source_vertex_count=len(v),
        rule="each entire fragment belongs to one same separating plane",
    )


def verify_piece_coverage(pieces, normals, bounds, owners, *, eps_m=0.0):
    if not scpsc_native_enabled():
        return _verify_piece_coverage_python(pieces, normals, bounds, owners, eps_m=eps_m)
    pieces.validate()
    A, b, owner = np.asarray(normals), np.asarray(bounds), np.asarray(owners)
    if (
        A.ndim != 2 or A.shape[1] != 3 or b.shape != (len(A),)
        or not np.isfinite(A).all() or not np.isfinite(b).all()
        or owner.shape != (len(pieces),) or owner.dtype.kind not in "iu"
        or np.any(owner < 0) or np.any(owner >= len(A))
        or np.any(A[:, 2] > 0.0) or np.any(np.linalg.norm(A, axis=1) <= 0.0)
    ):
        raise ValueError("INVALID_PIECE_COVERAGE_WITNESS")
    minimum = scpsc_module().verify(
        np.ascontiguousarray(pieces.vertices, dtype=np.float64),
        np.ascontiguousarray(pieces.offsets, dtype=np.int64),
        np.ascontiguousarray(A, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64),
        np.ascontiguousarray(owner, dtype=np.int64),
    )
    return dict(passed=bool(minimum >= eps_m), minimum_source_separation_m=minimum,
                covered_piece_count=len(pieces), source_vertex_count=len(pieces.vertices),
                rule="each entire fragment belongs to one same separating plane")
