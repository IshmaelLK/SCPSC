"""Geometry-only input preparation and explicit independent task-volume schema."""

from pathlib import Path
import json
import time
import numpy as np
from .envelope import fixed_support, planning_envelope_mode, planning_envelope_parameters



def build_voxel_grid(source, domain, config, *, checkpoint=None):
    """Rebuild closed-cell + selected-planning-envelope checks from source only.

    No prior grid, field volume, validity-mask or per-column field floor is read.
    Unknown source cells remain blocked. The source maximum is conservative over
    every source triangle intersecting the cell+planning-envelope XY footprint.
    """
    started = time.perf_counter()
    h = float(config["frontend"].get("voxel_spacing_m", 4.0))
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Positive finite voxel spacing required")
    domain = np.asarray(domain, float)
    mode = planning_envelope_mode(config["ubsqp"])
    # Legacy point retains its historical frontend; bare_point is end to end.
    front_mode = mode if mode in ("load_only", "bare_point") else "fixed_system"
    planning = planning_envelope_parameters(config["dynamics"]["ell"], config["limits"], front_mode)
    extent = fixed_support(np.r_[-np.eye(3), np.eye(3)], *planning)
    minus, plus = extent[:3], extent[3:]
    lower = domain[0].copy()
    counts = np.floor((domain[1] - lower) / h).astype(np.int64)
    if np.any(counts < 1) or np.prod(counts, dtype=object) > int(config["frontend"].get("max_voxels", 30_000_000)):
        raise ValueError("GRID_SIZE_LIMIT: change registered volume/resolution explicitly")
    upper = lower + h * counts
    origin = lower + h / 2
    xx, yy, zz = [origin[k] + h * np.arange(counts[k]) for k in range(3)]
    grid = np.zeros((counts[2], counts[1], counts[0]), np.uint8)
    for j, y in enumerate(yy):
        if checkpoint is not None:
            checkpoint("preparation")
        for i, x in enumerate(xx):
            lo = np.array([x, y]) - h / 2 - minus[:2]
            hi = np.array([x, y]) + h / 2 + plus[:2]
            if (
                np.any(lo < domain[0, :2])
                or np.any(hi > domain[1, :2])
                or np.any(lo < source.bounds[0])
                or np.any(hi > source.bounds[1])
            ):
                continue
            try:
                height = source.max_height(lo, hi)
            except ValueError as error:
                if "NoData" in str(error):
                    continue
                raise
            floor = max(height, domain[0, 2]) + minus[2] + h / 2
            ceiling = domain[1, 2] - plus[2] - h / 2
            # Eq. (14), mark closed voxels whose envelope-expanded volume clears the terrain and domain faces.
            grid[:, j, i] = ((zz > floor) & (zz < ceiling)).astype(np.uint8)
    return grid, dict(
        schema="SCPSC_HOP_CLOSED_VOXELS_V2",
        nx=int(counts[0]),
        ny=int(counts[1]),
        nz=int(counts[2]),
        h=h,
        origin=origin.tolist(),
        bounds=[lower.tolist(), upper.tolist()],
        operation_domain=domain.tolist(),
        storage="uint8 C[z,y,x], 1=certified free",
        support_negative_m=minus.tolist(),
        support_positive_m=plus.tolist(),
        free_count=int(grid.sum()),
        cell_count=int(grid.size),
        preparation_seconds=time.perf_counter() - started,
        planning_envelope=front_mode,
        requested_planning_envelope=mode,
        legacy_point_frontend=(mode == "point"),
        source_geometry_version=getattr(source, "source_geometry_version", "LEGACY_H0_TEST_ONLY"),
        source_geometry_id=getattr(source, "source_geometry_id", None),
        definition="closed voxel + " + front_mode + " six-axis support above fixed Hop, inside independent task volume",
    )


def save_grid(directory, grid, meta):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(grid, dtype=np.uint8).tofile(d / "grid.bin")
    (d / "grid.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return d / "grid.bin"
