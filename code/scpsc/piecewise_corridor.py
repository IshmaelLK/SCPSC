"""Seed-compatible, whole-fragment separation with reusable distance brackets.

Only an entire clipped fragment separated by ONE emitted plane may be removed.
Each raw halfspace retains the complete fixed-envelope seed.  Projection bounds
and feasible witnesses select a numerically nearest uncovered fragment; the
selected normal is reused, not solved again.  The payload-ball shortcut is used
only when the complete envelope support and a distance bracket certify it.

No grid approximation, terrain relaxation, random ordering, wall-time cutoff,
reference modification, or extra erosion is introduced.  IEEE error guards are
used; these are floating-point certificates, not interval-arithmetic proofs.
"""

from dataclasses import dataclass, asdict
from time import perf_counter
import hashlib

import numpy as np

from .envelope import effective_radii, fixed_support
from .plane_contract import PlaneOptions, SeedPlaneFailure, SeedPlaneSolverError
from .piecewise_coverage import verify_piece_coverage
from .piecewise_geometry import SourcePieces, clipped_source_pieces, payload_distance_brackets, _guard
from .piecewise_native import scpsc_module, scpsc_native_enabled
from .piecewise_oracle import FullPlaneOracle, _normal

SCHEMA = "SCPSC_WHOLE_FRAGMENT_SEPARATION_V1"


@dataclass(frozen=True)
class PieceOptions:
    payload_fast_path: bool = True
    lazy_nearest: bool = True
    reuse_solver: bool = True
    nearest_tolerance_m: float = 1e-6
    geometry_max_iter: int = 96

    def __post_init__(self):
        for key in ("payload_fast_path", "lazy_nearest", "reuse_solver"):
            if type(getattr(self, key)) is not bool:
                raise ValueError("piece_solver." + key + " must be bool")
        if not np.isfinite(self.nearest_tolerance_m) or self.nearest_tolerance_m <= 0.0:
            raise ValueError("piece_solver.nearest_tolerance_m must be finite positive")
        if type(self.geometry_max_iter) is not int or self.geometry_max_iter < 1:
            raise ValueError("piece_solver.geometry_max_iter must be a positive integer")


def _select_nearest_piece(active, lower, upper, available, tolerance, solve_full):
    while True:
        ids = np.flatnonzero(active)
        j = int(ids[np.argmin(upper[ids])])
        blocker = int(ids[np.argmin(lower[ids])])
        if lower[blocker] < upper[j] - tolerance:
            solve_full(blocker)
            continue
        if not available[j]:
            solve_full(j)
            continue
        nearest_gap = float(max(0.0, upper[j] - np.min(lower[ids])))
        return j, nearest_gap


def _covered_fragments(vertices, offsets, normal, active, bound, certificate_eps_m, guard):
    """A fragment is removed only when all its vertices share one witness plane."""
    projection = vertices @ normal
    mins = np.minimum.reduceat(projection, offsets[:-1])
    return active & (mins - bound >= certificate_eps_m + guard)


def _separate_pieces_python(pieces, seeds, ell, limits, *, plane_options=None, piece_options=None, checkpoint=None):
    """Raw world normals/bounds and a whole-fragment coverage certificate."""
    begin = perf_counter()
    pieces.validate()
    opt = plane_options if isinstance(plane_options, PlaneOptions) else PlaneOptions(**(plane_options or {}))
    popt = piece_options if isinstance(piece_options, PieceOptions) else PieceOptions(**(piece_options or {}))
    if opt.plane_selection_mode != "normal_margin" or opt.envelope_approximation != "exact":
        raise ValueError("piecewise_socp requires normal_margin and the exact fixed-system envelope")
    s_world = np.asarray(seeds, float)
    if s_world.shape != (2, 3) or not np.isfinite(s_world).all():
        raise ValueError("Expected two finite seed endpoints")
    rhoL, rhoQ = effective_radii(limits)
    theta = float(limits["theta"])
    if not np.isfinite([ell, rhoL, rhoQ, theta]).all() or min(ell, rhoL, rhoQ) <= 0.0 or not 0.0 <= theta < np.pi / 2:
        raise ValueError("Invalid fixed-system envelope")
    origin = s_world.mean(axis=0)
    s = s_world - origin
    local = SourcePieces(
        np.ascontiguousarray(pieces.vertices - origin), pieces.offsets, pieces.source_ids, pieces.input_triangle_count
    )
    v, offsets, count = local.vertices, local.offsets, len(local)
    if checkpoint is not None:
        checkpoint("corridor")
    tick = perf_counter()
    geom = payload_distance_brackets(
        local, s, min(popt.nearest_tolerance_m * 0.05, opt.tol * 10.0), popt.geometry_max_iter
    )
    payload_seconds = perf_counter() - tick
    normals = geom[:, :3].copy()
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    lengths = np.linalg.norm(normals, axis=1)
    supports = fixed_support(normals, ell, rhoL, rhoQ, theta)
    vnorm = np.repeat(normals, np.diff(offsets), axis=0)
    projection = np.sum(v * vnorm, axis=1)
    source_min = np.minimum.reduceat(projection, offsets[:-1])
    seed_max = np.max(normals @ s.T, axis=1)
    guard = _guard(v, s, [ell, rhoL, rhoQ])
    lower = np.maximum(0.0, (source_min - seed_max - supports) / lengths - guard)
    # A ball at every seed is a subset of the full envelope. A feasible segment
    # distance witness therefore gives this full-envelope distance upper bound.
    upper = np.maximum(0.0, geom[:, 4] - rhoL) + guard
    fast_eligible = (
        (supports <= rhoL * lengths)
        & (upper - lower <= popt.nearest_tolerance_m)
        & (geom[:, 6] == 1.0)
        & (lower > opt.certificate_eps_m + guard)
    )
    available = fast_eligible & popt.payload_fast_path
    # Distances/normals are cached per fragment, never reset after coverage.
    oracle = None
    active = np.ones(count, dtype=bool)
    owners = np.full(count, -1, dtype=np.int64)
    emitted_n: list[np.ndarray] = []
    emitted_b: list[float] = []
    selection: list[dict] = []
    full_done = np.zeros(count, dtype=bool)
    coverage_seconds = nearest_seconds = statistics_seconds = 0.0
    vertex_projections = 0
    full_solves = 0

    def solve_full(j):
        nonlocal oracle, full_solves
        if full_done[j]:
            raise SeedPlaneSolverError("PIECE_DISTANCE_UNCERTAIN: repeated unresolved cached distance")
        if checkpoint is not None:
            checkpoint("corridor")
        if oracle is None:
            oracle = FullPlaneOracle(s, ell, limits, opt, popt.reuse_solver)
        n, lb, ub, record = oracle.solve(local.piece(j), popt.nearest_tolerance_m * 0.25)
        normals[j] = n
        lower[j] = max(lower[j], lb)
        upper[j] = min(upper[j], ub)
        if lower[j] > upper[j] + guard:
            raise SeedPlaneSolverError("PIECE_DISTANCE_BRACKET_CONFLICT")
        upper[j] = max(lower[j], upper[j])
        full_done[j] = True
        available[j] = True
        full_solves += 1

    # Constant over this unchanged window; do not scan source magnitudes per face.
    world_guard = _guard(pieces.vertices, origin, [ell, rhoL, rhoQ])
    if not popt.lazy_nearest:
        for j in range(count):
            if not available[j]:
                solve_full(j)
    while np.any(active):
        if checkpoint is not None:
            checkpoint("corridor")
        tick = perf_counter()
        j, nearest_gap = _select_nearest_piece(active, lower, upper, available, popt.nearest_tolerance_m, solve_full)
        nearest_seconds += perf_counter() - tick
        n = _normal(normals[j])
        # The raw plane is anchored by the COMPLETE selected fragment only.
        # It must not be moved back onto the global-window single majorant.
        b_local = float(np.min(local.piece(j) @ n) - opt.certificate_eps_m - world_guard)
        b_world = float(b_local + n @ origin)
        actual_local_b = float(np.longdouble(b_world) - np.sum(n.astype(np.longdouble) * origin.astype(np.longdouble)))
        h = float(fixed_support(n[None], ell, rhoL, rhoQ, theta)[0])
        seed_margin = float((actual_local_b - np.max(s @ n) - h) / np.linalg.norm(n))
        if seed_margin < -opt.seed_tolerance_m:
            if upper[j] <= opt.certificate_eps_m + world_guard + opt.seed_tolerance_m:
                raise SeedPlaneFailure(
                    dict(
                        schema=SCHEMA,
                        message="a source fragment cannot retain the required envelope seed",
                        selected_piece=j,
                        distance_upper_m=float(upper[j]),
                        minimum_normal_margin_m=seed_margin,
                        covered_piece_count=int(np.sum(~active)),
                        source_piece_count=count,
                        physical_task_infeasibility_proven=False,
                        timing=dict(report_statistics_seconds=statistics_seconds
                                    + (0.0 if oracle is None else oracle.report_statistics_seconds)),
                    )
                )
            raise SeedPlaneSolverError(
                "PIECE_SOCP_CERTIFICATE_UNCERTAIN: selected normal lost positive seed separation"
            )
        tick = perf_counter()
        # One batched projection per new plane; removal requires all vertices
        # of the SAME fragment on its obstacle side. Previously covered pieces
        # retain their earlier certificate owner.
        covered = _covered_fragments(v, offsets, n, active, actual_local_b, opt.certificate_eps_m, guard)
        vertex_projections += len(v)
        if not covered[j]:
            raise SeedPlaneSolverError("PIECE_COVERAGE_NUMERICALLY_UNCERTAIN: target fragment not certified")
        face = len(emitted_n)
        owners[covered] = face
        active[covered] = False
        emitted_n.append(n)
        emitted_b.append(b_world)
        coverage_seconds += perf_counter() - tick
        statistics_started = perf_counter()
        selection.append(
            dict(
                face=face,
                selected_piece=j,
                input_source_triangle=int(pieces.source_ids[j]),
                new_covered_pieces=int(covered.sum()),
                distance_lower_m=float(lower[j]),
                distance_upper_m=float(upper[j]),
                nearest_gap_bound_m=nearest_gap,
                minimum_seed_margin_m=seed_margin,
                selected_solver=("full_socp" if full_done[j] else "payload_ball_certified"),
                normal_reused_from_distance=True,
            )
        )
        statistics_seconds += perf_counter() - statistics_started
        if len(emitted_n) > count:
            raise AssertionError("each round must remove a previously uncovered piece")
    A, b = np.array(emitted_n), np.array(emitted_b)
    tick = perf_counter()
    certificate = verify_piece_coverage(pieces, A, b, owners)
    if not certificate["passed"]:
        raise SeedPlaneSolverError("PIECE_FINAL_COVERAGE_FAILED")
    certify_seconds = perf_counter() - tick
    statistics_started = perf_counter()
    cache_fingerprint = hashlib.sha256(v.tobytes() + offsets.tobytes() + s.tobytes()).hexdigest()
    elapsed = perf_counter() - begin
    proof = dict(
        schema=SCHEMA,
        method="whole_fragment_normal_margin",
        plane_selection_mode="normal_margin",
        planning_mode="fixed_system",
        envelope_approximation="exact",
        physical_model="NO_AERODYNAMIC_TAUT_CABLE",
        source_piece_count=count,
        source_triangle_count=pieces.input_triangle_count,
        source_vertex_count=len(v),
        source_filter="none: complete per-fragment vertices",
        source_geometry="complete clipped affine fragments and their downward prisms",
        source_piece_offsets=offsets.tolist(),
        source_piece_triangle_ids=pieces.source_ids.tolist(),
        piece_coverage_owner=owners.tolist(),
        coverage_certificate=certificate,
        original_points_certified=True,
        coverage_scope="one entire fragment per witness plane",
        terrain_face_count=len(A),
        selected_faces=selection,
        minimum_normal_margin_m=min(x["minimum_seed_margin_m"] for x in selection),
        planning_minimum_margin_m=min(x["minimum_seed_margin_m"] for x in selection),
        payload_geometry_queries=count,
        payload_geometry_iterations=int(geom[:, 5].sum()),
        payload_fast_eligible_count=int(fast_eligible.sum()),
        payload_fast_selected_count=sum(x["selected_solver"] == "payload_ball_certified" for x in selection),
        full_socp_piece_count=full_solves,
        solver_calls=0 if oracle is None else oracle.calls,
        solver_setups=0 if oracle is None else oracle.setups,
        solver_updates=0 if oracle is None else oracle.updates,
        optimization_records=[] if oracle is None else oracle.records,
        selected_normal_resolves=0,
        unneeded_full_socp_skips=count - full_solves,
        optimality_status="GEOMETRIC_DISTANCE_BRACKET_CERTIFIED",
        nearest_order="numerically nearest within recorded bracket tolerance",
        vertex_projection_count=vertex_projections,
        geometry_cache_scope="one unchanged seed window",
        geometry_cache_fingerprint=cache_fingerprint,
        options=asdict(opt),
        piece_options=asdict(popt),
        timing=dict(
            payload_distance_seconds=payload_seconds,
            nearest_selection_seconds=nearest_seconds,
            assembly_seconds=0.0 if oracle is None else oracle.assembly_seconds,
            setup_update_seconds=0.0 if oracle is None else oracle.setup_update_seconds,
            solver_seconds=0.0 if oracle is None else oracle.solve_seconds,
            coverage_seconds=coverage_seconds,
            certification_seconds=certify_seconds,
            core_method_seconds=elapsed - certify_seconds,
            core_method_scope="whole segment: distance brackets, nearest selection, all local solves, incremental whole-fragment coverage",
            total_seconds=elapsed,
        ),
    )
    statistics_seconds += perf_counter() - statistics_started
    if oracle is not None:
        statistics_seconds += oracle.report_statistics_seconds
    proof['timing']['report_statistics_seconds'] = statistics_seconds
    proof['timing']['total_seconds'] = perf_counter() - begin
    proof['timing']['core_method_seconds'] = proof['timing']['total_seconds'] - certify_seconds - statistics_seconds
    return A, b, proof


def _separate_pieces_native(pieces, seeds, ell, limits, *, plane_options=None, piece_options=None, checkpoint=None):
    """C++ segment flow; Python retains validation, the unchanged Clarabel calls and proof I/O."""
    begin = perf_counter()
    pieces.validate()
    opt = plane_options if isinstance(plane_options, PlaneOptions) else PlaneOptions(**(plane_options or {}))
    popt = piece_options if isinstance(piece_options, PieceOptions) else PieceOptions(**(piece_options or {}))
    if opt.plane_selection_mode != "normal_margin" or opt.envelope_approximation != "exact":
        raise ValueError("piecewise_socp requires normal_margin and the exact fixed-system envelope")
    s_world = np.asarray(seeds, float)
    if s_world.shape != (2, 3) or not np.isfinite(s_world).all():
        raise ValueError("Expected two finite seed endpoints")
    rhoL, rhoQ = effective_radii(limits)
    theta = float(limits["theta"])
    if not np.isfinite([ell, rhoL, rhoQ, theta]).all() or min(ell, rhoL, rhoQ) <= 0.0 or not 0.0 <= theta < np.pi / 2:
        raise ValueError("Invalid fixed-system envelope")
    oracle = None

    def solve_full(j, vertices, local_seeds, bracket_tolerance):
        nonlocal oracle
        if oracle is None:
            oracle = FullPlaneOracle(local_seeds, ell, limits, opt, popt.reuse_solver)
            oracle._native_core = scpsc_module()
        n, lb, ub, record = oracle.solve(vertices, bracket_tolerance)
        return np.ascontiguousarray(n, dtype=np.float64), lb, ub

    try:
        result = scpsc_module().separate(
            np.ascontiguousarray(pieces.vertices, dtype=np.float64),
            np.ascontiguousarray(pieces.offsets, dtype=np.int64),
            np.ascontiguousarray(pieces.source_ids, dtype=np.int64),
            np.ascontiguousarray(s_world),
            np.array([ell, rhoL, rhoQ, theta, opt.certificate_eps_m, opt.seed_tolerance_m,
                      opt.tol, popt.nearest_tolerance_m], dtype=np.float64),
            int(popt.payload_fast_path), int(popt.lazy_nearest), popt.geometry_max_iter,
            solve_full, checkpoint,
        )
    except SeedPlaneFailure as error:
        timing = error.proof.setdefault('timing', {})
        timing['report_statistics_seconds'] = (timing.get('report_statistics_seconds', 0.)
            + (0.0 if oracle is None else oracle.report_statistics_seconds))
        raise
    A, b, owners = result["A"], result["b"], result["owners"]
    statistics_started = perf_counter()
    v, s, offsets, count = result["local_vertices"], result["local_seeds"], pieces.offsets, len(pieces)
    selection = []
    for face, (indices, values) in enumerate(zip(result["face_ids"], result["face_values"])):
        j, source_id, covered, full_done = indices
        lb, ub, gap, margin = values
        selection.append(dict(
            face=face, selected_piece=int(j), input_source_triangle=int(source_id),
            new_covered_pieces=int(covered), distance_lower_m=float(lb), distance_upper_m=float(ub),
            nearest_gap_bound_m=float(gap), minimum_seed_margin_m=float(margin),
            selected_solver="full_socp" if full_done else "payload_ball_certified",
            normal_reused_from_distance=True,
        ))
    certificate = dict(
        passed=True, minimum_source_separation_m=result["coverage_minimum"],
        covered_piece_count=count, source_vertex_count=len(v),
        rule="each entire fragment belongs to one same separating plane",
    )
    full_solves, vertex_projections = result["full_solves"], result["vertex_projections"]
    payload_seconds, nearest_seconds = result["payload_seconds"], result["nearest_seconds"]
    coverage_seconds, certify_seconds = result["coverage_seconds"], result["certify_seconds"]
    fingerprint = hashlib.sha256()
    for array in (v, offsets, s):
        fingerprint.update(np.ascontiguousarray(array))
    cache_fingerprint = fingerprint.hexdigest()
    elapsed = perf_counter() - begin
    proof = dict(
        schema=SCHEMA,
        method="whole_fragment_normal_margin",
        plane_selection_mode="normal_margin",
        planning_mode="fixed_system",
        envelope_approximation="exact",
        physical_model="NO_AERODYNAMIC_TAUT_CABLE",
        source_piece_count=count,
        source_triangle_count=pieces.input_triangle_count,
        source_vertex_count=len(v),
        source_filter="none: complete per-fragment vertices",
        source_geometry="complete clipped affine fragments and their downward prisms",
        source_piece_offsets=offsets.tolist(),
        source_piece_triangle_ids=pieces.source_ids.tolist(),
        piece_coverage_owner=owners.tolist(),
        coverage_certificate=certificate,
        original_points_certified=True,
        coverage_scope="one entire fragment per witness plane",
        terrain_face_count=len(A),
        selected_faces=selection,
        minimum_normal_margin_m=min(x["minimum_seed_margin_m"] for x in selection),
        planning_minimum_margin_m=min(x["minimum_seed_margin_m"] for x in selection),
        payload_geometry_queries=count,
        payload_geometry_iterations=int(result["iterations"]),
        payload_fast_eligible_count=int(result["fast_count"]),
        payload_fast_selected_count=sum(x["selected_solver"] == "payload_ball_certified" for x in selection),
        full_socp_piece_count=full_solves,
        solver_calls=0 if oracle is None else oracle.calls,
        solver_setups=0 if oracle is None else oracle.setups,
        solver_updates=0 if oracle is None else oracle.updates,
        optimization_records=[] if oracle is None else oracle.records,
        selected_normal_resolves=0,
        unneeded_full_socp_skips=count - full_solves,
        optimality_status="GEOMETRIC_DISTANCE_BRACKET_CERTIFIED",
        nearest_order="numerically nearest within recorded bracket tolerance",
        vertex_projection_count=vertex_projections,
        geometry_cache_scope="one unchanged seed window",
        geometry_cache_fingerprint=cache_fingerprint,
        options=asdict(opt),
        piece_options=asdict(popt),
        timing=dict(
            payload_distance_seconds=payload_seconds,
            nearest_selection_seconds=nearest_seconds,
            assembly_seconds=0.0 if oracle is None else oracle.assembly_seconds,
            setup_update_seconds=0.0 if oracle is None else oracle.setup_update_seconds,
            solver_seconds=0.0 if oracle is None else oracle.solve_seconds,
            coverage_seconds=coverage_seconds,
            certification_seconds=certify_seconds,
            core_method_seconds=elapsed - certify_seconds,
            core_method_scope="whole segment: distance brackets, nearest selection, all local solves, incremental whole-fragment coverage",
            total_seconds=elapsed,
        ),
    )
    statistics_seconds = result['statistics_seconds'] + perf_counter() - statistics_started
    if oracle is not None:
        statistics_seconds += oracle.report_statistics_seconds
    proof['timing']['report_statistics_seconds'] = statistics_seconds
    proof['timing']['total_seconds'] = perf_counter() - begin
    proof['timing']['core_method_seconds'] = proof['timing']['total_seconds'] - certify_seconds - statistics_seconds
    return A, b, proof


def separate_pieces(pieces, seeds, ell, limits, *, plane_options=None, piece_options=None, checkpoint=None):
    implementation = _separate_pieces_native if scpsc_native_enabled() else _separate_pieces_python
    return implementation(pieces, seeds, ell, limits, plane_options=plane_options,
                          piece_options=piece_options, checkpoint=checkpoint)


def terrain_seed_polyhedron(
    terrain, lo, hi, seeds, ell, limits, eta=0.0, *, plane_options=None, piece_options=None, checkpoint=None
):
    begin = perf_counter()
    pieces = clipped_source_pieces(terrain, lo, hi, eta)
    query = perf_counter() - begin
    A, b, proof = separate_pieces(
        pieces, seeds, ell, limits, plane_options=plane_options, piece_options=piece_options, checkpoint=checkpoint
    )
    statistics_started = perf_counter()
    proof["source_eta_m"] = float(eta)
    proof["timing"]["source_query_clip_seconds"] = query
    proof["timing"]["total_seconds"] += query
    proof['timing']['report_statistics_seconds'] += perf_counter() - statistics_started
    return A, b, proof
