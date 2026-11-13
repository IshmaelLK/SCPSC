"""Fixed-envelope corridor contract for the QP adapter only.

The default Corridor rows describe RAW workspace. The normal QP applies fixed
support once. The axis-box control encodes its eroded center bounds for that
same QP subtraction and retains the source plane separately in its proof.
The load_only baseline uses the payload ball in source windows,
plane selection and erosion. Legacy point planning omits erosion but retains
its original full-envelope window. center_margin_m is center freedom.
"""
import numpy as np
from time import perf_counter

from .envelope import effective_radii, fixed_support, original_halfspaces, planning_envelope_parameters, center_halfspaces, fixed_seed_margins
from .geometry import Corridor, CorridorIncompatible, remove_duplicates
from .plane_contract import PlaneOptions, SeedPlaneFailure
from .seed_plane import axis_box_vertices, terrain_seed_plane


_XY_NORMALS = np.array([[1., 0., 0.], [-1., 0., 0.],
                        [0., 1., 0.], [0., -1., 0.]])
_CONTRACT = 'raw_workspace_no_aero_normal_socp_v1'


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)




def build_fixed_envelope_corridors(terrain, points, domain, ell, limits,
                                   center_margin_m=8., eta=0.,
                                   seed_tolerance_m=1e-7, checkpoint=None,
                                   use_system_envelope=True, plane_options=None, deadline=None,
                                   planning_envelope=None, piece_options=None):
    """Re-prove raw terrain corridors sized for center freedom + envelope.

    Each source polyline edge keeps its own corridor and time association.  For
    an XY direction n the requested RAW bound is max(n.p_seed) + margin + h_E(n),
    so erosion by h_E leaves the requested margin.  Physical XY clipping can
    reduce that margin; the result is reported, never padded past source domains.
    Terrain lower planes are re-built on the actual expanded window. The
    load_only baseline uses its load sphere for the window and all planning
    support. Legacy point planning retains the full window and zero support.
    True-system seed margins remain separate diagnostics.
    """
    points, domain = _numpy(points), _numpy(domain)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
            or not np.isfinite(points).all()):
        raise ValueError('fixed corridor needs finite [N>=2,3] load reference points')
    if domain.shape != (2, 3) or not np.isfinite(domain).all() or np.any(domain[0] >= domain[1]):
        raise ValueError('fixed corridor requires ordered finite workspace [2,3] bounds')
    if not np.isfinite(ell) or ell <= 0.:
        raise ValueError('fixed corridor rope length must be positive')
    if (not np.isfinite(center_margin_m) or center_margin_m <= 0.
            or not np.isfinite(seed_tolerance_m) or seed_tolerance_m < 0.
            or not np.isfinite(eta) or eta < 0.):
        raise ValueError('positive center margin, nonnegative map eta and admission tolerance required')
    rhoL, rhoQ = effective_radii(limits)
    if min(rhoL, rhoQ) <= 0. or not np.isfinite([rhoL, rhoQ]).all():
        raise ValueError('fixed corridor needs finite positive effective object radii')
    xy_support = fixed_support(_XY_NORMALS, ell, rhoL, rhoQ, limits['theta'])
    planning_mode = planning_envelope or ('fixed_system' if use_system_envelope else 'point')
    use_system_envelope = planning_mode == 'fixed_system'
    options = plane_options if isinstance(plane_options, PlaneOptions) else PlaneOptions(**(plane_options or {}))
    progressive = piece_options is not None
    if progressive and (planning_mode != 'fixed_system' or options.envelope_approximation != 'exact'
                        or options.plane_selection_mode != 'normal_margin'):
        raise ValueError('piecewise_socp requires fixed_system, exact envelope and normal_margin')
    box_control = options.envelope_approximation == 'axis_box'
    if box_control and planning_mode != 'fixed_system':
        raise ValueError('Axis-box control requires the unchanged true physical envelope')
    box_vertices = axis_box_vertices(ell, limits) if box_control else None
    planning_parameters = planning_envelope_parameters(ell, limits, planning_mode)
    planning_xy_support = fixed_support(_XY_NORMALS, *planning_parameters)
    # Legacy point results retain their old window; load_only uses its own sphere throughout.
    window_support = xy_support if planning_mode == 'point' else planning_xy_support
    minus_support = window_support[[1, 3]]
    plus_support = window_support[[0, 2]]
    admission_prefix = dict(fixed_system='FIXED', load_only='LOAD_ONLY', point='POINT',
                            bare_point='BARE_POINT')[planning_mode]
    physical_lo = np.maximum(domain[0, :2], terrain.bounds[0])
    physical_hi = np.minimum(domain[1, :2], terrain.bounds[1])
    floor, ceiling = float(domain[0,2]), float(domain[1,2])
    if np.any(physical_lo >= physical_hi) or floor >= ceiling:
        raise CorridorIncompatible(0, None, 'FIXED_CORRIDOR_EMPTY_SOURCE_DOMAIN',
                                   dict(verification_seconds=0., report_statistics_seconds=0.))

    corridors = []
    statistics_started = perf_counter()
    report = dict(contract=_CONTRACT, raw_workspace_returned=True,
        use_system_envelope=bool(use_system_envelope), planning_mode=planning_mode,
        fixed_support_applied_by_backend=bool(use_system_envelope),
        source_window_uses_true_envelope=planning_mode in ('fixed_system','point'),
        planning_xy_support_m=planning_xy_support.tolist(),
        effective_radii_m=[float(rhoL), float(rhoQ)],
        fixed_cap_xy_support_m=xy_support.tolist(), requested_center_margin_m=float(center_margin_m),
        seed_tolerance_m=float(seed_tolerance_m), route_changed=False,
        timing_changed=False, fallback_calls=0, segments=[], seams=[],
        report_statistics_seconds=0., verification_seconds=0.)
    if box_control:
        report.update(envelope_approximation='axis_box', raw_workspace_returned=False,
            halfspace_semantics='LOAD_CENTER',
            envelope_box_lower_m=box_vertices.min(axis=0).tolist(),
            envelope_box_upper_m=box_vertices.max(axis=0).tolist())
    if progressive:
        report.update(contract='raw_workspace_whole_fragment_socp_v1', construction='whole_fragment_separation')
    report['report_statistics_seconds'] += perf_counter() - statistics_started
    for segment in range(len(points) - 1):
        if checkpoint is not None:
            checkpoint('corridor')
        seeds = points[segment:segment + 2]
        center_lo = seeds[:, :2].min(axis=0) - center_margin_m
        center_hi = seeds[:, :2].max(axis=0) + center_margin_m
        requested_lo, requested_hi = center_lo - minus_support, center_hi + plus_support
        lo, hi = np.maximum(requested_lo, physical_lo), np.minimum(requested_hi, physical_hi)
        statistics_started = perf_counter()
        proof = dict(contract=_CONTRACT, halfspace_semantics='RAW_WORKSPACE',
            use_system_envelope=bool(use_system_envelope), planning_mode=planning_mode,
            source_window_uses_true_envelope=planning_mode in ('fixed_system','point'),
            fixed_erosion_in_builder=False, clearance_in_effective_radii_once=True,
            requested_center_xy_lower=center_lo.tolist(), requested_center_xy_upper=center_hi.tolist(),
            requested_raw_xy_lower=requested_lo.tolist(), requested_raw_xy_upper=requested_hi.tolist(),
            xy_lower=lo.tolist(), xy_upper=hi.tolist(),
            xy_domain_clipped=bool(np.any(lo != requested_lo) or np.any(hi != requested_hi)),
            requested_center_margin_m=float(center_margin_m),
            physical_domain_unchanged=True, fixed_cap_angle_rad=float(limits['theta']))
        report['report_statistics_seconds'] += perf_counter() - statistics_started
        if np.any(lo >= hi):
            proof['report_statistics_seconds'] = report['report_statistics_seconds']
            proof['verification_seconds'] = report['verification_seconds']
            raise CorridorIncompatible(segment, None, 'FIXED_CORRIDOR_EMPTY_XY_WINDOW', proof)
        try:
            if progressive:
                from .piecewise_corridor import terrain_seed_polyhedron
                terrain_A, terrain_b, terrain_proof = terrain_seed_polyhedron(
                    terrain, lo, hi, seeds, ell, limits, eta, plane_options=options,
                    piece_options=piece_options, checkpoint=checkpoint)
            else:
                slope, beta, terrain_proof = terrain_seed_plane(terrain, lo, hi, seeds, ell, limits, eta,
                    use_system_envelope, options=plane_options, deadline=deadline,
                    checkpoint=checkpoint, planning_envelope=planning_mode)
                terrain_A, terrain_b = np.array([np.r_[slope, -1.]]), np.array([-beta])
        except SeedPlaneFailure as error:
            timing = error.proof.get('timing', {})
            report['report_statistics_seconds'] += float(timing.get('report_statistics_seconds', 0.))
            report['verification_seconds'] += float(timing.get(
                'certification_seconds' if progressive else 'verification_seconds', 0.))
            proof['terrain_envelope'] = error.proof
            proof['admitted_prefix_report'] = report
            proof['report_statistics_seconds'] = report['report_statistics_seconds']
            proof['verification_seconds'] = report['verification_seconds']
            raise CorridorIncompatible(segment, 4, str(error), proof) from error
        except (ValueError, RuntimeError) as error:
            diagnostics = dict(getattr(error, 'diagnostics', {}))
            diagnostics['verification_seconds'] = (report['verification_seconds']
                + float(diagnostics.get('verification_seconds', 0.)))
            diagnostics['report_statistics_seconds'] = (report['report_statistics_seconds']
                + float(diagnostics.get('report_statistics_seconds', 0.)))
            error.diagnostics = diagnostics
            # Unknown source/NoData remains an error, never a smaller safety radius.
            if not isinstance(error, ValueError) or not str(error).startswith('CORRIDOR_INCOMPATIBLE:'):
                raise
            proof.update(diagnostics)
            raise CorridorIncompatible(segment, 4, str(error), proof) from error
        proof['terrain_envelope'] = terrain_proof
        timing = terrain_proof.get('timing', {})
        report['report_statistics_seconds'] += float(timing.get('report_statistics_seconds', 0.))
        report['verification_seconds'] += float(timing.get(
            'certification_seconds' if progressive else 'verification_seconds', 0.))
        # Independent task volume + canopy only. The envelope is eroded once by QP.
        normals = np.vstack((_XY_NORMALS, terrain_A, [0.,0.,1.], [0.,0.,-1.]))
        bounds = np.r_[[hi[0],-lo[0],hi[1],-lo[1]], terrain_b, ceiling, -floor]
        if not progressive:
            lengths = np.linalg.norm(normals, axis=1)
            normals, bounds = normals / lengths[:, None], bounds / lengths
        # New terrain rows are already unit normals. Preserve the exact
        # certified world halfspaces and their fragment-owner witnesses.
        if box_control:
            # Select and erode with B, then encode for the unchanged backend's
            # h_S subtraction: (b_raw - h_B + h_S) - h_S = b_raw - h_B.
            raw_bounds = bounds.copy()
            box_support = np.max(normals @ box_vertices.T, axis=1)
            true_support = fixed_support(normals, ell, rhoL, rhoQ, limits['theta'])
            bounds = bounds - box_support
            proof.update(envelope_approximation='axis_box',
                halfspace_semantics='LOAD_CENTER',
                center_space_source='workspace planes eroded by the fixed-envelope axis box',
                center_envelope_parameters=[float(ell), float(rhoL), float(rhoQ), float(limits['theta'])],
                workspace_bounds_before_box_m=raw_bounds.tolist(),
                box_support_m=box_support.tolist(), true_support_m=true_support.tolist(),
                box_center_bounds_m=(raw_bounds-box_support).tolist(),
                qp_support_preadded=False, envelope_box_applied_in_builder=True,
                encoded_planes_are_workspace_safety_region=False)
        statistics_started = perf_counter()
        terrain_names = ([f'terrain_lower_{j:03d}' for j in range(len(terrain_b))]
                         if progressive else ['terrain_lower'])
        names = ['xy_upper_x', 'xy_lower_x', 'xy_upper_y', 'xy_lower_y'] + terrain_names + [
                 'operation_upper', 'operation_lower']
        if progressive:
            proof.update(contract='raw_workspace_whole_fragment_socp_v1',
                terrain_face_rows=list(range(4, 4+len(terrain_b))),
                lower_representation='whole_fragment_polyhedron')
        report['report_statistics_seconds'] += perf_counter() - statistics_started
        active_n, active_b = remove_duplicates(normals, bounds)
        proof['original_face_labels'] = names
        proof['exact_duplicate_faces_removed'] = len(bounds) - len(active_b)
        corridor = Corridor(active_n, active_b, proof, normals.copy(), bounds.copy())
        verification_started = perf_counter()
        planning_support = fixed_support(normals, *planning_parameters)
        if progressive:
            # SCPSC rows are the original certified raw workspace halfspaces.
            # Use their same erosion for admission and both ends of each seam.
            if not np.isfinite(bounds).all():
                raise ValueError('One finite bound per corridor normal required')
            center_bounds = bounds - planning_support  # Eq. (12), erode each raw half-space once.
        else:
            _, center_bounds = center_halfspaces(corridor, ell, limits, planning_mode)
        lengths = np.linalg.norm(normals, axis=1)
        margin = (center_bounds - seeds @ normals.T) / lengths  # Eqs. (2), (13), endpoint containment of S_i in C_i.
        true_margin = margin if use_system_envelope else fixed_seed_margins(corridor, seeds, ell, limits)
        report['verification_seconds'] += perf_counter() - verification_started
        # Report preparation is separate from the support/margin verification.
        statistics_started = perf_counter()
        proof['planning_support_m'] = planning_support.tolist()
        proof['planning_seed_margins_m'] = margin.tolist()
        proof['planning_seed_minimum_margin_m'] = float(margin.min())
        proof['fixed_seed_margins_m'] = true_margin.tolist()
        proof['fixed_seed_minimum_margin_m'] = float(true_margin.min())
        proof['full_envelope_seed_contained'] = bool(true_margin.min() >= -seed_tolerance_m)
        # Diagnostic center bounds only.  Do NOT pass these to qp_backend, which
        # already subtracts fixed_support from raw bounds.
        proof['eroded_center_xy_lower'] = (lo + xy_support[[1, 3]]).tolist()
        proof['eroded_center_xy_upper'] = (hi - xy_support[[0, 2]]).tolist()
        proof['planning_center_xy_lower'] = (lo + planning_xy_support[[1, 3]]).tolist()
        proof['planning_center_xy_upper'] = (hi - planning_xy_support[[0, 2]]).tolist()
        proof['fixed_seed_scope'] = 'fixed-cap polyline containment in constructed convex corridor; not dynamics'
        proof['planning_seed_scope'] = planning_mode + ' polyline admission; not physical acceptance'
        report['report_statistics_seconds'] += perf_counter() - statistics_started
        verification_started = perf_counter()
        seed_incompatible = float(margin.min()) < -seed_tolerance_m
        report['verification_seconds'] += perf_counter() - verification_started
        if seed_incompatible:
            seed_index, face = np.unravel_index(np.argmin(margin), margin.shape)
            diagnostic = dict(proof, seed_object=planning_mode,
                original_normals=normals.tolist(), original_bounds=bounds.tolist(),
                admitted_prefix_report=report,
                report_statistics_seconds=report['report_statistics_seconds'],
                verification_seconds=report['verification_seconds'],
                seed_position=seeds[seed_index].tolist(), original_face_label=names[face],
                residual_m=float(-margin[seed_index, face]),
                physical_task_infeasibility_proven=False)
            reason = admission_prefix + '_ENVELOPE_SEED_INCOMPATIBLE'
            raise CorridorIncompatible(segment, int(face),
                f'{reason}: {names[face]}, '
                f'residual {-margin[seed_index, face]:.9g} m', diagnostic)
        if corridors:
            verification_started = perf_counter()
            prev_normals, prev_bounds, prev_lengths = previous_center_rows
            # Keep the single-point matmul here: its rounding can differ from
            # the two-point segment projection used for seed admission.
            left = (prev_bounds - points[segment:segment + 1] @ prev_normals.T) / prev_lengths
            seam_margin = float(min(left.min(), margin[0].min()))
            report['verification_seconds'] += perf_counter() - verification_started
            statistics_started = perf_counter()
            seam = dict(route_vertex=segment, shared_load_point=points[segment].tolist(),
                planning_mode=planning_mode,
                minimum_original_face_margin_m=seam_margin,
                inscribed_ball_radius_at_shared_point_m=max(0., seam_margin),
                status=('POSITIVE_SHARED_NEIGHBORHOOD' if seam_margin > seed_tolerance_m
                        else 'TOUCHING_OR_NUMERICAL_TOLERANCE_ONLY'),
                scope=planning_mode + ' eroded corridor intersection; not spline feasibility')
            proof['left_seam'] = seam
            report['seams'].append(seam)
            report['report_statistics_seconds'] += perf_counter() - statistics_started
        previous_center_rows = normals, center_bounds, lengths
        corridors.append(corridor)
        statistics_started = perf_counter()
        report['segments'].append(dict(segment=segment, lower_plane=terrain_proof, xy_domain_clipped=proof['xy_domain_clipped'],
            planning_mode=planning_mode, planning_seed_minimum_margin_m=float(margin.min()),
            planning_seed_margins_m=margin.tolist(),
            fixed_seed_minimum_margin_m=proof['fixed_seed_minimum_margin_m'],
            fixed_seed_margins_m=true_margin.tolist(),
            xy_lower=lo.tolist(), xy_upper=hi.tolist(),
            eroded_center_xy_lower=proof['eroded_center_xy_lower'],
            eroded_center_xy_upper=proof['eroded_center_xy_upper']))
        report['report_statistics_seconds'] += perf_counter() - statistics_started
    report.update(status=admission_prefix + '_CORRIDOR_ADMISSION_PASS',
        corridor_count=len(corridors))
    statistics_started = perf_counter()
    report.update(
        minimum_planning_seed_margin_m=min(c.proof['planning_seed_minimum_margin_m'] for c in corridors),
        minimum_fixed_seed_margin_m=min(c.proof['fixed_seed_minimum_margin_m'] for c in corridors))
    report['report_statistics_seconds'] += perf_counter() - statistics_started
    return corridors, report
