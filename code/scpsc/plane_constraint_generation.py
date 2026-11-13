"""Constraint generation for the unchanged whole-window normal-margin SOCP."""
from time import perf_counter

import numpy as np

from .plane_contract import SeedPlaneSolverError


def solve_generated(vertices, seeds, ell, limits, options, assemble, solve, checkpoint=None):
    """Add violating source vertices; accept only a converged full-source solution.

    A subset optimum bounds the complete problem from above. It solves that
    problem only once the original, unshifted solution satisfies every source
    constraint. Recomputing a safe plane intercept is not this stopping test.
    """
    started = perf_counter()
    active = np.zeros(len(vertices), dtype=bool)
    # Axis extremes and the highest point give a deterministic small start.
    ids = np.unique(np.r_[np.argmin(vertices, axis=0), np.argmax(vertices, axis=0)])
    if len(ids) < 3:
        ids = np.union1d(ids, np.arange(min(3, len(vertices))))
    active[ids] = True
    initial_seconds = perf_counter() - started
    local = vertices - seeds.mean(axis=0)
    records = []
    assembly_seconds = scan_seconds = statistics_seconds = 0.0
    try:
        while True:
            if checkpoint is not None:
                checkpoint('corridor')
            tick = perf_counter()
            data = assemble(vertices[ids], seeds, ell, limits, True,
                            planning_envelope='fixed_system', envelope_approximation='exact')
            assembly_seconds += perf_counter() - tick
            solution, record = solve(data, options)
            statistics_seconds += record.get('report_statistics_seconds', 0.0)
            record['active_vertex_count'] = len(ids)
            records.append(record)
            if record['status'] != 'Solved':
                raise SeedPlaneSolverError(
                    f"CONSTRAINT_GENERATION_UNRESOLVED: {record['status']}; "
                    f'round={len(records)}, active_vertices={len(ids)}')
            x = np.asarray(solution.x, dtype=float)
            if x.shape != (7,) or not np.isfinite(x).all():
                raise SeedPlaneSolverError('CONSTRAINT_GENERATION_UNRESOLVED: nonfinite solution')
            length = float(np.linalg.norm(x[:3]))
            if length <= 1e-10 or x[2] >= -1e-10:
                raise SeedPlaneSolverError('CONSTRAINT_GENERATION_UNRESOLVED: no valid downward normal')
            primal, dual = float(solution.obj_val), float(solution.obj_val_dual)
            gap = abs(primal-dual)
            r_prim, r_dual = float(solution.r_prim), float(solution.r_dual)
            record.update(primal_residual=r_prim, dual_residual=r_dual, objective_gap=gap)
            if (not np.isfinite([primal, dual, r_prim, r_dual]).all()
                    or max(r_prim, r_dual) > options.tol
                    or gap > options.tol*max(1.0, abs(primal), abs(dual))):
                raise SeedPlaneSolverError(
                    'CONSTRAINT_GENERATION_UNRESOLVED: solver residual or objective gap; '
                    f'round={len(records)}, active_vertices={len(ids)}, '
                    f'r_prim={r_prim:.9g}, r_dual={r_dual:.9g}, gap={gap:.9g}, tol={options.tol:.9g}')
            tick = perf_counter()
            residual = (x[3] - local @ x[:3]) / length
            violation = max(0.0, float(residual.max()))
            scan_seconds += perf_counter() - tick
            statistics_started = perf_counter()
            # Retain raw seed/domain/SOC residuals as well as the solver's scaled
            # residuals above; no intercept repair enters any of these quantities.
            slack = data[3] - data[2] @ x
            m = data[4][0]
            record.update(
                raw_source_violation_m=violation,
                raw_non_source_linear_violation=max(0.0, float(-slack[len(ids):m].min())),
                raw_soc_violation=max(0.0, float(np.linalg.norm(slack[m+1:m+4])-slack[m]),
                                      float(np.linalg.norm(slack[m+5:m+7])-slack[m+4])),
            )
            statistics_seconds += perf_counter() - statistics_started
            if violation <= options.seed_tolerance_m:
                statistics_started = perf_counter()
                record['added_vertex_count'] = 0
                info = dict(mode='constraint_generation', input_count=len(vertices),
                            selected_count=len(ids), fallback=None, seconds=initial_seconds)
                proof = dict(rounds=len(records), active_vertex_count=len(ids),
                             raw_source_violation_m=violation, scan_seconds=scan_seconds,
                             status='FULL_SOURCE_CONSTRAINTS_SATISFIED')
                proof['report_statistics_seconds'] = statistics_seconds + perf_counter() - statistics_started
                return data, solution, records, info, assembly_seconds, proof
            tick = perf_counter()
            missing = np.flatnonzero((residual > options.seed_tolerance_m) & ~active)
            if len(missing) == 0:
                raise SeedPlaneSolverError(
                    'CONSTRAINT_GENERATION_UNRESOLVED: active source constraint violated; '
                    f'round={len(records)}, active_vertices={len(ids)}, violation_m={violation:.9g}')
            # At most eight most violated points per round, with index tie-breaking.
            order = np.lexsort((missing, -residual[missing]))
            added = missing[order[:8]]
            active[added] = True
            ids = np.flatnonzero(active)
            record['added_vertex_count'] = len(added)
            scan_seconds += perf_counter() - tick
    except SeedPlaneSolverError as error:
        diagnostics = dict(getattr(error, 'diagnostics', {}))
        diagnostics['report_statistics_seconds'] = statistics_seconds + diagnostics.get('report_statistics_seconds', 0.)
        error.diagnostics = diagnostics
        raise
