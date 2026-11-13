"""The per-window SOCP oracle and its geometric distance witnesses."""
from time import perf_counter

import numpy as np
from scipy import sparse

from .envelope import effective_radii, fixed_support
from .plane_contract import SeedPlaneSolverError
from .piecewise_geometry import _guard


def _normal(n):
    n = np.asarray(n, dtype=float).copy()
    if n.shape != (3,) or not np.isfinite(n).all():
        raise SeedPlaneSolverError("PIECE_SOCP: invalid normal")
    # Projection handles a positive z caused by a numerical inequality residual.
    # All subsequent support bounds and geometry certificates use the projection.
    n[2] = min(0.0, n[2])
    length = np.linalg.norm(n)
    if length <= 1e-12:
        raise SeedPlaneSolverError("PIECE_SOCP: zero normal")
    return n / length


def _cap_point(n, theta):
    q = np.array([np.sin(theta), 0.0, np.cos(theta)])
    length = np.linalg.norm(n[:2])
    if length > 0.0:
        q[:2] = np.sin(theta) * n[:2] / length
    return q


class FullPlaneOracle:
    """One seven-variable, two-cone SOCP workspace per seed window.

    Seven source rows are retained, padding smaller fragments with an existing
    vertex.  Explicit structural zero entries keep the CSC pattern invariant.
    update() reuses allocation/structure; this is NOT a warm-start claim.
    """

    def __init__(self, seeds, ell, limits, plane_options, reuse=True):
        self.seeds = np.asarray(seeds, float)
        self.ell = float(ell)
        self.rhoL, self.rhoQ = effective_radii(limits)
        self.theta = float(limits["theta"])
        self.options = plane_options
        self.reuse = reuse
        self.solver = None
        self.solver_tolerance = None
        self.calls = self.setups = self.updates = 0
        self.assembly_seconds = self.setup_update_seconds = self.solve_seconds = 0.0
        self.report_statistics_seconds = 0.0
        self.records = []
        self._initialise_matrix()

    def _initialise_matrix(self):
        start = perf_counter()
        d = np.zeros((21, 7))  # Eq. (11), columns (n_x, n_y, n_z, b, zeta, u, r).
        d[:7, 3] = 1.0
        for i, s in enumerate(self.seeds):
            d[7 + 2 * i, :] = [*s, -1.0, 1.0, self.rhoL, 0.0]
            p = s.copy()
            p[2] += self.ell * np.cos(self.theta)
            d[8 + 2 * i, :] = [*p, -1.0, 1.0, self.rhoQ, self.ell * np.sin(self.theta)]
        d[11, 5] = 1.0
        d[12, 2] = 1.0
        d[13, 4] = -1.0
        d[14, 5] = -1.0
        d[15, 0] = -1.0
        d[16, 1] = -1.0
        d[17, 2] = -1.0
        d[18, 6] = -1.0
        d[19, 0] = -1.0
        d[20, 1] = -1.0
        mask = d != 0.0
        mask[:7, :4] = True
        mask[7:11, :3] = True
        self.dense = d
        self.A = sparse.csc_matrix(mask.astype(float))
        self.rows = self.A.indices.copy()
        self.cols = np.repeat(np.arange(7), np.diff(self.A.indptr))
        self.A.data[:] = d[self.rows, self.cols]
        self.P = sparse.csc_matrix((7, 7))
        self.q = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0])  # Eq. (11), maximize zeta.
        self.b = np.zeros(21)
        self.b[11] = 1.0
        self.assembly_seconds += perf_counter() - start

    def _run(self, vertices, tol):
        try:
            import clarabel
        except ImportError as e:
            raise SeedPlaneSolverError("CLARABEL_DEPENDENCY_UNAVAILABLE: install clarabel==0.11.1") from e
        started = perf_counter()
        v = np.asarray(vertices, float)
        vp = v[np.minimum(np.arange(7), len(v) - 1)]
        self.dense[:7, :3] = -vp  # Eq. (11), b <= n·v for every patch vertex.
        self.A.data[:] = self.dense[self.rows, self.cols]
        self.assembly_seconds += perf_counter() - started
        started = perf_counter()
        if not self.reuse or self.solver is None or self.solver_tolerance != tol:
            settings = clarabel.DefaultSettings()
            settings.verbose = False
            settings.max_threads = 1
            settings.max_iter = self.options.max_iter
            settings.time_limit = float("inf")
            settings.tol_gap_abs = tol
            settings.tol_gap_rel = tol
            settings.tol_feas = tol
            settings.presolve_enable = False
            settings.chordal_decomposition_enable = False
            if hasattr(settings, "input_sparse_dropzeros"):
                settings.input_sparse_dropzeros = False
            self.solver = clarabel.DefaultSolver(
                self.P,
                self.q,
                self.A,
                self.b,
                [clarabel.NonnegativeConeT(14), clarabel.SecondOrderConeT(4), clarabel.SecondOrderConeT(3)],
                settings,
            )
            self.solver_tolerance = tol
            self.setups += 1
        else:
            self.solver.update(A=self.A.data)
            self.updates += 1
        self.setup_update_seconds += perf_counter() - started
        started = perf_counter()
        solution = self.solver.solve()
        wall = perf_counter() - started
        self.solve_seconds += wall
        self.calls += 1
        statistics_started = perf_counter()
        rec = dict(
            status=str(solution.status),
            iterations=int(solution.iterations),
            solve_wall_seconds=wall,
            solver_reported_seconds=float(solution.solve_time),
            runtime="clarabel",
            version=clarabel.__version__,
            tolerance=tol,
        )
        self.records.append(rec)
        self.report_statistics_seconds += perf_counter() - statistics_started
        return solution, vp, rec

    def solve(self, vertices, bracket_tolerance):
        """Return a valid normal and geometric [lower, upper] distance bracket.

        Dual multipliers are used only to propose feasible convex witnesses.
        They are clipped and normalised; the reported dual objective is NOT
        treated as a rigorous distance upper bound.
        """
        if getattr(self, "_native_core", None) is not None:
            def raw_run(v, tolerance):
                solution, padded_vertices, record = self._run(v, tolerance)
                return (np.ascontiguousarray(solution.x, dtype=np.float64),
                        np.ascontiguousarray(solution.z, dtype=np.float64), str(solution.status), record)

            first_record = len(self.records)
            try:
                return self._native_core.solve_oracle(
                    np.ascontiguousarray(vertices, dtype=np.float64), np.ascontiguousarray(self.seeds),
                    np.array([self.ell, self.rhoL, self.rhoQ, self.theta, self.options.tol], dtype=np.float64),
                    bracket_tolerance, raw_run,
                )
            finally:
                self.report_statistics_seconds += sum(
                    record.get('native_report_statistics_seconds', 0.) for record in self.records[first_record:])
        v, s = np.asarray(vertices, float), self.seeds
        for attempt in range(2):
            sol, vp, record = self._run(v, self.options.tol * (0.01 if attempt else 1.0))
            if str(sol.status) not in ("Solved", "AlmostSolved", "MaxIterations", "InsufficientProgress"):
                raise SeedPlaneSolverError("PIECE_SOCP_UNRESOLVED: " + str(sol.status))
            x, z = np.asarray(sol.x), np.asarray(sol.z)
            if x.shape != (7,) or z.shape != (21,) or not np.isfinite(x).all() or not np.isfinite(z).all():
                raise SeedPlaneSolverError("PIECE_SOCP_UNRESOLVED: nonfinite solution")
            n = _normal(x[:3])
            h = float(fixed_support(n[None], self.ell, self.rhoL, self.rhoQ, self.theta)[0])
            guard = _guard(v, s, [self.ell, self.rhoL, self.rhoQ])
            gap = float((np.min(v @ n) - np.max(s @ n) - h) / np.linalg.norm(n))
            lower = max(0.0, gap - guard)
            weights = np.maximum(0.0, z[:7])
            weights_sum = weights.sum()
            masses = np.maximum(0.0, z[7:11]).reshape(2, 2)
            masses_sum = masses.sum()
            if weights_sum <= 0.0 or masses_sum <= 0.0:
                raise SeedPlaneSolverError("PIECE_SOCP_UNRESOLVED: missing feasible-witness weights")
            weights /= weights_sum
            masses /= masses_sum
            alpha, beta = masses[:, 0], masses[:, 1]
            k = (alpha + beta) @ s + (alpha.sum() * self.rhoL + beta.sum() * self.rhoQ) * n
            qcap = _cap_point(n, self.theta)
            if np.linalg.norm(n[:2]) <= 1e-8 and z[18] > 0.0:
                # At the pole the cap support is nonunique. The SOC dual
                # proposes a point in the convex hull of the boundary circle;
                # clipping to its disk preserves feasibility independently of
                # the multiplier residuals.
                direction = -z[19:21] / z[18]
                direction /= max(1.0, float(np.linalg.norm(direction)))
                qcap = np.r_[np.sin(self.theta) * direction, np.cos(self.theta)]
            k += beta.sum() * self.ell * qcap
            o = weights @ vp
            o[2] = min(o[2], k[2])  # remains in the downward source prism
            upper = float(np.linalg.norm(o - k) + guard)
            statistics_started = perf_counter()
            record.update(distance_lower_m=lower, distance_upper_m=upper, geometric_primal_dual_gap_m=upper - lower)
            self.report_statistics_seconds += perf_counter() - statistics_started
            if lower <= upper + guard and upper - lower <= bracket_tolerance:
                return n, lower, max(lower, upper), record
        raise SeedPlaneSolverError(
            "PIECE_SOCP_DISTANCE_UNCERTAIN: valid plane but nearest-distance bracket did not close; "
            f"gap={upper - lower:.9g} m"
        )
