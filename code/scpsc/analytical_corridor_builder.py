"""Configure SCPSC for one task and its window trials."""
from pathlib import Path
import time
import warnings

from .corridor_dependencies import SCPSCDependencyError, dependency_status
from .envelope import planning_envelope_mode
from .geometry import CorridorIncompatible
from .qp_corridor import build_fixed_envelope_corridors


def make_analytical_builder(config, output, constructor=build_fixed_envelope_corridors):
    settings = config["ubsqp"]
    method = settings.get("corridor_method", "piecewise_socp")
    if method != "piecewise_socp":
        raise ValueError("SCPSC_ONLY: corridor_method must be piecewise_socp")
    if planning_envelope_mode(settings) != "fixed_system":
        raise ValueError("SCPSC requires the fixed_system planning envelope")
    opts = settings.get("corridor_options", {})
    dep = dependency_status(method)
    if not dep["available"]:
        raise SCPSCDependencyError("SCPSC_NATIVE_DEPENDENCY_UNAVAILABLE: " + dep["reason"])
    if opts.get("route_wall_limit_s") is not None:
        warnings.warn("corridor wall-clock settings are deprecated and ignored", FutureWarning)
    backoff = float(settings.get("corridor_backoff_m", 0.0))
    trialroot = Path(output) / "corridor_trials"

    def builder(
        terrain, points, domain, ell, limits, center_margin_m=8.0, eta=0.0,
        seed_tolerance_m=1e-7, checkpoint=None, use_system_envelope=True,
        planning_envelope=None,
    ):
        if not use_system_envelope:
            raise ValueError("SCPSC requires the fixed_system planning envelope")
        planning_mode = planning_envelope or ("fixed_system" if use_system_envelope else "point")
        if planning_mode != "fixed_system":
            raise ValueError("SCPSC requires the fixed_system planning envelope")

        def ck(stage):
            if checkpoint is not None:
                checkpoint(stage)

        ck("corridor")
        trial = trialroot / ("m" + format(center_margin_m, "g"))
        io_started = time.perf_counter()
        trial.mkdir(parents=True, exist_ok=False)
        output_io_seconds = time.perf_counter() - io_started
        original_started = time.perf_counter()
        try:
            original, original_report = constructor(
                terrain,
                points,
                domain,
                ell,
                limits,
                center_margin_m=center_margin_m,
                eta=eta,
                seed_tolerance_m=seed_tolerance_m,
                checkpoint=ck,
                use_system_envelope=use_system_envelope,
                plane_options=opts.get("plane_solver", {}),
                planning_envelope=planning_mode,
                piece_options=opts.get("piece_solver", {}),
            )
        except CorridorIncompatible as error:
            error.diagnostics["report_statistics_seconds"] = (
                float(error.diagnostics.get("report_statistics_seconds", 0.0)) + output_io_seconds
            )
            raise
        original_construction_seconds = time.perf_counter() - original_started
        if backoff != 0.0:
            raise ValueError("SCPSC uses certificate_eps_m, require corridor_backoff_m=0")
        statistics_started = time.perf_counter()
        original_report.update(
            method=method,
            independent_source_solids_verified=False,
            source_verification="whole-fragment coverage witnesses + downward prisms + exact support",
            aggregate_route_wall_limit_s=None,
            seconds=original_construction_seconds,
            original_construction_seconds=original_construction_seconds,
            output_io_seconds=output_io_seconds,
        )
        original_report["report_statistics_seconds"] = (
            float(original_report.get("report_statistics_seconds", 0.0)) + output_io_seconds
            + time.perf_counter() - statistics_started
        )
        ck("corridor")
        return original, original_report


    # The uniform builder API includes close(); analytical builds own no handle.
    def close():
        pass

    builder.close = close
    return builder
