"""Connect the frozen reference to corridor construction for the QP."""
import time
from pathlib import Path

from .envelope import planning_envelope_mode
from .geometry import CorridorIncompatible
from .analytical_corridor_builder import make_analytical_builder
from .qp_corridor_policy import build_with_window_policy
from .reference_preparation import prepare_reference
from .result_io import save_json



def reference_and_corridors(
    env,
    params,
    config,
    output,
    checkpoint,
    replay_from=None,
    frontend_candidate=None,
    register_reference_contract=True,
    retime_route_from=None,
):
    """Keep the complete original polyline; only its reference clock is changed."""
    output = Path(output)
    settings = config["ubsqp"]
    route, times, approach, front_seconds, timing_changed = prepare_reference(
        env, params, config, output, checkpoint, replay_from, frontend_candidate,
        register_reference_contract, retime_route_from,
    )
    checkpoint("corridor")
    # Separate load-center freedom from the fixed system's footprint.  Returned
    # corridors remain RAW workspace; the unchanged QP tightens them once.
    builder = make_analytical_builder(config, output)
    begin = time.perf_counter()
    # Corridor-level result files are timed and deducted with report statistics
    # for SCPSC. Verification has its own timer and remains mandatory.
    output_io = [0.0]

    def emit(state):
        io_started = time.perf_counter()
        save_json(output / "corridor_window_attempts.json", state)
        output_io[0] += time.perf_counter() - io_started

    try:
        try:
            corridors, corridor_report = build_with_window_policy(
                env,
                route,
                params,
                config,
                checkpoint,
                builder=builder,
                emit=emit,
            )
        except TimeoutError as error:
            save_json(
                output / "corridor_adapter.json",
                dict(
                    getattr(error, "window_policy_report", {}),
                    status="CORRIDOR_EXTERNAL_TIMEOUT",
                    failure=str(error),
                    qp_calls=0,
                ),
            )
            raise
        except CorridorIncompatible as error:
            policy_report = getattr(error, "window_policy_report", {})
            save_json(
                output / "corridor_adapter.json",
                dict(
                    policy_report,
                    status=(
                        dict(fixed_system="FIXED", load_only="LOAD_ONLY", point="POINT", bare_point="BARE_POINT")[
                            planning_envelope_mode(settings)
                        ]
                        + "_CORRIDOR_ADMISSION_FAILED"
                    ),
                    failure=error.to_dict(),
                    requested_center_margin_m=policy_report.get("initial_center_margin_m"),
                    route_changed=False,
                    timing_changed=timing_changed,
                    qp_calls=0,
                    forbidden_recovery_calls=0,
                ),
            )
            raise
        statistics_started = time.perf_counter()
        corridor_report.update(
            inherited_corridor_width_m=config["optimization"]["corridor_width"],
            geometry_source=(
                "source triangles and independent operation domain; "
                + planning_envelope_mode(settings)
                + " support applied once by QP; true-system certificate checked separately"
            ),
            timing_changed=timing_changed,
            forbidden_recovery_calls=0,
        )
        source = env["source"]
        corridor_report.update(
            source_geometry_version=source.source_geometry_version, source_geometry_id=source.source_geometry_id
        )
        for corridor in corridors:
            corridor.proof.update(
                source_geometry_version=source.source_geometry_version, source_geometry_id=source.source_geometry_id
            )
        adapter_statistics_seconds = time.perf_counter() - statistics_started
        io_started = time.perf_counter()
        save_json(output / "corridor_adapter.json", corridor_report)
        save_json(
            output / "corridors.json",
            [
                dict(
                    normals=c.normals,
                    bounds=c.bounds,
                    proof=c.proof,
                    original_normals=c.original_normals,
                    original_bounds=c.original_bounds,
                )
                for c in corridors
            ],
        )
        output_io[0] += time.perf_counter() - io_started
        corridor_seconds = time.perf_counter() - begin
        # report_statistics_seconds already includes the builder's per-segment
        # file I/O of every window attempt; add the corridor-level files here.
        corridor_statistics_seconds = (float(corridor_report.get("report_statistics_seconds", 0.0))
            + adapter_statistics_seconds + output_io[0])
        return dict(
            times=times,
            points=route,
            corridors=corridors,
            approach=approach,
            corridor_status=corridor_report["status"],
            frontend_seconds=front_seconds,
            corridor_seconds=corridor_seconds,
            corridor_statistics_seconds=corridor_statistics_seconds,
            corridor_verification_seconds=float(corridor_report.get("verification_seconds", 0.0)),
            corridor_representation_seconds=float(corridor_report.get("representation_seconds", 0.0)),
            corridor_output_io_seconds=float(corridor_report.get("output_io_seconds", 0.0)) + output_io[0],
            frontend_reused=replay_from is not None or retime_route_from is not None,
        )
    finally:
        if hasattr(builder, "close"):
            builder.close()
