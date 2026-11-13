"""Orchestrate one SCPSC task; algorithms and acceptance live in their modules."""
import time
from pathlib import Path

from .final_acceptance import _postplanning_checks
from .run_state import Budget, _initial_record, _record_failure, _finish_record

def _check_dependencies(config, stop_after, record):
    if stop_after == "complete":
        try:
            import osqp
        except ImportError as e:
            raise RuntimeError("OSQP_DEPENDENCY_UNAVAILABLE") from e
        if osqp.__version__ != "1.1.3":
            raise RuntimeError("OSQP_VERSION_MISMATCH: production requires 1.1.3")
        record["osqp_version"] = osqp.__version__
    try:
        import clarabel
    except ImportError as e:
        raise RuntimeError("CLARABEL_DEPENDENCY_UNAVAILABLE") from e
    if clarabel.__version__ != "0.11.1":
        raise RuntimeError("CLARABEL_VERSION_MISMATCH: use 0.11.1")
    record["clarabel_version"] = clarabel.__version__

def worker(config_path, output, replay=None, stop_after="complete", retime_route_from=None, swing_mode="required"):
    """Time construction separately; finish planning timing after final acceptance."""
    from datetime import datetime, timezone

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    from scpsc.configuration import load_config
    from scpsc.scene_inputs import load_scene
    from scpsc.result_io import save_json
    from scpsc.reference_preparation import make_endpoint
    from scpsc.qp_adapter import reference_and_corridors
    from scpsc.trajectory_acceptance import accept_same_curve
    from scpsc.qp_backend import solve_fixed_time_qp
    from scpsc.dynamics import DynamicsParams

    record = _initial_record(stop_after, swing_mode)
    budget = None
    started = time.perf_counter()
    statistics_seconds = 0.0
    planning_wall = None
    try:
        if swing_mode not in ("off", "report", "required"):
            raise ValueError("swing_mode must be off, report or required")
        config = load_config(config_path)
        save_json(output / "effective_config.json", config)
        budget = Budget(None, output)
        _check_dependencies(config, stop_after, record)
        budget("preparation")
        env = load_scene(config)
        params = DynamicsParams(**config["dynamics"])
        endpoint, eq = make_endpoint(env, params, config)
        save_json(output / "endpoint_reference.json", eq)
        ref = reference_and_corridors(
            env, params, config, output, budget, replay_from=replay, retime_route_from=retime_route_from
        )
        statistics_seconds = float(ref["corridor_statistics_seconds"])
        corridor_verification_seconds = float(ref.get("corridor_verification_seconds", 0.0))
        corridor_construction_seconds = ref["corridor_seconds"] - statistics_seconds - corridor_verification_seconds
        corridor_representation_seconds = float(ref["corridor_representation_seconds"])
        record.update(
            corridor_pass=True,
            frontend_seconds=ref["frontend_seconds"],
            corridor_seconds_including_statistics=ref["corridor_seconds"],
            corridor_statistics_seconds=statistics_seconds,
            corridor_verification_seconds=corridor_verification_seconds,
            corridor_output_io_seconds=float(ref.get("corridor_output_io_seconds", 0.0)),
            corridor_seconds=corridor_construction_seconds,
            corridor_online_seconds=corridor_construction_seconds,
            corridor_representation_seconds=corridor_representation_seconds,
            corridor_ready_seconds=corridor_construction_seconds - corridor_representation_seconds,
            flight_time_s=float(ref["times"][-1]),
            corridor_status=ref["corridor_status"],
        )
        if stop_after == "corridor":
            record.update(status="CORRIDOR_ONLY_PASS", corridor_pass=True, transport_evaluated=False)
            return record
        budget("qp")
        record["qp_calls"] = 1
        qp = solve_fixed_time_qp(
            ref["times"],
            ref["points"],
            ref["corridors"],
            eq,
            params,
            config["limits"],
            config["ubsqp"],
            approach=None,
            output_dir=output / "qp",
        )
        record["qp_status"] = qp["status"]
        record["qp_seconds"] = qp["diagnostics"]["total_seconds"]
        record["qp_report_statistics_seconds"] = float(qp["diagnostics"].get("report_statistics_seconds") or 0.0)
        statistics_seconds += record["qp_report_statistics_seconds"]
        if not qp["numerical_pass"]:
            record.update(status=qp["status"], failure_reason="QP did not pass its original constraints")
            return record
        budget("trajectory_check")
        trajectory, trajectory_check = accept_same_curve(qp, ref, endpoint, env, params, config, output)
        record["trajectory_status"] = trajectory_check["status"]
        record["trajectory_check_seconds"] = trajectory_check["seconds"]
        record["planning_candidate_pass"] = bool(trajectory_check["passed"])
        record["specific_force_status"] = trajectory_check["specific_force_status"]
        if not trajectory_check["passed"]:
            record["status"] = trajectory_check["status"]
            return record
        # A candidate is saved, but planning is not complete until final acceptance.
        record.update(
            planning_seconds_including_statistics=None,
            online_report_statistics_seconds=statistics_seconds,
            planning_seconds=None,
            status="PLANNING_PENDING_FINAL_ACCEPTANCE",
        )
        save_json(output / "run_record.json", record)
        _postplanning_checks(
            trajectory, qp, ref, params, config, output, trajectory_check, record, budget, swing_mode
        )
        planning_wall = time.perf_counter() - started
        statistics_seconds += record["output_statistics_seconds"]
        record.update(
            planning_seconds_including_statistics=planning_wall,
            online_report_statistics_seconds=statistics_seconds,
            planning_seconds=planning_wall - statistics_seconds,
            planning_finished_utc=datetime.now(timezone.utc).isoformat(),
        )
    except (Exception, KeyboardInterrupt) as error:
        _record_failure(error, budget, record, output, swing_mode)
    finally:
        _finish_record(started, planning_wall, statistics_seconds, budget, record, output)
    return record
