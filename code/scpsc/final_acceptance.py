"""Common independent trajectory and swing acceptance after a candidate."""
import time

def _postplanning_checks(
    trajectory, qp, ref, params, config, output, trajectory_check, record, budget, swing_mode
):
    from scpsc.result_io import save_json
    from scpsc.trajectory_acceptance import write_trajectory_diagnostics
    from scpsc.independent_verify import verify_saved
    from scpsc.envelope import planning_envelope_mode

    budget("output_statistics")
    phase_start = time.perf_counter()
    write_trajectory_diagnostics(trajectory, qp, ref, params, config, output, trajectory_check)
    record["output_statistics_seconds"] = time.perf_counter() - phase_start
    budget("independent_check")
    phase_start = time.perf_counter()
    verify = verify_saved(output, config)
    save_json(output / "independent_check.json", verify)
    record["independent_status"] = verify["status"]
    record["independent_seconds"] = verify["seconds"]
    record["independent_stage_seconds"] = time.perf_counter() - phase_start
    record["success"] = bool(trajectory_check["passed"] and verify["passed"])
    if record["success"]:
        record["status"] = (
            "POINT_TRAJECTORY_PASS" if planning_envelope_mode(config["ubsqp"]) == "bare_point" else "TRAJECTORY_PASS"
        )
    elif not trajectory_check["passed"]:
        record["status"] = trajectory_check["status"]
    else:
        record["status"] = "INDEPENDENT_CHECK_FAILED"
    record["transport_evaluated"] = False
    record["acceptance_scope"] = "trajectory-layer geometry, endpoints, load speed and continuous specific force"
    record.update(length_m=verify["length_m"], snap_integral=verify["snap_integral"])
    record["legacy_trajectory_pass"] = record["success"]
    record["legacy_status"] = record["status"]
    if swing_mode != "off":
        from scpsc.swing_validation import validate_saved_swing

        budget("swing_validation")
        phase_start = time.perf_counter()
        record["_swing_started"] = phase_start
        swing = validate_saved_swing(output, output / "swing", config.get("swing_validation"))
        record["swing_seconds"] = time.perf_counter() - phase_start
        record.pop("_swing_started")
        record["swing_forward_pass"] = bool(swing["passed"])
        record["swing_status"] = swing["status"]
    from scpsc.run_acceptance import apply_swing_acceptance

    apply_swing_acceptance(record)
    budget("complete")
