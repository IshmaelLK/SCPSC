"""Prepare or replay the frozen route, endpoint state and reference clock."""
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .bottleneck_clock import audit as audit_bottleneck, bottleneck_reference, eased_reference_times
from .dynamics import endpoint_equilibrium
from .lazytheta_frontend import search_lazytheta
from .reference_snapshot import validate_final_reference, write_contract
from .result_io import save_json
from .velocity_constraints import from_config


def make_endpoint(env, params, config):
    head = np.zeros((3, 4))
    tail = head.copy()
    head[:, 0] = env["start"]
    tail[:, 0] = env["goal"]
    eq = endpoint_equilibrium(np.stack((env["start"], env["goal"])), params)
    endpoint = SimpleNamespace(head=head, tail=tail)
    return endpoint, [{k: v[i].detach().numpy() for k, v in eq.items()} for i in range(2)]


def approach_settings(times, points, settings):
    if settings.get("approach_mode") != "none":
        raise ValueError("This profile covers airborne hover-to-hover transport only")
    return None


def _retime_saved_route(retime_route_from, output, env, settings, config):
    # Reuse only the original polyline. A final eased reference must never
    # become input to a second subdivision or easing operation.
    source = Path(retime_route_from).resolve()
    if source == output.resolve():
        raise ValueError("RETIME_OUTPUT_COLLISION")
    with np.load(source / "frontend.npz", allow_pickle=False) as saved:
        for key in ("source_geometry_version", "source_geometry_id"):
            if key not in saved or str(saved[key].item()) != getattr(env["source"], key):
                raise ValueError("RETIME_SOURCE_GEOMETRY_MISMATCH: old route requires explicit source revalidation")
        route = saved["positions"].copy()
    if (
        route.ndim != 2
        or route.shape[1] != 3
        or len(route) < 2
        or not np.isfinite(route).all()
        or np.any(np.linalg.norm(np.diff(route, axis=0), axis=1) <= 1e-11)
    ):
        raise ValueError("RETIME_INVALID_ROUTE: expected a finite nondegenerate XYZ polyline")
    if not np.allclose(
        route[[0, -1]], np.stack((env["start"], env["goal"])), rtol=0.0, atol=config["validation"]["endpoint_tolerance"]
    ):
        raise ValueError("RETIME_ENDPOINT_MISMATCH")
    model = from_config(settings, config["limits"]["vL"])
    front_times = np.r_[0.0, np.cumsum(model.gauge(np.diff(route, axis=0)))]
    front = dict(
        algorithm="SAVED_LAZY_THETA_STAR_GEOMETRY",
        geometry_reused=True,
        source_route_file=str(source / "frontend.npz"),
        search_calls=0,
        R_calls=0,
        route_point_count=len(route),
        velocity_constraint=model.metadata(),
        frontend_times_rule="current velocity gauge on saved geometric edges; before K26 and easing",
    )
    return route, front_times, front


def _new_reference(env, params, config, settings, output, checkpoint, frontend_candidate, begin):
    if frontend_candidate is None:
        route, front_times, front = search_lazytheta(env, params, config, output / "frontend_native", checkpoint)
    else:
        route, front_times, front = frontend_candidate
    np.savez_compressed(
        output / "frontend.npz",
        positions=route,
        times=front_times,
        source_geometry_version=env["source"].source_geometry_version,
        source_geometry_id=env["source"].source_geometry_id,
    )
    save_json(output / "frontend.json", dict(front, status="ROUTE_FOUND"))
    front_seconds = time.perf_counter() - begin
    checkpoint("reference")
    box = from_config(settings, config["limits"]["vL"])
    mode = settings.get("reference_timing_mode", "original")
    if mode == "bottleneck_equal_arclength":
        route, dt, bottle = bottleneck_reference(
            route, box, settings["timing_kappa"], int(settings["bottleneck_sections"])
        )
        save_json(output / "bottleneck_allocation.json", bottle)
        timing_rule = "K*max(section integrated velocity-gauge base times); K=26 fixed equal-arclength sections"
        geometry_rule = "all original bends retained; insert fixed section boundaries and <=4m collinear subdivisions; no lift or refit"
    else:
        raise ValueError("Unknown reference timing mode: " + mode)
    if np.any(dt <= 0):
        raise ValueError("REFERENCE_NONPOSITIVE_DURATION")
    times = np.r_[0.0, np.cumsum(dt)]
    approach = approach_settings(times, route, settings)
    np.savez_compressed(
        output / "reference.npz",
        positions=route,
        times=times,
        original_frontend_times=front_times,
        durations=dt,
        **box.npz_fields(),
    )
    save_json(
        output / "time_and_approach.json",
        dict(
            total_time_s=times[-1],
            timing_kappa=settings["timing_kappa"],
            rule=timing_rule,
            reference_timing_mode=mode,
            geometry=geometry_rule,
            velocity_constraint=box.metadata(),
            approach=approach,
            frozen_before_qp=True,
        ),
    )
    return route, times, approach, front_seconds


def _replay_reference(replay_from, output, env, config, checkpoint, begin):
    replay_from = Path(replay_from).resolve()
    if output.resolve() == replay_from:
        raise ValueError("REPLAY_OUTPUT_COLLISION: source and destination must differ")
    checkpoint("reference")
    saved = validate_final_reference(replay_from, env, config)
    route, times, dt, approach = (saved[key] for key in ("points", "times", "durations", "approach"))
    front_seconds = time.perf_counter() - begin
    for filename in ("frontend.npz", "frontend.json", "reference.npz", "time_and_approach.json"):
        if (output / filename).exists():
            raise FileExistsError("REPLAY_OUTPUT_COLLISION: " + str(output / filename))
    for filename in ("frontend.npz", "frontend.json", "reference.npz", "time_and_approach.json"):
        shutil.copy2(replay_from / filename, output / filename)
    for filename in ("bottleneck_allocation.json", "reference_timing_audit.json", "reference_timing_audit.npz"):
        if (replay_from / filename).exists():
            shutil.copy2(replay_from / filename, output / filename)
    save_json(output / "reference_replay.json", saved["audit"])
    return route, times, approach, front_seconds


def _finalize_reference(
    replay_from, settings, output, route, times, approach, config, env, register_reference_contract
):
    # A saved hard-approach record must not re-enable a disabled landing cone.
    if replay_from is None and settings.get("approach_mode", "hard") == "none":
        approach = None
        timing_file = output / "time_and_approach.json"
        timing = json.loads(timing_file.read_text(encoding="utf-8-sig"))
        timing.update(
            approach=None,
            approach_mode="none",
            approach_registration="Disabled: airborne hover-to-hover transport; existing soft reference retained",
        )
        save_json(timing_file, timing)
    clock_r = settings.get("reference_clock_r")
    timing_changed = replay_from is None and clock_r is not None
    if timing_changed:
        # The approach is registered above against the ORIGINAL clock.  Its
        # interval, direction and limits are retained while corridor switches
        # and soft reference positions use the start/stop-compatible clock.
        original_times = times.copy()
        times = eased_reference_times(original_times, float(clock_r))
        with np.load(output / "reference.npz") as saved:
            reference_arrays = {key: saved[key].copy() for key in saved.files}
        reference_arrays.update(
            times=times,
            durations=np.diff(times),
            original_reference_times=original_times,
            original_reference_durations=np.diff(original_times),
        )
        np.savez_compressed(output / "reference.npz", **reference_arrays)
        timing = json.loads((output / "time_and_approach.json").read_text(encoding="utf-8-sig"))
        timing.update(
            reference_clock="T*inverse_F(original_reference_time/T)",
            reference_clock_r=float(clock_r),
            total_time_changed=False,
            reference_clock_changed=True,
            approach_registration=(
                "Disabled: airborne hover-to-hover; existing soft reference retained"
                if approach is None
                else "Original reference clock before easing; interval and direction unchanged"
            ),
        )
        save_json(output / "time_and_approach.json", timing)
    if replay_from is None and settings.get("reference_timing_mode") == "bottleneck_equal_arclength":
        audit_bottleneck(output, route, times, config)
    if replay_from is None:
        timing_file = output / "time_and_approach.json"
        timing = json.loads(timing_file.read_text(encoding="utf-8-sig"))
        timing["reference_stage"] = "final_eased" if clock_r is not None else "final_uneased"
        save_json(timing_file, timing)
    if register_reference_contract and settings.get("register_reference_contract", True):
        write_contract(output, config, env)
    return times, approach, timing_changed


def prepare_reference(
    env, params, config, output, checkpoint, replay_from,
    frontend_candidate, register_reference_contract, retime_route_from,
):
    """Return the registered route and clock used by both corridors and the QP."""
    begin = time.perf_counter()
    settings = config["ubsqp"]
    checkpoint("frontend")
    if sum(x is not None for x in (replay_from, frontend_candidate, retime_route_from)) > 1:
        raise ValueError("Choose final-reference replay, saved-route retiming, or a current frontend candidate")
    if retime_route_from is not None:
        frontend_candidate = _retime_saved_route(retime_route_from, output, env, settings, config)
    if replay_from is None:
        route, times, approach, front_seconds = _new_reference(
            env, params, config, settings, output, checkpoint, frontend_candidate, begin
        )
    else:
        route, times, approach, front_seconds = _replay_reference(replay_from, output, env, config, checkpoint, begin)
    times, approach, timing_changed = _finalize_reference(
        replay_from, settings, output, route, times, approach, config, env, register_reference_contract
    )
    return route, times, approach, front_seconds, timing_changed
