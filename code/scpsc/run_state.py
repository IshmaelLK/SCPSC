"""Stage timing, run status and failure records; no planning algorithms."""
import json
import os
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

class Budget:
    def __init__(self, seconds, output):
        self.start = time.perf_counter()
        self.deadline = float("inf")
        self.stage = "input"
        self.output = Path(output)
        self.stage_start = self.start
        self.durations = {}
        if seconds is not None:
            warnings.warn("task_budget_s is retired and ignored", FutureWarning)

    def __call__(self, stage):
        if stage != self.stage:
            now = time.perf_counter()
            self.durations[self.stage] = self.durations.get(self.stage, 0.0) + now - self.stage_start
            self.stage = stage
            self.stage_start = now
            write(
                self.output / "stage.json",
                dict(
                    stage=stage,
                    elapsed_seconds=now - self.start,
                    entered_monotonic_s=now,
                    completed_stage_seconds=self.durations,
                ),
            )

def _cpu_affinity():
    if os.name != "nt":
        return sorted(os.sched_getaffinity(0))
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
    ]
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(
        kernel.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return [i for i in range(8 * ctypes.sizeof(process_mask)) if process_mask.value & (1 << i)]

def _initial_record(stop_after, swing_mode):
    from datetime import datetime, timezone

    return dict(
        schema="SCPSC_TRAJECTORY_LAYER_RUN_V2",
        reference_model="MASS_INDEPENDENT_TAUT_CABLE",
        success=False,
        status="STARTED",
        qp_calls=0,
        frontend_algorithm="LAZY_THETA_STAR",
        R_calls=0,
        python=platform.python_version(),
        python_executable=sys.executable,
        platform=platform.platform(),
        source_tree=str(ROOT),
        cpu_affinity=_cpu_affinity(),
        numerical_threads=1,
        stop_after=stop_after,
        timing_protocol="SCPSC_CONSTRUCTION_AND_FINAL_ACCEPTANCE_V4",
        started_utc=datetime.now(timezone.utc).isoformat(),
        planning_candidate_pass=False,
        swing_mode=swing_mode,
        legacy_trajectory_pass=False,
        swing_forward_pass=None,
        swing_status="NOT_RUN",
        independent_status="NOT_RUN",
        full_validation_pass=False,
        swing_seconds=0.0,
        acceptance_scope="not yet validated",
    )

def _record_failure(error, budget, record, output, swing_mode):
    stage = budget.stage if budget is not None else "input"
    if isinstance(error, KeyboardInterrupt):
        record["status"] = "INTERRUPTED"
    elif isinstance(error, TimeoutError):
        record["status"] = "EXTERNAL_TIMEOUT"
    elif (
        type(error).__name__ == "SCPSCDependencyError"
        or "DEPENDENCY_UNAVAILABLE" in str(error)
        or str(error).startswith(("OSQP_VERSION_MISMATCH", "CLARABEL_VERSION_MISMATCH"))
    ):
        record["status"] = "DEPENDENCY_UNAVAILABLE"
    elif type(error).__name__ == "CorridorIncompatible":
        record["status"] = "CANDIDATE_WINDOWS_ADMISSION_FAILED"
    elif "HSEAL" in str(error) or isinstance(error, FileNotFoundError):
        record["status"] = "MISSING_INPUT"
    else:
        record["status"] = "INPUT_OR_NUMERICAL_ERROR"
    record["success"] = False
    if stage == "swing_validation":
        swing_end = time.perf_counter()
        swing_start = record.pop("_swing_started", swing_end)
        record.update(
            swing_forward_pass=False,
            swing_status="SWING_VALIDATION_ERROR",
            swing_seconds=swing_end - swing_start,
        )
        if swing_mode == "report" and not isinstance(error, KeyboardInterrupt):
            record["success"] = record["legacy_trajectory_pass"]
            record["status"] = record["legacy_status"]
    record.update(failure_stage=stage, failure_reason=str(error), exception_type=type(error).__name__)
    (output / "error_trace.txt").write_text(traceback.format_exc(), encoding="utf-8")

def _finish_record(started, planning_wall, statistics_seconds, budget, record, output):
    from datetime import datetime, timezone
    from scpsc.result_io import save_json

    elapsed = time.perf_counter() - started
    record["total_seconds"] = elapsed
    record["total_with_diagnostics_seconds"] = elapsed
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    if planning_wall is None:
        record.update(
            planning_seconds=None,
            planning_seconds_including_statistics=None,
            online_report_statistics_seconds=statistics_seconds,
        )
    if budget is not None:
        durations = dict(budget.durations)
        durations[budget.stage] = durations.get(budget.stage, 0.0) + time.perf_counter() - budget.stage_start
        record["stage_seconds"] = durations
    if not record.get("success"):
        record.setdefault("later_stages", "NOT_RUN where not reached")
    save_json(output / "run_record.json", record)
