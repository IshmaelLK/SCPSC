"""Command-line arguments and child-process execution for one SCPSC task."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .run_state import write

ROOT = Path(__file__).resolve().parents[2]

def configure_threads():
    """Original one-thread benchmark setting, applied only at CLI startup."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"

def launch(config, output, replay=None, stop_after="complete", retime_route_from=None, swing_mode="required"):
    if replay is not None and retime_route_from is not None:
        raise ValueError("Choose final-reference replay or saved-route retiming")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Output already exists: " + str(output))
    from scpsc.configuration import load_config

    load_config(config)  # Validate and report retired fields before dispatch.
    args = [
        sys.executable,
        str(ROOT / "scripts/run_scpsc.py"),
        "--worker",
        "--config",
        str(Path(config).resolve()),
        "--output",
        str(output),
        "--stop-after",
        stop_after,
        "--swing-mode",
        swing_mode,
    ]
    if replay is not None:
        args += ["--replay-from", str(Path(replay).resolve())]
    if retime_route_from is not None:
        args += ["--retime-route-from", str(Path(retime_route_from).resolve())]
    started = time.perf_counter()
    kwargs = (
        {"start_new_session": True}
        if os.name != "nt"
        else {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP")}
    )
    process = subprocess.Popen(args, **kwargs)
    try:
        process.wait()
    except KeyboardInterrupt:
        # Explicit user cancellation is the only process-tree termination here.
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        process.wait()
        output.mkdir(parents=True, exist_ok=True)
        record = dict(status="INTERRUPTED", success=False, total_seconds=time.perf_counter() - started)
        path = output / "run_record.json"
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved.update(record)
            record = saved
        write(path, record)
        return record
    path = output / "run_record.json"
    if not path.is_file():
        output.mkdir(parents=True, exist_ok=True)
        record = dict(
            status="WORKER_CRASH",
            success=False,
            returncode=process.returncode,
            total_seconds=time.perf_counter() - started,
        )
        write(path, record)
        return record
    result = json.loads(path.read_text(encoding="utf-8"))
    if process.returncode and result.get("success"):
        result.update(success=False, status="WORKER_EXIT_ERROR", returncode=process.returncode)
        write(path, result)
    return result

def main():
    configure_threads()
    from .application import worker

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--replay-from", type=Path)
    source.add_argument(
        "--retime-route-from", type=Path, help="reuse saved frontend geometry and recompute the selected velocity clock"
    )
    p.add_argument("--stop-after", choices=["corridor", "complete"], default="complete")
    p.add_argument(
        "--swing-mode",
        choices=["off", "report", "required"],
        default="required",
        help="required gates full validation; report preserves legacy success; off is historical regression",
    )
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    f = worker if a.worker else launch
    result = f(a.config, a.output, a.replay_from, a.stop_after, a.retime_route_from, a.swing_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") or result.get("status") == "CORRIDOR_ONLY_PASS" else 1
