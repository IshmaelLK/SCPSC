"""Check the native geometry dependency used by SCPSC."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class SCPSCDependencyError(RuntimeError):
    pass


def dependency_status(method="piecewise_socp"):
    if method != "piecewise_socp":
        raise ValueError("SCPSC_ONLY: corridor_method must be piecewise_socp")
    path = ROOT / "native/build" / ("piece_geometry.dll" if os.name == "nt" else "piece_geometry.so")
    return dict(
        available=path.is_file(),
        status="AVAILABLE" if path.is_file() else "DEPENDENCY_UNAVAILABLE",
        reason="run python scripts/build_piece_geometry.py",
        binary=str(path),
    )
