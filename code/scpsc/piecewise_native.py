"""Bindings for the existing piece-geometry library and SCPSC native kernel."""
from functools import lru_cache
from pathlib import Path
import ctypes
import importlib.util
import os
import sysconfig

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def piece_geometry_library():
    path = ROOT / "native/build" / ("piece_geometry.dll" if os.name == "nt" else "piece_geometry.so")
    if not path.is_file():
        raise RuntimeError("PIECE_GEOMETRY_DEPENDENCY_UNAVAILABLE: run python scripts/build_piece_geometry.py")
    lib = ctypes.CDLL(str(path))
    real = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    integer = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    index = ctypes.c_int64
    lib.scpsc_raster_window.argtypes = [real, real, real, index, index, index, index, index, index, real, index]
    lib.scpsc_raster_window.restype = ctypes.c_int
    lib.scpsc_clip_raster_pieces.argtypes = [
        real, real, real, index, index, index, index, index, index,
        real, real, real, index, integer, integer,
    ]
    lib.scpsc_clip_raster_pieces.restype = index
    lib.scpsc_clip_pieces.argtypes = [real, ctypes.c_int64, real, real, real, ctypes.c_int64, integer, integer]
    lib.scpsc_clip_pieces.restype = ctypes.c_int64
    lib.scpsc_payload_brackets.argtypes = [
        real,
        ctypes.c_int64,
        integer,
        ctypes.c_int64,
        real,
        ctypes.c_double,
        ctypes.c_int,
        real,
    ]
    lib.scpsc_payload_brackets.restype = ctypes.c_int
    return lib


def scpsc_native_enabled():
    """Native path by default since the 2026-10-15 acceptance; 0 selects the Python reference."""
    mode = os.environ.get("SCPSC_NATIVE", "1")
    if mode not in ("0", "1"):
        raise ValueError("SCPSC_NATIVE must be 0 or 1")
    return mode == "1"


@lru_cache(maxsize=1)
def scpsc_module():
    # Only the immutable library binding is cached; never segment/task data.
    path = ROOT / "native/build" / ("_scpsc_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    if not path.is_file():
        raise RuntimeError("SCPSC_NATIVE_DEPENDENCY_UNAVAILABLE: run python scripts/build_piece_geometry.py "
                           "with the current Python and NumPy; no silent Python fallback")
    spec = importlib.util.spec_from_file_location("_scpsc_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.longdouble_mantissa_bits != np.finfo(np.longdouble).nmant + 1:
        raise RuntimeError("SCPSC_NATIVE_LONGDOUBLE_MISMATCH")
    return module
