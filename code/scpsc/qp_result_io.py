"""QP result serialization at the application-selected output boundary."""

import json
from pathlib import Path

import numpy as np


def json_value(value):
    """Convert numerical proof records to JSON without object-valued arrays."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_assembled_matrices(output_dir, matrices):
    """Keep raw matrices even when subsequent mathematical preprocessing fails."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "qp_matrices.npz", **matrices, raw_x=np.empty(0), raw_y=np.empty(0))


def save_qp_result(output_dir, matrices, reduced, raw, x, y, diagnostics):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    # raw_x is the original-coordinate primal point; raw_y is empty.
    # Only native reduced duals are recorded below; no dual lifting is done.
    np.savez_compressed(destination / "qp_matrices.npz", **matrices, raw_x=x, raw_y=y)
    np.savez_compressed(
        destination / "reduced_matrices.npz",
        P=reduced["P"],
        q=reduced["q"],
        A=reduced["A"],
        l=reduced["l"],
        u=reduced["u"],
        xbase=reduced["xbase"],
        M=reduced["M"],
        p=reduced["p"],
        N=reduced["N"],
        Eaug=reduced["Eaug"],
        baug=reduced["baug"],
        row_den=reduced["row_den"],
        equality_singular_values=reduced["equality_singular_values"],
        augmented_equality_rank=len(reduced["Eaug"]),
        constant_original_rows=reduced["constant_original_rows"],
        constant_row_weights=reduced["constant_row_weights"],
        near_constant_rows=reduced["near_constant_rows"],
        raw_reduced_x=raw.x,
        raw_reduced_y=raw.y,
        reduced_prim_inf_cert=raw.prim_inf_cert,
        reduced_dual_inf_cert=raw.dual_inf_cert,
    )
    (destination / "qp_status.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
