"""Existing JSON result encodings, separated from planning mathematics."""

from pathlib import Path
import json
import numpy as np


def save_json(path, value):
    """Write numerical output without importing the legacy optimization pipeline."""

    def convert(item):
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, np.generic):
            return convert(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    Path(path).write_text(json.dumps(convert(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def dump(path, data):
    def cv(x):
        if isinstance(x, dict):
            return {str(k): cv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [cv(v) for v in x]
        if isinstance(x, np.ndarray):
            return cv(x.tolist())
        if isinstance(x, np.generic):
            return cv(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cv(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
