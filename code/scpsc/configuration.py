"""Frozen model configuration, explicit migration and input path resolution."""

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
from .dynamics import DynamicsParams
from .envelope import planning_envelope_mode
from .velocity_constraints import from_config

SCHEMA = "SCPSC_TRAJECTORY_LAYER_V1"
DEPRECATED_TIME_KEYS = frozenset(
    (
        "native_budget_s",
        "route_wall_limit_s",
        "time_limit_s",
        "time_limit",
        "task_budget_s",
        "iris_segment_timeout_s",
        "segment_timeout_sec",
        "task_timeout_sec",
        "deadline",
        "remaining",
        "remaining_s",
        "timeout_s",
        "timeout_sec",
        "preparation_timeout_s",
        "corridor_timeout_s",
        "qp_timeout_s",
        "validation_timeout_s",
    )
)


def migrate_config(config):
    """Report obsolete model fields and remove every project wall-clock limit."""
    config = deepcopy(config)
    changes = list(config.get("config_migration", []))
    limits = config.get("limits", {})
    if "rR" in limits and limits["rR"] > min(limits.get("rL", 0.0), limits.get("rQ", 0.0)):
        raise ValueError(
            "ROPE_CROSS_SECTION_NOT_COVERED: rL and rQ must both cover the previous rR; do not shrink the envelope"
        )

    def remove(mapping, key, path):
        changes.append(dict(path=path + "." + key, old_value=mapping.pop(key), action="removed obsolete field"))

    def visit(value, path):
        if isinstance(value, dict):
            for key in list(value):
                if key == "config_migration":
                    continue
                if key in DEPRECATED_TIME_KEYS:
                    remove(value, key, path)
                else:
                    visit(value[key], path + "." + key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                visit(item, path + "[" + str(i) + "]")

    previous = len(changes)
    if config.get("schema") == "SCPSC_NO_AERO_V1":
        changes.append(
            dict(
                path="config.schema",
                old_value=config["schema"],
                new_value=SCHEMA,
                action="trajectory-layer model migration",
            )
        )
        config["schema"] = SCHEMA
    visit(config, "config")
    for key in ("mL", "mQ", "yaw", "tension_guard", "thrust_guard", "upward_guard"):
        if key in config.get("dynamics", {}):
            remove(config["dynamics"], key, "config.dynamics")
    for key in ("Tmin", "Tmax", "fmin", "fmax", "vQ", "phi", "omega", "rR"):
        if key in limits:
            remove(limits, key, "config.limits")
    if "limits" in config and "gamma" not in limits:
        limits["gamma"] = 0.1
        changes.append(dict(path="config.limits.gamma", new_value=0.1, action="registered paper default"))
    if len(changes) > previous:
        config["config_migration"] = changes
        warnings.warn(
            "SCPSC_CONFIG_MIGRATED: " + ", ".join(row["path"] for row in changes[previous:]), UserWarning, stacklevel=2
        )
    return config


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def validate_config(config):
    if config.get("schema") != SCHEMA:
        raise ValueError("NO_AERO_SCHEMA_REQUIRED: use the explicit migration script, not a legacy profile")
    for k in ("wind", "wind_policy"):
        if k in config:
            raise ValueError("Unsupported environmental field setting: " + k)
    DynamicsParams(**config["dynamics"])
    limits = config["limits"]
    if not 0 < float(limits["gamma"]) < 1:
        raise ValueError("Specific-force gamma must lie in (0,1)")
    if not 0 <= float(limits["theta"]) < np.pi / 2:
        raise ValueError("Fixed rope cap theta must lie in [0,pi/2)")
    scene = config["mechanism_scene"]
    if any(k.startswith("wind") or k in ("analytic", "physical_configuration") for k in scene):
        raise ValueError("Legacy field-bearing scene metadata must be migrated explicitly")
    D = np.asarray(scene.get("operation_domain"), float)
    if D.shape != (2, 3) or not np.isfinite(D).all() or np.any(D[0] >= D[1]):
        raise ValueError("EXPLICIT_OPERATION_DOMAIN_REQUIRED: finite ordered [lower_xyz,upper_xyz]")
    if not isinstance(scene.get("operation_domain_source"), str) or not scene["operation_domain_source"].strip():
        raise ValueError("Provide operation_domain_source independently of any environmental-field mask")
    for key in ("start", "goal"):
        p = np.asarray(scene[key], float)
        if p.shape != (3,) or not np.isfinite(p).all() or np.any(p <= D[0]) or np.any(p >= D[1]):
            raise ValueError("Task endpoint outside its explicit operation domain: " + key)
    if scene["start"][2] <= scene["goal"][2]:
        raise ValueError("SCPSC task must have positive net descent")
    front = config["frontend"]
    q = config["ubsqp"]
    if q.get("corridor_method", "piecewise_socp") != "piecewise_socp":
        raise ValueError("SCPSC_ONLY: corridor_method must be piecewise_socp")
    if planning_envelope_mode(q) != "fixed_system":
        raise ValueError("SCPSC requires the fixed_system planning envelope")
    if (
        front.get("algorithm") != "lazy_theta_star"
        or front.get("heuristic_weight") != 1
        or front.get("r_enabled", False)
    ):
        raise ValueError("Current front end is Lazy Theta*, h=1, with no route-adjustment stage")
    for key, val in [
        ("n_control", 33),
        ("bottleneck_sections", 26),
        ("reference_clock_r", 0.2),
        ("reference_timing_mode", "bottleneck_equal_arclength"),
        ("approach_mode", "none"),
    ]:
        if q.get(key) != val:
            raise ValueError("Unsupported frozen backend setting: " + key)
    from_config(q, config["limits"]["vL"])
    if q.get("velocity_constraint") not in ("polytope26", "box"):
        raise ValueError("Select polytope26 or the explicitly registered inscribed-box ablation")
    return config


def load_config(path):
    path = Path(path).resolve()
    cfg = migrate_config(read_json(path))
    validate_config(cfg)
    for key in ("source_terrain", "planning_terrain", "canopy_seal"):
        value = cfg["mechanism_scene"].get(key)
        if value is not None:
            asset = Path(value)
            cfg["mechanism_scene"][key] = str((path.parent / asset).resolve() if not asset.is_absolute() else asset)
    return cfg
