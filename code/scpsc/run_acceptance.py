"""Separate original trajectory acceptance from a completed forward-swing solve."""

LEGACY_SCOPE = (
    "trajectory-layer geometry, endpoints, load speed and continuous specific force"
)
FULL_SCOPE = (
    LEGACY_SCOPE
    + "; single nominal taut-cable forward integration under ideal UAV acceleration"
)


def apply_swing_acceptance(record):
    """Update only postplanning status; never rescue a failed original certificate."""
    mode = record["swing_mode"]
    if mode not in ("off", "report", "required"):
        raise ValueError("Unknown swing mode: " + str(mode))
    legacy = bool(record["legacy_trajectory_pass"])
    full = mode == "required" and legacy and record["swing_forward_pass"] is True
    record["full_validation_pass"] = full
    record["success"] = full if mode == "required" else legacy
    point_only = record["legacy_status"] == "POINT_TRAJECTORY_PASS"
    scope = (
        "point-envelope trajectory geometry, endpoints, load speed and continuous specific force"
        if point_only
        else LEGACY_SCOPE
    )
    record["acceptance_scope"] = (
        scope + "; single nominal taut-cable forward integration under ideal UAV acceleration"
        if full
        else scope
    )
    record["transport_evaluated"] = False
    if full:
        record["status"] = (
            "FULL_POINT_TRAJECTORY_VALIDATION_PASS"
            if point_only
            else "FULL_TRAJECTORY_VALIDATION_PASS"
        )
    elif mode == "required" and legacy:
        record["status"] = "SWING_VALIDATION_FAILED"
    else:
        record["status"] = record["legacy_status"]
    return record
