"""Load the unchanged continuous source and explicit operating domain."""

from copy import deepcopy
from pathlib import Path
import numpy as np
from .geometry import Terrain
from .canopy_source import OperationSource, load_seal
from .configuration import validate_config


def load_scene(config):
    validate_config(config)
    s = config["mechanism_scene"]
    if not s.get("canopy_seal"):
        raise ValueError(
            "HSEAL_INPUT_REQUIRED: provide mechanism_scene.canopy_seal; H0 alone does not close forest gaps"
        )
    seal = Path(s["canopy_seal"])
    if not seal.is_file():
        raise FileNotFoundError("HSEAL_INPUT_MISSING: " + str(seal))
    original = Terrain.load(s["source_terrain"])
    source = OperationSource.from_inputs(
        original, load_seal(seal), config.get("geometry", {}).get("query_mode", "scan")
    )
    # All geometry consumers share the same fixed source, including planning.
    terrain = source
    domain = np.asarray(s["operation_domain"], float)
    if np.any(domain[0, :2] < source.bounds[0]) or np.any(domain[1, :2] > source.bounds[1]):
        raise ValueError("OPERATION_DOMAIN_OUTSIDE_KNOWN_SOURCE_MAP")
    if s.get("obstacles"):
        raise ValueError(
            "This profile models canopy solids; independent obstacle meshes require a separate registered adapter"
        )
    return dict(
        source=source,
        terrain=terrain,
        domain=domain,
        start=np.asarray(s["start"], float),
        goal=np.asarray(s["goal"], float),
        environment_reference=deepcopy(s),
        source_geometry_version=source.source_geometry_version,
        source_geometry_id=source.source_geometry_id,
    )
