# SCPSC

SCPSC constructs safe flight corridors above piecewise-affine terrain and canopy surfaces. The source includes SCPSC and its accompanying planning pipeline: Lazy Theta* reference search, fixed-time B-spline trajectory optimization, and trajectory and suspended-load validation.

This repository contains the SCPSC source, build and run entry points, and runtime dependencies. Map files, experiment configurations and results, test datasets, compiled binaries, and comparison-method implementations are not included.

Python modules are in `code/scpsc/`; C++ kernels are in `native/`.

## Build

Use Python 3.12 and a C++17 compiler. On Linux, install the development headers for the same Python interpreter. On Windows, use an x64 MSVC developer terminal so that `cl.exe` and its headers and libraries are available. Run from the project root:

```text
python -m pip install -r requirements.txt
python scripts/build_lazytheta.py
python scripts/build_source_clip.py
python scripts/build_piece_geometry.py
```

Each build script accepts `--compiler`, for example `--compiler cl.exe`. Native outputs are written to `native/build/` and must be built with the Python environment used to run SCPSC. SOCPs use Clarabel **0.11.1**; the trajectory QP uses OSQP **1.1.3**.

## Run with your own inputs

Supply a JSON task configuration and terrain and canopy files:

```text
python scripts/run_scpsc.py --config /path/to/your_task.json --output runs/task --swing-mode required
```

The task configuration uses schema `SCPSC_TRAJECTORY_LAYER_V1`. Its fields specify system parameters (`dynamics`, `limits`), terrain paths, endpoints and an explicit operation domain (`mechanism_scene`), reference search (`frontend`), and trajectory and corridor settings (`ubsqp`). Select `ubsqp.corridor_method = "piecewise_socp"`. Required settings and mathematical checks are defined in [configuration.py](code/scpsc/configuration.py).

For raster inputs:

- The terrain NPZ contains increasing one-dimensional `x`, `y` coordinates and `dtm`, `dsm` arrays of shape `(len(y), len(x))`.
- The canopy NPZ contains `x`, `y`, and `hseal` on the same grid.
- Coordinates and heights are in metres in a consistent coordinate system. Grid cells use the southwest-to-northeast diagonal. The planning surface is the maximum of terrain, surface, and canopy heights on this grid.
- Input paths are resolved relative to the configuration file. Task endpoints and the operation domain must lie within known map coverage.

The output directory must not already exist. Complete success requires `success: true` and `full_validation_pass: true` in `run_record.json`. For corridor construction alone, add `--stop-after corridor`; `CORRIDOR_ONLY_PASS` is not a validated trajectory.

## Scripts

| Script | Purpose |
|---|---|
| `build_lazytheta.py` | Compile the reference-search program |
| `build_source_clip.py` | Compile surface clipping |
| `build_piece_geometry.py` | Compile piece geometry and the SCPSC native kernel |
| `run_scpsc.py` | Run one planning task |

Use `--help` for arguments. Required geometric, trajectory, and suspended-load checks remain part of the planning implementation.

## Results and timing

The planner writes the effective configuration, reference, corridor halfspaces, trajectory coefficients, independent checks, and completion or failure status into the selected output directory.

`corridor_ready_seconds` measures corridor construction, excluding representation preparation, independent acceptance, report-only statistics, and output writing. SCPSC does not build an additional obstacle representation. `planning_seconds` extends through all required final checks, including suspended-load validation when `--swing-mode required` is selected.

## License

This project is distributed under the [MIT License](LICENSE).
