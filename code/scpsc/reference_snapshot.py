"""Validated, immutable final-reference replay for corridor-only experiments.

A snapshot contains the FINAL densified points and physical times. It is not a
frontend seed and must never be passed through subdivision or easing again.
Only current-model snapshots with explicit final-clock evidence and a
matching effective configuration are accepted; external map bytes must also be traceable.
This module performs no search, corridor construction, retiming or QP solve.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .velocity_constraints import from_config

SCHEMA = 'SCPSC_FINAL_REFERENCE_TRAJECTORY_LAYER_V3'
FILES = ('frontend.npz', 'frontend.json', 'reference.npz', 'time_and_approach.json')
ASSETS = ('planning_terrain', 'source_terrain', 'canopy_seal')
# Only corridor construction may differ in a corridor-only replay. Inactive
# descriptive strings do not alter the mathematical experiment contract.
CORRIDOR_KEYS = {'corridor_center_margin_m', 'corridor_window_policy',
                 'corridor_method', 'corridor_options', 'corridor_adaptation', 'corridor_backoff_m',
                 'source', 'approach_registration', 'use_system_envelope'}
ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def frozen_config(config):
    """Numerical/physical contract; path relocation is not a model change."""
    scene = config.get('mechanism_scene', {})
    return dict(
        dynamics=config.get('dynamics', {}), limits=config.get('limits', {}),
        validation=config.get('validation', {}), model_schema=config.get('schema'), frontend=config.get('frontend', {}),
        geometry=config.get('geometry', {}),
        ubsqp={k: v for k, v in config['ubsqp'].items() if k not in CORRIDOR_KEYS},
        scene={k: v for k, v in scene.items() if k not in ASSETS and k not in
               ('physical_case_id', 'parameter_source', 'source_task')})


def current_assets(config):
    result = {}
    for key in ASSETS:
        value = config.get('mechanism_scene', {}).get(key)
        if value is not None:
            path = Path(value)
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_file():
                raise ValueError('REPLAY_ASSET_MISSING: ' + str(path))
            result[key] = dict(name=path.name, sha256=sha256(path))
    return result



def _clock(times, count, name):
    if (times.shape != (count,) or not np.isfinite(times).all() or times[0] != 0.
            or np.any(np.diff(times) <= 0.)):
        raise ValueError('REPLAY_INVALID_REFERENCE: invalid ' + name)


def _route(points, name):
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
            or not np.isfinite(points).all()
            or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-11)):
        raise ValueError('REPLAY_INVALID_REFERENCE: invalid ' + name)


def check_polyline_refinement(front, points, tolerance=1e-8):
    """Retain every frontend vertex in order; intermediate points stay on each edge.

    Ordered projection checks also reject excursions or backtracking on an edge.
    This does not generate or replace a point. Repeated nonconsecutive vertices
    are handled by their order, not a nearest-point global shortcut.
    """
    if not np.allclose(front[0], points[0], rtol=0., atol=tolerance):
        raise ValueError('REPLAY_REFERENCE_MISMATCH: different route start')
    index = 0
    for left, right in zip(front[:-1], front[1:]):
        delta = right - left
        length = float(np.linalg.norm(delta))
        previous = 0.
        found = False
        while index + 1 < len(points):
            index += 1
            offset = points[index] - left
            distance = float(offset @ delta / length)
            residual = float(np.linalg.norm(offset - distance * delta / length))
            if residual > tolerance or distance < previous - tolerance or distance > length + tolerance:
                raise ValueError('REPLAY_REFERENCE_MISMATCH: bend lost, off-edge point or reversed order')
            if np.linalg.norm(points[index] - right) <= tolerance:
                found = True
                break
            previous = distance
        if not found:
            raise ValueError('REPLAY_REFERENCE_MISMATCH: original vertex not retained')
    if index != len(points) - 1:
        raise ValueError('REPLAY_REFERENCE_MISMATCH: trailing points beyond original endpoint')


def _progress(s, r):
    # Forward check of the existing inverse clock, not a second time mapping.
    def integral(u):
        return 21*u**6 - 60*u**7 + 67.5*u**8 - 35*u**9 + 7*u**10
    out = np.empty_like(s)
    left = s < r
    right = s > 1.-r
    mid = ~(left | right)
    out[left] = r*integral(s[left]/r)/(1.-r)
    out[mid] = (s[mid]-r/2.)/(1.-r)
    out[right] = 1.-r*integral((1.-s[right])/r)/(1.-r)
    return out


def validate_final_reference(source, env, config):
    source = Path(source).resolve()
    if not (source/'reference_contract.json').is_file():
        raise ValueError('CURRENT_NO_AERO_REFERENCE_CONTRACT_REQUIRED')
    from .configuration import validate_config
    validate_config(config)
    for name in FILES + ('effective_config.json',):
        if not (source/name).is_file():
            raise ValueError('REPLAY_MISSING_METADATA: ' + name)
    saved_config = read_json(source/'effective_config.json')
    validate_config(saved_config)
    timing = read_json(source/'time_and_approach.json')
    with np.load(source/'frontend.npz', allow_pickle=False) as saved:
        front, front_times = saved['positions'].copy(), saved['times'].copy()
    with np.load(source/'reference.npz', allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    try:
        points, times, dt = (arrays[k] for k in ('positions', 'times', 'durations'))
        _route(front, 'frontend points'); _route(points, 'final reference points')
        _clock(front_times, len(front), 'frontend clock'); _clock(times, len(points), 'final reference clock')
        if (dt.shape != (len(points)-1,) or not np.isfinite(dt).all() or np.any(dt <= 0.)
                or not np.allclose(dt, np.diff(times), rtol=0., atol=1e-10)):
            raise ValueError('REPLAY_DURATION_MISMATCH: durations differ from final time differences')
        if not np.array_equal(arrays['original_frontend_times'], front_times):
            raise ValueError('REPLAY_REFERENCE_MISMATCH: different frontend clock provenance')
    except KeyError as error:
        raise ValueError('REPLAY_MISSING_METADATA: ' + str(error)) from error
    check_polyline_refinement(front, points)
    endpoint_error = float(np.max(np.abs(points[[0, -1]] - np.stack((env['start'], env['goal'])))))
    if endpoint_error > config['validation']['endpoint_tolerance']:
        raise ValueError('REPLAY_ENDPOINT_MISMATCH: saved reference belongs to another task')
    if float(timing['total_time_s']) != float(times[-1]):
        raise ValueError('REPLAY_TIME_MISMATCH: metadata differs from final clock')
    settings = config['ubsqp']
    model = from_config(settings, config.get('limits', {}).get('vL'))
    if timing.get('velocity_constraint') != model.metadata():
        raise ValueError('REPLAY_VELOCITY_MISMATCH: saved velocity model/bounds differ')
    for key, value in model.npz_fields().items():
        if key not in arrays or not np.array_equal(arrays[key], value):
            raise ValueError('REPLAY_VELOCITY_MISMATCH: reference array ' + key)
    saved_problem,current_problem=frozen_config(saved_config),frozen_config(config)
    # Compute budgets do not change the saved route, clock or mathematical QP.
    # Keep the saved contract check below exact; only replay may change budgets.
    for key in ('max_iter','time_limit','task_budget_s'):
        saved_problem['ubsqp'].pop(key,None)
        current_problem['ubsqp'].pop(key,None)
    if canonical(saved_problem) != canonical(current_problem):
        raise ValueError('REPLAY_CONFIG_MISMATCH: physics, timing, QP or validation changed')
    mode = settings.get('reference_timing_mode', 'original')
    if timing.get('reference_timing_mode') != mode or timing.get('timing_kappa') != settings['timing_kappa']:
        raise ValueError('REPLAY_CONFIG_MISMATCH: timing metadata does not match settings')
    r = settings.get('reference_clock_r')
    if r is not None:
        if (timing.get('reference_clock') != 'T*inverse_F(original_reference_time/T)'
                or timing.get('reference_clock_changed') is not True
                or timing.get('reference_clock_r') != r):
            raise ValueError('REPLAY_NOT_FINAL: missing explicit completed easing metadata')
        if not np.isfinite(r) or not 0. < r < .5:
            raise ValueError('REPLAY_INVALID_REFERENCE: invalid clock parameter')
        for name in ('original_reference_times', 'original_reference_durations'):
            if name not in arrays:
                raise ValueError('REPLAY_NOT_FINAL: missing ' + name)
        raw = arrays['original_reference_times']; raw_dt = arrays['original_reference_durations']
        _clock(raw, len(points), 'pre-easing clock')
        if (raw_dt.shape != dt.shape or not np.isfinite(raw_dt).all()
                or not np.allclose(raw_dt, np.diff(raw), rtol=0., atol=1e-10)
                or raw[-1] != times[-1]):
            raise ValueError('REPLAY_DURATION_MISMATCH: invalid pre-easing evidence')
        error = float(np.max(np.abs(times[-1]*_progress(times/times[-1], r) - raw)))
        if error > 1e-8:
            raise ValueError('REPLAY_TIME_MISMATCH: final times are not the saved once-eased clock')
        stage = 'final_eased'
    else:
        if timing.get('reference_stage') != 'final_uneased' or timing.get('reference_clock_changed', False):
            raise ValueError('REPLAY_NOT_FINAL: explicit final_uneased stage required')
        error = 0.; stage = 'final_uneased'
    if timing.get('reference_stage', stage) != stage:
        raise ValueError('REPLAY_NOT_FINAL: inconsistent reference stage')
    saved_mode = saved_config['ubsqp'].get('approach_mode', 'hard')
    if saved_mode == 'none' and timing.get('approach') is not None:
        raise ValueError('REPLAY_APPROACH_MISMATCH: disabled approach has restrictions')
    contract_path = source/'reference_contract.json'
    if contract_path.exists():
        contract = read_json(contract_path)
        if (contract.get('schema') != SCHEMA or contract.get('reference_stage') != stage
                or contract.get('frozen_config') != frozen_config(saved_config)):
            raise ValueError('REPLAY_CONTRACT_MISMATCH: invalid schema or frozen configuration')
        for name in FILES:
            if contract.get('files_sha256', {}).get(name) != sha256(source/name):
                raise ValueError('REPLAY_HASH_MISMATCH: ' + name)
        expected_assets = contract.get('assets', {})
        provenance = 'versioned_snapshot'
    else:
        raise ValueError('CURRENT_NO_AERO_REFERENCE_CONTRACT_REQUIRED')
    if expected_assets != current_assets(config):
        raise ValueError('REPLAY_ASSET_MISMATCH: source geometry bytes changed')
    if (contract.get('source_geometry_version') != env['source'].source_geometry_version
            or contract.get('source_geometry_id') != env['source'].source_geometry_id):
        raise ValueError('REPLAY_SOURCE_GEOMETRY_MISMATCH: operation source differs')
    return dict(points=points, times=times, durations=dt, approach=timing.get('approach'),
        audit=dict(input_directory=str(source), frontend_reused=True, search_calls=0,
            subdivision_calls=0, time_mapping_calls=0, route_changed=False, timing_changed=False,
            approach_reused=True, frozen_reference_time_s=float(times[-1]),
            saved_route_endpoint_error_m=endpoint_error, forward_clock_check_error_s=error,
            frontend_point_count=len(front), reference_point_count=len(points),
            original_vertices_retained=True, source_provenance=provenance,
            assets_verified=True, reference_stage=stage))


def write_contract(output, config, env=None):
    output = Path(output)
    timing = read_json(output/'time_and_approach.json')
    stage = ('final_eased' if timing.get('reference_clock_changed') else 'final_uneased')
    result = dict(schema=SCHEMA, reference_stage=stage, frozen_config=frozen_config(config),
                  assets=current_assets(config), files_sha256={n: sha256(output/n) for n in FILES})
    if env is None:
        from .scene_inputs import load_scene
        env = load_scene(config)
    result.update(source_geometry_version=env['source'].source_geometry_version,
                  source_geometry_id=env['source'].source_geometry_id)
    (output/'reference_contract.json').write_text(json.dumps(result, ensure_ascii=False,
        indent=2, allow_nan=False)+'\n', encoding='utf-8')
    # Low-level adapter callers also receive a self-contained effective profile.
    if not (output/'effective_config.json').exists():
        (output/'effective_config.json').write_text(json.dumps(config, ensure_ascii=False,
            indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result
