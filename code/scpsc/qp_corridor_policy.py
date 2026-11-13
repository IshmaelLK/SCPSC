"""Bounded, shared C3 window selection; never a route or QP retry.

Historical configurations default to a single fixed window. The explicit
bounded_shrink policy reads optimization.corridor_width_candidates, rebuilds
the ENTIRE corridor sequence for each width, and returns the first admission
    pass. It forwards the explicitly configured planning-envelope mode; source
windows, physical limits and reference times are not changed by this policy.
"""
from copy import deepcopy
import math
import time
from .geometry import CorridorIncompatible
from .envelope import planning_envelope_mode


def window_policy(settings, optimization):
    mode = settings.get('corridor_window_policy', 'fixed')
    if mode not in ('fixed', 'bounded_shrink'):
        raise ValueError('C3_WINDOW_POLICY: expected fixed or bounded_shrink')
    value = settings.get('corridor_center_margin_m', optimization.get('corridor_width'))
    if isinstance(value, bool):
        raise ValueError('C3_WINDOW_POLICY: invalid center margin')
    try:
        margin = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('C3_WINDOW_POLICY: finite positive center margin required') from error
    if not math.isfinite(margin) or margin <= 0.:
        raise ValueError('C3_WINDOW_POLICY: finite positive center margin required')
    if mode == 'fixed':
        return mode, [margin]
    values = optimization.get('corridor_width_candidates')
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 4:
        raise ValueError('C3_WINDOW_POLICY: one to four registered candidate margins required')
    try:
        candidates = [float(x) for x in values]
    except (TypeError, ValueError) as error:
        raise ValueError('C3_WINDOW_POLICY: invalid candidate margin') from error
    if (any(isinstance(x, bool) for x in values)
            or not all(math.isfinite(x) and x > 0. for x in candidates)
            or candidates[0] != margin
            or any(a <= b for a, b in zip(candidates, candidates[1:]))):
        raise ValueError('C3_WINDOW_POLICY: positive strictly descending margins starting at configured margin required')
    return mode, candidates


def build_with_window_policy(env, route, params, config, checkpoint, builder, emit=None):
    """Only CorridorIncompatible permits a smaller window; propagate all else.

    emit receives an immutable snapshot after each attempt, including a rejected
    window's full source/seed diagnostic. Runtime errors and cancellation are
    recorded and propagated, never treated as geometry rejection. No QP is called here.
    """
    settings = config['ubsqp']
    use_system_envelope = settings.get('use_system_envelope', True)
    if not isinstance(use_system_envelope, bool):
        raise ValueError('ubsqp.use_system_envelope must be a bool')
    planning_mode = planning_envelope_mode(settings)
    use_system_envelope = planning_mode == 'fixed_system'
    prefix = dict(fixed_system='FIXED', load_only='LOAD_ONLY', point='POINT',
                  bare_point='BARE_POINT')[planning_mode]
    admission_pass = prefix + '_CORRIDOR_ADMISSION_PASS'
    admission_failed = prefix + '_CORRIDOR_ADMISSION_FAILED'
    mode, candidates = window_policy(settings, config['optimization'])
    attempts = []
    policy_statistics_seconds = 0.0

    def snapshot(status, selected=None):
        nonlocal policy_statistics_seconds
        statistics_started = time.perf_counter()
        state = dict(center_margin_policy=mode,
            use_system_envelope=use_system_envelope, planning_mode=planning_mode,
            center_margin_candidates_m=list(candidates),
            initial_center_margin_m=candidates[0],
            selected_center_margin_m=selected,
            window_attempt_count=len(attempts),
            window_fallback_count=max(0, len(attempts)-1),
            window_attempts=deepcopy(attempts), status=status,
            report_statistics_seconds=sum(a['report_statistics_seconds'] for a in attempts),
            verification_seconds=sum(a['verification_seconds'] for a in attempts),
            representation_seconds=sum(a['representation_seconds'] for a in attempts),
            full_route_rebuilt_each_attempt=True,
            route_changed_by_window_policy=False, time_changed_by_window_policy=False,
            physical_limits_changed=False, qp_calls_in_window_policy=0,
            qp_failure_retries=False, global_task_infeasibility_proven=False)
        policy_statistics_seconds += time.perf_counter() - statistics_started
        state['policy_report_statistics_seconds'] = policy_statistics_seconds
        state['report_statistics_seconds'] += policy_statistics_seconds
        return state

    for margin in candidates:
        checkpoint('corridor')
        begin = time.perf_counter()
        try:
            corridors, report = builder(env['source'], route, env['domain'],
                params.ell, config['limits'], center_margin_m=margin,
                seed_tolerance_m=float(settings.get('corridor_seed_tolerance_m', 1e-7)),
                checkpoint=checkpoint, use_system_envelope=use_system_envelope,
                **({'planning_envelope': planning_mode} if planning_mode in ('load_only','bare_point') else {}))
        except CorridorIncompatible as error:
            attempts.append(dict(index=len(attempts), center_margin_m=margin,
                status=admission_failed,
                seconds=time.perf_counter()-begin, failure=error.to_dict(),
                report_statistics_seconds=float(error.diagnostics.get('report_statistics_seconds', 0.)),
                verification_seconds=float(error.diagnostics.get('verification_seconds', 0.)),
                representation_seconds=float(error.diagnostics.get('representation_seconds', 0.))))
            last = margin == candidates[-1]
            state = snapshot(admission_failed if last else 'TRY_SMALLER_WINDOW')
            if emit is not None:
                emit(state)
            if last:
                error.window_policy_report = state
                raise
            continue
        except (Exception, KeyboardInterrupt) as error:
            status=('CORRIDOR_INTERRUPTED' if isinstance(error, KeyboardInterrupt) else
                    'CORRIDOR_EXTERNAL_TIMEOUT' if isinstance(error, TimeoutError) else 'CORRIDOR_ERROR')
            attempts.append(dict(index=len(attempts),center_margin_m=margin,status=status,
                seconds=time.perf_counter()-begin,failure=str(error),exception_type=type(error).__name__,
                report_statistics_seconds=float(getattr(error, 'diagnostics', {}).get('report_statistics_seconds', 0.)),
                verification_seconds=float(getattr(error, 'diagnostics', {}).get('verification_seconds', 0.)),
                representation_seconds=float(getattr(error, 'diagnostics', {}).get('representation_seconds', 0.))))
            state=snapshot(status)
            if emit is not None:emit(state)
            error.window_policy_report=state
            raise
        attempts.append(dict(index=len(attempts), center_margin_m=margin,
            status=admission_pass, seconds=time.perf_counter()-begin,
            report_statistics_seconds=float(report.get('report_statistics_seconds', 0.)),
            verification_seconds=float(report.get('verification_seconds', 0.)),
            representation_seconds=float(report.get('representation_seconds', 0.)),
            corridor_count=len(corridors),
            planning_mode=planning_mode,
            minimum_planning_seed_margin_m=report.get('minimum_planning_seed_margin_m'),
            minimum_fixed_seed_margin_m=report.get('minimum_fixed_seed_margin_m')))
        state = snapshot(admission_pass, margin)
        if emit is not None:
            emit(state)
        # Keep the original builder diagnostics, adding only policy metadata.
        report.update(state)
        report['fallback_calls'] = state['window_fallback_count']
        checkpoint('corridor')
        return corridors, report
    raise AssertionError('C3_WINDOW_POLICY: empty validated candidates')
