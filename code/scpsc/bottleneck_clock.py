"""Experimental, representation-invariant worst-section reference clock.

Exactly K equal-Euclidean-arclength sections. All original bends are retained.
Each section's base time is the integral of the selected velocity-gauge time
along that section, not the norm of its endpoint chord. Assign K * max(base
section times) in total. This is at least the old sum clock for a fixed route.
Global endpoint easing remains a separate, unchanged downstream operation.
This clock is a reference policy, NOT a dynamically time-optimal certificate.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import brentq
from .result_io import save_json
from .velocity_constraints import coerce_velocity, from_config


def _data(route, box, kappa, sections):
    p=np.asarray(route,dtype=float);model=coerce_velocity(box)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<2 or not np.isfinite(p).all():
        raise ValueError('BOTTLENECK: invalid finite XYZ route')
    if not np.isfinite(kappa) or kappa<=0:
        raise ValueError('BOTTLENECK: positive kappa required')
    if not isinstance(sections,(int,np.integer)) or sections<1:
        raise ValueError('BOTTLENECK: fixed positive integer section count required')
    p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-11]]
    d=np.diff(p,axis=0);length=np.linalg.norm(d,axis=1)
    if len(length)==0:raise ValueError('BOTTLENECK: empty path')
    s=np.r_[0.,np.cumsum(length)];base=kappa*model.gauge(d)
    base_cum=np.r_[0.,np.cumsum(base)];cuts=np.linspace(0.,s[-1],sections+1)
    bc=np.interp(cuts,s,base_cum);need=np.diff(bc)  # Eq. (15), demand on each equal-arclength section.
    if np.any(need<=0):raise ValueError('BOTTLENECK: nonpositive section demand')
    return p,s,base,base_cum,cuts,bc,need


def analyze_route(route, box, kappa, sections=26):
    p,s,base,cum,cuts,bc,need=_data(route,box,kappa,sections)
    endpoints=np.column_stack([np.interp(cuts,s,p[:,j]) for j in range(3)])
    imax=int(np.argmax(need));delta=float(need[imax]);total=sections*delta  # Eq. (16).
    report=dict(policy='FIXED_EQUAL_ARCLENGTH_SECTIONS_GLOBAL_MAX',sections=int(sections),
        partition='K equal Euclidean arclength sections of complete exact-endpoint polyline; preserve every bend',
        demand='integral of kappa*velocity_gauge(dp) along each section, NOT endpoint chord',
        velocity_constraint=coerce_velocity(box).metadata(),
        imbalance_beta=float(total/base.sum()),
        total_rule='T=K*max_i(section_base_time_i)',section_base_times_s=need.tolist(),
        section_boundary_positions=endpoints.tolist(),section_boundary_arclength_m=cuts.tolist(),
        longest_section_index0=imax,longest_section_index1=imax+1,longest_section_start=endpoints[imax].tolist(),
        longest_section_end=endpoints[imax+1].tolist(),longest_section_base_time_s=delta,
        uniform_section_assigned_time_s=delta,total_time_s=total,old_sum_time_s=float(base.sum()),
        added_uniform_slack_s=float(total-base.sum()),route_length_m=float(s[-1]),
        arbitrary_collinear_splitting_changes_score=False,not_dynamic_time_optimal=True)
    return report


def scheduled_raw_times(route, box, kappa, sections=26):
    p,s,base,cum,cuts,bc,need=_data(route,box,kappa,sections)
    idx=np.clip(np.searchsorted(cuts,s,side='right')-1,0,sections-1)
    longest=float(need.max());t=(idx+(cum-bc[idx])/need[idx])*longest;t[0]=0.;t[-1]=sections*longest
    if len(p)!=len(route):raise ValueError('Remove zero-length edges before assigning raw clock')
    return t


def bottleneck_reference(route, box, kappa, sections=26, max_edge_m=4.):
    if max_edge_m<=0:raise ValueError('positive subdivision required')
    p,s,base,cum,cuts,bc,need=_data(route,box,kappa,sections)
    all_s=list(cuts)
    for left,right in zip(s[:-1],s[1:]):
        n=max(1,int(np.ceil((right-left)/max_edge_m)))
        all_s.extend(np.linspace(left,right,n+1))
    ss=np.sort(np.asarray(all_s));ss=ss[np.r_[True,np.diff(ss)>1e-10]];ss[0]=0.;ss[-1]=s[-1]
    pts=np.column_stack([np.interp(ss,s,p[:,j]) for j in range(3)])
    # Reinsert each original vertex exactly (no shortcut/refit).
    for j,a in enumerate(s):
        near=int(np.argmin(abs(ss-a)))
        if abs(ss[near]-a)>1e-8:raise AssertionError('Lost original bend')
        pts[near]=p[j]
    cc=np.interp(ss,s,cum);idx=np.clip(np.searchsorted(cuts,ss,side='right')-1,0,sections-1)
    longest=float(need.max());times=(idx+(cc-bc[idx])/need[idx])*longest  # Eq. (17), before endpoint easing.
    times[0]=0.;times[-1]=sections*longest;dt=np.diff(times)
    if np.any(dt<=0) or not np.isfinite(dt).all():raise ValueError('BOTTLENECK: invalid final clock')
    report=analyze_route(route,box,kappa,sections)
    report['subdivision_max_edge_m']=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).max())
    return pts,dt,report


def eased_reference_times(times, r=0.2):
    """Invert the fixed start/stop progress clock, keeping route order and T."""
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or times[0] != 0.0
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("REFERENCE_CLOCK: finite, strictly increasing times from zero required")
    if not np.isfinite(r) or not 0.0 < r < 0.5:
        raise ValueError("REFERENCE_CLOCK: endpoint fraction must lie in (0, 0.5)")
    total = times[-1]
    coefficients = ((5, 126.0), (6, -420.0), (7, 540.0), (8, -315.0), (9, 70.0))

    def integral(u):
        return sum(c * u ** (k + 1) / (k + 1) for k, c in coefficients)

    def progress(s):
        if s < r:
            return r * integral(s / r) / (1.0 - r)
        if s <= 1.0 - r:
            return (s - r / 2.0) / (1.0 - r)
        return 1.0 - r * integral((1.0 - s) / r) / (1.0 - r)

    mapped = np.array(
        [
            0.0
            if t == 0.0
            else total
            if t == total
            else total * brentq(lambda s: progress(s) - t / total, 0.0, 1.0, xtol=1e-14)  # Eq. (17), inverse F_rho.
            for t in times
        ]
    )
    if not np.isfinite(mapped).all() or np.any(np.diff(mapped) <= 0.0):
        raise ValueError("REFERENCE_CLOCK: inverse clock lost finite, strictly increasing times")
    return mapped


def audit(output, route, mapped_times, config):
    output=Path(output);cfg=config['ubsqp'];K=int(cfg['bottleneck_sections']);box=from_config(cfg, config['limits']['vL'])
    with np.load(output/'frontend.npz') as z:raw=z['positions'];native_times=z['times']
    points,dt,report=bottleneck_reference(raw,box,cfg['timing_kappa'],K)
    base=np.r_[0.,np.cumsum(dt)];mapped=eased_reference_times(base,cfg['reference_clock_r'])
    if points.shape!=route.shape or np.max(abs(points-route))>1e-8:raise ValueError('BOTTLENECK_AUDIT: route mismatch')
    error=float(np.max(abs(mapped-mapped_times)))
    if error>1e-8:raise ValueError('BOTTLENECK_AUDIT: final time mismatch')
    olddt=cfg['timing_kappa']*box.gauge(np.diff(raw,axis=0))
    report.update(status='FIXED_SECTION_GLOBAL_MAX_AND_EASING_VERIFIED',final_clock_max_error_s=error,
        edge_reference_velocity_is_constant=True,
        reason='The unchanged Lazy Theta geometry is assigned one K26 reference clock, followed by one fixed monotone easing map.',
        final_mapping_matches_adapter=True,total_time_s=float(mapped[-1]),
        raw_frontend_times=np.asarray(native_times).tolist(),old_search_base_durations_s=olddt.tolist())
    save_json(output/'reference_timing_audit.json',report)
    np.savez_compressed(output/'reference_timing_audit.npz',positions=points,base_times=base,qp_reference_times=mapped,
                        base_durations=dt,qp_reference_durations=np.diff(mapped))
    return report
