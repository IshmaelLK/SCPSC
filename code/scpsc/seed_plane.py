"""Maximum certified normal-margin lower plane: seven-variable, two-cone SOCP.

The final plane always covers every original clipped source vertex. Admission
uses the selected planning support; true-system margins are reported separately.
No bisection, LP tangent approximation, field mask or aerodynamic model is used.
The two selection ablations retain Clarabel: vertical_margin fixes n_z=-1;
point_then_shrink selects with zero support, then certifies the chosen plane
with the original full-system support. Both retain true-envelope admission.
horizontal_max uses the highest complete-window source vertex directly,
without a solver or vertex filtering, and retains true-envelope admission.
"""
from dataclasses import asdict
from time import perf_counter
import warnings
import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull, QhullError
from .envelope import effective_radii, fixed_support, planning_envelope_parameters


from .plane_contract import PlaneOptions, SeedPlaneFailure, SeedPlaneSolverError
from .plane_constraint_generation import solve_generated
from .source_clipping import clipped_source_vertices


def downward_support(n,ell,rhoL,rhoQ,theta):
    """Closed-form exact support for n_z<=0, including n=0."""
    n=np.asarray(n,float)
    if n.shape[-1:]!=(3,) or not np.isfinite(n).all() or np.any(n[...,2]>1e-12):
        raise ValueError('A finite downward normal is required')
    norm=np.linalg.norm(n,axis=-1)
    cap=np.sin(theta)*np.linalg.norm(n[...,:2],axis=-1)+np.cos(theta)*n[...,2]
    return np.maximum(rhoL*norm,rhoQ*norm+ell*cap)


def axis_box_vertices(ell,limits):
    """Eight corners of the true fixed-envelope box from its six axial supports."""
    rhoL,rhoQ=effective_radii(limits)
    h=fixed_support(np.vstack((np.eye(3),-np.eye(3))),ell,rhoL,rhoQ,limits['theta'])
    lo,hi=-h[3:],h[:3]
    return np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1])
                     for z in (lo[2],hi[2])])


def _inputs(vertices,seeds,ell,limits):
    V,S=np.asarray(vertices,float),np.asarray(seeds,float)
    if V.ndim!=2 or V.shape[1]!=3 or len(V)<3 or S.shape!=(2,3):
        raise ValueError('Expected source vertices [N>=3,3] and two seeds [2,3]')
    if not np.isfinite(V).all() or not np.isfinite(S).all():
        raise ValueError('Unknown/nonfinite source or seeds cannot be certified')
    rhoL,rhoQ=effective_radii(limits);theta=float(limits['theta'])
    if not np.isfinite([ell,rhoL,rhoQ,theta]).all() or min(ell,rhoL,rhoQ)<=0 or not 0<=theta<np.pi/2:
        raise ValueError('Positive dimensions and cap angle in [0,pi/2) required')
    return V,S,float(rhoL),float(rhoQ),theta


def filter_vertices(vertices,mode):
    """Select actual source indices, never perturbed Qhull coordinates."""
    V=np.asarray(vertices,float);start=perf_counter();note=None
    indices=np.arange(len(V))
    if mode!='all' and len(V)>4:
        try:
            hull=ConvexHull(V-V.mean(axis=0))  # deliberately no QJ coordinate joggling
            if mode=='hull': indices=hull.vertices
            elif mode=='upper_hull':
                # Include vertical boundary faces so near-horizontal queries remain covered.
                indices=np.unique(hull.simplices[hull.equations[:,2]>=-1e-12])
                if not len(indices):raise ValueError('Empty upper hull')
            else:raise ValueError('Unknown filter mode')
        except (QhullError,ValueError) as error:
            indices=np.arange(len(V));note=type(error).__name__+': using all original points'
    return np.asarray(indices,dtype=int),dict(mode=mode,input_count=len(V),selected_count=len(indices),
        fallback=note,seconds=perf_counter()-start)


def assemble_socp(vertices,seeds,ell,limits,use_system_envelope=True,*,vertical_margin=False,
                  planning_envelope=None,envelope_approximation='exact'):
    """Return min q'x s.t. Ax+s=b, s in R_+^m x SOC4 x SOC3.

    x=(nx,ny,nz,beta,t,u,r). beta is the intercept in seed-centred coordinates.
    Cone dimensions are returned as integers, so this assembly is testable without
    importing a native runtime. P is identically zero.
    """
    V,S,rhoL,rhoQ,theta=_inputs(vertices,seeds,ell,limits)
    origin=S.mean(0);V=V-origin;S=S-origin
    planning_mode=planning_envelope or ('fixed_system' if use_system_envelope else 'point')
    if envelope_approximation not in ('exact','axis_box'):
        raise ValueError('envelope_approximation must be exact or axis_box')
    if envelope_approximation=='axis_box' and (planning_mode!='fixed_system' or vertical_margin):
        raise ValueError('axis_box requires fixed_system and normal_margin')
    ell,rhoL,rhoQ,theta=planning_envelope_parameters(ell,limits,planning_mode)
    source_count=len(V)
    seed_rows=16 if envelope_approximation=='axis_box' else 4
    m=source_count+seed_rows+3
    matrix=np.zeros((m+7,7));rhs=np.zeros(m+7)
    # beta - n'v <= 0
    matrix[:source_count,:3]=-V
    matrix[:source_count,3]=1.
    seed_matrix=matrix[source_count:source_count+seed_rows]
    seed_matrix[:,3]=-1.;seed_matrix[:,4]=1.
    if envelope_approximation=='axis_box':
        # h_B(n)=max_c n.c: all eight actual box corners give linear seed rows.
        corners=axis_box_vertices(ell,limits)
        seed_matrix[:,:3]=(S[:,None,:]+corners[None,:,:]).reshape(16,3)
    else:
        seed_matrix[::2,:3]=S;seed_matrix[1::2,:3]=S
        seed_matrix[1::2,2]+=ell*np.cos(theta)
        seed_matrix[::2,5]=rhoL;seed_matrix[1::2,5]=rhoQ
        seed_matrix[1::2,6]=ell*np.sin(theta)
    domain=matrix[m-3:m]
    if vertical_margin:
        # n_z=-1 replaces u<=1 and n_z<=0. The two inequalities encode
        # this equality in the existing nonnegative cone; u has no upper bound.
        domain[0,2]=1.;domain[1,2]=-1.
        rhs[m-3:m-1]=(-1.,1.)
    else:
        domain[0,5]=1.;domain[1,2]=1.;rhs[m-3]=1.
    domain[2,4]=-1.
    # -Ax=(u,nx,ny,nz) and (r,nx,ny) are SOC slacks.
    C=matrix[m:];C[0,5]=-1;C[1,0]=-1;C[2,1]=-1;C[3,2]=-1
    C[4,6]=-1;C[5,0]=-1;C[6,1]=-1
    A=sparse.csc_matrix(matrix)
    q=np.zeros(7);q[4]=-1
    return sparse.csc_matrix((7,7)),q,A,rhs,(m,4,3),origin


def certify_normal(normal,vertices,seeds,ell,limits,eps_m=1e-8,use_system_envelope=True,
                   planning_envelope=None,envelope_approximation='exact'):
    """Recompute the intercept in local coordinates from the complete source."""
    V,S,rhoL,rhoQ,theta=_inputs(vertices,seeds,ell,limits)
    n=np.asarray(normal,float);length=float(np.linalg.norm(n))
    if n.shape!=(3,) or not np.isfinite(n).all() or length<=1e-10:
        raise ValueError('Zero/nonfinite normal cannot define a plane')
    n=n/length
    if n[2]>=-1e-10:raise ValueError('Lower plane must have a strictly downward z component')
    core_started=perf_counter()
    origin=S.mean(0);v=V-origin;s=S-origin
    source_projection=v@n
    b_local=float(np.min(source_projection)-eps_m)
    intercept_core_seconds=perf_counter()-core_started
    true_support=float(fixed_support(n[None],ell,rhoL,rhoQ,theta)[0])
    planning_mode=planning_envelope or ('fixed_system' if use_system_envelope else 'point')
    if envelope_approximation=='axis_box':
        if planning_mode!='fixed_system':
            raise ValueError('axis_box requires fixed_system and normal_margin')
        corners=axis_box_vertices(ell,limits)
        support=float(np.max(corners@n))
    elif envelope_approximation=='exact':
        support=(true_support if planning_mode=='fixed_system' else
                 float(fixed_support(n[None],*planning_envelope_parameters(ell,limits,planning_mode))[0]))
    else:
        raise ValueError('envelope_approximation must be exact or axis_box')
    seed_projection=s@n
    margins=b_local-seed_projection-support
    true_margins=b_local-seed_projection-true_support
    b_world=float(b_local+n@origin)
    slope=n[:2]/(-n[2]);beta=b_world/n[2]
    minimum_margin=float(margins.min())
    statistics_started=perf_counter()
    source_cover_min=float(np.min(source_projection-b_local))
    proof=dict(unit_normal=n.tolist(),unit_bound_m=b_world,
        local_origin=origin.tolist(),local_unit_bound_m=b_local,
        source_cover_min_m=source_cover_min,
        source_majorant_min=source_cover_min/(-n[2]),
        fixed_seed_margins_m=true_margins.tolist(),fixed_margin_m=float(true_margins.min()),
        planning_minimum_margin_m=minimum_margin,minimum_normal_margin_m=minimum_margin,
        support_m=support,source_vertex_count=len(V),
        intercept_core_seconds=intercept_core_seconds)
    if envelope_approximation=='axis_box':
        proof.update(envelope_approximation='axis_box',true_support_m=true_support,
            box_support_m=support,box_lower_m=corners.min(axis=0).tolist(),
            box_upper_m=corners.max(axis=0).tolist(),box_seed_margins_m=margins.tolist(),
            box_minimum_margin_m=float(margins.min()))
    proof['report_statistics_seconds']=perf_counter()-statistics_started
    return slope,float(beta),proof


def _clarabel_solve(data,options):
    try:
        import clarabel
    except ImportError as error:
        raise SeedPlaneSolverError('CLARABEL_DEPENDENCY_UNAVAILABLE: install clarabel==0.11.1') from error
    P,q,A,b,dims,origin=data
    settings=clarabel.DefaultSettings()
    settings.verbose=False;settings.max_threads=1
    settings.max_iter=options.max_iter
    # Clarabel's native infinity disables its time limit; JSON records use null.
    settings.time_limit=float('inf')
    settings.tol_gap_abs=options.tol;settings.tol_gap_rel=options.tol;settings.tol_feas=options.tol
    wall_start=perf_counter()
    solver=clarabel.DefaultSolver(P,q,A,b,[clarabel.NonnegativeConeT(dims[0]),
        clarabel.SecondOrderConeT(4),clarabel.SecondOrderConeT(3)],settings)
    solution=solver.solve()
    solve_wall=perf_counter()-wall_start
    statistics_started=perf_counter()
    record=dict(runtime='clarabel',version=clarabel.__version__,status=str(solution.status),
        iterations=int(solution.iterations),solve_seconds=float(solution.solve_time),
        native_setup_and_solve_wall_seconds=solve_wall,
        primal_objective=float(solution.obj_val),dual_objective=float(solution.obj_val_dual))
    record['report_statistics_seconds']=perf_counter()-statistics_started
    return solution,record


def seed_plane(vertices,seeds,ell,limits,use_system_envelope=True,*,options=None,deadline=None,checkpoint=None,
               planning_envelope=None):
    options=options if isinstance(options,PlaneOptions) else PlaneOptions(**(options or {}))
    if deadline is not None:
        warnings.warn('seed_plane deadline is deprecated and ignored', FutureWarning)
    mode=options.plane_selection_mode
    planning_mode=planning_envelope or ('fixed_system' if use_system_envelope else 'point')
    generated=options.vertex_filter=='constraint_generation'
    if generated and planning_mode!='fixed_system':
        raise ValueError('constraint_generation requires fixed_system')
    if options.envelope_approximation=='axis_box' and planning_mode!='fixed_system':
        raise ValueError('axis_box requires fixed_system and normal_margin')
    if mode!='normal_margin' and planning_mode!='fixed_system':
        raise ValueError('Plane-selection ablations require true-envelope admission and QP erosion')
    selection_mode='point' if mode=='point_then_shrink' else planning_mode
    selection_envelope=selection_mode=='fixed_system'
    def selection_score(candidate):
        proof=candidate[2]
        margin=proof['minimum_normal_margin_m']
        return margin/(-proof['unit_normal'][2]) if mode=='vertical_margin' else margin
    V,S,*_=_inputs(vertices,seeds,ell,limits);begin=perf_counter()
    def progress():
        if checkpoint is not None:checkpoint('corridor')
    progress()
    if mode=='horizontal_max' or generated:
        # Neither horizontal height selection nor constraint generation uses a hull.
        ids=np.arange(len(V))
        filter_info=dict(mode='all',requested_mode=options.vertex_filter,
                         input_count=len(V),selected_count=len(V),fallback=None,seconds=0.)
    else:
        ids,filter_info=filter_vertices(V,options.vertex_filter)
    ct=perf_counter()
    horizontal=certify_normal([0,0,-1],V,S,ell,limits,options.certificate_eps_m,selection_envelope,
                              planning_envelope=selection_mode,envelope_approximation=options.envelope_approximation)
    best=horizontal;choice='horizontal';solve_records=[];assembly_s=0.;certification_s=perf_counter()-ct
    statistics_s=horizontal[2]['report_statistics_seconds']
    generated_proof=None
    try:
        for attempt in range(0 if mode=='horizontal_max' else (1 if generated else 2)):
            progress();at=perf_counter()
            if generated:
                data,sol,solve_records,filter_info,assembly_s,generated_proof=solve_generated(
                    V,S,ell,limits,options,assemble_socp,_clarabel_solve,checkpoint)
                record=solve_records[-1]
                statistics_s+=generated_proof['report_statistics_seconds']
            else:
                data=assemble_socp(V[ids],S,ell,limits,selection_envelope,
                               vertical_margin=mode=='vertical_margin',
                               planning_envelope=selection_mode,
                               envelope_approximation=options.envelope_approximation);assembly_s+=perf_counter()-at
                sol,record=_clarabel_solve(data,options);solve_records.append(record)
                statistics_s+=record.get('report_statistics_seconds',0.)
            if mode=='vertical_margin' and record['status']=='PrimalInfeasible':
                # Fixed n_z removes the homogeneous zero solution. An infeasible
                # nonnegative vertical gap is a corridor failure, allowing C3.
                if horizontal[2]['minimum_normal_margin_m']>=-options.seed_tolerance_m:
                    raise SeedPlaneSolverError('SOCP_UNRESOLVED: infeasible status conflicts with horizontal certificate')
                proof=dict(horizontal[2],method='vertical_margin_socp',plane_selection_mode=mode,
                    selected_candidate=None,optimization_records=solve_records,options=asdict(options),
                    message='vertical-margin SOCP primal infeasible with t>=0 in this window',
                    timing=dict(report_statistics_seconds=statistics_s))
                raise SeedPlaneFailure(proof)
            if record['status'] not in ('Solved','AlmostSolved','MaxIterations','InsufficientProgress'):
                raise SeedPlaneSolverError('SOCP_UNRESOLVED: '+record['status'])
            x=np.asarray(sol.x,float);ct=perf_counter()
            candidate=None;original_violation=0.
            if x.shape==(7,) and np.isfinite(x).all() and np.linalg.norm(x[:3])>1e-10 and x[2]<-1e-10:
                original_violation=(generated_proof['raw_source_violation_m'] if generated else
                    max(0.,float(x[3]-np.min((V-data[-1])@x[:3])))/np.linalg.norm(x[:3]))
                candidate=certify_normal(x[:3],V,S,ell,limits,options.certificate_eps_m,selection_envelope,
                                         planning_envelope=selection_mode,envelope_approximation=options.envelope_approximation)
                statistics_s+=candidate[2]['report_statistics_seconds']
                if selection_score(candidate)>selection_score(best):
                    best=candidate;choice='socp'
            certification_s+=perf_counter()-ct
            # A reduced floating-point hull is only an optimization aid. Repair the
            # constraint set, not the original geometry, if omitted points matter.
            if original_violation>options.seed_tolerance_m and len(ids)<len(V) and attempt==0:
                ids=np.arange(len(V));filter_info['full_point_retry']=True;continue
            break
    except SeedPlaneSolverError as error:
        diagnostics=dict(getattr(error,'diagnostics',{}))
        diagnostics['report_statistics_seconds']=statistics_s+diagnostics.get('report_statistics_seconds',0.)
        error.diagnostics=diagnostics
        raise
    progress()
    score=selection_score(best)
    selection_normal_margin=best[2]['minimum_normal_margin_m']
    if mode=='point_then_shrink':
        # The point objective has already selected the normal. Certify and
        # admit that same plane using the true support, without selecting again.
        ct=perf_counter()
        best=certify_normal(best[2]['unit_normal'],V,S,ell,limits,
                            options.certificate_eps_m,True)
        statistics_s+=best[2]['report_statistics_seconds']
        certification_s+=perf_counter()-ct
    slope,beta,proof=best
    statistics_started=perf_counter()
    proof.update(method=mode if mode=='horizontal_max' else mode+'_socp',physical_model='NO_AERODYNAMIC_TAUT_CABLE',
        plane_selection_mode=mode,selection_uses_system_envelope=selection_envelope,
        selection_margin_m=score,selection_normal_margin_m=selection_normal_margin,
        selection_margin_definition='vertical' if mode=='vertical_margin' else 'normal',
        actual_vertical_margin_m=proof['minimum_normal_margin_m']/(-proof['unit_normal'][2]),
        planning_mode=planning_mode,selected_candidate=choice,
        horizontal_margin_m=(horizontal[2]['fixed_margin_m'] if planning_mode=='fixed_system'
                             else horizontal[2]['minimum_normal_margin_m']),
        horizontal_selection_margin_m=selection_score(horizontal),filter=filter_info,
        original_points_certified=True,optimization_records=solve_records,options=asdict(options),
        solver_calls=len(solve_records),iterations=sum(x['iterations'] for x in solve_records),
        optimality_status=('EXACT_HORIZONTAL_MAX' if mode=='horizontal_max' else
                           'NUMERICAL_SOLVED' if solve_records[-1]['status']=='Solved' and choice=='socp'
                           else 'CERTIFIED_CANDIDATE_ONLY'),
        timing=dict(filter_seconds=filter_info['seconds'],assembly_seconds=assembly_s,
                    solver_seconds=sum(x['solve_seconds'] for x in solve_records),
                    native_setup_and_solve_wall_seconds=sum(x.get('native_setup_and_solve_wall_seconds',x['solve_seconds']) for x in solve_records),
                    certification_seconds=certification_s,
                    certification_excluding_core_seconds=(certification_s-horizontal[2]['intercept_core_seconds']
                        if mode=='horizontal_max' else certification_s),
                    core_method_seconds=(horizontal[2]['intercept_core_seconds'] if mode=='horizontal_max'
                        else sum(x.get('native_setup_and_solve_wall_seconds',x['solve_seconds']) for x in solve_records)),
                    core_method_scope=('source vertex scan and closed-form intercept' if mode=='horizontal_max'
                        else 'Clarabel setup and solve wall time; solver_seconds is nested'),
                    total_seconds=perf_counter()-begin))
    if generated_proof is not None:
        proof['constraint_generation']=generated_proof
        proof['timing']['constraint_scan_seconds']=generated_proof['scan_seconds']
    proof['timing']['report_statistics_seconds']=statistics_s+perf_counter()-statistics_started
    verification_started=perf_counter()
    seed_incompatible=proof['minimum_normal_margin_m'] < -options.seed_tolerance_m
    proof['timing']['verification_seconds']=perf_counter()-verification_started
    if seed_incompatible:
        if mode not in ('point_then_shrink','horizontal_max') and solve_records[-1]['status']!='Solved':
            error=SeedPlaneSolverError('SOCP_UNRESOLVED: no certified seed-compatible candidate')
            error.diagnostics=dict(verification_seconds=proof['timing']['verification_seconds'],
                report_statistics_seconds=proof['timing']['report_statistics_seconds'])
            raise error
        message=('point-selected lower plane fails true-envelope tightening in this window'
                 if mode=='point_then_shrink' else
                 'window-maximum horizontal plane fails true-envelope seed inclusion in this window'
                 if mode=='horizontal_max' else 'no certified seed-compatible lower plane in this window')
        raise SeedPlaneFailure(dict(proof,message=message))
    return slope,beta,proof


def terrain_seed_plane(terrain,lo,hi,seeds,ell,limits,eta=0.,use_system_envelope=True,
                       *,options=None,deadline=None,checkpoint=None,planning_envelope=None):
    start=perf_counter()
    if checkpoint is not None:checkpoint('corridor')
    V,meta=clipped_source_vertices(terrain,lo,hi,eta);query_s=perf_counter()-start
    try:
        slope,beta,proof=seed_plane(V,seeds,ell,limits,use_system_envelope,
                                  options=options,deadline=deadline,checkpoint=checkpoint,
                                  planning_envelope=planning_envelope)
    except SeedPlaneFailure as error:
        error.proof.update(meta)
        timing=error.proof.setdefault('timing',{})
        timing['source_query_clip_seconds']=query_s
        if 'total_seconds' in timing:timing['total_seconds']+=query_s
        raise
    statistics_started=perf_counter()
    proof.update(meta);proof['timing']['source_query_clip_seconds']=query_s
    proof['timing']['total_seconds']+=query_s
    proof['timing']['report_statistics_seconds']+=perf_counter()-statistics_started
    return slope,beta,proof
