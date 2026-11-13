"""Pure geometric native Lazy Theta*: h=1, no route adjustment or force queries."""
from pathlib import Path
import os,json,subprocess,time,warnings
import numpy as np
from .frontend_grid import build_voxel_grid, save_grid
from .velocity_constraints import from_config
from .envelope import planning_envelope_mode

ROOT=Path(__file__).resolve().parents[2]

def search_lazytheta(env,params,config,output,checkpoint):
    begin=time.perf_counter();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    checkpoint('preparation')
    grid,meta=build_voxel_grid(env['source'],env['domain'],config,checkpoint=checkpoint)
    grid_file=save_grid(out/'prepared_grid',grid,meta)
    scene=config['mechanism_scene'];anchors=np.asarray(scene.get('fixed_anchors',[env['start'],env['goal']]),float)
    if anchors.shape!=(2,3) or not np.isfinite(anchors).all():raise ValueError('fixed_anchors must be two XYZ positions')
    # Registered off-grid physical endpoints keep their exact connector segments.
    # Check the same planning envelope as the voxel grid; no anchor search.
    connectors=[]
    planning_mode=planning_envelope_mode(config['ubsqp'])
    front_mode=planning_mode if planning_mode in ('load_only','bare_point') else 'fixed_system'
    from .qp_corridor import build_fixed_envelope_corridors
    for p,a in zip((env['start'],env['goal']),anchors):
        if np.linalg.norm(p-a)<=1e-10:connectors.append(None);continue
        _,proof=build_fixed_envelope_corridors(env['source'],np.asarray([p,a]),env['domain'],params.ell,config['limits'],
            center_margin_m=config['ubsqp']['corridor_center_margin_m'],checkpoint=checkpoint,
            plane_options=config['ubsqp'].get('corridor_options',{}).get('plane_solver',{}),
            use_system_envelope=True,planning_envelope=front_mode)
        connectors.append(proof)
    preparation_seconds=time.perf_counter()-begin
    executable=ROOT/'native/build'/('lazytheta_runner.exe' if os.name=='nt' else 'lazytheta_runner')
    if not executable.is_file():raise RuntimeError('LAZY_NATIVE_DEPENDENCY_UNAVAILABLE: run scripts/build_lazytheta.py')
    request=out/'request.txt';route_file=out/'route.txt';status_file=out/'native_status.json'
    if config['frontend'].get('native_budget_s') is not None:
        warnings.warn('frontend.native_budget_s is deprecated and ignored; search has no wall-clock limit', FutureWarning)
    checkpoint('frontend')
    with request.open('w',encoding='utf-8',newline='\n') as f:
        f.write(str(grid_file.resolve())+'\n')
        f.write(f"{meta['nx']} {meta['ny']} {meta['nz']} {meta['h']:.17g}\n")
        for p in (meta['origin'],anchors[0],anchors[1]):f.write(' '.join(format(float(x),'.17g') for x in p)+'\n')
        f.write('0\n')
    started=time.perf_counter()
    process=subprocess.run([str(executable),str(request),str(route_file),str(status_file)],
            capture_output=True,text=True)
    (out/'native.log').write_text(process.stdout+process.stderr,encoding='utf-8')
    report=json.loads(status_file.read_text()) if status_file.exists() else {'status':'NATIVE_CRASH'}
    if process.returncode or report['status']!='ROUTE_FOUND':
        raise ValueError('FRONTEND_'+report['status']+': '+report.get('detail',''))
    route=np.loadtxt(route_file).reshape(-1,3)
    if np.linalg.norm(route[0]-env['start'])>1e-10:route=np.vstack((env['start'],route))
    if np.linalg.norm(route[-1]-env['goal'])>1e-10:route=np.vstack((route,env['goal']))
    route=route[np.r_[True,np.linalg.norm(np.diff(route,axis=0),axis=1)>1e-10]]
    if len(route)<2:raise ValueError('FRONTEND_EMPTY_ROUTE')
    model=from_config(config['ubsqp'],config['limits']['vL'])
    times=np.r_[0.,np.cumsum(model.gauge(np.diff(route,axis=0)))]
    report.update(algorithm='LAZY_THETA_STAR',heuristic_weight=1,R_enabled=False,R_calls=0,
        grid_regenerated=True,physical_model='NO_AERODYNAMIC_TAUT_CABLE',planning_envelope=front_mode,
        requested_planning_envelope=planning_mode,legacy_point_frontend=(planning_mode=='point'),
        preparation_seconds=preparation_seconds,native_wall_seconds=time.perf_counter()-started,
        endpoint_connectors=connectors,grid=meta,route_point_count=len(route))
    checkpoint('reference')
    return route,times,report
