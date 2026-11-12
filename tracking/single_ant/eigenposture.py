"""Run the colony posture/state methods on a completed manual-cup dataset."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tracking.single_ant.manual_cup_tracking import write_json
from tracking.single_ant.full_cup_tracking import manifest

REPO = Path(__file__).resolve().parents[2]


def snapshot(destination):
    for folder in ['analysis', 'tracking', 'camera_cal']:
        for source in (REPO/folder).rglob('*'):
            if source.is_file() and source.suffix in {'.py', '.md', '.sh', '.txt'}:
                target = destination/source.relative_to(REPO)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


def initialize(args):
    from analysis.full_behavior_extract import inventory
    source = args.dataset.resolve()
    m = manifest(source/'manifest.json')
    if not (source/'PUBLICATION_COMPLETE.json').exists() or m['fps'] != 24:
        raise ValueError('A completed 24-Hz single-ant dataset is required')
    destination = source/'analysis/eigenposture'
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Existing analysis: {destination}; resume its plan instead')
    base = Path('/flash/ReiterU/ant_tmp')/getpass.getuser()/'single_ant'
    run = base/('eigenposture_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    code, block, output = run/'prep_code', run/'block', run/'packet'
    snapshot(code)
    stitched = block/'stitched'; stitched.mkdir(parents=True)
    (stitched/'per_track').symlink_to(source/'per_track', target_is_directory=True)
    (stitched/'speed_vectors').symlink_to(source/'analysis/speed_vectors', target_is_directory=True)
    write_json(stitched/'block_combination.json', dict(start_datetime=m['start_datetime'],
        num_frames=m['n_frames'], fps=m['fps'], blocks=[dict(name=s['block']) for s in m['recording_segments']]))
    table = inventory(block, output)
    from tracking.single_ant.state_prediction import identity_ok
    table['identity_ok']=[identity_ok(source,int(tid)) for tid in table.track_id]
    table['selected'] &= table.identity_ok
    table['input_rows'] = [pq.ParquetFile(p).metadata.num_rows for p in table.source_path]
    table['exclusion_reason'] = np.where(table.input_rows.eq(0), 'no_pose_rows',
        np.where(~table.identity_ok,'resident_identity_excluded',
                 np.where(table.selected, '', 'position_coverage_not_over_60_percent')))
    table.to_csv(output/'inventory_all.csv', index=False)
    table = table.loc[table.input_rows.gt(0)].reset_index(drop=True)
    table['original_source_path'] = table.source_path
    for i, row in table.loc[table.selected].iterrows():
        table.loc[i, 'source_path'] = str(stitched/'posture_interpolated/per_track'/Path(row.track_name).stem/row.track_name)
    table.to_csv(output/'inventory.csv', index=False)
    write_json(output/'analysis_protocol.json', dict(mode='single_ant', social_features=False,
        body_geometry='calibrated from independent per-ant body-axis samples',
        geometry_rule='Upper axis bound = 2 x 95th percentile of ant-median tag-petiole distances; scale all colony geometry QC bounds by the same factor, at least 1',
        state_cohort='Discovery uses position coverage strictly >60%; prediction also evaluates locally valid full-posture windows outside that cohort',
        interpolation='native bounded node interpolation <=1 second; same camera/recording; no whole-pose or identity filling',
        state_features='native posture and motion features; unmeasured contact columns are NaN and excluded',
        unavailable=['social contacts', 'nest occupancy', 'foraging trips', 'return-contact responses'],
        source=str(source)))
    plan = dict(run=str(run), source=str(source), block=str(block), output=str(output),
        destination=str(destination), code=str(code), python=sys.executable, count=len(table),
        selected=int(table.selected.sum()), cpus=args.cpus, exclude_nodes=args.exclude_nodes,
        prepare_mem=getattr(args, 'prepare_mem', '64G'))
    write_json(run/'plan.json', plan)
    print(json.dumps(plan, indent=2), flush=True)
    return run/'plan.json'


def command(plan, module, *args):
    subprocess.run([plan['python'], '-u', '-m', module, *map(str, args)], check=True)


def geometry_sample(track):
    pf = pq.ParquetFile(track)
    groups = sorted(set([0, pf.num_row_groups//2, pf.num_row_groups-1]))
    values = []
    for index in groups:
        d = pf.read_row_group(index, columns=['Frame', 'Bodypoint', 'X', 'Y']).to_pandas()
        d = d.loc[d.Bodypoint.isin([0, 2])].pivot(index='Frame', columns='Bodypoint', values=['X', 'Y']).dropna()
        if len(d) and all(bp in d['X'] for bp in [0, 2]):
            lengths = np.hypot(d['X'][0]-d['X'][2], d['Y'][0]-d['Y'][2])*.016
            values.extend(lengths[np.isfinite(lengths)&(lengths>.05)].to_numpy())
    if not values:
        raise ValueError(f'No body-axis observations for {track}')
    return dict(track=str(track), samples=len(values), median_mm=float(np.median(values)),
                q01_mm=float(np.quantile(values,.01)), q99_mm=float(np.quantile(values,.99)))


def prepare(plan, index):
    output, block = Path(plan['output']), Path(plan['block'])
    row = pd.read_csv(output/'inventory.csv').iloc[index]
    marker = Path(plan['run'])/'prepared'/f'{index:03d}.json'
    if marker.exists():
        return
    raw = Path(row.original_source_path)
    if row.selected:
        command(plan, 'analysis.compute_track_posture_interpolation', '--track', raw,
                '--out', Path(row.source_path).parent, '--fps', 24, '--max_gap_seconds', 1)
        write_json(output/'geometry_samples'/f'{row.track_id:03d}.json', geometry_sample(raw))
    command(plan, 'analysis.compute_track_sleep_motion', '--track', raw,
            '--out', block/'stitched/sleep_motion/per_track'/Path(row.track_name).stem,
            '--fps', 24, '--mm_per_px', .016)
    write_json(marker, dict(ant=row.ant, selected=bool(row.selected)))


def geometry(plan):
    output = Path(plan['output'])
    shutil.copy2(Path(plan['source'])/'manifest.json', output/'tracking_manifest.json')
    samples = [manifest(p) for p in sorted((output/'geometry_samples').glob('*.json'))]
    if len(samples) != plan['selected']:
        raise ValueError('Geometry sampling did not finish for every selected ant')
    q95 = float(np.quantile([s['median_mm'] for s in samples], .95))
    factor = max(1., 2*q95/1.25)
    config = dict(axis_bounds_mm=[.1*factor, 1.25*factor], max_landmark_mm=5*factor,
                  factor=factor, ant_median_q95_mm=q95, n_ants=len(samples), samples=samples)
    write_json(output/'geometry.json', config)
    print('GEOMETRY', json.dumps(config), flush=True)


def extract(plan, index):
    from analysis.full_behavior_extract import extract_one
    output = Path(plan['output'])
    row = pd.read_csv(output/'inventory.csv').iloc[index]
    if row.selected:
        print(extract_one((Path(plan['block']), output, row.to_dict())), flush=True)


def models(plan):
    from analysis.full_behavior_extract import fit_pca
    from analysis.full_behavior_roles import run as fit_roles
    from analysis.s0_motion_sleep import run as motion_sleep
    output, block = Path(plan['output']), Path(plan['block'])
    table = pd.read_csv(output/'inventory.csv')
    fit_pca(output, table.loc[table.selected])
    for phase in ['features', 'learn', 'assign']:
        command(plan, 'analysis.full_behavior_states', phase, '--root', output, '--workers', plan['cpus'])
    from tracking.single_ant.state_prediction import predict_additional
    predict_additional(plan)
    command(plan, 'analysis.full_behavior_profiles', '--root', output)
    fit_roles(output, side='single')
    motion_sleep(output, block, output/'motion_work', output/'sleep_from_s0_motion_20261011', workers=plan['cpus'])
    report(plan)


def report(plan):
    output, block = Path(plan['output']), Path(plan['block'])
    (output/'COMPLETE.json').unlink(missing_ok=True)
    (Path(plan['run'])/'PUBLISHED.json').unlink(missing_ok=True)
    from analysis import s0_motion_sleep as s0
    from analysis.single_ant_timeline import recording_support
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    support=recording_support(output)
    readout=output/'sleep_from_s0_motion_20261011'
    if manifest(readout/'definition.json').get('recording_support',{}).get('manifest_sha256') != support['manifest_sha256']:
        # Reuse measured raw motion and fitted states; only refresh labels.
        s0.calibrate(output,output/'motion_work',readout)
        with ProcessPoolExecutor(max_workers=plan['cpus']) as pool:
            list(pool.map(partial(s0.label,output,output/'motion_work',readout),range(plan['count'])))
        s0.finalize(output,readout)
    os.environ.update(MPLBACKEND='Agg', ANTS_DATASET_ROOT=str(block), ANTS_BEHAVIOR_SOURCE=str(output),
        ANTS_COLONY_SIDE='single', ANTS_ROLE_MODELS=str(output/'roles'), ANTS_ROLE_OUTPUT=str(output/'report'))
    command(plan, 'analysis.eigenposture_task_report', '--source', output, '--output', output/'report', '--single-ant')
    if not (output/'report/single_behavior_classes.pdf').is_file():
        raise FileNotFoundError('Full single-ant report was not produced')
    from tracking.single_ant.state_labels import export_labels
    s01=export_labels(output,Path(plan['source']))
    snapshot(output/'code')
    launcher = output/'run_report.sh'
    launcher.write_text('#!/bin/bash\nset -euo pipefail\n'+
        'packet="$(cd -- "$(dirname -- "$0")" && pwd)"\n'+
        'dataset="$(dirname -- "$(dirname -- "$packet")")"\n'+
        'export MPLBACKEND=Agg ANTS_DATASET_ROOT="$dataset" ANTS_COLONY_SIDE=single\n'+
        'export ANTS_BEHAVIOR_SOURCE="$packet" ANTS_ROLE_MODELS="$packet/roles"\n'+
        'export ANTS_MOTION_SLEEP="$packet/sleep_from_s0_motion_20261011"\n'+
        'cd "$packet/code"\n'+shlex.quote(plan['python'])+
        ' -m analysis.eigenposture_task_report --source "$packet" --output "$packet/report" --single-ant\n')
    launcher.chmod(0o755)
    write_json(output/'COMPLETE.json', dict(source=plan['source'], cohort=plan['selected'], side='single',
        report=str(output/'report/single_behavior_classes.pdf'), full_colony_methods=True,
        report_sha256=hashlib.sha256((output/'report/single_behavior_classes.pdf').read_bytes()).hexdigest(),
        social_features=False, state_interpretation=manifest(output/'state_interpretation.json'),
        s01_labels=s01,
        unavailable=manifest(output/'analysis_protocol.json')['unavailable']))


def submit(plan_path, stage):
    plan = manifest(plan_path); run = Path(plan['run'])
    jobs_path = run/'jobs.json'; jobs = manifest(jobs_path) if jobs_path.exists() else {}
    if stage in ['continue','retry-report']:
        code = run/('model_code' if stage=='continue' else 'report_code_'+datetime.now(timezone.utc).strftime('%H%M%S%f')); snapshot(code)
        plan['model_code'] = str(code); write_json(plan_path, plan)
    if stage=='retry-report' and 'models_report' in jobs:
        path=run/'job_history.json';history=manifest(path) if path.exists() else []
        history.append(dict(job=jobs.pop('models_report'),stage='models_report',note='Report retry; retain fitted models'))
        write_json(path,history);write_json(jobs_path,jobs)
    def job(name, phase, deps=(), array=None, mem='64G', cpus=2, wall='02:00:00'):
        if name in jobs:
            return jobs[name]
        mem = plan.get(name+'_mem', mem)
        wall = plan.get(name+'_wall', wall)
        code = plan['code'] if phase=='prepare' else plan['model_code']
        folder = run/'jobs';folder.mkdir(exist_ok=True);(run/'logs').mkdir(exist_ok=True)
        path = folder/(name+'.sbatch')
        cmd = shlex.join([plan['python'], '-u', '-m', 'tracking.single_ant.eigenposture', phase, '--plan', str(plan_path)])
        if array is not None:cmd+=' --index "$SLURM_ARRAY_TASK_ID"'
        path.write_text(f'''#!/bin/bash
#SBATCH -J single_eigen_{name}
#SBATCH -p compute
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {wall}
#SBATCH --exclude={plan['exclude_nodes']}
#SBATCH -o {run}/logs/{name}_%A_%a.out
#SBATCH -e {run}/logs/{name}_%A_%a.err
set -euo pipefail
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS={max(cpus,4)}
cd {shlex.quote(str(code))}
{cmd}
''')
        args = ['sbatch', '--parsable']
        if array is not None:args += ['--array='+array]
        pending = []
        for dep in deps:
            state = subprocess.check_output(['sacct','-X','-n','-P','-j',dep,'--format=State'],text=True).splitlines()
            if not state or any(s.strip('| ')!='COMPLETED' for s in state):pending.append(dep)
        if pending:args += ['--dependency=afterok:'+':'.join(pending)]
        jobs[name] = subprocess.check_output(args+[str(path)],text=True).strip().split(';')[0]
        write_json(jobs_path,jobs)
        return jobs[name]
    prepared = job('prepare','prepare',array=f"0-{plan['count']-1}",mem=plan.get('prepare_mem','64G'))
    if stage == 'continue':
        calibrated = job('geometry','geometry',[prepared],mem='8G',cpus=1)
        extracted = job('extract','extract',[calibrated],array=f"0-{plan['count']-1}",mem='16G',cpus=2)
        job('models_report','models',[extracted],mem='128G',cpus=plan['cpus'],wall='2-00:00:00')
    elif stage=='retry-report':
        job('models_report','report',mem='64G',cpus=plan['cpus'],wall='04:00:00')
    print(json.dumps(dict(plan=str(plan_path),jobs=jobs,destination=plan['destination']),indent=2),flush=True)


def publish(plan):
    """Run on the login node, whose bucket mount permits writes."""
    import fcntl
    run, output = Path(plan['run']), Path(plan['output'])
    with (run/'publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run/'PUBLISHED.json').exists(): return
        while not (output/'COMPLETE.json').is_file():
            jobs = manifest(run/'jobs.json')
            accounting = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j',
                ','.join(jobs.values()), '--format=JobID,State,ExitCode'], text=True)
            failures = [s for s in accounting.splitlines() if any(x in s for x in
                ['FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','CANCELLED','DEPENDENCY'])]
            if failures: raise RuntimeError('Eigenposture jobs failed: '+str(failures))
            print('Waiting for complete model packet', flush=True)
            time.sleep(30)
        destination = Path(plan['destination'])
        transfers = [(Path(plan['block'])/'stitched'/name, destination.parent/name)
                     for name in ['posture_interpolated', 'sleep_motion']]
        transfers.append((output, destination))
        transfers.append((output/'s01_labels',destination.parent/'sleep_s01_exclusion'))
        for source, target in transfers:
            if any(p.is_symlink() for p in source.rglob('*')):
                raise ValueError(f'Unexpected symlink in published analysis: {source}')
            target.mkdir(parents=True, exist_ok=True)
            subprocess.run(['rsync','-rt','--no-perms','--omit-dir-times',
                '--exclude=COMPLETE.json', str(source)+'/', str(target)+'/'], check=True)
        # Keep provenance/signature bytes unchanged; document their scratch paths.
        speed_metadata=destination.parent/'speed_by_metadata';speed_metadata.mkdir(exist_ok=True)
        shutil.copyfile(output/'s01_labels/per_ant_summary.csv',speed_metadata/'per_ant_summary.csv')
        for target in [destination/'s01_labels',destination.parent/'sleep_s01_exclusion']:
            shutil.copyfile(output/'s01_labels/COMPLETE.json',target/'COMPLETE.json')
        write_json(destination/'publication.json', dict(plan=plan, published=datetime.now(timezone.utc).isoformat(),
            paths={str(s):str(t) for s,t in transfers}, provenance_paths='Original compute paths preserved'))
        shutil.copyfile(output/'COMPLETE.json', destination/'COMPLETE.json')
        write_json(run/'PUBLISHED.json', dict(output=str(destination), report=str(destination/'report/single_behavior_classes.pdf')))
        print('Published '+str(destination), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['submit','continue','retry-report','prepare','geometry','extract','models','report','publish'])
    p.add_argument('dataset',type=Path,nargs='?');p.add_argument('--plan',type=Path)
    p.add_argument('--index',type=int,default=0);p.add_argument('--cpus',type=int,default=16)
    p.add_argument('--prepare-mem',default='64G',help='Memory per preparation worker, e.g. 128G for long combined tracks')
    p.add_argument('--exclude-nodes',default='deigo011019');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    if a.stage=='submit':
        path=initialize(a);submit(path,'prepare' if a.prepare_only else 'continue')
    elif a.stage in ['continue','retry-report']:submit(a.plan,a.stage)
    else:globals()[a.stage](manifest(a.plan),a.index) if a.stage in ['prepare','extract'] else globals()[a.stage](manifest(a.plan))


if __name__=='__main__':main()
