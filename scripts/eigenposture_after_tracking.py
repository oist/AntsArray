#!/usr/bin/env python3
"""Submit the standard eigenposture report once tracking inputs are complete.

The lightweight watcher runs on the login node; computation runs in Slurm.
The colony transfer watcher publishes the report only after its success marker.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


MODULES = ['numpy', 'scipy', 'pandas', 'pyarrow', 'matplotlib', 'sklearn',
           'umap', 'numba', 'igraph', 'leidenalg', 'shapely']


def check_environment(python):
    code = ('import importlib; from sklearn.cluster import HDBSCAN; '
            '[importlib.import_module(m) for m in ' + repr(MODULES) + ']')
    subprocess.run([str(python), '-c', code], check=True,
                   env={**os.environ, 'PYTHONNOUSERSITE': '1', 'MPLBACKEND': 'Agg'})


def ready(paths, epoch):
    return all(p.is_file() and p.stat().st_size > 0 and p.stat().st_mtime >= epoch for p in paths)


def watch(config_path):
    config = json.loads(config_path.read_text())
    with config_path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = [Path(p) for p in config['prerequisites']]
        submitted = config_path.with_name('eigenposture_job_id.txt')
        if submitted.exists() and submitted.stat().st_mtime >= config['epoch']:
            print('Already submitted:', submitted.read_text().strip(), flush=True)
            return
        while not ready(paths, config['epoch']):
            missing = [str(p) for p in paths if not ready([p], config['epoch'])]
            print(time.strftime('%F %T'), 'Waiting for', missing, flush=True)
            if time.time() - config['epoch'] > 14 * 86400:
                raise TimeoutError('Eigenposture prerequisites did not finish within 14 days')
            time.sleep(config['poll_seconds'])
        job = subprocess.check_output([config['sbatch_bin'], '--parsable', config['job_script']], text=True).strip()
        if not job.split(';')[0].isdigit():
            raise ValueError(f'Invalid Slurm job ID: {job!r}')
        submitted.write_text(job + '\n')
        print('Submitted eigenposture_interactive:', job, flush=True)


def prepare(args):
    repo = Path(__file__).resolve().parents[1]
    args.state_dir.mkdir(parents=True, exist_ok=True)
    scripts = args.state_dir.parent / 'scripts'
    logs = args.state_dir.parent / 'logs/eigenposture'
    scripts.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    script = scripts / 'eigenposture.sbatch'
    output = args.work_block / 'analysis_outputs/eigenposture'
    command = [str(args.python_bin), '-m', 'analysis.eigenposture_pipeline',
               '--block', str(args.work_block), '--source-block', str(args.source_block),
               '--output', str(output), '--workers', str(args.cpus), '--grid-size-mm', str(args.grid_size_mm),
               '--grid-output-name', args.grid_output_name]
    script.write_text('\n'.join([
        '#!/usr/bin/env bash', f'#SBATCH -J eigenposture_{args.source_block.parent.name}_{args.source_block.name}',
        f'#SBATCH -p {args.partition}', f'#SBATCH -c {args.cpus}', f'#SBATCH --mem={args.mem}',
        f'#SBATCH -t {args.time}', '#SBATCH -o ' + shlex.quote(str(logs / 'eigenposture_%j.out')),
        '#SBATCH -e ' + shlex.quote(str(logs / 'eigenposture_%j.err')),
        *([f'#SBATCH --mail-user={args.notify_email}', '#SBATCH --mail-type=FAIL'] if args.notify_email else []),
        'set -euo pipefail', 'umask 0002', 'export PYTHONNOUSERSITE=1 MPLBACKEND=Agg',
        f'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4',
        'cd ' + shlex.quote(str(repo)), shlex.join(command), '',
    ]))
    block = args.source_block.name
    prerequisites = [args.state_dir / f'stitch_{block}.ok',
                     args.state_dir / f'interactions_complete_{block}.ok']
    prerequisites += [args.work_block / 'stitched' / folder / marker for folder, marker in [
        ('speed_vectors', 'speed_vector_complete.ok'), ('sleep_motion', 'sleep_motion_complete.ok'),
        ('posture_interpolated', 'posture_interpolation_complete.ok'),
        (args.grid_output_name, 'grid_occupancy_complete.ok')]]
    config_path = args.state_dir / 'eigenposture_watch.json'
    config = dict(epoch=args.run_epoch or int(time.time()), prerequisites=list(map(str, prerequisites)),
                  job_script=str(script), sbatch_bin=args.sbatch_bin, poll_seconds=args.poll_seconds,
                  output=str(output), python=str(args.python_bin), repo=str(repo))
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    subprocess.run(['bash', '-n', str(script)], check=True)
    if args.dry_run:
        print(f'[dry-run] eigenposture job: {script}')
        return
    check_environment(args.python_bin)
    with (logs / 'watcher.log').open('a') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--watch', str(config_path)],
                                   start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    config_path.with_suffix('.pid').write_text(str(process.pid) + '\n')
    print(f'Started eigenposture watcher PID {process.pid}; log: {logs / "watcher.log"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', type=Path)
    parser.add_argument('--check-environment', type=Path)
    parser.add_argument('--source-block', type=Path)
    parser.add_argument('--work-block', type=Path)
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--python-bin', type=Path, default=Path(sys.executable))
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--mem', default='128G')
    parser.add_argument('--time', default='2-00:00:00')
    parser.add_argument('--partition', default='compute')
    parser.add_argument('--sbatch-bin', default='sbatch')
    parser.add_argument('--grid-size-mm', type=float, default=.25)
    parser.add_argument('--grid-output-name', default='grid_occupancy_histograms')
    parser.add_argument('--poll-seconds', type=int, default=120)
    parser.add_argument('--run-epoch', type=int)
    parser.add_argument('--notify-email', default='')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.check_environment:
        check_environment(args.check_environment)
    elif args.watch:
        watch(args.watch)
    else:
        if not all((args.source_block, args.work_block, args.state_dir)):
            parser.error('--source-block, --work-block and --state-dir are required')
        prepare(args)


if __name__ == '__main__':
    main()
