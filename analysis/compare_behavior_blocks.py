"""Transfer a baseline behavior dictionary and ant classes across recording blocks.

Fit nothing on the follow-up recording at the state level: its body-aligned
sequences use the baseline PCA, normalizer, dictionary, and state IDs. Spatial
measurements are independent validation, never classifier inputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

FPS, HZ, SEED = 24., 12., 7241012


def fingerprint(path):
    path = Path(path)
    return dict(path=str(path), size=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def clock_window(before_start, before_seconds, after_start, after_seconds):
    """Select the earliest full baseline interval with the follow-up clock hours."""
    offset = ((after_start - after_start.replace(hour=0, minute=0, second=0)).total_seconds()
              - (before_start - before_start.replace(hour=0, minute=0, second=0)).total_seconds()) % 86400
    if after_seconds > 86400 or offset + after_seconds > before_seconds + 1e-6:
        raise ValueError('No complete baseline interval matches the follow-up clock window')
    return offset, offset + after_seconds


def state_counts(starts, lengths, states, edges, n_states):
    """Exact duration weighting; split segments that cross requested boundaries."""
    result = np.zeros((len(edges)-1, n_states), float)
    ends = starts + lengths
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        weight = np.maximum(0, np.minimum(ends, hi) - np.maximum(starts, lo))
        result[i] = np.bincount(states, weights=weight, minlength=n_states)
    return result


def probabilities(counts):
    count = np.asarray(counts, float)
    total = count.sum(axis=-1, keepdims=True)
    return np.divide(count, total, out=np.full(count.shape, np.nan), where=total > 0)


def centroids(profiles, labels):
    classes = np.unique(labels)
    x = np.sqrt(profiles)
    return classes, np.stack([x[labels == label].mean(axis=0) for label in classes])


def predict_profiles(profiles, classes, centers):
    classes = np.asarray(classes)
    x = np.sqrt(np.atleast_2d(profiles))
    distance = ((x[:, None, :] - centers[None, :, :])**2).sum(axis=2)
    valid = np.isfinite(x).all(axis=1)
    label = np.full(len(x), -1, int)
    label[valid] = classes[distance[valid].argmin(axis=1)]
    return label, distance


def role_coordinate(profile, centers, nurse_index, forager_index):
    direction = centers[forager_index] - centers[nurse_index]
    return (np.sqrt(profile) - centers[nurse_index]) @ direction / (direction @ direction)


def _link(source, target):
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def prepare_phase(block, packet, workspace, ants, reference_pca, workers):
    from analysis.full_behavior_extract import contacts, extract_one
    from analysis.full_behavior_states import make_features
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reference_pca, workspace/'full_body_pca.npz')
    inventory = ants.copy().assign(original_selected=ants.selected, selected=True)
    inventory.to_csv(workspace/'inventory.csv', index=False)
    contacts(block, workspace, inventory)
    tasks = []
    for row in inventory.to_dict('records'):
        name = row['ant'].replace(':', '_')
        source = packet/'sequences'/name
        if (source/'metadata.json').is_file():
            _link(source, workspace/'sequences'/name)
        else:
            tasks.append((block, workspace, row))
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(extract_one, tasks):
                print('EXTRACT', json.dumps(result), flush=True)
    # Baseline feature caches already use the baseline PCA. Follow-up caches
    # have a separately fitted PCA and therefore must be recomputed.
    if packet/'full_body_pca.npz' == reference_pca:
        for ant in inventory.ant:
            name = ant.replace(':', '_')
            if (packet/'features'/name/'metadata.json').is_file():
                _link(packet/'features'/name, workspace/'features'/name)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(make_features, [(workspace, ant) for ant in inventory.ant]):
            print('FEATURES', result['ant'], result['n_windows'], flush=True)
    return inventory


def scan_positions(task):
    """Observed positions plus contemporaneous ArUco corroboration, with no filling."""
    import pyarrow.dataset as ds
    n = int(np.ceil(task['n_frames']/FPS))
    names = ['observed', 'outside', 'aruco_observed', 'aruco_outside']
    counts = {name: np.zeros(n, np.int32) for name in names}
    rect = task['rectangle']
    dataset = ds.dataset(task['track_path'], format='parquet')
    scanner = dataset.scanner(columns=['Frame', 'TrackX', 'TrackY', 'ArucoX', 'ArucoY'],
        filter=ds.field('Bodypoint') == 0, batch_size=262144, use_threads=False)
    for batch in scanner.to_batches():
        frame, x, y, ax, ay = [batch.column(i).to_numpy(zero_copy_only=False) for i in range(5)]
        for prefix, xx, yy in [('', x, y), ('aruco_', ax, ay)]:
            valid = np.isfinite(xx)&np.isfinite(yy)&(frame >= 0)&(frame < task['n_frames'])
            bins = (frame[valid]/FPS).astype(int)
            inside = ((xx[valid] >= rect['x_min_px'])&(xx[valid] <= rect['x_max_px'])
                      &(yy[valid] >= rect['y_min_px'])&(yy[valid] <= rect['y_max_px']))
            counts[prefix+'observed'] += np.bincount(bins, minlength=n)
            counts[prefix+'outside'] += np.bincount(bins[~inside], minlength=n)
    return counts


def summarize_positions(counts, low, high, minimum_frames=1):
    start = np.arange(len(counts['observed']), dtype=float)
    exposure = np.maximum(0, np.minimum(start+1, high)-np.maximum(start, low))
    known = counts['observed'] >= minimum_frames
    outside = np.divide(counts['outside'], counts['observed'],
                        out=np.zeros(len(start)), where=counts['observed'] > 0)
    seconds = float(exposure[known].sum())
    outside_seconds = float(np.dot(exposure*known, outside))
    tag = counts['aruco_observed'] > 0
    tag_fraction = np.divide(counts['aruco_outside'], counts['aruco_observed'],
                            out=np.zeros(len(start)), where=tag)
    tag_seconds = float(exposure[tag].sum())
    return dict(available_seconds=high-low, observed_seconds=seconds,
        position_coverage=seconds/(high-low), outside_seconds=outside_seconds,
        outside_fraction=outside_seconds/seconds if seconds else np.nan,
        aruco_observed_seconds=tag_seconds,
        aruco_outside_seconds=float(np.dot(exposure, tag_fraction)),
        aruco_outside_fraction=float(np.dot(exposure, tag_fraction))/tag_seconds if tag_seconds else np.nan)


def position_worker(task):
    from analysis import trip_phenotyping_utils as trips
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    destination = Path(task['output'])
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination/(task['ant'].replace(':', '_')+'.npz')
    counts = scan_positions(task)
    np.savez_compressed(cache, **counts)
    if task['existing_trip_csv']:
        table = pd.read_csv(task['existing_trip_csv'])
        table = table[table.track_id.eq(task['track_id'])].copy()
    else:
        settings = dict(task, fps=FPS, bodypoint=0, position_bin_seconds=1.,
            frame_max=task['n_frames']-1, cluster_id='comparison', max_state_gap_seconds=30.,
            min_state_run_seconds=5., min_colony_anchor_seconds=5., min_trip_seconds=30.,
            min_trip_coverage=.2, max_path_gap_seconds=3., trip_entropy_bin_mm=10.)
        table, _, _ = trips._scan_track_to_position_bins(settings)
    if table.empty:
        table = pd.DataFrame(columns=['exit_frame', 'return_frame', 'duration_seconds'])
    table.to_csv(destination/(task['ant'].replace(':', '_')+'_trips.csv'), index=False)
    rows = []
    scopes = [('full', 0., task['n_frames']/FPS), ('matched', task['low'], task['high'])]
    scopes += [(f'hour_{hour:02}', lo, hi) for hour, (lo, hi) in
               enumerate(zip(task['hour_edges'][:-1], task['hour_edges'][1:]))]
    for scope, low, high in scopes:
        row = summarize_positions(counts, low, high)
        strict = summarize_positions(counts, low, high, minimum_frames=12)
        # Require complete trips within the analysis window on both days.
        complete = table[(table.exit_frame >= low*FPS)&(table.return_frame <= high*FPS)]
        row.update(ant=task['ant'], side=task['side'], track_id=task['track_id'],
            phase=task['phase'], scope=scope, n_completed_trips=len(complete),
            trip_seconds=float(complete.duration_seconds.sum()),
            strict_position_coverage=strict['position_coverage'],
            strict_outside_fraction=strict['outside_fraction'])
        row['trips_per_observed_hour'] = len(complete)/(row['observed_seconds']/3600) if row['observed_seconds'] else np.nan
        rows.append(row)
    return rows


def assign_states(workspaces, packets, inventories, output, reference, windows, workers):
    from analysis.full_behavior_states import encode, vote
    from sklearn.neighbors import NearestNeighbors
    choice = json.loads((reference/'models/selection.json').read_text())
    normalizer = json.loads((reference/'models'/f"{choice['representation']}_normalizer.json").read_text())
    with np.load(reference/'models/dictionary.npz') as z:
        dictionary = dict(z)
    n_states = int(dictionary['state'].max()+1)
    columns = np.concatenate([np.asarray(n['columns']) for n in normalizer])
    variants = {'standard': np.ones(len(columns), bool), 'without_contacts': columns < 29}
    rows, audit = [], []
    for variant, keep in variants.items():
        index = NearestNeighbors(n_neighbors=5, algorithm='kd_tree', n_jobs=workers).fit(dictionary['x'][:, keep])
        for phase in ('before', 'after'):
            root = workspaces[phase]
            for row in inventories[phase].itertuples():
                name = row.ant.replace(':', '_')
                feature = root/'features'/name
                raw = np.load(feature/'raw.npy', mmap_mode='r')
                starts = np.load(feature/'start.npy')
                lengths = np.load(feature/'length.npy')
                existing = packets[phase]/'assignments'/name/'state.npy'
                state = np.empty(len(raw), np.int16)
                confidence = np.empty(len(raw), np.float32)
                distance = np.empty(len(raw), np.float32)
                for lo in range(0, len(raw), 30000):
                    high = min(len(raw), lo+30000)
                    d, ix = index.kneighbors(encode(raw[lo:high], normalizer)[:, keep])
                    state[lo:high], confidence[lo:high] = vote(dictionary['state'], ix, d)
                    distance[lo:high] = d[:, 0]
                agreement = np.nan
                if phase == 'before' and variant == 'standard' and existing.is_file():
                    original = np.load(existing)
                    assert original.shape == state.shape
                    agreement = float(np.mean(original == state))
                    # Preserve the published baseline, including its exact
                    # landmark-label overrides, rather than relabeling it.
                    state = original
                folder = output/'assignments'/variant/phase/name
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder/'windows.npz', state=state, confidence=confidence,
                                    nearest_distance=distance, start=starts, length=lengths)
                low, high = windows[phase]
                edges = np.r_[np.arange(low, high, 3600), high]
                scopes = [('full', np.array([0., row.n_frames/FPS])),
                          ('matched', np.array([low, high]))]
                scopes += [(f'hour_{i:02}', np.array([a, b])) for i, (a,b) in enumerate(zip(edges[:-1], edges[1:]))]
                for scope, bounds in scopes:
                    count = state_counts(starts, lengths, state, bounds*HZ, n_states)[0]
                    values = probabilities(count)
                    rows.append(dict(phase=phase, ant=row.ant, side=row.side, track_id=row.track_id,
                        variant=variant, scope=scope, available_seconds=bounds[-1]-bounds[0],
                        usable_seconds=float(count.sum()/HZ), behavior_coverage=float(count.sum()/HZ/np.diff(bounds)[0]),
                        **{f'count_S{i}':v for i,v in enumerate(count)},
                        **{f'S{i}':v for i,v in enumerate(values)}))
                audit.append(dict(phase=phase, ant=row.ant, variant=variant, n_windows=len(state),
                    exact_vs_published_agreement=agreement, mean_vote_confidence=float(confidence.mean()),
                    median_nearest_distance=float(np.median(distance)), p95_nearest_distance=float(np.quantile(distance,.95))))
                print('ASSIGNED', variant, phase, row.ant, len(state), flush=True)
    pd.DataFrame(rows).to_csv(output/'state_profiles.csv', index=False)
    pd.DataFrame(audit).to_csv(output/'state_assignment_audit.csv', index=False)


def run(before, after, work, output, workers):
    from tracking.colony.combine_blocks import recording_start
    from camera_cal.region_paths import panorama_regions_path
    from analysis import grid_occupancy_utils as go
    from analysis.trip_phenotyping_utils import colony_rectangles_from_regions
    output.mkdir(parents=True, exist_ok=True)
    blocks = dict(before=before, after=after)
    packets = {phase:block/'analysis_outputs/eigenposture' for phase,block in blocks.items()}
    original = {phase:pd.read_csv(root/'inventory.csv') for phase,root in packets.items()}
    selected_ids = set().union(*[set(table.loc[table.selected,'ant']) for table in original.values()])
    inventories = {phase:table[table.ant.isin(selected_ids)].copy() for phase,table in original.items()}
    identity = original['before'][['ant','side','track_id','selected','detection_fraction']].merge(
        original['after'][['ant','selected','detection_fraction']], on='ant', how='outer',
        suffixes=('_before','_after'), indicator='presence')
    identity['in_comparison'] = identity.ant.isin(selected_ids)
    identity['side'] = identity.ant.str.split(':').str[0]
    identity['track_id'] = identity.ant.str.split(':').str[1].astype(int)
    identity.to_csv(output/'identity_and_coverage.csv', index=False)
    starts = {phase:recording_start(block,list((block/'stitched/per_track').glob('*.parquet')))[0]
              for phase,block in blocks.items()}
    duration = {phase:float(table.n_frames.iloc[0]/FPS) for phase,table in original.items()}
    low, high = clock_window(starts['before'], duration['before'], starts['after'], duration['after'])
    windows = dict(before=(low,high),after=(0.,duration['after']))
    representation = json.loads((packets['before']/'models/selection.json').read_text())['representation']
    provenance = dict(created_epoch=time.time(), fps=FPS, sampling_hz=HZ, seed=SEED,
        before=str(before), after=str(after), reference_state_model=json.loads((packets['before']/'models/selection.json').read_text()),
        treatment_colonies=['left','right'], confirmed_removed_ids=None, removal_time=None,
        identity_policy='colony side plus ArUco-derived TrackID; no inferred ID remapping',
        light_schedule=dict(on_hour=5.5,off_hour=19.5,source='existing colony report settings; controller log not supplied'),
        recording_start={k:v.isoformat() for k,v in starts.items()}, duration_seconds=duration,
        matched_windows={phase:dict(elapsed_seconds=list(bounds),
            start=(starts[phase]+timedelta(seconds=bounds[0])).isoformat(),
            end=(starts[phase]+timedelta(seconds=bounds[1])).isoformat()) for phase,bounds in windows.items()},
        reference_files=[fingerprint(packets['before']/name) for name in ['full_body_pca.npz','models/dictionary.npz',
            'models/selection.json',f'models/{representation}_normalizer.json','roles/ant_classes.csv','roles_left/ant_classes.csv']],
        cohort='union of ants meeting the original >60% tracking threshold in either dataset; lower-coverage transfers labeled separately',
        model_transfer='baseline PCA, normalizer, exact five-neighbor inverse-distance state vote; original baseline labels retained',
        code=fingerprint(Path(__file__)))
    (output/'comparison_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    reference = output/'reference_model';reference.mkdir(exist_ok=True)
    for item in provenance['reference_files']:
        src=Path(item['path']);dst=reference/src.relative_to(packets['before']);dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    base_classes=[]
    for side,folder in [('left','roles_left'),('right','roles')]:
        base_classes.append(pd.read_csv(packets['before']/folder/'ant_classes.csv'))
    pd.concat(base_classes,ignore_index=True).to_csv(output/'baseline_ant_classes.csv',index=False)
    workspaces = {phase:work/phase for phase in blocks}
    for phase in blocks:
        print('PREPARE',phase,len(inventories[phase]),flush=True)
        inventories[phase] = prepare_phase(blocks[phase], packets[phase], workspaces[phase],
            inventories[phase], packets['before']/'full_body_pca.npz', workers)
    position_tasks=[]
    for phase,block in blocks.items():
        source=Path(json.loads((packets[phase]/'pipeline_run.json').read_text())['source_block'])
        annotations=panorama_regions_path(source,search_earlier_dates=True)
        rectangles=colony_rectangles_from_regions(go.load_panorama_regions(annotations,x_split_px=None))
        provenance.setdefault('annotations',{})[phase]=fingerprint(annotations)
        low,high=windows[phase]
        for row in inventories[phase].itertuples():
            existing = packets[phase]/('report_left' if row.side=='left' else 'report')/'completed_trips.csv'
            position_tasks.append(dict(phase=phase,ant=row.ant,side=row.side,track_id=row.track_id,
                track_name=row.track_name,n_frames=row.n_frames,rectangle=rectangles[row.side],
                track_path=str(block/'stitched/per_track'/row.track_name),low=low,high=high,
                hour_edges=np.r_[np.arange(low,high,3600),high].tolist(),
                output=str(output/'positions'/phase),start_clock_seconds=(starts[phase]-starts[phase].replace(hour=0,minute=0,second=0)).total_seconds(),
                existing_trip_csv=str(existing) if row.original_selected else None))
    position_rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, rows in enumerate(pool.map(position_worker,position_tasks),1):
            position_rows.extend(rows)
            print('POSITIONS',n,'/',len(position_tasks),rows[0]['phase'],rows[0]['ant'],flush=True)
    pd.DataFrame(position_rows).to_csv(output/'position_and_trip_metrics.csv',index=False)
    (output/'comparison_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    assign_states(workspaces,packets,inventories,output,packets['before'],windows,workers)
    (output/'measurement_complete.ok').write_text('completed common-reference states and spatial measurements\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['before','after','work','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    run(args.before,args.after,args.work,args.output,args.workers)
