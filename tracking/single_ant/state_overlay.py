"""Exact S0 and classified-time intervals for single-ant speed inspection."""
import json
import hashlib
from pathlib import Path
import numpy as np


def intervals(mask, frame_min=0):
    """Half-open frame intervals; never fill unknown or recording gaps."""
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return np.column_stack((np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))) + frame_min


def load_s0(dataset, track_id):
    if track_id is None:
        return None
    folder = Path(dataset)/'analysis/eigenposture/sleep_from_states/per_track'/f'TrackID_{track_id:04d}_all_single'
    metadata = folder/'sleep_motion_label_metadata.json'
    if not metadata.is_file():
        return None
    meta = json.loads(metadata.read_text())
    parameters = meta['classifier_parameters']
    if (meta['classifier_type'] != 'behavior_cluster_membership'
            or parameters['quiet_states'] != [0] or parameters['minimum_seconds'] != 0):
        raise ValueError(f'Expected direct S0 membership, not motion-based labels: {metadata}')
    values = np.load(folder/meta['files']['sleep_state'], mmap_mode='r')
    if values.shape != (meta['frame_max']-meta['frame_min']+1,) or not np.isin(values, [-1,0,1]).all():
        raise ValueError(f'Invalid S0 frame vector: {folder}')
    return dict(s0=intervals(values == 1, meta['frame_min']),
                known=intervals(values >= 0, meta['frame_min']),
                fps=float(meta['fps']), metadata=metadata)


def clip_intervals(spans, first, stop):
    """Find all intersecting intervals, including one starting before the view."""
    spans = np.asarray(spans).reshape(-1, 2)
    left = np.searchsorted(spans[:,1], first, side='right')
    right = np.searchsorted(spans[:,0], stop, side='left')
    result = spans[left:right].copy()
    if len(result):
        result[:,0] = np.maximum(result[:,0], first)
        result[:,1] = np.minimum(result[:,1], stop)
    return result


def load_sleep_categories(dataset, track_id):
    """Load the separate S0/S1 display cache; keep the original states intact."""
    if track_id is None:return None
    root=Path(dataset)/'analysis/sleep_s01_exclusion'
    folder=root/'per_track'/f'TrackID_{track_id:04d}_all_single'
    if not (root/'COMPLETE.json').is_file() or not (folder/'labels.npz').is_file():return None
    definition=json.loads((root/'definition.json').read_text())
    meta=json.loads((folder/'metadata.json').read_text())
    if meta['definition_sha256']!=hashlib.sha256((root/'definition.json').read_bytes()).hexdigest():
        raise ValueError('Sleep display labels belong to different thresholds')
    with np.load(folder/'labels.npz') as z:
        category=z['category'];evidence=z['evidence'];observed=z['observed']
    if category.shape!=(definition['n_frames'],) or not np.isin(category,[-1,0,1]).all():
        raise ValueError('Invalid three-category sleep display')
    if evidence.shape!=category.shape or observed.shape!=category.shape:raise ValueError('State/evidence timeline mismatch')
    return dict(mode='sleep01',category=category,evidence=evidence,observed=observed,
        sleep=intervals(category==1),not_sleep=intervals(category==0),known=intervals(category>=0),
        fps=float(definition['fps']),thresholds=definition['thresholds_mm_s'],metadata=folder/'metadata.json')


def state_status(labels,frame):
    if labels is None:return 'Ambiguous: no state evidence'
    if labels.get('mode')=='sleep01':
        if not 0<=frame<len(labels['category']):return 'Ambiguous: outside recording'
        c=int(labels['category'][frame]);observed=int(labels['observed'][frame]);evidence=int(labels['evidence'][frame])
        if c==1:return f'S0/S1 sleep: observed S{observed}'
        if c<0:return 'Ambiguous: insufficient posture/motion evidence'
        if evidence==1:return f'Not S0/S1: observed S{observed}'
        why=[]
        if evidence&2:why.append('high velocity')
        if evidence&4:why.append('fast antenna')
        return 'Not S0/S1: inferred from '+' and '.join(why)
    def contains(spans):
        i=np.searchsorted(spans[:,1],frame,side='right')
        return i<len(spans) and spans[i,0]<=frame
    return ('state unknown' if not contains(labels['known']) else 'S0' if contains(labels['s0']) else 'other state')
