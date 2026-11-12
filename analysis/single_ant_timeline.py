"""Structural recording support for manual-cup block combinations."""
import hashlib
import json
from pathlib import Path
import numpy as np


def recording_support(root):
    root = Path(root)
    protocol = root/'analysis_protocol.json'
    if not protocol.is_file() or json.loads(protocol.read_text()).get('mode') != 'single_ant':
        return None
    path = root/'tracking_manifest.json'
    manifest = json.loads(path.read_text())
    intervals = [(int(s['frame_offset']), int(s['frame_offset'])+int(s['num_frames']))
                 for s in manifest['recording_segments']]
    if any(a < 0 or b <= a or b > manifest['n_frames'] for a,b in intervals):
        raise ValueError('Invalid recording intervals')
    if any(a < previous[1] for previous,(a,b) in zip(intervals, intervals[1:])):
        raise ValueError('Overlapping recording intervals')
    return dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                intervals=intervals,n_frames=int(manifest['n_frames']))


def support_mask(support, length, frame_min=0, window_frames=1):
    """Unknown outside recordings; restart a trailing window in each block."""
    mask = np.zeros(length, bool)
    for start, stop in support['intervals']:
        lo = max(0, start+window_frames-1-frame_min)
        hi = min(length, stop-frame_min)
        if hi > lo: mask[lo:hi] = True
    return mask
