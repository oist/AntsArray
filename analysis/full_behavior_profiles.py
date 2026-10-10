"""Mean observed posture and physical features for each assigned behavior state.

Run after assignment, while the dense local pose arrays are available. The small
summary travels with the published model packet, so plots need no dense arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.full_behavior_extract import HZ

MOTION_NAMES = ['forward_peak', 'lateral_peak', 'antenna_A_mean', 'antenna_B_mean',
                'turn_mean', 'exact_contact_mean', 'exact_contact_peak']


def state_sums(pose, starts, lengths, states, raw, n_states, batch_size=20000):
    """Sum actual aligned coordinates and duration-weighted window features.

    A short tail contributes only its observed samples. Unassigned frames,
    including gaps and excluded tails, do not enter either average.
    """
    assert len(starts) == len(lengths) == len(states) == len(raw)
    assert (lengths > 0).all() and ((states >= 0) & (states < n_states)).all()
    assert (starts >= 0).all() and (starts + lengths <= len(pose)).all()
    assert (starts[1:] >= starts[:-1] + lengths[:-1]).all()
    count = np.bincount(states, weights=lengths, minlength=n_states).astype(np.int64)
    posture = np.zeros((n_states, 20))
    motion = np.zeros((n_states, 7))
    for lo in range(0, len(starts), batch_size):
        hi = min(lo + batch_size, len(starts))
        size, label = lengths[lo:hi], states[lo:hi]
        offset = np.arange(size.max())[None, :]
        mask = offset < size[:, None]
        index = starts[lo:hi, None] + np.minimum(offset, size[:, None] - 1)
        observed = pose[index]
        assert np.isfinite(observed).all()
        window_sum = np.sum(observed * mask[:, :, None], axis=1, dtype=np.float64)
        features = np.asarray(raw[lo:hi, 24:31], dtype=np.float64)
        assert np.isfinite(features).all()
        for j in range(20):
            posture[:, j] += np.bincount(label, weights=window_sum[:, j], minlength=n_states)
        for j in range(7):
            motion[:, j] += np.bincount(label, weights=features[:, j] * size, minlength=n_states)
    return count, posture, motion


def build(root):
    """Recompute the compact summary; never refit or relabel the state model."""
    root = Path(root)
    cohort = pd.read_csv(root / 'inventory.csv').query('selected')
    dictionary_path = root / 'models/dictionary.npz'
    with np.load(dictionary_path) as z:
        n_states = int(z['state'].max()) + 1
    counts, posture, motion, provenance = [], [], [], []
    for ant in cohort.ant:
        name = ant.replace(':', '_')
        sequence, features = root / 'sequences' / name, root / 'features' / name
        paths = [sequence / 'pose.npy', features / 'start.npy', features / 'length.npy',
                 root / 'assignments' / name / 'state.npy', features / 'raw.npy']
        arrays = [np.load(path, mmap_mode='r') for path in paths]
        count, shape, values = state_sums(*arrays, n_states)
        counts.append(count); posture.append(shape); motion.append(values)
        provenance.append(dict(ant=ant, extraction=json.loads((sequence / 'metadata.json').read_text()),
            inputs=[dict(file=str(p.relative_to(root)), bytes=p.stat().st_size,
                         mtime_ns=p.stat().st_mtime_ns) for p in paths]))
        print(f'{ant}: {count.sum():,} classified posture samples', flush=True)
    count = np.sum(counts, axis=0)
    assert (count > 0).all()
    mean_pose = (np.sum(posture, axis=0) / count[:, None]).reshape(n_states, 10, 2)
    mean_features = np.sum(motion, axis=0) / count[:, None]
    np.savez(root / 'models/state_means.npz', mean_pose=mean_pose, mean_features=mean_features,
             sample_count=count, ant_sample_count=np.asarray(counts), ants=cohort.ant.to_numpy(str),
             feature_names=np.asarray(MOTION_NAMES))
    metadata = dict(dictionary_sha256=hashlib.sha256(dictionary_path.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_ants=len(cohort), n_samples=int(count.sum()), sampling_hz=HZ,
        weighting='Pooled observed duration across both colonies; each assigned 12-Hz sample has equal weight',
        posture='Mean actual aligned coordinates, normalized by each ant median body-axis length; no PCA reconstruction',
        features='Duration-weighted means of window features; forward/lateral and peak contacts remain window maxima',
        inputs=provenance)
    (root / 'models/state_means.json').write_text(json.dumps(metadata, indent=2) + '\n')
    table = pd.DataFrame(mean_features, columns=MOTION_NAMES)
    table.insert(0, 'observed_hours', count / HZ / 3600)
    table.insert(0, 'sample_count', count)
    table.insert(0, 'state', np.arange(n_states))
    table.to_csv(root / 'models/state_means.csv', index=False)
    print(f'Mean profiles: {count.sum():,} samples, {n_states} states', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    build(parser.parse_args().root)
