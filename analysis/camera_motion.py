"""Motion and smoothing within camera views, with explicit recording boundaries."""
import numpy as np
import pandas as pd

RECORDING_COLUMNS = ('source_block', 'recording_id', 'source_file')
MOTION_VERSION = 'same_camera_v1'


def recording_keys(rows):
    """Adjacent chunks share a recording; timestamped recordings do not."""
    keys = None
    for name in RECORDING_COLUMNS:
        if name in rows:
            # Normalize unique strings only: a full track contains millions of
            # repeated chunk filenames. Categorical storage avoids copying a
            # Python string for every landmark observation.
            codes, unique = pd.factorize(rows[name].fillna('<unknown>'), sort=False)
            normalized = np.array([name + '=' + (str(value).split('_chunk', 1)[0]
                if name == 'source_file' else str(value)) for value in unique], dtype=object)
            values = normalized[codes]
            keys = values if keys is None else pd.Series(keys).str.cat(pd.Series(values), sep='|').to_numpy()
    if keys is None:
        keys = np.full(len(rows), '', dtype=object)
    return pd.Series(pd.Categorical(keys), index=rows.index)


def camera_local_fit(values, camera, recording, centers, *, radius=5):
    """Quadratic five-observation fit at observed centers within one view.

    Require two earlier and two later finite observations, each within radius
    native frames. Fit using their actual timestamps, never extrapolate or fill
    a missing center. Ordinary consecutive observations reproduce SG(5, 2).
    Returns fitted values and derivatives per native frame; missing is NaN.
    """
    values = np.asarray(values)
    centers = np.asarray(centers, dtype=int)
    flat = values.reshape(len(values), -1)
    fitted = np.full((len(centers), flat.shape[1]), np.nan)
    derivative = fitted.copy()
    valid = np.isfinite(flat).all(axis=1) & (camera >= 0) & (recording >= 0)
    for rec in np.unique(recording[centers][valid[centers]]):
        for cam in np.unique(camera[centers][valid[centers] & (recording[centers] == rec)]):
            observations = np.flatnonzero(valid & (camera == cam) & (recording == rec))
            target = np.flatnonzero(valid[centers] & (camera[centers] == cam) & (recording[centers] == rec))
            ranks = np.searchsorted(observations, centers[target])
            keep = (ranks >= 2) & (ranks + 2 < len(observations))
            target, ranks = target[keep], ranks[keep]
            if not len(target):
                continue
            ix = observations[ranks[:, None] + np.arange(-2, 3)]
            dt = ix - centers[target, None]
            keep = (np.abs(dt) <= radius).all(axis=1)
            ix, dt, target = ix[keep], dt[keep], target[keep]
            # Only a small number of integer timestamp patterns occur. Reuse
            # their polynomial weights instead of solving millions of fits.
            patterns, inverse = np.unique(dt, axis=0, return_inverse=True)
            for i, offsets in enumerate(patterns):
                selected = inverse == i
                weights = np.linalg.pinv(np.column_stack([np.ones(5), offsets, offsets**2]))
                data = flat[ix[selected]]
                fitted[target[selected]] = np.einsum('j,ijk->ik', weights[0], data)
                derivative[target[selected]] = np.einsum('j,ijk->ik', weights[1], data)
    shape = (len(centers),) + values.shape[1:]
    return fitted.reshape(shape), derivative.reshape(shape)
