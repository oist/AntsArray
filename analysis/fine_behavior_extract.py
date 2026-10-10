"""Sample continuous one-minute pose sequences throughout July 24–26.

One random minute per clock hour per ant. No behavior, spatial or sleep labels
enter sampling. Coordinates are body aligned; absolute position is not exported.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from analysis.postural_dynamics_extract import intrinsic_directions, stamp
from analysis.eigenposture_features import coordinates

FPS = 24
HZ = 12
START = 41520
N_HOURS = 48
SECONDS = 60
MM_PER_PX = .016
SEED = 7241011


def sample_starts(ant_index):
    rng = np.random.default_rng(SEED + int(ant_index))
    return START + np.arange(N_HOURS) * 3600 * FPS + rng.integers(0, (3600-SECONDS)*FPS+1, N_HOURS)


def fill_short_gaps(x, camera, maximum=3):
    """Linear fills of <=125 ms, bounded by observations in one known camera."""
    x = np.array(x, dtype=float, copy=True)
    filled = np.zeros(x.shape[:-1], dtype=bool)
    for clip in range(len(x)):
        for point in range(x.shape[2]):
            finite = np.isfinite(x[clip, :, point]).all(axis=1)
            edges = np.diff(np.r_[True, finite, True].astype(int))
            starts, stops = np.flatnonzero(edges == -1), np.flatnonzero(edges == 1)
            for start, stop in zip(starts, stops):
                if start == 0 or stop == x.shape[1] or stop-start > maximum:
                    continue
                cams = camera[clip, start-1:stop+1]
                if not np.isfinite(cams).all() or cams[0] < 0 or np.ptp(cams) != 0:
                    continue
                fraction = np.arange(1, stop-start+1) / (stop-start+1)
                x[clip, start:stop, point] = x[clip, start-1, point] * (1-fraction[:, None]) + x[clip, stop, point] * fraction[:, None]
                filled[clip, start:stop, point] = True
    return x, filled


def measurements(xy, position, camera, position_camera, scale, center, modes):
    xy, filled = fill_short_gaps(xy, camera)
    direction, lengths = intrinsic_directions(xy)
    lengths *= MM_PER_PX
    direction = direction.reshape(*xy.shape[:2], 16)
    pose = coordinates(direction, lengths, scale)
    articulation = direction[:, :, :4]  # head and gaster directions, cosine/sine
    geometry = ((lengths[:, :, 0] >= .1) & (lengths[:, :, 0] <= 1.25)
                & ((lengths[:, :, 1:] >= .05) & (lengths[:, :, 1:] <= 2)).all(axis=2))
    base_valid = geometry & np.isfinite(pose).all(axis=2) & np.isfinite(articulation).all(axis=2)
    base_valid &= np.isfinite(camera) & (camera >= 0)
    # Five-frame quadratic smoothing has substantially less high-frequency
    # attenuation than the earlier four-frame moving average.
    pose[~base_valid] = np.nan
    articulation[~base_valid] = np.nan
    smooth = savgol_filter(pose, 5, 2, axis=1, mode='interp')
    art = savgol_filter(articulation, 5, 2, axis=1, mode='interp')
    index = np.arange(2, xy.shape[1]-2, 2)
    support = index[:, None] + np.arange(-2, 3)
    same = np.isfinite(camera[:, support]).all(axis=2) & (np.ptp(camera[:, support], axis=2) == 0)
    good = base_valid[:, support].all(axis=2) & same
    smooth, art = smooth[:, index], art[:, index]
    smooth[~good] = np.nan; art[~good] = np.nan
    pc = (smooth-center) @ modes[:4].T
    axis = xy[:, :, 0] - xy[:, :, 2]
    axis /= np.maximum(np.linalg.norm(axis, axis=2, keepdims=True), 1e-12)
    anterior = np.mean(axis[:, support], axis=2)
    anterior /= np.maximum(np.linalg.norm(anterior, axis=2, keepdims=True), 1e-12)
    lateral = np.stack([-anterior[:, :, 1], anterior[:, :, 0]], axis=2)
    pp = position[:, support] * MM_PER_PX
    velocity = ((pp-pp[:, :, 2:3]) * np.arange(-2, 3)[None, None, :, None]).sum(axis=2) * FPS/10
    velocity = np.stack([(velocity*anterior).sum(axis=2), (velocity*lateral).sum(axis=2)], axis=2)
    velocity_ok = good & (position_camera[:, support] == camera[:, support]).all(axis=2)
    velocity_ok &= np.isfinite(velocity).all(axis=2) & (np.linalg.norm(velocity, axis=2) <= 20)
    velocity[~velocity_ok] = np.nan
    derivative = np.diff(smooth.reshape(len(xy), len(index), 6, 2), axis=1)*HZ*scale
    antenna = np.linalg.norm(derivative, axis=3).reshape(len(xy), len(index)-1, 2, 3).mean(axis=3)
    turning = np.abs(np.arctan2(anterior[:, 1:, 1]*anterior[:, :-1, 0]-anterior[:, 1:, 0]*anterior[:, :-1, 1],
                              (anterior[:, 1:]*anterior[:, :-1]).sum(axis=2))) * HZ
    pair_ok = good[:, 1:] & good[:, :-1] & (camera[:, index[1:]] == camera[:, index[:-1]])
    antenna[~pair_ok] = np.nan; turning[~pair_ok] = np.nan
    antenna = np.concatenate([np.full((len(xy), 1, 2), np.nan), antenna], axis=1)
    turning = np.concatenate([np.full((len(xy), 1), np.nan), turning], axis=1)
    signals = np.concatenate([pc, art, np.abs(velocity), antenna, turning[:, :, None]], axis=2)
    # Residuals and interpolation remain quality diagnostics, never features.
    residual = np.sqrt(np.nanmean((pose[:, index]-smooth)**2, axis=2))*scale
    return dict(signals=signals.astype(np.float32), pose=smooth.astype(np.float32),
                camera=camera[:, index], sample_offsets=index,
                interpolated_fraction=filled[:, support].mean(axis=(2, 3)).astype(np.float32),
                residual_mm=residual.astype(np.float32))


def extract_one(arguments):
    import pyarrow.parquet as pq
    block, source, basis, output, ant_index = arguments
    ants = pd.read_csv(source / 'all_ant_coverage.csv')
    ant = ants.iloc[ant_index]
    track = block / 'stitched/per_track' / ant.track_name
    before = stamp(track)
    target = output / (ant.ant.replace(':', '_') + '.npz')
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    signature = dict(source=before, code_sha256=code_hash, seed=SEED+ant_index,
                     hours=N_HOURS, seconds=SECONDS, basis_sha256=hashlib.sha256(basis.read_bytes()).hexdigest())
    if target.exists() and target.with_suffix('.json').exists():
        metadata = json.loads(target.with_suffix('.json').read_text())
        if metadata['signature'] == signature:
            return {'ant': ant.ant, 'cached': True}
    starts = sample_starts(ant_index)
    frames = (starts[:, None] + np.arange(SECONDS*FPS)).ravel()
    parquet = pq.ParquetFile(track)
    n_frames = int(parquet.schema_arrow.metadata[b'num_frames'])
    assert n_frames == 4247196 and frames[-1] < n_frames
    lookup = np.full(n_frames, -1, dtype=np.int32); lookup[frames] = np.arange(len(frames))
    xy = np.full((len(frames), 10, 2), np.nan, dtype=float)
    seen = np.zeros((len(frames), 10), dtype=np.uint8)
    camera = np.full(len(frames), np.nan)
    position = np.full((len(frames), 2), np.nan)
    position_camera = np.full(len(frames), np.nan)
    for batch in parquet.iter_batches(batch_size=262144,
            columns=['Frame', 'Bodypoint', 'X', 'Y', 'SleapCam', 'TrackX', 'TrackY', 'CameraID'], use_threads=False):
        fr, bp = [batch.column(i).to_numpy(zero_copy_only=False) for i in (0, 1)]
        selected = np.flatnonzero((fr >= 0) & (fr < n_frames) & (bp >= 0) & (bp < 10))
        ix = lookup[fr[selected]]; keep = ix >= 0; selected, ix = selected[keep], ix[keep]
        point = bp[selected].astype(int)
        np.add.at(seen, (ix, point), 1)
        for j in (0, 1): xy[ix, point, j] = batch.column(2+j).to_numpy(zero_copy_only=False)[selected]
        anchor = point == 0; at = selected[anchor]; ix0 = ix[anchor]
        camera[ix0] = batch.column(4).to_numpy(zero_copy_only=False)[at]
        for j in (0, 1): position[ix0, j] = batch.column(5+j).to_numpy(zero_copy_only=False)[at]
        position_camera[ix0] = batch.column(7).to_numpy(zero_copy_only=False)[at]
    duplicate = (seen > 1).any(axis=1)
    xy[duplicate] = np.nan; camera[duplicate] = np.nan
    with np.load(basis) as z: center, modes = z['posture_center'], z['posture_modes']
    result = measurements(xy.reshape(N_HOURS, SECONDS*FPS, 10, 2),
        position.reshape(N_HOURS, SECONDS*FPS, 2), camera.reshape(N_HOURS, SECONDS*FPS),
        position_camera.reshape(N_HOURS, SECONDS*FPS), ant.body_scale_mm, center, modes)
    assert stamp(track) == before
    np.savez_compressed(target, **result, starts=starts, ant_index=ant_index,
                        ant=str(ant.ant), body_scale_mm=ant.body_scale_mm)
    metadata = dict(signature=signature, ant=str(ant.ant), ant_index=ant_index,
        complete_samples=int(np.isfinite(result['signals']).all(axis=2).sum()),
        total_samples=int(np.prod(result['signals'].shape[:2])),
        filled_raw_landmark_limit='<=3 raw frames, within one known camera; no long-gap interpolation',
        signal_names=['PC1','PC2','PC3','PC4','head_cos','head_sin','gaster_cos','gaster_sin',
                      'forward_unsigned','lateral_unsigned','antenna_A_unsigned','antenna_B_unsigned','turn_unsigned'],
        sampling_hz=HZ, output_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    target.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return {k:metadata[k] for k in ('ant','complete_samples','total_samples')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('block','source','basis','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--ant-index', type=int)
    a = parser.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    indices = range(114) if a.ant_index is None else [a.ant_index]
    tasks = [(a.block,a.source,a.basis,a.output,i) for i in indices]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(extract_one, tasks): print(json.dumps(result), flush=True)
