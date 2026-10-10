"""Within-clip posture rates and finite-support Morlet features from cached poses.

No interpolation or convolution across missing samples, camera changes, or clips.
Run on the machine holding the unfitted pose cache. Outputs are small enough to
reuse locally for all subsequent bin-width / feature / Leiden comparisons.
"""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.signal import convolve

from analysis.eigenposture_features import DAY1_FRAME, windows

FREQUENCIES = (2., 3., 4.)
SAMPLE_HZ = 12.


def mean_missing(x, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


def morlet_power(scores, cameras=None, frequencies=FREQUENCIES, fs=SAMPLE_HZ):
    """Mean squared unit-energy Morlet coefficients, using full valid support.

    scores: clip x sample x PC. Kernel = Gaussian * complex sinusoid, omega0=3,
    truncated at +/-2 sigma and corrected to have exactly zero DC response.
    No padding. At least three valid coefficient centers per frequency/clip.
    """
    powers, counts = [], []
    finite = np.isfinite(scores).all(axis=2)
    if cameras is None:
        cameras = np.zeros(scores.shape[:2])
    for frequency in frequencies:
        sigma = 3 / (2 * np.pi * frequency)
        radius = int(np.ceil(2 * sigma * fs))
        t = np.arange(-radius, radius + 1) / fs
        envelope = np.exp(-0.5 * (t / sigma) ** 2)
        carrier = np.exp(2j * np.pi * frequency * t)
        kernel = envelope * (carrier - np.sum(envelope * carrier) / envelope.sum())
        kernel /= np.sqrt(np.sum(np.abs(kernel) ** 2))
        if len(kernel) > scores.shape[1]:
            raise ValueError("Wavelet does not fit inside a sampled clip")
        support = np.lib.stride_tricks.sliding_window_view(finite, len(kernel), axis=1).all(axis=-1)
        camera_windows = np.lib.stride_tricks.sliding_window_view(cameras, len(kernel), axis=1)
        support &= np.isfinite(camera_windows).all(axis=-1) & (np.ptp(camera_windows, axis=-1) == 0)
        coefficient = convolve(np.nan_to_num(scores), kernel.conj()[None, ::-1, None],
                               mode="valid", method="fft")
        energy = np.abs(coefficient) ** 2
        energy[~support] = np.nan
        n = support.sum(axis=1)
        power = mean_missing(energy, axis=1)
        power[n < 3] = np.nan
        powers.append(power)
        counts.append(n)
    return np.stack(powers, axis=2), np.stack(counts, axis=1)


def clip_features(x, velocity, cameras, center, modes, body_scale_mm=1.):
    scores = (x - center) @ modes[:8].T
    n_pose = np.isfinite(scores).all(axis=2).sum(axis=1)
    pose_mean = mean_missing(scores, 1)
    pose_std = np.sqrt(mean_missing((scores[:, :, :4] - pose_mean[:, None, :4]) ** 2, 1))
    rate = np.diff(scores[:, :, :4], axis=1) * SAMPLE_HZ
    same_camera = cameras[:, 1:] == cameras[:, :-1]
    rate[~same_camera] = np.nan
    n_rate = np.isfinite(rate).all(axis=2).sum(axis=1)
    rate_rms = np.sqrt(mean_missing(rate ** 2, 1))
    rate_rms[n_rate < 7] = np.nan
    wavelets, wavelet_counts = morlet_power(scores[:, :, :4], cameras)
    n_velocity = np.isfinite(velocity).all(axis=2).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peaks = np.nanmax(np.abs(velocity), axis=1)
    rms = np.sqrt(mean_missing(velocity ** 2, 1))
    means = mean_missing(velocity, 1)
    velocity_std = np.sqrt(mean_missing((velocity - means[:, None]) ** 2, 1))
    acceleration = np.diff(velocity, axis=1) * SAMPLE_HZ
    acceleration[~same_camera] = np.nan
    n_acceleration = np.isfinite(acceleration).all(axis=2).sum(axis=1)
    acceleration_rms = np.sqrt(mean_missing(acceleration ** 2, 1))
    acceleration_rms[n_acceleration < 7] = np.nan
    pose_mean[n_pose < 8] = np.nan
    pose_std[n_pose < 8] = np.nan
    for a in (peaks, rms, velocity_std):
        a[n_velocity < 8] = np.nan
    # Physical unsigned antennal velocity in the head-relative body frame.
    antenna_velocity = np.diff(x.reshape(len(x), x.shape[1], 6, 2), axis=1) * (SAMPLE_HZ * body_scale_mm)
    antenna_velocity[~same_camera] = np.nan
    speed = np.linalg.norm(antenna_velocity, axis=-1)
    n_antenna = np.isfinite(speed).all(axis=2).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        antenna_peak = np.nanmax(speed, axis=(1, 2))
    antenna = np.column_stack([
        mean_missing(speed, (1, 2)), antenna_peak,
        mean_missing(np.abs(antenna_velocity[..., 0]), (1, 2)),
        mean_missing(np.abs(antenna_velocity[..., 1]), (1, 2)),
        mean_missing(speed[:, :, 2], 1), mean_missing(speed[:, :, 5], 1)])
    antenna[n_antenna < 7] = np.nan
    values = np.column_stack([peaks, rms, velocity_std, acceleration_rms,
                              pose_mean, pose_std, rate_rms, wavelets.reshape(len(x), -1), antenna])
    return values.astype(np.float32), wavelet_counts


def feature_names():
    names = [f"{axis}_{kind}" for kind in ("peak", "rms", "std", "acceleration_rms")
             for axis in ("forward", "lateral")]
    names += [f"posture_PC{i}_mean" for i in range(1, 9)]
    names += [f"posture_PC{i}_within_std" for i in range(1, 5)]
    names += [f"posture_PC{i}_rate" for i in range(1, 5)]
    names += [f"posture_PC{i}_wavelet_{f:g}Hz" for i in range(1, 5) for f in FREQUENCIES]
    names += ['antenna_unsigned_mean', 'antenna_unsigned_peak',
              'antenna_forward_unsigned_mean', 'antenna_lateral_unsigned_mean',
              'antenna_A_tip_unsigned_mean', 'antenna_B_tip_unsigned_mean']
    return names


def extract(pose_cache, source, basis, output):
    output.mkdir(parents=True, exist_ok=True)
    coverage = pd.read_csv(source / "all_ant_coverage.csv")
    with np.load(basis) as z:
        np.testing.assert_array_equal(z["ants"], coverage.ant)
        center, modes = z["posture_center"], z["posture_modes"]
    with np.load(source / "full_rank/landmark_pca.npz") as z:
        old_center, old_modes = z["mean"], z["components"]
    with np.load(source / "full_rank/eigenposture_measurements.npz") as z:
        published_mean = z["minute"][:, :, 4:16] @ old_modes + old_center
    all_values, all_counts, sources, examples = [], [], [], []
    for i, ant in enumerate(coverage.itertuples()):
        path = pose_cache / (ant.ant.replace(":", "_") + ".npz")
        signature = json.loads(path.with_suffix(".json").read_text())["signature"]
        track = Path(signature["task"]["source"]["path"])
        expected = signature["task"]["source"]
        if (track.stat().st_size, track.stat().st_mtime_ns) != (expected["size"], expected["mtime_ns"]):
            raise ValueError(f"Changed tracking source: {track}")
        assert signature["feature_version"] == "posture_velocity_v2"
        assert signature["task"]["ant"] == ant.ant
        with np.load(path) as z:
            np.testing.assert_array_equal((z["frames"] - DAY1_FRAME) // (60 * 24), np.arange(2880))
            x, velocity, _, cameras = windows(z, ant.body_scale_mm)
        values, counts = clip_features(x, velocity, cameras, center, modes, ant.body_scale_mm)
        expected_scores = (published_mean[i] - center) @ modes[:8].T
        np.testing.assert_allclose(values[:, 8:16], expected_scores, rtol=1e-4, atol=2e-6, equal_nan=True)
        all_values.append(values)
        all_counts.append(counts)
        complete = np.flatnonzero(np.isfinite(x).all(axis=(1, 2)) & (np.ptp(cameras, axis=1) == 0))
        if len(complete):
            rng = np.random.default_rng(7241010 + i)
            ix = rng.choice(complete, min(8, len(complete)), replace=False)
            for j in ix:
                examples.append(dict(ant_index=i, minute=int(j), scores=((x[j] - center) @ modes[:4].T)))
        sources.append(dict(ant=ant.ant, cache=str(path), cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), tracking=expected))
        print(f"{i+1}/114 {ant.ant}: jointly complete clips={np.isfinite(values).all(axis=1).sum()}", flush=True)
    np.savez_compressed(output / "clip_features.npz", ants=coverage.ant.to_numpy(str),
                        values=np.stack(all_values), wavelet_counts=np.stack(all_counts),
                        names=np.array(feature_names()), frequencies=FREQUENCIES,
                        example_ant=[e['ant_index'] for e in examples],
                        example_minute=[e['minute'] for e in examples],
                        example_scores=np.stack([e['scores'] for e in examples]),
                        posture_center=center, posture_modes=modes)
    (output / "clip_features.json").write_text(json.dumps(dict(
        sampling="One 2.5-second clip per minute; 28 smoothed samples at 12 Hz",
        antennal_velocity="Norm/absolute components of head-relative six-landmark displacement in body axes, converted to mm/s; camera changes/gaps invalid; separate tip speeds",
        sleep_labels_used=False,
        frequencies_hz=FREQUENCIES, wavelet="Unit-energy, zero-DC complex Morlet; omega0=3; +/-2 sigma finite support",
        edge_rule="Only full, finite, same-camera supports; no padding or gap interpolation; >=3 centers",
        units="Posture normalized by ant body-axis length; velocity mm/s; acceleration mm/s^2; rates body lengths/s",
        basis=str(basis), basis_sha256=hashlib.sha256(basis.read_bytes()).hexdigest(), sources=sources,
    ), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pose-cache", "source", "basis", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    extract(a.pose_cache, a.source, a.basis, a.output)
