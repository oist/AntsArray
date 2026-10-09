"""Physical hourly phenotype measurements, without spatial labels or fitted motifs.

The input is an unfitted cache of uniformly sampled 2.5-second skeleton clips.
Each feature uses its valid short windows; one missing landmark does not reject
the whole clip. Every minute and every measured hour have equal weight.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

FPS = 24
SAMPLE_ENDS = np.arange(4, 59, 2)
DAY1_FRAME = (28 * 60 + 50) * FPS
SEED = 7241009
MIN_SAMPLES = 8
MIN_MINUTES = 5
MIN_HOURS = 12
FEATURES = [
    ("forward_velocity", "Mean forward velocity", "mm/s", "velocity"),
    ("lateral_velocity", "Mean lateral velocity", "mm/s", "velocity"),
    ("speed", "Mean translational speed", "mm/s", "velocity"),
    ("speed_p90", "90th-percentile speed", "mm/s", "velocity"),
    ("moving_fraction", "Fraction moving >0.2 mm/s", "fraction", "velocity"),
    ("backward_fraction", "Fraction backward <−0.05 mm/s", "fraction", "velocity"),
    ("lateral_magnitude", "Mean absolute lateral velocity", "mm/s", "velocity"),
    ("head_bend", "Head bending", "degrees", "posture"),
    ("gaster_bend", "Gaster bending", "degrees", "posture"),
    ("antenna_spread", "Angle between antenna tips", "degrees", "posture"),
    ("antenna_extension", "Antennal straightness", "fraction", "posture"),
    ("antenna_asymmetry", "Difference in antennal straightness", "fraction", "posture"),
    ("head_motion", "Head angular motion", "degrees/s", "dynamics"),
    ("gaster_motion", "Gaster angular motion", "degrees/s", "dynamics"),
    ("antenna_motion", "Antennal angular motion", "degrees/s", "dynamics"),
]


def nan_stat(x, operation="mean", axis=1, minimum=MIN_SAMPLES):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = (
            np.nanmean(x, axis=axis)
            if operation == "mean"
            else np.nanquantile(x, 0.9, axis=axis)
        )
    return np.where(np.isfinite(x).sum(axis=axis) >= minimum, value, np.nan)


def posture_geometry(directions, lengths):
    """Static, body-relative geometry; directions end in [8 segments,2 axes]."""
    angles = np.arctan2(directions[..., 1], directions[..., 0])
    head = np.abs(angles[..., 0])
    gaster = np.abs(np.angle(np.exp(1j * (angles[..., 1] - np.pi))))
    segments = directions * lengths[..., 1:, None]
    tips = np.stack(
        [segments[..., 2:5, :].sum(axis=-2), segments[..., 5:8, :].sum(axis=-2)],
        axis=-2,
    )
    reach = np.linalg.norm(tips, axis=-1)
    chain = np.stack(
        [lengths[..., 3:6].sum(axis=-1), lengths[..., 6:9].sum(axis=-1)], axis=-1
    )
    straight = np.clip(reach / chain, 0, 1)
    cosine = (tips[..., 0, :] * tips[..., 1, :]).sum(axis=-1) / (
        reach[..., 0] * reach[..., 1]
    )
    spread = np.arccos(np.clip(cosine, -1, 1))
    return np.stack(
        [
            np.degrees(head),
            np.degrees(gaster),
            np.degrees(spread),
            straight.mean(axis=-1),
            np.abs(straight[..., 0] - straight[..., 1]),
        ],
        axis=-1,
    )


def clip_measurements(z):
    """Return fifteen per-minute measurements plus observation/camera controls."""
    shape = z["shape"].reshape(-1, 60, 8, 2)
    lengths = z["lengths_mm"]
    cam = z["cameras"]
    position_cam = z["position_cameras"]
    duplicate = z["duplicate_frame"]
    axis_ok = (
        np.isfinite(lengths[..., 0])
        & (lengths[..., 0] >= 0.1)
        & (lengths[..., 0] <= 1.25)
    )
    camera_ok = np.isfinite(cam) & (cam >= 0) & ~duplicate
    raw_pose_ok = (
        axis_ok
        & camera_ok
        & np.isfinite(shape).all(axis=(-1, -2))
        & ((lengths[..., 1:] >= 0.05) & (lengths[..., 1:] <= 2)).all(axis=-1)
    )
    windows = SAMPLE_ENDS[:, None] + np.arange(-3, 1)
    smooth = shape[:, windows].mean(axis=2)
    norm = np.linalg.norm(smooth, axis=-1, keepdims=True)
    smooth = np.divide(smooth, norm, out=np.full_like(smooth, np.nan), where=norm > 0.1)
    smooth_lengths = lengths[:, windows].mean(axis=2)
    pose_ok = raw_pose_ok[:, windows].all(axis=-1) & (
        np.ptp(cam[:, windows], axis=-1) == 0
    )
    smooth[~pose_ok] = np.nan
    geometry = posture_geometry(smooth, smooth_lengths)
    geometry[~pose_ok] = np.nan
    vwindows = SAMPLE_ENDS[:, None] + np.arange(-4, 1)
    velocity = z["velocity_mm_s"].copy()
    speed = np.linalg.norm(velocity, axis=-1)
    vframe_ok = axis_ok & camera_ok & np.isfinite(position_cam) & (position_cam == cam)
    velocity_ok = (
        vframe_ok[:, vwindows].all(axis=-1)
        & (np.ptp(cam[:, vwindows], axis=-1) == 0)
        & np.isfinite(velocity).all(axis=-1)
        & (speed <= 20)
    )
    velocity[~velocity_ok] = np.nan
    speed[~velocity_ok] = np.nan
    angles = np.arctan2(smooth[..., 1], smooth[..., 0])
    angular = np.degrees(np.abs(np.angle(np.exp(1j * np.diff(angles, axis=1))))) * 12
    pair_ok = (
        pose_ok[:, 1:]
        & pose_ok[:, :-1]
        & (cam[:, SAMPLE_ENDS[1:]] == cam[:, SAMPLE_ENDS[:-1]])
    )
    angular[~pair_ok] = np.nan
    values = [
        nan_stat(velocity[..., 0]),
        nan_stat(velocity[..., 1]),
        nan_stat(speed),
        nan_stat(speed, "p90"),
        nan_stat(np.where(velocity_ok, speed > 0.2, np.nan)),
        nan_stat(np.where(velocity_ok, velocity[..., 0] < -0.05, np.nan)),
        nan_stat(np.abs(velocity[..., 1])),
    ]
    values.extend(nan_stat(geometry[..., j]) for j in range(5))
    values.extend(
        [
            nan_stat(angular[..., 0], minimum=MIN_SAMPLES - 1),
            nan_stat(angular[..., 1], minimum=MIN_SAMPLES - 1),
            nan_stat(angular[..., 2:].mean(axis=-1), minimum=MIN_SAMPLES - 1),
        ]
    )
    minute = np.stack(values, axis=-1).astype(np.float32)
    # These controls do not enter the ant clustering.
    controls = dict(
        pose_samples=pose_ok.sum(axis=1),
        velocity_samples=velocity_ok.sum(axis=1),
        modal_camera=np.full(len(minute), -1, dtype=np.int16),
    )
    for i in range(len(minute)):
        valid = cam[i][camera_ok[i]]
        if len(valid):
            controls["modal_camera"][i] = np.bincount(valid.astype(int)).argmax()
    return minute, controls


def hourly_measurements(minute, *, parity=None):
    """Equal-minute means; no observed value becomes zero due to missingness."""
    x = np.asarray(minute).reshape(48, 60, -1).copy()
    if parity is not None:
        x[:, np.arange(60) % 2 != parity] = np.nan
    minimum = MIN_MINUTES if parity is None else 3
    counts = np.isfinite(x).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(x, axis=1)
    means[counts < minimum] = np.nan
    return means, counts


def extract(pose_cache, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(pose_cache.glob("*.npz"))
    if len(paths) != 114:
        raise ValueError(f"Expected all 114 ant caches; got {len(paths)}")
    minutes = []
    hourly = []
    even = []
    odd = []
    counts = []
    rows = []
    provenance = []
    camera_counts = []
    for index, path in enumerate(paths, 1):
        meta = json.loads(path.with_suffix(".json").read_text())
        signature = meta["signature"]
        task = signature["task"]
        source = Path(task["source"]["path"])
        stat = source.stat()
        if (
            stat.st_size != task["source"]["size"]
            or stat.st_mtime_ns != task["source"]["mtime_ns"]
        ):
            raise ValueError(f"Stale source cache: {source}")
        if signature["feature_version"] != "posture_velocity_v2":
            raise ValueError("Missing physical velocities")
        with np.load(path) as z:
            minute, control = clip_measurements(z)
            expected = (z["frames"] - DAY1_FRAME) // (60 * FPS)
            if not np.array_equal(expected, np.arange(2880)):
                raise ValueError("Wrong minute sampling schedule")
        h, n = hourly_measurements(minute)
        e, _ = hourly_measurements(minute, parity=0)
        o, _ = hourly_measurements(minute, parity=1)
        minutes.append(minute)
        hourly.append(h)
        even.append(e)
        odd.append(o)
        counts.append(n)
        row = dict(
            ant=task["ant"],
            side=task["side"],
            track_id=task["track_id"],
            track_name=task["track_name"],
        )
        for day in range(2):
            valid = np.isfinite(h[day * 24 : (day + 1) * 24]).sum(axis=0)
            row[f"day{day+1}_minimum_feature_hours"] = int(valid.min())
            row[f"day{day+1}_eligible"] = bool(valid.min() >= MIN_HOURS)
            row[f"day{day+1}_pose_samples"] = int(
                control["pose_samples"][day * 1440 : (day + 1) * 1440].sum()
            )
            row[f"day{day+1}_velocity_samples"] = int(
                control["velocity_samples"][day * 1440 : (day + 1) * 1440].sum()
            )
            row[f"day{day+1}_observed_minutes"] = int(
                np.isfinite(minute[day * 1440 : (day + 1) * 1440]).any(axis=1).sum()
            )
        cameras = control["modal_camera"].reshape(2, 1440)
        camera_counts.append(
            np.stack([np.bincount(c[c >= 0], minlength=25)[:25] for c in cameras])
        )
        row["dominant_camera_day1"] = int(camera_counts[-1][0].argmax())
        rows.append(row)
        provenance.append(
            dict(
                ant=task["ant"],
                tracking_source=task["source"],
                input_cache=str(path),
                cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                extraction_signature=signature,
            )
        )
        if index % 10 == 0 or index == len(paths):
            print(f"MEASURED {index}/{len(paths)}", flush=True)
    inventory = pd.DataFrame(rows)
    inventory.to_csv(output / "all_ant_coverage.csv", index=False)
    data = dict(
        minute=np.stack(minutes),
        hourly=np.stack(hourly),
        minute_counts=np.stack(counts),
        hourly_even=np.stack(even),
        hourly_odd=np.stack(odd),
        camera_counts=np.stack(camera_counts),
        ants=inventory.ant.to_numpy(str),
        feature_names=np.array([f[0] for f in FEATURES]),
    )
    np.savez_compressed(output / "physical_measurements.npz", **data)
    manifest = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        feature_definitions=FEATURES,
        day1_start="2026-07-24T10:00:00+09:00",
        day2_start="2026-07-25T10:00:00+09:00",
        minutes_per_hour=MIN_MINUTES,
        minimum_hours_per_feature=MIN_HOURS,
        sampling="One uniformly sampled 2.5-second clip per minute per ant; cached unfitted skeleton geometry and body-axis velocities only.",
        model_inputs="Hourly means, then the 25th, 50th and 75th percentiles across measured hours. No absolute position, camera, ant ID or spatial class enters clustering.",
        provenance=provenance,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "measurement_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        inventory.groupby("side")[["day1_eligible", "day2_eligible"]].sum().to_string(),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.pose_cache, args.output)


if __name__ == "__main__":
    main()
