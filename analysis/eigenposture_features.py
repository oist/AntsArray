"""Learn antennal posture directly from aligned landmark coordinates.

No hand-defined shape descriptor or spatial class is an input to this module.
The existing unfitted cache preserves segment directions AND lengths, so their
products reconstruct the six head-relative antennal landmark positions exactly.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles

SEED = 7241009
ENDS = np.arange(4, 59, 2)
DAY1_FRAME = (28 * 60 + 50) * 24
RANK_VARIANCE = 0.90


def mean_missing(x, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


def coordinates(directions, lengths, scale=1.0):
    """Six landmark xy positions relative to head, in body-axis coordinates."""
    vectors = (
        directions.reshape(*directions.shape[:-1], 8, 2)[..., 2:, :]
        * lengths[..., 3:, None]
    )
    chains = vectors.reshape(*vectors.shape[:-2], 2, 3, 2)
    return (np.cumsum(chains, axis=-2) / scale).reshape(*vectors.shape[:-2], 12)


def windows(z, scale):
    raw = coordinates(z["shape"], z["lengths_mm"], scale)
    lengths = z["lengths_mm"]
    cam = z["cameras"]
    axis_ok = (
        np.isfinite(lengths[..., 0])
        & (lengths[..., 0] >= 0.1)
        & (lengths[..., 0] <= 1.25)
    )
    camera_ok = np.isfinite(cam) & (cam >= 0) & ~z["duplicate_frame"]
    pose_ok = (
        axis_ok
        & camera_ok
        & np.isfinite(raw).all(axis=-1)
        & ((lengths[..., 3:] >= 0.05) & (lengths[..., 3:] <= 2)).all(axis=-1)
    )
    ix = ENDS[:, None] + np.arange(-3, 1)
    smooth = raw[:, ix].mean(axis=2)
    valid = pose_ok[:, ix].all(axis=-1) & (np.ptp(cam[:, ix], axis=-1) == 0)
    smooth[~valid] = np.nan
    vx = ENDS[:, None] + np.arange(-4, 1)
    velocity = z["velocity_mm_s"].copy()
    vok = (
        axis_ok
        & camera_ok
        & np.isfinite(z["position_cameras"])
        & (z["position_cameras"] == cam)
    )
    vok = vok[:, vx].all(axis=-1) & (np.ptp(cam[:, vx], axis=-1) == 0)
    vok &= np.isfinite(velocity).all(axis=-1) & (
        np.linalg.norm(velocity, axis=-1) <= 20
    )
    velocity[~vok] = np.nan
    differences = np.diff(smooth, axis=1) * 12
    same_camera = cam[:, ENDS[1:]] == cam[:, ENDS[:-1]]
    differences[~same_camera] = np.nan
    return smooth.astype(np.float32), velocity, differences, cam[:, ENDS]


def clip_moments(x, velocity, delta):
    counts = np.isfinite(x).all(axis=-1).sum(axis=1)
    delta_counts = np.isfinite(delta).all(axis=-1).sum(axis=1)
    vcounts = np.isfinite(velocity).all(axis=-1).sum(axis=1)
    mean = mean_missing(x, 1)
    second = np.einsum("nti,ntj->nij", np.nan_to_num(x), np.nan_to_num(x)) / np.maximum(
        counts[:, None, None], 1
    )
    dsecond = np.einsum(
        "nti,ntj->nij", np.nan_to_num(delta), np.nan_to_num(delta)
    ) / np.maximum(delta_counts[:, None, None], 1)
    v = np.concatenate(
        [mean_missing(velocity, 1), np.sqrt(mean_missing(velocity**2, 1))], axis=1
    )
    mean[counts < 8] = np.nan
    second[counts < 8] = np.nan
    dsecond[delta_counts < 7] = np.nan
    v[vcounts < 8] = np.nan
    return mean, second, dsecond, v, np.stack([counts, delta_counts, vcounts], axis=-1)


def hourly(x, parity=None):
    shape = (48, 60) + x.shape[1:]
    data = x.reshape(shape).copy()
    if parity is not None:
        data[:, np.arange(60) % 2 != parity] = np.nan
    n = np.isfinite(data).sum(axis=1)
    h = mean_missing(data, 1)
    h[n < (5 if parity is None else 3)] = np.nan
    return h, n


def ant_moments(mean, second, day=0, parity=None):
    h, _ = hourly(mean, parity)
    hs, _ = hourly(second, parity)
    sl = slice(day * 24, (day + 1) * 24)
    return mean_missing(h[sl], 0), mean_missing(hs[sl], 0)


def weighted_basis(means, seconds, sides, eligible, rank=None):
    """Equal colony/ant weights; ant moments already balance hours/minutes."""
    first = []
    second = []
    for side in ("left", "right"):
        ix = np.flatnonzero((np.asarray(sides) == side) & eligible)
        if len(ix) < 4:
            raise ValueError(f"Too few eligible ants in {side}")
        first.append(np.mean(means[ix], axis=0))
        second.append(np.mean(seconds[ix], axis=0))
    center = np.mean(first, axis=0)
    moment = np.mean(second, axis=0)
    covariance = moment - np.outer(center, center)
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0)
    components = vectors[:, order].T
    signs = np.sign(components[np.arange(12), np.abs(components).argmax(axis=1)])
    components *= signs[:, None]
    variance = values / values.sum()
    if rank is None:
        rank = int(np.searchsorted(np.cumsum(variance), RANK_VARIANCE) + 1)
    return dict(
        mean=center,
        components=components,
        eigenvalues=values,
        variance_ratio=variance,
        rank=rank,
    )


def project(mean, dsecond, velocity, basis):
    c = basis["components"][: int(basis["rank"])]
    amplitudes = (mean - basis["mean"]) @ c.T
    rate = np.sqrt(np.maximum(np.einsum("ki,nij,kj->nk", c, dsecond, c), 0))
    return np.concatenate([velocity, amplitudes, rate], axis=1).astype(np.float32)


def extract(pose_cache, output):
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "coordinate_moments"
    cache.mkdir(exist_ok=True)
    paths = sorted(pose_cache.glob("*.npz"))
    if len(paths) != 114:
        raise ValueError(f"Expected 114 source identities, got {len(paths)}")
    rows = []
    sources = []
    all_moments = []
    all_seconds = []
    sample_coordinates = []
    sample_velocity = []
    sample_camera = []
    for i, path in enumerate(paths):
        meta = json.loads(path.with_suffix(".json").read_text())
        signature = meta["signature"]
        task = signature["task"]
        source = Path(task["source"]["path"])
        stat = source.stat()
        if (
            stat.st_size != task["source"]["size"]
            or stat.st_mtime_ns != task["source"]["mtime_ns"]
        ):
            raise ValueError(f"Changed tracking source: {source}")
        if signature["feature_version"] != "posture_velocity_v2":
            raise ValueError("Wrong source cache version")
        with np.load(path) as z:
            if not np.array_equal(
                (z["frames"] - DAY1_FRAME) // (60 * 24), np.arange(2880)
            ):
                raise ValueError("Wrong sampling clock")
            lengths = z["lengths_mm"][:1440, :, 0]
            valid = (
                np.isfinite(lengths)
                & (lengths >= 0.1)
                & (lengths <= 1.25)
                & ~z["duplicate_frame"][:1440]
            )
            scale = float(np.median(lengths[valid])) if valid.any() else 1.0
            x, v, d, cameras = windows(z, scale)
        mean, second, dsecond, velocity, counts = clip_moments(x, v, d)
        np.savez_compressed(
            cache / (task["ant"].replace(":", "_") + ".npz"),
            mean=mean,
            dsecond=dsecond,
            velocity=velocity,
            counts=counts,
        )
        row = dict(
            ant=task["ant"],
            side=task["side"],
            track_id=task["track_id"],
            track_name=task["track_name"],
            body_scale_mm=scale,
        )
        channels = np.stack(
            [counts[:, 0] >= 8, counts[:, 1] >= 7, counts[:, 2] >= 8], axis=-1
        )
        for day in range(2):
            hours = (channels.reshape(48, 60, 3).sum(axis=1) >= 5)[
                day * 24 : (day + 1) * 24
            ].sum(axis=0)
            row[f"day{day+1}_posture_hours"] = int(hours[0])
            row[f"day{day+1}_dynamics_hours"] = int(hours[1])
            row[f"day{day+1}_velocity_hours"] = int(hours[2])
            row[f"day{day+1}_eligible"] = bool(hours.min() >= 12)
        for parity, label in [(0, "even"), (1, "odd")]:
            h, _ = hourly(
                np.where(channels, np.ones_like(channels, float), np.nan), parity
            )
            row[f"{label}_eligible"] = bool(np.isfinite(h[:24]).sum(axis=0).min() >= 8)
        rows.append(row)
        moments = [
            ant_moments(mean, second, day=0),
            ant_moments(mean, second, day=1),
            ant_moments(mean, second, day=0, parity=0),
            ant_moments(mean, second, day=0, parity=1),
        ]
        all_moments.append(np.stack([a for a, b in moments]))
        all_seconds.append(np.stack([b for a, b in moments]))
        # Samples for post-fit visualizations and interpretation, balanced by hour.
        rng = np.random.default_rng(SEED + i)
        sx = np.full((2, 24, 10, 12), np.nan, np.float32)
        sv = np.full((2, 24, 10, 2), np.nan, np.float32)
        sc = np.full((2, 24, 10), -1, np.int16)
        for hour in range(48):
            clips = (
                np.flatnonzero(counts[hour * 60 : (hour + 1) * 60, 0] >= 8) + hour * 60
            )
            if len(clips) < 5:
                continue
            chosen = rng.choice(clips, 10, replace=len(clips) < 10)
            for j, clip in enumerate(chosen):
                times = np.flatnonzero(np.isfinite(x[clip]).all(axis=-1))
                t = int(rng.choice(times))
                sx[hour // 24, hour % 24, j] = x[clip, t]
                sv[hour // 24, hour % 24, j] = v[clip, t]
                sc[hour // 24, hour % 24, j] = int(cameras[clip, t])
        sample_coordinates.append(sx)
        sample_velocity.append(sv)
        sample_camera.append(sc)
        sources.append(
            dict(
                ant=task["ant"],
                tracking_source=task["source"],
                cache=str(path),
                cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
        if (i + 1) % 10 == 0 or i == len(paths) - 1:
            print("COORDINATES", i + 1, len(paths), flush=True)
    q = pd.DataFrame(rows)
    q.to_csv(output / "all_ant_coverage.csv", index=False)
    m = np.stack(all_moments)
    s = np.stack(all_seconds)
    eligible = q.day1_eligible.to_numpy(bool)
    sides = q.side.to_numpy()
    basis = weighted_basis(m[:, 0], s[:, 0], sides, eligible)
    rank = basis["rank"]
    repeat = eligible & q.even_eligible & q.odd_eligible
    even_basis = weighted_basis(m[:, 2], s[:, 2], sides, repeat, rank)
    odd_basis = weighted_basis(m[:, 3], s[:, 3], sides, repeat, rank)
    np.savez_compressed(
        output / "landmark_pca.npz",
        **basis,
        even_mean=even_basis["mean"],
        even_components=even_basis["components"],
        odd_mean=odd_basis["mean"],
        odd_components=odd_basis["components"],
    )
    np.savez_compressed(
        output / "ant_coordinate_moments.npz",
        means=m,
        seconds=s,
        ants=q.ant.to_numpy(str),
        versions=np.array(["day1", "day2", "even", "odd"]),
    )
    np.savez_compressed(
        output / "sampled_landmarks.npz",
        coordinates=np.stack(sample_coordinates),
        velocity=np.stack(sample_velocity),
        cameras=np.stack(sample_camera),
        ants=q.ant.to_numpy(str),
    )
    names = (
        ["forward_mean", "lateral_mean", "forward_rms", "lateral_rms"]
        + [f"posture_pc{i+1}_mean" for i in range(rank)]
        + [f"posture_pc{i+1}_rate" for i in range(rank)]
    )
    families = ["velocity"] * 4 + ["posture"] * rank + ["dynamics"] * rank
    minutes = []
    hours = []
    evens = []
    odds = []
    hcounts = []
    for row in q.itertuples():
        with np.load(cache / (row.ant.replace(":", "_") + ".npz")) as z:
            raw = project(z["mean"], z["dsecond"], z["velocity"], basis)
            eraw = project(z["mean"], z["dsecond"], z["velocity"], even_basis)
            oraw = project(z["mean"], z["dsecond"], z["velocity"], odd_basis)
        h, n = hourly(raw)
        e, _ = hourly(eraw, 0)
        o, _ = hourly(oraw, 1)
        minutes.append(raw)
        hours.append(h)
        hcounts.append(n)
        evens.append(e)
        odds.append(o)
    np.savez_compressed(
        output / "eigenposture_measurements.npz",
        minute=np.stack(minutes),
        hourly=np.stack(hours),
        hourly_even=np.stack(evens),
        hourly_odd=np.stack(odds),
        counts=np.stack(hcounts),
        feature_names=np.array(names),
        families=np.array(families),
        ants=q.ant.to_numpy(str),
    )
    stability = []
    rng = np.random.default_rng(SEED)
    for repeat_i in range(100):
        ix = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(eligible & (sides == side)),
                    size=int((eligible & (sides == side)).sum()),
                    replace=True,
                )
                for side in ["left", "right"]
            ]
        )
        b = weighted_basis(m[ix, 0], s[ix, 0], sides[ix], np.ones(len(ix), bool), rank)
        angles = np.degrees(
            subspace_angles(basis["components"][:rank].T, b["components"][:rank].T)
        )
        stability.append(
            dict(
                repeat=repeat_i,
                max_angle_degrees=float(angles.max()),
                median_angle_degrees=float(np.median(angles)),
            )
        )
    pd.DataFrame(stability).to_csv(output / "landmark_basis_bootstrap.csv", index=False)
    split_angles = np.degrees(
        subspace_angles(
            even_basis["components"][:rank].T, odd_basis["components"][:rank].T
        )
    )
    pca_summary = dict(
        rank=rank,
        training_variance_explained=float(basis["variance_ratio"][:rank].sum()),
        split_minute_subspace_angles_degrees=split_angles.tolist(),
        n_training_ants=int(eligible.sum()),
        posture_inputs="12 head-relative body-aligned coordinates; fixed per-ant day-1 body length normalization",
    )
    (output / "landmark_pca_summary.json").write_text(
        json.dumps(pca_summary, indent=2) + "\n"
    )
    manifest = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        seed=SEED,
        day1_start="2026-07-24T10:00:00+09:00",
        day2_start="2026-07-25T10:00:00+09:00",
        rank_rule="Smallest rank explaining at least 90% of equal-colony/ant/hour/minute/sample coordinate variance",
        feature_names=names,
        families=families,
        spatial_inputs_loaded=False,
        handcrafted_shape_descriptors_used=False,
        sources=sources,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "measurement_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("BASIS", json.dumps(pca_summary), flush=True)
    print(
        q.groupby("side")[["day1_eligible", "day2_eligible"]].sum().to_string(),
        flush=True,
    )


def reproject(source, output, variance_target):
    """Sensitivity to retained coordinate variance; reuses unfitted moments."""
    output.mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(source / "all_ant_coverage.csv")
    with np.load(source / "ant_coordinate_moments.npz") as z:
        m = z["means"]
        s = z["seconds"]
    sides = q.side.to_numpy()
    eligible = q.day1_eligible.to_numpy(bool)
    full = weighted_basis(m[:, 0], s[:, 0], sides, eligible)
    rank = (
        12
        if variance_target == 1
        else int(
            np.searchsorted(np.cumsum(full["variance_ratio"]), variance_target) + 1
        )
    )
    basis = weighted_basis(m[:, 0], s[:, 0], sides, eligible, rank)
    repeat = eligible & q.even_eligible & q.odd_eligible
    eb = weighted_basis(m[:, 2], s[:, 2], sides, repeat, rank)
    ob = weighted_basis(m[:, 3], s[:, 3], sides, repeat, rank)
    for name in [
        "all_ant_coverage.csv",
        "ant_coordinate_moments.npz",
        "sampled_landmarks.npz",
    ]:
        shutil.copy2(source / name, output / name)
    np.savez_compressed(
        output / "landmark_pca.npz",
        **basis,
        even_mean=eb["mean"],
        even_components=eb["components"],
        odd_mean=ob["mean"],
        odd_components=ob["components"],
    )
    minutes = []
    hours = []
    evens = []
    odds = []
    counts = []
    for row in q.itertuples():
        with np.load(
            source / "coordinate_moments" / (row.ant.replace(":", "_") + ".npz")
        ) as z:
            raw = project(z["mean"], z["dsecond"], z["velocity"], basis)
            eraw = project(z["mean"], z["dsecond"], z["velocity"], eb)
            oraw = project(z["mean"], z["dsecond"], z["velocity"], ob)
        h, n = hourly(raw)
        e, _ = hourly(eraw, 0)
        o, _ = hourly(oraw, 1)
        minutes.append(raw)
        hours.append(h)
        evens.append(e)
        odds.append(o)
        counts.append(n)
    names = (
        ["forward_mean", "lateral_mean", "forward_rms", "lateral_rms"]
        + [f"posture_pc{i+1}_mean" for i in range(rank)]
        + [f"posture_pc{i+1}_rate" for i in range(rank)]
    )
    families = ["velocity"] * 4 + ["posture"] * rank + ["dynamics"] * rank
    np.savez_compressed(
        output / "eigenposture_measurements.npz",
        minute=np.stack(minutes),
        hourly=np.stack(hours),
        hourly_even=np.stack(evens),
        hourly_odd=np.stack(odds),
        counts=np.stack(counts),
        feature_names=np.array(names),
        families=np.array(families),
        ants=q.ant.to_numpy(str),
    )
    stability = []
    rng = np.random.default_rng(SEED)
    for rep in range(100):
        ix = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(eligible & (sides == side)),
                    size=int((eligible & (sides == side)).sum()),
                    replace=True,
                )
                for side in ["left", "right"]
            ]
        )
        b = weighted_basis(m[ix, 0], s[ix, 0], sides[ix], np.ones(len(ix), bool), rank)
        angles = np.degrees(
            subspace_angles(basis["components"][:rank].T, b["components"][:rank].T)
        )
        stability.append(
            dict(
                repeat=rep,
                max_angle_degrees=float(angles.max()),
                median_angle_degrees=float(np.median(angles)),
            )
        )
    pd.DataFrame(stability).to_csv(output / "landmark_basis_bootstrap.csv", index=False)
    angles = np.degrees(
        subspace_angles(eb["components"][:rank].T, ob["components"][:rank].T)
    )
    (output / "landmark_pca_summary.json").write_text(
        json.dumps(
            dict(
                rank=rank,
                variance_target=variance_target,
                training_variance_explained=float(basis["variance_ratio"][:rank].sum()),
                split_minute_subspace_angles_degrees=angles.tolist(),
                n_training_ants=int(eligible.sum()),
            ),
            indent=2,
        )
        + "\n"
    )
    manifest = json.loads((source / "measurement_manifest.json").read_text())
    manifest.update(
        feature_names=names,
        families=families,
        variance_target=variance_target,
        rank=rank,
        rank_rule=f"Smallest rank explaining {variance_target:.0%} of coordinate variance; all 12 modes for 100%.",
        sensitivity_note="Retained-variance sensitivity specified after inspecting the initial 90% run; no spatial labels used in reprojecting or selecting rank.",
        reprojection_code_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    )
    (output / "measurement_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("REPROJECTED", variance_target, rank, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--pose-cache", type=Path)
    inputs.add_argument("--source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--variance-target", type=float, choices=[0.9, 0.95, 1.0], default=0.9
    )
    a = p.parse_args()
    if a.source:
        reproject(a.source, a.output, a.variance_target)
    elif a.variance_target != 0.9:
        p.error(
            "Extract the source once, then use --source to change retained variance."
        )
    else:
        extract(a.pose_cache, a.output)


if __name__ == "__main__":
    main()
