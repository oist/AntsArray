"""Read-only detector diagnostic; no tracking or detector settings are changed.

One jittered frame per 30 seconds, synchronized across cameras. Raw SLEAP
anchors are the denominator, not successfully identified/stitched tracks.
Own outputs are isolated under analysis_outputs/tag_detection_spatial.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Circle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.grid_occupancy_utils import load_panorama_regions

RADII = (0.25, 0.5, 1.0, 1.6)
ZONES = ("colony interior", "colony edge", "open floor", "outer wall", "resource")
SEED = 8102026
VERSION = 1


def transform(xy, H):
    p = np.column_stack([xy, np.ones(len(xy))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def match_unique(xy, tags, radius_px):
    """Maximum-cardinality, then minimum-distance one-to-one matching."""
    hit = np.zeros(len(xy), bool)
    distance = np.full(len(xy), np.nan)
    assigned = np.full(len(xy), -1, dtype=int)
    if not len(xy) or not len(tags):
        return hit, distance, assigned
    d = np.linalg.norm(xy[:, None] - tags[None, :], axis=2)
    penalty = min(len(xy), len(tags)) + 1.0
    cost = np.full((len(xy), len(tags) + len(xy)), penalty)
    cost[:, :len(tags)] = np.where(d <= radius_px, d / radius_px, penalty * 3)
    a, b = linear_sum_assignment(cost)
    good = b < len(tags)
    a, b = a[good], b[good]
    hit[a], distance[a], assigned[a] = True, d[a, b], b
    return hit, distance, assigned


def owner_camera(xy, hmats, width=4024, height=3036):
    """Fixed spatial camera ownership: greatest margin from image boundary.

    Avoids counting an ant twice in camera overlaps or choosing a camera based
    on whether its tag decoded. Same-camera matching avoids sync assumptions.
    """
    scores = []
    for H in hmats:
        p = transform(xy, np.linalg.inv(H))
        scores.append(np.minimum.reduce([p[:, 0] / width, 1 - p[:, 0] / width,
                                         p[:, 1] / height, 1 - p[:, 1] / height]))
    return np.argmax(scores, axis=0)


def signed_rectangle_distance(xy, bounds):
    """Positive outside, negative inside; Euclidean distance at outer corners."""
    x0, y0, x1, y1 = bounds
    outside = np.hypot(np.maximum.reduce([x0-xy[:, 0], xy[:, 0]-x1, np.zeros(len(xy))]),
                       np.maximum.reduce([y0-xy[:, 1], xy[:, 1]-y1, np.zeros(len(xy))]))
    inside = np.minimum.reduce([xy[:, 0]-x0, x1-xy[:, 0], xy[:, 1]-y0, y1-xy[:, 1]])
    return np.where(inside >= 0, -inside, outside)


def bounds(row):
    return np.array([row.tracking_x_min_px, row.tracking_y_min_px,
                     row.tracking_x_max_px, row.tracking_y_max_px])


def fingerprint(path):
    s = path.stat()
    return dict(path=str(path), size=s.st_size, mtime_ns=s.st_mtime_ns)


def sample_frames(chunk, step, nframes):
    rng = np.random.default_rng(SEED + chunk)
    starts = np.arange(0, nframes, step)
    return starts + np.array([rng.integers(min(step, nframes-s)) for s in starts])


def process_camera(args):
    slp, hmats, scale, step, cache_root, hmat_fingerprint = args
    slp = Path(slp)
    camera, chunk = map(int, re.match(r"cam(\d+)_.*_(\d{3})\.slp$", slp.name).groups())
    aruco = slp.with_name(slp.stem + "_aruco_tracks.h5")
    path = Path(cache_root) / (slp.stem + ".parquet")
    audit_path = path.with_suffix(".json")
    signature = dict(version=VERSION, sleap=fingerprint(slp), aruco=fingerprint(aruco),
                     step=step, scale=scale, hmats=hmat_fingerprint, seed=SEED, radii=RADII)
    signature = json.loads(json.dumps(signature))
    if path.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit.get("signature") == signature:
            return str(path), audit
    start = time.monotonic()
    with h5py.File(aruco) as f:
        nframes = len(f["aruco_tracks"])
        sampled = sample_frames(chunk, step, nframes)
        tag_array = f["aruco_tracks"][sampled]
    with h5py.File(slp) as f:
        frames = f["frames"][:]
        if len(np.unique(frames["video"])) != 1 or len(np.unique(frames["frame_idx"])) != len(frames):
            raise ValueError(f"Ambiguous frame/video table: {slp}")
        if frames["frame_idx"].max(initial=0) >= nframes:
            raise ValueError(f"SLEAP/ArUco frame-span mismatch: {slp}")
        chosen = frames[np.isin(frames["frame_idx"], sampled)]
        ids = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in
                              zip(chosen["instance_id_start"], chosen["instance_id_end"])])
        inst = f["instances"][ids]
        if not np.all(inst["instance_type"] == 1):
            raise ValueError(f"Non-predicted instances: {slp}")
        pts = f["pred_points"][inst["point_id_start"].astype(np.int64)]
        frame_ids = np.repeat(chosen["frame_idx"], (chosen["instance_id_end"] - chosen["instance_id_start"]).astype(np.int64))
    native = np.column_stack([pts["x"], pts["y"]])
    good = np.isfinite(native).all(axis=1) & np.isfinite(pts["score"]) & (pts["score"] >= 0.5)
    native, frame_ids, scores = native[good], frame_ids[good], pts["score"][good]
    xy = transform(native, hmats[camera-1])
    owner = owner_camera(xy, hmats)
    rows = pd.DataFrame(dict(frame=frame_ids.astype(np.int64), camera=camera, chunk=chunk,
                             x=xy[:, 0], y=xy[:, 1], native_x=native[:, 0], native_y=native[:, 1],
                             score=scores, primary=owner == camera-1))
    for radius in RADII:
        rows[f"hit_{radius:g}"] = False
    rows["tag_id"] = -1
    rows["match_mm"] = np.nan
    rows["nearest_tag_mm"] = np.inf
    rows["nearest_ant_mm"] = np.inf
    tag_count = 0
    for i, fr in enumerate(sampled):
        idx = np.flatnonzero(frame_ids == fr)
        a = tag_array[i]
        valid = np.isfinite(a).all(axis=1) & (a != 0).any(axis=1)
        tag_ids = np.flatnonzero(valid)
        tags = transform(a[valid], hmats[camera-1])
        tag_count += len(tags)
        if not len(idx):
            continue
        s = xy[idx]
        if len(tags):
            rows.loc[idx, "nearest_tag_mm"] = np.linalg.norm(s[:, None]-tags[None, :], axis=2).min(axis=1)*scale
        if len(s) > 1:
            d = np.linalg.norm(s[:, None]-s[None, :], axis=2)
            np.fill_diagonal(d, np.inf)
            rows.loc[idx, "nearest_ant_mm"] = d.min(axis=1)*scale
        for radius in RADII:
            hit, distance, assigned = match_unique(s, tags, radius/scale)
            rows.loc[idx, f"hit_{radius:g}"] = hit
            if radius == 0.5:
                rows.loc[idx, "match_mm"] = distance*scale
                rows.loc[idx[hit], "tag_id"] = tag_ids[assigned[hit]]
    rows.to_parquet(path, index=False)
    audit = dict(signature=signature, camera=camera, chunk=chunk, nframes=nframes,
                 requested_frames=len(sampled), available_sleap_frames=len(chosen),
                 raw_instances=len(ids), accepted_instances=len(rows), tag_reads=tag_count,
                 seconds=time.monotonic()-start)
    audit_path.write_text(json.dumps(audit, indent=2))
    return str(path), audit


def add_regions(rows, regions, scale):
    rows = rows.copy()
    arenas = regions[regions.region_type.eq("arena")].set_index("side")
    colonies = regions[regions.region_type.eq("colony")].set_index("side")
    split = (arenas.loc["left"].tracking_x_max_px + arenas.loc["right"].tracking_x_min_px)/2
    rows["side"] = np.where(rows.x < split, "left", "right")
    rows["colony_distance_mm"] = np.nan
    rows["wall_distance_mm"] = np.nan
    rows["zone"] = "outside annotation"
    for side in ("left", "right"):
        mask = rows.side.eq(side)
        xy = rows.loc[mask, ["x", "y"]].to_numpy()
        dc = signed_rectangle_distance(xy, bounds(colonies.loc[side]))*scale
        dw = -signed_rectangle_distance(xy, bounds(arenas.loc[side]))*scale
        resource = np.zeros(len(xy), bool)
        for row in regions[regions.side.eq(side) & regions.region_type.isin(["food", "water"])].itertuples():
            resource |= np.linalg.norm(xy - [row.tracking_center_x_px, row.tracking_center_y_px], axis=1) <= row.radius_px
        zone = np.full(len(xy), "open floor", dtype=object)
        zone[resource] = "resource"
        zone[dw <= 2] = "outer wall"
        zone[np.abs(dc) <= 2] = "colony edge"
        zone[dc < -2] = "colony interior"
        # Keep ants just outside the drawn rectangle: excluding them would
        # selectively discard the very wall-climbing cases under investigation.
        zone[dw < -2] = "outside annotation"
        rows.loc[mask, "colony_distance_mm"] = dc
        rows.loc[mask, "wall_distance_mm"] = dw
        rows.loc[mask, "zone"] = zone
    return rows


def block_summary(rows, keys, hit="hit_0.5", nboot=1000):
    """Resample whole 30-minute chunks, not autocorrelated ant observations."""
    blocks = rows.groupby(keys + ["chunk"], observed=True)[hit].agg(["sum", "size"]).reset_index()
    results = []
    chunks = np.sort(rows.chunk.unique())
    rng = np.random.default_rng(SEED)
    draw = rng.integers(len(chunks), size=(nboot, len(chunks)))
    for key, g in blocks.groupby(keys, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        g = g.set_index("chunk").reindex(chunks, fill_value=0)
        hits, total = g["sum"].to_numpy(), g["size"].to_numpy()
        den = total[draw].sum(axis=1)
        boot = hits[draw].sum(axis=1) / np.where(den > 0, den, np.nan)
        lo, hi = np.nanquantile(boot, [.025, .975])
        results.append(dict(zip(keys, key), n=int(total.sum()), detected=int(hits.sum()),
                            rate=hits.sum()/total.sum(), lower=lo, upper=hi,
                            n_chunks=int((total>0).sum())))
    return pd.DataFrame(results)


def outlines(ax, regions, side, scale):
    for r in regions[regions.side.eq(side)].itertuples():
        color = {"arena":"white", "colony":"cyan", "food":"tomato", "water":"dodgerblue"}[r.region_type]
        if r.shape == "rectangle":
            b = bounds(r)*scale
            ax.add_patch(Rectangle(b[:2], b[2]-b[0], b[3]-b[1], fill=False, ec=color, lw=1))
        else:
            ax.add_patch(Circle((r.tracking_center_x_px*scale, r.tracking_center_y_px*scale),
                                r.radius_px*scale, fill=False, ec=color, lw=1))


def make_results(rows, regions, scale, meta, root):
    primary = rows[rows.primary & rows.zone.ne("outside annotation")].copy()
    summary = block_summary(primary, ["side", "zone"])
    summary.to_csv(root / "region_detection_rates.csv", index=False)
    per_block = primary.groupby(["side", "chunk", "camera", "zone"], observed=True)["hit_0.5"].agg(["sum", "size"])
    per_block.to_csv(root / "rates_by_camera_chunk_region.csv")
    sensitivity = []
    for name, subset in [("primary", primary), ("all camera views", rows[rows.zone.ne("outside annotation")]),
                         ("score >= 0.9", primary[primary.score >= .9]),
                         ("nearest ant > 2 mm", primary[primary.nearest_ant_mm > 2])]:
        for radius in RADII:
            table = block_summary(subset, ["side", "zone"], hit=f"hit_{radius:g}", nboot=500)
            table["selection"], table["match_radius_mm"] = name, radius
            sensitivity.append(table)
    pd.concat(sensitivity).to_csv(root / "sensitivity.csv", index=False)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    bg = plt.imread(meta["panorama_image"])
    extent = np.array([meta["raw_tracking_bounds_px"][k] for k in ["x_min", "x_max", "y_max", "y_min"]])*scale
    global_x = np.arange(np.floor(primary.x.min()*scale), np.ceil(primary.x.max()*scale)+1)
    global_y = np.arange(np.floor(primary.y.min()*scale), np.ceil(primary.y.max()*scale)+1)
    coverage_max = max(100, np.histogram2d(primary.x*scale, primary.y*scale, bins=(global_x, global_y))[0].max())
    fig, axs = plt.subplots(2, 2, figsize=(10, 11), constrained_layout=True)
    bins_rows = []
    for col, side in enumerate(("left", "right")):
        s = primary[primary.side.eq(side)]
        arena = regions[regions.side.eq(side) & regions.region_type.eq("arena")].iloc[0]
        b = bounds(arena)*scale
        xe, ye = np.arange(np.floor(b[0])-2, np.ceil(b[2])+3), np.arange(np.floor(b[1])-2, np.ceil(b[3])+3)
        n = np.histogram2d(s.x*scale, s.y*scale, bins=(xe, ye))[0].T
        h = np.histogram2d(s.x*scale, s.y*scale, bins=(xe, ye), weights=s["hit_0.5"].astype(int))[0].T
        rate = np.divide(h, n, out=np.full_like(h, np.nan), where=n >= 25)
        for row, values in enumerate([rate*100, np.where(n > 0, n, np.nan)]):
            ax = axs[row, col]
            ax.imshow(bg, extent=extent, cmap="gray", alpha=.6)
            if row == 0:
                im = ax.pcolormesh(xe, ye, values, cmap="RdYlGn", vmin=0, vmax=100)
            else:
                im = ax.pcolormesh(xe, ye, values, cmap="magma", norm=LogNorm(1, coverage_max))
            outlines(ax, regions, side, scale)
            ax.set(xlim=(b[0]-2, b[2]+2), ylim=(b[3]+2, b[1]-2), aspect="equal", xlabel="panorama x (mm)", ylabel="panorama y (mm)")
            ax.set_title(f"{side.capitalize()}: " + ("tag detection (%)" if row == 0 else "SLEAP sample coverage"))
            fig.colorbar(im, ax=ax, shrink=.8, label="% with saved tag match" if row==0 else "sampled ant detections / 1 mm²")
        xx, yy = np.meshgrid((xe[1:]+xe[:-1])/2, (ye[1:]+ye[:-1])/2)
        bins_rows.append(pd.DataFrame(dict(side=side, x_mm=xx.ravel(), y_mm=yy.ravel(), n=n.ravel(), hits=h.ravel(), rate=rate.ravel())))
    fig.suptitle("Raw SLEAP detections with a saved same-camera ArUco match\n1 mm bins; gray background = <25 samples; one spatially chosen camera")
    fig.savefig(root / "01_spatial_detection.png", dpi=180)
    plt.close(fig)
    pd.concat(bins_rows).to_csv(root / "spatial_bins.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for side, color, offset in [("left", "#2878b5", -.12), ("right", "#d4562e", .12)]:
        tab = summary[summary.side.eq(side)].set_index("zone").reindex(ZONES)
        axs[0].errorbar(np.arange(len(ZONES))+offset, tab.rate*100,
                        yerr=np.vstack([tab.rate-tab.lower, tab.upper-tab.rate])*100,
                        fmt="o", color=color, capsize=3, label=side)
        s = primary[primary.side.eq(side)].copy()
        for ax, column, edges, limit in [(axs[1], "colony_distance_mm", np.arange(-10, 16, 1), (-10, 15)),
                                        (axs[2], "wall_distance_mm", np.arange(-2, 17, 1), (-2, 16))]:
            use = s if column.startswith("colony") else s[s.colony_distance_mm > 2]
            use = use.copy()
            use["distance_bin"] = pd.cut(use[column], edges, labels=False)
            use = use.dropna(subset=["distance_bin"])
            out = block_summary(use, ["distance_bin"])
            out = out[out.n >= 100]
            x = edges[out.distance_bin.astype(int)]+.5
            ax.plot(x, out.rate*100, color=color, label=side)
            ax.fill_between(x, out.lower*100, out.upper*100, color=color, alpha=.18)
            ax.set_xlim(limit)
            out["side"], out["distance_mm"] = side, x
            out.to_csv(root / f"{side}_{column}_profile.csv", index=False)
    axs[0].set_xticks(np.arange(len(ZONES)), ["Colony\ninterior", "Colony edge\n±2 mm", "Open\nfloor", "Outer wall\n±2 mm", "Food /\nwater"])
    axs[0].set_title("Region comparison")
    axs[1].axvspan(-2, 2, color="gray", alpha=.12)
    axs[1].axvline(0, color="gray", ls=":")
    axs[1].set(xlabel="Distance from colony boundary (mm)\nnegative = inside; positive = outside", title="Colony-edge gradient")
    axs[2].axvspan(-2, 2, color="gray", alpha=.12)
    axs[2].axvline(0, color="gray", ls=":")
    axs[2].set(xlabel="Distance inward from drawn arena boundary (mm)\nnegative = outside drawn rectangle", title="Outer-wall gradient (outside colony)")
    for ax in axs:
        ax.set(ylim=(0, 100), ylabel="SLEAP detections with saved tag match (%)")
        ax.legend(frameon=False)
    fig.suptitle("Are saved tag detections missing preferentially near walls?\n0.5 mm one-to-one match; 95% intervals resample 30-minute chunks")
    fig.savefig(root / "02_boundary_comparison.png", dpi=180)
    plt.close(fig)
    print(summary.to_string(index=False), flush=True)
    return primary


def read_sampled_track_chunk(args):
    dataset, chunk, step, output = args
    paths = sorted(Path(dataset).joinpath("tracks").glob(f"*chunk{chunk:03d}_*.parquet"))
    if len(paths) != 2:
        raise ValueError(f"Expected left and right finished tracks for chunk {chunk}; found {paths}")
    path = Path(output) / f"tracks_chunk{chunk:03d}.parquet"
    sig = dict(version=VERSION, paths=[fingerprint(p) for p in paths], step=step, seed=SEED)
    audit = path.with_suffix(".json")
    if path.exists() and audit.exists() and json.loads(audit.read_text()) == sig:
        return str(path)
    sampled = sample_frames(chunk, step, 43200)
    parts = []
    for p in paths:
        f = pq.ParquetFile(p)
        for i in range(f.num_row_groups):
            stat = f.metadata.row_group(i).column(f.schema.names.index("Frame")).statistics
            if stat is not None and stat.has_min_max and not ((sampled >= stat.min) & (sampled <= stat.max)).any():
                continue
            table = f.read_row_group(i, columns=["Frame", "Bodypoint", "SleapAnchorX", "SleapAnchorY", "SleapCam", "TrackID"], use_threads=False).to_pandas()
            table = table[table.Bodypoint.eq(0) & table.Frame.isin(sampled)].dropna(subset=["SleapAnchorX", "SleapAnchorY", "SleapCam"])
            parts.append(table)
    result = pd.concat(parts, ignore_index=True)
    result.to_parquet(path, index=False)
    audit.write_text(json.dumps(sig, indent=2))
    return str(path)


def compare_tracks(rows, dataset, step, scale, output, workers):
    cache = output / "sample_cache"
    tasks = [(str(dataset), int(chunk), step, str(cache)) for chunk in sorted(rows.chunk.unique())]
    paths = []
    with ProcessPoolExecutor(max_workers=min(4, workers)) as pool:
        for i, path in enumerate(pool.map(read_sampled_track_chunk, tasks), 1):
            paths.append(path)
            print(f"Finished tracking sampled: {i}/{len(tasks)} chunks", flush=True)
    rows = rows.copy()
    rows["retained"] = False
    rows["retained_same_camera"] = False
    rows["tracking_distance_mm"] = np.inf
    for args, path in zip(tasks, paths):
        chunk = args[1]
        tracks = pd.read_parquet(path)
        tracks["camera"] = tracks.SleapCam.astype(int)+1
        frame_groups = {int(frame): group for frame, group in tracks.groupby("Frame")}
        for frame, sample in rows[rows.chunk.eq(chunk)].groupby("frame"):
            t = frame_groups.get(int(frame))
            if t is None:
                continue
            distance, _ = cKDTree(t[["SleapAnchorX", "SleapAnchorY"]]).query(sample[["x", "y"]])
            rows.loc[sample.index, "tracking_distance_mm"] = distance*scale
            rows.loc[sample.index, "retained"] = distance*scale <= .25
        groups = {(int(frame), int(cam)): group for (frame, cam), group in tracks.groupby(["Frame", "camera"])}
        for (frame, cam), sample in rows[rows.chunk.eq(chunk)].groupby(["frame", "camera"]):
            t = groups.get((int(frame), int(cam)))
            if t is None:
                continue
            distance, _ = cKDTree(t[["SleapAnchorX", "SleapAnchorY"]]).query(sample[["x", "y"]])
            rows.loc[sample.index, "retained_same_camera"] = distance*scale <= .05
    primary = rows[rows.primary & rows.zone.ne("outside annotation")].copy()
    primary["tag_state"] = np.where(primary["hit_0.5"], "saved tag match", "no saved tag")
    tab = block_summary(primary, ["side", "zone", "tag_state"], hit="retained")
    tab.to_csv(output / "tracking_retention_by_tag_and_region.csv", index=False)
    sensitivity = []
    for radius in (.05, .15, .25, .5):
        primary["retention_sensitivity"] = primary.tracking_distance_mm <= radius
        check = block_summary(primary, ["side", "tag_state"], hit="retention_sensitivity")
        check["tracking_radius_mm"] = radius
        sensitivity.append(check)
    pd.concat(sensitivity).to_csv(output / "tracking_match_sensitivity.csv", index=False)
    rows.to_parquet(output / "sampled_detections_with_tracking.parquet", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, side in zip(axs, ("left", "right")):
        for state, color, offset in [("saved tag match", "#239b56", -.12), ("no saved tag", "#c0392b", .12)]:
            s = tab[tab.side.eq(side) & tab.tag_state.eq(state)].set_index("zone").reindex(ZONES)
            ax.errorbar(np.arange(len(ZONES))+offset, s.rate*100,
                        yerr=np.maximum(0, np.vstack([s.rate-s.lower, s.upper-s.rate])*100),
                        fmt="o", color=color, capsize=3, label=state)
        ax.set(ylim=(0, 100), title=side.capitalize(), ylabel="SLEAP positions present in finished tracking (%)")
        ax.set_xticks(np.arange(len(ZONES)), ["Colony\ninterior", "Colony\nedge", "Open\nfloor", "Outer\nwall", "Food /\nwater"])
        ax.legend(frameon=False)
    fig.suptitle("Do missing tag reads coincide with gaps in finished tracking?\nFinished SLEAP anchor within 0.25 mm in any camera; identity correctness is not assessed")
    fig.savefig(output / "03_tracking_retention.png", dpi=180)
    plt.close(fig)
    print(tab.to_string(index=False), flush=True)


def make_visibility_examples(rows, dataset, output):
    """Raw-camera crops, without labeling unseen tags as detector failures."""
    import cv2
    # One early common timestamp keeps this small and makes source decoding
    # auditable. This is an illustration, not a randomly sampled visibility study.
    first = rows[rows.chunk.eq(0)].frame.min()
    candidates = rows[rows.chunk.eq(0) & rows.frame.eq(first) & rows.primary & rows.score.ge(.9)]
    selected = []
    for zone in ("colony edge", "colony interior", "open floor", "outer wall"):
        group = candidates[candidates.zone.eq(zone) & ~candidates["hit_0.5"]]
        # Most isolated first: useful for spotting plainly visible missed tags.
        selected.extend(group.sort_values("nearest_ant_mm", ascending=False).head(2).index)
    selected.extend(candidates[candidates["hit_0.5"]].sort_values("nearest_ant_mm", ascending=False).head(2).index)
    chosen = rows.loc[selected].copy()
    chosen.to_csv(output / "visibility_example_coordinates.csv", index=False)
    if chosen.empty:
        return
    frames = {}
    for camera in chosen.camera.unique():
        video = sorted(Path(dataset).glob(f"cam{camera:02d}_*.mkv"))
        if len(video) != 1:
            raise ValueError(f"Cannot choose video for camera {camera}")
        cap = cv2.VideoCapture(str(video[0]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(first))
        ok, image = cap.read()
        reported = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.release()
        if not ok or abs(reported-(first+1)) > 1:
            raise ValueError(f"Could not decode frame {first} of {video[0]} (position {reported})")
        frames[camera] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Raw visibility frame decoded: camera {camera}, frame {first}", flush=True)
    fig, axs = plt.subplots(int(np.ceil(len(chosen)/4)), 4, figsize=(13, 3.6*int(np.ceil(len(chosen)/4))), squeeze=False)
    for ax in axs.flat:
        ax.set_axis_off()
    for ax, (idx, row) in zip(axs.flat, chosen.iterrows()):
        im = frames[row.camera]
        x, y = int(round(row.native_x)), int(round(row.native_y))
        x0, x1, y0, y1 = max(0, x-220), min(im.shape[1], x+220), max(0, y-220), min(im.shape[0], y+220)
        ax.imshow(im[y0:y1, x0:x1], extent=[x0-x, x1-x, y1-y, y0-y])
        ax.add_patch(Circle((0, 0), 75, fill=False, ec="cyan", lw=.7))
        state = "tag saved" if row["hit_0.5"] else "NO saved tag"
        ax.set_title(f"{row.zone} • {state}\ncam {int(row.camera):02d}; SLEAP score {row.score:.2f}", fontsize=10)
    fig.suptitle(f"Raw-camera checks: SLEAP tag anchor at circle center\nFrame {int(first)}; selected examples, not a visibility-rate estimate")
    fig.tight_layout(rect=(0, 0, 1, .94), h_pad=2.5)
    fig.savefig(output / "04_raw_visibility_examples.png", dpi=180)
    plt.close(fig)


def robustness_and_cross_camera(rows, output, scale=.016):
    """Compare like camera/time strata and audit other-view tag evidence."""
    primary = rows[rows.primary & rows.zone.ne("outside annotation")].copy()
    primary["tag_in_other_view"] = False
    primary["possible_duplicate_id_loss"] = False
    for (_, _), frame in rows.groupby(["chunk", "frame"]):
        tagged = frame[frame["hit_0.5"]]
        for camera, sample in frame[frame.primary & ~frame["hit_0.5"] & frame.zone.ne("outside annotation")].groupby("camera"):
            others = tagged[tagged.camera.ne(camera)]
            if others.empty:
                continue
            d, idx = cKDTree(others[["x", "y"]]).query(sample[["x", "y"]])
            supported = d*scale <= .25
            primary.loc[sample.index, "tag_in_other_view"] = supported
            # A conservative flag, not proof: the corroborating ID is already
            # stored on another animal in this camera. Hidden tags can also
            # coexist with the same ID elsewhere, so actual replay is needed.
            used_ids = set(tagged.loc[tagged.camera.eq(camera), "tag_id"])
            collisions = supported & np.isin(others.iloc[idx].tag_id.to_numpy(), list(used_ids))
            primary.loc[sample.index, "possible_duplicate_id_loss"] = collisions
    primary["tag_in_any_view"] = primary["hit_0.5"] | primary.tag_in_other_view
    block_summary(primary, ["side", "zone"], hit="tag_in_any_view").to_csv(output / "any_view_tag_rates.csv", index=False)
    missing = primary[~primary["hit_0.5"]]
    missing.groupby(["side", "zone"]).agg(n_missing=("tag_in_other_view", "size"),
        other_view_support=("tag_in_other_view", "sum"), possible_duplicate_id_loss=("possible_duplicate_id_loss", "sum")).to_csv(output / "missing_tag_cross_camera_audit.csv")
    primary.to_parquet(output / "primary_detections_cross_camera.parquet", index=False)
    strata = primary.groupby(["side", "camera", "chunk", "zone"])["hit_0.5"].agg(["sum", "size"]).reset_index()
    results = []
    for side in ("left", "right"):
        s = strata[strata.side.eq(side)]
        base = s[s.zone.eq("open floor")].drop(columns="zone")
        for zone in ("colony interior", "colony edge", "outer wall"):
            g = s[s.zone.eq(zone)].merge(base, on=["side", "camera", "chunk"], suffixes=("_zone", "_floor"))
            g = g[(g.size_zone >= 10) & (g.size_floor >= 10)].copy()
            if g.empty:
                continue
            g["weight"] = g.size_zone*g.size_floor/(g.size_zone+g.size_floor)
            g["weighted_delta"] = g.weight*(g.sum_zone/g.size_zone-g.sum_floor/g.size_floor)
            blocks = g.groupby("chunk")[["weight", "weighted_delta"]].sum().reindex(sorted(primary.chunk.unique()), fill_value=0)
            rng = np.random.default_rng(SEED)
            draw = rng.integers(len(blocks), size=(2000, len(blocks)))
            numerator = blocks.weighted_delta.to_numpy()[draw].sum(1)
            denominator = blocks.weight.to_numpy()[draw].sum(1)
            delta = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
            lo, hi = np.nanquantile(delta, [.025, .975])
            results.append(dict(side=side, zone=zone, difference_from_open_floor=g.weighted_delta.sum()/g.weight.sum(),
                                lower=lo, upper=hi, n_camera_chunk_strata=len(g), n_cameras=g.camera.nunique()))
    pd.DataFrame(results).to_csv(output / "within_camera_time_contrasts.csv", index=False)
    contribution = primary.groupby(["side", "zone"])["hit_0.5"].agg(n="size", matched="sum").reset_index()
    contribution["missing"] = contribution.n-contribution.matched
    contribution["share_of_all_missing"] = contribution.missing / contribution.groupby("side").missing.transform("sum")
    contribution.to_csv(output / "missing_tag_contributions.csv", index=False)
    rows[rows.primary].groupby(["side", "zone"])["hit_0.5"].agg(["size", "mean"]).to_csv(output / "arena_boundary_inclusion_audit.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--step-seconds", type=float, default=30)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunks", type=int, nargs="+")
    p.add_argument("--compare-tracks", action="store_true")
    p.add_argument("--examples", action="store_true")
    a = p.parse_args()
    root = a.dataset / "analysis_outputs" / "tag_detection_spatial"
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "sample_cache"
    cache.mkdir(exist_ok=True)
    meta = json.loads((a.dataset / "panorama_from_hmats_metadata.json").read_text())
    regions = load_panorama_regions(a.dataset / "panorama_regions.csv")
    scale = float(meta["mm_per_pixel"])
    hpath = Path(meta["homographies"])
    hmats = np.load(hpath)["H"]
    slps = sorted((a.dataset / "data").glob("cam*_*.slp"))
    if a.chunks:
        slps = [f for f in slps if int(f.stem[-3:]) in a.chunks]
    if not slps:
        raise FileNotFoundError("No raw SLEAP files")
    tasks = [(str(f), hmats, scale, int(round(a.step_seconds*24)), str(cache), fingerprint(hpath)) for f in slps]
    paths, audits = [], []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(process_camera, args) for args in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            path, audit = future.result()
            paths.append(path)
            audits.append({k:v for k,v in audit.items() if k != "signature"})
            if i == 1 or i % 25 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} camera-chunks sampled ({time.monotonic()-start:.0f}s)", flush=True)
    pd.DataFrame(audits).sort_values(["chunk", "camera"]).to_csv(root / "input_audit.csv", index=False)
    rows = add_regions(pd.concat([pd.read_parquet(f) for f in paths], ignore_index=True), regions, scale)
    rows.to_parquet(root / "sampled_detections.parquet", index=False)
    primary = make_results(rows, regions, scale, meta, root)
    robustness_and_cross_camera(rows, root, scale)
    if a.compare_tracks:
        compare_tracks(rows, a.dataset, int(round(a.step_seconds*24)), scale, root, a.workers)
    if a.examples:
        make_visibility_examples(rows, a.dataset, root)
    settings = dict(version=VERSION, dataset=str(a.dataset), step_seconds=a.step_seconds, fps=24,
                    radii_mm=RADII, score_min=.5, primary_radius_mm=.5, edge_width_mm=2, arena_margin_mm=2,
                    source_camera_chunks=len(tasks), chunks=sorted(rows.chunk.unique().tolist()),
                    rows=len(rows), primary_rows=len(primary), regions=fingerprint(a.dataset / "panorama_regions.csv"),
                    hmats=fingerprint(hpath), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    note="Conditional on SLEAP anchor detection; does not measure actual tag visibility or all true ants. No tracker/detector inputs modified.")
    (root / "settings.json").write_text(json.dumps(settings, indent=2))
    print(root, flush=True)


if __name__ == "__main__":
    main()
