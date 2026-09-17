"""Time-disjoint behavioral features from existing, fingerprinted analysis caches.

The spatial feature is an approximation made from cached one-second mean
positions, weighted by detected frames. It is not the original full-frame grid.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import grid_occupancy_utils as go
from analysis import return_sleep_utils as rs


@dataclass(frozen=True)
class FeatureSettings:
    day_seconds: int = 86400
    profile_bin_seconds: int = 7200
    min_day_coverage: float = .7
    min_bin_coverage: float = .5
    min_profile_bins: int = 9
    min_partner_onsets: int = 20

    def validate(self):
        if self.day_seconds != 86400:
            raise ValueError("This experiment compares complete 24-hour windows")
        if self.profile_bin_seconds <= 0 or self.day_seconds % self.profile_bin_seconds:
            raise ValueError("Profile bin size must divide 24 hours")
        if not 0 < self.min_day_coverage <= 1 or not 0 < self.min_bin_coverage <= 1:
            raise ValueError("Coverage thresholds must lie in (0, 1]")
        if not 1 <= self.min_profile_bins <= self.day_seconds // self.profile_bin_seconds:
            raise ValueError("Invalid minimum number of profile bins")


def mapped_source(path, block):
    """Map this block's published staging paths onto its current mount only."""
    block = Path(block)
    marker = f"/{block.parent.name}/{block.name}/"
    if marker not in str(path):
        raise ValueError(f"Source does not belong to the requested block: {path}")
    return block / str(path).split(marker, 1)[1]


def validate_stamp(stamp, path):
    stat = Path(path).stat()
    if (stat.st_size, stat.st_mtime_ns) != (stamp["size"], stamp["mtime_ns"]):
        raise ValueError(f"Source changed since the published analysis: {path}")


def load_sources(block, bundle):
    """A comparison manifest supplies the provenance of relocated context caches."""
    block, bundle = Path(block), Path(bundle)
    manifest_path = bundle / "comparison_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    suffix = f"/{block.parent.name}/{block.name}"
    entries = [w for w in manifest["windows"] if w["source"].endswith(suffix)]
    if len(entries) != 1:
        raise ValueError(f"Expected exactly one {suffix} window in {manifest_path}")
    info = entries[0]["info"]
    sources, contexts = [manifest_path], {}
    for stamp in info["sources"]:
        path = mapped_source(stamp["path"], block)
        validate_stamp(stamp, path)
        sources.append(path)
        if "/context/" in str(path):
            name = path.stem.rsplit("_", 1)[0] + ".parquet"
            if name in contexts:
                raise ValueError(f"Multiple context versions for {name}")
            contexts[name] = path
    region_path = go.panorama_regions_path(block)
    if go.load_panorama_regions(region_path).to_json(orient="records") != info["regions"]:
        raise ValueError("Annotations changed since the published analysis")
    grid = block / "stitched/grid_occupancy_histograms_arena"
    tracks = go.load_grid_tracks(grid)
    clusters = pd.read_csv(grid / "track_cluster_ids.csv").rename(columns={"TrackID": "track_id"})
    if clusters.duplicated(["side", "track_id"]).any():
        raise ValueError("Duplicate colony/tag identities in the occupancy handoff")
    tracks = tracks.merge(clusters[["side", "track_id", "track_name", "cluster_id"]],
                          on=["side", "track_id", "track_name"], validate="one_to_one")
    if len(tracks) != len(clusters):
        raise ValueError("Grid cache and cluster handoff disagree")
    for row in tracks.itertuples():
        if row.track_name not in contexts:
            raise FileNotFoundError(f"No fingerprinted context for {row.track_name}")
        meta = json.loads(Path(row.metadata_path).read_text())
        raw = block / "stitched/per_track" / row.track_name
        stamp = meta["arena_cache_inputs"]
        validate_stamp(dict(size=stamp["source_size"], mtime_ns=stamp["source_mtime_ns"]), raw)
        if meta.get("input_x_is_side_local") or meta.get("bodypoint_filter") != 0:
            raise ValueError("Expected global bodypoint-0 tracking coordinates")
        sources.extend([raw, Path(row.occupancy_path), Path(row.x_edges_path), Path(row.y_edges_path)])
    return tracks, info, contexts, sources


def load_contacts(block, info):
    """Reuse only the bout cache whose source hash matches the published fanout.

    The original staging path participates in the legacy cache key. Reconstruct
    that key using current file sizes/mtimes, instead of accepting an arbitrary
    similarly named cache after copying files between machines.
    """
    block = Path(block)
    root = rs.resolve_interaction_root(block)
    clock = pd.Timestamp(info["start_time"])
    clock_seconds = clock.hour * 3600 + clock.minute * 60 + clock.second - info["frame_start"] / info["fps"]
    chunks, run = rs.load_published_interaction_chunks(root, block / "tracks", fps=info["fps"],
                                                       start_clock_seconds=clock_seconds)
    settings = rs.ReturnSettings(**info["context_settings"])
    sources = [root / "run_manifest.json", root / "transfer_complete.ok"]
    for c in chunks:
        meta_path = c.interaction_path.with_suffix(".metadata.json")
        meta = json.loads(meta_path.read_text())
        if meta["parameters"] != run["parameters"] or c.interaction_path.stat().st_size != meta["output_size"]:
            raise ValueError(f"Interaction metadata does not match the published run: {meta_path}")
        validate_stamp(dict(size=meta["input_size"], mtime_ns=meta["input_mtime_ns"]), c.track_path)
        sources.extend([meta_path, c.interaction_path, c.track_path])
    candidates = set()
    for old_root in (str(root.resolve()), run["output_dir"], run["bucket_output_dir"]):
        stamps = []
        for c in chunks:
            stamp = rs.fingerprint(c.interaction_path)
            stamp["path"] = str(Path(old_root) / c.interaction_path.name)
            stamps.append(dict(offset=c.chunk_global_frame_offset, n_frames=c.chunk_frame_count, **stamp))
        key = rs.cache_key(dict(version=3, gap=settings.contact_gap_seconds, fps=settings.fps,
                                min_detection_frames=settings.contact_min_detection_frames, chunks=stamps))
        path = block / "stitched/analysis_cache/return_sleep" / f"pair_contact_bouts_{key}.parquet"
        if path.is_file():
            candidates.add(path)
    if len(candidates) != 1:
        raise FileNotFoundError(f"Need exactly one current, source-validated pair-contact cache; found {candidates}")
    path = candidates.pop()
    sources.append(path)
    bouts = pd.read_parquet(path)
    n = int(np.ceil(info["frame_stop"] / settings.fps))
    coverage = {side: np.zeros(n, bool) for side in ("left", "right")}
    for c in chunks:
        lo = int(np.ceil(c.chunk_global_frame_offset / settings.fps))
        hi = int((c.chunk_global_frame_offset + c.chunk_frame_count) // settings.fps)
        if coverage[c.side][lo:hi].any():
            raise ValueError("Overlapping contact chunks")
        coverage[c.side][lo:hi] = True
    return bouts, coverage, sources, run["run_id"]


def second_speed(path, metadata, n_seconds, fps):
    """Frame-weighted speed sums and valid-frame exposure on global seconds."""
    if metadata["fps"] != fps or fps != int(fps):
        raise ValueError("Speed/context FPS mismatch or noninteger FPS")
    speed = np.load(path, mmap_mode="r")
    if len(speed) != metadata["n_frames"]:
        raise ValueError(f"Speed length disagrees with metadata: {path}")
    sums, counts = np.zeros(n_seconds), np.zeros(n_seconds, np.int32)
    for start in range(0, len(speed), 500_000):
        values = np.asarray(speed[start:start + 500_000])
        frames = np.arange(len(values)) + start + int(metadata["frame_min"])
        valid = np.isfinite(values) & (values >= 0) & (frames < n_seconds * fps) & (frames >= 0)
        bins = frames[valid] // int(fps)
        sums += np.bincount(bins, weights=values[valid], minlength=n_seconds)
        counts += np.bincount(bins, minlength=n_seconds).astype(np.int32)
    mean = np.divide(sums, counts, out=np.full(n_seconds, np.nan), where=counts > 0)
    return mean, counts


def summarize(values, valid, lo, hi, weights=None):
    values, valid = np.asarray(values)[lo:hi], np.asarray(valid)[lo:hi]
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=None if weights is None else weights[lo:hi][valid]))


def contact_vectors(bouts, side, ant, n, fps):
    selected = bouts[bouts.side.eq(side) & (bouts.ant_a.eq(ant) | bouts.ant_b.eq(ant))].copy()
    selected["partner"] = np.where(selected.ant_a.eq(ant), selected.ant_b, selected.ant_a)
    start = selected.start_frame.to_numpy(np.int64) // int(fps)
    stop = selected.end_frame.to_numpy(np.int64) // int(fps) + 1
    delta = np.zeros(n + 1, np.int32)
    np.add.at(delta, np.clip(start, 0, n), 1)
    np.add.at(delta, np.clip(stop, 0, n), -1)
    active = np.cumsum(delta[:-1]) > 0
    return selected, active


def extract_ant(row, block, info, context_path, bouts, interaction_coverage, settings):
    with np.load(context_path) as data:
        context = {name: data[name] for name in data.files}
    fps = info["fps"]
    if info["context_settings"]["position_bin_seconds"] != 1:
        raise ValueError("Feature extraction requires existing one-second contexts")
    n = len(context["inside"])
    speed_root = block / "stitched/speed_vectors/per_track" / Path(row.track_name).stem
    speed_meta = json.loads((speed_root / "speed_metadata.json").read_text())
    speed, speed_count = second_speed(speed_root / "speed_mm_s.npy", speed_meta, n, fps)
    position_valid = np.isfinite(context["x_mm"]) & np.isfinite(context["y_mm"])
    contact_valid = position_valid & interaction_coverage[row.side]
    selected, active = contact_vectors(bouts, row.side, row.track_id, n, fps)
    new = selected[selected.is_new_onset].copy()
    new["second"] = new.start_frame.to_numpy(np.int64) // int(fps)
    new = new[new.second.between(0, n - 1)]
    new = new[contact_valid[new.second.to_numpy()]]
    onsets = np.bincount(new.second.to_numpy(), minlength=n)
    vectors = dict(speed=speed, body=context["body_speed"], antenna=context["antenna_speed"],
                   sleep=(context["state"] == 1).astype(float), outside=(context["inside"] == 0).astype(float),
                   contacts=onsets.astype(float) * 3600, bout_presence=active.astype(float))
    valid = {key: np.isfinite(value) for key, value in vectors.items()}
    valid.update(sleep=context["state"] >= 0, outside=context["inside"] >= 0,
                 contacts=contact_valid, bout_presence=contact_valid)
    # Speed coverage is valid frame exposure; context motion coverage is valid seconds.
    weights = {"speed": speed_count}
    start = int(np.ceil(info["frame_start"] / fps))
    x_edges, y_edges = np.load(row.x_edges_path), np.load(row.y_edges_path)
    meta = json.loads(Path(row.metadata_path).read_text())
    x = context["x_mm"] - meta["input_x_origin_px"] * meta["mm_per_px"]
    y = context["y_mm"] - meta["y_origin_px"] * meta["mm_per_px"]
    bounds = meta["arena_bounds_px"]
    in_arena = position_valid & (x >= 0) & (y >= 0)
    in_arena &= x <= (bounds["x_max_px"] - bounds["x_min_px"]) * meta["mm_per_px"]
    in_arena &= y <= (bounds["y_max_px"] - bounds["y_min_px"]) * meta["mm_per_px"]
    days, profiles, histograms = [], [], []
    for day in range(2):
        lo, hi = start + day * settings.day_seconds, start + (day + 1) * settings.day_seconds
        daily = dict(side=row.side, track_id=row.track_id, track_name=row.track_name,
                     original_cluster=row.cluster_id, day=day, start_second=lo, stop_second=hi)
        for key, values in vectors.items():
            daily[key] = summarize(values, valid[key], lo, hi, weights.get(key))
            daily[key + "_coverage"] = float(np.mean(valid[key][lo:hi]))
        daily["speed_coverage"] = float(speed_count[lo:hi].sum() / ((hi - lo) * fps))
        daily["position_coverage"] = float(position_valid[lo:hi].mean())
        daily["detected_frame_fraction"] = float(context["position_count"][lo:hi].sum() / ((hi - lo) * fps))
        counts = new.loc[new.second.ge(lo) & new.second.lt(hi), "partner"].value_counts()
        daily["onsets"] = int(counts.sum())
        daily["partner_diversity"] = (float(1 - np.square(counts / counts.sum()).sum())
                                        if counts.sum() >= settings.min_partner_onsets else np.nan)
        complete = selected[selected.is_new_onset & selected.start_frame.ge(lo * fps)
                            & selected.end_frame.lt(hi * fps)]
        durations = (complete.end_frame - complete.start_frame + 1) / fps
        daily["median_bout_seconds"] = float(durations.median()) if len(durations) else np.nan
        daily["unique_partners"] = len(counts)
        # This presence outcome is not used as an input feature.
        exposure = context["position_count"][lo:hi].sum() / fps
        daily["resource_fraction"] = float(context["resource_seconds"][lo:hi].sum() / exposure) if exposure else np.nan
        keep = in_arena[lo:hi]
        h = np.histogram2d(y[lo:hi][keep], x[lo:hi][keep], bins=(y_edges, x_edges),
                           weights=context["position_count"][lo:hi][keep])[0]
        histograms.append((h / h.sum() if h.sum() else h).ravel())
        daily["in_arena_fraction"] = float(context["position_count"][lo:hi][keep].sum() / (exposure * fps)) if exposure else np.nan
        for slot, a in enumerate(range(lo, hi, settings.profile_bin_seconds)):
            z = a + settings.profile_bin_seconds
            profile = dict(side=row.side, track_id=row.track_id, day=day, slot=slot)
            for key in ("speed", "sleep", "contacts"):
                cov = (speed_count[a:z].sum() / ((z - a) * fps) if key == "speed"
                       else float(valid[key][a:z].mean()))
                profile[key] = (summarize(vectors[key], valid[key], a, z, weights.get(key))
                                if cov >= settings.min_bin_coverage else np.nan)
                profile[key + "_coverage"] = cov
            profiles.append(profile)
        days.append(daily)
    return days, profiles, np.stack(histograms)


def extract_features(block, bundle, output, settings=FeatureSettings(), workers=4):
    settings.validate()
    block, output = Path(block).resolve(), Path(output)
    tracks, info, contexts, sources = load_sources(block, bundle)
    if info["frame_stop"] / info["fps"] < np.ceil(info["frame_start"] / info["fps"]) + 2 * settings.day_seconds:
        raise ValueError("Need at least 48 hours for two clock-matched 24-hour windows")
    bouts, coverage, contact_sources, run_id = load_contacts(block, info)
    sources.extend(contact_sources)
    sources.append(Path(__file__))
    signature = dict(settings=asdict(settings), sources=[rs.fingerprint(p) for p in sorted(set(sources))])
    key = rs.cache_key(signature)
    cache = output / "cache" / key
    if (cache / "features.npz").exists():
        print(f"Feature cache hit: {cache}", flush=True)
    else:
        cache.mkdir(parents=True, exist_ok=True)
        def read(row):
            return extract_ant(row, block, info, contexts[row.track_name], bouts, coverage, settings)
        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(read, tracks.itertuples()), 1):
                results.append(result)
                if i == 1 or i % 10 == 0 or i == len(tracks):
                    print(f"Behavioral features: {i}/{len(tracks)} ants", flush=True)
        pd.DataFrame([r for result in results for r in result[0]]).to_csv(cache / "daily_features.csv", index=False)
        pd.DataFrame([r for result in results for r in result[1]]).to_csv(cache / "clock_profiles.csv", index=False)
        arrays = {}
        for side, group in tracks.groupby("side", sort=True):
            arrays[side] = np.stack([results[i][2] for i in group.index])
        np.savez_compressed(cache / "features.npz", **arrays)
        (cache / "sources.json").write_text(json.dumps(signature, indent=2) + "\n")
    daily = pd.read_csv(cache / "daily_features.csv")
    profiles = pd.read_csv(cache / "clock_profiles.csv")
    with np.load(cache / "features.npz") as data:
        spatial = {k: data[k] for k in data.files}
    audit = daily.copy()
    metrics = ["position", "speed", "body", "antenna", "sleep", "contacts"]
    audit["daily_coverage_ok"] = audit[[m + "_coverage" for m in metrics]].ge(settings.min_day_coverage).all(axis=1)
    support = profiles.groupby(["side", "track_id", "day"])[["speed", "sleep", "contacts"]].count()
    support = support.rename(columns=lambda c: c + "_profile_bins").reset_index()
    audit = audit.merge(support, on=["side", "track_id", "day"], validate="one_to_one")
    audit["profile_coverage_ok"] = audit[[m + "_profile_bins" for m in ("speed", "sleep", "contacts")]].ge(settings.min_profile_bins).all(axis=1)
    audit["eligible_day"] = audit.daily_coverage_ok & audit.profile_coverage_ok
    audit["included"] = audit.groupby(["side", "track_id"]).eligible_day.transform("all")
    audit.to_csv(output / "coverage_audit.csv", index=False)
    daily.to_csv(output / "daily_features.csv", index=False)
    profiles.to_csv(output / "clock_profiles.csv", index=False)
    return dict(tracks=tracks, daily=daily, profiles=profiles, spatial=spatial, audit=audit, info=info,
                feature_cache=str(cache), feature_signature=key, contact_run_id=run_id, settings=asdict(settings))
