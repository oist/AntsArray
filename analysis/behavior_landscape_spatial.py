"""Exact frame-level spatial baseline for the held-out landscape experiment."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import return_sleep_utils as rs


def frame_histograms(frame, x_px, y_px, meta, x_edges, y_edges, windows):
    """Canonical arena coordinates and fraction-of-all-detections normalization."""
    x = (x_px - meta["input_x_origin_px"]) * meta["mm_per_px"]
    y = (y_px - meta["y_origin_px"]) * meta["mm_per_px"]
    b = meta["arena_bounds_px"]
    finite = np.isfinite(x) & np.isfinite(y)
    inside = finite & (x_px >= b["x_min_px"]) & (x_px <= b["x_max_px"])
    inside &= (y_px >= b["y_min_px"]) & (y_px <= b["y_max_px"])
    histograms, detected = [], []
    for lo, hi in windows:
        time = (frame >= lo) & (frame < hi)
        count = int((time & finite).sum())
        valid = time & inside
        h = np.histogram2d(y[valid], x[valid], bins=(y_edges, x_edges))[0]
        histograms.append((h / count if count else h).ravel())
        detected.append(count)
    return np.stack(histograms), np.array(detected)


def use_exact_spatial(features, block, output, workers=2):
    """Read finished bodypoint-0 positions once; cache per-ant daily histograms.

    Also reconstruct the whole-recording histogram and require it to agree with
    the saved canonical grid. This validates normalization, coordinates and the
    handling of duplicate frames before any held-out comparison is accepted.
    """
    import pyarrow.dataset as ds

    block, output = Path(block), Path(output)
    fps = features["info"]["fps"]
    start = int(np.ceil(features["info"]["frame_start"] / fps) * fps)
    day = int(86400 * fps)
    windows = [(start, start + day), (start + day, start + 2 * day),
               (features["info"]["frame_start"], features["info"]["frame_stop"])]
    cache = output / "cache/exact_spatial"
    cache.mkdir(parents=True, exist_ok=True)

    def read(row):
        path = block / "stitched/per_track" / row.track_name
        sources = [path, Path(row.metadata_path), Path(row.x_edges_path), Path(row.y_edges_path),
                   Path(row.occupancy_path), Path(__file__)]
        signature = dict(windows=windows, sources=[rs.fingerprint(p) for p in sources])
        key = rs.cache_key(signature)
        cached = cache / f"{Path(row.track_name).stem}_{key}.npz"
        if cached.is_file():
            with np.load(cached) as data:
                return data["histograms"], data["detected"], float(data["maximum_error"]), key
        data = ds.dataset(path, format="parquet").to_table(
            columns=["Frame", "TrackX", "TrackY"], filter=ds.field("Bodypoint") == 0).to_pandas()
        data = data.dropna(subset=["Frame", "TrackX", "TrackY"])
        data["Frame"] = data.Frame.round().astype(np.int64)
        if data.Frame.duplicated().any():
            data = data.groupby("Frame", as_index=False)[["TrackX", "TrackY"]].mean()
        meta = json.loads(Path(row.metadata_path).read_text())
        histograms, detected = frame_histograms(data.Frame.to_numpy(), data.TrackX.to_numpy(),
            data.TrackY.to_numpy(), meta, np.load(row.x_edges_path), np.load(row.y_edges_path), windows)
        original = np.load(row.occupancy_path).ravel()
        error = float(np.abs(histograms[2] - original).max())
        if not np.allclose(histograms[2], original, atol=2e-8, rtol=1e-6):
            raise ValueError(f"Exact spatial reconstruction disagrees with original grid ({error:g}): {path}")
        np.savez_compressed(cached, histograms=histograms, detected=detected, maximum_error=error)
        cached.with_suffix(".json").write_text(json.dumps(signature, indent=2) + "\n")
        return histograms, detected, error, key

    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 2))) as pool:
        for i, result in enumerate(pool.map(read, features["tracks"].itertuples()), 1):
            results.append(result)
            if i == 1 or i % 10 == 0 or i == len(features["tracks"]):
                print(f"Exact spatial windows: {i}/{len(features['tracks'])} ants", flush=True)
    features["spatial_proxy"] = features["spatial"]
    features["spatial"] = {}
    features["exact_spatial_counts"] = {}
    features["exact_spatial_whole"] = {}
    for side, group in features["tracks"].groupby("side"):
        features["spatial"][side] = np.stack([results[i][0][:2] for i in group.index])
        features["exact_spatial_counts"][side] = np.stack([results[i][1][:2] for i in group.index])
        features["exact_spatial_whole"][side] = np.stack([results[i][0][2] for i in group.index])
    audit = features["tracks"][["side", "track_id", "track_name"]].copy()
    audit["maximum_whole_grid_error"] = [r[2] for r in results]
    audit["source_key"] = [r[3] for r in results]
    audit.to_csv(output / "exact_spatial_audit.csv", index=False)
    features["exact_spatial_keys"] = audit.source_key.tolist()
    return features
