"""Exact half-hour spatial counts from finished tracks, for temporal analysis.

Run one task per ant on a Slurm array. Each task reads that ant's available
blocks once. Existing grid geometry and load_track_xy frame averaging are
reused; summed counts must reproduce the published whole-block histogram.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis import compute_track_grid_occupancy as compute


def stamp(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def temporal_counts(position, meta, x_edges, y_edges, origin_seconds, fps, first_bin, n_bins, width=1800):
    """Same endpoint/arena clipping and detected-frame normalization as grids."""
    if not float(fps).is_integer():
        raise ValueError("Exact integer calendar binning requires integral fps")
    fps = int(fps)
    frames = position.Frame.to_numpy(np.int64)
    x, y = position.X.to_numpy(float), position.Y.to_numpy(float)
    finite = np.isfinite(x) & np.isfinite(y)
    atom = (frames + int(origin_seconds)*fps) // (width*fps) - first_bin
    if ((atom < 0) | (atom >= n_bins)).any():
        raise ValueError("Track frames extend beyond the recorded time window")
    detected = np.bincount(atom[finite], minlength=n_bins).astype(np.int64)
    bounds = meta["arena_bounds_px"]
    xm = (x - meta["input_x_origin_px"]) * meta["mm_per_px"]
    ym = (y - meta["y_origin_px"]) * meta["mm_per_px"]
    valid = finite & (x >= bounds["x_min_px"]) & (x <= bounds["x_max_px"]) & (y >= bounds["y_min_px"]) & (y <= bounds["y_max_px"])
    valid &= (xm >= x_edges[0]) & (xm <= x_edges[-1]) & (ym >= y_edges[0]) & (ym <= y_edges[-1])
    nx, ny = len(x_edges)-1, len(y_edges)-1
    ix = np.minimum(np.searchsorted(x_edges, xm[valid], side="right")-1, nx-1)
    iy = np.minimum(np.searchsorted(y_edges, ym[valid], side="right")-1, ny-1)
    flat = (atom[valid]*ny + iy)*nx + ix
    counts = np.bincount(flat, minlength=n_bins*ny*nx).reshape(n_bins, ny, nx).astype(np.int64)
    return counts, detected


def extract(entry, output):
    root = Path(entry["block"]) / "stitched/grid_occupancy_histograms_arena/per_track" / Path(entry["track_name"]).stem
    metadata_path = root / "grid_occupancy_metadata.json"
    meta = json.loads(metadata_path.read_text())
    track = Path(entry["block"]) / "stitched/per_track" / entry["track_name"]
    inputs = meta["arena_cache_inputs"]
    stat = track.stat()
    if (stat.st_size, stat.st_mtime_ns) != (inputs["source_size"], inputs["source_mtime_ns"]):
        raise ValueError(f"Finished tracking changed since grid generation: {track}")
    files = [track, metadata_path, root/"grid_occupancy_f4.npy", root/"grid_x_edges_mm.npy", root/"grid_y_edges_mm.npy"]
    signature = dict(entry=entry, inputs=[stamp(p) for p in files], width_seconds=1800,
                     code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(compute.__file__))})
    target = Path(output) / "atoms" / str(entry["block_index"]) / (entry["ant"].replace(":", "_") + ".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = target.with_suffix(".json")
    if target.is_file() and manifest_path.is_file():
        saved = json.loads(manifest_path.read_text())
        if saved.get("signature") == signature:
            print("ATOM_CACHE_HIT", entry["ant"], entry["block_index"], flush=True)
            return
    position = compute.load_track_xy(track, frame_col="Frame", x_col=meta["x_col"], y_col=meta["y_col"], bodypoint=meta["bodypoint_filter"])
    start, stop = pd.Timestamp(entry["start"]), pd.Timestamp(entry["stop"])
    origin = start - pd.Timedelta(seconds=entry["frame_start"] / entry["fps"])
    if origin.value % 1_000_000_000:
        raise ValueError("Expected a whole-second recording clock origin")
    first, last = start.value // (1800*10**9), stop.value // (1800*10**9)
    x, y = np.load(root/"grid_x_edges_mm.npy"), np.load(root/"grid_y_edges_mm.npy")
    counts, detected = temporal_counts(position, meta, x, y, origin.value//10**9, entry["fps"], first, last-first+1)
    original = np.load(root/"grid_occupancy_f4.npy")
    combined = (counts.sum(axis=0) / detected.sum()).astype(np.float32)
    np.testing.assert_array_equal(combined, original, err_msg=f"Temporal counts do not reproduce original grid: {track}")
    assert detected.sum() == meta["n_detected_frames"]
    assert counts.sum() == meta["n_in_grid_frames"]
    # Detect upstream changes during extraction as well.
    assert [stamp(p) for p in files] == signature["inputs"]
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, counts=counts.astype(np.int32), detected=detected,
                        calendar_bin=np.arange(first, last+1), x_edges=x, y_edges=y)
    temporary.replace(target)
    result = dict(signature=signature, n_detected=int(detected.sum()), n_in_arena=int(counts.sum()),
                  atoms=len(detected), original_histogram_exact=True)
    manifest_path.write_text(json.dumps(result, indent=2)+"\n")
    print("ATOM_CACHE_COMPLETE", entry["ant"], entry["block_index"], len(detected), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(args.tasks.read_text())[args.task_index]
    for entry in task["entries"]:
        extract(entry, args.output)
    print("ANT_COMPLETE", task["ant"], flush=True)


if __name__ == "__main__":
    main()
