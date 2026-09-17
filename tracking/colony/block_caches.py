"""Combine the pipeline's versioned per-ant caches on a shared block timeline.

Temporal caches retain their original within-block estimates. Unrecorded time
is unknown; no velocity, interpolation, or sleep bout is inferred across a
recording boundary. Geometry caches can be rebuilt using common annotations.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from analysis.sleep_motion_utils import load_sleep_motion_cache, _atomic_write_json
from tracking.stitch_tracks import concatenate_shifted_parquets


CACHE_METADATA = {
    "speed_vectors": "speed_metadata.json",
    "colony_presence_vectors": "colony_presence_metadata.json",
    "grid_occupancy_histograms": "grid_occupancy_metadata.json",
    "sleep_motion": "sleep_motion_metadata.json",
    "sleep_motion_labels": "sleep_motion_label_metadata.json",
    "sleep_predictions": "sleep_prediction_metadata.json",
}
PARAMETERS = {
    "speed_vectors": ("fps", "mm_per_px", "x_col", "y_col", "bodypoint_filter",
                      "max_interp_gap_frames", "smooth_sigma_frames", "max_speed_mm_s"),
    "colony_presence_vectors": ("mm_per_px", "x_col", "y_col", "bodypoint_filter", "colony_boxes_mm"),
    "sleep_motion": ("fps", "mm_per_px", "cache_max_gap_frames", "format_version", "speed_assignment"),
    "sleep_motion_labels": ("fps", "classifier_type", "classifier_parameters"),
    "sleep_predictions": ("fps", "mm_per_px", "model_path", "model_feature_set", "feature_mode",
                          "n_model_features", "speed_windows_seconds"),
    "grid_occupancy_histograms": ("mm_per_px", "grid_size_mm", "grid_pad_mm", "x_col", "y_col",
                                 "bodypoint_filter", "input_x_origin_px", "y_origin_px",
                                 "input_x_is_side_local", "same_shape_sides"),
}
# file, unknown value, metadata path key
DENSE = {
    "speed_vectors": [("speed_mm_s.npy", np.nan, "speed_path")],
    "colony_presence_vectors": [("colony_presence_i1.npy", -1, "presence_path")],
    "sleep_motion_labels": [("sleep_state_i1.npy", -1, None), ("quiet_fraction_f2.npy", np.nan, None),
                            ("valid_fraction_f2.npy", 0, None)],
    "sleep_predictions": [("sleep_probability_f4.npy", np.nan, "sleep_probability_path"),
                          ("wake_probability_f4.npy", np.nan, "wake_probability_path"),
                          ("predicted_sleep_i1.npy", -1, "predicted_sleep_path")],
}


def cache_segments(ant, kind):
    result = []
    for source in ant["sources"]:
        cache = source["caches"].get(kind)
        if cache:
            path = Path(cache["metadata"]["path"])
            meta = json.loads(path.read_text())
            name = meta.get("track_name", meta.get("summary", {}).get("track_name"))
            if name is not None and name != Path(source["track"]["path"]).name:
                raise ValueError(f"Cache identity mismatch: {path}")
            result.append((source, path.parent, meta))
    return result


def require_compatible(segments, kind):
    if not segments:
        return
    first = {key: segments[0][2].get(key) for key in PARAMETERS[kind]}
    for _, path, meta in segments[1:]:
        changed = [key for key, value in first.items() if meta.get(key) != value]
        if changed:
            raise ValueError(f"Incompatible {kind} parameters at {path}: {', '.join(changed)}")


def common_metadata(segments, ant, group, final_root):
    result = copy.deepcopy(segments[0][2])
    result.update(track_name=ant["name"], track_id=ant["track_id"], side=ant["side"],
                  track_path=str(final_root / "per_track" / ant["name"]),
                  frame_min=0, frame_max=group["num_frames"] - 1, n_frames=group["num_frames"],
                  fps=group["fps"], start_datetime=group["start_datetime"],
                  generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  combination_method="preserve_within_block_estimates",
                  boundary_policy="unknown gaps; no cross-block interpolation or bout merging",
                  source_segments=[dict(block=s["block"], frame_offset=s["frame_offset"], source_cache=str(path))
                                   for s, path, _ in segments])
    return result


def merge_dense(segments, kind, ant, group, root, final_root):
    require_compatible(segments, kind)
    out = root / kind / "per_track" / Path(ant["name"]).stem
    final = final_root / kind / "per_track" / out.name
    out.mkdir(parents=True, exist_ok=True)
    meta = common_metadata(segments, ant, group, final_root)
    vectors = {}
    for filename, fill, path_key in DENSE[kind]:
        arrays = [np.load(path / filename, mmap_mode="r") for _, path, _ in segments]
        dtype = np.result_type(*(array.dtype for array in arrays))
        temporary = out / (filename + ".tmp")
        merged = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=(group["num_frames"],))
        merged[:] = fill
        for (source, path, part_meta), array in zip(segments, arrays):
            local_min = part_meta.get("frame_min")
            local_max = part_meta.get("frame_max")
            if len(array) == 0 and local_min is None:
                continue
            if array.ndim != 1 or len(array) != int(local_max) - int(local_min) + 1:
                raise ValueError(f"Dense cache span/shape mismatch: {path / filename}")
            start = source["frame_offset"] + int(local_min)
            stop = start + len(array)
            if int(local_min) < 0 or int(local_max) >= source["num_frames"] or stop > len(merged):
                raise ValueError(f"Cache extends outside its recording: {path / filename}")
            merged[start:stop] = array
        merged.flush()
        del merged
        temporary.replace(out / filename)
        vectors[filename] = np.load(out / filename, mmap_mode="r")
        if path_key:
            meta[path_key] = str(final / filename)
    counts = lambda key: sum(int(m.get(key, 0)) for _, _, m in segments)
    if kind == "speed_vectors":
        meta.update(n_observed_frames=counts("n_observed_frames"),
                    n_valid_speed_frames=int(np.isfinite(vectors["speed_mm_s.npy"]).sum()))
    elif kind == "colony_presence_vectors":
        values = vectors["colony_presence_i1.npy"]
        inside, outside = int((values == 1).sum()), int((values == 0).sum())
        meta.update(n_observed_frames=counts("n_observed_frames"), n_valid_position_frames=inside + outside,
                    n_inside_colony_frames=inside, n_outside_colony_frames=outside,
                    inside_colony_frac_valid=inside / (inside + outside) if inside + outside else None)
    elif kind in {"sleep_motion_labels", "sleep_predictions"}:
        filename = "sleep_state_i1.npy" if kind == "sleep_motion_labels" else "predicted_sleep_i1.npy"
        values = vectors[filename]
        sleep, wake = int((values == 1).sum()), int((values == 0).sum())
        tables = {"sleep_bouts.parquet": ("frame_start", "frame_end")}
        if kind == "sleep_predictions":
            tables["sleep_predictions.parquet"] = ("Frame",)
            if any((path / "sleep_features.parquet").exists() for _, path, _ in segments):
                tables["sleep_features.parquet"] = ("Frame",)
        table_counts = {}
        for filename, frame_columns in tables.items():
            inputs = [(path / filename, source["frame_offset"], source["block"])
                      for source, path, _ in segments if (path / filename).is_file()]
            if len(inputs) != len(segments) and filename != "sleep_features.parquet":
                raise ValueError(f"Missing {filename} in {kind}")
            table_counts[filename] = concatenate_shifted_parquets(
                inputs, out / filename, frame_columns=frame_columns, track_name=ant["name"])["rows"]
        if kind == "sleep_motion_labels":
            meta["summary"] = dict(track_name=ant["name"], track_id=ant["track_id"], side=ant["side"],
                                   frame_min=0, frame_max=group["num_frames"] - 1, n_frames=group["num_frames"],
                                   n_cached_frames=sum(int(m["summary"]["n_cached_frames"]) for _, _, m in segments),
                                   n_sleep_frames=sleep, n_wake_frames=wake,
                                   n_unknown_frames=int((values == -1).sum()),
                                   sleep_fraction_classified=sleep / (sleep + wake) if sleep + wake else None,
                                   n_sleep_bouts=table_counts["sleep_bouts.parquet"])
            motion = root / "sleep_motion/per_track" / out.name / CACHE_METADATA["sleep_motion"]
            if motion.is_file():
                stat = motion.stat()
                meta.update(source_sleep_motion_cache=str(final_root / "sleep_motion/per_track" / out.name),
                            source_metadata_mtime_ns=stat.st_mtime_ns, source_metadata_size_bytes=stat.st_size)
            else:
                for key in ("source_sleep_motion_cache", "source_metadata_mtime_ns", "source_metadata_size_bytes"):
                    meta.pop(key, None)
        else:
            probabilities = vectors["sleep_probability_f4.npy"]
            meta.update(n_predicted_frames=sleep + wake, n_sleep_frames=sleep, n_wake_frames=wake,
                        sleep_fraction_predicted_frames=sleep / (sleep + wake) if sleep + wake else None,
                        mean_sleep_probability=float(np.nanmean(probabilities)) if sleep + wake else None,
                        prediction_table_path=str(final / "sleep_predictions.parquet"),
                        bouts_path=str(final / "sleep_bouts.parquet"),
                        features_path=str(final / "sleep_features.parquet") if "sleep_features.parquet" in tables else None,
                        speed_root=str(final_root / "speed_vectors"))
    _atomic_write_json(out / CACHE_METADATA[kind], meta)
    return {"action": "merged", "blocks": [s["block"] for s, _, _ in segments]}


def merge_motion(segments, ant, group, root, final_root):
    require_compatible(segments, "sleep_motion")
    out = root / "sleep_motion/per_track" / Path(ant["name"]).stem
    out.mkdir(parents=True, exist_ok=True)
    caches = [load_sleep_motion_cache(path) for _, path, _ in segments]
    ids = np.unique(np.concatenate([cache.bodypoint_ids for cache in caches])).astype(np.int16)
    nrows = sum(len(cache.frames) for cache in caches)
    speed = np.lib.format.open_memmap(out / "bodypoint_speed_mm_s.npy", mode="w+", dtype="float32", shape=(nrows, len(ids)))
    gaps = np.lib.format.open_memmap(out / "bodypoint_frame_gap_u1.npy", mode="w+", dtype="uint8", shape=speed.shape)
    frames = np.lib.format.open_memmap(out / "frames.npy", mode="w+", dtype="int64", shape=(nrows,))
    speed[:] = np.nan
    gaps[:] = 0
    cursor = 0
    for (source, path, _), cache in zip(segments, caches):
        count = len(cache.frames)
        if count and (cache.frames[0] < 0 or cache.frames[-1] >= source["num_frames"] or np.any(np.diff(cache.frames) <= 0)):
            raise ValueError(f"Invalid sparse cache frames: {path}")
        frames[cursor:cursor + count] = cache.frames + source["frame_offset"]
        for column, bodypoint in enumerate(cache.bodypoint_ids):
            target = int(np.searchsorted(ids, bodypoint))
            speed[cursor:cursor + count, target] = cache.speed_mm_s[:, column]
            gaps[cursor:cursor + count, target] = cache.frame_gap[:, column]
        cursor += count
    valid = {str(int(bodypoint)): int(np.isfinite(speed[:, i]).sum()) for i, bodypoint in enumerate(ids)}
    for array in (speed, gaps, frames):
        array.flush()
    del speed, gaps, frames
    np.save(out / "bodypoint_ids.npy", ids)
    meta = common_metadata(segments, ant, group, final_root)
    stat = (root / "per_track" / ant["name"]).stat()
    meta.update(bodypoint_ids=ids.tolist(), n_bodypoints=len(ids), n_cached_frames=nrows,
                speed_shape=[nrows, len(ids)], n_valid_speed_values=sum(valid.values()),
                valid_speed_values_by_bodypoint=valid, source_size_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns)
    for key in ("n_input_pose_rows", "n_duplicate_pose_rows_removed"):
        meta[key] = sum(int(m.get(key, 0)) for _, _, m in segments)
    _atomic_write_json(out / CACHE_METADATA["sleep_motion"], meta)
    return {"action": "merged", "blocks": [s["block"] for s, _, _ in segments], "cached_frames": nrows}


def merge_grid(segments, ant, group, root, final_root):
    require_compatible(segments, "grid_occupancy_histograms")
    out = root / "grid_occupancy_histograms/per_track" / Path(ant["name"]).stem
    out.mkdir(parents=True, exist_ok=True)
    x = np.load(segments[0][1] / "grid_x_edges_mm.npy")
    y = np.load(segments[0][1] / "grid_y_edges_mm.npy")
    weighted = np.zeros((len(y) - 1, len(x) - 1), dtype=np.float64)
    for _, path, meta in segments:
        if not np.array_equal(x, np.load(path / "grid_x_edges_mm.npy")) or not np.array_equal(y, np.load(path / "grid_y_edges_mm.npy")):
            raise ValueError("Grid edges differ; provide shared panorama annotations to rebuild occupancy")
        weighted += np.load(path / "grid_occupancy_f4.npy") * int(meta["n_detected_frames"])
    meta = common_metadata(segments, ant, group, final_root)
    for key in ("n_detected_frames", "n_observed_frames", "n_in_grid_frames", "n_out_of_grid_detected_frames"):
        meta[key] = sum(int(m[key]) for _, _, m in segments)
    if meta["n_detected_frames"]:
        weighted /= meta["n_detected_frames"]
    final = final_root / "grid_occupancy_histograms/per_track" / out.name
    for name, array, key in [("grid_occupancy_f4.npy", weighted.astype(np.float32), "occupancy_path"),
                             ("grid_x_edges_mm.npy", x, "x_edges_mm_path"), ("grid_y_edges_mm.npy", y, "y_edges_mm_path")]:
        np.save(out / name, array)
        meta[key] = str(final / name)
    meta["occupancy_sum"] = float(weighted.sum())
    _atomic_write_json(out / CACHE_METADATA["grid_occupancy_histograms"], meta)
    return {"action": "weighted_merge", "blocks": [s["block"] for s, _, _ in segments]}


def rebuild_geometry(ant, group, root, final_root):
    """Use existing production calculators for the common annotated geometry."""
    repo = Path(__file__).resolve().parents[2]
    track = root / "per_track" / ant["name"]
    geometry = group["geometry"]
    outputs = {}
    commands = {
        "colony_presence_vectors": ("compute_track_colony_presence_vector.py",
                                    ["--colony_boxes_mm=" + ";".join(",".join(map(str, box)) for box in geometry["colony_boxes_mm"])]),
        "grid_occupancy_histograms": ("compute_track_grid_occupancy.py",
                                      ["--bounds_json", str(root / "grid_bounds_from_panorama_regions.json"),
                                       "--side", ant["side"], "--grid_size_mm", str(geometry["grid_size_mm"]),
                                       "--grid_pad_mm", str(geometry["grid_pad_mm"])])}
    for kind, (script, extra) in commands.items():
        out = root / kind / "per_track" / Path(ant["name"]).stem
        subprocess.run([sys.executable, str(repo / "analysis" / script), "--track", str(track), "--out", str(out),
                        "--mm_per_px", str(geometry["mm_per_px"]), *extra], check=True)
        meta_path = out / CACHE_METADATA[kind]
        meta = json.loads(meta_path.read_text())
        # The generators intentionally use their usual per-track indexing. Paths
        # must address the published tree, not scratch, after the transfer.
        def rewrite(value):
            if isinstance(value, dict): return {key: rewrite(val) for key, val in value.items()}
            if isinstance(value, list): return [rewrite(val) for val in value]
            if isinstance(value, str) and value.startswith(str(root) + "/"):
                return str(final_root) + value[len(str(root)):]
            return value
        meta = rewrite(meta)
        meta.update(start_datetime=group["start_datetime"], fps=group["fps"],
                    combination_method="recomputed_from_combined_track_and_panorama_regions",
                    regions_sources=geometry["sources"])
        _atomic_write_json(meta_path, meta)
        outputs[kind] = {"action": "recomputed_with_shared_annotations"}
    return outputs


def combine_caches(ant, group, root, final_root):
    report = {}
    for kind in ("speed_vectors", "sleep_motion", "sleep_motion_labels", "sleep_predictions"):
        segments = cache_segments(ant, kind)
        if not segments:
            report[kind] = {"action": "unavailable", "blocks": []}
            continue
        report[kind] = (merge_motion(segments, ant, group, root, final_root) if kind == "sleep_motion" else
                        merge_dense(segments, kind, ant, group, root, final_root))
        report[kind]["missing_cache_blocks"] = [s["block"] for s in ant["sources"] if kind not in s["caches"]]
    if group.get("geometry"):
        report.update(rebuild_geometry(ant, group, root, final_root))
    else:
        for kind in ("colony_presence_vectors", "grid_occupancy_histograms"):
            segments = cache_segments(ant, kind)
            report[kind] = ({"action": "unavailable", "blocks": []} if not segments else
                            merge_dense(segments, kind, ant, group, root, final_root) if kind in DENSE else
                            merge_grid(segments, ant, group, root, final_root))
    return report
