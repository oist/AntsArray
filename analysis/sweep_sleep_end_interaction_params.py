#!/usr/bin/env python3
"""Sweep interaction-detection parameters against labeled sleep-end transitions.

The sweep uses frame-level sleep/wake labels from crop videos, maps each crop
frame back to the original chunk track parquet, recomputes focal-ant contact
distances around labeled sleep->wake transitions, and scores parameter sets by
how enriched interaction-bout onsets are near sleep ends versus matched sleep
control times.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from analysis.export_sleep_crop_videos import ChunkTrackIndex, DEFAULT_CHUNK_FRAMES
from analysis.sleep_classifier import expand_label_paths, load_label_files, parse_crop_name


DEFAULT_CROP_ROOT = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/sleep_crop_videos/"
    "random_100_600s_min75pct_480px_20260622_161247"
)
DEFAULT_BLOCK_ROOT = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02")
DEFAULT_ANTENNA_SETS: dict[str, tuple[int, ...]] = {
    "tips": (6, 9),
    "distal": (5, 6, 8, 9),
    "default": (4, 5, 6, 7, 8, 9),
    "all_points": tuple(range(10)),
}


@dataclass(frozen=True)
class WindowSpec:
    side: str
    chunk: int
    chunk_start: int
    local_start: int
    local_stop: int


def parse_float_csv(raw: str) -> list[float]:
    return [float(part.strip()) for part in str(raw).split(",") if part.strip()]


def parse_antenna_sets(raw: str) -> dict[str, tuple[int, ...]]:
    if not str(raw).strip():
        return dict(DEFAULT_ANTENNA_SETS)
    out: dict[str, tuple[int, ...]] = {}
    for spec in str(raw).split(";"):
        spec = spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            if spec not in DEFAULT_ANTENNA_SETS:
                raise ValueError(f"Unknown antenna set {spec!r}; known: {sorted(DEFAULT_ANTENNA_SETS)}")
            out[spec] = DEFAULT_ANTENNA_SETS[spec]
            continue
        name, values = spec.split(":", 1)
        points = tuple(int(part.strip()) for part in values.split(",") if part.strip())
        if not points:
            raise ValueError(f"Antenna set {spec!r} has no bodypoints")
        out[name.strip()] = points
    return out


def parse_roles(raw: str) -> list[str]:
    roles = [part.strip().lower() for part in str(raw).split(",") if part.strip()]
    bad = sorted(set(roles).difference({"body", "antenna", "either"}))
    if bad:
        raise ValueError(f"Unknown role(s): {bad}; use body, antenna, either")
    return roles


def crop_window_table(label_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in label_paths:
        raw = pd.read_parquet(path, columns=["Frame", "video_path"])
        if raw.empty:
            continue
        video_path = Path(str(raw["video_path"].dropna().iloc[0])) if raw["video_path"].notna().any() else path
        crop_info = parse_crop_name(video_path)
        if crop_info is None:
            crop_info = parse_crop_name(path)
        if crop_info is None:
            raise ValueError(f"Could not parse crop frame/track from {path}")
        local_min = int(pd.to_numeric(raw["Frame"], errors="coerce").min())
        local_max = int(pd.to_numeric(raw["Frame"], errors="coerce").max())
        crop_start = int(crop_info["crop_frame_start"])
        rows.append(
            {
                "label_file": str(path),
                "video_path": str(video_path),
                "side": str(crop_info["side"]),
                "track_id": int(crop_info["track_id"]),
                "crop_frame_start": crop_start,
                "crop_frame_stop": crop_start + local_max + 1,
                "n_crop_frames": local_max - local_min + 1,
            }
        )
    if not rows:
        raise ValueError("No crop label windows found")
    return pd.DataFrame(rows).drop_duplicates("label_file").reset_index(drop=True)


def labeled_sleep_end_table(
    labels: pd.DataFrame,
    crop_windows: pd.DataFrame,
    *,
    fps: float,
    min_sleep_seconds: float,
    max_transition_gap_seconds: float,
) -> pd.DataFrame:
    min_sleep_frames = int(round(float(min_sleep_seconds) * float(fps)))
    max_gap_frames = int(round(float(max_transition_gap_seconds) * float(fps)))
    rows = []
    for label_file, group in labels.sort_values(["label_file", "frame_start"]).groupby("label_file", sort=True):
        group = group.reset_index(drop=True)
        for idx, row in group.iterrows():
            if str(row["label"]).lower() != "sleep":
                continue
            duration_frames = int(row["frame_end"]) - int(row["frame_start"]) + 1
            if duration_frames < min_sleep_frames:
                continue
            if idx + 1 >= len(group):
                continue
            next_row = group.iloc[idx + 1]
            if str(next_row["label"]).lower() != "wake":
                continue
            transition_gap = int(next_row["frame_start"]) - int(row["frame_end"]) - 1
            if transition_gap > max_gap_frames:
                continue
            rows.append(
                {
                    "transition_id": len(rows),
                    "label_file": str(label_file),
                    "video_path": str(row.get("video_path", "")),
                    "side": str(row["side"]),
                    "track_id": int(row["track_id"]),
                    "sleep_start_frame": int(row["frame_start"]),
                    "sleep_end_frame": int(row["frame_end"]),
                    "wake_start_frame": int(next_row["frame_start"]),
                    "sleep_end_center_frame": int(next_row["frame_start"]),
                    "sleep_duration_seconds": float(duration_frames / float(fps)),
                    "transition_gap_frames": int(max(0, transition_gap)),
                }
            )
    sleep_ends = pd.DataFrame(rows)
    if sleep_ends.empty:
        raise ValueError("No sleep->wake transitions found after filtering")
    sleep_ends = sleep_ends.merge(
        crop_windows[["label_file", "crop_frame_start", "crop_frame_stop", "n_crop_frames"]],
        on="label_file",
        how="left",
        validate="many_to_one",
    )
    return sleep_ends.reset_index(drop=True)


def frame_mask_near_targets(frames: np.ndarray, targets: np.ndarray, exclusion_frames: int) -> np.ndarray:
    if len(targets) == 0:
        return np.zeros(len(frames), dtype=bool)
    targets = np.sort(np.asarray(targets, dtype=np.int64))
    idx = np.searchsorted(targets, frames)
    near = np.zeros(len(frames), dtype=bool)
    valid_left = idx > 0
    near[valid_left] |= np.abs(frames[valid_left] - targets[idx[valid_left] - 1]) <= int(exclusion_frames)
    valid_right = idx < len(targets)
    near[valid_right] |= np.abs(frames[valid_right] - targets[idx[valid_right]]) <= int(exclusion_frames)
    return near


def control_frame_table(
    label_paths: list[Path],
    sleep_ends: pd.DataFrame,
    *,
    fps: float,
    controls_per_transition: int,
    exclude_seconds: float,
    pre_seconds: float,
    post_seconds: float,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_state))
    rows = []
    exclude_frames = int(round(float(exclude_seconds) * float(fps)))
    pre_frames = int(round(float(pre_seconds) * float(fps)))
    post_frames = int(round(float(post_seconds) * float(fps)))
    ends_by_file = {
        str(label_file): group["sleep_end_center_frame"].to_numpy(np.int64)
        for label_file, group in sleep_ends.groupby("label_file", sort=False)
    }
    transition_rows_by_file = {
        str(label_file): group.reset_index(drop=True)
        for label_file, group in sleep_ends.groupby("label_file", sort=False)
    }

    for path in label_paths:
        label_file = str(path)
        if label_file not in transition_rows_by_file:
            continue
        raw = pd.read_parquet(path, columns=["Frame", "label_value", "video_path"])
        if raw.empty:
            continue
        video_path = Path(str(raw["video_path"].dropna().iloc[0])) if raw["video_path"].notna().any() else path
        crop_info = parse_crop_name(video_path)
        if crop_info is None:
            crop_info = parse_crop_name(path)
        if crop_info is None:
            continue
        crop_start = int(crop_info["crop_frame_start"])
        local = pd.to_numeric(raw["Frame"], errors="coerce").to_numpy(np.float64)
        label_value = pd.to_numeric(raw["label_value"], errors="coerce").to_numpy(np.float64)
        valid = np.isfinite(local) & (label_value == 1)
        global_frames = crop_start + np.rint(local).astype(np.int64)
        global_frames = global_frames[valid]
        if len(global_frames) == 0:
            continue
        crop_stop = crop_start + int(np.nanmax(local)) + 1
        edge_ok = (global_frames >= crop_start + pre_frames) & (global_frames < crop_stop - post_frames)
        far_from_transition = ~frame_mask_near_targets(global_frames, ends_by_file.get(label_file, np.asarray([], dtype=np.int64)), exclude_frames)
        candidates = np.unique(global_frames[edge_ok & far_from_transition])
        if len(candidates) == 0:
            candidates = np.unique(global_frames[edge_ok])
        if len(candidates) == 0:
            continue

        for transition in transition_rows_by_file[label_file].itertuples(index=False):
            size = min(len(candidates), int(controls_per_transition))
            chosen = rng.choice(candidates, size=size, replace=False)
            for control_idx, frame in enumerate(np.sort(chosen)):
                rows.append(
                    {
                        "transition_id": int(transition.transition_id),
                        "control_id": int(control_idx),
                        "label_file": label_file,
                        "side": str(transition.side),
                        "track_id": int(transition.track_id),
                        "control_center_frame": int(frame),
                    }
                )
    return pd.DataFrame(rows)


def split_global_window(track_index: ChunkTrackIndex, side: str, global_start: int, global_stop: int) -> list[WindowSpec]:
    specs = []
    for spec in track_index.specs:
        overlap_start = max(int(global_start), int(spec.start))
        overlap_stop = min(int(global_stop), int(spec.stop))
        if overlap_stop <= overlap_start:
            continue
        if track_index.path_for(spec.chunk, side) is None:
            continue
        specs.append(
            WindowSpec(
                side=str(side),
                chunk=int(spec.chunk),
                chunk_start=int(spec.start),
                local_start=int(overlap_start - spec.start),
                local_stop=int(overlap_stop - spec.start),
            )
        )
    return specs


def merge_window_specs(windows: list[WindowSpec]) -> list[WindowSpec]:
    out: list[WindowSpec] = []
    for key, group_df in pd.DataFrame([w.__dict__ for w in windows]).groupby(["side", "chunk", "chunk_start"], sort=True):
        side, chunk, chunk_start = key
        intervals = group_df[["local_start", "local_stop"]].sort_values("local_start").to_numpy(np.int64)
        merged: list[list[int]] = []
        for start, stop in intervals:
            if not merged or int(start) > merged[-1][1]:
                merged.append([int(start), int(stop)])
            else:
                merged[-1][1] = max(merged[-1][1], int(stop))
        for start, stop in merged:
            out.append(
                WindowSpec(
                    side=str(side),
                    chunk=int(chunk),
                    chunk_start=int(chunk_start),
                    local_start=int(start),
                    local_stop=int(stop),
                )
            )
    return out


def required_windows(
    centers: pd.DataFrame,
    track_index: ChunkTrackIndex,
    *,
    fps: float,
    pre_seconds: float,
    post_seconds: float,
    event_gap_seconds_values: list[float],
) -> list[WindowSpec]:
    max_gap = max(event_gap_seconds_values) if event_gap_seconds_values else 0.0
    pad_frames = int(round((max(float(pre_seconds), float(post_seconds)) + max_gap + 2.0) * float(fps)))
    windows = []
    for center in centers.itertuples(index=False):
        frame = int(getattr(center, "center_frame"))
        windows.extend(
            split_global_window(
                track_index,
                str(getattr(center, "side")),
                frame - pad_frames,
                frame + pad_frames + 1,
            )
        )
    if not windows:
        return []
    return merge_window_specs(windows)


def load_track_window(path: Path, *, local_start: int, local_stop: int) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    columns = ["Frame", "TrackID", "Bodypoint", "X", "Y", "TrackX", "TrackY"]
    table = ds.dataset(path, format="parquet").to_table(
        columns=columns,
        filter=(pc.field("Frame") >= int(local_start)) & (pc.field("Frame") < int(local_stop)),
        use_threads=True,
    )
    df = table.to_pandas()
    if df.empty:
        return df
    for col in ["Frame", "TrackID", "Bodypoint"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["X", "Y", "TrackX", "TrackY"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=columns).copy()
    df[["Frame", "TrackID", "Bodypoint"]] = np.rint(df[["Frame", "TrackID", "Bodypoint"]]).astype(np.int64)
    return df.sort_values(["Frame", "TrackID", "Bodypoint"], kind="mergesort").reset_index(drop=True)


def track_arrays_for_frame(frame_df: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    tracks: dict[int, dict[str, np.ndarray]] = {}
    for track_id, group in frame_df.groupby("TrackID", sort=False):
        center = group[["TrackX", "TrackY"]].dropna()
        if center.empty:
            continue
        xy = group[["X", "Y"]].to_numpy(np.float64, copy=True)
        bodypoints = group["Bodypoint"].to_numpy(np.int64, copy=False)
        finite = np.isfinite(xy).all(axis=1)
        xy = xy[finite]
        bodypoints = bodypoints[finite]
        if xy.size == 0:
            continue
        tracks[int(track_id)] = {
            "center": center.iloc[0].to_numpy(np.float64),
            "xy": xy,
            "bodypoints": bodypoints,
        }
    return tracks


def min_distance_sq(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if points_a.size == 0 or points_b.size == 0:
        return np.inf
    delta = points_a[:, None, :] - points_b[None, :, :]
    dist_sq = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
    return float(np.min(dist_sq))


def antenna_xy(track: dict[str, np.ndarray], bodypoints: tuple[int, ...]) -> np.ndarray:
    mask = np.isin(track["bodypoints"], np.asarray(bodypoints, dtype=np.int64))
    return track["xy"][mask]


def compute_contact_distances(
    windows: list[WindowSpec],
    track_index: ChunkTrackIndex,
    focus_tracks: pd.DataFrame,
    antenna_sets: dict[str, tuple[int, ...]],
    *,
    mm_per_px: float,
    max_interaction_radius_mm: float,
    max_micro_distance_mm: float,
    cache_path: Path,
    force: bool = False,
) -> pd.DataFrame:
    if cache_path.exists() and not force:
        print(f"cache hit: {cache_path}")
        return pd.read_parquet(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    focus_by_side = {
        str(side): set(group["track_id"].astype(int).to_list())
        for side, group in focus_tracks.groupby("side", sort=False)
    }
    max_radius_px = float(max_interaction_radius_mm) / float(mm_per_px)
    max_radius_sq = max_radius_px * max_radius_px
    max_micro_px = float(max_micro_distance_mm) / float(mm_per_px)
    max_micro_sq = max_micro_px * max_micro_px
    rows = []
    start_time = time.perf_counter()
    total_frames = 0

    for win_idx, window in enumerate(windows, start=1):
        path = track_index.path_for(window.chunk, window.side)
        if path is None:
            continue
        print(
            f"contact distances {win_idx}/{len(windows)}: {path.name} "
            f"frames {window.local_start}-{window.local_stop - 1}",
            flush=True,
        )
        chunk = load_track_window(path, local_start=window.local_start, local_stop=window.local_stop)
        if chunk.empty:
            continue
        focus_ids = focus_by_side.get(window.side, set())
        if not focus_ids:
            continue
        for frame, frame_df in chunk.groupby("Frame", sort=True):
            tracks = track_arrays_for_frame(frame_df)
            present_focus = [track_id for track_id in focus_ids if track_id in tracks]
            if not present_focus or len(tracks) < 2:
                continue
            global_frame = int(frame) + int(window.chunk_start)
            track_ids = np.asarray(sorted(tracks), dtype=np.int64)
            for focus_id in present_focus:
                focus = tracks[int(focus_id)]
                other_ids = track_ids[track_ids != int(focus_id)]
                for other_id in other_ids:
                    other = tracks[int(other_id)]
                    center_delta = focus["center"] - other["center"]
                    center_dist_sq = float(np.dot(center_delta, center_delta))
                    if center_dist_sq > max_radius_sq:
                        continue
                    center_dist_mm = float(np.sqrt(center_dist_sq) * float(mm_per_px))
                    for set_name, bps in antenna_sets.items():
                        other_antenna = antenna_xy(other, bps)
                        focus_antenna = antenna_xy(focus, bps)
                        body_dist_sq = min_distance_sq(other_antenna, focus["xy"])
                        if body_dist_sq <= max_micro_sq:
                            rows.append(
                                {
                                    "global_frame": global_frame,
                                    "side": window.side,
                                    "track_id": int(focus_id),
                                    "partner_track_id": int(other_id),
                                    "interaction_role": "body",
                                    "antenna_set": str(set_name),
                                    "center_distance_mm": center_dist_mm,
                                    "min_distance_mm": float(np.sqrt(body_dist_sq) * float(mm_per_px)),
                                }
                            )
                        antenna_dist_sq = min_distance_sq(focus_antenna, other["xy"])
                        if antenna_dist_sq <= max_micro_sq:
                            rows.append(
                                {
                                    "global_frame": global_frame,
                                    "side": window.side,
                                    "track_id": int(focus_id),
                                    "partner_track_id": int(other_id),
                                    "interaction_role": "antenna",
                                    "antenna_set": str(set_name),
                                    "center_distance_mm": center_dist_mm,
                                    "min_distance_mm": float(np.sqrt(antenna_dist_sq) * float(mm_per_px)),
                                }
                            )
            total_frames += 1
        elapsed = time.perf_counter() - start_time
        print(
            f"  accumulated rows={len(rows):,}, frames={total_frames:,}, "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    contacts = pd.DataFrame(
        rows,
        columns=[
            "global_frame",
            "side",
            "track_id",
            "partner_track_id",
            "interaction_role",
            "antenna_set",
            "center_distance_mm",
            "min_distance_mm",
        ],
    )
    contacts.to_parquet(cache_path, index=False)
    print(f"wrote contact cache: {cache_path} ({len(contacts):,} rows)")
    return contacts


def contact_onsets(
    contacts: pd.DataFrame,
    *,
    antenna_set: str,
    interaction_radius_mm: float,
    micro_distance_mm: float,
    role: str,
    event_gap_seconds: float,
    fps: float,
) -> pd.DataFrame:
    if contacts.empty:
        return pd.DataFrame(columns=["side", "track_id", "partner_track_id", "global_frame"])
    work = contacts[
        (contacts["antenna_set"].astype(str) == str(antenna_set))
        & (pd.to_numeric(contacts["center_distance_mm"], errors="coerce") <= float(interaction_radius_mm))
        & (pd.to_numeric(contacts["min_distance_mm"], errors="coerce") <= float(micro_distance_mm))
    ].copy()
    if role != "either":
        work = work[work["interaction_role"].astype(str) == str(role)].copy()
    if work.empty:
        return pd.DataFrame(columns=["side", "track_id", "partner_track_id", "global_frame"])
    work["global_frame"] = pd.to_numeric(work["global_frame"], errors="coerce").round().astype(np.int64)
    work["track_id"] = pd.to_numeric(work["track_id"], errors="coerce").astype(np.int64)
    work["partner_track_id"] = pd.to_numeric(work["partner_track_id"], errors="coerce").astype(np.int64)
    gap_frames = max(1, int(round(float(event_gap_seconds) * float(fps))))
    keys = ["side", "track_id", "partner_track_id"] if role == "either" else ["side", "track_id", "partner_track_id", "interaction_role"]
    work = work.drop_duplicates(keys + ["global_frame"]).sort_values(keys + ["global_frame"], kind="mergesort")
    frame_gap = work.groupby(keys, sort=False)["global_frame"].diff()
    onsets = work[frame_gap.isna() | (frame_gap > gap_frames)].copy()
    return onsets[["side", "track_id", "partner_track_id", "global_frame"]].drop_duplicates().reset_index(drop=True)


def nearest_event_metrics(
    onsets: pd.DataFrame,
    centers: pd.DataFrame,
    *,
    center_frame_col: str,
    pre_seconds: float,
    post_seconds: float,
    fps: float,
) -> pd.DataFrame:
    pre_frames = int(round(float(pre_seconds) * float(fps)))
    post_frames = int(round(float(post_seconds) * float(fps)))
    event_map = {
        (str(side), int(track_id)): np.sort(group["global_frame"].to_numpy(np.int64))
        for (side, track_id), group in onsets.groupby(["side", "track_id"], sort=False)
    }
    rows = []
    for center in centers.itertuples(index=False):
        frame = int(getattr(center, center_frame_col))
        key = (str(getattr(center, "side")), int(getattr(center, "track_id")))
        events = event_map.get(key, np.asarray([], dtype=np.int64))
        left = np.searchsorted(events, frame - pre_frames, side="left")
        right = np.searchsorted(events, frame + post_frames, side="right")
        nearby = events[left:right]
        if len(nearby):
            rel = (nearby - frame) / float(fps)
            nearest_idx = int(np.argmin(np.abs(rel)))
            nearest_latency = float(rel[nearest_idx])
            hit = True
            pre_hit = bool(np.any(rel < 0))
            post_hit = bool(np.any(rel >= 0))
            n_events = int(len(rel))
        else:
            nearest_latency = np.nan
            hit = False
            pre_hit = False
            post_hit = False
            n_events = 0
        row = {
            "hit": hit,
            "pre_hit": pre_hit,
            "post_hit": post_hit,
            "nearest_latency_s": nearest_latency,
            "n_events_in_window": n_events,
        }
        for col in ["transition_id", "control_id", "label_file", "side", "track_id"]:
            if hasattr(center, col):
                row[col] = getattr(center, col)
        rows.append(row)
    return pd.DataFrame(rows)


def score_parameter_grid(
    contacts: pd.DataFrame,
    sleep_ends: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    radii_mm: list[float],
    micro_distances_mm: list[float],
    antenna_set_names: list[str],
    roles: list[str],
    event_gap_seconds_values: list[float],
    pre_seconds: float,
    post_seconds: float,
    fps: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_rows = []
    detail_tables: dict[str, pd.DataFrame] = {}
    sleep_centers = sleep_ends.rename(columns={"sleep_end_center_frame": "center_frame"}).copy()
    control_centers = controls.rename(columns={"control_center_frame": "center_frame"}).copy()
    total_combos = len(radii_mm) * len(micro_distances_mm) * len(antenna_set_names) * len(roles) * len(event_gap_seconds_values)
    combo_idx = 0
    for antenna_set in antenna_set_names:
        for radius in radii_mm:
            for micro in micro_distances_mm:
                for event_gap in event_gap_seconds_values:
                    for role in roles:
                        combo_idx += 1
                        if combo_idx == 1 or combo_idx == total_combos or combo_idx % 100 == 0:
                            print(f"scoring {combo_idx}/{total_combos}", flush=True)
                        onsets = contact_onsets(
                            contacts,
                            antenna_set=antenna_set,
                            interaction_radius_mm=radius,
                            micro_distance_mm=micro,
                            role=role,
                            event_gap_seconds=event_gap,
                            fps=fps,
                        )
                        sleep_metrics = nearest_event_metrics(
                            onsets,
                            sleep_centers,
                            center_frame_col="center_frame",
                            pre_seconds=pre_seconds,
                            post_seconds=post_seconds,
                            fps=fps,
                        )
                        control_metrics = nearest_event_metrics(
                            onsets,
                            control_centers,
                            center_frame_col="center_frame",
                            pre_seconds=pre_seconds,
                            post_seconds=post_seconds,
                            fps=fps,
                        )
                        hit_rate = float(sleep_metrics["hit"].mean()) if len(sleep_metrics) else np.nan
                        control_hit_rate = float(control_metrics["hit"].mean()) if len(control_metrics) else np.nan
                        post_hit_rate = float(sleep_metrics["post_hit"].mean()) if len(sleep_metrics) else np.nan
                        pre_hit_rate = float(sleep_metrics["pre_hit"].mean()) if len(sleep_metrics) else np.nan
                        event_count_rate = float(sleep_metrics["n_events_in_window"].mean()) if len(sleep_metrics) else np.nan
                        latency = sleep_metrics.loc[sleep_metrics["hit"], "nearest_latency_s"].to_numpy(np.float64)
                        summary_rows.append(
                            {
                                "antenna_set": antenna_set,
                                "interaction_radius_mm": float(radius),
                                "micro_interaction_distance_mm": float(micro),
                                "event_gap_seconds": float(event_gap),
                                "role": role,
                                "n_sleep_ends": int(len(sleep_metrics)),
                                "n_controls": int(len(control_metrics)),
                                "n_event_onsets": int(len(onsets)),
                                "sleep_end_hit_rate": hit_rate,
                                "control_hit_rate": control_hit_rate,
                                "enrichment_diff": hit_rate - control_hit_rate,
                                "enrichment_ratio": hit_rate / max(control_hit_rate, 1e-9),
                                "sleep_end_pre_hit_rate": pre_hit_rate,
                                "sleep_end_post_hit_rate": post_hit_rate,
                                "mean_events_per_sleep_end_window": event_count_rate,
                                "median_nearest_latency_s": float(np.nanmedian(latency)) if len(latency) else np.nan,
                                "median_abs_nearest_latency_s": float(np.nanmedian(np.abs(latency))) if len(latency) else np.nan,
                            }
                        )
    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "enrichment_diff",
            "sleep_end_hit_rate",
            "median_abs_nearest_latency_s",
            "n_event_onsets",
        ],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return summary.reset_index(drop=True), detail_tables


def relative_event_curve(
    onsets: pd.DataFrame,
    centers: pd.DataFrame,
    *,
    center_frame_col: str,
    condition: str,
    pre_seconds: float,
    post_seconds: float,
    bin_seconds: float,
    fps: float,
) -> pd.DataFrame:
    bins = np.arange(-float(pre_seconds), float(post_seconds) + float(bin_seconds), float(bin_seconds), dtype=np.float64)
    if len(bins) < 2:
        raise ValueError("curve binning produced fewer than 2 edges")
    event_map = {
        (str(side), int(track_id)): np.sort(group["global_frame"].to_numpy(np.int64))
        for (side, track_id), group in onsets.groupby(["side", "track_id"], sort=False)
    }
    counts = np.zeros(len(bins) - 1, dtype=np.float64)
    hit = np.zeros(len(bins) - 1, dtype=np.float64)
    n_centers = int(len(centers))
    for center in centers.itertuples(index=False):
        frame = int(getattr(center, center_frame_col))
        key = (str(getattr(center, "side")), int(getattr(center, "track_id")))
        events = event_map.get(key, np.asarray([], dtype=np.int64))
        left = np.searchsorted(events, frame - int(round(pre_seconds * fps)), side="left")
        right = np.searchsorted(events, frame + int(round(post_seconds * fps)), side="right")
        rel = (events[left:right] - frame) / float(fps)
        if len(rel):
            hist, _ = np.histogram(rel, bins=bins)
            counts += hist
            hit += hist > 0
    centers_s = (bins[:-1] + bins[1:]) * 0.5
    return pd.DataFrame(
        {
            "condition": condition,
            "relative_seconds": centers_s,
            "event_onsets_per_transition_per_s": counts / max(n_centers, 1) / float(bin_seconds),
            "transition_hit_fraction": hit / max(n_centers, 1),
            "n_centers": n_centers,
            "bin_seconds": float(bin_seconds),
        }
    )


def onsets_for_summary_row(contacts: pd.DataFrame, row: pd.Series, *, fps: float) -> pd.DataFrame:
    return contact_onsets(
        contacts,
        antenna_set=str(row["antenna_set"]),
        interaction_radius_mm=float(row["interaction_radius_mm"]),
        micro_distance_mm=float(row["micro_interaction_distance_mm"]),
        role=str(row["role"]),
        event_gap_seconds=float(row["event_gap_seconds"]),
        fps=fps,
    )


def plot_summary(
    summary: pd.DataFrame,
    contacts: pd.DataFrame,
    sleep_ends: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    out_dir: Path,
    fps: float,
    curve_pre_seconds: float,
    curve_post_seconds: float,
    curve_bin_seconds: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    top = summary.head(25).copy()
    top["label"] = (
        top["antenna_set"].astype(str)
        + " r"
        + top["interaction_radius_mm"].map(lambda x: f"{x:g}")
        + " m"
        + top["micro_interaction_distance_mm"].map(lambda x: f"{x:g}")
        + " gap"
        + top["event_gap_seconds"].map(lambda x: f"{x:g}")
        + " "
        + top["role"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(11, max(5, 0.32 * len(top))), constrained_layout=True)
    y = np.arange(len(top))[::-1]
    ax.barh(y, top["sleep_end_hit_rate"], color="tab:blue", alpha=0.75, label="sleep-end hit")
    ax.barh(y, top["control_hit_rate"], color="0.45", alpha=0.45, label="control hit")
    ax.set_yticks(y, labels=top["label"])
    ax.set_xlabel("fraction with interaction onset in scoring window")
    ax.set_title("Top interaction-parameter sets by sleep-end enrichment")
    ax.legend(loc="lower right")
    fig.savefig(out_dir / "top_parameter_sets_hit_rates.png", dpi=180)
    plt.close(fig)

    best_subset = summary.iloc[0]
    heat_data = summary[
        (summary["antenna_set"] == best_subset["antenna_set"])
        & (summary["event_gap_seconds"] == best_subset["event_gap_seconds"])
        & (summary["role"] == best_subset["role"])
    ].copy()
    radii = sorted(heat_data["interaction_radius_mm"].unique())
    micros = sorted(heat_data["micro_interaction_distance_mm"].unique())
    pivot = heat_data.pivot_table(
        index="micro_interaction_distance_mm",
        columns="interaction_radius_mm",
        values="enrichment_diff",
        aggfunc="max",
    ).reindex(index=micros, columns=radii)
    fig, ax = plt.subplots(figsize=(8, 5.8), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(np.float64), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(radii)), labels=[f"{x:g}" for x in radii])
    ax.set_yticks(np.arange(len(micros)), labels=[f"{x:g}" for x in micros])
    ax.set_xlabel("interaction_radius_mm")
    ax.set_ylabel("micro_interaction_distance_mm")
    ax.set_title(
        "Sleep-end enrichment heatmap\n"
        f"antenna_set={best_subset['antenna_set']}, role={best_subset['role']}, gap={best_subset['event_gap_seconds']:g}s"
    )
    fig.colorbar(im, ax=ax, label="sleep hit rate - control hit rate")
    fig.savefig(out_dir / "enrichment_heatmap_best_slice.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    scatter = ax.scatter(
        summary["control_hit_rate"],
        summary["sleep_end_hit_rate"],
        c=summary["median_abs_nearest_latency_s"],
        s=np.clip(summary["n_event_onsets"] / max(summary["n_event_onsets"].max(), 1) * 90, 12, 90),
        cmap="magma_r",
        alpha=0.72,
        edgecolors="none",
    )
    lim = max(float(summary["control_hit_rate"].max()), float(summary["sleep_end_hit_rate"].max()), 0.05)
    ax.plot([0, lim], [0, lim], color="0.5", lw=1, ls="--")
    ax.set_xlabel("control hit rate")
    ax.set_ylabel("sleep-end hit rate")
    ax.set_title("Parameter sweep: sleep-end specificity")
    fig.colorbar(scatter, ax=ax, label="median abs latency (s)")
    fig.savefig(out_dir / "sleep_end_vs_control_hit_scatter.png", dpi=180)
    plt.close(fig)

    sleep_centers = sleep_ends.rename(columns={"sleep_end_center_frame": "center_frame"}).copy()
    control_centers = controls.rename(columns={"control_center_frame": "center_frame"}).copy()
    chosen_rows = [("best", summary.iloc[0])]
    default = summary[
        (summary["antenna_set"] == "default")
        & np.isclose(summary["interaction_radius_mm"], 8.0)
        & np.isclose(summary["micro_interaction_distance_mm"], 1.0)
        & np.isclose(summary["event_gap_seconds"], 2.0)
        & (summary["role"] == "body")
    ]
    if not default.empty:
        chosen_rows.append(("current_default_body", default.iloc[0]))

    curves = []
    latency_tables = []
    for label, row in chosen_rows:
        onsets = onsets_for_summary_row(contacts, row, fps=fps)
        sleep_curve = relative_event_curve(
            onsets,
            sleep_centers,
            center_frame_col="center_frame",
            condition=f"{label}: sleep_end",
            pre_seconds=curve_pre_seconds,
            post_seconds=curve_post_seconds,
            bin_seconds=curve_bin_seconds,
            fps=fps,
        )
        control_curve = relative_event_curve(
            onsets,
            control_centers,
            center_frame_col="center_frame",
            condition=f"{label}: control",
            pre_seconds=curve_pre_seconds,
            post_seconds=curve_post_seconds,
            bin_seconds=curve_bin_seconds,
            fps=fps,
        )
        curves.extend([sleep_curve, control_curve])
        metrics = nearest_event_metrics(
            onsets,
            sleep_centers,
            center_frame_col="center_frame",
            pre_seconds=curve_pre_seconds,
            post_seconds=curve_post_seconds,
            fps=fps,
        )
        metrics["parameter_label"] = label
        latency_tables.append(metrics)

    curve_table = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    if not curve_table.empty:
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        for condition, group in curve_table.groupby("condition", sort=False):
            ax.plot(
                group["relative_seconds"],
                group["event_onsets_per_transition_per_s"],
                lw=2,
                label=condition,
            )
        ax.axvline(0, color="0.25", lw=1, ls="--")
        ax.set_xlabel("seconds relative to labeled sleep end")
        ax.set_ylabel("interaction-bout onsets / transition / s")
        ax.set_title("Interaction timing around labeled sleep ends")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.savefig(out_dir / "event_triggered_interaction_rate.png", dpi=180)
        plt.close(fig)
        curve_table.to_csv(out_dir / "event_triggered_interaction_rate.csv", index=False)

    if latency_tables:
        latencies = pd.concat(latency_tables, ignore_index=True)
        latencies = latencies[latencies["hit"] & latencies["nearest_latency_s"].notna()].copy()
        if not latencies.empty:
            fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
            bins = np.arange(-float(curve_pre_seconds), float(curve_post_seconds) + 1.0, 1.0)
            for label, group in latencies.groupby("parameter_label", sort=False):
                ax.hist(
                    group["nearest_latency_s"],
                    bins=bins,
                    alpha=0.5,
                    density=True,
                    label=label,
                )
            ax.axvline(0, color="0.25", lw=1, ls="--")
            ax.set_xlabel("nearest interaction onset latency from sleep end (s)")
            ax.set_ylabel("density")
            ax.set_title("Latency of nearest interaction onset")
            ax.legend(fontsize=8)
            fig.savefig(out_dir / "nearest_latency_histogram.png", dpi=180)
            plt.close(fig)
            latencies.to_csv(out_dir / "nearest_latency_details.csv", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop_root", type=Path, default=DEFAULT_CROP_ROOT)
    parser.add_argument("--block_root", type=Path, default=DEFAULT_BLOCK_ROOT)
    parser.add_argument("--label_glob", default="*_labels.parquet")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--mm_per_px", type=float, default=0.016)
    parser.add_argument("--min_sleep_seconds", type=float, default=5.0)
    parser.add_argument("--max_transition_gap_seconds", type=float, default=5.0)
    parser.add_argument("--score_pre_seconds", type=float, default=10.0)
    parser.add_argument("--score_post_seconds", type=float, default=20.0)
    parser.add_argument("--control_exclude_seconds", type=float, default=30.0)
    parser.add_argument("--controls_per_transition", type=int, default=5)
    parser.add_argument("--curve_pre_seconds", type=float, default=30.0)
    parser.add_argument("--curve_post_seconds", type=float, default=30.0)
    parser.add_argument("--curve_bin_seconds", type=float, default=1.0)
    parser.add_argument("--interaction_radii_mm", default="4,6,8,10,12")
    parser.add_argument("--micro_distances_mm", default="0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--event_gap_seconds", default="0,0.5,1,2,5")
    parser.add_argument(
        "--antenna_sets",
        default="tips;distal;default;all_points",
        help="Semicolon-separated names from defaults or name:bp,bp,... specs.",
    )
    parser.add_argument("--roles", default="body,antenna,either")
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--force_contact_cache", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    crop_root = Path(args.crop_root)
    label_dir = crop_root / "label_vectors"
    out_dir = Path(args.out_dir) if args.out_dir is not None else crop_root / "interaction_parameter_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    radii = parse_float_csv(args.interaction_radii_mm)
    micros = parse_float_csv(args.micro_distances_mm)
    event_gaps = parse_float_csv(args.event_gap_seconds)
    antenna_sets = parse_antenna_sets(args.antenna_sets)
    roles = parse_roles(args.roles)
    label_paths = expand_label_paths(None, label_dir, args.label_glob)
    labels = load_label_files(None, labels_dir=label_dir, label_glob=args.label_glob)
    crop_windows = crop_window_table(label_paths)
    sleep_ends = labeled_sleep_end_table(
        labels,
        crop_windows,
        fps=float(args.fps),
        min_sleep_seconds=float(args.min_sleep_seconds),
        max_transition_gap_seconds=float(args.max_transition_gap_seconds),
    )
    controls = control_frame_table(
        label_paths,
        sleep_ends,
        fps=float(args.fps),
        controls_per_transition=int(args.controls_per_transition),
        exclude_seconds=float(args.control_exclude_seconds),
        pre_seconds=float(args.score_pre_seconds),
        post_seconds=float(args.score_post_seconds),
        random_state=int(args.random_state),
    )
    if controls.empty:
        raise ValueError("No matched sleep-control frames could be sampled")

    sleep_ends.to_csv(out_dir / "labeled_sleep_end_transitions.csv", index=False)
    controls.to_csv(out_dir / "matched_sleep_control_frames.csv", index=False)
    crop_windows.to_csv(out_dir / "labeled_crop_windows.csv", index=False)
    print(
        f"loaded {len(label_paths)} label files; "
        f"{len(sleep_ends)} sleep->wake transitions; {len(controls)} controls",
        flush=True,
    )

    centers = pd.concat(
        [
            sleep_ends.rename(columns={"sleep_end_center_frame": "center_frame"})[
                ["side", "track_id", "center_frame"]
            ],
            controls.rename(columns={"control_center_frame": "center_frame"})[
                ["side", "track_id", "center_frame"]
            ],
        ],
        ignore_index=True,
    ).drop_duplicates()

    track_index = ChunkTrackIndex(args.block_root / "tracks", chunk_frames=DEFAULT_CHUNK_FRAMES, chunk_offset_mode="metadata")
    windows = required_windows(
        centers,
        track_index,
        fps=float(args.fps),
        pre_seconds=float(args.curve_pre_seconds),
        post_seconds=float(args.curve_post_seconds),
        event_gap_seconds_values=event_gaps,
    )
    pd.DataFrame([window.__dict__ for window in windows]).to_csv(out_dir / "contact_detection_windows.csv", index=False)
    focus_tracks = centers[["side", "track_id"]].drop_duplicates().reset_index(drop=True)
    print(f"detecting contacts for {len(focus_tracks)} focal tracks across {len(windows)} chunk windows", flush=True)
    contacts = compute_contact_distances(
        windows,
        track_index,
        focus_tracks,
        antenna_sets,
        mm_per_px=float(args.mm_per_px),
        max_interaction_radius_mm=max(radii),
        max_micro_distance_mm=max(micros),
        cache_path=out_dir / "focal_contact_distance_cache.parquet",
        force=bool(args.force_contact_cache),
    )

    summary, _details = score_parameter_grid(
        contacts,
        sleep_ends,
        controls,
        radii_mm=radii,
        micro_distances_mm=micros,
        antenna_set_names=list(antenna_sets),
        roles=roles,
        event_gap_seconds_values=event_gaps,
        pre_seconds=float(args.score_pre_seconds),
        post_seconds=float(args.score_post_seconds),
        fps=float(args.fps),
    )
    summary.to_csv(out_dir / "parameter_sweep_summary.csv", index=False)
    plot_summary(
        summary,
        contacts,
        sleep_ends,
        controls,
        out_dir=out_dir,
        fps=float(args.fps),
        curve_pre_seconds=float(args.curve_pre_seconds),
        curve_post_seconds=float(args.curve_post_seconds),
        curve_bin_seconds=float(args.curve_bin_seconds),
    )
    metadata = {
        "crop_root": str(crop_root),
        "block_root": str(args.block_root),
        "label_files": len(label_paths),
        "n_sleep_ends": int(len(sleep_ends)),
        "n_controls": int(len(controls)),
        "n_contact_rows": int(len(contacts)),
        "radii_mm": radii,
        "micro_distances_mm": micros,
        "event_gap_seconds": event_gaps,
        "antenna_sets": {name: list(points) for name, points in antenna_sets.items()},
        "roles": roles,
        "score_window_seconds": [-float(args.score_pre_seconds), float(args.score_post_seconds)],
        "best_parameter_set": summary.iloc[0].to_dict() if not summary.empty else {},
    }
    (out_dir / "sweep_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print("best parameter set:")
    print(summary.head(10).to_string(index=False))
    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
