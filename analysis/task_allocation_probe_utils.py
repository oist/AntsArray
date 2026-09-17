"""Utilities for exploratory task-allocation transition analyses."""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis.interaction_analysis_utils as ia


SIDES = ("left", "right")


def _save_current_figures_if_enabled() -> None:
    try:
        from analysis.figure_saving import save_new_figures

        save_new_figures(plt)
    except Exception as exc:
        print(f"Warning: failed to save task-allocation figure: {exc}")


def resolve_cluster_table_path(
    dataset_root: Path,
    *,
    preferred_grid_dirs: tuple[str, ...] = (
        "grid_occupancy_histograms_inferred_bounds",
        "grid_occupancy_histograms_0p5mm_inferred_bounds",
        "grid_occupancy_histograms",
    ),
) -> Path:
    stitched = Path(dataset_root) / "stitched"
    for dirname in preferred_grid_dirs:
        candidate = stitched / dirname / "track_cluster_ids.csv"
        if candidate.exists():
            return candidate
    searched = [str(stitched / dirname / "track_cluster_ids.csv") for dirname in preferred_grid_dirs]
    raise FileNotFoundError(f"No track_cluster_ids.csv found. Searched: {searched}")


def load_task_proxy_clusters(
    cluster_table_path: Path,
    *,
    sides: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    clusters = pd.read_csv(cluster_table_path)
    required = {"TrackID", "side", "cluster_id", "leiden_cluster_id"}
    missing = required.difference(clusters.columns)
    if missing:
        raise ValueError(f"{cluster_table_path} is missing required columns: {sorted(missing)}")

    out = clusters.copy()
    out["TrackID"] = pd.to_numeric(out["TrackID"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["TrackID", "side"]).astype({"TrackID": "int64"})
    out["side"] = out["side"].astype(str)
    out["cluster_id"] = out["cluster_id"].astype(str)
    if sides is not None:
        out = out[out["side"].isin([str(side) for side in sides])].copy()
    return out.drop_duplicates(["side", "TrackID"]).reset_index(drop=True)


def load_speed_tracks_with_clusters(
    speed_root: Path,
    clusters: pd.DataFrame,
    *,
    min_present_frac: float = 0.40,
) -> pd.DataFrame:
    speed_tracks = ia.load_speed_tracks(speed_root)
    cluster_lookup = clusters.rename(columns={"TrackID": "track_id"})[
        ["side", "track_id", "cluster_id", "leiden_cluster_id"]
    ].drop_duplicates(["side", "track_id"])
    out = speed_tracks.merge(cluster_lookup, on=["side", "track_id"], how="inner", validate="one_to_one")
    out = out[
        (out["present_frac"] >= float(min_present_frac))
        & out["speed_path"].map(lambda path: Path(path).exists())
    ].copy()
    if out.empty:
        raise ValueError(f"No speed tracks matched clusters with present_frac >= {min_present_frac}")
    return out.sort_values(["side", "track_id"], kind="mergesort").reset_index(drop=True)


def attach_side_aware_cluster_labels(
    interactions: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    drop_unclustered: bool = True,
) -> pd.DataFrame:
    required = {"side", "antenna_track_id", "body_track_id"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"interactions is missing columns: {sorted(missing)}")

    lookup = clusters.rename(columns={"TrackID": "track_id"})[
        ["side", "track_id", "cluster_id", "leiden_cluster_id"]
    ].drop_duplicates(["side", "track_id"])
    lookup["side"] = lookup["side"].astype(str)

    out = interactions.copy()
    out["side"] = out["side"].astype(str)
    out = out.merge(
        lookup.rename(
            columns={
                "track_id": "antenna_track_id",
                "cluster_id": "antenna_cluster_id",
                "leiden_cluster_id": "antenna_leiden_cluster_id",
            }
        ),
        on=["side", "antenna_track_id"],
        how="left",
        validate="many_to_one",
    )
    out = out.merge(
        lookup.rename(
            columns={
                "track_id": "body_track_id",
                "cluster_id": "body_cluster_id",
                "leiden_cluster_id": "body_leiden_cluster_id",
            }
        ),
        on=["side", "body_track_id"],
        how="left",
        validate="many_to_one",
    )
    if drop_unclustered:
        out = out.dropna(subset=["antenna_cluster_id", "body_cluster_id"]).copy()
    for col in ["antenna_cluster_id", "body_cluster_id"]:
        out[col] = out[col].fillna("unclustered").astype(str)
    return out.reset_index(drop=True)


def chunk_bounds_by_side(chunks: list[ia.InteractionChunk]) -> dict[str, list[tuple[int, int]]]:
    bounds: dict[str, list[tuple[int, int]]] = {}
    for chunk in chunks:
        start = int(chunk.chunk_global_frame_offset)
        stop = int(chunk.chunk_global_frame_offset + chunk.chunk_frame_count)
        bounds.setdefault(chunk.side, []).append((start, stop))
    return bounds


def recording_start_clock_seconds(chunks: list[ia.InteractionChunk]) -> int:
    if not chunks:
        raise ValueError("No chunks were selected")
    return int(chunks[0].recording_start_clock_seconds)


def _state_label(quiet: np.ndarray, active: np.ndarray, valid: np.ndarray) -> np.ndarray:
    labels = np.full(len(quiet), "missing", dtype=object)
    labels[valid & quiet] = "quiet"
    labels[valid & active] = "active"
    labels[valid & ~(quiet | active)] = "mixed"
    return labels


def build_state_bins_from_speed(
    speed_tracks: pd.DataFrame,
    chunks: list[ia.InteractionChunk],
    *,
    bin_seconds: float,
    fps: float,
    quiet_speed_threshold_mm_s: float,
    active_speed_threshold_mm_s: float,
    quiet_fraction_threshold: float = 0.80,
    active_fraction_threshold: float = 0.50,
    min_valid_fraction: float = 0.20,
    light_on_hour: float = 5.5,
) -> pd.DataFrame:
    if speed_tracks.empty:
        raise ValueError("No speed tracks supplied")
    bin_frames = max(1, int(round(float(bin_seconds) * float(fps))))
    rec_start_clock = recording_start_clock_seconds(chunks)
    side_bounds = chunk_bounds_by_side(chunks)

    rows = []
    for i, row in speed_tracks.reset_index(drop=True).iterrows():
        if i == 0 or i == len(speed_tracks) - 1 or (i + 1) % 25 == 0:
            print(f"state bins: loading speed {i + 1}/{len(speed_tracks)} {row['track_name']}")
        side = str(row["side"])
        bounds = side_bounds.get(side, [])
        if not bounds:
            continue
        speed = np.load(row["speed_path"], mmap_mode="r")
        speed_frame_min = int(row["frame_min"])
        speed_frame_stop = speed_frame_min + len(speed)

        for chunk_start, chunk_stop in bounds:
            global_start = max(speed_frame_min, int(chunk_start))
            global_stop = min(speed_frame_stop, int(chunk_stop))
            if global_stop <= global_start:
                continue
            local_start = global_start - speed_frame_min
            local_stop = global_stop - speed_frame_min
            values = np.asarray(speed[local_start:local_stop], dtype=np.float32)
            global_frames = np.arange(global_start, global_stop, dtype=np.int64)
            bin_idx = global_frames // bin_frames
            first_bin = int(bin_idx.min())
            n_bins = int(bin_idx.max() - first_bin + 1)
            local_bin = bin_idx - first_bin

            finite = np.isfinite(values)
            expected_count = np.bincount(local_bin, minlength=n_bins).astype(np.int64)
            valid_count = np.bincount(local_bin[finite], minlength=n_bins).astype(np.int64)
            speed_sum = np.bincount(local_bin[finite], weights=values[finite], minlength=n_bins)
            quiet_sum = np.bincount(
                local_bin[finite],
                weights=(values[finite] <= float(quiet_speed_threshold_mm_s)).astype(np.float64),
                minlength=n_bins,
            )
            active_sum = np.bincount(
                local_bin[finite],
                weights=(values[finite] >= float(active_speed_threshold_mm_s)).astype(np.float64),
                minlength=n_bins,
            )

            for offset in range(n_bins):
                rows.append(
                    {
                        "side": side,
                        "track_name": row["track_name"],
                        "track_id": int(row["track_id"]),
                        "cluster_id": str(row["cluster_id"]),
                        "leiden_cluster_id": row["leiden_cluster_id"],
                        "bin_index": int(first_bin + offset),
                        "bin_start_frame": int((first_bin + offset) * bin_frames),
                        "bin_end_frame": int((first_bin + offset + 1) * bin_frames - 1),
                        "_speed_sum": float(speed_sum[offset]),
                        "_quiet_sum": float(quiet_sum[offset]),
                        "_active_sum": float(active_sum[offset]),
                        "n_valid_speed_frames": int(valid_count[offset]),
                        "n_expected_frames": int(expected_count[offset]),
                    }
                )

    if not rows:
        raise ValueError("No state bins overlapped the selected chunks")

    grouped = (
        pd.DataFrame(rows)
        .groupby(
            [
                "side",
                "track_name",
                "track_id",
                "cluster_id",
                "leiden_cluster_id",
                "bin_index",
                "bin_start_frame",
                "bin_end_frame",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            _speed_sum=("_speed_sum", "sum"),
            _quiet_sum=("_quiet_sum", "sum"),
            _active_sum=("_active_sum", "sum"),
            n_valid_speed_frames=("n_valid_speed_frames", "sum"),
            n_expected_frames=("n_expected_frames", "sum"),
        )
        .sort_values(["side", "track_id", "bin_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    denom = grouped["n_valid_speed_frames"].replace(0, np.nan).astype(float)
    grouped["mean_speed_mm_s"] = grouped["_speed_sum"] / denom
    grouped["quiet_fraction"] = grouped["_quiet_sum"] / denom
    grouped["active_fraction"] = grouped["_active_sum"] / denom
    grouped["valid_fraction"] = grouped["n_valid_speed_frames"] / grouped["n_expected_frames"].replace(0, np.nan)
    grouped["quiet_now"] = grouped["quiet_fraction"] >= float(quiet_fraction_threshold)
    grouped["active_now"] = grouped["active_fraction"] >= float(active_fraction_threshold)
    grouped["state_valid"] = grouped["valid_fraction"] >= float(min_valid_fraction)
    grouped.loc[~grouped["state_valid"], ["quiet_now", "active_now"]] = False
    grouped["state"] = _state_label(
        grouped["quiet_now"].to_numpy(bool),
        grouped["active_now"].to_numpy(bool),
        grouped["state_valid"].to_numpy(bool),
    )

    elapsed_s = grouped["bin_start_frame"].to_numpy(np.float64) / float(fps)
    clock_h = ((float(rec_start_clock) + elapsed_s) / 3600.0) % 24.0
    hours_since_light_on = (clock_h - float(light_on_hour)) % 24.0
    grouped["elapsed_time_h"] = elapsed_s / 3600.0
    grouped["clock_hour"] = clock_h
    grouped["hours_since_light_on"] = hours_since_light_on
    angle = 2.0 * np.pi * hours_since_light_on / 24.0
    grouped["time_since_light_on_sin"] = np.sin(angle)
    grouped["time_since_light_on_cos"] = np.cos(angle)
    return grouped.drop(columns=["_speed_sum", "_quiet_sum", "_active_sum"])


def add_next_state_columns(state_bins: pd.DataFrame, *, bin_seconds: float) -> pd.DataFrame:
    out = state_bins.sort_values(["side", "track_id", "bin_index"], kind="mergesort").copy()
    next_cols = [
        "bin_index",
        "state",
        "state_valid",
        "quiet_now",
        "active_now",
        "mean_speed_mm_s",
        "quiet_fraction",
        "active_fraction",
    ]
    for col in next_cols:
        next_col = f"next_{col}"
        if col == "state":
            out[next_col] = pd.Series([None] * len(out), index=out.index, dtype=object)
        elif col in {"state_valid", "quiet_now", "active_now"}:
            out[next_col] = pd.Series([pd.NA] * len(out), index=out.index, dtype="boolean")
        else:
            out[next_col] = np.nan

    quiet_run_bins = np.zeros(len(out), dtype=np.int64)
    for (_side, _track_id), idx in out.groupby(["side", "track_id"], sort=False).groups.items():
        group = out.loc[idx].sort_values("bin_index", kind="mergesort")
        group_idx = group.index.to_numpy()
        group_positions = out.index.get_indexer(group_idx)
        for col in next_cols:
            shifted = group[col].shift(-1)
            out.loc[group_idx, f"next_{col}"] = shifted.to_numpy()

        bin_values = group["bin_index"].to_numpy(np.int64)
        quiet_values = group["quiet_now"].to_numpy(bool)
        run = 0
        prev_bin = None
        for row_pos, (frame_bin, is_quiet) in enumerate(zip(bin_values, quiet_values)):
            if prev_bin is None or frame_bin != prev_bin + 1 or not is_quiet:
                run = 1 if is_quiet else 0
            else:
                run += 1
            quiet_run_bins[group_positions[row_pos]] = run
            prev_bin = int(frame_bin)

    out["next_bin_is_contiguous"] = out["next_bin_index"].astype(float) == out["bin_index"].astype(float) + 1
    out["next_state_valid"] = out["next_state_valid"].fillna(False).astype(bool)
    out["next_quiet_now"] = out["next_quiet_now"].fillna(False).astype(bool)
    out["next_active_now"] = out["next_active_now"].fillna(False).astype(bool)
    out["quiet_run_bins"] = quiet_run_bins
    out["quiet_run_seconds"] = out["quiet_run_bins"] * float(bin_seconds)
    out["quiet_run_minutes"] = out["quiet_run_seconds"] / 60.0
    out["quiet_to_active_next_bin"] = (
        out["quiet_now"].astype(bool)
        & out["next_bin_is_contiguous"].astype(bool)
        & out["next_state_valid"].astype(bool)
        & out["next_active_now"].astype(bool)
    )
    return out.reset_index(drop=True)


def build_position_context_bins(
    chunks: list[ia.InteractionChunk],
    *,
    bin_seconds: float,
    fps: float,
    mm_per_px: float,
    bodypoint: int = 0,
    frame_step: int = 24,
    local_radius_mm: float | None = 15.0,
) -> pd.DataFrame:
    bin_frames = max(1, int(round(float(bin_seconds) * float(fps))))
    rows = []
    radius_sq = None if local_radius_mm is None else float(local_radius_mm) ** 2

    for i, chunk in enumerate(chunks, start=1):
        print(f"position context: loading chunk{chunk.chunk} {chunk.side} ({i}/{len(chunks)})")
        positions = ia.load_track_positions(chunk.track_path, bodypoint=bodypoint)
        if positions.empty:
            continue
        if int(frame_step) > 1:
            positions = positions[(positions["Frame"] % int(frame_step)) == 0].copy()
        if positions.empty:
            continue

        positions["side"] = chunk.side
        positions["global_frame"] = positions["Frame"].astype(np.int64) + int(chunk.chunk_global_frame_offset)
        positions["bin_index"] = positions["global_frame"] // bin_frames
        positions["x_mm"] = positions["x_px"].astype(float) * float(mm_per_px)
        positions["y_mm"] = positions["y_px"].astype(float) * float(mm_per_px)
        visible = positions.groupby("Frame", sort=False)["TrackID"].transform("nunique")
        positions["n_visible_ants_frame"] = visible.astype(float)

        if radius_sq is not None:
            local_counts = np.zeros(len(positions), dtype=np.float32)
            for _, frame_group in positions.groupby("Frame", sort=False):
                xy = frame_group[["x_mm", "y_mm"]].to_numpy(np.float64)
                if len(xy) <= 1:
                    counts = np.zeros(len(xy), dtype=np.float32)
                else:
                    delta = xy[:, None, :] - xy[None, :, :]
                    distance_sq = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
                    counts = ((distance_sq <= radius_sq) & (distance_sq > 0)).sum(axis=1).astype(np.float32)
                local_counts[positions.index.get_indexer(frame_group.index)] = counts
            positions["local_neighbor_count_frame"] = local_counts

        agg = {
            "x_mm": ("x_mm", "mean"),
            "y_mm": ("y_mm", "mean"),
            "mean_n_visible_ants": ("n_visible_ants_frame", "mean"),
            "n_position_samples": ("Frame", "size"),
        }
        if radius_sq is not None:
            agg["mean_local_neighbor_count"] = ("local_neighbor_count_frame", "mean")

        binned = (
            positions.groupby(["side", "TrackID", "bin_index"], as_index=False)
            .agg(**agg)
            .rename(columns={"TrackID": "track_id"})
        )
        rows.append(binned)

    if not rows:
        return pd.DataFrame(
            columns=[
                "side",
                "track_id",
                "bin_index",
                "x_mm",
                "y_mm",
                "mean_n_visible_ants",
                "mean_local_neighbor_count",
                "n_position_samples",
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(["side", "track_id", "bin_index"]).reset_index(drop=True)


def directed_interaction_onsets(
    interactions: pd.DataFrame,
    *,
    fps: float,
    event_gap_seconds: float = 2.0,
    collapse_contacts: bool = True,
    keep_first_observed_contact: bool = True,
) -> pd.DataFrame:
    if interactions.empty:
        return pd.DataFrame()
    required = {"side", "global_frame", "antenna_track_id", "body_track_id"}
    missing = required.difference(interactions.columns)
    if missing:
        raise ValueError(f"interactions is missing columns: {sorted(missing)}")

    work = interactions.copy()
    work = work[work["antenna_track_id"] != work["body_track_id"]].copy()
    for col in ["global_frame", "antenna_track_id", "body_track_id"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["global_frame", "antenna_track_id", "body_track_id"]).copy()
    work["global_frame"] = np.rint(work["global_frame"]).astype(np.int64)
    work["antenna_track_id"] = work["antenna_track_id"].astype(np.int64)
    work["body_track_id"] = work["body_track_id"].astype(np.int64)

    if collapse_contacts:
        gap_frames = max(1, int(round(float(event_gap_seconds) * float(fps))))
        sort_cols = ["side", "antenna_track_id", "body_track_id", "global_frame"]
        work = work.sort_values(sort_cols, kind="mergesort").drop_duplicates(sort_cols)
        frame_gap = work.groupby(["side", "antenna_track_id", "body_track_id"], sort=False)["global_frame"].diff()
        is_new_contact = frame_gap > gap_frames
        if keep_first_observed_contact:
            is_new_contact = frame_gap.isna() | is_new_contact
        work = work[is_new_contact].copy()
    else:
        work = work.drop_duplicates(["side", "antenna_track_id", "body_track_id", "global_frame"]).copy()

    work = work.sort_values(["side", "global_frame", "antenna_track_id", "body_track_id"], kind="mergesort")
    work = work.reset_index(drop=True)
    work.insert(0, "directed_event_id", np.arange(len(work), dtype=np.int64))

    antenna = pd.DataFrame(
        {
            "directed_event_id": work["directed_event_id"],
            "side": work["side"],
            "global_frame": work["global_frame"],
            "interaction_role": "antenna",
            "focal_track_id": work["antenna_track_id"],
            "partner_track_id": work["body_track_id"],
            "focal_cluster_id": work.get("antenna_cluster_id", pd.Series(index=work.index, dtype=object)),
            "partner_cluster_id": work.get("body_cluster_id", pd.Series(index=work.index, dtype=object)),
            "focal_leiden_cluster_id": work.get("antenna_leiden_cluster_id", pd.Series(index=work.index, dtype=object)),
            "partner_leiden_cluster_id": work.get("body_leiden_cluster_id", pd.Series(index=work.index, dtype=object)),
        }
    )
    body = pd.DataFrame(
        {
            "directed_event_id": work["directed_event_id"],
            "side": work["side"],
            "global_frame": work["global_frame"],
            "interaction_role": "body",
            "focal_track_id": work["body_track_id"],
            "partner_track_id": work["antenna_track_id"],
            "focal_cluster_id": work.get("body_cluster_id", pd.Series(index=work.index, dtype=object)),
            "partner_cluster_id": work.get("antenna_cluster_id", pd.Series(index=work.index, dtype=object)),
            "focal_leiden_cluster_id": work.get("body_leiden_cluster_id", pd.Series(index=work.index, dtype=object)),
            "partner_leiden_cluster_id": work.get("antenna_leiden_cluster_id", pd.Series(index=work.index, dtype=object)),
        }
    )
    focal = pd.concat([antenna, body], ignore_index=True)
    focal["side"] = focal["side"].astype(str)
    for col in ["focal_cluster_id", "partner_cluster_id"]:
        focal[col] = focal[col].fillna("unclustered").astype(str)
    return focal.sort_values(["side", "focal_track_id", "global_frame"], kind="mergesort").reset_index(drop=True)


def _safe_suffix_seconds(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return f"{int(value)}s"
    return f"{value:g}s".replace(".", "p")


def add_interaction_history_features(
    state_bins: pd.DataFrame,
    focal_events: pd.DataFrame,
    *,
    fps: float,
    cumulative_windows_seconds: tuple[float, ...] = (30.0, 120.0, 600.0),
    lag_windows_seconds: tuple[tuple[float, float], ...] = ((0.0, 30.0), (30.0, 120.0), (120.0, 600.0)),
    leaky_tau_seconds: tuple[float, ...] = (60.0, 300.0),
    partner_cluster_window_seconds: float = 600.0,
    max_partner_cluster_features: int = 8,
) -> pd.DataFrame:
    out = state_bins.sort_values(["side", "track_id", "bin_index"], kind="mergesort").copy()
    out["n_interactions_prev_any"] = 0
    max_history_seconds = max(
        [*map(float, cumulative_windows_seconds), *[float(high) for _low, high in lag_windows_seconds], *map(float, leaky_tau_seconds)]
    )
    out["time_since_last_interaction_seconds"] = np.nan
    out["time_since_last_interaction_capped_seconds"] = float(max_history_seconds)

    for window_s in cumulative_windows_seconds:
        suffix = _safe_suffix_seconds(window_s)
        for prefix in ["n_interactions", "n_interactions_as_antenna", "n_interactions_as_body"]:
            out[f"{prefix}_prev_{suffix}"] = 0
        out[f"n_unique_partners_prev_{suffix}"] = 0
        out[f"n_partner_clusters_prev_{suffix}"] = 0
        out[f"has_interaction_prev_{suffix}"] = False

    for low_s, high_s in lag_windows_seconds:
        suffix = f"{_safe_suffix_seconds(low_s)}_{_safe_suffix_seconds(high_s)}"
        out[f"n_interactions_lag_{suffix}"] = 0
        out[f"n_interactions_as_antenna_lag_{suffix}"] = 0
        out[f"n_interactions_as_body_lag_{suffix}"] = 0

    for tau_s in leaky_tau_seconds:
        out[f"leaky_interactions_tau_{_safe_suffix_seconds(tau_s)}"] = 0.0

    partner_clusters = []
    if not focal_events.empty and max_partner_cluster_features > 0:
        partner_clusters = (
            focal_events["partner_cluster_id"].dropna().astype(str).value_counts().head(int(max_partner_cluster_features)).index.tolist()
        )
        partner_suffix = _safe_suffix_seconds(partner_cluster_window_seconds)
        for label in partner_clusters:
            safe_label = str(label).replace(" ", "_").replace("/", "_")
            out[f"n_partner_cluster_{safe_label}_prev_{partner_suffix}"] = 0

    if focal_events.empty:
        return out.reset_index(drop=True)

    events = focal_events.copy()
    events["side"] = events["side"].astype(str)
    events["focal_track_id"] = pd.to_numeric(events["focal_track_id"], errors="coerce").astype("Int64")
    events["global_frame"] = pd.to_numeric(events["global_frame"], errors="coerce").astype("Int64")
    events = events.dropna(subset=["focal_track_id", "global_frame"]).astype(
        {"focal_track_id": "int64", "global_frame": "int64"}
    )

    for (side, track_id), idx in out.groupby(["side", "track_id"], sort=False).groups.items():
        track_events = events[(events["side"] == side) & (events["focal_track_id"] == int(track_id))]
        if track_events.empty:
            continue
        track_events = track_events.sort_values("global_frame", kind="mergesort")
        frames = track_events["global_frame"].to_numpy(np.int64)
        starts = out.loc[idx, "bin_start_frame"].to_numpy(np.int64)
        roles = track_events["interaction_role"].astype(str).to_numpy()
        is_antenna = (roles == "antenna").astype(np.int64)
        is_body = (roles == "body").astype(np.int64)
        antenna_prefix = np.r_[0, np.cumsum(is_antenna)]
        body_prefix = np.r_[0, np.cumsum(is_body)]

        previous_right = np.searchsorted(frames, starts, side="left")
        has_previous = previous_right > 0
        out.loc[idx, "n_interactions_prev_any"] = previous_right
        last_frames = np.full(len(starts), np.nan, dtype=float)
        last_frames[has_previous] = frames[previous_right[has_previous] - 1]
        since_last = (starts.astype(float) - last_frames) / float(fps)
        out.loc[idx, "time_since_last_interaction_seconds"] = since_last
        out.loc[idx, "time_since_last_interaction_capped_seconds"] = np.where(
            np.isfinite(since_last),
            np.minimum(since_last, float(max_history_seconds)),
            float(max_history_seconds),
        )

        for window_s in cumulative_windows_seconds:
            suffix = _safe_suffix_seconds(window_s)
            left = np.searchsorted(frames, starts - int(round(float(window_s) * float(fps))), side="left")
            right = previous_right
            counts = right - left
            antenna_counts = antenna_prefix[right] - antenna_prefix[left]
            body_counts = body_prefix[right] - body_prefix[left]
            out.loc[idx, f"n_interactions_prev_{suffix}"] = counts
            out.loc[idx, f"n_interactions_as_antenna_prev_{suffix}"] = antenna_counts
            out.loc[idx, f"n_interactions_as_body_prev_{suffix}"] = body_counts
            out.loc[idx, f"has_interaction_prev_{suffix}"] = counts > 0

            partners = track_events["partner_track_id"].to_numpy()
            partner_cluster_values = track_events["partner_cluster_id"].astype(str).to_numpy()
            unique_partners = np.zeros(len(starts), dtype=np.int64)
            unique_partner_clusters = np.zeros(len(starts), dtype=np.int64)
            for row_pos, (lft, rgt) in enumerate(zip(left, right)):
                if rgt <= lft:
                    continue
                unique_partners[row_pos] = len(pd.unique(partners[lft:rgt]))
                unique_partner_clusters[row_pos] = len(pd.unique(partner_cluster_values[lft:rgt]))
            out.loc[idx, f"n_unique_partners_prev_{suffix}"] = unique_partners
            out.loc[idx, f"n_partner_clusters_prev_{suffix}"] = unique_partner_clusters

        for low_s, high_s in lag_windows_seconds:
            suffix = f"{_safe_suffix_seconds(low_s)}_{_safe_suffix_seconds(high_s)}"
            left = np.searchsorted(frames, starts - int(round(float(high_s) * float(fps))), side="left")
            right = np.searchsorted(frames, starts - int(round(float(low_s) * float(fps))), side="left")
            out.loc[idx, f"n_interactions_lag_{suffix}"] = right - left
            out.loc[idx, f"n_interactions_as_antenna_lag_{suffix}"] = antenna_prefix[right] - antenna_prefix[left]
            out.loc[idx, f"n_interactions_as_body_lag_{suffix}"] = body_prefix[right] - body_prefix[left]

        for tau_s in leaky_tau_seconds:
            suffix = _safe_suffix_seconds(tau_s)
            horizon = int(round(5.0 * float(tau_s) * float(fps)))
            left = np.searchsorted(frames, starts - horizon, side="left")
            leaky = np.zeros(len(starts), dtype=np.float64)
            for row_pos, (lft, rgt, start_frame) in enumerate(zip(left, previous_right, starts)):
                if rgt <= lft:
                    continue
                ages_s = (float(start_frame) - frames[lft:rgt].astype(float)) / float(fps)
                leaky[row_pos] = float(np.exp(-ages_s / float(tau_s)).sum())
            out.loc[idx, f"leaky_interactions_tau_{suffix}"] = leaky

        if partner_clusters:
            partner_suffix = _safe_suffix_seconds(partner_cluster_window_seconds)
            left = np.searchsorted(
                frames,
                starts - int(round(float(partner_cluster_window_seconds) * float(fps))),
                side="left",
            )
            right = previous_right
            partner_cluster_values = track_events["partner_cluster_id"].astype(str).to_numpy()
            for label in partner_clusters:
                safe_label = str(label).replace(" ", "_").replace("/", "_")
                is_label = (partner_cluster_values == str(label)).astype(np.int64)
                label_prefix = np.r_[0, np.cumsum(is_label)]
                out.loc[idx, f"n_partner_cluster_{safe_label}_prev_{partner_suffix}"] = label_prefix[right] - label_prefix[left]

    return out.reset_index(drop=True)


def make_quiet_to_active_design(state_bins: pd.DataFrame) -> pd.DataFrame:
    required = {"quiet_now", "next_bin_is_contiguous", "next_state_valid", "quiet_to_active_next_bin"}
    missing = required.difference(state_bins.columns)
    if missing:
        raise ValueError(f"state_bins is missing columns: {sorted(missing)}")

    design = state_bins[
        state_bins["quiet_now"].astype(bool)
        & state_bins["next_bin_is_contiguous"].astype(bool)
        & state_bins["next_state_valid"].astype(bool)
    ].copy()
    design["activate_next_bin"] = design["quiet_to_active_next_bin"].astype(int)
    return design.reset_index(drop=True)


def cluster_state_timeseries(state_bins: pd.DataFrame) -> pd.DataFrame:
    summary = (
        state_bins[state_bins["state_valid"].astype(bool)]
        .groupby(["side", "cluster_id", "bin_index"], as_index=False)
        .agg(
            elapsed_time_h=("elapsed_time_h", "first"),
            hours_since_light_on=("hours_since_light_on", "first"),
            quiet_fraction=("quiet_now", "mean"),
            active_fraction=("active_now", "mean"),
            mean_speed_mm_s=("mean_speed_mm_s", "mean"),
            n_ants=("track_id", "nunique"),
        )
        .sort_values(["side", "cluster_id", "bin_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    return summary


def plot_cluster_state_timeseries(
    state_bins: pd.DataFrame,
    *,
    value_col: str = "active_fraction",
    x_col: str = "elapsed_time_h",
    smooth_bins: float = 2.0,
    title: str | None = None,
    ylim: tuple[float, float] | None = (0.0, 1.0),
    cmap: str = "tab20",
) -> pd.DataFrame:
    summary = cluster_state_timeseries(state_bins)
    plot_df = summary.copy()
    y_col = value_col
    if smooth_bins > 0:
        y_col = f"smoothed_{value_col}"
        plot_df[y_col] = np.nan
        for (_side, _cluster), idx in plot_df.groupby(["side", "cluster_id"]).groups.items():
            group = plot_df.loc[idx].sort_values(x_col, kind="mergesort")
            plot_df.loc[group.index, y_col] = ia.smooth_series(group[value_col].to_numpy(float), smooth_bins)

    sides = [side for side in SIDES if side in set(plot_df["side"])]
    fig, axes = plt.subplots(len(sides), 1, figsize=(12, max(4.0, 3.2 * len(sides))), sharex=True, sharey=True, squeeze=False)
    for ax, side in zip(axes.ravel(), sides):
        side_df = plot_df[plot_df["side"] == side]
        clusters = side_df["cluster_id"].drop_duplicates().tolist()
        colors = plt.get_cmap(cmap, max(1, len(clusters)))
        for i, cluster_id in enumerate(clusters):
            group = side_df[side_df["cluster_id"] == cluster_id].sort_values(x_col, kind="mergesort")
            n_ants = int(group["n_ants"].max())
            ax.plot(group[x_col], group[y_col], lw=1.6, color=colors(i), label=f"{cluster_id} n={n_ants}")
        ax.set_ylabel(value_col)
        ax.set_title(f"{side} colony")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncols=2)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[-1, 0].set_xlabel("Elapsed time (h)" if x_col == "elapsed_time_h" else x_col)
    fig.suptitle(title or f"Cluster {value_col} over time")
    fig.tight_layout()
    _save_current_figures_if_enabled()
    return plot_df


def binned_transition_hazard(
    design: pd.DataFrame,
    *,
    predictor_col: str,
    group_col: str = "side",
    max_count_bin: int = 5,
) -> pd.DataFrame:
    if design.empty:
        raise ValueError("No transition design rows")
    work = design.copy()
    values = pd.to_numeric(work[predictor_col], errors="coerce").fillna(0)
    bins = np.minimum(values.to_numpy(float), float(max_count_bin)).astype(int)
    work["predictor_bin"] = bins
    work["predictor_bin_label"] = [f"{value}+" if value == max_count_bin else str(value) for value in bins]
    summary = (
        work.groupby([group_col, "predictor_bin", "predictor_bin_label"], as_index=False)
        .agg(
            n_bins=("activate_next_bin", "size"),
            n_transitions=("activate_next_bin", "sum"),
            transition_probability=("activate_next_bin", "mean"),
        )
        .sort_values([group_col, "predictor_bin"], kind="mergesort")
        .reset_index(drop=True)
    )
    return summary


def plot_transition_hazard_by_contact_count(
    design: pd.DataFrame,
    *,
    predictor_col: str,
    group_col: str = "side",
    max_count_bin: int = 5,
    title: str | None = None,
) -> pd.DataFrame:
    summary = binned_transition_hazard(
        design,
        predictor_col=predictor_col,
        group_col=group_col,
        max_count_bin=max_count_bin,
    )
    groups = summary[group_col].drop_duplicates().tolist()
    colors = plt.get_cmap("tab10", max(1, len(groups)))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, group_value in enumerate(groups):
        group = summary[summary[group_col] == group_value].sort_values("predictor_bin", kind="mergesort")
        ax.plot(
            group["predictor_bin"],
            group["transition_probability"],
            marker="o",
            lw=1.8,
            color=colors(i),
            label=str(group_value),
        )
        for _, row in group.iterrows():
            ax.text(
                row["predictor_bin"],
                row["transition_probability"],
                f"n={int(row['n_bins'])}",
                fontsize=7,
                ha="center",
                va="bottom",
            )
    tick_values = np.arange(max_count_bin + 1)
    ax.set_xticks(tick_values)
    ax.set_xticklabels([f"{value}+" if value == max_count_bin else str(value) for value in tick_values])
    ax.set_xlabel(predictor_col)
    ax.set_ylabel("P(active next bin | quiet now)")
    ax.set_title(title or f"Quiet-to-active hazard by {predictor_col}")
    ax.grid(True, alpha=0.25)
    ax.legend(title=group_col)
    fig.tight_layout()
    _save_current_figures_if_enabled()
    return summary


def plot_transition_hazard_by_time_since_contact(
    design: pd.DataFrame,
    *,
    group_col: str = "side",
    time_col: str = "time_since_last_interaction_seconds",
    bins_seconds: tuple[float, ...] = (0.0, 30.0, 120.0, 600.0),
    title: str | None = None,
) -> pd.DataFrame:
    work = design.copy()
    values = pd.to_numeric(work[time_col], errors="coerce")
    labels = []
    for value in values:
        if not np.isfinite(value):
            labels.append("no previous")
        elif value <= bins_seconds[1]:
            labels.append(f"0-{bins_seconds[1]:g}s")
        elif value <= bins_seconds[2]:
            labels.append(f"{bins_seconds[1]:g}-{bins_seconds[2]:g}s")
        elif value <= bins_seconds[3]:
            labels.append(f"{bins_seconds[2]:g}-{bins_seconds[3]:g}s")
        else:
            labels.append(f">{bins_seconds[3]:g}s")
    work["time_since_contact_bin"] = labels
    order = [f"0-{bins_seconds[1]:g}s", f"{bins_seconds[1]:g}-{bins_seconds[2]:g}s", f"{bins_seconds[2]:g}-{bins_seconds[3]:g}s", f">{bins_seconds[3]:g}s", "no previous"]
    summary = (
        work.groupby([group_col, "time_since_contact_bin"], as_index=False)
        .agg(
            n_bins=("activate_next_bin", "size"),
            n_transitions=("activate_next_bin", "sum"),
            transition_probability=("activate_next_bin", "mean"),
        )
    )
    summary["order"] = summary["time_since_contact_bin"].map({label: i for i, label in enumerate(order)})
    summary = summary.sort_values([group_col, "order"], kind="mergesort").reset_index(drop=True)

    groups = summary[group_col].drop_duplicates().tolist()
    colors = plt.get_cmap("tab10", max(1, len(groups)))
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for i, group_value in enumerate(groups):
        group = summary[summary[group_col] == group_value].sort_values("order", kind="mergesort")
        ax.plot(group["order"], group["transition_probability"], marker="o", lw=1.8, color=colors(i), label=str(group_value))
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_xlabel("Time since last interaction")
    ax.set_ylabel("P(active next bin | quiet now)")
    ax.set_title(title or "Quiet-to-active hazard by time since last contact")
    ax.grid(True, alpha=0.25)
    ax.legend(title=group_col)
    fig.tight_layout()
    _save_current_figures_if_enabled()
    return summary.drop(columns=["order"])


def build_event_triggered_state_curve(
    state_bins: pd.DataFrame,
    focal_events: pd.DataFrame,
    *,
    fps: float,
    bin_seconds: float,
    pre_seconds: float,
    post_seconds: float,
    condition_col: str = "interaction_role",
    require_quiet_at_event: bool = True,
    max_events: int | None = 5000,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if focal_events.empty:
        raise ValueError("No focal events supplied")
    bin_frames = max(1, int(round(float(bin_seconds) * float(fps))))
    events = focal_events.copy()
    events["event_bin_index"] = (events["global_frame"].astype(np.int64) // bin_frames).astype(np.int64)
    events = events.drop_duplicates(["side", "focal_track_id", "event_bin_index", condition_col]).copy()
    event_state = state_bins[
        ["side", "track_id", "bin_index", "quiet_now", "active_now", "state_valid", "cluster_id"]
    ].rename(columns={"track_id": "focal_track_id", "bin_index": "event_bin_index", "cluster_id": "focal_state_cluster_id"})
    events = events.merge(event_state, on=["side", "focal_track_id", "event_bin_index"], how="inner", validate="many_to_one")
    if require_quiet_at_event:
        events = events[events["quiet_now"].astype(bool) & events["state_valid"].astype(bool)].copy()
    if max_events is not None and len(events) > int(max_events):
        events = events.sample(n=int(max_events), random_state=int(random_state)).sort_values(["side", "global_frame"], kind="mergesort")
    events = events.reset_index(drop=True)
    events["trigger_event_id"] = np.arange(len(events), dtype=np.int64)

    pre_bins = int(math.ceil(float(pre_seconds) / float(bin_seconds)))
    post_bins = int(math.ceil(float(post_seconds) / float(bin_seconds)))
    rel_bins = np.arange(-pre_bins, post_bins + 1, dtype=np.int64)
    state_lookup = state_bins.set_index(["side", "track_id", "bin_index"])[
        ["mean_speed_mm_s", "quiet_fraction", "active_fraction", "quiet_now", "active_now", "state_valid"]
    ]

    rows = []
    for event in events.itertuples(index=False):
        for rel_bin in rel_bins:
            key = (event.side, int(event.focal_track_id), int(event.event_bin_index + rel_bin))
            if key not in state_lookup.index:
                continue
            state = state_lookup.loc[key]
            if isinstance(state, pd.DataFrame):
                state = state.iloc[0]
            rows.append(
                {
                    "trigger_event_id": int(event.trigger_event_id),
                    "side": event.side,
                    "focal_track_id": int(event.focal_track_id),
                    "condition": getattr(event, condition_col),
                    "relative_bin": int(rel_bin),
                    "relative_seconds": float(rel_bin * float(bin_seconds)),
                    "mean_speed_mm_s": state["mean_speed_mm_s"],
                    "quiet_fraction": state["quiet_fraction"],
                    "active_fraction": state["active_fraction"],
                    "quiet_now": bool(state["quiet_now"]),
                    "active_now": bool(state["active_now"]),
                    "state_valid": bool(state["state_valid"]),
                }
            )

    if not rows:
        raise ValueError("No state bins overlapped the selected interaction events")
    curve_rows = pd.DataFrame(rows)
    summary = (
        curve_rows[curve_rows["state_valid"].astype(bool)]
        .groupby(["condition", "relative_bin", "relative_seconds"], as_index=False)
        .agg(
            mean_speed_mm_s=("mean_speed_mm_s", "mean"),
            active_fraction=("active_now", "mean"),
            quiet_fraction=("quiet_now", "mean"),
            n_events=("trigger_event_id", "nunique"),
        )
        .sort_values(["condition", "relative_bin"], kind="mergesort")
        .reset_index(drop=True)
    )
    return curve_rows, summary


def build_interaction_control_triggered_state_curve(
    state_bins: pd.DataFrame,
    focal_events: pd.DataFrame,
    *,
    fps: float,
    bin_seconds: float,
    pre_seconds: float,
    post_seconds: float,
    require_quiet_at_trigger: bool = True,
    control_replicates: int = 2,
    control_exclude_seconds: float = 120.0,
    control_exclusion_events: pd.DataFrame | None = None,
    control_match_levels: tuple[str, ...] = ("same_ant", "same_side_cluster"),
    separate_control_match_levels: bool = True,
    pool_interaction_roles: bool = True,
    max_events: int | None = 5000,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if focal_events.empty:
        raise ValueError("No focal events supplied")

    bin_frames = max(1, int(round(float(bin_seconds) * float(fps))))
    events = focal_events.copy()
    events["event_bin_index"] = (events["global_frame"].astype(np.int64) // bin_frames).astype(np.int64)
    events["trigger_condition"] = (
        "interaction"
        if pool_interaction_roles
        else "interaction_" + events["interaction_role"].astype(str)
    )
    dedup_cols = ["side", "focal_track_id", "event_bin_index", "trigger_condition"]
    events = events.drop_duplicates(dedup_cols).copy()

    event_state = state_bins[
        ["side", "track_id", "bin_index", "quiet_now", "active_now", "state_valid", "cluster_id"]
    ].rename(columns={"track_id": "focal_track_id", "bin_index": "event_bin_index"})
    events = events.merge(event_state, on=["side", "focal_track_id", "event_bin_index"], how="inner", validate="many_to_one")
    if require_quiet_at_trigger:
        events = events[events["quiet_now"].astype(bool) & events["state_valid"].astype(bool)].copy()
    if max_events is not None and len(events) > int(max_events):
        events = events.sample(n=int(max_events), random_state=int(random_state)).sort_values(
            ["side", "global_frame"],
            kind="mergesort",
        )
    events = events.reset_index(drop=True)
    events["matched_event_id"] = np.arange(len(events), dtype=np.int64)
    if events.empty:
        raise ValueError("No interaction triggers remained after state filtering")

    rng = np.random.default_rng(int(random_state))
    all_event_bins = (
        focal_events.copy()
        if control_exclusion_events is None
        else control_exclusion_events.copy()
    )
    all_event_bins["event_bin_index"] = (all_event_bins["global_frame"].astype(np.int64) // bin_frames).astype(np.int64)
    all_event_bins = all_event_bins[["side", "focal_track_id", "event_bin_index"]].drop_duplicates()
    event_bins_by_track = {
        (str(side), int(track_id)): np.sort(group["event_bin_index"].to_numpy(np.int64))
        for (side, track_id), group in all_event_bins.groupby(["side", "focal_track_id"], sort=False)
    }

    state_candidates = state_bins[
        ["side", "track_id", "bin_index", "quiet_now", "state_valid", "cluster_id"]
    ].copy()
    if require_quiet_at_trigger:
        state_candidates = state_candidates[
            state_candidates["quiet_now"].astype(bool) & state_candidates["state_valid"].astype(bool)
        ].copy()
    else:
        state_candidates = state_candidates[state_candidates["state_valid"].astype(bool)].copy()

    state_candidates["side"] = state_candidates["side"].astype(str)
    state_candidates["track_id"] = state_candidates["track_id"].astype(int)
    state_candidates["cluster_id"] = state_candidates["cluster_id"].astype(str)

    exclude_bins = max(0, int(math.ceil(float(control_exclude_seconds) / float(bin_seconds))))
    no_onset_parts = []
    for (side, track_id), group in state_candidates.groupby(["side", "track_id"], sort=False):
        bins = group["bin_index"].to_numpy(np.int64)
        focal_bins = event_bins_by_track.get((str(side), int(track_id)), np.asarray([], dtype=np.int64))
        keep = np.ones(len(group), dtype=bool)
        if focal_bins.size:
            left = np.searchsorted(focal_bins, bins - exclude_bins, side="left")
            right = np.searchsorted(focal_bins, bins + exclude_bins, side="right")
            keep = (right - left) == 0
        no_onset_parts.append(group.loc[keep])
    no_onset_candidates = (
        pd.concat(no_onset_parts, ignore_index=True)
        if no_onset_parts
        else state_candidates.iloc[0:0].copy()
    )

    allowed_match_levels = {"same_ant", "same_side_cluster", "same_side"}
    bad_levels = set(control_match_levels).difference(allowed_match_levels)
    if bad_levels:
        raise ValueError(f"Unknown control_match_levels: {sorted(bad_levels)}")

    control_rows = []
    controls_found = 0
    for event in events.itertuples(index=False):
        key = (str(event.side), int(event.focal_track_id))
        candidates = pd.DataFrame()
        match_level_used = None
        for match_level in control_match_levels:
            if match_level == "same_ant":
                candidate_mask = (
                    (no_onset_candidates["side"] == key[0])
                    & (no_onset_candidates["track_id"] == key[1])
                )
            elif match_level == "same_side_cluster":
                candidate_mask = (
                    (no_onset_candidates["side"] == key[0])
                    & (no_onset_candidates["cluster_id"] == str(event.cluster_id))
                    & (no_onset_candidates["track_id"] != key[1])
                )
            elif match_level == "same_side":
                candidate_mask = (
                    (no_onset_candidates["side"] == key[0])
                    & (no_onset_candidates["track_id"] != key[1])
                )
            else:
                continue
            candidates = no_onset_candidates.loc[candidate_mask].copy()
            candidates = candidates[candidates["bin_index"] != int(event.event_bin_index)].copy()
            if not candidates.empty:
                match_level_used = match_level
                break
        if candidates.empty:
            continue

        n_pick = min(int(control_replicates), len(candidates))
        chosen_idx = rng.choice(candidates.index.to_numpy(), size=n_pick, replace=False)
        chosen = candidates.loc[np.sort(chosen_idx)]
        for control_id, control in enumerate(chosen.itertuples(index=False)):
            control_rows.append(
                {
                    "matched_event_id": int(event.matched_event_id),
                    "control_id": int(control_id),
                    "side": key[0],
                    "event_focal_track_id": key[1],
                    "focal_track_id": int(control.track_id),
                    "center_bin_index": int(control.bin_index),
                    "trigger_condition": (
                        f"matched_no_interaction_{match_level_used}"
                        if separate_control_match_levels
                        else "matched_no_interaction"
                    ),
                    "control_match_level": match_level_used,
                    "cluster_id": getattr(control, "cluster_id"),
                }
            )
            controls_found += 1

    interaction_triggers = pd.DataFrame(
        {
            "matched_event_id": events["matched_event_id"],
            "control_id": -1,
            "side": events["side"].astype(str),
            "event_focal_track_id": events["focal_track_id"].astype(int),
            "focal_track_id": events["focal_track_id"].astype(int),
            "center_bin_index": events["event_bin_index"].astype(int),
            "trigger_condition": events["trigger_condition"].astype(str),
            "control_match_level": "interaction",
            "cluster_id": events["cluster_id"],
        }
    )
    control_triggers = pd.DataFrame(control_rows)
    if controls_found == 0:
        print(
            "WARNING: no matched no-interaction controls were found; "
            "plotting interaction triggers only. Try smaller control_exclude_seconds "
            "or include 'same_side' in control_match_levels."
        )
    trigger_table = pd.concat([interaction_triggers, control_triggers], ignore_index=True)
    trigger_table = trigger_table.sort_values(
        ["matched_event_id", "control_id", "trigger_condition"],
        kind="mergesort",
    ).reset_index(drop=True)
    trigger_table.insert(0, "trigger_row_id", np.arange(len(trigger_table), dtype=np.int64))

    pre_bins = int(math.ceil(float(pre_seconds) / float(bin_seconds)))
    post_bins = int(math.ceil(float(post_seconds) / float(bin_seconds)))
    rel_bins = np.arange(-pre_bins, post_bins + 1, dtype=np.int64)
    state_lookup = state_bins.set_index(["side", "track_id", "bin_index"])[
        ["mean_speed_mm_s", "quiet_fraction", "active_fraction", "quiet_now", "active_now", "state_valid"]
    ]

    rows = []
    for trigger in trigger_table.itertuples(index=False):
        for rel_bin in rel_bins:
            key = (trigger.side, int(trigger.focal_track_id), int(trigger.center_bin_index + rel_bin))
            if key not in state_lookup.index:
                continue
            state = state_lookup.loc[key]
            if isinstance(state, pd.DataFrame):
                state = state.iloc[0]
            rows.append(
                {
                    "trigger_row_id": int(trigger.trigger_row_id),
                    "matched_event_id": int(trigger.matched_event_id),
                    "control_id": int(trigger.control_id),
                    "side": trigger.side,
                    "event_focal_track_id": int(trigger.event_focal_track_id),
                    "focal_track_id": int(trigger.focal_track_id),
                    "condition": trigger.trigger_condition,
                    "control_match_level": trigger.control_match_level,
                    "relative_bin": int(rel_bin),
                    "relative_seconds": float(rel_bin * float(bin_seconds)),
                    "mean_speed_mm_s": state["mean_speed_mm_s"],
                    "quiet_fraction": state["quiet_fraction"],
                    "active_fraction": state["active_fraction"],
                    "quiet_now": bool(state["quiet_now"]),
                    "active_now": bool(state["active_now"]),
                    "state_valid": bool(state["state_valid"]),
                }
            )

    if not rows:
        raise ValueError("No state bins overlapped the selected interaction/control triggers")
    curve_rows = pd.DataFrame(rows)
    summary = (
        curve_rows[curve_rows["state_valid"].astype(bool)]
        .groupby(["condition", "control_match_level", "relative_bin", "relative_seconds"], as_index=False)
        .agg(
            mean_speed_mm_s=("mean_speed_mm_s", "mean"),
            active_fraction=("active_now", "mean"),
            quiet_fraction=("quiet_now", "mean"),
            n_triggers=("trigger_row_id", "nunique"),
            n_matched_events=("matched_event_id", "nunique"),
        )
        .sort_values(["condition", "relative_bin"], kind="mergesort")
        .reset_index(drop=True)
    )
    return trigger_table, curve_rows, summary


def plot_event_triggered_state_curve(
    summary: pd.DataFrame,
    *,
    y_col: str = "active_fraction",
    title: str | None = None,
    marker: str | None = "o",
) -> pd.DataFrame:
    if summary.empty:
        raise ValueError("No event-triggered summary rows")
    conditions = summary["condition"].drop_duplicates().tolist()
    colors = plt.get_cmap("tab10", max(1, len(conditions)))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, condition in enumerate(conditions):
        group = summary[summary["condition"] == condition].sort_values("relative_seconds", kind="mergesort")
        ax.plot(
            group["relative_seconds"] / 60.0,
            group[y_col],
            marker=marker,
            lw=1.8,
            markersize=4 if marker else 0,
            color=colors(i),
            label=str(condition),
        )
    ax.axvline(0, color="0.25", lw=1, linestyle="--")
    ax.set_xlabel("Minutes relative to interaction onset")
    ax.set_ylabel(y_col)
    ax.set_title(title or f"Interaction-triggered {y_col}")
    ax.grid(True, alpha=0.25)
    ax.legend(title="condition")
    fig.tight_layout()
    _save_current_figures_if_enabled()
    return summary


def _encoded_design_matrix(
    design: pd.DataFrame,
    *,
    predictors: list[str],
    categorical_predictors: list[str],
) -> pd.DataFrame:
    present = [col for col in predictors if col in design.columns]
    if not present:
        raise ValueError("No requested predictors are present in design")
    data = design[present].copy()
    categorical = [col for col in categorical_predictors if col in data.columns]
    for col in data.columns:
        if col in categorical:
            data[col] = data[col].astype(str)
        else:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    encoded = pd.get_dummies(data, columns=categorical, drop_first=True, dtype=float)
    encoded = encoded.replace([np.inf, -np.inf], np.nan)
    for col in encoded.columns:
        median = encoded[col].median()
        if not np.isfinite(median):
            median = 0.0
        encoded[col] = encoded[col].fillna(float(median))
    return encoded.astype(float)


def fit_transition_model_hierarchy(
    design: pd.DataFrame,
    specs: list[dict[str, object]],
    *,
    outcome_col: str = "activate_next_bin",
    ridge: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if design.empty:
        raise ValueError("No transition rows to model")
    model_rows = []
    coef_tables = []
    y = pd.to_numeric(design[outcome_col], errors="coerce")
    keep_y = y.notna()
    if y[keep_y].nunique() < 2:
        raise ValueError(f"{outcome_col} has only one class")

    for spec in specs:
        name = str(spec["name"])
        predictors = [str(col) for col in spec.get("predictors", [])]
        categorical = [str(col) for col in spec.get("categorical", [])]
        try:
            x = _encoded_design_matrix(design.loc[keep_y], predictors=predictors, categorical_predictors=categorical)
            fit_data = x.copy()
            fit_data[outcome_col] = y.loc[keep_y].astype(int).to_numpy()
            model, coefs = ia._fit_logistic_model(
                fit_data,
                predictors=x.columns.tolist(),
                outcome_col=outcome_col,
                ridge=float(ridge),
                max_iter=int(max_iter),
                tol=float(tol),
            )
            model["model"] = name
            model["n_encoded_predictors"] = len(x.columns)
            model["predictors"] = ", ".join(predictors)
            model_rows.append(model)
            coefs["model"] = name
            coef_tables.append(coefs)
        except ValueError as exc:
            model_rows.append(
                {
                    "model": name,
                    "n_rows": 0,
                    "n_wake_bins": 0,
                    "n_predictors": len(predictors),
                    "n_encoded_predictors": 0,
                    "log_likelihood": np.nan,
                    "log_loss": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                    "mcfadden_r2": np.nan,
                    "formula_standardized": str(exc),
                    "predictors": ", ".join(predictors),
                }
            )

    model_table = pd.DataFrame(model_rows).sort_values("aic", na_position="last", kind="mergesort").reset_index(drop=True)
    coef_table = pd.concat(coef_tables, ignore_index=True) if coef_tables else pd.DataFrame()
    return model_table, coef_table


def plot_transition_model_comparison(
    model_table: pd.DataFrame,
    *,
    metric: str = "aic",
    title: str = "Quiet-to-active model hierarchy",
) -> pd.DataFrame:
    plot_df = model_table.dropna(subset=[metric]).sort_values(metric, kind="mergesort")
    if plot_df.empty:
        raise ValueError(f"No finite {metric} values to plot")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(plot_df["model"], plot_df[metric], marker="o", lw=1.8, color="0.25")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_current_figures_if_enabled()
    return plot_df


def top_coefficients(
    coef_table: pd.DataFrame,
    *,
    model: str,
    n: int = 20,
) -> pd.DataFrame:
    if coef_table.empty:
        return pd.DataFrame()
    chosen = coef_table[(coef_table["model"] == model) & (coef_table["term"] != "intercept")].copy()
    if chosen.empty:
        return chosen
    chosen["abs_coef_standardized"] = chosen["coef_standardized"].abs()
    return chosen.sort_values("abs_coef_standardized", ascending=False, kind="mergesort").head(int(n)).reset_index(drop=True)
