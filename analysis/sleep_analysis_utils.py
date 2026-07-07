import json
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Iterable, Optional


ADDITIONAL_POSTURE_DISTANCE_PAIRS = [
    (1, 5),
    (1, 6),
    (1, 8),
    (1, 9),
    (4, 6),
    (4, 8),
    (4, 9),
    (5, 7),
    (5, 9),
    (6, 7),
    (6, 8),
    (6, 9),
    (7, 9),
]

BASE_POSTURE_FEATURE_COLUMNS = [
    "angle_in_left_deg",
    "angle_out_left_deg",
    "angle_in_right_deg",
    "angle_out_right_deg",
    "pose_width_mm",
    "pose_height_mm",
    "pose_area_mm2",
]

CORE_POSTURE_DISTANCE_FEATURE_COLUMNS = [
    "bp1_bp4_dist_mm",
    "bp1_bp7_dist_mm",
    "bp4_bp5_dist_mm",
    "bp5_bp6_dist_mm",
    "bp7_bp8_dist_mm",
    "bp8_bp9_dist_mm",
    "bp4_bp7_dist_mm",
    "bp5_bp8_dist_mm",
]

ADDITIONAL_POSTURE_DISTANCE_FEATURE_COLUMNS = [
    f"bp{left}_bp{right}_dist_mm"
    for left, right in ADDITIONAL_POSTURE_DISTANCE_PAIRS
]

POSTURE_NORMALIZED_DISTANCE_FEATURE_COLUMNS = [
    f"{feature}_per_pose_width"
    for feature in [
        *CORE_POSTURE_DISTANCE_FEATURE_COLUMNS,
        *ADDITIONAL_POSTURE_DISTANCE_FEATURE_COLUMNS,
    ]
]

POSTURE_ENGINEERED_FEATURE_COLUMNS = [
    "pose_aspect_ratio",
    "pose_extent_mm",
    "pose_compactness",
    "angle_in_mean_deg",
    "angle_out_mean_deg",
    "angle_all_mean_deg",
    "angle_in_abs_diff_deg",
    "angle_out_abs_diff_deg",
    "angle_left_sum_deg",
    "angle_right_sum_deg",
    "left_side_chain_mm",
    "right_side_chain_mm",
    "side_chain_mean_mm",
    "side_chain_abs_diff_mm",
    "side_chain_ratio",
    "left_segment_ratio",
    "right_segment_ratio",
    "mid_span_ratio",
    "tip_span_mm",
    "tip_span_to_pose_width",
    "head_to_left_tip_ratio",
    "head_to_right_tip_ratio",
]

POSTURE_GROUP_DISPLAY_LABELS = {
    "little_or_no_interaction": "little/no interaction",
    "more_interaction": "more interaction",
    "non_quiescent": "non-quiescent",
}


def side_from_track_name(track_name: str) -> str | None:
    name = str(track_name)
    if name.endswith("_left.parquet") or name.endswith("_left"):
        return "left"
    if name.endswith("_right.parquet") or name.endswith("_right"):
        return "right"
    return None


def sleep_prediction_metadata_paths(sleep_root: Path) -> list[Path]:
    root = Path(sleep_root)
    paths = sorted((root / "per_track").glob("*/sleep_prediction_metadata.json"))
    if not paths:
        paths = sorted(root.glob("*/sleep_prediction_metadata.json"))
    return paths


def load_sleep_prediction_tracks(sleep_root: Path) -> pd.DataFrame:
    rows = []
    for metadata_path in sleep_prediction_metadata_paths(Path(sleep_root)):
        meta = json.loads(metadata_path.read_text())
        track_name = str(meta.get("track_name", metadata_path.parent.name))
        side = meta.get("side") or side_from_track_name(track_name)
        if side is None:
            continue
        n_frames = int(meta.get("n_frames", 0))
        n_predicted = int(meta.get("n_predicted_frames", 0))
        local_dir = metadata_path.parent
        rows.append(
            {
                "track_name": track_name,
                "track_id": meta.get("track_id"),
                "side": side,
                "sleep_metadata_path": metadata_path,
                "sleep_predictions_path": local_dir / "sleep_predictions.parquet",
                "sleep_bouts_path": local_dir / "sleep_bouts.parquet",
                "predicted_sleep_path": local_dir / "predicted_sleep_i1.npy",
                "sleep_probability_path": local_dir / "sleep_probability_f4.npy",
                "wake_probability_path": local_dir / "wake_probability_f4.npy",
                "sleep_frame_min": int(meta.get("frame_min", 0)) if meta.get("frame_min") is not None else 0,
                "sleep_frame_max": int(meta.get("frame_max", -1)) if meta.get("frame_max") is not None else -1,
                "sleep_n_frames": n_frames,
                "n_predicted_sleep_frames": int(meta.get("n_sleep_frames", 0)),
                "n_predicted_wake_frames": int(meta.get("n_wake_frames", 0)),
                "n_predicted_frames": n_predicted,
                "sleep_prediction_present_frac": n_predicted / n_frames if n_frames else np.nan,
                "sleep_fraction_predicted_frames": meta.get("sleep_fraction_predicted_frames"),
                "mean_sleep_probability": meta.get("mean_sleep_probability"),
                "sleep_fps": float(meta.get("fps", 24.0)),
                "sleep_model_path": meta.get("model_path"),
                "sleep_model_feature_set": meta.get("model_feature_set"),
                "sleep_prediction_status": meta.get("status", "ok"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise FileNotFoundError(f"No sleep_prediction_metadata.json files found under {sleep_root}")
    return out.sort_values(["side", "track_id", "track_name"], kind="mergesort").reset_index(drop=True)


def attach_sleep_predictions_to_tracks(
    tracks: pd.DataFrame,
    sleep_root: Path,
    *,
    require_all: bool = False,
) -> pd.DataFrame:
    sleep_tracks = load_sleep_prediction_tracks(sleep_root)
    out = tracks.merge(sleep_tracks, on=["track_name", "track_id", "side"], how="left", validate="one_to_one")
    out["has_sleep_predictions"] = out["predicted_sleep_path"].map(
        lambda path: Path(path).exists() if pd.notna(path) else False
    )
    missing = int((~out["has_sleep_predictions"]).sum())
    if missing and require_all:
        examples = out.loc[~out["has_sleep_predictions"], "track_name"].head(10).to_list()
        raise FileNotFoundError(f"Missing sleep predictions for {missing} tracks, examples: {examples}")
    if missing:
        print(f"WARNING: missing sleep predictions for {missing}/{len(out)} selected tracks")
    return out


def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    ang1 = np.arctan2(v1[:, 1], v1[:, 0])
    ang2 = np.arctan2(v2[:, 1], v2[:, 0])
    ddeg = (np.degrees(ang2 - ang1) + 360.0) % 360.0

    l1 = np.linalg.norm(v1, axis=1)
    l2 = np.linalg.norm(v2, axis=1)
    bad = (l1 == 0) | (l2 == 0) | np.isnan(ang1) | np.isnan(ang2)
    ddeg[bad] = np.nan
    return ddeg


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(changes[0::2], changes[1::2])]


def classify_sleep_wake_from_sleap(
    sleap_df: pd.DataFrame,
    *,
    fps: float,
    track_ids: Optional[Iterable[int]] = None,
    speed_bodypoint: int = 0,
    speed_smooth_window: Optional[int] = None,
    max_speed_pix_s: Optional[float] = None,
    # XY smoothing for speed
    max_interp_gap: int = 5,
    xy_smooth_sigma: float = 3.0,
    speed_nan_mode: str = "interp",
    # angle smoothing (circular)
    angle_max_interp_gap: int = 5,
    angle_smooth_sigma: float = 3.0,
    angle_smooth_window: Optional[int] = None,
    # thresholds
    thr_inL: float = 90.0,
    thr_inR: float = 90.0,
    thr_outL: float = 130.0,
    thr_outR: float = 130.0,
    thr_speed_pix_s: float = 22.17,
    # sleep smoothing (median filter on binary sleep evidence)
    sleep_median_win_sec: float = 20.0,
    # duplicates handling
    duplicate_agg: str = "mean",
    # output row index policy
    output_frame_index: str = "observed",
) -> pd.DataFrame:
    """
    Returns per-frame smoothed angles + smoothed speed (+ smoothed XY used for speed) + sleep/wake classification.

    Speed:
      - Reindex X/Y onto dense frame grid
      - Interpolate short gaps (support smoothing)
      - Smooth X/Y (Gaussian preferred)
      - Compute speed from smoothed X/Y
      - Mask speed by `speed_nan_mode`:
          * "interp": endpoints must exist after limited interpolation
          * "raw": endpoints must exist in raw XY (legacy behavior)
          * "none": no endpoint-validity masking

    Angles:
      - Compute raw angles from pose geometry
      - Circularly smooth angles (smooth sin/cos then atan2), with short-gap interp for support
      - Mask smoothed angles back to NaN where raw angle was missing (no hallucinated angles)

    Sleep classification:
      - Compute framewise sleep evidence from smoothed angles + smoothed speed (NaN-aware)
      - Apply a centered rolling median (majority filter) over sleep evidence (no state machine)
      - Unknown (NaN) stays unknown unless surrounding evidence supports a median value

    Output frame rows:
      - output_frame_index="observed": rows at original observed pose frames only (legacy behavior)
      - output_frame_index="full": rows for every frame in [min_frame, max_frame]
    """

    # ---------- helpers ----------

    def col_or_nan(wide: pd.DataFrame, xy: str, bp: int) -> pd.Series:
        if (xy, bp) in wide.columns:
            return wide[(xy, bp)]
        return pd.Series(np.nan, index=wide.index)

    def get_vec(wide: pd.DataFrame, bp_from: int, bp_to: int) -> np.ndarray:
        dx = col_or_nan(wide, "X", bp_to) - col_or_nan(wide, "X", bp_from)
        dy = col_or_nan(wide, "Y", bp_to) - col_or_nan(wide, "Y", bp_from)
        return np.column_stack([dx.to_numpy(), dy.to_numpy()])

    def smooth_circular_angle_deg(
        angle_deg: pd.Series,
        full_index: pd.RangeIndex,
        *,
        max_interp_gap: int,
        smooth_sigma: float,
        smooth_window: Optional[int],
    ) -> pd.Series:
        """Circular smoothing for degrees in [0, 360). Returns a Series on full_index."""
        a = angle_deg.reindex(full_index).astype(float)
        raw_valid = np.isfinite(a.to_numpy())

        lim = int(max(0, max_interp_gap))
        if lim > 0:
            a_fill = a.interpolate("linear", limit=lim, limit_direction="both")
        else:
            a_fill = a

        rad = np.deg2rad(a_fill.to_numpy(dtype=float))
        s = pd.Series(np.sin(rad), index=full_index)
        c = pd.Series(np.cos(rad), index=full_index)

        if smooth_sigma is not None and float(smooth_sigma) > 0:
            sigma = float(smooth_sigma)
            win = int(max(3, round(6 * sigma + 1)))
            if win % 2 == 0:
                win += 1
            s_sm = s.rolling(win, win_type="gaussian", center=True, min_periods=1).mean(std=sigma)
            c_sm = c.rolling(win, win_type="gaussian", center=True, min_periods=1).mean(std=sigma)
        else:
            w = None if smooth_window is None else int(smooth_window)
            if w is None or w <= 1:
                s_sm, c_sm = s, c
            else:
                s_sm = s.rolling(w, center=True, min_periods=1).mean()
                c_sm = c.rolling(w, center=True, min_periods=1).mean()

        ang = (np.rad2deg(np.arctan2(s_sm.to_numpy(), c_sm.to_numpy())) + 360.0) % 360.0
        out = pd.Series(ang, index=full_index)
        out[~raw_valid] = np.nan  # do not output angles where raw was missing
        return out

    # ---------- main ----------

    df = sleap_df.copy()
    if "TrackID" not in df.columns:
        df["TrackID"] = 0

    required = {"Frame", "TrackID", "Bodypoint", "X", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if track_ids is not None:
        df = df[df["TrackID"].isin(list(track_ids))]

    if duplicate_agg not in {"mean", "median", "first"}:
        raise ValueError("duplicate_agg must be one of: 'mean', 'median', 'first'")
    if speed_nan_mode not in {"interp", "raw", "none"}:
        raise ValueError("speed_nan_mode must be one of: 'interp', 'raw', 'none'")
    if output_frame_index not in {"observed", "full"}:
        raise ValueError("output_frame_index must be one of: 'observed', 'full'")

    rows = []

    for tid, g in df.groupby("TrackID", sort=False):
        g = g.sort_values("Frame", kind="mergesort")

        # --- build wide pose table ---
        if duplicate_agg == "first":
            gg = (
                g.sort_values(["Frame", "Bodypoint"], kind="mergesort")
                .drop_duplicates(subset=["Frame", "Bodypoint"], keep="first")
            )
            wide = gg.pivot(index="Frame", columns="Bodypoint", values=["X", "Y"]).sort_index()
        else:
            wide = (
                g.pivot_table(
                    index="Frame",
                    columns="Bodypoint",
                    values=["X", "Y"],
                    aggfunc=duplicate_agg,
                    dropna=False,
                )
                .sort_index()
            )

        wide.columns = pd.MultiIndex.from_arrays(
            [wide.columns.get_level_values(0), wide.columns.get_level_values(1).astype(int)]
        )

        if wide.index.size == 0:
            continue

        fmin, fmax = int(wide.index.min()), int(wide.index.max())
        full_index = pd.RangeIndex(fmin, fmax + 1)

        # ---------- speed from smoothed XY ----------
        speed_X_raw = col_or_nan(wide, "X", int(speed_bodypoint)).astype(float)
        speed_Y_raw = col_or_nan(wide, "Y", int(speed_bodypoint)).astype(float)

        x = speed_X_raw.reindex(full_index)
        y = speed_Y_raw.reindex(full_index)

        raw_xy_valid = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())

        lim = int(max(0, max_interp_gap))
        if lim > 0:
            x_fill = x.interpolate("linear", limit=lim, limit_direction="both")
            y_fill = y.interpolate("linear", limit=lim, limit_direction="both")
        else:
            x_fill, y_fill = x, y
        interp_xy_valid = np.isfinite(x_fill.to_numpy()) & np.isfinite(y_fill.to_numpy())

        if xy_smooth_sigma is not None and float(xy_smooth_sigma) > 0:
            sigma = float(xy_smooth_sigma)
            win = int(max(3, round(6 * sigma + 1)))
            if win % 2 == 0:
                win += 1
            x_s = x_fill.rolling(win, win_type="gaussian", center=True, min_periods=1).mean(std=sigma)
            y_s = y_fill.rolling(win, win_type="gaussian", center=True, min_periods=1).mean(std=sigma)
        else:
            w = None if speed_smooth_window is None else int(speed_smooth_window)
            if w is None or w <= 1:
                x_s, y_s = x_fill, y_fill
            else:
                x_s = x_fill.rolling(w, center=True, min_periods=1).mean()
                y_s = y_fill.rolling(w, center=True, min_periods=1).mean()

        dx = x_s.diff()
        dy = y_s.diff()
        speed_full = np.sqrt(dx * dx + dy * dy).to_numpy(dtype=float) * float(fps)
        speed_full[0] = np.nan
        if speed_nan_mode == "raw":
            step_valid = raw_xy_valid[1:] & raw_xy_valid[:-1]
        elif speed_nan_mode == "interp":
            step_valid = interp_xy_valid[1:] & interp_xy_valid[:-1]
        else:  # "none"
            step_valid = np.ones(len(speed_full) - 1, dtype=bool)
        speed_full[1:][~step_valid] = np.nan

        speed_pix_s_full = pd.Series(speed_full, index=full_index)

        # ---------- raw angles ----------
        v_5_4 = get_vec(wide, 5, 4)
        v_5_6 = get_vec(wide, 5, 6)
        v_4_1 = get_vec(wide, 4, 1)
        v_4_5 = get_vec(wide, 4, 5)
        v_8_7 = get_vec(wide, 8, 7)
        v_8_9 = get_vec(wide, 8, 9)
        v_7_1 = get_vec(wide, 7, 1)
        v_7_8 = get_vec(wide, 7, 8)

        raw_angles = pd.DataFrame(
            {
                "Frame": wide.index.astype(int),
                "angle_InL_raw": 360.0 - angle_between(v_5_4, v_5_6),
                "angle_OutL_raw": angle_between(v_4_1, v_4_5),
                "angle_InR_raw": angle_between(v_8_7, v_8_9),
                "angle_OutR_raw": 360.0 - angle_between(v_7_1, v_7_8),
            }
        )

        # ---------- smooth angles (output ONLY smoothed) ----------
        smoothed_angles_full = {}
        for raw_col, out_col in [
            ("angle_InL_raw", "angle_InL_deg"),
            ("angle_OutL_raw", "angle_OutL_deg"),
            ("angle_InR_raw", "angle_InR_deg"),
            ("angle_OutR_raw", "angle_OutR_deg"),
        ]:
            s = pd.Series(raw_angles[raw_col].to_numpy(dtype=float),
                          index=raw_angles["Frame"].to_numpy(dtype=int))
            sm = smooth_circular_angle_deg(
                s,
                full_index,
                max_interp_gap=angle_max_interp_gap,
                smooth_sigma=angle_smooth_sigma,
                smooth_window=angle_smooth_window,
            )
            smoothed_angles_full[out_col] = sm

        frame_out = full_index if output_frame_index == "full" else wide.index

        out_track = pd.DataFrame(
            {
                "TrackID": tid,
                "Frame": frame_out.to_numpy(dtype=int),
                "angle_InL_deg": smoothed_angles_full["angle_InL_deg"].reindex(frame_out).to_numpy(dtype=float),
                "angle_OutL_deg": smoothed_angles_full["angle_OutL_deg"].reindex(frame_out).to_numpy(dtype=float),
                "angle_InR_deg": smoothed_angles_full["angle_InR_deg"].reindex(frame_out).to_numpy(dtype=float),
                "angle_OutR_deg": smoothed_angles_full["angle_OutR_deg"].reindex(frame_out).to_numpy(dtype=float),
                "speed_pix_s": speed_pix_s_full.reindex(frame_out).to_numpy(dtype=float),
                "speed_X_s": x_s.reindex(frame_out).to_numpy(dtype=float),
                "speed_Y_s": y_s.reindex(frame_out).to_numpy(dtype=float),
            }
        )

        # ---------- invalidate unrealistic jumps: WIPE speed + ALL angles + XY ----------
        if max_speed_pix_s is not None:
            bad = out_track["speed_pix_s"] > float(max_speed_pix_s)
            wipe = [
                "angle_InL_deg", "angle_OutL_deg",
                "angle_InR_deg", "angle_OutR_deg",
                "speed_pix_s", "speed_X_s", "speed_Y_s",
            ]
            out_track.loc[bad, wipe] = np.nan

        # ---------- sleep evidence (NaN-aware) + median filter ----------
        req = ["angle_InL_deg", "angle_InR_deg", "angle_OutL_deg", "angle_OutR_deg", "speed_pix_s"]
        valid = out_track[req].notna().all(axis=1)

        sleep_raw = pd.Series(np.nan, index=out_track.index, dtype=float)
        sleep_raw.loc[valid] = (
            (out_track.loc[valid, "angle_InL_deg"]  < thr_inL) &
            (out_track.loc[valid, "angle_InR_deg"]  < thr_inR) &
            (out_track.loc[valid, "angle_OutL_deg"] < thr_outL) &
            (out_track.loc[valid, "angle_OutR_deg"] < thr_outR) &
            (out_track.loc[valid, "speed_pix_s"]    < thr_speed_pix_s)
        ).astype(float)  # 1 sleep, 0 wake, NaN unknown

        win = int(round(float(sleep_median_win_sec) * float(fps)))
        win = max(3, win | 1)  # odd, >=3

        sleep_score = sleep_raw.rolling(window=win, center=True, min_periods=1).median()

        out_track["sleep_score"] = sleep_score.to_numpy(dtype=float)
        out_track["is_sleep"] = (sleep_score >= 0.5) & sleep_score.notna()
        out_track["is_wake"] = ~out_track["is_sleep"]

        rows.append(out_track)

    out = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["TrackID", "Frame"], kind="mergesort")
        .reset_index(drop=True)
    )
    return out


def _binned_track_interactions(
    track_counts: Optional[pd.DataFrame],
    track_row: pd.Series,
    bin_seconds: float,
) -> pd.DataFrame:
    fps = float(track_row["fps"])
    frame_min = int(track_row["frame_min"])
    n_frames = int(track_row["n_frames"])
    bin_frames = max(1, int(round(fps * float(bin_seconds))))
    first_bin = frame_min // bin_frames
    n_bins = int(np.ceil((frame_min + n_frames) / bin_frames)) - first_bin

    out = pd.DataFrame(
        {
            "time_h": ((first_bin + np.arange(n_bins)) * bin_frames) / fps / 3600.0,
            "n_interactions_as_antenna": np.zeros(n_bins, dtype=np.int64),
            "n_interactions_as_body": np.zeros(n_bins, dtype=np.int64),
            "n_interactions_total": np.zeros(n_bins, dtype=np.int64),
        }
    )
    if track_counts is None or track_counts.empty or n_bins <= 0:
        return out

    frame_stop = frame_min + n_frames
    frames = track_counts["global_frame"].to_numpy(np.int64)
    in_track = (frames >= frame_min) & (frames < frame_stop)
    if not in_track.any():
        return out

    local_bin = (frames[in_track] // bin_frames) - first_bin
    valid_bin = (local_bin >= 0) & (local_bin < n_bins)
    if not valid_bin.any():
        return out

    local_bin = local_bin[valid_bin]
    for col in ["n_interactions_as_antenna", "n_interactions_as_body", "n_interactions_total"]:
        values = track_counts.loc[in_track, col].to_numpy(np.float64)[valid_bin]
        out[col] = np.rint(
            np.bincount(local_bin, weights=values, minlength=n_bins)[:n_bins]
        ).astype(np.int64)
    return out


def _interaction_counts_for_frame_interval(
    track_counts: Optional[pd.DataFrame],
    *,
    start_frame: int,
    end_frame: int,
) -> tuple[int, int, int]:
    if track_counts is None or track_counts.empty:
        return 0, 0, 0
    frames = track_counts["global_frame"].to_numpy(np.int64)
    left = int(np.searchsorted(frames, int(start_frame), side="left"))
    right = int(np.searchsorted(frames, int(end_frame), side="right"))
    if right <= left:
        return 0, 0, 0
    selected = track_counts.iloc[left:right]
    n_antenna = int(selected["n_interactions_as_antenna"].sum())
    n_body = int(selected["n_interactions_as_body"].sum())
    return n_antenna, n_body, int(n_antenna + n_body)


def _speed_threshold_runs_for_track_row(
    track_row: pd.Series,
    *,
    speed_smooth_seconds: float,
    speed_threshold_mm_s: float,
    min_duration_seconds: float,
    state: str,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
) -> pd.DataFrame:
    from analysis import colony_speed_utils as cs

    speed = np.load(track_row["speed_path"], mmap_mode="r")
    fps = float(track_row["fps"])
    frame_min = int(track_row["frame_min"])
    smooth_frames = max(1, int(round(float(speed_smooth_seconds) * fps)))
    smoothed = cs.rolling_nanmean(np.asarray(speed, dtype=np.float32), smooth_frames)
    finite = np.isfinite(smoothed)
    if state == "below":
        mask = finite & (smoothed <= float(speed_threshold_mm_s))
    elif state == "above":
        mask = finite & (smoothed > float(speed_threshold_mm_s))
    else:
        raise ValueError("state must be 'below' or 'above'")

    if frame_start is not None or frame_stop is not None:
        keep = np.zeros(len(mask), dtype=bool)
        local_start = 0 if frame_start is None else max(0, int(frame_start) - frame_min)
        local_stop = len(mask) if frame_stop is None else min(len(mask), int(frame_stop) - frame_min)
        if local_stop > local_start:
            keep[local_start:local_stop] = True
        mask &= keep

    min_frames = max(1, int(round(float(min_duration_seconds) * fps)))
    rows = []
    for run_start, run_stop in contiguous_true_runs(mask):
        duration_frames = int(run_stop - run_start)
        if duration_frames < min_frames:
            continue
        start_frame = int(frame_min + run_start)
        end_frame = int(frame_min + run_stop - 1)
        rows.append(
            {
                "bout_start_frame": start_frame,
                "bout_end_frame": end_frame,
                "bout_duration_frames": duration_frames,
                "bout_duration_seconds": duration_frames / fps,
                "time_h": start_frame / fps / 3600.0,
                "bout_end_time_h": (end_frame + 1) / fps / 3600.0,
                "mean_smoothed_speed_mm_s": float(np.nanmean(smoothed[run_start:run_stop])),
                "min_smoothed_speed_mm_s": float(np.nanmin(smoothed[run_start:run_stop])),
                "max_smoothed_speed_mm_s": float(np.nanmax(smoothed[run_start:run_stop])),
                "speed_threshold_mm_s": float(speed_threshold_mm_s),
                "speed_smooth_seconds": float(speed_smooth_seconds),
                "min_bout_seconds": float(min_duration_seconds),
            }
        )
    return pd.DataFrame(rows)


def quiescent_threshold_bouts_for_track_row(
    track_row: pd.Series,
    counts_by_track: Optional[dict[int, pd.DataFrame]] = None,
    *,
    speed_smooth_seconds: float = 5 * 60.0,
    quiescence_speed_threshold_mm_s: float = 0.1,
    min_quiescent_seconds: float = 10.0,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
) -> pd.DataFrame:
    bouts = _speed_threshold_runs_for_track_row(
        track_row,
        speed_smooth_seconds=speed_smooth_seconds,
        speed_threshold_mm_s=quiescence_speed_threshold_mm_s,
        min_duration_seconds=min_quiescent_seconds,
        state="below",
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    if bouts.empty:
        return bouts
    track_counts = None
    if counts_by_track is not None and pd.notna(track_row.get("track_id")):
        track_counts = counts_by_track.get(int(track_row["track_id"]))
    rows = []
    for bout_id, bout in enumerate(bouts.itertuples(index=False)):
        n_antenna, n_body, n_total = _interaction_counts_for_frame_interval(
            track_counts,
            start_frame=int(bout.bout_start_frame),
            end_frame=int(bout.bout_end_frame),
        )
        row = bout._asdict()
        row.update(
            {
                "bout_id": int(bout_id),
                "is_quiescent": True,
                "n_interactions_as_antenna": n_antenna,
                "n_interactions_as_body": n_body,
                "n_interactions_total": n_total,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def non_quiescent_threshold_bouts_for_track_row(
    track_row: pd.Series,
    counts_by_track: Optional[dict[int, pd.DataFrame]] = None,
    *,
    speed_smooth_seconds: float = 5 * 60.0,
    quiescence_speed_threshold_mm_s: float = 0.1,
    min_non_quiescent_seconds: float = 10.0,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
) -> pd.DataFrame:
    bouts = _speed_threshold_runs_for_track_row(
        track_row,
        speed_smooth_seconds=speed_smooth_seconds,
        speed_threshold_mm_s=quiescence_speed_threshold_mm_s,
        min_duration_seconds=min_non_quiescent_seconds,
        state="above",
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    if bouts.empty:
        return bouts
    track_counts = None
    if counts_by_track is not None and pd.notna(track_row.get("track_id")):
        track_counts = counts_by_track.get(int(track_row["track_id"]))
    rows = []
    for bout_id, bout in enumerate(bouts.itertuples(index=False)):
        n_antenna, n_body, n_total = _interaction_counts_for_frame_interval(
            track_counts,
            start_frame=int(bout.bout_start_frame),
            end_frame=int(bout.bout_end_frame),
        )
        row = bout._asdict()
        row.update(
            {
                "bout_id": int(bout_id),
                "is_quiescent": False,
                "n_interactions_as_antenna": n_antenna,
                "n_interactions_as_body": n_body,
                "n_interactions_total": n_total,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _speed_interaction_timeseries_for_track_row(
    track_row: pd.Series,
    counts_by_track: dict[int, pd.DataFrame],
    *,
    bin_seconds: float,
    speed_smooth_seconds: float,
) -> pd.DataFrame:
    from analysis import colony_speed_utils as cs

    if pd.isna(track_row["track_id"]):
        raise ValueError(f"Selected track has no track_id: {track_row.get('track_name', '<unknown>')}")

    speed_df = cs.binned_track_speed(track_row, bin_seconds)
    window_bins = max(1, int(round(float(speed_smooth_seconds) / float(bin_seconds))))
    speed_df["smoothed_speed_mm_s"] = cs.rolling_nanmean(speed_df["speed_mm_s"].to_numpy(), window_bins)

    interaction_df = _binned_track_interactions(
        counts_by_track.get(int(track_row["track_id"])),
        track_row,
        bin_seconds,
    )

    out = pd.merge(speed_df, interaction_df, on="time_h", how="outer").sort_values(
        "time_h",
        kind="mergesort",
    )
    for col in ["n_interactions_as_antenna", "n_interactions_as_body", "n_interactions_total"]:
        out[col] = out[col].fillna(0).astype(np.int64)

    out["bin_seconds"] = float(bin_seconds)
    out["speed_smooth_seconds"] = float(speed_smooth_seconds)
    return out.reset_index(drop=True)


def single_ant_speed_interaction_timeseries(
    speed_tracks: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    row_number: Optional[int] = None,
    track_id: Optional[int] = None,
    side: Optional[str] = "left",
    bin_seconds: float = 60.0,
    speed_smooth_seconds: float = 5 * 60.0,
    chunk_global_frame_offset: int = 0,
    counts_by_track: Optional[dict[int, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Build a single-ant speed/interactions table from existing speed vectors.

    `speed_tracks` should come from `analysis.colony_speed_utils.load_speed_tracks`
    or the equivalent metadata table. `interactions` can be one chunk's raw
    interaction parquet rows or the multi-chunk table returned by
    `analysis.interaction_analysis_utils.load_interactions_for_chunks`.
    """
    from analysis import colony_speed_utils as cs
    from analysis import interaction_analysis_utils as ia

    row_numbers = [int(row_number)] if row_number is not None else None
    track_ids = [track_id] if track_id is not None else None
    chosen = cs.choose_tracks(
        speed_tracks,
        side=side,
        row_numbers=row_numbers,
        track_ids=track_ids,
        max_tracks=1,
        sort_tracks=False,
    )
    track_row = chosen.iloc[0]

    if counts_by_track is None:
        counts_by_track = ia.interaction_bout_counts_by_track(
            interactions,
            chunk_global_frame_offset=int(chunk_global_frame_offset),
            fps=float(track_row.get("fps", 24.0)),
        )
    return _speed_interaction_timeseries_for_track_row(
        track_row,
        counts_by_track,
        bin_seconds=bin_seconds,
        speed_smooth_seconds=speed_smooth_seconds,
    )


def plot_single_ant_speed_interactions(
    speed_tracks: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    row_number: Optional[int] = None,
    track_id: Optional[int] = None,
    side: Optional[str] = "left",
    bin_seconds: float = 60.0,
    speed_smooth_seconds: float = 5 * 60.0,
    chunk_global_frame_offset: int = 0,
    quiescence_speed_threshold_mm_s: Optional[float] = 0.1,
    min_quiescent_seconds: float = 10.0,
    analysis_frame_start: Optional[int] = None,
    analysis_frame_stop: Optional[int] = None,
    speed_ylim: Optional[tuple[float, float]] = None,
    interaction_ylim: Optional[tuple[float, float]] = None,
    counts_by_track: Optional[dict[int, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Plot one ant's 5-minute-smoothed speed with its interaction-bout counts.

    Interaction roles are directed bout starts: antenna/source means the
    selected ant's antenna contacted another ant; body/receiver means another
    ant's antenna contacted the selected ant.
    """
    import matplotlib.pyplot as plt

    plot_df = single_ant_speed_interaction_timeseries(
        speed_tracks,
        interactions,
        row_number=row_number,
        track_id=track_id,
        side=side,
        bin_seconds=bin_seconds,
        speed_smooth_seconds=speed_smooth_seconds,
        chunk_global_frame_offset=chunk_global_frame_offset,
        counts_by_track=counts_by_track,
    )
    track_row = selected_speed_track(
        speed_tracks,
        row_number=row_number,
        track_id=track_id,
        side=side,
    )

    selected = plot_df[["side", "track_id", "track_name", "track_row"]].dropna(how="all").head(1)
    if selected.empty:
        label = "selected ant"
    else:
        row = selected.iloc[0]
        label = f"{row['side']} track {row['track_id']} row {row['track_row']}"

    bin_width_h = float(bin_seconds) / 3600.0
    time_h = plot_df["time_h"].to_numpy(float)
    smoothed_speed = plot_df["smoothed_speed_mm_s"].to_numpy(float)
    plot_df["is_quiescent"] = False
    quiescent_bouts = pd.DataFrame()
    if quiescence_speed_threshold_mm_s is not None:
        quiescent_bouts = quiescent_threshold_bouts_for_track_row(
            track_row,
            counts_by_track,
            speed_smooth_seconds=speed_smooth_seconds,
            quiescence_speed_threshold_mm_s=float(quiescence_speed_threshold_mm_s),
            min_quiescent_seconds=float(min_quiescent_seconds),
            frame_start=analysis_frame_start,
            frame_stop=analysis_frame_stop,
        )
        if not quiescent_bouts.empty:
            fps = float(track_row["fps"])
            bin_frames = max(1, int(round(fps * float(bin_seconds))))
            bin_start = np.rint(time_h * 3600.0 * fps).astype(np.int64)
            bin_end = bin_start + bin_frames - 1
            in_bout = np.zeros(len(plot_df), dtype=bool)
            for bout in quiescent_bouts.itertuples(index=False):
                in_bout |= (bin_end >= int(bout.bout_start_frame)) & (bin_start <= int(bout.bout_end_frame))
            plot_df["is_quiescent"] = in_bout
        plot_df.attrs["quiescent_bouts"] = quiescent_bouts

    fig, (speed_ax, interaction_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.35]},
    )

    speed_ax.plot(
        plot_df["time_h"],
        plot_df["speed_mm_s"],
        lw=0.8,
        alpha=0.25,
        color="0.35",
        label=f"{bin_seconds:g} s binned speed",
    )
    speed_ax.plot(
        plot_df["time_h"],
        plot_df["smoothed_speed_mm_s"],
        lw=1.5,
        color="tab:blue",
        label=f"{speed_smooth_seconds / 60:g} min smoothed speed",
    )

    if quiescence_speed_threshold_mm_s is not None:
        threshold = float(quiescence_speed_threshold_mm_s)
        speed_ax.axhline(
            threshold,
            color="0.25",
            lw=1.0,
            ls="--",
            alpha=0.7,
            label="quiescence threshold",
        )
        for bout in quiescent_bouts.itertuples(index=False):
            start_h = float(bout.bout_start_frame) / float(track_row["fps"]) / 3600.0
            stop_h = float(bout.bout_end_frame + 1) / float(track_row["fps"]) / 3600.0
            speed_ax.axvspan(start_h, stop_h, color="0.7", alpha=0.18, lw=0)
            interaction_ax.axvspan(start_h, stop_h, color="0.7", alpha=0.18, lw=0)

    speed_ax.set_ylabel("Speed (mm/s)")
    speed_ax.set_title(f"Speed and directed interactions: {label}")
    if speed_ylim is not None:
        speed_ax.set_ylim(*speed_ylim)
    speed_ax.legend(fontsize=8, loc="best")
    speed_ax.grid(True, alpha=0.25)

    antenna = plot_df["n_interactions_as_antenna"].to_numpy(np.float64)
    body = plot_df["n_interactions_as_body"].to_numpy(np.float64)
    interaction_ax.bar(
        time_h,
        antenna,
        width=bin_width_h * 0.92,
        align="edge",
        color="tab:orange",
        alpha=0.85,
        label="as antenna/source",
    )
    interaction_ax.bar(
        time_h,
        body,
        bottom=antenna,
        width=bin_width_h * 0.92,
        align="edge",
        color="tab:purple",
        alpha=0.80,
        label="as body/receiver",
    )
    interaction_ax.set_ylabel(f"Interaction bouts / {bin_seconds:g} s")
    interaction_ax.set_xlabel("Elapsed time (h)")
    if interaction_ylim is not None:
        interaction_ax.set_ylim(*interaction_ylim)
    interaction_ax.legend(fontsize=8, loc="best")
    interaction_ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    plt.show()
    return plot_df


def selected_speed_track(
    speed_tracks: pd.DataFrame,
    *,
    row_number: Optional[int] = None,
    track_id: Optional[int] = None,
    side: Optional[str] = "left",
) -> pd.Series:
    from analysis import colony_speed_utils as cs

    row_numbers = [int(row_number)] if row_number is not None else None
    track_ids = [track_id] if track_id is not None else None
    chosen = cs.choose_tracks(
        speed_tracks,
        side=side,
        row_numbers=row_numbers,
        track_ids=track_ids,
        max_tracks=1,
        sort_tracks=False,
    )
    return chosen.iloc[0]


def binned_track_sleep_predictions(row: pd.Series, bin_seconds: float) -> pd.DataFrame:
    states = np.load(row["predicted_sleep_path"], mmap_mode="r")
    fps = float(row.get("sleep_fps", row.get("fps", 24.0)))
    frame_min = int(row.get("sleep_frame_min", row.get("frame_min", 0)))

    bin_frames = max(1, int(round(fps * float(bin_seconds))))
    first_bin = frame_min // bin_frames
    n_bins = int(np.ceil((frame_min + len(states)) / bin_frames)) - first_bin
    sleep_fraction = np.full(n_bins, np.nan, dtype=np.float32)
    wake_fraction = np.full(n_bins, np.nan, dtype=np.float32)
    n_predicted = np.zeros(n_bins, dtype=np.int64)
    n_sleep = np.zeros(n_bins, dtype=np.int64)
    n_wake = np.zeros(n_bins, dtype=np.int64)

    valid = states >= 0
    if valid.any():
        valid_idx = np.flatnonzero(valid)
        local_bin_idx = ((frame_min + valid_idx) // bin_frames) - first_bin
        n_predicted = np.bincount(local_bin_idx, minlength=n_bins)[:n_bins].astype(np.int64)
        n_sleep = np.bincount(
            local_bin_idx,
            weights=(states[valid_idx] == 1).astype(np.int64),
            minlength=n_bins,
        )[:n_bins].astype(np.int64)
        n_wake = np.bincount(
            local_bin_idx,
            weights=(states[valid_idx] == 0).astype(np.int64),
            minlength=n_bins,
        )[:n_bins].astype(np.int64)
        keep = n_predicted > 0
        sleep_fraction[keep] = n_sleep[keep] / n_predicted[keep]
        wake_fraction[keep] = n_wake[keep] / n_predicted[keep]

    out = pd.DataFrame(
        {
            "time_h": ((first_bin + np.arange(n_bins)) * bin_frames) / fps / 3600.0,
            "sleep_fraction": sleep_fraction,
            "wake_fraction": wake_fraction,
            "n_sleep_frames": n_sleep,
            "n_wake_frames": n_wake,
            "n_predicted_sleep_state_frames": n_predicted,
            "side": row["side"],
            "track_id": row["track_id"],
            "track_name": row["track_name"],
            "track_row": row.name,
        }
    )
    return out


def _sleep_prediction_row_for_track(
    sleep_tracks: pd.DataFrame,
    *,
    row_number: Optional[int] = None,
    track_id: Optional[int] = None,
    side: Optional[str] = "left",
) -> pd.Series:
    from analysis import colony_speed_utils as cs

    row_numbers = [int(row_number)] if row_number is not None else None
    track_ids = [track_id] if track_id is not None else None
    chosen = cs.choose_tracks(
        sleep_tracks,
        side=side,
        row_numbers=row_numbers,
        track_ids=track_ids,
        max_tracks=1,
        sort_tracks=False,
    )
    chosen = chosen[chosen.get("has_sleep_predictions", True).astype(bool)] if "has_sleep_predictions" in chosen else chosen
    if chosen.empty:
        raise ValueError("Selected track has no sleep prediction outputs")
    return chosen.iloc[0]


def predicted_sleep_bouts_for_track_row(
    track_row: pd.Series,
    counts_by_track: Optional[dict[int, pd.DataFrame]] = None,
    *,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
) -> pd.DataFrame:
    path = Path(track_row["sleep_bouts_path"])
    if not path.exists():
        raise FileNotFoundError(f"Missing sleep_bouts.parquet for {track_row.get('track_name')}: {path}")
    bouts = pd.read_parquet(path)
    if bouts.empty:
        return pd.DataFrame()

    label_value = pd.to_numeric(bouts.get("predicted_label_value"), errors="coerce")
    if "predicted_label" in bouts.columns:
        label_text = bouts["predicted_label"].astype(str).str.lower()
        sleep_mask = (label_value == 1) | (label_text == "sleep")
    else:
        sleep_mask = label_value == 1
    bouts = bouts[sleep_mask].copy()
    if bouts.empty:
        return pd.DataFrame()

    bouts["bout_start_frame"] = pd.to_numeric(bouts["frame_start"], errors="coerce")
    bouts["bout_end_frame"] = pd.to_numeric(bouts["frame_end"], errors="coerce")
    bouts = bouts.dropna(subset=["bout_start_frame", "bout_end_frame"]).copy()
    bouts["bout_start_frame"] = np.rint(bouts["bout_start_frame"]).astype(np.int64)
    bouts["bout_end_frame"] = np.rint(bouts["bout_end_frame"]).astype(np.int64)

    if frame_start is not None:
        bouts = bouts[bouts["bout_end_frame"] >= int(frame_start)].copy()
        bouts["bout_start_frame"] = np.maximum(bouts["bout_start_frame"].to_numpy(np.int64), int(frame_start))
    if frame_stop is not None:
        bouts = bouts[bouts["bout_start_frame"] < int(frame_stop)].copy()
        bouts["bout_end_frame"] = np.minimum(bouts["bout_end_frame"].to_numpy(np.int64), int(frame_stop) - 1)
    bouts = bouts[bouts["bout_end_frame"] >= bouts["bout_start_frame"]].copy()
    if bouts.empty:
        return bouts

    fps = float(track_row.get("sleep_fps", track_row.get("fps", 24.0)))
    bouts["bout_duration_frames"] = bouts["bout_end_frame"] - bouts["bout_start_frame"] + 1
    bouts["bout_duration_seconds"] = bouts["bout_duration_frames"] / fps
    bouts["time_h"] = bouts["bout_start_frame"] / fps / 3600.0
    bouts["bout_start_time_h"] = bouts["time_h"]
    bouts["bout_end_time_h"] = (bouts["bout_end_frame"] + 1) / fps / 3600.0
    bouts["track_name"] = track_row.get("track_name")
    bouts["track_id"] = track_row.get("track_id")
    bouts["side"] = track_row.get("side")
    bouts["track_row"] = track_row.name
    bouts["sleep_source"] = "sleep_prediction"
    bouts["classifier_bin"] = (
        bouts["track_name"].astype(str)
        + ":"
        + bouts["bout_start_frame"].astype(str)
        + "-"
        + bouts["bout_end_frame"].astype(str)
    )

    track_counts = None
    if counts_by_track is not None and pd.notna(track_row.get("track_id")):
        track_counts = counts_by_track.get(int(track_row["track_id"]))
    rows = []
    for bout_id, bout in enumerate(bouts.itertuples(index=False)):
        n_antenna, n_body, n_total = _interaction_counts_for_frame_interval(
            track_counts,
            start_frame=int(bout.bout_start_frame),
            end_frame=int(bout.bout_end_frame),
        )
        row = bout._asdict()
        row.update(
            {
                "bout_id": int(bout_id),
                "n_interactions_as_antenna": n_antenna,
                "n_interactions_as_body": n_body,
                "n_interactions_total": n_total,
                "n_interaction_onsets": n_total,
                "n_interaction_onsets_as_antenna": n_antenna,
                "n_interaction_onsets_as_body": n_body,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bout_start_frame", kind="mergesort").reset_index(drop=True)


def load_predicted_sleep_bouts(
    sleep_tracks: pd.DataFrame,
    counts_by_track: Optional[dict[int, pd.DataFrame]] = None,
    *,
    frame_start: Optional[int] = None,
    frame_stop: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    selected = sleep_tracks.copy()
    if "has_sleep_predictions" in selected.columns:
        selected = selected[selected["has_sleep_predictions"].astype(bool)].copy()
    for _, row in selected.iterrows():
        bouts = predicted_sleep_bouts_for_track_row(
            row,
            counts_by_track,
            frame_start=frame_start,
            frame_stop=frame_stop,
        )
        if not bouts.empty:
            rows.append(bouts)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["side", "track_id", "bout_start_frame"],
        kind="mergesort",
    ).reset_index(drop=True)


def single_ant_sleep_prediction_timeseries(
    sleep_tracks: pd.DataFrame,
    speed_tracks: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    row_number: Optional[int] = None,
    track_id: Optional[int] = None,
    side: Optional[str] = "left",
    bin_seconds: float = 60.0,
    speed_smooth_seconds: float = 5 * 60.0,
    chunk_global_frame_offset: int = 0,
    counts_by_track: Optional[dict[int, pd.DataFrame]] = None,
) -> pd.DataFrame:
    from analysis import interaction_analysis_utils as ia

    sleep_row = _sleep_prediction_row_for_track(
        sleep_tracks,
        row_number=row_number,
        track_id=track_id,
        side=side,
    )
    speed_row = selected_speed_track(
        speed_tracks,
        track_id=int(sleep_row["track_id"]) if pd.notna(sleep_row.get("track_id")) else track_id,
        side=sleep_row.get("side", side),
    )
    if counts_by_track is None:
        counts_by_track = ia.interaction_bout_counts_by_track(
            interactions,
            chunk_global_frame_offset=int(chunk_global_frame_offset),
            fps=float(speed_row.get("fps", 24.0)),
        )
    speed_df = _speed_interaction_timeseries_for_track_row(
        speed_row,
        counts_by_track,
        bin_seconds=bin_seconds,
        speed_smooth_seconds=speed_smooth_seconds,
    )
    sleep_df = binned_track_sleep_predictions(sleep_row, bin_seconds)
    merge_cols = ["time_h", "side", "track_id", "track_name", "track_row"]
    out = speed_df.merge(
        sleep_df.drop(columns=[col for col in ["side", "track_id", "track_name", "track_row"] if col in sleep_df]),
        on="time_h",
        how="outer",
    ).sort_values("time_h", kind="mergesort")
    out["is_sleep"] = pd.to_numeric(out["sleep_fraction"], errors="coerce") >= 0.5
    bouts = predicted_sleep_bouts_for_track_row(
        sleep_row,
        counts_by_track,
    )
    out.attrs["sleep_bouts"] = bouts
    return out.reset_index(drop=True)


def plot_single_ant_sleep_predictions_interactions(
    sleep_tracks: pd.DataFrame,
    speed_tracks: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    row_number: Optional[int] = None,
    track_id: Optional[int] = None,
    side: Optional[str] = "left",
    bin_seconds: float = 60.0,
    speed_smooth_seconds: float = 5 * 60.0,
    chunk_global_frame_offset: int = 0,
    analysis_frame_start: Optional[int] = None,
    analysis_frame_stop: Optional[int] = None,
    speed_ylim: Optional[tuple[float, float]] = None,
    interaction_ylim: Optional[tuple[float, float]] = None,
    counts_by_track: Optional[dict[int, pd.DataFrame]] = None,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    plot_df = single_ant_sleep_prediction_timeseries(
        sleep_tracks,
        speed_tracks,
        interactions,
        row_number=row_number,
        track_id=track_id,
        side=side,
        bin_seconds=bin_seconds,
        speed_smooth_seconds=speed_smooth_seconds,
        chunk_global_frame_offset=chunk_global_frame_offset,
        counts_by_track=counts_by_track,
    )
    sleep_row = _sleep_prediction_row_for_track(
        sleep_tracks,
        row_number=row_number,
        track_id=track_id,
        side=side,
    )
    sleep_bouts = predicted_sleep_bouts_for_track_row(
        sleep_row,
        counts_by_track,
        frame_start=analysis_frame_start,
        frame_stop=analysis_frame_stop,
    )
    plot_df.attrs["sleep_bouts"] = sleep_bouts

    label = f"{sleep_row['side']} track {sleep_row['track_id']}"
    bin_width_h = float(bin_seconds) / 3600.0
    time_h = plot_df["time_h"].to_numpy(float)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.4, 1.2]},
    )
    speed_ax, sleep_ax, interaction_ax = axes
    speed_ax.plot(
        plot_df["time_h"],
        plot_df["speed_mm_s"],
        lw=0.8,
        alpha=0.25,
        color="0.35",
        label=f"{bin_seconds:g} s binned speed",
    )
    speed_ax.plot(
        plot_df["time_h"],
        plot_df["smoothed_speed_mm_s"],
        lw=1.5,
        color="tab:blue",
        label=f"{speed_smooth_seconds / 60:g} min smoothed speed",
    )
    for bout in sleep_bouts.itertuples(index=False):
        start_h = float(bout.bout_start_frame) / float(sleep_row["sleep_fps"]) / 3600.0
        stop_h = float(bout.bout_end_frame + 1) / float(sleep_row["sleep_fps"]) / 3600.0
        speed_ax.axvspan(start_h, stop_h, color="tab:blue", alpha=0.10, lw=0)
        sleep_ax.axvspan(start_h, stop_h, color="tab:blue", alpha=0.10, lw=0)
        interaction_ax.axvspan(start_h, stop_h, color="tab:blue", alpha=0.10, lw=0)

    speed_ax.set_ylabel("Speed (mm/s)")
    speed_ax.set_title(f"Classifier sleep predictions and directed interactions: {label}")
    if speed_ylim is not None:
        speed_ax.set_ylim(*speed_ylim)
    speed_ax.legend(fontsize=8, loc="best")
    speed_ax.grid(True, alpha=0.25)

    sleep_ax.plot(
        plot_df["time_h"],
        plot_df["sleep_fraction"],
        lw=1.3,
        color="tab:blue",
        label="predicted sleep fraction",
    )
    sleep_ax.set_ylim(-0.05, 1.05)
    sleep_ax.set_ylabel("Sleep fraction")
    sleep_ax.grid(True, alpha=0.25)
    sleep_ax.legend(fontsize=8, loc="best")

    antenna = plot_df["n_interactions_as_antenna"].fillna(0).to_numpy(np.float64)
    body = plot_df["n_interactions_as_body"].fillna(0).to_numpy(np.float64)
    interaction_ax.bar(
        time_h,
        antenna,
        width=bin_width_h * 0.92,
        align="edge",
        color="tab:orange",
        alpha=0.85,
        label="as antenna/source",
    )
    interaction_ax.bar(
        time_h,
        body,
        bottom=antenna,
        width=bin_width_h * 0.92,
        align="edge",
        color="tab:purple",
        alpha=0.80,
        label="as body/receiver",
    )
    interaction_ax.set_ylabel(f"Interaction bouts / {bin_seconds:g} s")
    interaction_ax.set_xlabel("Elapsed time (h)")
    if interaction_ylim is not None:
        interaction_ax.set_ylim(*interaction_ylim)
    interaction_ax.legend(fontsize=8, loc="best")
    interaction_ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    plt.show()
    return plot_df


def resolve_track_path_for_speed_row(
    track_row: pd.Series,
    *,
    per_track_root: Optional[Path] = None,
) -> Path:
    candidates: list[Path] = []
    track_name = str(track_row.get("track_name", ""))

    if per_track_root is not None and track_name:
        candidates.append(Path(per_track_root) / track_name)

    metadata_value = track_row.get("metadata_path", track_row.get("speed_metadata_path", None))
    if pd.notna(metadata_value):
        metadata_path = Path(metadata_value)
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError:
                metadata = {}
            track_path_text = metadata.get("track_path")
            if track_path_text:
                candidates.append(Path(str(track_path_text)))
                if per_track_root is not None:
                    candidates.append(Path(per_track_root) / Path(str(track_path_text)).name)

    if track_name:
        candidates.append(Path(track_name))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve selected ant track parquet. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def quiescent_interaction_bin_groups(
    speed_interactions: pd.DataFrame,
    *,
    low_interaction_max: int = 0,
    high_interaction_min: Optional[int] = None,
    high_interaction_quantile: float = 0.75,
    include_non_quiescent: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if "is_quiescent" not in speed_interactions.columns:
        raise ValueError("speed_interactions must include is_quiescent; use plot_single_ant_speed_interactions first")
    required = {"time_h", "n_interactions_total"}
    missing = required.difference(speed_interactions.columns)
    if missing:
        raise ValueError(f"speed_interactions is missing columns: {sorted(missing)}")

    is_quiescent = speed_interactions["is_quiescent"].astype(bool)
    quiescent = speed_interactions[is_quiescent].copy()
    if quiescent.empty:
        raise ValueError("No quiescent bins available with the current speed threshold")

    counts = pd.to_numeric(quiescent["n_interactions_total"], errors="coerce").fillna(0).astype(int)
    if high_interaction_min is None:
        high_interaction_min = int(np.ceil(counts.quantile(float(high_interaction_quantile))))
        high_interaction_min = max(high_interaction_min, int(low_interaction_max) + 1)

    group = np.full(len(quiescent), "middle", dtype=object)
    group[counts.to_numpy() <= int(low_interaction_max)] = "little_or_no_interaction"
    group[counts.to_numpy() >= int(high_interaction_min)] = "more_interaction"
    quiescent["posture_interaction_group"] = group

    grouped = quiescent
    if include_non_quiescent:
        non_quiescent = speed_interactions[~is_quiescent].copy()
        non_quiescent["posture_interaction_group"] = "non_quiescent"
        grouped = pd.concat([quiescent, non_quiescent], ignore_index=True)

    thresholds = {
        "low_interaction_max": float(low_interaction_max),
        "high_interaction_min": float(high_interaction_min),
        "high_interaction_quantile": float(high_interaction_quantile),
    }
    return grouped.reset_index(drop=True), thresholds


def quiescent_interaction_bout_groups(
    track_row: pd.Series,
    counts_by_track: dict[int, pd.DataFrame],
    *,
    speed_smooth_seconds: float = 5 * 60.0,
    quiescence_speed_threshold_mm_s: float = 0.1,
    min_quiescent_seconds: float = 10.0,
    low_interaction_max: int = 0,
    high_interaction_min: Optional[int] = None,
    high_interaction_quantile: float = 0.75,
    include_non_quiescent: bool = False,
    min_non_quiescent_seconds: Optional[float] = None,
    analysis_frame_start: Optional[int] = None,
    analysis_frame_stop: Optional[int] = None,
    interaction_count_col: str = "n_interactions_total",
) -> tuple[pd.DataFrame, dict[str, float]]:
    quiescent = quiescent_threshold_bouts_for_track_row(
        track_row,
        counts_by_track,
        speed_smooth_seconds=speed_smooth_seconds,
        quiescence_speed_threshold_mm_s=quiescence_speed_threshold_mm_s,
        min_quiescent_seconds=min_quiescent_seconds,
        frame_start=analysis_frame_start,
        frame_stop=analysis_frame_stop,
    )
    if quiescent.empty:
        raise ValueError("No quiescent bouts available with the current speed threshold/minimum duration")

    interaction_count_col = str(interaction_count_col)
    if interaction_count_col not in quiescent.columns:
        raise ValueError(f"quiescent bouts are missing interaction_count_col={interaction_count_col!r}")
    counts = pd.to_numeric(quiescent[interaction_count_col], errors="coerce").fillna(0).astype(int)
    if high_interaction_min is None:
        high_interaction_min = int(np.ceil(counts.quantile(float(high_interaction_quantile))))
        high_interaction_min = max(high_interaction_min, int(low_interaction_max) + 1)

    group = np.full(len(quiescent), "middle", dtype=object)
    group[counts.to_numpy() <= int(low_interaction_max)] = "little_or_no_interaction"
    group[counts.to_numpy() >= int(high_interaction_min)] = "more_interaction"
    quiescent = quiescent.copy()
    quiescent["interaction_group_count_col"] = interaction_count_col
    quiescent["interaction_group_count"] = counts.to_numpy(np.int64)
    quiescent["posture_interaction_group"] = group
    quiescent["bout_type"] = "quiescent"

    grouped = quiescent
    if include_non_quiescent:
        non_quiescent = non_quiescent_threshold_bouts_for_track_row(
            track_row,
            counts_by_track,
            speed_smooth_seconds=speed_smooth_seconds,
            quiescence_speed_threshold_mm_s=quiescence_speed_threshold_mm_s,
            min_non_quiescent_seconds=float(min_non_quiescent_seconds or min_quiescent_seconds),
            frame_start=analysis_frame_start,
            frame_stop=analysis_frame_stop,
        )
        if not non_quiescent.empty:
            non_quiescent = non_quiescent.copy()
            non_quiescent["posture_interaction_group"] = "non_quiescent"
            non_quiescent["bout_type"] = "non_quiescent"
            grouped = pd.concat([quiescent, non_quiescent], ignore_index=True)

    thresholds = {
        "low_interaction_max": float(low_interaction_max),
        "high_interaction_min": float(high_interaction_min),
        "high_interaction_quantile": float(high_interaction_quantile),
        "quiescence_speed_threshold_mm_s": float(quiescence_speed_threshold_mm_s),
        "min_quiescent_seconds": float(min_quiescent_seconds),
        "speed_smooth_seconds": float(speed_smooth_seconds),
        "interaction_count_col": interaction_count_col,
    }
    if analysis_frame_start is not None:
        thresholds["analysis_frame_start"] = float(analysis_frame_start)
    if analysis_frame_stop is not None:
        thresholds["analysis_frame_stop"] = float(analysis_frame_stop)
    return grouped.reset_index(drop=True), thresholds


def _sample_frames_from_bins(
    bins: pd.DataFrame,
    *,
    fps: float,
    bin_seconds: float,
    max_frames: int,
    random_state: int,
) -> pd.DataFrame:
    if bins.empty:
        return pd.DataFrame(columns=["Frame", "posture_interaction_group", "time_h", "n_interactions_total"])

    rng = np.random.default_rng(int(random_state))
    bin_frames = max(1, int(round(float(fps) * float(bin_seconds))))
    rows = []
    for group_name, group_bins in bins.groupby("posture_interaction_group", sort=False):
        if group_name == "middle":
            continue
        group_bins = group_bins.sort_values("time_h", kind="mergesort").reset_index(drop=True)

        starts = np.rint(group_bins["time_h"].to_numpy(np.float64) * 3600.0 * float(fps)).astype(np.int64)
        lengths = np.full(len(starts), bin_frames, dtype=np.int64)
        total_frames = int(lengths.sum())
        if total_frames <= 0:
            continue

        if total_frames <= int(max_frames):
            sampled = np.concatenate([np.arange(start, start + bin_frames, dtype=np.int64) for start in starts])
        else:
            n_draw = int(max_frames)
            chosen_bins = rng.choice(len(starts), size=n_draw, replace=True, p=lengths / total_frames)
            offsets = rng.integers(0, lengths[chosen_bins], size=n_draw, endpoint=False)
            sampled = starts[chosen_bins] + offsets
            sampled = np.unique(sampled)

        frame_to_bin_idx = np.searchsorted(starts, sampled, side="right") - 1
        valid = (frame_to_bin_idx >= 0) & (frame_to_bin_idx < len(group_bins))
        sampled = sampled[valid]
        frame_to_bin_idx = frame_to_bin_idx[valid]
        source_bins = group_bins.iloc[frame_to_bin_idx].reset_index(drop=True)
        rows.append(
            pd.DataFrame(
                {
                    "Frame": sampled.astype(np.int64),
                    "posture_interaction_group": group_name,
                    "time_h": source_bins["time_h"].to_numpy(np.float64),
                    "n_interactions_total": source_bins["n_interactions_total"].to_numpy(np.int64),
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=["Frame", "posture_interaction_group", "time_h", "n_interactions_total"])
    return pd.concat(rows, ignore_index=True).drop_duplicates(
        ["Frame", "posture_interaction_group"],
        keep="first",
    )


def _sample_frames_from_intervals(
    intervals: pd.DataFrame,
    *,
    max_frames: int,
    random_state: int,
) -> pd.DataFrame:
    columns = [
        "Frame",
        "posture_interaction_group",
        "time_h",
        "bout_id",
        "bout_start_frame",
        "bout_end_frame",
        "bout_duration_seconds",
        "n_interactions_total",
        "n_interactions_as_antenna",
        "n_interactions_as_body",
    ]
    if intervals.empty:
        return pd.DataFrame(columns=columns)

    rng = np.random.default_rng(int(random_state))
    rows = []
    for group_name, group_intervals in intervals.groupby("posture_interaction_group", sort=False):
        if group_name == "middle":
            continue
        group_intervals = group_intervals.sort_values("bout_start_frame", kind="mergesort").reset_index(drop=True)
        starts = group_intervals["bout_start_frame"].to_numpy(np.int64)
        stops = group_intervals["bout_end_frame"].to_numpy(np.int64) + 1
        lengths = np.maximum(0, stops - starts)
        keep = lengths > 0
        if not keep.any():
            continue
        group_intervals = group_intervals.loc[keep].reset_index(drop=True)
        starts = starts[keep]
        stops = stops[keep]
        lengths = lengths[keep]
        total_frames = int(lengths.sum())
        if total_frames <= 0:
            continue

        if total_frames <= int(max_frames):
            sampled = np.concatenate([np.arange(start, stop, dtype=np.int64) for start, stop in zip(starts, stops)])
        else:
            n_draw = int(max_frames)
            chosen_intervals = rng.choice(len(starts), size=n_draw, replace=True, p=lengths / total_frames)
            offsets = rng.integers(0, lengths[chosen_intervals], size=n_draw, endpoint=False)
            sampled = starts[chosen_intervals] + offsets
            sampled = np.unique(sampled)

        interval_idx = np.searchsorted(starts, sampled, side="right") - 1
        valid = (interval_idx >= 0) & (interval_idx < len(group_intervals)) & (sampled < stops[interval_idx])
        sampled = sampled[valid]
        interval_idx = interval_idx[valid]
        source = group_intervals.iloc[interval_idx].reset_index(drop=True)
        rows.append(
            pd.DataFrame(
                {
                    "Frame": sampled.astype(np.int64),
                    "posture_interaction_group": group_name,
                    "time_h": source["time_h"].to_numpy(np.float64),
                    "bout_id": source["bout_id"].to_numpy(np.int64),
                    "bout_start_frame": source["bout_start_frame"].to_numpy(np.int64),
                    "bout_end_frame": source["bout_end_frame"].to_numpy(np.int64),
                    "bout_duration_seconds": source["bout_duration_seconds"].to_numpy(np.float64),
                    "n_interactions_total": source["n_interactions_total"].to_numpy(np.int64),
                    "n_interactions_as_antenna": source["n_interactions_as_antenna"].to_numpy(np.int64),
                    "n_interactions_as_body": source["n_interactions_as_body"].to_numpy(np.int64),
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True).drop_duplicates(
        ["Frame", "posture_interaction_group"],
        keep="first",
    )


def _read_track_rows_for_frames(track_path: Path, frames: np.ndarray) -> pd.DataFrame:
    from analysis import sleep_classifier_features as scf

    frames = np.unique(np.asarray(frames, dtype=np.int64))
    if len(frames) == 0:
        return pd.DataFrame()

    read_cols = ["Frame", "Bodypoint", "X", "Y", "TrackX", "TrackY"]
    columns = set(scf.parquet_columns(Path(track_path)))
    if "TrackID" in columns:
        read_cols.append("TrackID")

    try:
        import pyarrow.compute as pc
        import pyarrow.dataset as ds

        table = ds.dataset(Path(track_path), format="parquet").to_table(
            columns=read_cols,
            filter=pc.field("Frame").isin(frames.tolist()),
            use_threads=True,
        )
        track_rows = table.to_pandas()
    except Exception:
        track_rows = scf.read_track_rows(
            Path(track_path),
            frame_min=int(frames.min()),
            frame_max=int(frames.max()),
        )
        track_rows = track_rows[track_rows["Frame"].isin(frames)].copy()

    for col in ["Frame", "Bodypoint"]:
        track_rows[col] = pd.to_numeric(track_rows[col], errors="coerce")
    for col in ["X", "Y", "TrackX", "TrackY"]:
        track_rows[col] = pd.to_numeric(track_rows[col], errors="coerce")
    track_rows = track_rows.dropna(subset=["Frame", "Bodypoint"]).copy()
    track_rows["Frame"] = track_rows["Frame"].round().astype(np.int64)
    track_rows["Bodypoint"] = track_rows["Bodypoint"].round().astype(np.int64)
    return track_rows


def _xy_or_nan(wide: pd.DataFrame, bodypoint: int) -> tuple[pd.Series, pd.Series]:
    if ("X", bodypoint) in wide.columns:
        x = wide[("X", bodypoint)].astype(float)
    else:
        x = pd.Series(np.nan, index=wide.index, dtype=float)
    if ("Y", bodypoint) in wide.columns:
        y = wide[("Y", bodypoint)].astype(float)
    else:
        y = pd.Series(np.nan, index=wide.index, dtype=float)
    return x, y


def _safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(num.shape, np.nan, dtype=np.float64)
    keep = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    out[keep] = num[keep] / den[keep]
    return out


def add_posture_engineered_features(
    posture: pd.DataFrame,
    wide: Optional[pd.DataFrame] = None,
    *,
    mm_per_px: float = 0.016,
) -> pd.DataFrame:
    out = posture.copy()

    if wide is not None and not wide.empty:
        for left, right in ADDITIONAL_POSTURE_DISTANCE_PAIRS:
            feature = f"bp{left}_bp{right}_dist_mm"
            if feature in out.columns:
                continue
            x0, y0 = _xy_or_nan(wide, left)
            x1, y1 = _xy_or_nan(wide, right)
            dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) * float(mm_per_px)
            frame_to_dist = pd.Series(dist.to_numpy(np.float64), index=wide.index)
            out[feature] = out["Frame"].map(frame_to_dist)

    def numeric(col: str) -> pd.Series:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype=float)

    width = numeric("pose_width_mm")
    height = numeric("pose_height_mm")
    area = numeric("pose_area_mm2")
    extent = np.sqrt(width.to_numpy(np.float64) ** 2 + height.to_numpy(np.float64) ** 2)
    out["pose_aspect_ratio"] = _safe_divide(width, height)
    out["pose_extent_mm"] = extent
    out["pose_compactness"] = _safe_divide(area, extent ** 2)

    angle_in = pd.DataFrame(
        {
            "left": numeric("angle_in_left_deg"),
            "right": numeric("angle_in_right_deg"),
        }
    )
    angle_out = pd.DataFrame(
        {
            "left": numeric("angle_out_left_deg"),
            "right": numeric("angle_out_right_deg"),
        }
    )
    out["angle_in_mean_deg"] = angle_in.mean(axis=1)
    out["angle_out_mean_deg"] = angle_out.mean(axis=1)
    out["angle_all_mean_deg"] = pd.concat([angle_in, angle_out], axis=1).mean(axis=1)
    out["angle_in_abs_diff_deg"] = (angle_in["left"] - angle_in["right"]).abs()
    out["angle_out_abs_diff_deg"] = (angle_out["left"] - angle_out["right"]).abs()
    out["angle_left_sum_deg"] = numeric("angle_in_left_deg") + numeric("angle_out_left_deg")
    out["angle_right_sum_deg"] = numeric("angle_in_right_deg") + numeric("angle_out_right_deg")

    d14 = numeric("bp1_bp4_dist_mm")
    d17 = numeric("bp1_bp7_dist_mm")
    d45 = numeric("bp4_bp5_dist_mm")
    d56 = numeric("bp5_bp6_dist_mm")
    d78 = numeric("bp7_bp8_dist_mm")
    d89 = numeric("bp8_bp9_dist_mm")
    d47 = numeric("bp4_bp7_dist_mm")
    d58 = numeric("bp5_bp8_dist_mm")
    d16 = numeric("bp1_bp6_dist_mm")
    d19 = numeric("bp1_bp9_dist_mm")
    d69 = numeric("bp6_bp9_dist_mm")

    left_chain = d14 + d45 + d56
    right_chain = d17 + d78 + d89
    side_chain_mean = pd.concat([left_chain, right_chain], axis=1).mean(axis=1)
    out["left_side_chain_mm"] = left_chain
    out["right_side_chain_mm"] = right_chain
    out["side_chain_mean_mm"] = side_chain_mean
    out["side_chain_abs_diff_mm"] = (left_chain - right_chain).abs()
    out["side_chain_ratio"] = _safe_divide(left_chain, right_chain)
    out["left_segment_ratio"] = _safe_divide(d45, d56)
    out["right_segment_ratio"] = _safe_divide(d78, d89)
    out["mid_span_ratio"] = _safe_divide(d58, d47)
    out["tip_span_mm"] = d69
    out["tip_span_to_pose_width"] = _safe_divide(d69, width)
    out["head_to_left_tip_ratio"] = _safe_divide(d16, left_chain)
    out["head_to_right_tip_ratio"] = _safe_divide(d19, right_chain)

    all_distance_cols = [
        *CORE_POSTURE_DISTANCE_FEATURE_COLUMNS,
        *ADDITIONAL_POSTURE_DISTANCE_FEATURE_COLUMNS,
    ]
    for col in all_distance_cols:
        if col in out.columns:
            out[f"{col}_per_pose_width"] = _safe_divide(numeric(col), width)

    return out


def _quiescent_posture_samples_for_track_row(
    speed_interactions: pd.DataFrame,
    track_row: pd.Series,
    *,
    counts_by_track: dict[int, pd.DataFrame],
    per_track_root: Optional[Path] = None,
    fps: float = 24.0,
    mm_per_px: float = 0.016,
    speed_smooth_seconds: float = 5 * 60.0,
    quiescence_speed_threshold_mm_s: float = 0.1,
    min_quiescent_seconds: float = 10.0,
    analysis_frame_start: Optional[int] = None,
    analysis_frame_stop: Optional[int] = None,
    low_interaction_max: int = 0,
    high_interaction_min: Optional[int] = None,
    high_interaction_quantile: float = 0.75,
    include_non_quiescent: bool = False,
    max_frames_per_group: int = 20000,
    random_state: int = 0,
    interaction_count_col: str = "n_interactions_total",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from analysis import sleep_classifier_features as scf

    track_path = resolve_track_path_for_speed_row(track_row, per_track_root=per_track_root)
    grouped_intervals, thresholds = quiescent_interaction_bout_groups(
        track_row,
        counts_by_track,
        speed_smooth_seconds=speed_smooth_seconds,
        quiescence_speed_threshold_mm_s=quiescence_speed_threshold_mm_s,
        min_quiescent_seconds=min_quiescent_seconds,
        analysis_frame_start=analysis_frame_start,
        analysis_frame_stop=analysis_frame_stop,
        low_interaction_max=low_interaction_max,
        high_interaction_min=high_interaction_min,
        high_interaction_quantile=high_interaction_quantile,
        include_non_quiescent=include_non_quiescent,
        interaction_count_col=interaction_count_col,
    )
    sampled_frames = _sample_frames_from_intervals(
        grouped_intervals,
        max_frames=int(max_frames_per_group),
        random_state=int(random_state),
    )
    if sampled_frames.empty:
        raise ValueError("No posture frames were sampled; loosen interaction thresholds or increase max_frames_per_group")

    track_rows = _read_track_rows_for_frames(track_path, sampled_frames["Frame"].to_numpy(np.int64))
    wide, anchor = scf.pose_wide(track_rows)
    if wide.empty:
        raise ValueError(f"No pose rows loaded from {track_path}")

    posture = scf.posture_features(wide, anchor, mm_per_px=float(mm_per_px))
    posture = add_posture_engineered_features(posture, wide, mm_per_px=float(mm_per_px))
    samples = posture.merge(sampled_frames, on="Frame", how="inner", validate="one_to_one")
    samples["track_name"] = track_row.get("track_name")
    samples["track_id"] = track_row.get("track_id")
    samples["side"] = track_row.get("side")
    samples["track_row"] = track_row.name
    samples["track_path"] = str(track_path)
    if "bin_seconds" in speed_interactions and speed_interactions["bin_seconds"].notna().any():
        samples["plot_bin_seconds"] = float(speed_interactions["bin_seconds"].dropna().iloc[0])

    summarized_intervals = grouped_intervals[grouped_intervals["posture_interaction_group"] != "middle"].copy()
    interval_summary = (
        summarized_intervals.groupby("posture_interaction_group", sort=False)
        .agg(
            n_bouts=("time_h", "size"),
            median_bout_duration_s=("bout_duration_seconds", "median"),
            total_bout_duration_s=("bout_duration_seconds", "sum"),
            min_interactions=("n_interactions_total", "min"),
            median_interactions=("n_interactions_total", "median"),
            max_interactions=("n_interactions_total", "max"),
        )
        .reset_index()
    )
    sample_counts = samples.groupby("posture_interaction_group", sort=False).size().rename("n_sampled_frames")
    interval_summary = interval_summary.merge(sample_counts, on="posture_interaction_group", how="left")
    for key, value in thresholds.items():
        interval_summary[key] = value
    interval_summary["track_name"] = track_row.get("track_name")
    interval_summary["track_id"] = track_row.get("track_id")
    interval_summary["side"] = track_row.get("side")
    interval_summary["track_row"] = track_row.name
    interval_summary["track_path"] = str(track_path)
    return samples.reset_index(drop=True), interval_summary


def quiescent_posture_samples_by_interaction(
    speed_interactions: pd.DataFrame,
    speed_tracks: pd.DataFrame,
    *,
    row_number: Optional[int] = None,
    track_id: Optional[int] = None,
    side: Optional[str] = "left",
    counts_by_track: Optional[dict[int, pd.DataFrame]] = None,
    per_track_root: Optional[Path] = None,
    fps: float = 24.0,
    mm_per_px: float = 0.016,
    speed_smooth_seconds: float = 5 * 60.0,
    quiescence_speed_threshold_mm_s: float = 0.1,
    min_quiescent_seconds: float = 10.0,
    analysis_frame_start: Optional[int] = None,
    analysis_frame_stop: Optional[int] = None,
    low_interaction_max: int = 0,
    high_interaction_min: Optional[int] = None,
    high_interaction_quantile: float = 0.75,
    include_non_quiescent: bool = False,
    max_frames_per_group: int = 20000,
    random_state: int = 0,
    interaction_count_col: str = "n_interactions_total",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    track_row = selected_speed_track(
        speed_tracks,
        row_number=row_number,
        track_id=track_id,
        side=side,
    )
    if counts_by_track is None:
        raise ValueError("counts_by_track is required so interactions can be counted over each quiescent bout")
    return _quiescent_posture_samples_for_track_row(
        speed_interactions,
        track_row,
        counts_by_track=counts_by_track,
        per_track_root=per_track_root,
        fps=fps,
        mm_per_px=mm_per_px,
        speed_smooth_seconds=speed_smooth_seconds,
        quiescence_speed_threshold_mm_s=quiescence_speed_threshold_mm_s,
        min_quiescent_seconds=min_quiescent_seconds,
        analysis_frame_start=analysis_frame_start,
        analysis_frame_stop=analysis_frame_stop,
        low_interaction_max=low_interaction_max,
        high_interaction_min=high_interaction_min,
        high_interaction_quantile=high_interaction_quantile,
        include_non_quiescent=include_non_quiescent,
        max_frames_per_group=max_frames_per_group,
        random_state=random_state,
        interaction_count_col=interaction_count_col,
    )


def quiescent_posture_samples_for_tracks(
    speed_tracks: pd.DataFrame,
    interactions: Optional[pd.DataFrame] = None,
    *,
    row_numbers: Optional[list[int]] = None,
    track_ids: Optional[list[int | str]] = None,
    side: Optional[str] = "left",
    max_tracks: Optional[int] = None,
    per_track_root: Optional[Path] = None,
    fps: float = 24.0,
    mm_per_px: float = 0.016,
    bin_seconds: float = 60.0,
    speed_smooth_seconds: float = 5 * 60.0,
    quiescence_speed_threshold_mm_s: Optional[float] = 0.1,
    min_quiescent_seconds: float = 10.0,
    analysis_frame_start: Optional[int] = None,
    analysis_frame_stop: Optional[int] = None,
    low_interaction_max: int = 0,
    high_interaction_min: Optional[int] = None,
    high_interaction_quantile: float = 0.75,
    include_non_quiescent: bool = False,
    max_frames_per_group: int = 5000,
    random_state: int = 0,
    chunk_global_frame_offset: int = 0,
    counts_by_track: Optional[dict[int, pd.DataFrame]] = None,
    interaction_count_col: str = "n_interactions_total",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from analysis import colony_speed_utils as cs
    from analysis import interaction_analysis_utils as ia

    if quiescence_speed_threshold_mm_s is None:
        raise ValueError("quiescence_speed_threshold_mm_s is required for bout-based posture sampling")

    chosen = cs.choose_tracks(
        speed_tracks,
        side=side,
        row_numbers=row_numbers,
        track_ids=track_ids,
        max_tracks=max_tracks,
        sort_tracks=True,
    )
    if counts_by_track is None:
        if interactions is None:
            raise ValueError("Provide interactions or precomputed counts_by_track")
        counts_by_track = ia.interaction_bout_counts_by_track(
            interactions,
            chunk_global_frame_offset=int(chunk_global_frame_offset),
            fps=fps,
        )

    all_samples = []
    all_bin_summaries = []
    all_speed_interactions = []
    for ant_number, (_, track_row) in enumerate(chosen.iterrows()):
        speed_interactions = _speed_interaction_timeseries_for_track_row(
            track_row,
            counts_by_track,
            bin_seconds=bin_seconds,
            speed_smooth_seconds=speed_smooth_seconds,
        )
        speed_interactions["is_quiescent"] = False
        if quiescence_speed_threshold_mm_s is not None:
            smoothed_speed = speed_interactions["smoothed_speed_mm_s"].to_numpy(np.float64)
            speed_interactions["is_quiescent"] = np.isfinite(smoothed_speed) & (
                smoothed_speed <= float(quiescence_speed_threshold_mm_s)
            )
        speed_interactions["source_ant_number"] = int(ant_number)
        all_speed_interactions.append(speed_interactions)

        samples, bin_summary = _quiescent_posture_samples_for_track_row(
            speed_interactions,
            track_row,
            counts_by_track=counts_by_track,
            per_track_root=per_track_root,
            fps=fps,
            mm_per_px=mm_per_px,
            speed_smooth_seconds=speed_smooth_seconds,
            quiescence_speed_threshold_mm_s=float(quiescence_speed_threshold_mm_s),
            min_quiescent_seconds=min_quiescent_seconds,
            analysis_frame_start=analysis_frame_start,
            analysis_frame_stop=analysis_frame_stop,
            low_interaction_max=low_interaction_max,
            high_interaction_min=high_interaction_min,
            high_interaction_quantile=high_interaction_quantile,
            include_non_quiescent=include_non_quiescent,
            max_frames_per_group=max_frames_per_group,
            random_state=int(random_state) + int(ant_number),
            interaction_count_col=interaction_count_col,
        )
        samples["source_ant_number"] = int(ant_number)
        bin_summary["source_ant_number"] = int(ant_number)
        all_samples.append(samples)
        all_bin_summaries.append(bin_summary)

    if not all_samples:
        raise ValueError("No posture samples were produced for the selected tracks")

    return (
        pd.concat(all_samples, ignore_index=True),
        pd.concat(all_bin_summaries, ignore_index=True),
        pd.concat(all_speed_interactions, ignore_index=True),
    )


def posture_distribution_summary(
    posture_samples: pd.DataFrame,
    *,
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    if feature_cols is None:
        feature_cols = [
            "angle_in_left_deg",
            "angle_out_left_deg",
            "angle_in_right_deg",
            "angle_out_right_deg",
            "pose_width_mm",
            "pose_height_mm",
            "pose_area_mm2",
        ]

    rows = []
    for group_name, group in posture_samples.groupby("posture_interaction_group", sort=False):
        for feature in feature_cols:
            values = pd.to_numeric(group[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            rows.append(
                {
                    "posture_interaction_group": group_name,
                    "feature": feature,
                    "n": int(len(values)),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "median": float(values.median()) if len(values) else np.nan,
                    "q25": float(values.quantile(0.25)) if len(values) else np.nan,
                    "q75": float(values.quantile(0.75)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_quiescent_posture_distributions(
    posture_samples: pd.DataFrame,
    *,
    feature_cols: Optional[list[str]] = None,
    bins: int = 50,
    density: bool = True,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    if feature_cols is None:
        feature_cols = [
            "angle_in_left_deg",
            "angle_out_left_deg",
            "angle_in_right_deg",
            "angle_out_right_deg",
            "pose_width_mm",
            "pose_height_mm",
            "pose_area_mm2",
        ]

    missing_features = [feature for feature in feature_cols if feature not in posture_samples.columns]
    if missing_features:
        raise ValueError(f"posture_samples is missing plot feature columns: {missing_features}")
    if "posture_interaction_group" not in posture_samples.columns:
        raise ValueError("posture_samples must include 'posture_interaction_group'")

    groups = [
        group
        for group in ["little_or_no_interaction", "more_interaction", "non_quiescent"]
        if group in set(posture_samples["posture_interaction_group"])
    ]
    if not groups:
        raise ValueError("No low/high interaction posture groups available to plot")

    ncols = 2
    nrows = int(np.ceil(len(feature_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.2 * nrows), squeeze=False)
    colors = {
        "little_or_no_interaction": "tab:blue",
        "more_interaction": "tab:orange",
        "non_quiescent": "0.35",
    }

    for ax, feature in zip(axes.ravel(), feature_cols):
        values_by_group: dict[str, np.ndarray] = {}
        for group in groups:
            values = (
                pd.to_numeric(
                    posture_samples.loc[posture_samples["posture_interaction_group"] == group, feature],
                    errors="coerce",
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy(np.float64)
            )
            if len(values) == 0:
                continue
            values_by_group[group] = values

        if not values_by_group:
            ax.set_title(feature)
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        all_values = np.concatenate(list(values_by_group.values()))
        if np.nanmin(all_values) == np.nanmax(all_values):
            center = float(all_values[0])
            pad = max(abs(center) * 0.05, 1e-6)
            bin_edges = np.linspace(center - pad, center + pad, int(bins) + 1)
        else:
            bin_edges = np.histogram_bin_edges(all_values, bins=int(bins))

        for group, values in values_by_group.items():
            ax.hist(
                values,
                bins=bin_edges,
                density=bool(density),
                histtype="step",
                linewidth=1.8,
                color=colors.get(group),
                label=f"{POSTURE_GROUP_DISPLAY_LABELS.get(group, group)} finite n={len(values):,}",
            )
        ax.set_title(feature)
        ax.set_ylabel("density" if density else "frames")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes.ravel()[len(feature_cols):]:
        ax.axis("off")
    fig.suptitle("Posture distributions by quiescence/interaction group")
    fig.tight_layout()
    plt.show()
    return posture_distribution_summary(posture_samples, feature_cols=feature_cols)


def multivariate_quiescent_posture_analysis(
    posture_samples: pd.DataFrame,
    *,
    feature_cols: Optional[list[str]] = None,
    groups: tuple[str, str] = ("little_or_no_interaction", "more_interaction"),
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if feature_cols is None:
        feature_cols = [
            "angle_in_left_deg",
            "angle_out_left_deg",
            "angle_in_right_deg",
            "angle_out_right_deg",
            "pose_width_mm",
            "pose_height_mm",
            "pose_area_mm2",
        ]

    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    group_col = "posture_interaction_group"
    data = posture_samples[posture_samples[group_col].isin(groups)].copy()
    if data[group_col].nunique() < 2:
        raise ValueError(f"Need both groups for multivariate analysis: {groups}")

    numeric = data[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep_features = [col for col in numeric.columns if numeric[col].notna().sum() >= 2]
    if not keep_features:
        raise ValueError("No posture feature has enough finite values for multivariate analysis")
    numeric = numeric[keep_features]
    keep_rows = numeric.notna().any(axis=1)
    data = data.loc[keep_rows].copy()
    numeric = numeric.loc[keep_rows]

    y = (data[group_col] == groups[1]).astype(int).to_numpy()
    group_labels = data[group_col].to_numpy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_imp = imputer.fit_transform(numeric)
    x_z = scaler.fit_transform(x_imp)

    pca = PCA(n_components=min(2, x_z.shape[1]), random_state=int(random_state))
    pcs = pca.fit_transform(x_z)
    embedding = data.reset_index(drop=True).copy()
    embedding["posture_pc1"] = pcs[:, 0]
    embedding["posture_pc2"] = pcs[:, 1] if pcs.shape[1] > 1 else 0.0

    loadings = pd.DataFrame(
        {
            "feature": keep_features,
            "pc1_loading": pca.components_[0],
            "pc2_loading": pca.components_[1] if pca.components_.shape[0] > 1 else np.nan,
        }
    )
    loadings["abs_pc1_loading"] = loadings["pc1_loading"].abs()
    loadings["abs_pc2_loading"] = loadings["pc2_loading"].abs()
    loadings = loadings.sort_values("abs_pc1_loading", ascending=False, kind="mergesort").reset_index(drop=True)

    rows = [
        {
            "metric": "n_features",
            "value": float(len(keep_features)),
            "detail": ", ".join(keep_features),
        },
        {
            "metric": "pc1_explained_variance",
            "value": float(pca.explained_variance_ratio_[0]),
            "detail": "",
        },
        {
            "metric": "pc2_explained_variance",
            "value": float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else np.nan,
            "detail": "",
        },
    ]

    centroids = {
        group: x_z[group_labels == group].mean(axis=0)
        for group in groups
    }
    rows.append(
        {
            "metric": "centroid_distance_z_units",
            "value": float(np.linalg.norm(centroids[groups[1]] - centroids[groups[0]])),
            "detail": f"{groups[0]} to {groups[1]}",
        }
    )

    class_counts = pd.Series(group_labels).value_counts()
    rows.extend(
        {
            "metric": f"n_{group}",
            "value": float(class_counts.get(group, 0)),
            "detail": "",
        }
        for group in groups
    )

    if class_counts.min() >= 3:
        n_splits = int(min(5, class_counts.min()))
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=int(random_state)),
                ),
            ]
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
        pred = cross_val_predict(model, numeric, y, cv=cv, method="predict")
        score = cross_val_predict(model, numeric, y, cv=cv, method="predict_proba")[:, 1]
        rows.append(
            {
                "metric": "cv_balanced_accuracy",
                "value": float(balanced_accuracy_score(y, pred)),
                "detail": f"{n_splits}-fold logistic regression, all features jointly",
            }
        )
        rows.append(
            {
                "metric": "cv_roc_auc",
                "value": float(roc_auc_score(y, score)),
                "detail": f"{n_splits}-fold logistic regression, all features jointly",
            }
        )
        embedding["joint_posture_interaction_score"] = score
    else:
        embedding["joint_posture_interaction_score"] = np.nan

    metrics = pd.DataFrame(rows)
    return embedding, metrics, loadings


def plot_multivariate_quiescent_posture_analysis(
    posture_samples: pd.DataFrame,
    *,
    feature_cols: Optional[list[str]] = None,
    groups: tuple[str, str] = ("little_or_no_interaction", "more_interaction"),
    random_state: int = 0,
    max_points_per_group: int = 3000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import matplotlib.pyplot as plt

    embedding, metrics, loadings = multivariate_quiescent_posture_analysis(
        posture_samples,
        feature_cols=feature_cols,
        groups=groups,
        random_state=random_state,
    )

    rng = np.random.default_rng(int(random_state))
    plot_rows = []
    for group, group_df in embedding.groupby("posture_interaction_group", sort=False):
        if max_points_per_group is not None and len(group_df) > int(max_points_per_group):
            idx = rng.choice(group_df.index.to_numpy(), size=int(max_points_per_group), replace=False)
            plot_rows.append(group_df.loc[idx])
        else:
            plot_rows.append(group_df)
    plot_df = pd.concat(plot_rows, ignore_index=True)

    colors = {"little_or_no_interaction": "tab:blue", "more_interaction": "tab:orange"}
    fig, (scatter_ax, score_ax) = plt.subplots(1, 2, figsize=(12, 4.8))
    for group in groups:
        subset = plot_df[plot_df["posture_interaction_group"] == group]
        scatter_ax.scatter(
            subset["posture_pc1"],
            subset["posture_pc2"],
            s=8,
            alpha=0.25,
            color=colors.get(group),
            label=f"{group} n={len(subset):,}",
        )
    scatter_ax.set_xlabel("joint posture PC1")
    scatter_ax.set_ylabel("joint posture PC2")
    scatter_ax.set_title("Quiescent posture space")
    scatter_ax.grid(True, alpha=0.25)
    scatter_ax.legend(fontsize=8)

    if embedding["joint_posture_interaction_score"].notna().any():
        for group in groups:
            values = embedding.loc[
                embedding["posture_interaction_group"] == group,
                "joint_posture_interaction_score",
            ].dropna()
            score_ax.hist(
                values,
                bins=50,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=colors.get(group),
                label=group,
            )
        score_ax.set_xlabel("joint posture interaction score")
        score_ax.set_ylabel("density")
        score_ax.set_title("Cross-validated logistic score")
        score_ax.grid(True, alpha=0.25)
        score_ax.legend(fontsize=8)
    else:
        score_ax.axis("off")

    fig.tight_layout()
    plt.show()
    return embedding, metrics, loadings


def _default_joint_posture_feature_columns(posture_samples: pd.DataFrame) -> list[str]:
    preferred = [
        *BASE_POSTURE_FEATURE_COLUMNS,
        *CORE_POSTURE_DISTANCE_FEATURE_COLUMNS,
        *ADDITIONAL_POSTURE_DISTANCE_FEATURE_COLUMNS,
        *POSTURE_ENGINEERED_FEATURE_COLUMNS,
        *POSTURE_NORMALIZED_DISTANCE_FEATURE_COLUMNS,
    ]
    return [col for col in preferred if col in posture_samples.columns]


def _classifier_bin_key(data: pd.DataFrame) -> pd.Series:
    if {"bout_start_frame", "bout_end_frame"}.issubset(data.columns):
        start_value = pd.to_numeric(data["bout_start_frame"], errors="coerce").astype("Int64").astype(str)
        end_value = pd.to_numeric(data["bout_end_frame"], errors="coerce").astype("Int64").astype(str)
        bin_value = start_value + "-" + end_value
    elif "bout_id" in data.columns:
        bin_value = pd.to_numeric(data["bout_id"], errors="coerce").astype("Int64").astype(str)
    elif "time_h" in data.columns:
        bin_value = pd.to_numeric(data["time_h"], errors="coerce").round(9).astype(str)
    else:
        bin_value = pd.to_numeric(data["Frame"], errors="coerce").astype("Int64").astype(str)

    if "track_name" in data.columns:
        track_value = data["track_name"].astype(str)
    elif "track_id" in data.columns:
        track_value = data["track_id"].astype(str)
    else:
        track_value = pd.Series("", index=data.index)

    if "side" in data.columns:
        side_value = data["side"].astype(str)
    else:
        side_value = pd.Series("", index=data.index)
    return side_value + "|" + track_value + "|" + bin_value


def _posture_classifier_data(
    posture_samples: pd.DataFrame,
    *,
    feature_cols: Optional[list[str]],
    groups: tuple[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    group_col = "posture_interaction_group"
    if group_col not in posture_samples.columns:
        raise ValueError(f"posture_samples must include {group_col!r}")

    if feature_cols is None:
        feature_cols = _default_joint_posture_feature_columns(posture_samples)
    missing_features = [col for col in feature_cols if col not in posture_samples.columns]
    if missing_features:
        raise ValueError(f"posture_samples is missing classifier feature columns: {missing_features}")
    if not feature_cols:
        raise ValueError("No posture feature columns are available for classifier training")

    data = posture_samples[posture_samples[group_col].isin(groups)].copy()
    if data[group_col].nunique() < 2:
        raise ValueError(f"Need both posture groups for classifier training: {groups}")

    numeric = data[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep_features = [col for col in numeric.columns if numeric[col].notna().sum() >= 2]
    if not keep_features:
        raise ValueError("No classifier feature has enough finite values")
    numeric = numeric[keep_features]
    keep_rows = numeric.notna().any(axis=1)
    data = data.loc[keep_rows].copy()
    numeric = numeric.loc[keep_rows]
    y = (data[group_col] == groups[1]).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError(f"Need both posture groups after removing all-NaN feature rows: {groups}")
    return data, numeric, y, keep_features


def _classifier_split_masks(
    data: pd.DataFrame,
    y: np.ndarray,
    *,
    groups: tuple[str, str],
    split_by: str,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedShuffleSplit

    group_col = "posture_interaction_group"
    split_by = str(split_by)
    if split_by == "time_bout":
        split_by = "time_bin"
    elif split_by == "random_bout":
        split_by = "random_bin"
    test_size = float(test_size)
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")

    if split_by == "random_frame":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=int(random_state))
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
        train_mask = np.zeros(len(y), dtype=bool)
        test_mask = np.zeros(len(y), dtype=bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        return train_mask, test_mask

    bin_key = _classifier_bin_key(data)
    bin_table = (
        pd.DataFrame(
            {
                "bin_key": bin_key.to_numpy(),
                "group": data[group_col].to_numpy(),
                "time_h": pd.to_numeric(data.get("time_h", pd.Series(np.arange(len(data)))), errors="coerce").to_numpy(),
            }
        )
        .drop_duplicates(["bin_key", "group"])
        .reset_index(drop=True)
    )
    if bin_table["bin_key"].duplicated().any():
        raise ValueError("A classifier split unit maps to multiple posture groups")

    if split_by == "random_bin":
        y_bin = (bin_table["group"] == groups[1]).astype(int).to_numpy()
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=int(random_state))
        train_bin_idx, test_bin_idx = next(splitter.split(np.zeros(len(y_bin)), y_bin))
        train_bins = set(bin_table.iloc[train_bin_idx]["bin_key"])
        test_bins = set(bin_table.iloc[test_bin_idx]["bin_key"])
    elif split_by == "time_bin":
        train_bins = set()
        test_bins = set()
        for group in groups:
            group_bins = bin_table[bin_table["group"] == group].sort_values(
                ["time_h", "bin_key"],
                kind="mergesort",
            )
            n_bins = len(group_bins)
            if n_bins < 2:
                raise ValueError(f"Need at least two split units for {group!r} to make a time-ordered holdout")
            n_test = int(np.ceil(n_bins * test_size))
            n_test = min(max(1, n_test), n_bins - 1)
            train_bins.update(group_bins.iloc[: n_bins - n_test]["bin_key"])
            test_bins.update(group_bins.iloc[n_bins - n_test :]["bin_key"])
    else:
        raise ValueError("split_by must be one of: 'time_bout', 'random_bout', 'time_bin', 'random_bin', 'random_frame'")

    train_mask = bin_key.isin(train_bins).to_numpy()
    test_mask = bin_key.isin(test_bins).to_numpy()
    return train_mask, test_mask


def _coerce_mask(mask: object, source_index: pd.Index, target_index: pd.Index, name: str) -> np.ndarray:
    if mask is None:
        raise ValueError(f"{name} cannot be None")
    if isinstance(mask, pd.Series):
        values = mask.reindex(source_index).fillna(False).astype(bool)
    else:
        values = pd.Series(mask, index=source_index).astype(bool)
    return values.reindex(target_index).fillna(False).to_numpy(bool)


def _classifier_metrics_frame(
    predictions: pd.DataFrame,
    *,
    groups: tuple[str, str],
    split_label: str,
    split_by: str,
) -> pd.DataFrame:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    rows = []
    for split_name, split_df in predictions[predictions["split"].isin(["train", "test"])].groupby("split", sort=False):
        y_true = split_df["y_true"].to_numpy(int)
        y_pred = split_df["y_pred"].to_numpy(int)
        score = split_df["prob_more_interaction"].to_numpy(float)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        n_units = int(split_df["classifier_bin"].nunique())
        has_bout_units = {"bout_start_frame", "bout_end_frame"}.issubset(split_df.columns)
        rows.append(
            {
                "split_label": split_label,
                "split_by": split_by,
                "split": split_name,
                "n_frames": int(len(split_df)),
                "n_bins": n_units,
                "n_bouts": n_units if has_bout_units else np.nan,
                f"n_{groups[0]}": int((y_true == 0).sum()),
                f"n_{groups[1]}": int((y_true == 1).sum()),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "roc_auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan,
                f"precision_{groups[1]}": float(precision_score(y_true, y_pred, zero_division=0)),
                f"recall_{groups[1]}": float(recall_score(y_true, y_pred, zero_division=0)),
                f"f1_{groups[1]}": float(f1_score(y_true, y_pred, zero_division=0)),
                "true_low_pred_low": int(tn),
                "true_low_pred_high": int(fp),
                "true_high_pred_low": int(fn),
                "true_high_pred_high": int(tp),
            }
        )
    return pd.DataFrame(rows)


def train_test_quiescent_posture_classifier(
    posture_samples: pd.DataFrame,
    *,
    feature_cols: Optional[list[str]] = None,
    groups: tuple[str, str] = ("little_or_no_interaction", "more_interaction"),
    split_by: str = "time_bin",
    test_size: float = 0.5,
    random_state: int = 0,
    train_mask: Optional[object] = None,
    test_mask: Optional[object] = None,
    split_label: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    source_index = posture_samples.index
    data, numeric, y, keep_features = _posture_classifier_data(
        posture_samples,
        feature_cols=feature_cols,
        groups=groups,
    )

    if train_mask is None and test_mask is None:
        train_mask_array, test_mask_array = _classifier_split_masks(
            data,
            y,
            groups=groups,
            split_by=split_by,
            test_size=test_size,
            random_state=random_state,
        )
    elif train_mask is not None and test_mask is not None:
        train_mask_array = _coerce_mask(train_mask, source_index, data.index, "train_mask")
        test_mask_array = _coerce_mask(test_mask, source_index, data.index, "test_mask")
        split_by = "explicit"
    else:
        raise ValueError("Provide both train_mask and test_mask, or neither")

    if np.any(train_mask_array & test_mask_array):
        raise ValueError("train_mask and test_mask overlap")
    if train_mask_array.sum() == 0 or test_mask_array.sum() == 0:
        raise ValueError("Classifier split produced an empty train or test set")
    if len(np.unique(y[train_mask_array])) < 2:
        raise ValueError("Classifier train set needs both posture groups")
    if len(np.unique(y[test_mask_array])) < 2:
        raise ValueError("Classifier test set needs both posture groups")

    train_numeric = numeric.loc[train_mask_array]
    keep_features = [col for col in keep_features if train_numeric[col].notna().sum() > 0]
    if not keep_features:
        raise ValueError("No classifier feature has finite values in the training split")
    numeric = numeric[keep_features]

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=int(random_state)),
            ),
        ]
    )
    model.fit(numeric.loc[train_mask_array], y[train_mask_array])

    used_mask = train_mask_array | test_mask_array
    pred_data = data.loc[used_mask].copy()
    pred_numeric = numeric.loc[used_mask]
    prob = model.predict_proba(pred_numeric)[:, 1]
    pred = model.predict(pred_numeric)

    predictions = pred_data.reset_index(drop=False).rename(columns={"index": "source_index"})
    used_y = y[used_mask]
    predictions["classifier_bin"] = _classifier_bin_key(pred_data).to_numpy()
    predictions["split"] = np.where(train_mask_array[used_mask], "train", "test")
    predictions["y_true"] = used_y
    predictions["y_pred"] = pred.astype(int)
    predictions["prob_more_interaction"] = prob.astype(float)
    predictions["predicted_posture_interaction_group"] = np.where(pred == 1, groups[1], groups[0])

    split_label = split_label or split_by
    metrics = _classifier_metrics_frame(
        predictions,
        groups=groups,
        split_label=split_label,
        split_by=split_by,
    )

    coef = model.named_steps["logistic"].coef_[0]
    weights = pd.DataFrame(
        {
            "feature": keep_features,
            f"coefficient_for_{groups[1]}": coef,
            "abs_coefficient": np.abs(coef),
        }
    ).sort_values("abs_coefficient", ascending=False, kind="mergesort").reset_index(drop=True)
    return predictions, metrics, weights, model


def plot_quiescent_posture_classifier(
    posture_samples: pd.DataFrame,
    *,
    feature_cols: Optional[list[str]] = None,
    groups: tuple[str, str] = ("little_or_no_interaction", "more_interaction"),
    split_by: str = "time_bin",
    test_size: float = 0.5,
    random_state: int = 0,
    train_mask: Optional[object] = None,
    test_mask: Optional[object] = None,
    split_label: Optional[str] = None,
    max_weight_features: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    import matplotlib.pyplot as plt

    predictions, metrics, weights, model = train_test_quiescent_posture_classifier(
        posture_samples,
        feature_cols=feature_cols,
        groups=groups,
        split_by=split_by,
        test_size=test_size,
        random_state=random_state,
        train_mask=train_mask,
        test_mask=test_mask,
        split_label=split_label,
    )

    colors = {groups[0]: "tab:blue", groups[1]: "tab:orange"}
    test_predictions = predictions[predictions["split"] == "test"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for group in groups:
        values = test_predictions.loc[
            test_predictions["posture_interaction_group"] == group,
            "prob_more_interaction",
        ].dropna()
        axes[0].hist(
            values,
            bins=40,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=colors.get(group),
            label=f"{group} n={len(values):,}",
        )
    axes[0].set_xlabel(f"P({groups[1]})")
    axes[0].set_ylabel("density")
    axes[0].set_title("Held-out classifier scores")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    confusion_values = metrics.loc[metrics["split"] == "test"].iloc[0]
    confusion = np.array(
        [
            [confusion_values["true_low_pred_low"], confusion_values["true_low_pred_high"]],
            [confusion_values["true_high_pred_low"], confusion_values["true_high_pred_high"]],
        ],
        dtype=float,
    )
    image = axes[1].imshow(confusion, cmap="Blues")
    axes[1].set_xticks([0, 1], labels=[groups[0], groups[1]], rotation=30, ha="right")
    axes[1].set_yticks([0, 1], labels=[groups[0], groups[1]])
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("true")
    axes[1].set_title("Held-out confusion")
    for row in range(2):
        for col in range(2):
            axes[1].text(col, row, f"{int(confusion[row, col]):,}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    coefficient_col = f"coefficient_for_{groups[1]}"
    top_weights = weights.head(int(max_weight_features)).iloc[::-1]
    axes[2].barh(
        top_weights["feature"],
        top_weights[coefficient_col],
        color=np.where(top_weights[coefficient_col] >= 0, "tab:orange", "tab:blue"),
        alpha=0.85,
    )
    axes[2].axvline(0, color="0.25", lw=1)
    axes[2].set_xlabel(f"standardized coefficient for {groups[1]}")
    axes[2].set_title("Largest classifier weights")
    axes[2].grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    plt.show()
    return predictions, metrics, weights, model


def _numeric_or_nan(frame: pd.DataFrame, col: str) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _correlation_summary(df: pd.DataFrame, *, x_col: str, y_col: str) -> pd.Series:
    valid = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3:
        return pd.Series(
            {
                "x_col": x_col,
                "y_col": y_col,
                "n": int(len(valid)),
                "pearson_r": np.nan,
                "spearman_r": np.nan,
                "pearson_p": np.nan,
                "spearman_p": np.nan,
            }
        )
    if valid[x_col].nunique(dropna=True) < 2 or valid[y_col].nunique(dropna=True) < 2:
        return pd.Series(
            {
                "x_col": x_col,
                "y_col": y_col,
                "n": int(len(valid)),
                "pearson_r": np.nan,
                "spearman_r": np.nan,
                "pearson_p": np.nan,
                "spearman_p": np.nan,
            }
        )

    pearson_r = valid[x_col].corr(valid[y_col], method="pearson")
    spearman_r = valid[x_col].corr(valid[y_col], method="spearman")
    pearson_p = np.nan
    spearman_p = np.nan
    try:
        from scipy.stats import pearsonr, spearmanr

        pearson_r, pearson_p = pearsonr(valid[x_col], valid[y_col])
        spearman_r, spearman_p = spearmanr(valid[x_col], valid[y_col])
    except Exception:
        pass
    return pd.Series(
        {
            "x_col": x_col,
            "y_col": y_col,
            "n": int(len(valid)),
            "pearson_r": float(pearson_r),
            "spearman_r": float(spearman_r),
            "pearson_p": float(pearson_p) if np.isfinite(pearson_p) else np.nan,
            "spearman_p": float(spearman_p) if np.isfinite(spearman_p) else np.nan,
        }
    )


def low_interaction_quiescent_classifier_bins(
    classifier_predictions: pd.DataFrame,
    *,
    groups: tuple[str, str] = ("little_or_no_interaction", "more_interaction"),
    split: Optional[str] = "test",
    low_score_threshold: float = 0.5,
    min_predicted_low_fraction: float = 0.5,
) -> pd.DataFrame:
    required = {
        "classifier_bin",
        "posture_interaction_group",
        "predicted_posture_interaction_group",
        "prob_more_interaction",
    }
    missing = required.difference(classifier_predictions.columns)
    if missing:
        raise ValueError(f"classifier_predictions is missing columns: {sorted(missing)}")

    data = classifier_predictions.copy()
    if split is not None:
        if "split" not in data.columns:
            raise ValueError("classifier_predictions has no split column")
        data = data[data["split"].astype(str) == str(split)].copy()
    if data.empty:
        raise ValueError(f"No classifier predictions available for split={split!r}")

    sleep_group = str(groups[1])
    non_sleep_group = str(groups[0])
    data["prob_sleep_quiescent"] = pd.to_numeric(data["prob_more_interaction"], errors="coerce")
    data["prob_former_high_interaction_quiescent_sleep"] = data["prob_sleep_quiescent"]
    data["prob_more_interaction_quiescent_sleep"] = data["prob_sleep_quiescent"]
    data["predicted_sleep_quiescent"] = data["predicted_posture_interaction_group"].astype(str) == sleep_group
    data["predicted_former_high_interaction_quiescent_sleep"] = data["predicted_sleep_quiescent"]
    data["predicted_more_interaction_quiescent_sleep"] = data["predicted_sleep_quiescent"]

    rows = []
    for classifier_bin, group in data.groupby("classifier_bin", sort=True):
        group = group.sort_values("Frame", kind="mergesort") if "Frame" in group.columns else group
        prob_sleep = pd.to_numeric(group["prob_sleep_quiescent"], errors="coerce")
        pred_sleep = group["predicted_sleep_quiescent"].astype(bool)
        true_sleep = group["posture_interaction_group"].astype(str) == sleep_group
        n_interactions = _numeric_or_nan(group, "n_interactions_total")

        row = {
            "classifier_bin": classifier_bin,
            "n_sampled_frames": int(len(group)),
            "sleep_classifier_target_group": sleep_group,
            "sleep_classifier_non_sleep_group": non_sleep_group,
            "mean_prob_sleep_quiescent": float(prob_sleep.mean()),
            "median_prob_sleep_quiescent": float(prob_sleep.median()),
            "mean_prob_former_high_interaction_quiescent_sleep": float(prob_sleep.mean()),
            "median_prob_former_high_interaction_quiescent_sleep": float(prob_sleep.median()),
            "mean_prob_more_interaction_quiescent_sleep": float(prob_sleep.mean()),
            "median_prob_more_interaction_quiescent_sleep": float(prob_sleep.median()),
            "frac_frames_predicted_sleep_quiescent": float(pred_sleep.mean()),
            "frac_frames_true_sleep_quiescent": float(true_sleep.mean()),
            "frac_frames_predicted_former_high_interaction_quiescent_sleep": float(pred_sleep.mean()),
            "frac_frames_true_former_high_interaction_quiescent_sleep": float(true_sleep.mean()),
            "frac_frames_predicted_more_interaction_quiescent_sleep": float(pred_sleep.mean()),
            "frac_frames_true_more_interaction_quiescent_sleep": float(true_sleep.mean()),
            "n_interactions_total": float(n_interactions.median()) if n_interactions.notna().any() else np.nan,
            "has_interaction": bool(n_interactions.fillna(0).median() > 0),
        }
        for col in [
            "split",
            "side",
            "track_id",
            "track_name",
            "track_path",
            "source_ant_number",
            "time_h",
            "bout_id",
            "bout_start_frame",
            "bout_end_frame",
            "bout_duration_seconds",
            "bout_end_time_h",
            "bin_seconds",
            "plot_bin_seconds",
            "speed_smooth_seconds",
        ]:
            if col in group.columns:
                row[col] = group[col].iloc[0]
        rows.append(row)

    out = pd.DataFrame(rows)
    out["is_sleep_quiescent_cluster"] = (
        (out["mean_prob_sleep_quiescent"] >= float(low_score_threshold))
        & (
            out["frac_frames_predicted_sleep_quiescent"]
            >= float(min_predicted_low_fraction)
        )
    )
    out["is_more_interaction_quiescent_sleep"] = out["is_sleep_quiescent_cluster"]
    out["is_former_high_interaction_quiescent_sleep"] = out["is_sleep_quiescent_cluster"]
    out["classifier_quiescent_cluster"] = np.where(
        out["is_sleep_quiescent_cluster"],
        "former_high_interaction_quiescent_sleep",
        "non_sleep_quiescent",
    )
    out["classifier_quiescent_cluster_label"] = np.where(
        out["is_sleep_quiescent_cluster"],
        "former high-interaction quiescent sleep",
        "non-sleep quiescent",
    )
    # Backward-compatible aliases for notebooks that still reference the older
    # low-interaction names. These now refer to the sleep score, whose target is
    # the more-interaction quiescent classifier side.
    out["mean_prob_low_interaction_quiescent"] = out["mean_prob_sleep_quiescent"]
    out["median_prob_low_interaction_quiescent"] = out["median_prob_sleep_quiescent"]
    out["frac_frames_predicted_low_interaction_quiescent"] = out["frac_frames_predicted_sleep_quiescent"]
    out["frac_frames_true_low_interaction_quiescent"] = out["frac_frames_true_sleep_quiescent"]
    out["is_low_interaction_quiescent_cluster"] = out["is_sleep_quiescent_cluster"]
    return out.sort_values(["track_id", "time_h"], kind="mergesort").reset_index(drop=True)


def summarize_low_interaction_quiescent_bins(scored_bins: pd.DataFrame) -> pd.DataFrame:
    if scored_bins.empty:
        return pd.DataFrame()
    return (
        scored_bins.groupby("classifier_quiescent_cluster", sort=False)
        .agg(
            n_bins=("classifier_bin", "size"),
            n_bouts=("classifier_bin", "size"),
            n_tracks=("track_id", "nunique"),
            n_sampled_frames=("n_sampled_frames", "sum"),
            frac_bins_with_interaction=("has_interaction", "mean"),
            frac_bouts_with_interaction=("has_interaction", "mean"),
            mean_interactions_per_bin=("n_interactions_total", "mean"),
            median_interactions_per_bin=("n_interactions_total", "median"),
            mean_interactions_per_bout=("n_interactions_total", "mean"),
            median_interactions_per_bout=("n_interactions_total", "median"),
            mean_prob_sleep_quiescent=("mean_prob_sleep_quiescent", "mean"),
            median_prob_sleep_quiescent=("median_prob_sleep_quiescent", "median"),
            mean_prob_low_interaction_quiescent=("mean_prob_sleep_quiescent", "mean"),
            median_prob_low_interaction_quiescent=("median_prob_sleep_quiescent", "median"),
            mean_prob_former_high_interaction_quiescent_sleep=(
                "mean_prob_former_high_interaction_quiescent_sleep",
                "mean",
            ),
            median_prob_former_high_interaction_quiescent_sleep=(
                "median_prob_former_high_interaction_quiescent_sleep",
                "median",
            ),
            mean_prob_more_interaction_quiescent_sleep=("mean_prob_more_interaction_quiescent_sleep", "mean"),
            median_prob_more_interaction_quiescent_sleep=("median_prob_more_interaction_quiescent_sleep", "median"),
        )
        .reset_index()
    )


def low_interaction_quiescent_bouts_from_bins(
    scored_bins: pd.DataFrame,
    *,
    bin_seconds: float,
) -> pd.DataFrame:
    if scored_bins.empty:
        return pd.DataFrame()
    sleep_col = (
        "is_sleep_quiescent_cluster"
        if "is_sleep_quiescent_cluster" in scored_bins.columns
        else "is_low_interaction_quiescent_cluster"
    )
    sleep_bins = scored_bins[scored_bins[sleep_col]].copy()
    if sleep_bins.empty:
        return pd.DataFrame(
            columns=[
                "track_id",
                "track_name",
                "bout_start_frame",
                "bout_end_frame",
                "bout_start_time_h",
                "bout_end_time_h",
                "bout_duration_seconds",
                "n_bins",
                "n_interactions_total",
                "interaction_rate_per_min",
            ]
        )

    if {"bout_start_frame", "bout_end_frame", "bout_duration_seconds"}.issubset(sleep_bins.columns):
        out = sleep_bins.copy()
        if "time_h" in out.columns:
            out["bout_start_time_h"] = pd.to_numeric(out["time_h"], errors="coerce")
        else:
            out["bout_start_time_h"] = np.nan
        if "bout_end_time_h" not in out.columns:
            out["bout_end_time_h"] = np.nan
        out["interaction_rate_per_min"] = (
            pd.to_numeric(out["n_interactions_total"], errors="coerce").fillna(0)
            / (pd.to_numeric(out["bout_duration_seconds"], errors="coerce") / 60.0)
        )
        return out.sort_values(["track_id", "bout_start_frame"], kind="mergesort").reset_index(drop=True)

    sleep_bins["bin_index"] = np.rint(
        pd.to_numeric(sleep_bins["time_h"], errors="coerce").to_numpy(np.float64)
        * 3600.0
        / float(bin_seconds)
    ).astype(np.int64)

    rows = []
    for _, group in sleep_bins.groupby("track_id", dropna=False, sort=True):
        group = group.sort_values("bin_index", kind="mergesort").reset_index(drop=True)
        if group.empty:
            continue
        breaks = np.flatnonzero(group["bin_index"].diff().fillna(1).to_numpy(np.float64) != 1)
        starts = np.r_[0, breaks]
        stops = np.r_[breaks, len(group)]
        for start, stop in zip(starts, stops):
            bout = group.iloc[start:stop]
            n_bins = int(len(bout))
            duration_seconds = float(n_bins * float(bin_seconds))
            n_interactions = float(pd.to_numeric(bout["n_interactions_total"], errors="coerce").fillna(0).sum())
            row = {
                "track_id": bout["track_id"].iloc[0] if "track_id" in bout.columns else np.nan,
                "track_name": bout["track_name"].iloc[0] if "track_name" in bout.columns else None,
                "side": bout["side"].iloc[0] if "side" in bout.columns else None,
                "bout_start_time_h": float(bout["time_h"].iloc[0]),
                "bout_end_time_h": float(bout["time_h"].iloc[-1] + float(bin_seconds) / 3600.0),
                "bout_duration_seconds": duration_seconds,
                "n_bins": n_bins,
                "n_sampled_frames": int(bout["n_sampled_frames"].sum()),
                "n_interactions_total": n_interactions,
                "interaction_rate_per_min": n_interactions / (duration_seconds / 60.0),
                "frac_bins_with_interaction": float(bout["has_interaction"].mean()),
                "mean_prob_sleep_quiescent": float(bout["mean_prob_sleep_quiescent"].mean()),
                "median_prob_sleep_quiescent": float(bout["median_prob_sleep_quiescent"].median()),
                "mean_prob_low_interaction_quiescent": float(bout["mean_prob_sleep_quiescent"].mean()),
                "median_prob_low_interaction_quiescent": float(bout["median_prob_sleep_quiescent"].median()),
                "mean_prob_former_high_interaction_quiescent_sleep": float(
                    bout["mean_prob_former_high_interaction_quiescent_sleep"].mean()
                ),
                "median_prob_former_high_interaction_quiescent_sleep": float(
                    bout["median_prob_former_high_interaction_quiescent_sleep"].median()
                ),
                "mean_prob_more_interaction_quiescent_sleep": float(
                    bout["mean_prob_more_interaction_quiescent_sleep"].mean()
                ),
                "median_prob_more_interaction_quiescent_sleep": float(
                    bout["median_prob_more_interaction_quiescent_sleep"].median()
                ),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["track_id", "bout_start_time_h"], kind="mergesort").reset_index(drop=True)


def low_interaction_quiescent_cluster_analysis(
    classifier_predictions: pd.DataFrame,
    *,
    groups: tuple[str, str] = ("little_or_no_interaction", "more_interaction"),
    split: Optional[str] = "test",
    bin_seconds: float = 30.0,
    low_score_threshold: float = 0.5,
    min_predicted_low_fraction: float = 0.5,
    correlation_x_col: str = "n_interactions_total",
    correlation_y_col: str = "bout_duration_seconds",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    scored_bins = low_interaction_quiescent_classifier_bins(
        classifier_predictions,
        groups=groups,
        split=split,
        low_score_threshold=low_score_threshold,
        min_predicted_low_fraction=min_predicted_low_fraction,
    )
    bin_summary = summarize_low_interaction_quiescent_bins(scored_bins)
    bouts = low_interaction_quiescent_bouts_from_bins(scored_bins, bin_seconds=bin_seconds)
    corr_stats = _correlation_summary(bouts, x_col=correlation_x_col, y_col=correlation_y_col)
    corr_stats["split"] = split if split is not None else "all"
    corr_stats["sleep_score_threshold"] = float(low_score_threshold)
    corr_stats["min_predicted_sleep_fraction"] = float(min_predicted_low_fraction)
    corr_stats["sleep_classifier_target_group"] = str(groups[1])
    return scored_bins, bin_summary, bouts, corr_stats


def plot_low_interaction_quiescent_cluster_analysis(
    classifier_predictions: pd.DataFrame,
    *,
    groups: tuple[str, str] = ("little_or_no_interaction", "more_interaction"),
    split: Optional[str] = "test",
    bin_seconds: float = 30.0,
    low_score_threshold: float = 0.5,
    min_predicted_low_fraction: float = 0.5,
    correlation_x_col: str = "n_interactions_total",
    correlation_y_col: str = "bout_duration_seconds",
    title: str = "Former high-interaction quiescent sleep classifier state",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    import matplotlib.pyplot as plt

    scored_bins, bin_summary, bouts, corr_stats = low_interaction_quiescent_cluster_analysis(
        classifier_predictions,
        groups=groups,
        split=split,
        bin_seconds=bin_seconds,
        low_score_threshold=low_score_threshold,
        min_predicted_low_fraction=min_predicted_low_fraction,
        correlation_x_col=correlation_x_col,
        correlation_y_col=correlation_y_col,
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    unit_label = "bout" if "bout_start_frame" in scored_bins.columns else "bin"
    order = ["former_high_interaction_quiescent_sleep", "non_sleep_quiescent"]
    colors = {
        "former_high_interaction_quiescent_sleep": "tab:blue",
        "non_sleep_quiescent": "0.45",
    }
    display_labels = {
        "former_high_interaction_quiescent_sleep": "former high-interaction quiescent sleep",
        "non_sleep_quiescent": "non-sleep quiescent",
    }
    for label in order:
        values = scored_bins.loc[
            scored_bins["classifier_quiescent_cluster"] == label,
            "n_interactions_total",
        ].dropna()
        if values.empty:
            continue
        axes[0].hist(
            values,
            bins=30,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=colors[label],
            label=f"{display_labels.get(label, label)} n={len(values):,}",
        )
    axes[0].set_xlabel(f"interaction bouts / {unit_label}")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"{unit_label.title()} interaction-bout counts")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].scatter(
        scored_bins["mean_prob_sleep_quiescent"],
        scored_bins["n_interactions_total"],
        c=np.where(scored_bins["is_sleep_quiescent_cluster"], "tab:blue", "0.45"),
        s=np.maximum(10, scored_bins["n_sampled_frames"].to_numpy(np.float64) * 0.4),
        alpha=0.6,
    )
    axes[1].axvline(float(low_score_threshold), color="0.25", lw=1, ls="--")
    axes[1].set_xlabel("P(former high-interaction quiescent sleep)")
    axes[1].set_ylabel(f"interaction bouts / {unit_label}")
    axes[1].set_title("Classifier score vs interactions")
    axes[1].grid(True, alpha=0.25)

    if not bouts.empty:
        axes[2].scatter(
            bouts[correlation_x_col],
            bouts[correlation_y_col],
            s=32,
            alpha=0.75,
            c="tab:blue",
        )
        axes[2].set_title(
            "Sleep-classified bouts\n"
            f"n={int(corr_stats['n'])}, Spearman r={corr_stats['spearman_r']:.3f}"
        )
    else:
        axes[2].text(0.5, 0.5, "No sleep-classified bouts", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("Sleep-classified bouts")
    axes[2].set_xlabel(correlation_x_col)
    axes[2].set_ylabel(correlation_y_col)
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    return scored_bins, bin_summary, bouts, corr_stats


def posture_state_sample_table(
    posture_samples: pd.DataFrame,
    scored_quiescent_bins: pd.DataFrame,
    *,
    state_order: tuple[str, ...] = ("sleep", "non_sleep_quiescent", "non_quiescent"),
) -> pd.DataFrame:
    if posture_samples.empty:
        return pd.DataFrame()
    required = {"Frame", "posture_interaction_group", "track_path"}
    missing = required.difference(posture_samples.columns)
    if missing:
        raise ValueError(f"posture_samples is missing columns: {sorted(missing)}")
    if scored_quiescent_bins.empty:
        raise ValueError("scored_quiescent_bins is empty; run the quiescent sleep classifier scoring first")
    if "classifier_bin" not in scored_quiescent_bins.columns or "classifier_quiescent_cluster" not in scored_quiescent_bins.columns:
        raise ValueError("scored_quiescent_bins must include classifier_bin and classifier_quiescent_cluster")

    samples = posture_samples.copy()
    samples["Frame"] = pd.to_numeric(samples["Frame"], errors="coerce")
    samples = samples.dropna(subset=["Frame"]).copy()
    samples["Frame"] = samples["Frame"].round().astype(np.int64)
    samples["classifier_bin"] = _classifier_bin_key(samples)

    scored = scored_quiescent_bins[["classifier_bin", "classifier_quiescent_cluster"]].drop_duplicates("classifier_bin")
    samples = samples.merge(scored, on="classifier_bin", how="left", validate="many_to_one")
    samples["posture_state"] = pd.NA

    group = samples["posture_interaction_group"].astype(str)
    cluster = samples["classifier_quiescent_cluster"].astype(str)
    is_non_quiescent = group == "non_quiescent"
    is_quiescent = group.isin(["little_or_no_interaction", "more_interaction"])
    sleep_clusters = {
        "former_high_interaction_quiescent_sleep",
        "more_interaction_quiescent_sleep",
    }
    non_sleep_clusters = {
        "non_sleep_quiescent",
        "low_interaction_quiescent",
        "other_quiescent",
    }
    samples.loc[is_non_quiescent, "posture_state"] = "non_quiescent"
    samples.loc[
        is_quiescent & cluster.isin(sleep_clusters),
        "posture_state",
    ] = "sleep"
    samples.loc[
        is_quiescent & cluster.isin(non_sleep_clusters),
        "posture_state",
    ] = "non_sleep_quiescent"

    samples = samples[samples["posture_state"].isin(state_order)].copy()
    if samples.empty:
        raise ValueError("No posture samples could be assigned to sleep/non-sleep/non-quiescent states")
    return samples.reset_index(drop=True)


def aligned_posture_points_for_states(
    posture_samples: pd.DataFrame,
    scored_quiescent_bins: pd.DataFrame,
    *,
    mm_per_px: float = 0.016,
    body_bodypoint: int = 0,
    head_bodypoint: int = 1,
    bodypoints: tuple[int, ...] = (0, 1, 4, 5, 6, 7, 8, 9),
    max_frames_per_state: int | None = 8000,
    random_state: int = 0,
    state_order: tuple[str, ...] = ("sleep", "non_sleep_quiescent", "non_quiescent"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from analysis import sleep_classifier_features as scf

    state_samples = posture_state_sample_table(
        posture_samples,
        scored_quiescent_bins,
        state_order=state_order,
    )
    if max_frames_per_state is not None:
        rng = np.random.default_rng(int(random_state))
        sampled_parts = []
        for state, group in state_samples.groupby("posture_state", sort=False):
            frames = group[["track_path", "Frame"]].drop_duplicates().reset_index(drop=True)
            if len(frames) > int(max_frames_per_state):
                chosen = rng.choice(len(frames), size=int(max_frames_per_state), replace=False)
                frames = frames.iloc[np.sort(chosen)]
                group = group.merge(frames, on=["track_path", "Frame"], how="inner")
            sampled_parts.append(group)
        state_samples = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
    if state_samples.empty:
        raise ValueError("No sampled posture frames remain for aligned heatmaps")

    rows = []
    needed_bodypoints = tuple(dict.fromkeys([int(body_bodypoint), int(head_bodypoint), *[int(bp) for bp in bodypoints]]))
    metadata_cols = [
        col
        for col in ["track_name", "track_id", "side", "track_row", "source_ant_number"]
        if col in state_samples.columns
    ]
    for track_path, track_samples in state_samples.groupby("track_path", sort=False):
        path = Path(str(track_path))
        if not path.exists():
            raise FileNotFoundError(f"Track path from posture_samples does not exist: {path}")
        frame_state = track_samples[["Frame", "posture_state", *metadata_cols]].drop_duplicates(
            ["Frame", "posture_state"]
        )
        frames = frame_state["Frame"].dropna().astype(np.int64).unique()
        if len(frames) == 0:
            continue

        track_rows = _read_track_rows_for_frames(path, frames)
        wide, _anchor = scf.pose_wide(track_rows)
        if wide.empty:
            continue
        frame_state = frame_state[frame_state["Frame"].isin(wide.index)].copy()
        if frame_state.empty:
            continue

        frame_index = pd.Index(frame_state["Frame"].to_numpy(np.int64))
        body_x, body_y = _xy_or_nan(wide, int(body_bodypoint))
        head_x, head_y = _xy_or_nan(wide, int(head_bodypoint))
        origin_x = body_x.reindex(frame_index).to_numpy(np.float64)
        origin_y = body_y.reindex(frame_index).to_numpy(np.float64)
        axis_x = head_x.reindex(frame_index).to_numpy(np.float64) - origin_x
        axis_y = head_y.reindex(frame_index).to_numpy(np.float64) - origin_y
        axis_length = np.sqrt(axis_x**2 + axis_y**2)
        valid_axis = np.isfinite(origin_x) & np.isfinite(origin_y) & np.isfinite(axis_x) & np.isfinite(axis_y) & (axis_length > 1e-9)
        if not valid_axis.any():
            continue
        theta = np.arctan2(axis_y, axis_x)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        for bodypoint in needed_bodypoints:
            x, y = _xy_or_nan(wide, int(bodypoint))
            px = x.reindex(frame_index).to_numpy(np.float64)
            py = y.reindex(frame_index).to_numpy(np.float64)
            dx = px - origin_x
            dy = py - origin_y
            aligned_x = (dx * cos_t + dy * sin_t) * float(mm_per_px)
            aligned_y = (-dx * sin_t + dy * cos_t) * float(mm_per_px)
            keep = valid_axis & np.isfinite(aligned_x) & np.isfinite(aligned_y)
            if not keep.any():
                continue
            part = frame_state.loc[keep, ["Frame", "posture_state", *metadata_cols]].copy()
            part["track_path"] = str(path)
            part["bodypoint"] = int(bodypoint)
            part["aligned_x_mm"] = aligned_x[keep]
            part["aligned_y_mm"] = aligned_y[keep]
            part["axis_length_mm"] = axis_length[keep] * float(mm_per_px)
            rows.append(part)

    if not rows:
        raise ValueError("No aligned posture points could be computed from the sampled frames")
    aligned_points = pd.concat(rows, ignore_index=True)
    aligned_points["posture_state"] = pd.Categorical(
        aligned_points["posture_state"],
        categories=list(state_order),
        ordered=True,
    )
    state_summary = (
        aligned_points.groupby("posture_state", observed=True)
        .agg(
            n_aligned_points=("aligned_x_mm", "size"),
            n_frames=("Frame", "nunique"),
            n_tracks=("track_path", "nunique"),
            median_axis_length_mm=("axis_length_mm", "median"),
        )
        .reset_index()
    )
    return aligned_points, state_summary


def plot_aligned_posture_state_heatmaps(
    posture_samples: pd.DataFrame,
    scored_quiescent_bins: pd.DataFrame,
    *,
    mm_per_px: float = 0.016,
    body_bodypoint: int = 0,
    head_bodypoint: int = 1,
    bodypoints: tuple[int, ...] = (0, 1, 4, 5, 6, 7, 8, 9),
    skeleton_edges: tuple[tuple[int, int], ...] = ((0, 1), (1, 4), (4, 5), (5, 6), (1, 7), (7, 8), (8, 9)),
    max_frames_per_state: int | None = 8000,
    random_state: int = 0,
    bins: int = 160,
    extent_mm: float | None = None,
    extent_quantile: float = 0.995,
    state_order: tuple[str, ...] = ("sleep", "non_sleep_quiescent", "non_quiescent"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    aligned_points, state_summary = aligned_posture_points_for_states(
        posture_samples,
        scored_quiescent_bins,
        mm_per_px=mm_per_px,
        body_bodypoint=body_bodypoint,
        head_bodypoint=head_bodypoint,
        bodypoints=bodypoints,
        max_frames_per_state=max_frames_per_state,
        random_state=random_state,
        state_order=state_order,
    )
    aligned_points = aligned_points[aligned_points["bodypoint"].isin([int(bp) for bp in bodypoints])].copy()
    if aligned_points.empty:
        raise ValueError("No aligned points remain after applying bodypoint filter")
    aligned_points["plot_x_mm"] = -aligned_points["aligned_y_mm"]
    aligned_points["plot_y_mm"] = aligned_points["aligned_x_mm"]

    bodypoint_summary = (
        aligned_points.groupby(["posture_state", "bodypoint"], observed=True)
        .agg(
            median_aligned_x_mm=("aligned_x_mm", "median"),
            median_aligned_y_mm=("aligned_y_mm", "median"),
            median_plot_x_mm=("plot_x_mm", "median"),
            median_plot_y_mm=("plot_y_mm", "median"),
            n_points=("aligned_x_mm", "size"),
        )
        .reset_index()
    )

    if extent_mm is None:
        xy_abs = np.abs(
            aligned_points[["plot_x_mm", "plot_y_mm"]]
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(np.float64)
            .ravel()
        )
        xy_abs = xy_abs[np.isfinite(xy_abs)]
        extent = float(np.nanquantile(xy_abs, float(extent_quantile))) if len(xy_abs) else 1.0
        extent = max(0.25, extent * 1.08)
    else:
        extent = float(extent_mm)
    edges = np.linspace(-extent, extent, int(bins) + 1)

    histograms: dict[str, np.ndarray] = {}
    positive_values = []
    for state in state_order:
        group = aligned_points[aligned_points["posture_state"].astype(str) == str(state)]
        if group.empty:
            histograms[state] = np.zeros((int(bins), int(bins)), dtype=np.float64)
            continue
        hist, _xedges, _yedges = np.histogram2d(
            group["plot_x_mm"].to_numpy(np.float64),
            group["plot_y_mm"].to_numpy(np.float64),
            bins=[edges, edges],
        )
        prob = hist / hist.sum() if hist.sum() > 0 else hist
        histograms[state] = prob
        positive_values.append(prob[prob > 0])
    positive = np.concatenate(positive_values) if positive_values else np.asarray([], dtype=np.float64)
    min_probability = float(np.nanmin(positive)) if len(positive) else 1e-10
    vmax = float(np.nanpercentile(positive, 99.7)) if len(positive) else 1.0
    vmin = max(min_probability, 1e-10)
    vmax = max(vmax, vmin * 10.0)

    labels = {
        "sleep": "sleep",
        "non_sleep_quiescent": "non-sleep quiescent",
        "non_quiescent": "non-quiescent",
    }
    fig, axes = plt.subplots(
        1,
        len(state_order),
        figsize=(4.6 * len(state_order), 4.8),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for ax, state in zip(axes.ravel(), state_order):
        prob = histograms[state].copy()
        prob[prob <= 0] = vmin
        image = ax.pcolormesh(
            edges,
            edges,
            prob.T,
            shading="auto",
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        med = bodypoint_summary[bodypoint_summary["posture_state"].astype(str) == str(state)].set_index("bodypoint")
        for left, right in skeleton_edges:
            if int(left) in med.index and int(right) in med.index:
                ax.plot(
                    [med.loc[int(left), "median_plot_x_mm"], med.loc[int(right), "median_plot_x_mm"]],
                    [med.loc[int(left), "median_plot_y_mm"], med.loc[int(right), "median_plot_y_mm"]],
                    color="white",
                    lw=1.2,
                    alpha=0.85,
                )
        if not med.empty:
            ax.scatter(
                med["median_plot_x_mm"],
                med["median_plot_y_mm"],
                s=28,
                c="white",
                edgecolors="0.1",
                linewidths=0.5,
                zorder=3,
            )
            for bodypoint, row in med.iterrows():
                ax.text(
                    float(row["median_plot_x_mm"]),
                    float(row["median_plot_y_mm"]),
                    str(int(bodypoint)),
                    color="black",
                    fontsize=7,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=4,
                )
        state_row = state_summary[state_summary["posture_state"].astype(str) == str(state)]
        n_frames = int(state_row["n_frames"].iloc[0]) if not state_row.empty else 0
        ax.set_title(f"{labels.get(state, state)}\nframes={n_frames:,}")
        ax.axhline(0, color="white", lw=0.4, alpha=0.35)
        ax.axvline(0, color="white", lw=0.4, alpha=0.35)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.grid(False)
    axes[0, 0].set_ylabel("display y after 90 deg CCW rotation (mm)")
    for ax in axes.ravel():
        ax.set_xlabel("display x after 90 deg CCW rotation (mm)")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="probability per spatial bin")
    fig.suptitle(
        f"Posture density after aligning bodypoint {body_bodypoint} -> {head_bodypoint} to +x, rotated 90 deg CCW"
    )
    plt.show()
    return aligned_points, bodypoint_summary, state_summary


def get_event_trig_avg(sig, event_inds, backlag, forwardlag):
    """
    Calculate the event-triggered average.

    Parameters:
    - sig (numpy.ndarray): Input signal.
    - event_inds (numpy.ndarray): Indices of events.
    - backlag (int): Backward time lag.
    - forwardlag (int): Forward time lag.

    Returns:
    - ev_avg (numpy.ndarray): Event-triggered average.
    - ev_mat (numpy.ndarray): Event-triggered matrix.

    """
    event_inds = np.round(event_inds).astype(int) 
    if sig.ndim==1:
        sig=np.expand_dims(sig,0)
        
    min_nevents = 1  # minimum number of events where we will even compute a triggered avg

    orig_size = sig.shape

    lags = np.arange(-backlag, forwardlag + 1)

    # get rid of events that happen within the lag-range of the end points
    bad_ids = np.where(event_inds <= backlag)[0]
    if len(bad_ids) > 0:
        print(f'Dropping {len(bad_ids)} early events')
        event_inds = np.delete(event_inds, bad_ids)

    bad_ids = np.where(event_inds >= (orig_size[1] - forwardlag))[0]
    if len(bad_ids) > 0:
        print(f'Dropping {len(bad_ids)} late events')
        event_inds = np.delete(event_inds, bad_ids)

    n_events = len(event_inds)

    # check that we have at least the minimum number of events to work with
    if n_events < min_nevents:
        ev_avg = np.full((orig_size[0], len(lags)), np.nan)
        ev_mat = np.nan
        return ev_avg, ev_mat

    ev_avg = np.zeros((orig_size[0], len(lags)))
    ev_mat = np.zeros((n_events, orig_size[0], len(lags)))

    for i in range(n_events):
        cur_ids = np.arange(event_inds[i] - backlag, event_inds[i] + forwardlag + 1)
        temp_sig = sig[:, cur_ids]
        ev_avg += temp_sig
        ev_mat[i,:, :] = temp_sig

    ev_avg /= n_events

    return np.squeeze(ev_avg), np.squeeze(ev_mat)
