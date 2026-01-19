import numpy as np
import pandas as pd
from typing import Iterable, Optional



def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    ang1 = np.arctan2(v1[:, 1], v1[:, 0])
    ang2 = np.arctan2(v2[:, 1], v2[:, 0])
    ddeg = (np.degrees(ang2 - ang1) + 360.0) % 360.0

    l1 = np.linalg.norm(v1, axis=1)
    l2 = np.linalg.norm(v2, axis=1)
    bad = (l1 == 0) | (l2 == 0) | np.isnan(ang1) | np.isnan(ang2)
    ddeg[bad] = np.nan
    return ddeg

import numpy as np
import pandas as pd
from typing import Optional, Iterable

import numpy as np
import pandas as pd
from typing import Optional, Iterable

import numpy as np
import pandas as pd
from typing import Optional, Iterable

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
) -> pd.DataFrame:
    """
    Returns per-frame smoothed angles + smoothed speed (+ smoothed XY used for speed) + sleep/wake classification.

    Speed:
      - Reindex X/Y onto dense frame grid
      - Interpolate short gaps (support smoothing)
      - Smooth X/Y (Gaussian preferred)
      - Compute speed from smoothed X/Y
      - Mask speed unless both endpoints of the step exist in RAW X/Y

    Angles:
      - Compute raw angles from pose geometry
      - Circularly smooth angles (smooth sin/cos then atan2), with short-gap interp for support
      - Mask smoothed angles back to NaN where raw angle was missing (no hallucinated angles)

    Sleep classification:
      - Compute framewise sleep evidence from smoothed angles + smoothed speed (NaN-aware)
      - Apply a centered rolling median (majority filter) over sleep evidence (no state machine)
      - Unknown (NaN) stays unknown unless surrounding evidence supports a median value
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
        raw_xy_step_valid = raw_xy_valid[1:] & raw_xy_valid[:-1]

        lim = int(max(0, max_interp_gap))
        if lim > 0:
            x_fill = x.interpolate("linear", limit=lim, limit_direction="both")
            y_fill = y.interpolate("linear", limit=lim, limit_direction="both")
        else:
            x_fill, y_fill = x, y

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
        speed_full[1:][~raw_xy_step_valid] = np.nan

        speed_pix_s = pd.Series(speed_full, index=full_index).reindex(wide.index)

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
        smoothed_angles = {}
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
            smoothed_angles[out_col] = sm.reindex(raw_angles["Frame"]).to_numpy(dtype=float)

        out_track = pd.DataFrame(
            {
                "TrackID": tid,
                "Frame": raw_angles["Frame"].to_numpy(dtype=int),
                **smoothed_angles,
                "speed_pix_s": speed_pix_s.to_numpy(dtype=float),
                "speed_X_s": x_s.reindex(wide.index).to_numpy(dtype=float),
                "speed_Y_s": y_s.reindex(wide.index).to_numpy(dtype=float),
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
