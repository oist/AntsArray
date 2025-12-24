
def nan_aware_speed_from_df(
    df: pd.DataFrame,
    bodypoint: int = 0,
    *,
    smooth_window: Optional[int] = None,
) -> np.ndarray:
    """
    Nan-aware per-frame displacement (px/frame), computed from an in-memory
    SLEAP dataframe, with optional NaN-aware smoothing.

    Required columns: Frame, Bodypoint, X, Y

    Parameters
    ----------
    bodypoint : int
        Bodypoint used for locomotion.
    smooth_window : int or None
        Odd window size for NaN-aware moving average smoothing.
        If None, no smoothing is applied.

    Returns
    -------
    speed : np.ndarray
        Array of length (max_frame + 1), indexed by absolute frame number.
        speed[f] is displacement from f-1 -> f when both frames have finite XY.
        speed[0] is NaN.
    """
    if df.empty:
        return np.array([], dtype=float)

    # Filter to the bodypoint used for locomotion
    d0 = df[df["Bodypoint"] == bodypoint].copy()
    if d0.empty:
        return np.array([], dtype=float)

    # Coerce numeric
    d0["Frame"] = pd.to_numeric(d0["Frame"], errors="coerce")
    d0["X"] = pd.to_numeric(d0["X"], errors="coerce")
    d0["Y"] = pd.to_numeric(d0["Y"], errors="coerce")

    d0 = (
        d0.dropna(subset=["Frame"])
          .sort_values("Frame", kind="mergesort")
          .drop_duplicates(subset=["Frame"], keep="first")
    )

    frames = d0["Frame"].astype(int).to_numpy()
    x = d0["X"].to_numpy(float)
    y = d0["Y"].to_numpy(float)

    if frames.size == 0:
        return np.array([], dtype=float)

    max_f = int(frames.max())
    x_full = np.full(max_f + 1, np.nan, dtype=float)
    y_full = np.full(max_f + 1, np.nan, dtype=float)
    x_full[frames] = x
    y_full[frames] = y

    dx = np.diff(x_full)
    dy = np.diff(y_full)

    valid = (
        np.isfinite(x_full[1:]) & np.isfinite(x_full[:-1]) &
        np.isfinite(y_full[1:]) & np.isfinite(y_full[:-1])
    )

    speed = np.full_like(x_full, np.nan, dtype=float)
    speed[1:][valid] = np.sqrt(dx[valid] ** 2 + dy[valid] ** 2)

    # ---------- optional NaN-aware smoothing ----------
    if smooth_window is not None:
        if smooth_window % 2 == 0:
            raise ValueError("smooth_window must be odd")

        mask = np.isfinite(speed)
        speed_filled = np.where(mask, speed, 0.0)
        kernel = np.ones(smooth_window, dtype=float)

        num = np.convolve(speed_filled, kernel, mode="same")
        den = np.convolve(mask.astype(float), kernel, mode="same")

        speed = num / den
        speed[den == 0] = np.nan

    return speed



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
from typing import Iterable, Optional


def classify_sleep_wake_from_sleap(
    sleap_df: pd.DataFrame,
    *,
    fps: float,
    track_ids: Optional[Iterable[int]] = None,
    instance: Optional[int] = None,
    speed_bodypoint: int = 0,
    speed_smooth_window: Optional[int] = None,  # <-- NEW
    # thresholds
    thr_inL: float = 90.0,
    thr_inR: float = 90.0,
    thr_outL: float = 130.0,
    thr_outR: float = 130.0,
    thr_speed_pix_s: float = 22.17,
    # duplicates handling for (Frame,Bodypoint)
    duplicate_agg: str = "mean",  # "mean", "median", "first"
) -> pd.DataFrame:
    """
    Frame-level sleep/wake classification from SLEAP-like data.

    Required columns: Frame, Bodypoint, X, Y
    TrackID optional (defaults to 0).

    Speed is computed using nan_aware_speed_from_df(df, bodypoint=speed_bodypoint,
    smooth_window=speed_smooth_window) (px/frame -> px/s), then joined by Frame.

    If speed_smooth_window is not None, the speed used for classification is smoothed.

    Returns columns:
      TrackID, Frame,
      angle_InL_deg, angle_OutL_deg, angle_InR_deg, angle_OutR_deg,
      speed_pix_s, is_sleep, is_wake
    """
    df = sleap_df.copy()

    # Ensure TrackID exists
    if "TrackID" not in df.columns:
        df["TrackID"] = 0

    required = {"Frame", "TrackID", "Bodypoint", "X", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if track_ids is not None:
        df = df[df["TrackID"].isin(list(track_ids))]

    if instance is not None and "Instance" in df.columns:
        df = df[df["Instance"] == instance]

    if duplicate_agg not in {"mean", "median", "first"}:
        raise ValueError("duplicate_agg must be one of: 'mean', 'median', 'first'")

    # ---------- speed (nan-aware; px/frame -> px/s), keyed by Frame ----------
    sp_pf = nan_aware_speed_from_df(
        df,
        bodypoint=speed_bodypoint,
        smooth_window=speed_smooth_window,  # <-- pass through
    )  # px/frame (optionally smoothed)
    sp_ps = sp_pf * float(fps)  # px/s
    speed_by_frame = pd.Series(sp_ps, index=pd.RangeIndex(len(sp_ps)), name="speed_pix_s")

    # ---------- helpers ----------
    def col_or_nan(wide: pd.DataFrame, xy: str, bp: int) -> pd.Series:
        if (xy, bp) in wide.columns:
            return wide[(xy, bp)]
        return pd.Series(np.nan, index=wide.index)

    def get_vec(wide: pd.DataFrame, bp_from: int, bp_to: int) -> np.ndarray:
        dx = col_or_nan(wide, "X", bp_to) - col_or_nan(wide, "X", bp_from)
        dy = col_or_nan(wide, "Y", bp_to) - col_or_nan(wide, "Y", bp_from)
        return np.column_stack([dx.to_numpy(), dy.to_numpy()])

    # ---------- angles ----------
    rows = []
    for tid, g in df.groupby("TrackID", sort=False):
        g = g.sort_values("Frame", kind="mergesort")

        # Resolve duplicates when pivoting
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
            [wide.columns.get_level_values(0),
             wide.columns.get_level_values(1).astype(int)]
        )

        v_5_4 = get_vec(wide, 5, 4)
        v_5_6 = get_vec(wide, 5, 6)
        v_4_1 = get_vec(wide, 4, 1)
        v_4_5 = get_vec(wide, 4, 5)
        v_8_7 = get_vec(wide, 8, 7)
        v_8_9 = get_vec(wide, 8, 9)
        v_7_1 = get_vec(wide, 7, 1)
        v_7_8 = get_vec(wide, 7, 8)

        angles = pd.DataFrame({
            "TrackID": tid,
            "Frame": wide.index.astype(int),
            "angle_InL_deg": 360.0 - angle_between(v_5_4, v_5_6),
            "angle_OutL_deg": angle_between(v_4_1, v_4_5),
            "angle_InR_deg":  angle_between(v_8_7, v_8_9),
            "angle_OutR_deg": 360.0 - angle_between(v_7_1, v_7_8),
        })

        # attach speed by Frame
        angles = angles.join(speed_by_frame, on="Frame")

        rows.append(angles)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["TrackID", "Frame"], kind="mergesort").reset_index(drop=True)

    # ---------- sleep rule ----------
    is_sleep = (
        (out["angle_InL_deg"]  < thr_inL) &
        (out["angle_InR_deg"]  < thr_inR) &
        (out["angle_OutL_deg"] < thr_outL) &
        (out["angle_OutR_deg"] < thr_outR) &
        (out["speed_pix_s"]    < thr_speed_pix_s)
    )

    out["is_sleep"] = is_sleep.fillna(False)
    out["is_wake"] = ~out["is_sleep"]

    return out