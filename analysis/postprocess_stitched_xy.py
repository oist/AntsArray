#!/usr/bin/env python3
"""Postprocess one stitched track parquet: interpolation, speed, and occupancy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PositionColumns:
    x: str
    y: str


def _parse_bins(value: str) -> tuple[int, int]:
    if "x" in value.lower():
        left, right = value.lower().split("x", 1)
        return int(left), int(right)
    bins = int(value)
    return bins, bins


def _choose_position_columns(df: pd.DataFrame, x_col: str | None, y_col: str | None) -> PositionColumns:
    if x_col is not None or y_col is not None:
        if x_col is None or y_col is None:
            raise SystemExit("--x_col and --y_col must be provided together.")
        missing = {x_col, y_col}.difference(df.columns)
        if missing:
            raise SystemExit(f"Missing requested position columns: {sorted(missing)}")
        return PositionColumns(x=x_col, y=y_col)

    for candidate_x, candidate_y in (("TrackX", "TrackY"), ("ArucoX", "ArucoY"), ("X", "Y")):
        if candidate_x not in df.columns or candidate_y not in df.columns:
            continue
        x = pd.to_numeric(df[candidate_x], errors="coerce")
        y = pd.to_numeric(df[candidate_y], errors="coerce")
        if (x.notna() & y.notna()).any():
            return PositionColumns(x=candidate_x, y=candidate_y)

    raise SystemExit("Could not find usable position columns. Tried TrackX/TrackY, ArucoX/ArucoY, X/Y.")


def _load_position_table(
    parquet_path: Path,
    *,
    frame_col: str,
    x_col: str | None,
    y_col: str | None,
    bodypoint: int | None,
    track_id: int | None,
) -> tuple[pd.DataFrame, PositionColumns]:
    df = pd.read_parquet(parquet_path)
    if frame_col not in df.columns:
        raise SystemExit(f"Missing frame column: {frame_col}")

    if track_id is not None:
        if "TrackID" not in df.columns:
            raise SystemExit("--track_id was provided, but the parquet has no TrackID column.")
        df = df[pd.to_numeric(df["TrackID"], errors="coerce") == int(track_id)]
        if df.empty:
            raise SystemExit(f"No rows found for TrackID {track_id}.")

    pos_cols = _choose_position_columns(df, x_col, y_col)

    # X/Y are skeleton bodypoint coordinates, so filter before reducing to one
    # row per frame. TrackX/TrackY and ArucoX/ArucoY are duplicated across
    # bodypoint rows in the stitched output and will be deduplicated below.
    if bodypoint is not None and pos_cols == PositionColumns("X", "Y") and "Bodypoint" in df.columns:
        df = df[pd.to_numeric(df["Bodypoint"], errors="coerce") == int(bodypoint)]
        if df.empty:
            raise SystemExit(f"No rows found for Bodypoint {bodypoint}.")

    keep_cols = [frame_col, pos_cols.x, pos_cols.y]
    for col in ("TrackID", "Bodypoint", "CameraID", "ArucoCam", "SleapCam", "source_file"):
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)
    df = df[keep_cols].copy()
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")
    df[pos_cols.x] = pd.to_numeric(df[pos_cols.x], errors="coerce")
    df[pos_cols.y] = pd.to_numeric(df[pos_cols.y], errors="coerce")
    df = df.dropna(subset=[frame_col]).sort_values(frame_col, kind="mergesort")
    if df.empty:
        raise SystemExit("No valid frame rows after filtering.")

    df["Frame"] = df[frame_col].round().astype(np.int64)
    df["X_raw_px"] = df[pos_cols.x].astype(float)
    df["Y_raw_px"] = df[pos_cols.y].astype(float)
    df = df.dropna(subset=["X_raw_px", "Y_raw_px"])
    if df.empty:
        raise SystemExit("No finite X/Y rows after filtering.")

    # Collapse repeated bodypoint rows for TrackX/TrackY or ArucoX/ArucoY.
    # Mean is stable for true duplicates and harmless if tiny numeric jitter is
    # present across rows for a frame.
    grouped = df.groupby("Frame", sort=True, as_index=False)
    out = grouped.agg(
        X_raw_px=("X_raw_px", "mean"),
        Y_raw_px=("Y_raw_px", "mean"),
        n_rows=("X_raw_px", "size"),
    )
    for col in ("TrackID", "Bodypoint", "CameraID", "ArucoCam", "SleapCam", "source_file"):
        if col in df.columns:
            out[col] = grouped[col].first()[col]

    return out.sort_values("Frame", kind="mergesort").reset_index(drop=True), pos_cols


def _gap_sizes(frames: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    if len(frames) < 2:
        return out
    for prev_frame, next_frame in zip(frames[:-1], frames[1:], strict=False):
        missing = int(next_frame) - int(prev_frame) - 1
        if missing > 0:
            out[int(next_frame)] = missing
    return out


def interpolate_positions(df: pd.DataFrame, *, max_gap: int | None) -> pd.DataFrame:
    """Return a full-frame table with pixel X/Y linearly interpolated over short gaps."""
    if df.empty:
        return df.copy()
    if max_gap is not None and max_gap < 1:
        raise SystemExit("--max_interpolate_gap must be >= 1, or 0 for unlimited.")

    frames = df["Frame"].to_numpy(np.int64)
    start = int(frames.min())
    stop = int(frames.max())
    full = pd.DataFrame({"Frame": np.arange(start, stop + 1, dtype=np.int64)})
    out = full.merge(df, on="Frame", how="left")
    out["Observed"] = out["X_raw_px"].notna() & out["Y_raw_px"].notna()
    out["X_px"] = out["X_raw_px"]
    out["Y_px"] = out["Y_raw_px"]

    if out["Observed"].sum() <= 1:
        out["Interpolated"] = False
        return out

    x_interp = out["X_px"].interpolate(method="linear", limit_area="inside")
    y_interp = out["Y_px"].interpolate(method="linear", limit_area="inside")
    fill_mask = (~out["Observed"]) & x_interp.notna() & y_interp.notna()

    if max_gap is not None:
        observed_frames = out.loc[out["Observed"], "Frame"].to_numpy(np.int64)
        allowed = np.zeros(len(out), dtype=bool)
        for prev_frame, next_frame in zip(observed_frames[:-1], observed_frames[1:], strict=False):
            missing = int(next_frame) - int(prev_frame) - 1
            if 0 < missing <= max_gap:
                allowed |= (out["Frame"].to_numpy(np.int64) > prev_frame) & (
                    out["Frame"].to_numpy(np.int64) < next_frame
                )
        fill_mask &= allowed

    out.loc[fill_mask, "X_px"] = x_interp[fill_mask]
    out.loc[fill_mask, "Y_px"] = y_interp[fill_mask]
    out["Interpolated"] = fill_mask
    return out


def prepare_positions(df: pd.DataFrame, *, interpolate: bool, max_gap: int | None) -> pd.DataFrame:
    out = df.copy()
    out["Observed"] = True
    out["Interpolated"] = False
    out["X_px"] = out["X_raw_px"]
    out["Y_px"] = out["Y_raw_px"]
    if not interpolate:
        return out.reset_index(drop=True)
    return interpolate_positions(df, max_gap=max_gap).reset_index(drop=True)


def add_metric_units(df: pd.DataFrame, *, fps: float, mm_per_px: float) -> pd.DataFrame:
    if fps <= 0:
        raise SystemExit("--fps must be > 0.")
    if mm_per_px <= 0:
        raise SystemExit("--mm_per_px must be > 0.")
    out = df.sort_values("Frame", kind="mergesort").copy()
    out["TimeS"] = out["Frame"].astype(float) / float(fps)
    out["TimeRelativeS"] = (out["Frame"].astype(float) - float(out["Frame"].min())) / float(fps)
    out["X_mm"] = out["X_px"] * float(mm_per_px)
    out["Y_mm"] = out["Y_px"] * float(mm_per_px)
    return out


def add_speed(df: pd.DataFrame, *, fps: float, mm_per_px: float) -> pd.DataFrame:
    if fps <= 0:
        raise SystemExit("--fps must be > 0.")
    if mm_per_px <= 0:
        raise SystemExit("--mm_per_px must be > 0.")
    out = df.sort_values("Frame", kind="mergesort").copy()
    dx_px = out["X_px"].diff()
    dy_px = out["Y_px"].diff()
    dt_frames = out["Frame"].diff()
    valid = dt_frames > 0
    distance_px = np.sqrt(dx_px * dx_px + dy_px * dy_px)
    out["dFrame"] = dt_frames
    out["dTimeS"] = (dt_frames / float(fps)).where(valid)
    out["DistancePx"] = distance_px.where(valid)
    out["DistanceMm"] = (distance_px * float(mm_per_px)).where(valid)
    out["SpeedPxPerFrame"] = (distance_px / dt_frames).where(valid)
    out["SpeedPxPerSec"] = out["SpeedPxPerFrame"] * float(fps)
    out["SpeedMmPerSec"] = out["SpeedPxPerSec"] * float(mm_per_px)
    return out


def summarize(
    processed: pd.DataFrame,
    observed_frames: Sequence[int],
    *,
    fps: float,
    mm_per_px: float,
) -> list[str]:
    frames = np.asarray(observed_frames, dtype=np.int64)
    gaps = _gap_sizes(frames)
    speed = pd.to_numeric(processed["SpeedMmPerSec"], errors="coerce").dropna()
    observed = int(processed["Observed"].sum()) if "Observed" in processed else len(processed)
    interpolated = int(processed["Interpolated"].sum()) if "Interpolated" in processed else 0
    frame_min = int(processed["Frame"].min())
    frame_max = int(processed["Frame"].max())
    duration_s = (frame_max - frame_min) / float(fps) if frame_max >= frame_min else 0.0

    lines = [
        f"frames: {frame_min}..{frame_max} ({duration_s:.2f} s at {fps:g} fps)",
        f"observed frames: {observed}",
        f"interpolated frames: {interpolated}",
        f"gaps: {len(gaps)} total; max missing frames: {max(gaps.values()) if gaps else 0}",
        f"scale: {mm_per_px:g} mm/px",
    ]
    if not speed.empty:
        lines.extend(
            [
                f"speed mm/s mean: {speed.mean():.6g}",
                f"speed mm/s median: {speed.median():.6g}",
                f"speed mm/s p95: {speed.quantile(0.95):.6g}",
                f"speed mm/s max: {speed.max():.6g}",
            ]
        )
    return lines


def _maybe_smooth_density(hist: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return hist
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        print("Warning: scipy is unavailable; density smoothing skipped.")
        return hist
    return gaussian_filter(hist, sigma=float(sigma))


def plot_postprocessing(
    processed: pd.DataFrame,
    *,
    title: str,
    fps: float,
    density_bins: tuple[int, int],
    density_sigma: float,
    density_log: bool,
    density_include_interpolated: bool,
    cartesian_y: bool,
    output_png: Path | None,
    show: bool,
) -> None:
    import matplotlib

    if output_png is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = processed.dropna(subset=["X_mm", "Y_mm"]).copy()
    if plot_df.empty:
        raise SystemExit("No finite processed X/Y rows to plot.")

    density_df = plot_df if density_include_interpolated else plot_df[plot_df["Observed"]]
    if density_df.empty:
        density_df = plot_df

    x = plot_df["X_mm"].to_numpy(float)
    y = plot_df["Y_mm"].to_numpy(float)
    frames = plot_df["Frame"].to_numpy(np.int64)
    t = plot_df["TimeS"].to_numpy(float)
    speed = pd.to_numeric(plot_df["SpeedMmPerSec"], errors="coerce").to_numpy(float)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0])
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_speed = fig.add_subplot(gs[0, 1])
    ax_density = fig.add_subplot(gs[1, 1])

    sc = ax_xy.scatter(x, y, c=t, s=4, linewidths=0, alpha=0.75, rasterized=True)
    if "Interpolated" in plot_df.columns and plot_df["Interpolated"].any():
        interp = plot_df["Interpolated"].to_numpy(bool)
        ax_xy.scatter(
            x[interp],
            y[interp],
            s=8,
            facecolors="none",
            edgecolors="crimson",
            linewidths=0.6,
            alpha=0.8,
            rasterized=True,
            label="interpolated",
        )
        ax_xy.legend(loc="best", fontsize="small")
    ax_xy.set_xlabel("X (mm)")
    ax_xy.set_ylabel("Y (mm)")
    ax_xy.set_title("Trajectory")
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, alpha=0.2)
    if not cartesian_y:
        ax_xy.invert_yaxis()
    cbar = fig.colorbar(sc, ax=ax_xy, fraction=0.046, pad=0.02)
    cbar.set_label("time (s)")

    ax_speed.plot(plot_df["TimeS"], speed, color="0.1", linewidth=0.9)
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("Speed (mm/s)")
    ax_speed.set_title("Speed")
    ax_speed.grid(True, alpha=0.25)

    hist, xedges, yedges = np.histogram2d(
        density_df["X_mm"].to_numpy(float),
        density_df["Y_mm"].to_numpy(float),
        bins=density_bins,
    )
    hist = _maybe_smooth_density(hist, density_sigma)
    image = np.log1p(hist.T) if density_log else hist.T
    im = ax_density.imshow(
        image,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="equal",
        interpolation="nearest",
        cmap="magma",
    )
    ax_density.set_xlabel("X (mm)")
    ax_density.set_ylabel("Y (mm)")
    ax_density.set_title("2D occupancy density")
    if not cartesian_y:
        ax_density.invert_yaxis()
    density_label = "log1p(count)" if density_log else "count"
    fig.colorbar(im, ax=ax_density, fraction=0.046, pad=0.02, label=density_label)

    fig.suptitle(title)
    fig.tight_layout()

    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=180)
        print(f"Wrote {output_png}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess one stitched per-track parquet: optional interpolation, "
            "speed calculation, and 2D occupancy density visualization in seconds and millimeters."
        )
    )
    parser.add_argument("parquet", type=Path, help="Stitched per-track parquet file.")
    parser.add_argument("--frame_col", default="Frame")
    parser.add_argument("--x_col", default=None, help="Position X column. Default: auto.")
    parser.add_argument("--y_col", default=None, help="Position Y column. Default: auto.")
    parser.add_argument("--track_id", type=int, default=None, help="Filter to one TrackID if the file has many.")
    parser.add_argument(
        "--bodypoint",
        type=int,
        default=0,
        help="Bodypoint used when plotting X/Y skeleton coordinates. Ignored for TrackX/TrackY and ArucoX/ArucoY.",
    )
    parser.add_argument("--no_bodypoint_filter", action="store_true", help="Do not filter X/Y by bodypoint.")
    parser.add_argument("--fps", type=float, default=24.0, help="Frames per second. Default: 24.")
    parser.add_argument("--mm_per_px", type=float, default=0.016, help="Millimeters per pixel. Default: 0.016.")
    parser.add_argument("--interpolate", action="store_true", help="Fill missing frames by linear interpolation.")
    parser.add_argument(
        "--max_interpolate_gap",
        type=int,
        default=120,
        help="Largest missing-frame gap to interpolate. Use 0 for unlimited. Default: 120.",
    )
    parser.add_argument("--out_parquet", type=Path, default=None, help="Write processed table to parquet.")
    parser.add_argument("--out_csv", type=Path, default=None, help="Write processed table to CSV.")
    parser.add_argument("--save_png", type=Path, default=None, help="Write trajectory/speed/density figure to PNG.")
    parser.add_argument("--no_plot", action="store_true", help="Do not make a figure.")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively.")
    parser.add_argument(
        "--density_bins",
        type=_parse_bins,
        default=(160, 160),
        help="Occupancy bins as N or NxM. Default: 160x160.",
    )
    parser.add_argument(
        "--density_sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma for occupancy density. Use 0 for none.",
    )
    parser.add_argument("--density_log", action="store_true", help="Display log1p density counts.")
    parser.add_argument(
        "--density_include_interpolated",
        action="store_true",
        help="Include interpolated frames in occupancy density. Default uses observed frames only.",
    )
    parser.add_argument(
        "--cartesian_y",
        action="store_true",
        help="Do not invert Y in plots. Default uses image/panorama coordinates with Y downward.",
    )
    args = parser.parse_args()

    max_gap = None if args.max_interpolate_gap == 0 else int(args.max_interpolate_gap)
    bodypoint = None if args.no_bodypoint_filter else args.bodypoint
    mm_per_px = float(args.mm_per_px)

    observed, pos_cols = _load_position_table(
        args.parquet,
        frame_col=args.frame_col,
        x_col=args.x_col,
        y_col=args.y_col,
        bodypoint=bodypoint,
        track_id=args.track_id,
    )
    processed = prepare_positions(observed, interpolate=args.interpolate, max_gap=max_gap)
    processed = add_metric_units(processed, fps=args.fps, mm_per_px=mm_per_px)
    processed = add_speed(processed, fps=args.fps, mm_per_px=mm_per_px)
    processed["PositionXCol"] = pos_cols.x
    processed["PositionYCol"] = pos_cols.y

    for line in summarize(
        processed,
        observed["Frame"].to_numpy(np.int64),
        fps=args.fps,
        mm_per_px=mm_per_px,
    ):
        print(line)
    print(f"position columns: {pos_cols.x}/{pos_cols.y}")

    if args.out_parquet is not None:
        args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
        processed.to_parquet(args.out_parquet, index=False)
        print(f"Wrote {args.out_parquet}")
    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")

    if not args.no_plot:
        plot_postprocessing(
            processed,
            title=f"{args.parquet.name} | {pos_cols.x}/{pos_cols.y} | {mm_per_px:g} mm/px",
            fps=args.fps,
            density_bins=args.density_bins,
            density_sigma=args.density_sigma,
            density_log=args.density_log,
            density_include_interpolated=args.density_include_interpolated,
            cartesian_y=args.cartesian_y,
            output_png=args.save_png,
            show=args.show or args.save_png is None,
        )


if __name__ == "__main__":
    main()
