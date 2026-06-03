#!/usr/bin/env python3
"""Plot X and Y positions over time from one stitched ant parquet file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _track_id_from_path(path: Path) -> int | None:
    match = re.search(r"TrackID_(\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1))


def _resolve_tracks_dir(parquet_path: Path) -> Path | None:
    # Expected layout: <block>/stitched/per_track/<track>.parquet and
    # intermediate chunks at <block>/tracks/<source_file>.
    parents = parquet_path.resolve().parents
    if len(parents) < 3:
        return None
    block_dir = parents[2]
    candidate = block_dir / "tracks"
    return candidate if candidate.is_dir() else None


def _expand_source_group(source_files: list[str], tracks_dir: Path) -> list[str]:
    expanded = set(source_files)
    for source_file in source_files:
        match = re.match(
            r"\d{8}_(?P<time>\d{6})_chunk\d+_(?P<side>left|right)\.parquet$",
            source_file,
        )
        if not match:
            continue
        pattern = f"*_{match.group('time')}_chunk*_{match.group('side')}.parquet"
        expanded.update(p.name for p in tracks_dir.glob(pattern))
    return sorted(expanded)


def _source_frame_offsets(source_files: list[str], tracks_dir: Path) -> dict[str, tuple[int, int]]:
    """Return source_file -> (local_min_frame, global_offset) using stitcher rules."""
    offsets: dict[str, tuple[int, int]] = {}
    running_offset = 0
    for source_file in _expand_source_group(source_files, tracks_dir):
        source_path = tracks_dir / source_file
        if not source_path.exists():
            continue
        frame_df = pd.read_parquet(source_path, columns=["Frame"])
        frame = pd.to_numeric(frame_df["Frame"], errors="coerce").dropna()
        if frame.empty:
            continue
        local_min = int(frame.min())
        local_max = int(frame.max())
        local_len = local_max - local_min + 1
        offsets[source_file] = (local_min, running_offset)
        running_offset += local_len
    return offsets


def _aruco_pkl_for_source(source_file: str, tracks_dir: Path) -> Path | None:
    stem = Path(source_file).stem
    match = re.match(r"(?P<prefix>.+_chunk\d+)_(?P<side>left|right)$", stem)
    if not match:
        return None
    panorama_dir = tracks_dir.parent / "panorama_pkls"
    pattern = f"{match.group('prefix')}_aruco_panorama_x_{match.group('side')}*.pkl"
    matches = sorted(panorama_dir.glob(pattern))
    return matches[0] if matches else None


def _load_aruco_cam_map(source_file: str, tracks_dir: Path, track_id: int) -> dict[int, str]:
    pkl_path = _aruco_pkl_for_source(source_file, tracks_dir)
    if pkl_path is None:
        return {}

    payload = pd.read_pickle(pkl_path)
    det = payload.get("detections") if isinstance(payload, dict) else payload
    if not isinstance(det, pd.DataFrame):
        return {}
    required = {"Frame", "Instance", "Cam"}
    if not required.issubset(det.columns):
        return {}

    d = det[list(required)].copy()
    d["Frame"] = pd.to_numeric(d["Frame"], errors="coerce")
    d["Instance"] = pd.to_numeric(d["Instance"], errors="coerce")
    d["Cam"] = pd.to_numeric(d["Cam"], errors="coerce")
    d = d.dropna(subset=["Frame", "Instance", "Cam"])
    d = d[d["Instance"].astype(int) == int(track_id)]
    if d.empty:
        return {}

    cam_by_frame: dict[int, str] = {}
    for frame, group in d.groupby(d["Frame"].astype(int)):
        cams = sorted({int(v) for v in group["Cam"].to_numpy()})
        cam_by_frame[int(frame)] = ",".join(str(v) for v in cams)
    return cam_by_frame


def add_camera_provenance(df: pd.DataFrame, parquet_path: Path, frame_col: str) -> pd.DataFrame:
    """Add a CameraID column when camera provenance can be recovered."""
    out = df.copy()

    for col in ("CameraID", "Cam", "cam", "camera", "Camera"):
        if col in out.columns:
            out["CameraID"] = out[col].astype("Int64").astype(str)
            return out

    if "source_file" not in out.columns:
        return out

    track_id = None
    if "TrackID" in out.columns and out["TrackID"].notna().any():
        track_id = int(pd.to_numeric(out["TrackID"], errors="coerce").dropna().iloc[0])
    if track_id is None:
        track_id = _track_id_from_path(parquet_path)
    if track_id is None:
        return out

    tracks_dir = _resolve_tracks_dir(parquet_path)
    if tracks_dir is None:
        return out

    source_files = [str(v) for v in out["source_file"].dropna().unique()]
    frame_offsets = _source_frame_offsets(source_files, tracks_dir)
    cam_maps = {
        source_file: _load_aruco_cam_map(source_file, tracks_dir, track_id)
        for source_file in frame_offsets
    }

    cameras: list[str | None] = []
    for row in out[[frame_col, "source_file"]].itertuples(index=False, name=None):
        global_frame, source_file = row
        source_file = str(source_file)
        if source_file not in frame_offsets:
            cameras.append(None)
            continue
        local_min, global_offset = frame_offsets[source_file]
        local_frame = int(global_frame) - int(global_offset) + int(local_min)
        cameras.append(cam_maps.get(source_file, {}).get(local_frame))

    out["CameraID"] = cameras
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open an interactive Qt matplotlib plot for one stitched track parquet."
    )
    parser.add_argument("parquet", type=Path, help="Stitched per-track parquet file.")
    parser.add_argument("--frame_col", default="Frame")
    parser.add_argument("--x_col", default="X")
    parser.add_argument("--y_col", default="Y")
    parser.add_argument(
        "--bodypoint",
        type=int,
        default=0,
        help="Bodypoint value to plot when the file contains multiple bodypoints. Default: 0.",
    )
    parser.add_argument(
        "--all_bodypoints",
        action="store_true",
        help="Plot all bodypoints instead of filtering to one bodypoint.",
    )
    parser.add_argument(
        "--median_frames",
        type=int,
        default=1,
        help="Median-filter window in frames for X/Y before plotting. Use 1 for no filtering.",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt

    df = pd.read_parquet(args.parquet)
    required = {args.frame_col, args.x_col, args.y_col}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    has_bodypoint = "Bodypoint" in df.columns
    if has_bodypoint and not args.all_bodypoints:
        df = df[df["Bodypoint"] == args.bodypoint]
    elif args.bodypoint is not None and not has_bodypoint:
        print("No Bodypoint column found; plotting all rows.")

    df = add_camera_provenance(df, args.parquet, args.frame_col)

    keep_cols = [args.frame_col, args.x_col, args.y_col]
    for col in ("CameraID", "source_file", "TrackID"):
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)
    df = df[keep_cols].copy()
    df[args.frame_col] = pd.to_numeric(df[args.frame_col], errors="coerce")
    df[args.x_col] = pd.to_numeric(df[args.x_col], errors="coerce")
    df[args.y_col] = pd.to_numeric(df[args.y_col], errors="coerce")
    df = df.dropna(subset=[args.frame_col, args.x_col, args.y_col]).sort_values(args.frame_col)
    if df.empty:
        raise SystemExit("No valid X/Y rows to plot")
    if args.median_frames < 1:
        raise SystemExit("--median_frames must be >= 1")
    if args.median_frames > 1:
        df[args.x_col] = (
            df[args.x_col]
            .rolling(args.median_frames, center=True, min_periods=1)
            .median()
        )
        df[args.y_col] = (
            df[args.y_col]
            .rolling(args.median_frames, center=True, min_periods=1)
            .median()
        )

    frames = df[args.frame_col].to_numpy()
    xs = df[args.x_col].to_numpy()
    ys = df[args.y_col].to_numpy()
    camera_ids = df["CameraID"].fillna("?").astype(str).to_numpy() if "CameraID" in df.columns else None
    source_files = df["source_file"].fillna("").astype(str).to_numpy() if "source_file" in df.columns else None

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.25])
    ax_x = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[1, 0], sharex=ax_x)
    ax_xy = fig.add_subplot(gs[:, 1])
    fig.canvas.manager.set_window_title(args.parquet.name)

    ax_x.scatter(frames, xs, s=2, color="tab:blue", alpha=0.75, rasterized=True)
    ax_x.set_ylabel(args.x_col)
    ax_x.grid(True, alpha=0.25)

    ax_y.scatter(frames, ys, s=2, color="tab:orange", alpha=0.75, rasterized=True)
    ax_y.set_ylabel(args.y_col)
    ax_y.set_xlabel(args.frame_col)
    ax_y.grid(True, alpha=0.25)

    if camera_ids is not None and np.any(camera_ids != "?"):
        cam_levels = sorted({cam for cam in camera_ids if cam != "?"})
        cmap = plt.get_cmap("tab20", max(len(cam_levels), 1))
        cam_to_color = {cam: cmap(i) for i, cam in enumerate(cam_levels)}
        point_colors = [cam_to_color.get(cam, (0.35, 0.35, 0.35, 0.35)) for cam in camera_ids]
        ax_xy.scatter(xs, ys, s=2, c=point_colors, alpha=0.45, rasterized=True)
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color=cam_to_color[cam], label=f"Cam {cam}", markersize=5)
            for cam in cam_levels[:12]
        ]
        if handles:
            ax_xy.legend(handles=handles, loc="lower right", fontsize="small", framealpha=0.85)
    else:
        ax_xy.scatter(xs, ys, s=2, color="0.15", alpha=0.35, rasterized=True)
    ax_xy.set_xlabel(args.x_col)
    ax_xy.set_ylabel(args.y_col)
    ax_xy.set_title("XY trajectory: click near a point")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.set_aspect("equal", adjustable="datalim")

    selected_x = ax_x.axvline(frames[0], color="crimson", lw=1.2, alpha=0.9)
    selected_y = ax_y.axvline(frames[0], color="crimson", lw=1.2, alpha=0.9)
    selected_x_point = ax_x.scatter([frames[0]], [xs[0]], s=55, facecolors="none", edgecolors="crimson", linewidths=1.6)
    selected_y_point = ax_y.scatter([frames[0]], [ys[0]], s=55, facecolors="none", edgecolors="crimson", linewidths=1.6)
    selected_xy = ax_xy.scatter([xs[0]], [ys[0]], s=70, facecolors="none", edgecolors="crimson", linewidths=1.8)
    selected_text = ax_xy.text(
        0.02,
        0.98,
        "",
        transform=ax_xy.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )

    x_span = max(float(np.nanmax(xs) - np.nanmin(xs)), 1.0)
    y_span = max(float(np.nanmax(ys) - np.nanmin(ys)), 1.0)

    def set_selected(index: int) -> None:
        frame = frames[index]
        x = xs[index]
        y = ys[index]
        selected_x.set_xdata([frame, frame])
        selected_y.set_xdata([frame, frame])
        selected_x_point.set_offsets([[frame, x]])
        selected_y_point.set_offsets([[frame, y]])
        selected_xy.set_offsets([[x, y]])
        lines = [f"Frame {int(frame)}", f"X {x:.2f}", f"Y {y:.2f}"]
        if camera_ids is not None:
            lines.append(f"Cam {camera_ids[index]}")
        if source_files is not None:
            lines.append(source_files[index])
        selected_text.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.xdata is None:
            return
        if event.inaxes is ax_xy:
            if event.ydata is None:
                return
            dx = (xs - event.xdata) / x_span
            dy = (ys - event.ydata) / y_span
            index = int(np.nanargmin(dx * dx + dy * dy))
        elif event.inaxes in {ax_x, ax_y}:
            index = int(np.nanargmin(np.abs(frames - event.xdata)))
        else:
            return
        set_selected(index)

    set_selected(0)
    fig.canvas.mpl_connect("button_press_event", on_click)

    title = args.parquet.name
    if has_bodypoint and not args.all_bodypoints:
        title += f" | Bodypoint {args.bodypoint}"
    if args.median_frames > 1:
        title += f" | median {args.median_frames} frames"
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
