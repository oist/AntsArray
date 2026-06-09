#!/usr/bin/env python3
"""Plot X and Y positions over time from one stitched ant parquet file."""

from __future__ import annotations

import argparse
import os
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


def _candidate_tracks_dirs(parquet_path: Path) -> list[Path]:
    tracks_dirs: list[Path] = []
    primary = _resolve_tracks_dir(parquet_path)
    if primary is not None:
        tracks_dirs.append(primary)

    resolved = parquet_path.resolve()
    parts = resolved.parts
    try:
        block_idx = next(i for i, part in enumerate(parts) if re.fullmatch(r"block\d+", part))
    except StopIteration:
        return tracks_dirs

    block_name = parts[block_idx]
    date = parts[block_idx - 1] if block_idx > 0 else None
    if date is None or not re.fullmatch(r"\d{8}", date):
        return tracks_dirs

    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    flash_roots = []
    if user:
        flash_roots.extend(
            [
                Path(f"/home/sam-reiter/flash/ant_tmp/{user}/colony_pipeline"),
                Path(f"/flash/ReiterU/ant_tmp/{user}/colony_pipeline"),
            ]
        )

    for root in flash_roots:
        candidate = root / date / block_name / "tracks"
        if candidate.is_dir() and candidate not in tracks_dirs:
            tracks_dirs.append(candidate)
    for root in (Path("/home/sam-reiter/flash/ant_tmp"), Path("/flash/ReiterU/ant_tmp")):
        if not root.is_dir():
            continue
        for candidate in sorted(root.glob(f"*/colony_pipeline/{date}/{block_name}/tracks")):
            if candidate.is_dir() and candidate not in tracks_dirs:
                tracks_dirs.append(candidate)
    return tracks_dirs


def _read_parquet_from_candidates(
    source_file: str,
    tracks_dirs: list[Path],
    *,
    columns: list[str],
) -> tuple[pd.DataFrame | None, Path | None]:
    errors: list[str] = []
    for tracks_dir in tracks_dirs:
        source_path = tracks_dir / source_file
        if not source_path.exists():
            continue
        try:
            return pd.read_parquet(source_path, columns=columns), source_path
        except Exception as exc:
            errors.append(f"{source_path}: {type(exc).__name__}: {exc}")
    if errors:
        print(f"Warning: no valid parquet source for {source_file}; tried:\n  " + "\n  ".join(errors))
    return None, None


def _expand_source_group(source_files: list[str], tracks_dirs: list[Path]) -> list[str]:
    expanded = set(source_files)
    for source_file in source_files:
        match = re.match(
            r"\d{8}_(?P<time>\d{6})_chunk\d+_(?P<side>left|right)\.parquet$",
            source_file,
        )
        if not match:
            continue
        pattern = f"*_{match.group('time')}_chunk*_{match.group('side')}.parquet"
        for tracks_dir in tracks_dirs:
            expanded.update(p.name for p in tracks_dir.glob(pattern))
    return sorted(expanded)


def _source_frame_offsets(source_files: list[str], tracks_dirs: list[Path]) -> dict[str, tuple[int, int]]:
    """Return source_file -> (local_min_frame, global_offset) using stitcher rules."""
    offsets: dict[str, tuple[int, int]] = {}
    running_offset = 0
    for source_file in _expand_source_group(source_files, tracks_dirs):
        n_frames = _source_num_frames_from_panorama(source_file, tracks_dirs)
        if n_frames is not None and n_frames > 0:
            offsets[source_file] = (0, running_offset)
            running_offset += int(n_frames)
            continue

        frame_df, _source_path = _read_parquet_from_candidates(
            source_file,
            tracks_dirs,
            columns=["Frame"],
        )
        if frame_df is None:
            continue
        frame = pd.to_numeric(frame_df["Frame"], errors="coerce").dropna()
        if frame.empty:
            continue
        local_min = int(frame.min())
        local_max = int(frame.max())
        local_len = local_max - local_min + 1
        offsets[source_file] = (local_min, running_offset)
        running_offset += local_len
    return offsets


def _aruco_pkl_for_source(source_file: str, tracks_dirs: list[Path]) -> Path | None:
    stem = Path(source_file).stem
    match = re.match(r"(?P<prefix>.+_chunk\d+)_(?P<side>left|right)$", stem)
    if not match:
        return None
    pattern = f"{match.group('prefix')}_aruco_panorama_x_{match.group('side')}*.pkl"
    for tracks_dir in tracks_dirs:
        panorama_dir = tracks_dir.parent / "panorama_pkls"
        matches = sorted(panorama_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _source_num_frames_from_panorama(source_file: str, tracks_dirs: list[Path]) -> int | None:
    pkl_path = _aruco_pkl_for_source(source_file, tracks_dirs)
    if pkl_path is None:
        return None
    try:
        payload = pd.read_pickle(pkl_path)
    except Exception as exc:
        print(f"Warning: could not read panorama metadata from {pkl_path}: {type(exc).__name__}: {exc}")
        return None
    if isinstance(payload, dict):
        num_frames = payload.get("num_frames")
        if num_frames is not None:
            try:
                return int(num_frames)
            except Exception:
                return None
    return None


def _camera_series_as_labels(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    labels: list[str | None] = []
    for raw, num in zip(series.to_numpy(), numeric.to_numpy(), strict=False):
        if pd.notna(num):
            labels.append(str(int(num)))
        elif pd.notna(raw):
            labels.append(str(raw))
        else:
            labels.append(None)
    return pd.Series(labels, index=series.index, dtype="object")


def _display_camera_label(label: str, *, zero_based: bool) -> str:
    if zero_based:
        return label
    parts = label.split(",")
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            out.append(str(int(part) + 1))
        except ValueError:
            out.append(part)
    return ",".join(out) if out else label


def _display_camera_labels(labels: np.ndarray, *, zero_based: bool) -> np.ndarray:
    if zero_based:
        return labels
    return np.array([_display_camera_label(str(label), zero_based=False) for label in labels], dtype=object)


def _load_aruco_cam_candidates(
    source_file: str,
    tracks_dirs: list[Path],
    track_id: int,
) -> dict[int, list[tuple[int, float, float]]]:
    pkl_path = _aruco_pkl_for_source(source_file, tracks_dirs)
    if pkl_path is None:
        return {}

    payload = pd.read_pickle(pkl_path)
    det = payload.get("detections") if isinstance(payload, dict) else payload
    if not isinstance(det, pd.DataFrame):
        return {}
    required = {"Frame", "Instance", "Cam", "X", "Y"}
    if not required.issubset(det.columns):
        return {}

    d = det[list(required)].copy()
    d["Frame"] = pd.to_numeric(d["Frame"], errors="coerce")
    d["Instance"] = pd.to_numeric(d["Instance"], errors="coerce")
    d["Cam"] = pd.to_numeric(d["Cam"], errors="coerce")
    d["X"] = pd.to_numeric(d["X"], errors="coerce")
    d["Y"] = pd.to_numeric(d["Y"], errors="coerce")
    d = d.dropna(subset=["Frame", "Instance", "Cam", "X", "Y"])
    d = d[d["Instance"].astype(int) == int(track_id)]
    if d.empty:
        return {}

    candidates_by_frame: dict[int, list[tuple[int, float, float]]] = {}
    for frame, group in d.groupby(d["Frame"].astype(int)):
        candidates_by_frame[int(frame)] = [
            (int(row.Cam), float(row.X), float(row.Y))
            for row in group.itertuples(index=False)
        ]
    return candidates_by_frame


def _xy_for_camera_recovery(row: pd.Series) -> tuple[float, float] | None:
    for x_col, y_col in (("ArucoX", "ArucoY"), ("TrackX", "TrackY"), ("X", "Y")):
        if x_col not in row or y_col not in row:
            continue
        x = pd.to_numeric(row[x_col], errors="coerce")
        y = pd.to_numeric(row[y_col], errors="coerce")
        if pd.notna(x) and pd.notna(y):
            return float(x), float(y)
    return None


def _choose_camera_candidate(
    candidates: list[tuple[int, float, float]],
    xy: tuple[float, float] | None,
) -> tuple[str | None, str | None, float | None]:
    if not candidates:
        return None, None, None

    candidate_label = ",".join(str(cam) for cam in sorted({cam for cam, _x, _y in candidates}))
    if xy is None:
        return candidate_label, candidate_label, None

    x, y = xy
    best_cam, best_x, best_y = min(
        candidates,
        key=lambda item: (item[1] - x) * (item[1] - x) + (item[2] - y) * (item[2] - y),
    )
    distance = float(np.hypot(best_x - x, best_y - y))
    return str(best_cam), candidate_label, distance


def add_camera_provenance(df: pd.DataFrame, parquet_path: Path, frame_col: str) -> pd.DataFrame:
    """Add a CameraID column when camera provenance can be recovered."""
    out = df.copy()

    for col in ("CameraID", "ArucoCam", "Cam", "cam", "camera", "Camera", "SleapCam"):
        if col in out.columns:
            out["CameraID"] = _camera_series_as_labels(out[col])
            out["CameraProvenance"] = col
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

    tracks_dirs = _candidate_tracks_dirs(parquet_path)
    if not tracks_dirs:
        return out

    source_files = [str(v) for v in out["source_file"].dropna().unique()]
    frame_offsets = _source_frame_offsets(source_files, tracks_dirs)
    candidate_maps = {
        source_file: _load_aruco_cam_candidates(source_file, tracks_dirs, track_id)
        for source_file in frame_offsets
    }

    cameras: list[str | None] = []
    camera_candidates: list[str | None] = []
    camera_distances: list[float | None] = []
    for _idx, row in out.iterrows():
        global_frame = row[frame_col]
        source_file = str(row["source_file"])
        if source_file not in frame_offsets:
            cameras.append(None)
            camera_candidates.append(None)
            camera_distances.append(None)
            continue
        local_min, global_offset = frame_offsets[source_file]
        local_frame = int(global_frame) - int(global_offset) + int(local_min)
        camera, candidates, distance = _choose_camera_candidate(
            candidate_maps.get(source_file, {}).get(local_frame, []),
            _xy_for_camera_recovery(row),
        )
        cameras.append(camera)
        camera_candidates.append(candidates)
        camera_distances.append(distance)

    out["CameraID"] = cameras
    out["CameraCandidates"] = camera_candidates
    out["CameraDistance"] = camera_distances
    out["CameraProvenance"] = "aruco_nearest"
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
    parser.add_argument(
        "--camera_provenance",
        action="store_true",
        help="Recover CameraID from chunk/panorama files. Off by default; this reads extra files.",
    )
    parser.add_argument(
        "--cartesian_y",
        action="store_true",
        help="Do not invert the XY trajectory panel. Default uses image/panorama coordinates with Y downward.",
    )
    parser.add_argument(
        "--zero_based_camera_ids",
        action="store_true",
        help="Display raw zero-based camera IDs. Default displays filename/video camera numbering.",
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

    if args.camera_provenance:
        df = add_camera_provenance(df, args.parquet, args.frame_col)

    keep_cols = [args.frame_col, args.x_col, args.y_col]
    for col in ("CameraID", "CameraCandidates", "CameraDistance", "CameraProvenance", "source_file", "TrackID"):
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
    raw_camera_ids = df["CameraID"].fillna("?").astype(str).to_numpy() if "CameraID" in df.columns else None
    camera_ids = (
        _display_camera_labels(raw_camera_ids, zero_based=args.zero_based_camera_ids)
        if raw_camera_ids is not None
        else None
    )
    camera_candidates = (
        _display_camera_labels(
            df["CameraCandidates"].fillna("").astype(str).to_numpy(),
            zero_based=args.zero_based_camera_ids,
        )
        if "CameraCandidates" in df.columns
        else None
    )
    camera_distances = (
        pd.to_numeric(df["CameraDistance"], errors="coerce").to_numpy()
        if "CameraDistance" in df.columns
        else None
    )
    camera_provenance = (
        df["CameraProvenance"].fillna("").astype(str).to_numpy()
        if "CameraProvenance" in df.columns
        else None
    )
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
    if not args.cartesian_y:
        ax_xy.invert_yaxis()

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
            camera_label = "Cam"
            if camera_provenance is not None and camera_provenance[index] == "aruco_nearest":
                camera_label = "Cam nearest ArUco"
            lines.append(f"{camera_label} {camera_ids[index]}")
            if (
                raw_camera_ids is not None
                and not args.zero_based_camera_ids
                and raw_camera_ids[index] != "?"
                and raw_camera_ids[index] != camera_ids[index]
            ):
                lines.append(f"raw Cam {raw_camera_ids[index]}")
            if (
                camera_candidates is not None
                and camera_candidates[index]
                and camera_candidates[index] != camera_ids[index]
            ):
                lines.append(f"Cam candidates {camera_candidates[index]}")
            if camera_distances is not None and np.isfinite(camera_distances[index]):
                lines.append(f"Cam match d {camera_distances[index]:.2f}")
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
