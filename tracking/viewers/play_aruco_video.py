#!/usr/bin/env python3
"""
Play a video and overlay ArUco detections from HDF5 per frame.

- No SLEAP
- No tracking
- Just draw ArUco (X,Y) marker detections from `aruco_tracks`.

Expected ArUco H5 datasets:
  - aruco_tracks: (num_frames, num_markers, 2)
Optional:
  - aruco_confidences: (num_frames, num_markers)

Controls:
  - Press 'q' to quit
"""

from pathlib import Path
import argparse

import cv2
import h5py
import numpy as np
import tkinter as tk


def get_screen_size(margin_w: float = 0.95, margin_h: float = 0.90) -> tuple[int, int]:
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return int(w * margin_w), int(h * margin_h)


def resize_to_screen(img: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)  # never upscale
    if scale >= 1.0:
        return img
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def valid_aruco_points(
    xy: np.ndarray,
    conf: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      marker_ids: (K,) integer marker indices
      points_xy:  (K,2) float points
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected xy shape (num_markers,2), got {xy.shape}")

    valid = np.isfinite(xy).all(axis=1)
    valid &= ~((xy[:, 0] == 0.0) & (xy[:, 1] == 0.0))

    if conf is not None:
        if conf.ndim != 1 or conf.shape[0] != xy.shape[0]:
            raise ValueError(
                f"Expected conf shape ({xy.shape[0]},), got {conf.shape}"
            )
        valid &= np.isfinite(conf) & (conf > 0)

    marker_ids = np.nonzero(valid)[0].astype(int)
    return marker_ids, xy[marker_ids]


def resolve_aruco_h5(video_path: Path) -> Path:
    """
    Resolve ArUco H5 path, supporting both filename variants:
      - <stem>_aruco_tracks_.h5
      - <stem>_aruco_tracks.h5
    Also supports .hdf5 variants.
    """
    stem = video_path.stem

    def candidates_in(base_dir: Path) -> list[Path]:
        direct = [
            base_dir / f"{stem}_aruco_tracks_.h5",
            base_dir / f"{stem}_aruco_tracks.h5",
            base_dir / f"{stem}_aruco_tracks_.hdf5",
            base_dir / f"{stem}_aruco_tracks.hdf5",
        ]
        existing_direct = [p for p in direct if p.exists()]
        if existing_direct:
            return existing_direct

        pattern_hits = sorted(base_dir.glob(f"{stem}*aruco_tracks*.h5")) + sorted(
            base_dir.glob(f"{stem}*aruco_tracks*.hdf5")
        )
        return [p for p in pattern_hits if p.exists()]

    hits = candidates_in(video_path.parent)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise FileExistsError(
            f"Multiple ArUco files match near video: {[p.name for p in hits]}."
        )
    raise FileNotFoundError(
        "Could not auto-find ArUco H5. Expected one of "
        f"'{stem}_aruco_tracks_.h5' or '{stem}_aruco_tracks.h5' "
        f"(or .hdf5) next to {video_path.name}."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True, help="Path to video file")
    ap.add_argument("--start", type=int, default=0, help="Start frame")
    ap.add_argument("--end", type=int, default=None, help="End frame (exclusive). Default: until video ends")
    ap.add_argument("--radius", type=int, default=6, help="Circle radius in pixels")
    ap.add_argument("--thickness", type=int, default=-1, help="Circle thickness (-1 fills)")
    ap.add_argument(
        "--tracks_dataset",
        type=str,
        default="aruco_tracks",
        help="H5 dataset name for ArUco XY tracks",
    )
    ap.add_argument(
        "--conf_dataset",
        type=str,
        default="aruco_confidences",
        help="Optional H5 dataset name for confidences (used if present)",
    )
    args = ap.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(args.video)
    aruco_h5 = resolve_aruco_h5(args.video)
    print(f"[INFO] Using ArUco file: {aruco_h5}")

    screen_w, screen_h = get_screen_size()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    start = max(int(args.start), 0)
    end = int(args.end) if args.end is not None else (
        total_frames if total_frames is not None else 10**12
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    cv2.namedWindow("ArUco overlay", cv2.WINDOW_NORMAL)

    with h5py.File(aruco_h5, "r") as h5f:
        if args.tracks_dataset not in h5f:
            raise KeyError(
                f"H5 file missing dataset '{args.tracks_dataset}': {aruco_h5}"
            )
        tracks_ds = h5f[args.tracks_dataset]
        if tracks_ds.ndim != 3 or tracks_ds.shape[2] != 2:
            raise ValueError(
                f"Dataset '{args.tracks_dataset}' must have shape (frames, markers, 2), "
                f"got {tracks_ds.shape}"
            )

        conf_ds = None
        if args.conf_dataset in h5f:
            cand = h5f[args.conf_dataset]
            if cand.ndim == 2 and cand.shape[:2] == tracks_ds.shape[:2]:
                conf_ds = cand

        aruco_n_frames = int(tracks_ds.shape[0])
        frame_idx = start

        while frame_idx < end:
            ok, frame = cap.read()
            if not ok:
                break

            disp = frame  # draw directly on the frame

            marker_count = 0
            if 0 <= frame_idx < aruco_n_frames:
                xy = np.asarray(tracks_ds[frame_idx], dtype=np.float64)
                conf = (
                    np.asarray(conf_ds[frame_idx], dtype=np.float64)
                    if conf_ds is not None
                    else None
                )

                marker_ids, points = valid_aruco_points(xy, conf)
                marker_count = int(marker_ids.size)
                ids_txt = ""
                if marker_count > 0:
                    ids_txt = ",".join(str(int(mid)) for mid in marker_ids.tolist())

                for marker_id, (x, y) in zip(marker_ids, points):
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue

                    px = int(round(float(x)))
                    py = int(round(float(y)))

                    cv2.circle(
                        disp,
                        (px, py),
                        args.radius,
                        (0, 255, 0),  # green
                        args.thickness,
                    )

                    cv2.putText(
                        disp,
                        str(int(marker_id)),
                        (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                disp,
                f"Frame {frame_idx} | ArUco detections {marker_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            if marker_count > 0:
                shown_ids = ids_txt if len(ids_txt) <= 120 else (ids_txt[:117] + "...")
                cv2.putText(
                    disp,
                    f"IDs: {shown_ids}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            disp_show = resize_to_screen(disp, screen_w, screen_h)
            cv2.imshow("ArUco overlay", disp_show)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
