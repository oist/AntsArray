#!/usr/bin/env python3
"""
Play a video and overlay SLEAP detections (sleep bodypoint) per frame.

- No ArUco
- No tracking
- Just draw SLEAP (X,Y) for a chosen bodypoint on each frame.

Expected SLEAP detections columns: Frame, Bodypoint, X, Y
Optional: TrackID, Instance (ignored; we draw all detections for that bodypoint)

Controls:
  - Press 'q' to quit
"""

from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd
import tkinter as tk


def load_sleap_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, engine="pyarrow")
    else:
        df = pd.read_csv(path)

    required = {"Frame", "Bodypoint", "X", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SLEAP detections missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
    df["Bodypoint"] = pd.to_numeric(df["Bodypoint"], errors="coerce")
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["Frame", "Bodypoint", "X", "Y"]).copy()
    df["Frame"] = df["Frame"].astype(int)
    df["Bodypoint"] = df["Bodypoint"].astype(int)
    return df


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True, help="Path to video file")
    ap.add_argument("--sleap", type=Path, required=True, help="Path to SLEAP detections (.parquet or .csv)")
    ap.add_argument("--bodypoint", type=int, default=0, help="Bodypoint index to overlay (sleep bodypoint)")
    ap.add_argument("--start", type=int, default=0, help="Start frame")
    ap.add_argument("--end", type=int, default=None, help="End frame (exclusive). Default: until video ends")
    ap.add_argument("--radius", type=int, default=6, help="Circle radius in pixels")
    ap.add_argument("--thickness", type=int, default=-1, help="Circle thickness (-1 fills)")
    args = ap.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if not args.sleap.exists():
        raise FileNotFoundError(args.sleap)

    screen_w, screen_h = get_screen_size()

    sleap_df = load_sleap_df(args.sleap)
    sleap_bp = sleap_df[sleap_df["Bodypoint"] == int(args.bodypoint)].copy()

    # Group by frame for fast lookup
    grouped = {f: g for f, g in sleap_bp.groupby("Frame")}

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    start = max(int(args.start), 0)
    end = int(args.end) if args.end is not None else (total_frames if total_frames is not None else 10**12)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    # Create a resizable window; actual displayed frame is scaled to screen anyway.
    cv2.namedWindow("SLEAP overlay", cv2.WINDOW_NORMAL)

    frame_idx = start
    while frame_idx < end:
        ok, frame = cap.read()
        if not ok:
            break

        disp = frame  # draw directly on the frame

        # Overlay detections for this frame (all instances / trackIDs, no tracking)
        g = grouped.get(frame_idx)
        if g is not None and not g.empty:
            for x, y in g[["X", "Y"]].itertuples(index=False, name=None):
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(
                        disp,
                        (int(round(x)), int(round(y))),
                        args.radius,
                        (0, 255, 0),  # green
                        args.thickness,
                    )

        # Frame label
        cv2.putText(
            disp,
            f"Frame {frame_idx} | BP {args.bodypoint}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        disp_show = resize_to_screen(disp, screen_w, screen_h)
        cv2.imshow("SLEAP overlay", disp_show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
