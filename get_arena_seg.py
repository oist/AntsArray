#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# --------------------
# Parameters
# --------------------
INPUT_DIR = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/single_ants/test_12h_vib/20260122-002925")
OUTPUT_DIR = INPUT_DIR
EXT = ".avi"
OUTPUT_SUFFIX = "_arena_seg.png"
WINDOW_MARGIN_PX = 120


def get_screen_size() -> Tuple[int, int]:
    """
    Return screen dimensions in pixels.
    Falls back to 1920x1080 if Tk cannot query the display.
    """
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w = int(root.winfo_screenwidth())
        h = int(root.winfo_screenheight())
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def fit_scale(width: int, height: int, max_w: int, max_h: int) -> float:
    if width <= 0 or height <= 0:
        return 1.0
    return min(max_w / float(width), max_h / float(height), 1.0)


def read_first_frame(video_path: Path) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def annotate_circle(
    frame_bgr: np.ndarray, window_name: str
) -> Optional[Tuple[int, int, int, float]]:
    """
    Return circle (cx, cy, r, scale), where circle values are in display coords.
    Returns None if user skips/quits.
    """
    screen_w, screen_h = get_screen_size()
    max_w = max(200, screen_w - WINDOW_MARGIN_PX)
    max_h = max(200, screen_h - WINDOW_MARGIN_PX)

    h, w = frame_bgr.shape[:2]
    scale = fit_scale(w, h, max_w, max_h)
    disp_w = max(1, int(round(w * scale)))
    disp_h = max(1, int(round(h * scale)))
    disp = cv2.resize(frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    state = {
        "center": None,       # (x, y)
        "radius": 0,
        "drawing": False,
        "has_circle": False,
        "cursor": (disp_w // 2, disp_h // 2),
    }

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata) -> None:
        state["cursor"] = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            state["center"] = (x, y)
            state["radius"] = 0
            state["drawing"] = True
            state["has_circle"] = False
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"] and state["center"] is not None:
            cx, cy = state["center"]
            state["radius"] = int(round(np.hypot(x - cx, y - cy)))
        elif event == cv2.EVENT_LBUTTONUP and state["center"] is not None:
            cx, cy = state["center"]
            state["radius"] = int(round(np.hypot(x - cx, y - cy)))
            state["drawing"] = False
            state["has_circle"] = state["radius"] > 0

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)
    cv2.setMouseCallback(window_name, on_mouse)

    print(f"\n{window_name}")
    print("  - Left click + drag: draw arena circle")
    print("  - r: reset circle")
    print("  - Enter / s / Space: save mask and continue")
    print("  - n: skip this file")
    print("  - q or Esc: quit")

    while True:
        canvas = disp.copy()
        cx_cur, cy_cur = state["cursor"]
        cv2.drawMarker(
            canvas,
            (cx_cur, cy_cur),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=1,
        )

        if state["center"] is not None and (state["drawing"] or state["has_circle"]):
            cx, cy = state["center"]
            r = int(state["radius"])
            if r > 0:
                cv2.circle(canvas, (cx, cy), r, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        cv2.putText(
            canvas,
            "Drag circle | Enter/s/Space: save | r:reset | n:skip | q/Esc:quit",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10, 32, ord("s")):
            if state["has_circle"] and state["center"] is not None and state["radius"] > 0:
                cx, cy = state["center"]
                r = int(state["radius"])
                cv2.destroyWindow(window_name)
                return cx, cy, r, scale
            print("No valid circle yet. Draw one first.")
        elif key == ord("r"):
            state["center"] = None
            state["radius"] = 0
            state["drawing"] = False
            state["has_circle"] = False
        elif key == ord("n"):
            cv2.destroyWindow(window_name)
            return None
        elif key in (27, ord("q")):
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt("User requested quit.")


def save_mask(
    out_path: Path,
    frame_shape: Tuple[int, int, int],
    circle_disp: Tuple[int, int, int],
    scale: float,
) -> None:
    h, w = frame_shape[:2]
    cx_d, cy_d, r_d = circle_disp

    # Convert display coords back to original-resolution coords.
    cx = int(round(cx_d / scale))
    cy = int(round(cy_d / scale))
    r = max(1, int(round(r_d / scale)))

    mask = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, (255, 255, 255), -1, lineType=cv2.LINE_8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), mask)
    if not ok:
        raise RuntimeError(f"Failed to write segmentation PNG: {out_path}")


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    avi_files = sorted(INPUT_DIR.glob(f"*{EXT}"))
    if not avi_files:
        print(f"No {EXT} files found in {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for avi_path in avi_files:
            frame = read_first_frame(avi_path)
            if frame is None:
                print(f"WARNING: could not read first frame of {avi_path.name}")
                continue

            win_name = f"{avi_path.name}"
            circle_out = annotate_circle(frame, win_name)
            if circle_out is None:
                print(f"Skipped {avi_path.name}")
                continue

            cx, cy, r, scale = circle_out
            out_path = OUTPUT_DIR / f"{avi_path.stem}{OUTPUT_SUFFIX}"
            save_mask(out_path, frame.shape, (cx, cy, r), scale)
            print(f"Saved {out_path.name}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
