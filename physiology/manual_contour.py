#!/usr/bin/env python3
"""Draw per-camera ROI masks from representative video frames."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def enhance_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)


def read_video_frame(video_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    return frame


def draw_mask(frame, title, window_size):
    display = enhance_contrast(frame)
    preview = display.copy()
    contour = []
    drawing = False

    def redraw():
        preview[:] = display
        if len(contour) > 1:
            cv2.polylines(preview, [np.asarray(contour, dtype=np.int32)], False, (255, 0, 0), 2)
        cv2.imshow(title, preview)

    def on_mouse(event, x, y, _flags, _param):
        nonlocal drawing, contour
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            contour = [(x, y)]
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            contour.append((x, y))
            redraw()
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            contour.append((x, y))
            redraw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, window_size[0], window_size[1])
    cv2.setMouseCallback(title, on_mouse)
    redraw()

    print(f"Draw ROI for {title}")
    print("  left-drag: draw contour")
    print("  r: reset")
    print("  s/enter/q: save and continue")
    print("  esc: skip without saving")

    save = False
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("s"), ord("q"), 13):
            save = True
            break
        if key == ord("r"):
            contour = []
            redraw()
        if key == 27:
            break

    cv2.destroyWindow(title)
    if not save or not contour:
        return None

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [np.asarray(contour, dtype=np.int32)], -1, 255, thickness=cv2.FILLED)
    return mask


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Interactively draw one ROI mask per camera for each folder. "
            "Masks are saved as cam0_mask.png and cam1_mask.png by default."
        )
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=[
            "/home/sam-reiter/bucket/ReiterU/electrophysiology/color_experiments_April_2026/2026_05_28*"
        ],
        help="Folder paths or glob patterns containing camera videos.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["cam0", "cam1"],
        help="Camera prefixes to mask.",
    )
    parser.add_argument("--frame-index", type=int, default=0, help="Representative frame to display.")
    parser.add_argument(
        "--output-template",
        default="{camera}_mask.png",
        help="Mask filename template relative to each video folder.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Redraw masks that already exist.")
    parser.add_argument("--window-width", type=int, default=1500)
    parser.add_argument("--window-height", type=int, default=900)
    return parser


def expand_folders(patterns):
    folders = []
    for pattern in patterns:
        matches = sorted(Path(path) for path in glob.glob(pattern))
        folders.extend(path for path in matches if path.is_dir())
    return sorted(dict.fromkeys(folders))


def main():
    args = build_parser().parse_args()
    folders = expand_folders(args.folders)
    if not folders:
        raise SystemExit("No folders matched.")

    for folder in folders:
        for camera in args.cameras:
            videos = sorted(folder.glob(f"{camera}_*.avi"))
            if not videos:
                print(f"Skipping {folder} {camera}: no videos found")
                continue

            output_path = folder / args.output_template.format(camera=camera)
            if output_path.exists() and not args.overwrite:
                print(f"Skipping existing mask: {output_path}")
                continue

            frame = read_video_frame(videos[0], args.frame_index)
            title = f"{folder.name} {camera} frame {args.frame_index}"
            mask = draw_mask(frame, title, (args.window_width, args.window_height))
            if mask is None:
                print(f"Skipped mask: {output_path}")
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), mask)
            print(f"Saved mask: {output_path}")


if __name__ == "__main__":
    main()
