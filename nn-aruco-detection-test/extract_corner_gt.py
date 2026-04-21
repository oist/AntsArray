#!/usr/bin/env python3
"""Extract ArUco crops with OpenCV corner annotations for GT review.

Runs OpenCV ArUco detection on video frames, saves 128x128 crops with
the detected corner positions. The crops are organized for manual review:
keep good ones, delete bad ones, then use the remaining set as GT for
corner refiner training.

Output per detection:
  - {id}_{frame}_{x}_{y}.png  — the 128x128 grayscale crop
  - {id}_{frame}_{x}_{y}.json — corners in crop-local coords + metadata

Usage:
    # Dense nest videos (multi-ant)
    python nn-aruco-detection-test/extract_corner_gt.py `
        --video "Z:\\ReiterU\\Ants\\basler\\20251020_1_30min_vibration\\cam*.avi" `
        --output-dir nn-aruco-detection-test/corner_gt_review `
        --n-frames 200 --sample-every 50

    # Single-ant videos
    python nn-aruco-detection-test/extract_corner_gt.py `
        --video "Z:\\ReiterU\\Ants\\basler\\QRcodes_test\\ARUCO_size_comparison_15_20_25\\20260331_01\\cam01*.avi" `
        --output-dir nn-aruco-detection-test/corner_gt_review `
        --n-frames 200
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from tqdm import tqdm

CROP_SIZE = 128


def extract_from_video(
    video_path: str,
    output_dir: str,
    n_frames: int = 200,
    sample_every: int | None = None,
    crop_size: int = CROP_SIZE,
):
    """Extract crops with corners from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vname = Path(video_path).stem

    # ArUco detector
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshConstant = 3
    params.adaptiveThreshWinSizeMin = 10
    params.adaptiveThreshWinSizeMax = 40
    params.adaptiveThreshWinSizeStep = 10
    params.errorCorrectionRate = 1.0
    detector = aruco.ArucoDetector(aruco_dict, params)

    half = crop_size // 2
    n_extracted = 0

    if sample_every:
        # Sequential read with skipping (fast on network drives)
        pbar = tqdm(total=total, desc=f"  {Path(video_path).name}", leave=False)
        for frame_idx in range(total):
            ret = cap.grab()
            if not ret:
                break
            pbar.update(1)

            if frame_idx % sample_every != 0:
                continue
            if n_extracted >= n_frames * 20:  # rough cap
                break

            ret, frame = cap.retrieve()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is None:
                continue

            for i, marker_id in enumerate(ids.flatten()):
                corner_pts = corners[i][0]  # (4, 2) in full-frame coords
                centre = corner_pts.mean(axis=0)
                cx, cy = int(centre[0]), int(centre[1])

                # Extract crop
                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(fw, x1 + crop_size)
                y2 = min(fh, y1 + crop_size)
                x1 = max(0, x2 - crop_size)
                y1 = max(0, y2 - crop_size)

                crop = gray[y1:y2, x1:x2]
                if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                    continue

                # Corners in crop-local coords
                corners_local = corner_pts.copy()
                corners_local[:, 0] -= x1
                corners_local[:, 1] -= y1

                mid = int(marker_id)
                fname = f"{mid:04d}_{frame_idx:06d}_{cx}_{cy}"

                cv2.imwrite(os.path.join(output_dir, f"{fname}.png"), crop)
                with open(os.path.join(output_dir, f"{fname}.json"), "w") as f:
                    json.dump({
                        "marker_id": mid,
                        "frame_idx": frame_idx,
                        "video": vname,
                        "centre_full": [cx, cy],
                        "crop_offset": [int(x1), int(y1)],
                        "corners_local": corners_local.tolist(),
                        "corners_full": corner_pts.tolist(),
                    }, f)
                n_extracted += 1

        pbar.close()
    else:
        # Uniform sampling via seeking
        indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
        for frame_idx in tqdm(indices, desc=f"  {Path(video_path).name}", leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is None:
                continue

            for i, marker_id in enumerate(ids.flatten()):
                corner_pts = corners[i][0]
                centre = corner_pts.mean(axis=0)
                cx, cy = int(centre[0]), int(centre[1])

                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(fw, x1 + crop_size)
                y2 = min(fh, y1 + crop_size)
                x1 = max(0, x2 - crop_size)
                y1 = max(0, y2 - crop_size)

                crop = gray[y1:y2, x1:x2]
                if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                    continue

                corners_local = corner_pts.copy()
                corners_local[:, 0] -= x1
                corners_local[:, 1] -= y1

                mid = int(marker_id)
                fname = f"{mid:04d}_{frame_idx:06d}_{cx}_{cy}"

                cv2.imwrite(os.path.join(output_dir, f"{fname}.png"), crop)
                with open(os.path.join(output_dir, f"{fname}.json"), "w") as f:
                    json.dump({
                        "marker_id": mid,
                        "frame_idx": frame_idx,
                        "video": vname,
                        "centre_full": [cx, cy],
                        "crop_offset": [int(x1), int(y1)],
                        "corners_local": corners_local.tolist(),
                        "corners_full": corner_pts.tolist(),
                    }, f)
                n_extracted += 1

    cap.release()
    return n_extracted


def generate_review_sheet(
    output_dir: str,
    sheet_path: str,
    cols: int = 10,
    thumb_size: int = 128,
):
    """Generate a contact sheet PNG with corner overlays for quick visual review.

    Each crop is drawn with the 4 detected corners overlaid as colored dots.
    The marker ID is shown below each crop. This allows quick scanning for
    obviously wrong detections (no visible marker, wrong corners, etc.).
    """
    import glob as _glob
    pngs = sorted(_glob.glob(os.path.join(output_dir, "*.png")))
    if not pngs:
        print("  No crops to review")
        return

    # Limit to a manageable number for the sheet
    if len(pngs) > 500:
        import random
        random.seed(42)
        pngs = sorted(random.sample(pngs, 500))

    rows = (len(pngs) + cols - 1) // cols
    margin = 20  # space for ID text below each crop
    sheet_w = cols * thumb_size
    sheet_h = rows * (thumb_size + margin)
    sheet = np.full((sheet_h, sheet_w, 3), 240, dtype=np.uint8)

    corner_colors = [
        (0, 0, 255),    # red = TL
        (0, 255, 0),    # green = TR
        (255, 0, 0),    # blue = BR
        (0, 255, 255),  # yellow = BL
    ]

    for idx, png_path in enumerate(pngs):
        row = idx // cols
        col = idx % cols
        x_off = col * thumb_size
        y_off = row * (thumb_size + margin)

        crop = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if crop is None:
            continue
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        # Load corners
        json_path = png_path.replace(".png", ".json")
        if os.path.isfile(json_path):
            with open(json_path) as f:
                meta = json.load(f)
            for i, (cx, cy) in enumerate(meta["corners_local"]):
                cv2.circle(crop_bgr, (int(cx), int(cy)), 3, corner_colors[i], -1)
            marker_id = meta["marker_id"]
        else:
            marker_id = "?"

        sheet[y_off:y_off + thumb_size, x_off:x_off + thumb_size] = crop_bgr

        # ID text
        cv2.putText(
            sheet, str(marker_id),
            (x_off + 2, y_off + thumb_size + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
        )

    cv2.imwrite(sheet_path, sheet)
    print(f"  Review sheet saved: {sheet_path} ({len(pngs)} crops, {rows}x{cols})")


def main():
    p = argparse.ArgumentParser(description="Extract ArUco crops with corners for GT review")
    p.add_argument("--video", type=str, nargs="+", required=True, help="Video files/globs")
    p.add_argument("--output-dir", default="nn-aruco-detection-test/corner_gt_review")
    p.add_argument("--n-frames", type=int, default=200, help="Frames to sample per video")
    p.add_argument("--sample-every", type=int, default=None,
                   help="Sequential read, process every N-th frame (faster for network drives)")
    p.add_argument("--no-sheet", action="store_true", help="Skip review sheet generation")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_paths = []
    for pattern in args.video:
        video_paths.extend(sorted(glob.glob(pattern)))

    if not video_paths:
        print("[ERROR] No videos found")
        return

    print(f"Found {len(video_paths)} videos")
    total = 0
    for vpath in video_paths:
        n = extract_from_video(vpath, args.output_dir, args.n_frames, args.sample_every)
        total += n
        print(f"  Extracted {n} crops")

    print(f"\nTotal: {total} crops in {args.output_dir}")
    print(f"  Each has .png (crop) + .json (corners + metadata)")

    # Generate review sheet
    if not args.no_sheet:
        print("\nGenerating review sheet...")
        generate_review_sheet(
            args.output_dir,
            os.path.join(args.output_dir, "_review_sheet.png"),
        )

    print(f"\n--- Review instructions ---")
    print(f"1. Open {args.output_dir}/_review_sheet.png for a visual overview")
    print(f"2. Colored dots = detected corners (red=TL, green=TR, blue=BR, yellow=BL)")
    print(f"3. Delete any .png + .json pairs that are clearly wrong:")
    print(f"   - No visible ArUco marker in the crop")
    print(f"   - Corners obviously misplaced")
    print(f"   - Crop shows arena edge / artifact, not an ant")
    print(f"4. Keep crops where the marker is visible and corners look reasonable")
    print(f"5. Then run the fine-tuning with --gt-dir pointing to this directory")


if __name__ == "__main__":
    main()
