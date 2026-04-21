#!/usr/bin/env python3
"""
Harvest training data from real experiment videos using OpenCV detections.

Two types of crops are extracted:
1. **Positive crops**: Regions where OpenCV confidently detected a marker.
   Organized as ImageFolder (class = marker ID) for classifier training
   and YOLO-format for detector training.
2. **Negative/hard crops**: Regions where SLEAP found an ant but OpenCV
   did NOT detect a marker. These are the "missed" detections that we
   want the NN to recover.

This is the data foundation for Scenarios 2 and 3:
  Scenario 2: Fine-tune a model using OpenCV detections as pseudo-labels
  Scenario 3: Hybrid — use these crops to train a model that finds more tags

All outputs go to nn-aruco-detection-test/training_data/ — no files are
created in the source data directories.

Usage:
    # Harvest from dense-nest chunks (which have ArUco + SLEAP data)
    python nn-aruco-detection-test/harvest_training_data.py \\
        --data-dir "Z:/ReiterU/Ants/basler/20251020_1_30min_vibration/data" \\
        --output-dir nn-aruco-detection-test/training_data \\
        --max-chunks 3

    # Harvest from raw videos (runs OpenCV live)
    python nn-aruco-detection-test/harvest_training_data.py \\
        --video "Z:/ReiterU/Ants/basler/20251020_1_30min_vibration/cam04_cam3_2025-10-20-13-46-00.avi" \\
        --output-dir nn-aruco-detection-test/training_data \\
        --n-frames 500
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aruco_detection.nn_detection.opencv_baseline import OpenCVArucoDetector


def harvest_from_chunk(
    video_path: str,
    aruco_csv_path: str,
    sleap_csv_path: str | None,
    output_dir: str,
    crop_size: int = 128,
    sample_every: int = 10,
    max_crops_per_id: int = 500,
    max_hard_negatives: int = 5000,
    tile_size: int = 1280,
    tile_overlap: float = 0.2,
):
    """Harvest training crops from a single video chunk with existing detections.

    YOLO images are saved as **tiles** (default 1280x1280) from the 4K frames
    so that marker scale at training time matches tiled inference. This avoids
    the scale mismatch where full-frame 640px resizing makes markers too small.

    Classification crops are saved as bbox-style crops (what YOLO would produce
    at inference) rather than tightly-centered crops, so the classifier sees
    the same framing during training and inference.

    Parameters
    ----------
    video_path : str
        Path to the chunk video (.avi).
    aruco_csv_path : str
        Path to the existing ArUco detection CSV (Frame, Instance, X, Y, Confidence).
    sleap_csv_path : str or None
        Path to SLEAP detection CSV. If provided, also harvests hard negatives.
    output_dir : str
        Root output directory for training data.
    crop_size : int
        Size of square crop around each detection for classification.
    sample_every : int
        Sample every N-th frame (to avoid near-duplicate crops).
    max_crops_per_id : int
        Cap crops per marker ID to prevent class imbalance.
    max_hard_negatives : int
        Maximum hard negative crops to save (per chunk).
    tile_size : int
        Tile dimension for YOLO training images (must match inference tile_size).
    tile_overlap : float
        Overlap ratio between adjacent tiles.
    """
    # Load detections
    aruco_df = pd.read_csv(aruco_csv_path)
    aruco_grouped = dict(iter(aruco_df.groupby("Frame")))

    sleap_grouped = {}
    if sleap_csv_path and os.path.isfile(sleap_csv_path):
        sleap_df = pd.read_csv(sleap_csv_path)
        # SLEAP CSV may have different column names; adapt
        if "X" in sleap_df.columns and "Frame" in sleap_df.columns:
            sleap_grouped = dict(iter(sleap_df.groupby("Frame")))

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output dirs
    cls_dir = os.path.join(output_dir, "classification")
    yolo_img_dir = os.path.join(output_dir, "yolo", "images", "train")
    yolo_lbl_dir = os.path.join(output_dir, "yolo", "labels", "train")
    hard_dir = os.path.join(output_dir, "hard_negatives")
    os.makedirs(yolo_img_dir, exist_ok=True)
    os.makedirs(yolo_lbl_dir, exist_ok=True)
    os.makedirs(hard_dir, exist_ok=True)

    vname = Path(video_path).stem
    crops_per_id: dict[int, int] = {}
    yolo_count = 0
    hard_count = 0
    half = crop_size // 2

    # Read sequentially to avoid slow random seeks on network drives.
    # Skip frames we don't need without decoding them.
    pbar = tqdm(total=total_frames, desc=f"  {Path(video_path).name}", leave=False)

    for frame_idx in range(total_frames):
        ret = cap.grab()  # grab without decoding — fast even on network drives
        if not ret:
            break
        pbar.update(1)

        if frame_idx % sample_every != 0:
            continue

        ret, frame = cap.retrieve()  # decode only the frames we need
        if not ret:
            continue

        # --- Collect detections for this frame ---
        aruco_dets = aruco_grouped.get(frame_idx, pd.DataFrame())
        frame_markers: list[tuple[int, float, float]] = []  # (id, x, y)

        if isinstance(aruco_dets, pd.DataFrame) and len(aruco_dets) > 0:
            for _, row in aruco_dets.iterrows():
                marker_id = int(row["Instance"])
                x, y = float(row["X"]), float(row["Y"])
                frame_markers.append((marker_id, x, y))

                # Classification crop — bbox-style with random jitter to match
                # what YOLO bboxes look like at inference (not perfectly centered)
                count = crops_per_id.get(marker_id, 0)
                if count < max_crops_per_id:
                    import random
                    jitter = crop_size // 8
                    jx = random.randint(-jitter, jitter)
                    jy = random.randint(-jitter, jitter)
                    x1 = max(0, int(x - half + jx))
                    y1 = max(0, int(y - half + jy))
                    x2 = min(frame_width, x1 + crop_size)
                    y2 = min(frame_height, y1 + crop_size)

                    crop = frame[y1:y2, x1:x2]
                    if crop.shape[0] > 0 and crop.shape[1] > 0:
                        id_dir = os.path.join(cls_dir, str(marker_id))
                        os.makedirs(id_dir, exist_ok=True)
                        crop_path = os.path.join(id_dir, f"{vname}_f{frame_idx}_{count:04d}.png")
                        cv2.imwrite(crop_path, crop)
                        crops_per_id[marker_id] = count + 1

        # --- Save YOLO tiles (1280x1280 crops from 4K frame) ---
        if frame_markers:
            step = max(1, int(tile_size * (1 - tile_overlap)))
            for ty in range(0, frame_height, step):
                for tx in range(0, frame_width, step):
                    # Snap tile to frame edges
                    tx2 = min(tx + tile_size, frame_width)
                    ty2 = min(ty + tile_size, frame_height)
                    tx1 = max(0, tx2 - tile_size)
                    ty1 = max(0, ty2 - tile_size)
                    tw = tx2 - tx1
                    th = ty2 - ty1

                    # Find markers inside this tile
                    tile_labels: list[str] = []
                    for mid, mx, my in frame_markers:
                        if tx1 <= mx < tx2 and ty1 <= my < ty2:
                            # Bbox in tile-local normalised coords
                            lx = max(0, mx - half - tx1) / tw
                            ly = max(0, my - half - ty1) / th
                            rx = min(tw, mx + half - tx1) / tw
                            ry = min(th, my + half - ty1) / th
                            cx = (lx + rx) / 2
                            cy = (ly + ry) / 2
                            bw = rx - lx
                            bh = ry - ly
                            if bw > 0.005 and bh > 0.005:
                                tile_labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                    if tile_labels:
                        tile_img = frame[ty1:ty2, tx1:tx2]
                        gray_tile = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
                        img_name = f"{vname}_f{frame_idx:06d}_t{tx1}_{ty1}.jpg"
                        cv2.imwrite(
                            os.path.join(yolo_img_dir, img_name),
                            gray_tile,
                            [cv2.IMWRITE_JPEG_QUALITY, 95],
                        )
                        with open(os.path.join(yolo_lbl_dir, img_name.replace(".jpg", ".txt")), "w") as f:
                            f.write("\n".join(tile_labels))
                        yolo_count += 1

        # --- Hard negatives: SLEAP detections where ArUco missed ---
        if hard_count >= max_hard_negatives:
            continue

        sleap_dets = sleap_grouped.get(frame_idx, pd.DataFrame())
        if isinstance(sleap_dets, pd.DataFrame) and len(sleap_dets) > 0:
            aruco_positions = set()
            if isinstance(aruco_dets, pd.DataFrame) and len(aruco_dets) > 0:
                for _, row in aruco_dets.iterrows():
                    aruco_positions.add((round(float(row["X"])), round(float(row["Y"]))))

            for _, srow in sleap_dets.iterrows():
                if hard_count >= max_hard_negatives:
                    break
                sx, sy = float(srow["X"]), float(srow["Y"])
                if np.isnan(sx) or np.isnan(sy):
                    continue

                is_near_aruco = any(
                    np.hypot(sx - ax, sy - ay) < crop_size
                    for ax, ay in aruco_positions
                )
                if not is_near_aruco:
                    x1 = max(0, int(sx - half))
                    y1 = max(0, int(sy - half))
                    x2 = min(frame_width, int(sx + half))
                    y2 = min(frame_height, int(sy + half))

                    crop = frame[y1:y2, x1:x2]
                    if crop.shape[0] > 0 and crop.shape[1] > 0:
                        crop_path = os.path.join(
                            hard_dir, f"{vname}_f{frame_idx}_x{int(sx)}_y{int(sy)}.png"
                        )
                        cv2.imwrite(crop_path, crop)
                        hard_count += 1

    cap.release()
    pbar.close()

    total_cls = sum(crops_per_id.values())
    n_ids = len(crops_per_id)
    print(
        f"  Harvested: {total_cls} classification crops ({n_ids} IDs), "
        f"{yolo_count} YOLO frames, {hard_count} hard negatives"
    )


def harvest_from_raw_video(
    video_path: str,
    output_dir: str,
    crop_size: int = 128,
    n_frames: int = 500,
    tile_size: int = 1280,
    tile_overlap: float = 0.2,
):
    """Harvest from a raw video by running OpenCV detection live.

    Saves YOLO images as tiles (matching inference scale).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    detector = OpenCVArucoDetector()

    vname = Path(video_path).stem
    cls_dir = os.path.join(output_dir, "classification")
    yolo_img_dir = os.path.join(output_dir, "yolo", "images", "train")
    yolo_lbl_dir = os.path.join(output_dir, "yolo", "labels", "train")
    os.makedirs(yolo_img_dir, exist_ok=True)
    os.makedirs(yolo_lbl_dir, exist_ok=True)

    half = crop_size // 2
    crops_per_id: dict[int, int] = {}
    yolo_count = 0

    for idx in tqdm(indices, desc=f"  {Path(video_path).name}", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        dets = detector.detect(frame)
        frame_markers = [(d.marker_id, d.x, d.y) for d in dets]

        for d in dets:
            import random
            count = crops_per_id.get(d.marker_id, 0)
            if count < 500:
                jitter = crop_size // 8
                jx = random.randint(-jitter, jitter)
                jy = random.randint(-jitter, jitter)
                x1 = max(0, int(d.x - half + jx))
                y1 = max(0, int(d.y - half + jy))
                x2 = min(frame_width, x1 + crop_size)
                y2 = min(frame_height, y1 + crop_size)
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] > 0 and crop.shape[1] > 0:
                    id_dir = os.path.join(cls_dir, str(d.marker_id))
                    os.makedirs(id_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(id_dir, f"{vname}_f{int(idx)}_{count:04d}.png"), crop)
                    crops_per_id[d.marker_id] = count + 1

        # Save YOLO tiles
        if frame_markers:
            step = max(1, int(tile_size * (1 - tile_overlap)))
            for ty in range(0, frame_height, step):
                for tx in range(0, frame_width, step):
                    tx2 = min(tx + tile_size, frame_width)
                    ty2 = min(ty + tile_size, frame_height)
                    tx1 = max(0, tx2 - tile_size)
                    ty1 = max(0, ty2 - tile_size)
                    tw = tx2 - tx1
                    th = ty2 - ty1

                    tile_labels: list[str] = []
                    for mid, mx, my in frame_markers:
                        if tx1 <= mx < tx2 and ty1 <= my < ty2:
                            lx = max(0, mx - half - tx1) / tw
                            ly = max(0, my - half - ty1) / th
                            rx = min(tw, mx + half - tx1) / tw
                            ry = min(th, my + half - ty1) / th
                            cx = (lx + rx) / 2
                            cy = (ly + ry) / 2
                            bw = rx - lx
                            bh = ry - ly
                            if bw > 0.005 and bh > 0.005:
                                tile_labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                    if tile_labels:
                        tile_img = frame[ty1:ty2, tx1:tx2]
                        gray_tile = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
                        img_name = f"{vname}_f{int(idx):06d}_t{tx1}_{ty1}.jpg"
                        cv2.imwrite(
                            os.path.join(yolo_img_dir, img_name),
                            gray_tile,
                            [cv2.IMWRITE_JPEG_QUALITY, 95],
                        )
                        with open(os.path.join(yolo_lbl_dir, img_name.replace(".jpg", ".txt")), "w") as f:
                            f.write("\n".join(tile_labels))
                        yolo_count += 1

    cap.release()
    total_cls = sum(crops_per_id.values())
    print(f"  Harvested: {total_cls} crops ({len(crops_per_id)} IDs), {yolo_count} YOLO tiles")


def write_yolo_yaml(output_dir: str):
    """Write data.yaml for the harvested YOLO dataset."""
    yolo_dir = os.path.join(output_dir, "yolo")
    abs_dir = os.path.abspath(yolo_dir).replace("\\", "/")
    yaml_path = os.path.join(yolo_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/train  # split manually or use synthetic val\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  0: marker\n")
    print(f"YOLO data.yaml: {yaml_path}")


def main():
    p = argparse.ArgumentParser(description="Harvest ArUco training data from real videos")

    # Source: processed chunks with existing detections
    p.add_argument(
        "--data-dir", type=str,
        help="Directory with chunk videos + _aruco_detections.csv + _sleap_data.csv",
    )
    p.add_argument("--max-chunks", type=int, default=3, help="Max chunk videos to process")

    # Source: raw unprocessed videos
    p.add_argument("--video", type=str, nargs="*", help="Raw video path(s) (runs OpenCV live)")
    p.add_argument("--n-frames", type=int, default=500, help="Frames to sample per raw video")

    # Output
    p.add_argument(
        "--output-dir", type=str,
        default="nn-aruco-detection-test/training_data",
        help="Output directory for training data",
    )
    p.add_argument("--crop-size", type=int, default=128, help="Crop size in pixels")
    p.add_argument("--sample-every", type=int, default=10, help="Sample every N frames from chunks")
    p.add_argument("--max-hard-negatives", type=int, default=5000, help="Max hard negative crops per chunk")

    args = p.parse_args()

    # Harvest from processed chunks
    if args.data_dir and os.path.isdir(args.data_dir):
        import glob

        chunk_videos = sorted(glob.glob(os.path.join(args.data_dir, "*_???.avi")))
        print(f"Found {len(chunk_videos)} chunk videos in {args.data_dir}")

        for video_path in chunk_videos[: args.max_chunks]:
            stem = Path(video_path).stem  # e.g. cam04_cam3_..._000
            aruco_csv = os.path.join(args.data_dir, f"{stem}_aruco_detections.csv")
            sleap_csv = os.path.join(args.data_dir, f"{stem}_sleap_data.csv")

            if not os.path.isfile(aruco_csv):
                print(f"  [SKIP] No ArUco CSV for {stem}")
                continue

            print(f"\nProcessing chunk: {Path(video_path).name}")
            harvest_from_chunk(
                video_path, aruco_csv,
                sleap_csv if os.path.isfile(sleap_csv) else None,
                args.output_dir, args.crop_size, args.sample_every,
                max_hard_negatives=args.max_hard_negatives,
            )

    # Harvest from raw videos
    if args.video:
        import glob as _glob
        raw_videos = []
        for pattern in args.video:
            raw_videos.extend(_glob.glob(pattern))
        for video_path in raw_videos:
            print(f"\nProcessing raw video: {Path(video_path).name}")
            harvest_from_raw_video(video_path, args.output_dir, args.crop_size, args.n_frames)

    write_yolo_yaml(args.output_dir)

    # Summary
    cls_dir = os.path.join(args.output_dir, "classification")
    if os.path.isdir(cls_dir):
        n_classes = len([d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))])
        print(f"\nClassification dataset: {n_classes} classes in {cls_dir}")

    hard_dir = os.path.join(args.output_dir, "hard_negatives")
    if os.path.isdir(hard_dir):
        n_hard = len([f for f in os.listdir(hard_dir) if f.endswith(".png")])
        print(f"Hard negatives: {n_hard} crops in {hard_dir}")


if __name__ == "__main__":
    main()
