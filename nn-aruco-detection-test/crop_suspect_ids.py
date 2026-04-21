#!/usr/bin/env python3
"""Single-pass ghost/low-count ID detector with crop extraction.

Scans all frames once, records EVERY detection with its crop, then after
the scan identifies low-count IDs and builds a panel.  This avoids the
stochastic problem of scanning twice — rare ghosts may not reappear.

Usage:
    python nn-aruco-detection-test/crop_suspect_ids.py \
        --video "Z:\\...\\cam04*.avi" "Z:\\...\\cam05*.avi" \
        --n-frames 500 --count-threshold 10
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class _TeeLogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(self.log_path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def make_detector(dict_type):
    d = aruco.getPredefinedDictionary(dict_type)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    return aruco.ArucoDetector(d, params)


def single_pass_scan(
    video_paths: list[str],
    n_frames: int,
    crop_size: int = 128,
    max_crops_per_id: int = 10,
) -> tuple[Counter, dict[int, list[tuple[np.ndarray, str, int]]]]:
    """Single pass: detect with d1000, store counts + crops for ALL IDs.

    Keeps up to max_crops_per_id crops per ID (evenly spaced if more are found).
    Returns (counts, crops_dict).
    """
    detector = make_detector(aruco.DICT_4X4_1000)
    counts = Counter()
    # Store all crop info temporarily; we'll downsample later
    all_crops: dict[int, list[tuple[np.ndarray, str, int]]] = defaultdict(list)

    for vp in video_paths:
        vname = Path(vp).stem
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"  [ERROR] Cannot open {vp}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // n_frames)
        frames_read = 0

        print(f"  Scanning {vname}...")
        for fi in tqdm(range(0, total_frames, step), desc=f"    {vname}", leave=False):
            if frames_read >= n_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is None:
                continue

            h, w = frame.shape[:2]
            half = crop_size // 2

            for i, mid_raw in enumerate(ids.flatten()):
                mid = int(mid_raw)
                counts[mid] += 1

                # Keep crops (cap at 3x max to avoid memory blow-up, downsample later)
                if len(all_crops[mid]) >= max_crops_per_id * 3:
                    continue

                corner = corners[i][0]  # (4, 2)
                cx, cy = np.mean(corner, axis=0)
                cx, cy = int(cx), int(cy)

                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(w, cx + half)
                y2 = min(h, cy + half)

                crop = frame[y1:y2, x1:x2].copy()

                # Draw marker outline
                offset = np.array([[x1, y1]], dtype=np.float32)
                local_corners = corner - offset
                pts = local_corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(crop, [pts], True, (0, 255, 0), 2)

                # Pad if near edge
                if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                    padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                    ph, pw = crop.shape[:2]
                    padded[:ph, :pw] = crop
                    crop = padded

                all_crops[mid].append((crop, vname, fi))

        cap.release()

    # Downsample crops to max_crops_per_id (evenly spaced)
    for mid in all_crops:
        if len(all_crops[mid]) > max_crops_per_id:
            indices = np.linspace(0, len(all_crops[mid]) - 1,
                                  max_crops_per_id, dtype=int)
            all_crops[mid] = [all_crops[mid][i] for i in indices]

    return counts, dict(all_crops)


def build_panel(
    crops: dict[int, list[tuple[np.ndarray, str, int]]],
    counts: Counter,
    crop_size: int = 128,
    label_height: int = 30,
) -> np.ndarray:
    """Build a panel image: one row per ID, columns are individual crops."""
    ids_sorted = sorted(crops.keys(), key=lambda x: (-counts[x], x))

    if not ids_sorted:
        return np.zeros((100, 400, 3), dtype=np.uint8)

    max_cols = max(len(crops[mid]) for mid in ids_sorted)
    max_cols = max(max_cols, 1)

    cell_h = crop_size + label_height
    cell_w = crop_size
    id_label_w = 140
    panel_w = id_label_w + max_cols * cell_w
    panel_h = len(ids_sorted) * cell_h

    panel = np.full((panel_h, panel_w, 3), 40, dtype=np.uint8)

    for row, mid in enumerate(ids_sorted):
        y_base = row * cell_h
        total_count = counts[mid]

        # ID label
        cv2.putText(panel, f"ID {mid}", (5, y_base + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, f"({total_count} total)", (5, y_base + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Status color
        if total_count <= 1:
            status, color = "GHOST", (0, 0, 255)
        elif total_count <= 5:
            status, color = "SUSPECT", (0, 165, 255)
        elif total_count <= 10:
            status, color = "LOW", (0, 255, 255)
        else:
            status, color = "BORDERLINE", (100, 255, 100)
        cv2.putText(panel, status, (5, y_base + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Range info
        if mid >= 250:
            rng = ">=250"
        elif mid >= 100:
            rng = "100-249"
        elif mid >= 50:
            rng = "50-99"
        else:
            rng = "0-49"
        cv2.putText(panel, rng, (5, y_base + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Crops
        for col, (crop, vname, fi) in enumerate(crops[mid]):
            x_base = id_label_w + col * cell_w
            panel[y_base:y_base + crop_size, x_base:x_base + crop_size] = crop

            cam = vname.split("_")[0]
            cv2.putText(panel, f"{cam} f{fi}", (x_base + 2, y_base + crop_size + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        cv2.line(panel, (0, y_base + cell_h - 1), (panel_w, y_base + cell_h - 1),
                 (80, 80, 80), 1)

    return panel


def main():
    p = argparse.ArgumentParser(description="Single-pass ghost ID detector with crops")
    p.add_argument("--video", nargs="+", required=True)
    p.add_argument("--n-frames", type=int, default=500)
    p.add_argument("--count-threshold", type=int, default=10,
                   help="IDs with total count <= this are shown in the panel")
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--max-crops-per-id", type=int, default=10)
    p.add_argument("--output-dir", default="nn-aruco-detection-test/results")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"suspect_ids_log_{timestamp}.txt"
    tee = _TeeLogger(log_path)
    sys.stdout = tee

    print(f"=== Single-Pass Ghost ID Detector ===")
    print(f"Time: {timestamp}")
    print(f"Frames per video: {args.n_frames}")
    print(f"Count threshold: <= {args.count_threshold}")
    print()

    video_paths = []
    for pattern in args.video:
        video_paths.extend(sorted(glob.glob(pattern)))

    if not video_paths:
        print("[ERROR] No videos found")
        tee.close()
        return

    print(f"Videos: {len(video_paths)}")

    # Single pass
    counts, all_crops = single_pass_scan(
        video_paths, args.n_frames,
        crop_size=args.crop_size, max_crops_per_id=args.max_crops_per_id,
    )

    # Full census
    print(f"\n{'='*60}")
    print(f"FULL CENSUS ({len(counts)} unique IDs)")
    print(f"{'='*60}")
    print(f"{'ID':>5} {'Count':>7} {'Status'}")
    print("-" * 40)
    for mid in sorted(counts.keys(), key=lambda x: -counts[x]):
        c = counts[mid]
        if c <= 1:
            status = "GHOST"
        elif c <= 5:
            status = "SUSPECT"
        elif c <= args.count_threshold:
            status = "LOW"
        else:
            status = ""
        print(f"{mid:>5} {c:>7}  {status}")

    # Filter to low-count IDs
    suspect_ids = {mid for mid, c in counts.items() if c <= args.count_threshold}
    suspect_crops = {mid: all_crops[mid] for mid in suspect_ids if mid in all_crops}

    print(f"\n{'='*60}")
    print(f"IDs with count <= {args.count_threshold}: {len(suspect_ids)}")
    print(f"{'='*60}")
    for mid in sorted(suspect_ids, key=lambda x: -counts[x]):
        n_crops = len(suspect_crops.get(mid, []))
        print(f"  ID {mid:>4}: {counts[mid]:>3} detections, {n_crops} crops saved")

    # Build panel
    if suspect_crops:
        panel = build_panel(suspect_crops, counts, crop_size=args.crop_size)
        panel_path = out_dir / f"suspect_ids_panel_{timestamp}.png"
        cv2.imwrite(str(panel_path), panel)
        print(f"\nPanel saved: {panel_path}")
        print(f"Panel size: {panel.shape[1]}x{panel.shape[0]}")
    else:
        print("\nNo suspect IDs found!")

    print(f"\nLog saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
