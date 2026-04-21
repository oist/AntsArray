#!/usr/bin/env python3
"""Per-ID detection census on dense-nest videos.

For each dictionary size, counts how many times each marker ID is detected
across N frames.  Low-count IDs in larger dictionaries that disappear in
smaller dictionaries are likely false positives from weak Hamming separation.

Usage:
    python nn-aruco-detection-test/id_census.py \
        --video "Z:\...\cam04*.avi" "Z:\...\cam05*.avi" \
        --n-frames 200
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter
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


def census_video(video_path: str, n_frames: int, detectors: dict):
    """Run all detectors on the same frames, return per-detector ID counts."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // n_frames)

    results = {name: Counter() for name in detectors}
    frames_read = 0

    for fi in tqdm(range(0, total_frames, step), desc="  frames", leave=False):
        if frames_read >= n_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_read += 1

        for name, det in detectors.items():
            corners, ids, _ = det.detectMarkers(gray)
            if ids is not None:
                for mid in ids.flatten():
                    results[name][int(mid)] += 1

    cap.release()
    return results, frames_read


def main():
    p = argparse.ArgumentParser(description="Per-ID detection census")
    p.add_argument("--video", nargs="+", required=True)
    p.add_argument("--n-frames", type=int, default=200)
    p.add_argument("--output-dir", default="nn-aruco-detection-test/results")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"id_census_log_{timestamp}.txt"
    tee = _TeeLogger(log_path)
    sys.stdout = tee

    print(f"=== ID Census ===")
    print(f"Time: {timestamp}")
    print(f"Frames per video: {args.n_frames}")
    print()

    video_paths = []
    for pattern in args.video:
        video_paths.extend(sorted(glob.glob(pattern)))

    if not video_paths:
        print("[ERROR] No videos found")
        tee.close()
        return

    detectors = {
        "4x4_1000": make_detector(aruco.DICT_4X4_1000),
        "4x4_250": make_detector(aruco.DICT_4X4_250),
        "4x4_100": make_detector(aruco.DICT_4X4_100),
        "4x4_50": make_detector(aruco.DICT_4X4_50),
    }

    all_counts = {name: Counter() for name in detectors}

    for vp in video_paths:
        vname = Path(vp).stem
        print(f"\n--- {vname} ---")
        results, n_read = census_video(vp, args.n_frames, detectors)
        print(f"  Frames read: {n_read}")

        for name in detectors:
            counts = results[name]
            all_counts[name] += counts
            print(f"  {name}: {len(counts)} unique IDs, {sum(counts.values())} total detections")

    # Grand summary: per-ID table sorted by 4x4_1000 count
    print(f"\n{'='*90}")
    print("GRAND CENSUS - All videos combined")
    print(f"{'='*90}")

    c1000 = all_counts["4x4_1000"]
    c250 = all_counts["4x4_250"]
    c100 = all_counts["4x4_100"]
    c50 = all_counts["4x4_50"]

    all_ids = sorted(c1000.keys())

    print(f"\n{'ID':>5} {'d1000':>7} {'d250':>7} {'d100':>7} {'d50':>7}  {'Status'}")
    print("-" * 70)

    real_ids = []
    suspect_ids = []
    ghost_ids = []

    for mid in sorted(all_ids, key=lambda x: -c1000[x]):
        n1000 = c1000[mid]
        n250 = c250.get(mid, 0)
        n100 = c100.get(mid, 0)
        n50 = c50.get(mid, 0)

        # Classify
        if mid >= 250:
            status = "OUT-OF-RANGE(>=250)"
        elif mid >= 100:
            status = "OUT-OF-RANGE(>=100)"
        elif mid >= 50:
            status = "OUT-OF-RANGE(>=50)"
        else:
            status = ""

        # Flag likely FPs: very low count in 1000, absent in smaller dicts
        if n1000 <= 5:
            if n250 == 0:
                status += " GHOST(absent in d250)"
                ghost_ids.append(mid)
            else:
                status += " LOW-COUNT"
                suspect_ids.append(mid)
        elif n250 > 0 and abs(n1000 - n250) / max(n1000, 1) < 0.1:
            real_ids.append(mid)
        elif mid < 250:
            real_ids.append(mid)

        print(f"{mid:>5} {n1000:>7} {n250:>7} {n100:>7} {n50:>7}  {status}")

    # Summary stats
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total unique IDs in d1000: {len(all_ids)}")
    print(f"  IDs with count <= 5 in d1000: {sum(1 for mid in all_ids if c1000[mid] <= 5)}")
    print(f"  IDs with count <= 5 AND absent in d250: {len(ghost_ids)}")

    # Range breakdown
    ranges = [(0, 50), (50, 100), (100, 250), (250, 1000)]
    print(f"\n  ID range breakdown (d1000 detections):")
    for lo, hi in ranges:
        ids_in_range = [mid for mid in all_ids if lo <= mid < hi]
        total_det = sum(c1000[mid] for mid in ids_in_range)
        print(f"    {lo:>4}-{hi-1:>4}: {len(ids_in_range):>4} IDs, {total_det:>8} detections")

    # IDs detected by d1000 but NOT by d250 (pure ghosts from large dict)
    only_in_1000 = set(c1000.keys()) - set(c250.keys())
    if only_in_1000:
        total_ghost_det = sum(c1000[mid] for mid in only_in_1000)
        print(f"\n  IDs detected ONLY by d1000 (absent in d250): {len(only_in_1000)} IDs, {total_ghost_det} detections")
        ghost_sorted = sorted(only_in_1000, key=lambda x: -c1000[x])
        for mid in ghost_sorted[:20]:
            print(f"    ID {mid:>4}: {c1000[mid]} detections")

    # IDs detected by d250 but NOT by d1000 (extra from d250)
    only_in_250 = set(c250.keys()) - set(c1000.keys())
    if only_in_250:
        total_extra = sum(c250[mid] for mid in only_in_250)
        print(f"\n  IDs detected ONLY by d250 (absent in d1000): {len(only_in_250)} IDs, {total_extra} detections")

    print(f"\nLog saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
