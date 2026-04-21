#!/usr/bin/env python3
"""Test custom ArUco dictionary detection on real camera footage.

Runs detection with both custom dictionaries (A and B) and the standard
DICT_4X4_1000 on filmed footage of printed custom tags. Reports per-camera
detection counts, unique IDs found, and frame-level statistics.

Usage:
    python nn-aruco-detection-test/test_custom_dict_detection.py \
        --video-dir "Z:/ReiterU/Ants/basler/QRcodes_test/custom_ARUCO_4x4_test/20260413" \
        --npz-a nn-aruco-detection-test/results/custom_dicts/custom_4x4_A100_d4_20260410_103938.npz \
        --npz-b nn-aruco-detection-test/results/custom_dicts/custom_4x4_B300_d3_20260410_103938.npz
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

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


def load_custom_dictionary(npz_path: str | Path):
    """Load a custom ArUco dictionary from an NPZ file."""
    data = np.load(str(npz_path), allow_pickle=True)
    d = aruco.Dictionary()
    d.bytesList = data["bytesList"]
    d.markerSize = 4
    d.maxCorrectionBits = int(data["max_correction_bits"])
    n_markers = d.bytesList.shape[0]
    min_d = int(data["min_distance"])
    return d, n_markers, min_d


def make_detector(dictionary: aruco.Dictionary, error_correction_rate: float = 1.0):
    """Create an ArUco detector with production-like parameters."""
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshConstant = 3
    params.adaptiveThreshWinSizeMin = 10
    params.adaptiveThreshWinSizeMax = 40
    params.adaptiveThreshWinSizeStep = 10
    params.errorCorrectionRate = error_correction_rate
    return aruco.ArucoDetector(dictionary, params)


def detect_video(
    video_path: Path,
    detector: aruco.ArucoDetector,
    max_frames: int = 0,
    sample_every: int = 1,
) -> dict:
    """Run detection on a video file.

    Returns dict with:
        total_frames, sampled_frames, frames_with_detections,
        total_detections, id_counts (Counter), detections_per_frame (list)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    id_counts = Counter()
    detections_per_frame = []
    frames_with_det = 0
    total_det = 0
    sampled = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detector.detectMarkers(gray)

            n_det = 0
            if ids is not None:
                n_det = len(ids)
                for mid in ids.flatten():
                    id_counts[int(mid)] += 1

            detections_per_frame.append(n_det)
            total_det += n_det
            if n_det > 0:
                frames_with_det += 1
            sampled += 1

        frame_idx += 1
        if max_frames > 0 and frame_idx >= max_frames:
            break

    cap.release()

    return {
        "total_frames": total_frames,
        "sampled_frames": sampled,
        "frames_with_detections": frames_with_det,
        "total_detections": total_det,
        "id_counts": id_counts,
        "detections_per_frame": detections_per_frame,
    }


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _worker_job(args_tuple):
    """Worker function for multiprocessing.

    Reconstructs the detector inside the worker (OpenCV objects are not picklable).
    """
    video_path, dict_spec, sample_every, max_frames, error_correction_rate = args_tuple

    # Reconstruct dictionary in worker process
    if dict_spec["type"] == "custom":
        data = np.load(dict_spec["npz_path"], allow_pickle=True)
        dictionary = aruco.Dictionary()
        dictionary.bytesList = data["bytesList"]
        dictionary.markerSize = 4
        dictionary.maxCorrectionBits = int(data["max_correction_bits"])
    elif dict_spec["type"] == "predefined":
        dictionary = aruco.getPredefinedDictionary(dict_spec["dict_id"])
    else:
        raise ValueError(f"Unknown dict type: {dict_spec['type']}")

    detector = make_detector(dictionary, error_correction_rate)
    result = detect_video(Path(video_path), detector, max_frames, sample_every)

    cam_name = Path(video_path).stem.split("_")[0]
    return dict_spec["name"], cam_name, result


def main():
    p = argparse.ArgumentParser(description="Test custom dictionary detection on real footage")
    p.add_argument("--video-dir", required=True, help="Directory with camera .avi files")
    p.add_argument("--npz-a", required=True, help="NPZ for dictionary A")
    p.add_argument("--npz-b", required=True, help="NPZ for dictionary B")
    p.add_argument("--sample-every", type=int, default=6,
                   help="Process every Nth frame (default: 6, ~4fps from 24fps)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Max frames per video (0=all)")
    p.add_argument("--error-correction-rate", type=float, default=1.0,
                   help="OpenCV errorCorrectionRate (default: 1.0)")
    p.add_argument("--output-dir", default="nn-aruco-detection-test/results/custom_dicts")
    p.add_argument("--workers", type=int, default=25,
                   help="Number of parallel workers (default: 25)")
    p.add_argument("--cameras", type=str, default=None,
                   help="Comma-separated camera list, e.g. cam01,cam05 (default: all)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"detection_test_log_{timestamp}.txt"
    tee = _TeeLogger(log_path)
    sys.stdout = tee

    print("=" * 70)
    print("Custom ArUco Dictionary Detection Test")
    print("=" * 70)
    print(f"  Time: {timestamp}")
    print(f"  Video dir: {args.video_dir}")
    print(f"  Sample every: {args.sample_every} frames")
    print(f"  Error correction rate: {args.error_correction_rate}")
    print()

    # Find video files
    video_dir = Path(args.video_dir)
    videos = sorted(video_dir.glob("cam*.avi"))
    if args.cameras:
        cam_filter = set(args.cameras.split(","))
        videos = [v for v in videos if v.stem.split("_")[0] in cam_filter]
    print(f"  Found {len(videos)} camera files")

    # Load dictionaries
    print("\nLoading dictionaries...")
    dict_a, n_a, min_d_a = load_custom_dictionary(args.npz_a)
    dict_b, n_b, min_d_b = load_custom_dictionary(args.npz_b)
    dict_1000 = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

    dict_specs = [
        {"name": f"Custom_A (n={n_a}, d={min_d_a})", "type": "custom", "npz_path": str(args.npz_a)},
        {"name": f"Custom_B (n={n_b}, d={min_d_b})", "type": "custom", "npz_path": str(args.npz_b)},
        {"name": "DICT_4X4_1000", "type": "predefined", "dict_id": aruco.DICT_4X4_1000},
    ]
    dict_names = [ds["name"] for ds in dict_specs]

    print(f"  Dict A: {n_a} markers, min_d={min_d_a}")
    print(f"  Dict B: {n_b} markers, min_d={min_d_b}")
    print(f"  DICT_4X4_1000: 1000 markers, min_d=2")

    # Build job list: (video_path, dict_spec, sample_every, max_frames, ecr)
    jobs = []
    for video_path in videos:
        for ds in dict_specs:
            jobs.append((
                str(video_path), ds,
                args.sample_every, args.max_frames, args.error_correction_rate,
            ))

    n_workers = min(args.workers, len(jobs))
    print(f"\n  Total jobs: {len(jobs)} ({len(videos)} cameras x {len(dict_specs)} dicts)")
    print(f"  Workers: {n_workers}")

    # Run in parallel
    print(f"\n{'='*70}")
    print(f"Running detection (parallel, {n_workers} workers)...")
    print(f"{'='*70}")
    t0 = time.time()

    all_results = {dn: {} for dn in dict_names}

    with mp.Pool(processes=n_workers) as pool:
        for i, (dict_name, cam_name, result) in enumerate(
            pool.imap_unordered(_worker_job, jobs)
        ):
            all_results[dict_name][cam_name] = result

            if "error" in result:
                print(f"  [{i+1}/{len(jobs)}] {cam_name} / {dict_name.split('(')[0].strip()}: "
                      f"ERROR - {result['error']}")
                continue

            n_unique = len(result["id_counts"])
            det_rate = (result["frames_with_detections"] / max(result["sampled_frames"], 1)) * 100
            avg_det = result["total_detections"] / max(result["sampled_frames"], 1)

            short_dict = dict_name.split("(")[0].strip()
            print(f"  [{i+1:>2}/{len(jobs)}] {cam_name} / {short_dict}: "
                  f"{result['total_detections']:>5} det, {n_unique:>3} IDs, "
                  f"{det_rate:.0f}% frames, avg {avg_det:.1f}/frame")

    total_elapsed = time.time() - t0
    print(f"\n  All detection done in {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f} min, {len(jobs)/total_elapsed:.1f} jobs/s)")

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")

    # Table header
    cam_names = sorted(set(
        cam for d in all_results.values() for cam in d.keys()
    ))

    print(f"\n{'Camera':<8}", end="")
    for dict_name in dict_names:
        short = dict_name.split("(")[0].strip()
        print(f"  {'Det':>6} {'IDs':>4} {'%frm':>5}", end="")
    print()

    print(f"{'':8}", end="")
    for dict_name in dict_names:
        short = dict_name.split("(")[0].strip()
        print(f"  {short:>17}", end="")
    print()
    print("-" * (8 + len(dict_names) * 19))

    totals = {dn: {"det": 0, "ids": set(), "frames_w_det": 0, "frames": 0}
              for dn in dict_names}

    for cam in cam_names:
        print(f"{cam:<8}", end="")
        for dict_name in dict_names:
            r = all_results[dict_name].get(cam)
            if r is None or "error" in r:
                print(f"  {'ERR':>6} {'':>4} {'':>5}", end="")
            else:
                n_det = r["total_detections"]
                n_ids = len(r["id_counts"])
                pct = (r["frames_with_detections"] / max(r["sampled_frames"], 1)) * 100
                print(f"  {n_det:>6} {n_ids:>4} {pct:>4.0f}%", end="")

                totals[dict_name]["det"] += n_det
                totals[dict_name]["ids"] |= set(r["id_counts"].keys())
                totals[dict_name]["frames_w_det"] += r["frames_with_detections"]
                totals[dict_name]["frames"] += r["sampled_frames"]
        print()

    print("-" * (8 + len(dict_names) * 19))
    print(f"{'TOTAL':<8}", end="")
    for dict_name in dict_names:
        t = totals[dict_name]
        pct = (t["frames_w_det"] / max(t["frames"], 1)) * 100
        print(f"  {t['det']:>6} {len(t['ids']):>4} {pct:>4.0f}%", end="")
    print()

    # Unique IDs found per dictionary
    print(f"\n{'='*70}")
    print("UNIQUE IDs DETECTED")
    print(f"{'='*70}")
    for dict_name in dict_names:
        t = totals[dict_name]
        ids_sorted = sorted(t["ids"])
        print(f"\n  {dict_name}:")
        print(f"    Count: {len(ids_sorted)}")
        if len(ids_sorted) <= 100:
            print(f"    IDs: {ids_sorted}")
        else:
            print(f"    IDs (first 50): {ids_sorted[:50]}")
            print(f"    IDs (last 50):  {ids_sorted[-50:]}")

    # Cross-check: IDs found by custom dicts that map to DICT_4X4_1000 IDs
    print(f"\n{'='*70}")
    print("CROSS-DICTIONARY ANALYSIS")
    print(f"{'='*70}")
    d1000_ids = totals["DICT_4X4_1000"]["ids"]
    for dict_name in dict_names[:2]:  # Custom A and B only
        custom_ids = totals[dict_name]["ids"]
        # Aggregate per-ID counts across all cameras
        all_id_counts = Counter()
        for cam in cam_names:
            r = all_results[dict_name].get(cam)
            if r and "error" not in r:
                all_id_counts.update(r["id_counts"])
        top_20 = all_id_counts.most_common(20)
        print(f"\n  {dict_name}:")
        print(f"    Total detections: {sum(all_id_counts.values()):,}")
        print(f"    Unique IDs detected: {len(custom_ids)}")
        print(f"    DICT_4X4_1000 unique IDs: {len(d1000_ids)}")
        print(f"    Top 20 IDs: {', '.join(f'{mid}({cnt})' for mid, cnt in top_20)}")

    print(f"\n  Log saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
