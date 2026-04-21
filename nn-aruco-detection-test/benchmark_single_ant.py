#!/usr/bin/env python3
"""Fair benchmark using single-ant videos with known ground truth IDs.

Each camera has exactly one ant with a known ArUco marker ID. The true
metric is simply: **in how many frames does the detector find the correct ID?**

This avoids the circular bias of using OpenCV as both detector and GT.

Known assignments (from benchmark/protocol.md):
    cam01: ID 25  (2.0mm, OpenCV 90.8%)
    cam05: ID  3  (2.5mm, OpenCV 99.8%)
    cam11: ID 25  (2.5mm, OpenCV 93.2%)
    cam12: ID 25  (1.5mm, OpenCV 87.9%)  ← hardest
    cam13: ID 17  (1.5mm, OpenCV 100%)
    cam17: ID 17  (2.0mm, OpenCV 100%)

Usage:
    python nn-aruco-detection-test/benchmark_single_ant.py `
        --video "Z:\\...\\cam01*.avi" "Z:\\...\\cam12*.avi" "Z:\\...\\cam17*.avi" `
        --detectors opencv yolo-hybrid yolo-warp `
        --yolo-weights "runs/detect/.../best.pt" `
        --n-frames 500
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class _TeeLogger:
    """Duplicate stdout to a log file."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(self.log_path, "w", encoding="utf-8")

    def write(self, msg: str):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

from aruco_detection.nn_detection.base import ArucoDetector, Detection

# Ground truth: camera name prefix → true marker ID
# Corrected 2026-04-08 after bit-level audit (audit_confused_cameras.py).
# 8 cameras had wrong labels — the ants in those arenas carry different
# markers than originally assumed.  The audit confirmed that the detected
# bits are consistently Hamming-0 to the corrected ID across 30 frames.
GROUND_TRUTH = {
    "cam01": 25,
    "cam02": 17,   # was 25 — audit: bits always match 17
    "cam03": 3,
    "cam04": 3,
    "cam05": 3,
    "cam06": 25,   # was 3  — audit: bits always match 25
    "cam07": 25,   # was 17 — audit: bits match 25 in 27/28
    "cam08": 17,
    "cam09": 3,
    "cam10": 3,
    "cam11": 25,
    "cam12": 25,
    "cam13": 17,
    "cam14": 3,    # was 17 — audit: bits match 3 in 29/31
    "cam15": 3,    # was 17 — audit: bits match 3 in 28/30
    "cam16": 25,   # was 17 — audit: bits match 25 in 29/30
    "cam17": 17,
    "cam18": 17,   # was 25 — audit: bits always match 17
    "cam19": 17,   # was 25 — audit: bits always match 17
}


def get_true_id(video_name: str) -> int | None:
    """Look up the true marker ID for a video."""
    for prefix, tid in GROUND_TRUTH.items():
        if video_name.startswith(prefix):
            return tid
    return None


def build_detector(name: str, args) -> ArucoDetector:
    if name == "opencv":
        from aruco_detection.nn_detection.opencv_baseline import OpenCVArucoDetector
        return OpenCVArucoDetector()
    elif name == "opencv-250":
        from aruco_detection.nn_detection.opencv_baseline import OpenCVArucoDetector
        import cv2.aruco as _aruco
        return OpenCVArucoDetector(dict_type=_aruco.DICT_4X4_250)
    elif name == "opencv-100":
        from aruco_detection.nn_detection.opencv_baseline import OpenCVArucoDetector
        import cv2.aruco as _aruco
        return OpenCVArucoDetector(dict_type=_aruco.DICT_4X4_100)
    elif name == "opencv-50":
        from aruco_detection.nn_detection.opencv_baseline import OpenCVArucoDetector
        import cv2.aruco as _aruco
        return OpenCVArucoDetector(dict_type=_aruco.DICT_4X4_50)
    elif name == "yolo-hybrid":
        from aruco_detection.nn_detection.yolo_opencv_hybrid import YOLOOpenCVHybridDetector
        return YOLOOpenCVHybridDetector(yolo_weights=args.yolo_weights, device=args.device)
    elif name == "yolo-cascade":
        from aruco_detection.nn_detection.yolo_cascade_hybrid import YOLOCascadeHybridDetector
        whitelist = None
        if getattr(args, "whitelist", None):
            from aruco_detection.nn_detection.whitelist import load_whitelist
            whitelist = load_whitelist(args.whitelist)
        return YOLOCascadeHybridDetector(yolo_weights=args.yolo_weights, device=args.device, whitelist=whitelist)
    elif name == "yolo-warp":
        from aruco_detection.nn_detection.yolo_warp_hybrid import YOLOWarpHybridDetector
        return YOLOWarpHybridDetector(yolo_weights=args.yolo_weights, device=args.device)
    else:
        raise ValueError(f"Unknown detector: {name}")


def benchmark_video(
    video_path: str,
    true_id: int,
    detectors: dict[str, ArucoDetector],
    n_frames: int = 500,
) -> list[dict]:
    """Benchmark all detectors on one video.

    Returns a list of result dicts, one per detector.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_frames, total)
    indices = np.linspace(0, total - 1, n, dtype=int)

    # Load frames
    frames: list[np.ndarray] = []
    for idx in tqdm(indices, desc="  Loading frames", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if not frames:
        return []

    results = []
    for det_name, detector in detectors.items():
        t0 = time.perf_counter()

        found_correct = 0      # frames where true ID was found
        found_any = 0          # frames where any marker was detected
        found_wrong_only = 0   # frames where markers found but not the true ID
        total_dets = 0
        false_positive_ids: dict[int, int] = {}

        for frame in tqdm(frames, desc=f"  {det_name}", leave=False):
            dets = detector.detect(frame)
            total_dets += len(dets)

            ids_found = set(d.marker_id for d in dets if d.marker_id >= 0)

            if ids_found:
                found_any += 1
                if true_id in ids_found:
                    found_correct += 1
                else:
                    found_wrong_only += 1
                # Count FP IDs
                for mid in ids_found:
                    if mid != true_id:
                        false_positive_ids[mid] = false_positive_ids.get(mid, 0) + 1

        elapsed = time.perf_counter() - t0
        n_frames_actual = len(frames)

        results.append({
            "detector": det_name,
            "video": Path(video_path).name,
            "true_id": true_id,
            "n_frames": n_frames_actual,
            "found_correct": found_correct,
            "found_any": found_any,
            "found_wrong_only": found_wrong_only,
            "missed": n_frames_actual - found_any,
            "detection_rate": found_correct / n_frames_actual,
            "any_detection_rate": found_any / n_frames_actual,
            "total_dets": total_dets,
            "mean_dets_per_frame": total_dets / n_frames_actual,
            "fp_ids": dict(sorted(false_positive_ids.items(), key=lambda x: -x[1])[:5]),
            "fps": n_frames_actual / elapsed if elapsed > 0 else 0,
            "runtime_sec": round(elapsed, 1),
        })

    return results


def main():
    p = argparse.ArgumentParser(description="Fair single-ant benchmark with known ground truth")
    p.add_argument("--video", type=str, nargs="+", required=True)
    p.add_argument(
        "--detectors", nargs="+",
        choices=["opencv", "opencv-250", "opencv-100", "opencv-50",
                 "yolo-hybrid", "yolo-cascade", "yolo-warp"],
        default=["opencv", "yolo-hybrid"],
    )
    p.add_argument("--yolo-weights", type=str)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--whitelist", type=str, default=None, help="Whitelist JSON path for yolo-cascade")
    p.add_argument("--n-frames", type=int, default=500, help="Frames to sample per video")
    p.add_argument("--output-dir", default="nn-aruco-detection-test/results")
    args = p.parse_args()

    # Expand globs
    video_paths = []
    for pattern in args.video:
        video_paths.extend(sorted(glob.glob(pattern)))

    if not video_paths:
        print("[ERROR] No videos found")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging — tee all output to a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.output_dir, f"single_ant_log_{timestamp}.txt")
    tee = _TeeLogger(log_path)
    sys.stdout = tee

    print(f"=== Single-Ant Benchmark ===")
    print(f"Time: {timestamp}")
    print(f"Detectors: {args.detectors}")
    print(f"Videos: {len(video_paths)}, n_frames: {args.n_frames}")
    print()

    # Build detectors
    detectors = {name: build_detector(name, args) for name in args.detectors}

    all_results: list[dict] = []
    for vpath in video_paths:
        vname = Path(vpath).stem
        cam = vname.split("_")[0]
        true_id = get_true_id(cam)
        if true_id is None:
            print(f"[SKIP] Unknown true ID for {cam}")
            continue

        print(f"\n--- {Path(vpath).name} (true ID={true_id}) ---")
        results = benchmark_video(vpath, true_id, detectors, args.n_frames)
        all_results.extend(results)

    if not all_results:
        return

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "single_ant_benchmark.csv")
    keys = ["detector", "video", "true_id", "n_frames", "found_correct",
            "found_any", "found_wrong_only", "missed", "detection_rate",
            "any_detection_rate", "total_dets", "mean_dets_per_frame", "fps", "runtime_sec"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    print(f"\nResults saved: {csv_path}")

    # Print summary
    print(f"\n{'='*110}")
    print(f"{'Detector':<14} {'Video':<20} {'ID':>3} {'Frames':>6} "
          f"{'Correct':>8} {'Any det':>8} {'Missed':>7} {'Det rate':>9} {'FP IDs':>20} {'FPS':>6}")
    print(f"{'-'*110}")
    for r in all_results:
        cam = r["video"].split("_")[0]
        fp = str(r.get("fp_ids", {}))[:20]
        print(f"{r['detector']:<14} {cam:<20} {r['true_id']:>3} {r['n_frames']:>6} "
              f"{r['found_correct']:>8} {r['found_any']:>8} {r['missed']:>7} "
              f"{r['detection_rate']:>8.1%} {fp:>20} {r['fps']:>6.1f}")
    print(f"{'='*110}")

    # Aggregate by detector
    print(f"\n--- Averages across cameras ---")
    import pandas as pd
    df = pd.DataFrame(all_results)
    for det in df["detector"].unique():
        sub = df[df["detector"] == det]
        print(f"  {det:<14}  detection_rate={sub['detection_rate'].mean():.1%}  "
              f"any_det_rate={sub['any_detection_rate'].mean():.1%}  "
              f"FPS={sub['fps'].mean():.1f}")

    print(f"\nLog saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
