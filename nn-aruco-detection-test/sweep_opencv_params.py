#!/usr/bin/env python3
"""OpenCV ArUco parameter sweep.

Benchmarks multiple DetectorParameters profiles on sampled video frames
to find the best settings for:
  (A) full-frame dense detection
  (B) single-crop rescue decoding (on YOLO bbox crops)

Profiles tested:
  - baseline:  current production settings (CORNER_REFINE_CONTOUR)
  - apriltag:  CORNER_REFINE_APRILTAG
  - aruco3:    useAruco3Detection=True
  - aruco3_at: useAruco3Detection=True + CORNER_REFINE_APRILTAG
  - subpix:    CORNER_REFINE_SUBPIX
  - aggressive: larger refinement window, more aggressive extraction
  - conservative: smaller refinement window, tighter rejection

Each profile runs on both the full frame and on padded YOLO crops (if
--yolo-weights is provided) to measure full-frame and crop-rescue behavior.

Usage:
    python nn-aruco-detection-test/sweep_opencv_params.py \\
        --video "Z:\\...\\cam04*.avi" \\
        --n-frames 100 \\
        --output-dir nn-aruco-detection-test/results

    # With YOLO crop rescue mode:
    python nn-aruco-detection-test/sweep_opencv_params.py \\
        --video "Z:\\...\\cam04*.avi" \\
        --yolo-weights runs/detect/.../best.pt \\
        --n-frames 100
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
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


# ---------------------------------------------------------------------------
# Parameter profiles
# ---------------------------------------------------------------------------

def _make_baseline() -> aruco.DetectorParameters:
    p = aruco.DetectorParameters()
    p.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    p.adaptiveThreshConstant = 3
    p.adaptiveThreshWinSizeMin = 10
    p.adaptiveThreshWinSizeMax = 40
    p.adaptiveThreshWinSizeStep = 10
    p.errorCorrectionRate = 1.0
    p.minMarkerPerimeterRate = 0.03
    p.maxMarkerPerimeterRate = 4.0
    return p


def _make_apriltag() -> aruco.DetectorParameters:
    p = _make_baseline()
    p.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    return p


def _make_subpix() -> aruco.DetectorParameters:
    p = _make_baseline()
    p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return p


def _make_aruco3() -> aruco.DetectorParameters:
    p = _make_baseline()
    p.useAruco3Detection = True
    return p


def _make_aruco3_apriltag() -> aruco.DetectorParameters:
    p = _make_baseline()
    p.useAruco3Detection = True
    p.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    return p


def _make_aggressive() -> aruco.DetectorParameters:
    """Aggressive profile for isolated markers / crops."""
    p = aruco.DetectorParameters()
    p.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    p.adaptiveThreshConstant = 3
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 80
    p.adaptiveThreshWinSizeStep = 3
    p.errorCorrectionRate = 0.8
    p.minMarkerPerimeterRate = 0.01
    p.maxMarkerPerimeterRate = 4.0
    # Larger corner refinement window for isolated markers
    p.relativeCornerRefinmentWinSize = 0.5
    # More aggressive perspective extraction
    p.perspectiveRemovePixelPerCell = 8
    p.perspectiveRemoveIgnoredMarginPerCell = 0.2
    return p


def _make_conservative() -> aruco.DetectorParameters:
    """Conservative profile for dense scenes."""
    p = aruco.DetectorParameters()
    p.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    p.adaptiveThreshConstant = 5
    p.adaptiveThreshWinSizeMin = 5
    p.adaptiveThreshWinSizeMax = 30
    p.adaptiveThreshWinSizeStep = 5
    p.errorCorrectionRate = 0.6
    p.minMarkerPerimeterRate = 0.03
    p.maxMarkerPerimeterRate = 4.0
    # Smaller corner refinement window for close-together markers
    p.relativeCornerRefinmentWinSize = 0.15
    p.perspectiveRemovePixelPerCell = 6
    p.perspectiveRemoveIgnoredMarginPerCell = 0.15
    return p


PROFILES = {
    "baseline": _make_baseline,
    "apriltag": _make_apriltag,
    "subpix": _make_subpix,
    "aruco3": _make_aruco3,
    "aruco3_at": _make_aruco3_apriltag,
    "aggressive": _make_aggressive,
    "conservative": _make_conservative,
}


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_profile_fullframe(
    profile_name: str,
    detector: aruco.ArucoDetector,
    frames: list[np.ndarray],
) -> dict:
    """Run a parameter profile on full frames, measure detection count and speed."""
    total_dets = 0
    id_set: set[int] = set()
    t0 = time.perf_counter()

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None:
            total_dets += len(ids)
            id_set.update(int(x) for x in ids.flatten())

    elapsed = time.perf_counter() - t0
    fps = len(frames) / elapsed if elapsed > 0 else 0

    return {
        "profile": profile_name,
        "mode": "fullframe",
        "n_frames": len(frames),
        "total_dets": total_dets,
        "unique_ids": len(id_set),
        "mean_dets_per_frame": total_dets / len(frames) if frames else 0,
        "fps": fps,
        "elapsed_sec": round(elapsed, 2),
    }


def benchmark_profile_crops(
    profile_name: str,
    detector: aruco.ArucoDetector,
    frames: list[np.ndarray],
    yolo_model,
    yolo_conf: float = 0.25,
    crop_padding: float = 0.5,
) -> dict:
    """Run YOLO to get crops, then decode with the given profile."""
    total_yolo = 0
    total_decoded = 0
    total_undecoded = 0
    id_set: set[int] = set()
    t0 = time.perf_counter()

    for frame in frames:
        results = yolo_model.predict(frame, conf=yolo_conf, verbose=False)
        fh, fw = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                total_yolo += 1
                xyxy = result.boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                bw, bh = x2 - x1, y2 - y1
                pad_x, pad_y = bw * crop_padding, bh * crop_padding
                cx1 = max(0, int(x1 - pad_x))
                cy1 = max(0, int(y1 - pad_y))
                cx2 = min(fw, int(x2 + pad_x))
                cy2 = min(fh, int(y2 + pad_y))
                crop = gray[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                corners, ids, _ = detector.detectMarkers(crop)
                if ids is not None and len(ids) > 0:
                    total_decoded += 1
                    id_set.update(int(x) for x in ids.flatten())
                else:
                    total_undecoded += 1

    elapsed = time.perf_counter() - t0
    decode_rate = total_decoded / total_yolo if total_yolo else 0

    return {
        "profile": profile_name,
        "mode": "yolo_crop",
        "n_frames": len(frames),
        "total_yolo_dets": total_yolo,
        "total_decoded": total_decoded,
        "total_undecoded": total_undecoded,
        "decode_rate": round(decode_rate, 4),
        "unique_ids": len(id_set),
        "fps": len(frames) / elapsed if elapsed > 0 else 0,
        "elapsed_sec": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="OpenCV ArUco parameter sweep")
    p.add_argument("--video", nargs="+", required=True, help="Video paths (glob OK)")
    p.add_argument("--n-frames", type=int, default=100)
    p.add_argument("--output-dir", default="nn-aruco-detection-test/results")
    p.add_argument("--yolo-weights", type=str, default=None,
                   help="If provided, also benchmark crop-rescue mode")
    p.add_argument("--profiles", nargs="*", default=None,
                   help=f"Profiles to test (default: all). Choices: {list(PROFILES.keys())}")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"sweep_log_{timestamp}.txt"
    tee = _TeeLogger(log_path)
    sys.stdout = tee

    print(f"=== OpenCV Parameter Sweep ===")
    print(f"Time: {timestamp}")
    print(f"Args: {vars(args)}")
    print()

    # Expand globs
    video_paths = []
    for pattern in args.video:
        expanded = sorted(glob.glob(pattern))
        video_paths.extend(expanded if expanded else [pattern])

    if not video_paths:
        print("[ERROR] No videos found")
        tee.close()
        return

    profile_names = args.profiles or list(PROFILES.keys())
    print(f"Videos: {len(video_paths)}")
    print(f"Profiles: {profile_names}")
    print(f"Frames per video: {args.n_frames}")

    # Build detectors
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    detectors = {}
    for pname in profile_names:
        if pname not in PROFILES:
            print(f"[WARN] Unknown profile '{pname}', skipping")
            continue
        params = PROFILES[pname]()
        detectors[pname] = aruco.ArucoDetector(aruco_dict, params)

    # Load YOLO if provided
    yolo_model = None
    if args.yolo_weights:
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo_weights)
        print(f"YOLO model loaded: {args.yolo_weights}")

    all_results: list[dict] = []

    for vpath in video_paths:
        vname = Path(vpath).stem
        print(f"\n{'='*70}")
        print(f"Video: {vname}")
        print(f"{'='*70}")

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  [ERROR] Cannot open: {vpath}")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = min(args.n_frames, total)
        indices = np.linspace(0, total - 1, n, dtype=int)

        frames = []
        for idx in tqdm(indices, desc="  Loading frames", leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        cap.release()

        if not frames:
            print("  [SKIP] No frames loaded")
            continue

        print(f"  Loaded {len(frames)} frames")

        # Full-frame benchmarks
        print(f"\n  --- Full-frame mode ---")
        print(f"  {'Profile':<16} {'Dets':>6} {'UIDs':>5} {'Dets/fr':>8} {'FPS':>7}")
        print(f"  {'-'*50}")

        for pname, det in detectors.items():
            r = benchmark_profile_fullframe(pname, det, frames)
            r["video"] = vname
            all_results.append(r)
            print(f"  {pname:<16} {r['total_dets']:>6} {r['unique_ids']:>5} "
                  f"{r['mean_dets_per_frame']:>8.1f} {r['fps']:>7.1f}")

        # Crop-rescue benchmarks (if YOLO available)
        if yolo_model is not None:
            print(f"\n  --- YOLO crop-rescue mode ---")
            print(f"  {'Profile':<16} {'YOLO':>6} {'Decoded':>8} {'Undec':>6} {'Rate':>7} {'FPS':>7}")
            print(f"  {'-'*60}")

            for pname, det in detectors.items():
                r = benchmark_profile_crops(pname, det, frames, yolo_model)
                r["video"] = vname
                all_results.append(r)
                print(f"  {pname:<16} {r['total_yolo_dets']:>6} {r['total_decoded']:>8} "
                      f"{r['total_undecoded']:>6} {r['decode_rate']:>7.1%} {r['fps']:>7.1f}")

    # Save CSV
    if all_results:
        csv_path = out_dir / f"sweep_results_{timestamp}.csv"
        keys = sorted(set().union(*(r.keys() for r in all_results)))
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_results)
        print(f"\nResults CSV saved: {csv_path}")

    # Summary: best profile per mode
    if all_results:
        print(f"\n{'='*70}")
        print("Summary: best profiles")
        print(f"{'='*70}")

        ff_results = [r for r in all_results if r["mode"] == "fullframe"]
        if ff_results:
            # Best by total detections
            best_dets = max(ff_results, key=lambda r: r["total_dets"])
            best_fps = max(ff_results, key=lambda r: r["fps"])
            print(f"\n  Full-frame — most detections: {best_dets['profile']} ({best_dets['total_dets']} dets)")
            print(f"  Full-frame — fastest:         {best_fps['profile']} ({best_fps['fps']:.1f} FPS)")

        cr_results = [r for r in all_results if r["mode"] == "yolo_crop"]
        if cr_results:
            best_rate = max(cr_results, key=lambda r: r["decode_rate"])
            print(f"\n  Crop rescue — best decode rate: {best_rate['profile']} ({best_rate['decode_rate']:.1%})")

    print(f"\nLog saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
