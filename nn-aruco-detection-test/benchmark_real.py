#!/usr/bin/env python3
"""
Real-video benchmark for NN ArUco detectors.

Uses actual experiment videos with OpenCV detections as silver-standard
ground truth.  Measures each detector's ability to:
  1. Match known OpenCV detections (agreement)
  2. Discover additional detections that OpenCV missed (recovery)
  3. Maintain speed on 4K frames

Supports two filming setups:
  - single_ant: one ant per camera — known true ID
  - multi_ant:  dense nest area with ~220 visible tags

All outputs go to nn-aruco-detection-test/results/ — no files are written
to the original data directories.

Usage:
    # Run OpenCV-only benchmark on sampled frames
    python nn-aruco-detection-test/benchmark_real.py \\
        --config nn-aruco-detection-test/config.yaml

    # Compare YOLO against OpenCV after training
    python nn-aruco-detection-test/benchmark_real.py \\
        --config nn-aruco-detection-test/config.yaml \\
        --detectors opencv yolo \\
        --yolo-weights nn-aruco-detection-test/models/yolo_best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aruco_detection.nn_detection.base import ArucoDetector, Detection
from aruco_detection.nn_detection.opencv_baseline import OpenCVArucoDetector


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


# ---------------------------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------------------------
def sample_frames_from_video(
    video_path: str, n_frames: int, strategy: str = "uniform"
) -> tuple[list[np.ndarray], list[int]]:
    """Sample frames from a video, returning (frames, frame_indices)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Cannot get frame count: {video_path}")

    n = min(n_frames, total)
    if strategy == "uniform":
        indices = np.linspace(0, total - 1, n, dtype=int).tolist()
    else:
        indices = sorted(np.random.choice(total, n, replace=False).tolist())

    frames = []
    frame_indices = []
    for idx in tqdm(indices, desc=f"Sampling {Path(video_path).name}", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            frame_indices.append(idx)
    cap.release()
    return frames, frame_indices


# ---------------------------------------------------------------------------
# Load existing OpenCV detections as ground truth
# ---------------------------------------------------------------------------
def load_existing_detections(
    csv_path: str, frame_indices: list[int]
) -> dict[int, list[Detection]]:
    """Load pre-computed OpenCV detections for specific frame indices.

    Returns a dict mapping frame_index -> list of Detection.
    """
    df = pd.read_csv(csv_path)
    frame_set = set(frame_indices)
    gt: dict[int, list[Detection]] = {idx: [] for idx in frame_indices}

    for _, row in df.iterrows():
        f = int(row["Frame"])
        if f in frame_set:
            gt[f].append(
                Detection(
                    marker_id=int(row["Instance"]),
                    x=float(row["X"]),
                    y=float(row["Y"]),
                    confidence=float(row["Confidence"]),
                )
            )
    return gt


def run_opencv_ground_truth(
    frames: list[np.ndarray],
) -> list[list[Detection]]:
    """Run OpenCV detector on frames to generate silver-standard ground truth."""
    detector = OpenCVArucoDetector()
    return [detector.detect(f) for f in tqdm(frames, desc="  OpenCV GT", leave=False)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class RealBenchmarkMetrics:
    detector_name: str = ""
    video_name: str = ""
    n_frames: int = 0
    # Basic counts
    total_detections: int = 0
    unique_ids: int = 0
    mean_det_per_frame: float = 0.0
    std_det_per_frame: float = 0.0
    # vs ground truth
    true_positives: int = 0       # matched GT detections (correct ID + location)
    false_positives: int = 0      # NN detections not in GT
    missed: int = 0               # GT detections not found by NN
    # Recovery: detections found by NN but not by GT (potential new discoveries)
    recovered: int = 0
    # Rates
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    # Speed
    runtime_sec: float = 0.0
    fps: float = 0.0


def compute_metrics(
    detector_name: str,
    video_name: str,
    gt_per_frame: list[list[Detection]],
    pred_per_frame: list[list[Detection]],
    runtime: float,
    distance_thresh: float = 50.0,
) -> RealBenchmarkMetrics:
    """Compare predictions against ground truth frame-by-frame."""
    n_frames = len(gt_per_frame)
    all_ids: set[int] = set()
    per_frame_counts: list[int] = []
    total_tp = 0
    total_fp = 0
    total_missed = 0

    for gt_dets, pred_dets in zip(gt_per_frame, pred_per_frame):
        per_frame_counts.append(len(pred_dets))
        for d in pred_dets:
            all_ids.add(d.marker_id)

        # Match predictions to GT
        gt_matched = [False] * len(gt_dets)
        pred_matched = [False] * len(pred_dets)

        for i, pd_ in enumerate(pred_dets):
            best_dist = float("inf")
            best_j = -1
            for j, gd in enumerate(gt_dets):
                if gt_matched[j]:
                    continue
                if pd_.marker_id != gd.marker_id:
                    continue
                dist = np.hypot(pd_.x - gd.x, pd_.y - gd.y)
                if dist < distance_thresh and dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j >= 0:
                total_tp += 1
                gt_matched[best_j] = True
                pred_matched[i] = True

        total_fp += sum(1 for m in pred_matched if not m)
        total_missed += sum(1 for m in gt_matched if not m)

    counts = np.array(per_frame_counts)
    total_det = int(counts.sum())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_missed) if (total_tp + total_missed) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return RealBenchmarkMetrics(
        detector_name=detector_name,
        video_name=video_name,
        n_frames=n_frames,
        total_detections=total_det,
        unique_ids=len(all_ids),
        mean_det_per_frame=float(counts.mean()) if n_frames else 0.0,
        std_det_per_frame=float(counts.std()) if n_frames else 0.0,
        true_positives=total_tp,
        false_positives=total_fp,
        missed=total_missed,
        recovered=total_fp,  # potential recoveries = FP relative to OpenCV GT
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        runtime_sec=round(runtime, 3),
        fps=round(n_frames / runtime, 1) if runtime > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Detector construction
# ---------------------------------------------------------------------------
def build_detector(name: str, args) -> ArucoDetector:
    if name == "opencv":
        return OpenCVArucoDetector()
    elif name == "opencv-250":
        import cv2.aruco as _aruco
        return OpenCVArucoDetector(dict_type=_aruco.DICT_4X4_250)
    elif name == "opencv-100":
        import cv2.aruco as _aruco
        return OpenCVArucoDetector(dict_type=_aruco.DICT_4X4_100)
    elif name == "opencv-50":
        import cv2.aruco as _aruco
        return OpenCVArucoDetector(dict_type=_aruco.DICT_4X4_50)
    elif name == "yolo":
        from aruco_detection.nn_detection.yolo_detector import YOLOArucoDetector
        return YOLOArucoDetector(
            yolo_weights=args.yolo_weights,
            classifier_weights=args.classifier_weights,
            class_names_path=args.class_names,
            device=args.device,
        )
    elif name == "rtdetr":
        from aruco_detection.nn_detection.rtdetr_detector import RTDETRArucoDetector
        return RTDETRArucoDetector(
            rtdetr_weights=args.rtdetr_weights,
            classifier_weights=args.classifier_weights,
            class_names_path=args.class_names,
            device=args.device,
        )
    elif name == "yolo-hybrid":
        if not args.yolo_weights:
            raise ValueError("--yolo-weights required for yolo-hybrid detector")
        from aruco_detection.nn_detection.yolo_opencv_hybrid import YOLOOpenCVHybridDetector
        return YOLOOpenCVHybridDetector(
            yolo_weights=args.yolo_weights,
            device=args.device,
        )

    elif name == "yolo-cascade":
        if not args.yolo_weights:
            raise ValueError("--yolo-weights required for yolo-cascade detector")
        from aruco_detection.nn_detection.yolo_cascade_hybrid import YOLOCascadeHybridDetector
        whitelist = None
        if getattr(args, "whitelist", None):
            from aruco_detection.nn_detection.whitelist import load_whitelist
            whitelist = load_whitelist(args.whitelist)
        return YOLOCascadeHybridDetector(
            yolo_weights=args.yolo_weights,
            device=args.device,
            whitelist=whitelist,
        )

    elif name == "yolo-warp":
        if not args.yolo_weights:
            raise ValueError("--yolo-weights required for yolo-warp detector")
        from aruco_detection.nn_detection.yolo_warp_hybrid import YOLOWarpHybridDetector
        return YOLOWarpHybridDetector(
            yolo_weights=args.yolo_weights,
            device=args.device,
        )

    elif name == "deeparuco-pytorch":
        if not args.yolo_weights:
            raise ValueError("--yolo-weights required for deeparuco-pytorch detector")
        if not args.corner_refiner_weights:
            raise ValueError("--corner-refiner-weights required for deeparuco-pytorch detector")
        if not args.decoder_weights:
            raise ValueError("--decoder-weights required for deeparuco-pytorch detector")
        from aruco_detection.nn_detection.deeparuco_pytorch import DeepArucoPytorchDetector
        return DeepArucoPytorchDetector(
            yolo_weights=args.yolo_weights,
            corner_refiner_weights=args.corner_refiner_weights,
            decoder_weights=args.decoder_weights,
            device=args.device,
        )

    elif name == "deeparuco":
        from aruco_detection.nn_detection.deeparuco_detector import DeepArucoDetector
        return DeepArucoDetector(
            deeparuco_path=args.deeparuco_path,
            detection_model=args.deeparuco_detection_model or "",
            refinement_model=args.deeparuco_refinement_model or "",
            decoding_model=args.deeparuco_decoding_model or "",
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown detector: {name}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
RESULT_COLUMNS = [
    "detector_name", "video_name", "n_frames",
    "total_detections", "unique_ids", "mean_det_per_frame", "std_det_per_frame",
    "true_positives", "false_positives", "missed", "recovered",
    "precision", "recall", "f1",
    "runtime_sec", "fps",
]


def write_csv(results: list[RealBenchmarkMetrics], path: str):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        for m in results:
            w.writerow({k: getattr(m, k) for k in RESULT_COLUMNS})
    print(f"Results saved: {path}")


def print_summary(results: list[RealBenchmarkMetrics]):
    print(f"\n{'=' * 120}")
    print(
        f"{'Detector':<14} {'Video':<40} {'Frames':>6} "
        f"{'Det':>6} {'UIDs':>5} {'TP':>6} {'FP':>6} {'Miss':>6} "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'FPS':>7}"
    )
    print(f"{'-' * 120}")
    for m in results:
        vname = m.video_name[:38] if len(m.video_name) > 38 else m.video_name
        print(
            f"{m.detector_name:<14} {vname:<40} {m.n_frames:>6} "
            f"{m.total_detections:>6} {m.unique_ids:>5} "
            f"{m.true_positives:>6} {m.false_positives:>6} {m.missed:>6} "
            f"{m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f} {m.fps:>7.1f}"
        )
    print(f"{'=' * 120}")


def plot_comparison(results: list[RealBenchmarkMetrics], output_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Aggregate by detector
    from collections import defaultdict
    by_det: dict[str, list[RealBenchmarkMetrics]] = defaultdict(list)
    for m in results:
        by_det[m.detector_name].append(m)

    names = list(by_det.keys())
    avg_recall = [np.mean([m.recall for m in by_det[n]]) for n in names]
    avg_prec = [np.mean([m.precision for m in by_det[n]]) for n in names]
    avg_fps = [np.mean([m.fps for m in by_det[n]]) for n in names]
    avg_det = [np.mean([m.mean_det_per_frame for m in by_det[n]]) for n in names]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    x = range(len(names))

    axes[0].bar(x, avg_det, color="steelblue")
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(names)
    axes[0].set_ylabel("Mean det/frame"); axes[0].set_title("Detection Volume")

    axes[1].bar(x, avg_recall, color="seagreen")
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(names)
    axes[1].set_ylabel("Recall"); axes[1].set_title("Recall (vs OpenCV GT)")
    axes[1].set_ylim(0, 1.05)

    axes[2].bar(x, avg_prec, color="coral")
    axes[2].set_xticks(list(x)); axes[2].set_xticklabels(names)
    axes[2].set_ylabel("Precision"); axes[2].set_title("Precision")
    axes[2].set_ylim(0, 1.05)

    axes[3].bar(x, avg_fps, color="goldenrod")
    axes[3].set_xticks(list(x)); axes[3].set_xticklabels(names)
    axes[3].set_ylabel("FPS"); axes[3].set_title("Speed")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "real_benchmark_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved: {plot_path}")


# ---------------------------------------------------------------------------
# Per-frame detail export (for debugging missed detections)
# ---------------------------------------------------------------------------
def export_frame_details(
    video_name: str,
    frame_indices: list[int],
    gt_per_frame: list[list[Detection]],
    pred_per_frame: dict[str, list[list[Detection]]],
    output_dir: str,
    distance_thresh: float = 50.0,
):
    """Export per-frame detection details for all detectors.

    Creates a CSV with columns:
        frame_idx, detector, marker_id, x, y, confidence, status
    where status is: TP, FP (not in GT), or MISSED (in GT but not detected).
    """
    rows = []
    for fi, frame_idx in enumerate(frame_indices):
        gt_dets = gt_per_frame[fi]

        for det_name, all_preds in pred_per_frame.items():
            preds = all_preds[fi]

            gt_matched = [False] * len(gt_dets)
            for pd_ in preds:
                matched_gt = False
                for j, gd in enumerate(gt_dets):
                    if gt_matched[j]:
                        continue
                    if pd_.marker_id == gd.marker_id:
                        dist = np.hypot(pd_.x - gd.x, pd_.y - gd.y)
                        if dist < distance_thresh:
                            gt_matched[j] = True
                            matched_gt = True
                            break
                rows.append({
                    "frame_idx": frame_idx,
                    "detector": det_name,
                    "marker_id": pd_.marker_id,
                    "x": round(pd_.x, 1),
                    "y": round(pd_.y, 1),
                    "confidence": round(pd_.confidence, 3),
                    "status": "TP" if matched_gt else "FP",
                })

            # Record misses
            for j, gd in enumerate(gt_dets):
                if not gt_matched[j]:
                    rows.append({
                        "frame_idx": frame_idx,
                        "detector": det_name,
                        "marker_id": gd.marker_id,
                        "x": round(gd.x, 1),
                        "y": round(gd.y, 1),
                        "confidence": 0.0,
                        "status": "MISSED",
                    })

    vname = Path(video_name).stem
    out_path = os.path.join(output_dir, f"frame_details_{vname}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Frame details saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Real-video ArUco detector benchmark")

    # Input
    p.add_argument("--config", type=str, help="Path to config.yaml")
    p.add_argument("--video", type=str, nargs="*", help="Video file(s) to benchmark (overrides config)")
    p.add_argument("--image-dir", type=str, nargs="*", help="Directories of frame images (alternative to --video)")
    p.add_argument(
        "--existing-csv", type=str,
        help="Pre-existing OpenCV detection CSV for ground truth (overrides config)",
    )

    # Sampling
    p.add_argument("--n-frames", type=int, default=200, help="Frames to sample per video")
    p.add_argument("--strategy", choices=["uniform", "random"], default="uniform")

    # Detectors
    p.add_argument(
        "--detectors", nargs="+",
        choices=["opencv", "opencv-250", "opencv-100", "opencv-50",
                 "yolo", "yolo-hybrid", "yolo-cascade", "yolo-warp",
                 "deeparuco-pytorch", "rtdetr", "deeparuco"],
        default=["opencv"],
    )
    p.add_argument("--yolo-weights", type=str)
    p.add_argument("--rtdetr-weights", type=str)
    p.add_argument("--classifier-weights", type=str)
    p.add_argument("--class-names", type=str)
    p.add_argument("--corner-refiner-weights", type=str, help="Corner refiner U-Net weights")
    p.add_argument("--decoder-weights", type=str, help="Bit decoder CNN weights")
    p.add_argument("--deeparuco-path", type=str)
    p.add_argument("--deeparuco-detection-model", type=str)
    p.add_argument("--deeparuco-refinement-model", type=str)
    p.add_argument("--deeparuco-decoding-model", type=str)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--whitelist", type=str, default=None, help="Whitelist JSON path for yolo-cascade")

    # Output
    p.add_argument(
        "--output-dir", type=str,
        default="nn-aruco-detection-test/results",
    )
    p.add_argument("--export-details", action="store_true", help="Export per-frame detection details")

    args = p.parse_args()

    # Resolve video paths
    video_paths: list[str] = []
    existing_csv_map: dict[str, str | None] = {}

    image_dir_map: dict[str, str] = {}  # pseudo-path -> directory

    if args.image_dir:
        import glob as _glob
        for pattern in args.image_dir:
            for d in sorted(_glob.glob(pattern)):
                if os.path.isdir(d):
                    image_dir_map[d] = d
                    video_paths.append(d)
                    existing_csv_map[d] = args.existing_csv
    elif args.video:
        # Expand glob patterns (PowerShell doesn't expand wildcards)
        import glob as _glob
        for pattern in args.video:
            video_paths.extend(_glob.glob(pattern))
        for v in video_paths:
            existing_csv_map[v] = args.existing_csv
    elif args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        for bname, bcfg in cfg.get("benchmarks", {}).items():
            vdir = bcfg["video_dir"]
            pattern = bcfg.get("video_pattern", "*.avi")
            import glob
            vids = sorted(glob.glob(os.path.join(vdir, pattern)))
            video_paths.extend(vids)

            # Map existing detections
            det_dir = bcfg.get("existing_detections_dir")
            for v in vids:
                if det_dir:
                    stem = Path(v).stem
                    # Try chunk000 CSV naming convention
                    candidate = os.path.join(det_dir, f"{stem}_000_aruco_detections.csv")
                    if os.path.isfile(candidate):
                        existing_csv_map[v] = candidate
                    else:
                        existing_csv_map[v] = None
                else:
                    existing_csv_map[v] = None

        args.n_frames = cfg.get("sampling", {}).get("n_frames", args.n_frames)
        args.strategy = cfg.get("sampling", {}).get("strategy", args.strategy)
    else:
        p.error("Provide --config, --video, or --image-dir")

    if not video_paths:
        print("[ERROR] No videos found")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging — tee all output to a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.output_dir, f"benchmark_log_{timestamp}.txt")
    tee = _TeeLogger(log_path)
    sys.stdout = tee

    print(f"=== Real-Video Benchmark ===")
    print(f"Time: {timestamp}")
    print(f"Detectors: {args.detectors}")
    print(f"Videos: {len(video_paths)}, n_frames: {args.n_frames}")
    print()

    print(f"Found {len(video_paths)} videos to benchmark")

    # Build detectors
    detectors = {name: build_detector(name, args) for name in args.detectors}

    all_results: list[RealBenchmarkMetrics] = []

    for video_path in video_paths:
        vname = Path(video_path).name
        print(f"\n--- {vname} ---")

        # Load frames
        if video_path in image_dir_map:
            # Load from image directory
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}
            img_paths = sorted(
                p for p in Path(video_path).iterdir() if p.suffix.lower() in exts
            )
            n = min(args.n_frames, len(img_paths))
            selected = [img_paths[i] for i in np.linspace(0, len(img_paths) - 1, n, dtype=int)]
            frames = []
            frame_indices = []
            for ip in selected:
                img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
                if img is not None:
                    frames.append(img)
                    frame_indices.append(int(ip.stem.split("_")[-1]) if "_" in ip.stem else len(frames) - 1)
            print(f"  Loaded {len(frames)} frames from directory")
        else:
            frames, frame_indices = sample_frames_from_video(
                video_path, args.n_frames, args.strategy
            )
        if not frames:
            print(f"  [SKIP] No frames loaded from {vname}")
            continue

        # Ground truth: use existing CSV if available, otherwise run OpenCV live
        existing_csv = existing_csv_map.get(video_path)
        if existing_csv and os.path.isfile(existing_csv):
            print(f"  Using existing GT: {Path(existing_csv).name}")
            gt_map = load_existing_detections(existing_csv, frame_indices)
            gt_per_frame = [gt_map[idx] for idx in frame_indices]
        else:
            print("  Running OpenCV for ground truth...")
            gt_per_frame = run_opencv_ground_truth(frames)

        # Run each detector
        pred_per_frame: dict[str, list[list[Detection]]] = {}
        for det_name, detector in detectors.items():
            print(f"  Running {det_name}...")
            t0 = time.perf_counter()
            preds = [detector.detect(f) for f in tqdm(frames, desc=f"    {det_name}", leave=False)]
            elapsed = time.perf_counter() - t0
            pred_per_frame[det_name] = preds

            metrics = compute_metrics(
                det_name, vname, gt_per_frame, preds, elapsed
            )
            all_results.append(metrics)

        # Export per-frame details
        if args.export_details:
            export_frame_details(
                vname, frame_indices, gt_per_frame, pred_per_frame, args.output_dir
            )

    # Write results
    csv_path = os.path.join(args.output_dir, "real_benchmark_results.csv")
    write_csv(all_results, csv_path)
    print_summary(all_results)
    plot_comparison(all_results, args.output_dir)

    print(f"\nLog saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
