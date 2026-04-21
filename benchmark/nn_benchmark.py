#!/usr/bin/env python3
"""
Unified benchmark for NN-based ArUco detectors vs OpenCV baseline.

Runs all configured detectors on the same set of frames and reports
detection metrics, ID accuracy, and speed in a single comparison table.

Uses the same frame-loading infrastructure as ``aruco_benchmark.py`` so
results are directly comparable.

Example:
    # Compare OpenCV baseline with YOLO on sample frames
    python benchmark/nn_benchmark.py --image-dir benchmark/sample_frames \\
        --detectors opencv yolo \\
        --yolo-weights models/yolo_aruco.pt

    # Full comparison with all detectors
    python benchmark/nn_benchmark.py --video cam01.avi --sample-frames 100 \\
        --detectors opencv yolo rtdetr \\
        --yolo-weights models/yolo_aruco.pt \\
        --rtdetr-weights models/rtdetr_aruco.pt \\
        --classifier-weights models/antnet.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Reuse frame loading from the existing benchmark
from benchmark.aruco_benchmark import load_frames_from_dir, load_frames_from_video

from aruco_detection.nn_detection.base import ArucoDetector, Detection


@dataclass
class NNMetrics:
    """Extended metrics for NN detector benchmarking."""

    detector_name: str = ""
    total_detections: int = 0
    unique_ids: int = 0
    mean_detections_per_frame: float = 0.0
    std_detections_per_frame: float = 0.0
    runtime_sec: float = 0.0
    fps: float = 0.0
    # Agreement metrics (vs reference detector, typically OpenCV)
    agreement_rate: float = 0.0  # fraction of reference detections also found
    extra_detections: int = 0  # detections not in reference
    missed_detections: int = 0  # reference detections not found


RESULT_COLUMNS = [
    "detector_name",
    "total_detections",
    "unique_ids",
    "mean_detections_per_frame",
    "std_detections_per_frame",
    "runtime_sec",
    "fps",
    "agreement_rate",
    "extra_detections",
    "missed_detections",
]


def run_detector_benchmark(
    detector: ArucoDetector,
    frames: list[np.ndarray],
) -> tuple[NNMetrics, list[list[Detection]]]:
    """Run a detector on all frames and compute metrics.

    Returns metrics and per-frame detection lists for cross-detector comparison.
    """
    per_frame_counts: list[int] = []
    all_ids: set[int] = set()
    all_detections: list[list[Detection]] = []

    t0 = time.perf_counter()
    for gray in tqdm(frames, desc=f"  {detector.name}"):
        # Convert grayscale to BGR if needed (some detectors expect colour)
        if gray.ndim == 2:
            frame_in = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            frame_in = gray

        dets = detector.detect(frame_in)
        all_detections.append(dets)
        per_frame_counts.append(len(dets))
        for d in dets:
            all_ids.add(d.marker_id)

    elapsed = time.perf_counter() - t0
    counts = np.array(per_frame_counts)
    n_frames = len(frames)

    return (
        NNMetrics(
            detector_name=detector.name,
            total_detections=int(counts.sum()),
            unique_ids=len(all_ids),
            mean_detections_per_frame=float(counts.mean()) if n_frames else 0.0,
            std_detections_per_frame=float(counts.std()) if n_frames else 0.0,
            runtime_sec=round(elapsed, 3),
            fps=round(n_frames / elapsed, 1) if elapsed > 0 else 0.0,
        ),
        all_detections,
    )


def compute_agreement(
    ref_dets: list[list[Detection]],
    test_dets: list[list[Detection]],
    distance_threshold: float = 50.0,
) -> tuple[float, int, int]:
    """Compute agreement between a reference and test detector.

    For each reference detection, check if there is a test detection within
    ``distance_threshold`` px with the same marker ID.

    Returns (agreement_rate, extra_count, missed_count).
    """
    total_ref = 0
    matched = 0
    total_test = 0

    for ref_frame, test_frame in zip(ref_dets, test_dets):
        total_ref += len(ref_frame)
        total_test += len(test_frame)

        test_used = [False] * len(test_frame)
        for rd in ref_frame:
            found = False
            for j, td in enumerate(test_frame):
                if test_used[j]:
                    continue
                if td.marker_id == rd.marker_id:
                    dist = np.hypot(td.x - rd.x, td.y - rd.y)
                    if dist < distance_threshold:
                        matched += 1
                        test_used[j] = True
                        found = True
                        break
            # If rd was not matched, it counts as a miss (handled below)

    missed = total_ref - matched
    extra = total_test - matched

    agreement_rate = matched / total_ref if total_ref > 0 else 0.0
    return agreement_rate, extra, missed


def build_detectors(args) -> list[ArucoDetector]:
    """Instantiate the requested detectors from CLI arguments."""
    detectors: list[ArucoDetector] = []

    for name in args.detectors:
        if name == "opencv":
            from aruco_detection.nn_detection.opencv_baseline import OpenCVArucoDetector

            detectors.append(OpenCVArucoDetector())

        elif name == "yolo":
            if not args.yolo_weights:
                raise ValueError("--yolo-weights required for YOLO detector")
            from aruco_detection.nn_detection.yolo_detector import YOLOArucoDetector

            detectors.append(
                YOLOArucoDetector(
                    yolo_weights=args.yolo_weights,
                    classifier_weights=args.classifier_weights,
                    class_names_path=args.class_names,
                    device=args.device,
                )
            )

        elif name == "rtdetr":
            if not args.rtdetr_weights:
                raise ValueError("--rtdetr-weights required for RT-DETR detector")
            from aruco_detection.nn_detection.rtdetr_detector import RTDETRArucoDetector

            detectors.append(
                RTDETRArucoDetector(
                    rtdetr_weights=args.rtdetr_weights,
                    classifier_weights=args.classifier_weights,
                    class_names_path=args.class_names,
                    device=args.device,
                )
            )

        elif name == "deeparuco":
            if not args.deeparuco_path:
                raise ValueError("--deeparuco-path required for DeepArUco++ detector")
            from aruco_detection.nn_detection.deeparuco_detector import DeepArucoDetector

            detectors.append(
                DeepArucoDetector(
                    deeparuco_path=args.deeparuco_path,
                    detection_model=args.deeparuco_detection_model or "",
                    refinement_model=args.deeparuco_refinement_model or "",
                    decoding_model=args.deeparuco_decoding_model or "",
                    device=args.device,
                )
            )

        else:
            raise ValueError(f"Unknown detector: {name}")

    return detectors


def write_results(results: list[NNMetrics], output_path: str):
    """Write benchmark results to CSV."""
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        for m in results:
            w.writerow(
                {
                    "detector_name": m.detector_name,
                    "total_detections": m.total_detections,
                    "unique_ids": m.unique_ids,
                    "mean_detections_per_frame": f"{m.mean_detections_per_frame:.2f}",
                    "std_detections_per_frame": f"{m.std_detections_per_frame:.2f}",
                    "runtime_sec": m.runtime_sec,
                    "fps": m.fps,
                    "agreement_rate": f"{m.agreement_rate:.4f}",
                    "extra_detections": m.extra_detections,
                    "missed_detections": m.missed_detections,
                }
            )
    print(f"\nResults saved to: {output_path}")


def plot_comparison(results: list[NNMetrics], output_dir: str):
    """Generate comparison plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
        return

    names = [m.detector_name for m in results]
    x = range(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Detection count
    means = [m.mean_detections_per_frame for m in results]
    stds = [m.std_detections_per_frame for m in results]
    axes[0].bar(x, means, yerr=stds, capsize=4, color="steelblue")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("Mean detections / frame")
    axes[0].set_title("Detection Count")

    # Speed
    fps = [m.fps for m in results]
    axes[1].bar(x, fps, color="seagreen")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("FPS")
    axes[1].set_title("Speed")

    # Agreement with reference
    agreements = [m.agreement_rate * 100 for m in results]
    axes[2].bar(x, agreements, color="coral")
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(names)
    axes[2].set_ylabel("Agreement rate (%)")
    axes[2].set_title("Agreement with reference")
    axes[2].set_ylim(0, 105)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "nn_benchmark_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved: {plot_path}")


def main():
    p = argparse.ArgumentParser(description="NN ArUco detector benchmark")

    # Input
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--video", type=str, help="Path to input video")
    inp.add_argument("--image-dir", type=str, help="Directory of frame images")
    p.add_argument("--sample-frames", type=int, default=100, help="Frames to sample from video")

    # Detectors
    p.add_argument(
        "--detectors",
        nargs="+",
        choices=["opencv", "yolo", "rtdetr", "deeparuco"],
        default=["opencv"],
        help="Detectors to benchmark",
    )

    # Model weights
    p.add_argument("--yolo-weights", type=str, help="YOLO weights path")
    p.add_argument("--rtdetr-weights", type=str, help="RT-DETR weights path")
    p.add_argument("--classifier-weights", type=str, help="ResNet50 classifier weights")
    p.add_argument("--class-names", type=str, help="Class names .npy file")

    # DeepArUco++ specific
    p.add_argument("--deeparuco-path", type=str, help="Path to deeparuco repo")
    p.add_argument("--deeparuco-detection-model", type=str)
    p.add_argument("--deeparuco-refinement-model", type=str)
    p.add_argument("--deeparuco-decoding-model", type=str)

    # Runtime
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    p.add_argument("--output-dir", type=str, default=".", help="Output directory")

    args = p.parse_args()

    # Load frames
    if args.video:
        frames = load_frames_from_video(args.video, args.sample_frames)
    else:
        frames = load_frames_from_dir(args.image_dir)

    if not frames:
        print("[ERROR] No frames loaded")
        return

    # Build detectors
    detectors = build_detectors(args)
    print(f"\nBenchmarking {len(detectors)} detector(s) on {len(frames)} frames\n")

    # Run benchmarks
    all_metrics: list[NNMetrics] = []
    all_detections: dict[str, list[list[Detection]]] = {}

    for detector in detectors:
        metrics, dets = run_detector_benchmark(detector, frames)
        all_metrics.append(metrics)
        all_detections[detector.name] = dets

    # Compute agreement (use first detector as reference)
    ref_name = detectors[0].name
    ref_dets = all_detections[ref_name]

    for metrics in all_metrics:
        if metrics.detector_name == ref_name:
            metrics.agreement_rate = 1.0
            metrics.extra_detections = 0
            metrics.missed_detections = 0
        else:
            test_dets = all_detections[metrics.detector_name]
            agreement, extra, missed = compute_agreement(ref_dets, test_dets)
            metrics.agreement_rate = agreement
            metrics.extra_detections = extra
            metrics.missed_detections = missed

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "nn_benchmark_results.csv")
    write_results(all_metrics, csv_path)
    plot_comparison(all_metrics, args.output_dir)

    # Print summary
    print(f"\n{'=' * 100}")
    print(
        f"{'Detector':<15}  {'Total det':>10}  {'Unique IDs':>10}  "
        f"{'Mean/frame':>10}  {'FPS':>8}  {'Agreement':>10}  {'Extra':>8}  {'Missed':>8}"
    )
    print(f"{'-' * 100}")
    for m in all_metrics:
        print(
            f"{m.detector_name:<15}  {m.total_detections:>10}  {m.unique_ids:>10}  "
            f"{m.mean_detections_per_frame:>10.2f}  {m.fps:>8.1f}  "
            f"{m.agreement_rate:>9.1%}  {m.extra_detections:>8}  {m.missed_detections:>8}"
        )
    print(f"{'=' * 100}")
    print(f"\nReference detector: {ref_name}")


if __name__ == "__main__":
    main()
