#!/usr/bin/env python3
"""Visualize custom ArUco detections on a global panorama map.

Projects per-camera detections to global coordinates using homography matrices,
then plots all detections on a single map showing the 25-camera grid layout.

Usage:
    python nn-aruco-detection-test/visualize_global_detection.py \
        --video-dir "Z:/ReiterU/Ants/basler/QRcodes_test/custom_ARUCO_4x4_test/20260413" \
        --hmats "Z:/ReiterU/Ants/basler/2025_Sep_no_pertubation/calibration_dataset/set0_patterns_elevated_by_2mm/initial_H_mats.npz" \
        --npz-a nn-aruco-detection-test/results/custom_dicts/custom_4x4_A100_d4_20260410_103938.npz \
        --npz-b nn-aruco-detection-test/results/custom_dicts/custom_4x4_B300_d3_20260410_103938.npz
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import cv2.aruco as aruco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_custom_dictionary(npz_path):
    data = np.load(str(npz_path), allow_pickle=True)
    d = aruco.Dictionary()
    d.bytesList = data["bytesList"]
    d.markerSize = 4
    d.maxCorrectionBits = int(data["max_correction_bits"])
    n = d.bytesList.shape[0]
    min_d = int(data["min_distance"])
    return d, n, min_d


def apply_homography(xy, H):
    pts = np.hstack([xy, np.ones((xy.shape[0], 1))])
    proj = pts @ H.T
    return proj[:, :2] / proj[:, [2]]


def detect_one_video(args_tuple):
    """Worker: detect markers in sampled frames, return (cam_idx, dict_name, detections)."""
    video_path, cam_idx, dict_spec, sample_every, max_frames, ecr = args_tuple

    # Reconstruct dictionary
    if dict_spec["type"] == "custom":
        data = np.load(dict_spec["npz_path"], allow_pickle=True)
        dictionary = aruco.Dictionary()
        dictionary.bytesList = data["bytesList"]
        dictionary.markerSize = 4
        dictionary.maxCorrectionBits = int(data["max_correction_bits"])
    else:
        dictionary = aruco.getPredefinedDictionary(dict_spec["dict_id"])

    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshConstant = 3
    params.adaptiveThreshWinSizeMin = 10
    params.adaptiveThreshWinSizeMax = 40
    params.adaptiveThreshWinSizeStep = 10
    params.errorCorrectionRate = ecr
    detector = aruco.ArucoDetector(dictionary, params)

    cap = cv2.VideoCapture(str(video_path))
    detections = []  # list of (marker_id, cx, cy)  in camera coordinates
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    c = corners[i][0]  # (4, 2) corner points
                    cx, cy = c.mean(axis=0)
                    detections.append((int(mid), float(cx), float(cy)))
        frame_idx += 1
        if max_frames > 0 and frame_idx >= max_frames:
            break

    cap.release()
    return cam_idx, dict_spec["name"], detections


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_global_map(
    all_detections: dict,  # {dict_name: list of (global_x, global_y, marker_id, cam_idx)}
    H_stack: np.ndarray,
    img_w: int, img_h: int,
    output_path: Path,
    title: str = "Custom ArUco Detection - Global Map",
):
    """Plot all detections on a global coordinate map with camera boundaries."""
    fig, axes = plt.subplots(1, len(all_detections), figsize=(10 * len(all_detections), 12))
    if len(all_detections) == 1:
        axes = [axes]

    for ax, (dict_name, dets) in zip(axes, all_detections.items()):
        # Draw camera boundaries
        corners_local = np.array([
            [0, 0], [img_w, 0], [img_w, img_h], [0, img_h], [0, 0]
        ], dtype=float)

        for cam_idx in range(25):
            H = H_stack[cam_idx]
            corners_global = apply_homography(corners_local[:, :2].reshape(-1, 2), H)
            # Close the polygon
            xs = list(corners_global[:4, 0]) + [corners_global[0, 0]]
            ys = list(corners_global[:4, 1]) + [corners_global[0, 1]]
            ax.plot(xs, ys, "k-", linewidth=0.5, alpha=0.4)

            # Camera label at center
            center = corners_global[:4].mean(axis=0)
            ax.text(center[0], center[1], f"{cam_idx+1:02d}",
                    ha="center", va="center", fontsize=7, color="gray", alpha=0.6)

        # Plot detections
        if dets:
            xs = [d[0] for d in dets]
            ys = [d[1] for d in dets]
            ids = [d[2] for d in dets]

            # Color by marker ID
            scatter = ax.scatter(xs, ys, c=ids, cmap="hsv", s=3, alpha=0.5, edgecolors="none")
            plt.colorbar(scatter, ax=ax, label="Marker ID", shrink=0.6)

        n_det = len(dets)
        n_unique = len(set(d[2] for d in dets)) if dets else 0
        short_name = dict_name.split("(")[0].strip()
        ax.set_title(f"{short_name}\n{n_det:,} detections, {n_unique} unique IDs", fontsize=11)
        ax.set_xlabel("Global X (px)")
        ax.set_ylabel("Global Y (px)")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Global map saved: {output_path}")


def plot_per_camera_heatmap(
    all_detections: dict,
    H_stack: np.ndarray,
    img_w: int, img_h: int,
    output_path: Path,
):
    """Plot detection density heatmap per camera on global map."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Camera boundaries and detection counts
    cam_counts = {}
    for dict_name, dets in all_detections.items():
        for gx, gy, mid, cam_idx in dets:
            cam_counts[cam_idx] = cam_counts.get(cam_idx, 0) + 1

    max_count = max(cam_counts.values()) if cam_counts else 1
    cmap = plt.cm.YlOrRd

    corners_local = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=float)

    for cam_idx in range(25):
        H = H_stack[cam_idx]
        corners_global = apply_homography(corners_local, H)
        count = cam_counts.get(cam_idx, 0)
        intensity = count / max_count if max_count > 0 else 0

        polygon = plt.Polygon(corners_global, facecolor=cmap(intensity),
                              edgecolor="black", linewidth=0.8, alpha=0.7)
        ax.add_patch(polygon)

        center = corners_global.mean(axis=0)
        ax.text(center[0], center[1], f"cam{cam_idx+1:02d}\n{count}",
                ha="center", va="center", fontsize=7, fontweight="bold")

    ax.set_xlim(-7500, 11500)
    ax.set_ylim(-5800, 8800)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("Global X (px)")
    ax.set_ylabel("Global Y (px)")
    ax.set_title("Detection Count per Camera (All Dictionaries Combined)", fontsize=13)
    ax.grid(True, alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_count))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Detection count", shrink=0.6)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Visualize custom ArUco detections on global map")
    p.add_argument("--video-dir", required=True)
    p.add_argument("--hmats", required=True, help="Path to H_mats.npz (25, 3, 3)")
    p.add_argument("--npz-a", required=True, help="Custom dict A NPZ")
    p.add_argument("--npz-b", required=True, help="Custom dict B NPZ")
    p.add_argument("--sample-every", type=int, default=12,
                   help="Process every Nth frame (default: 12, ~2fps)")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--error-correction-rate", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=25)
    p.add_argument("--output-dir", default="nn-aruco-detection-test/results/custom_dicts")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Global Detection Visualization")
    print("=" * 60)

    # Load homographies
    H_stack = np.load(args.hmats)["H"]  # (25, 3, 3)
    print(f"  Loaded {H_stack.shape[0]} homography matrices")

    # Find videos
    video_dir = Path(args.video_dir)
    videos = sorted(video_dir.glob("cam*.avi"))
    print(f"  Found {len(videos)} cameras")

    # Map cam names to indices (cam01 -> 0, cam25 -> 24)
    cam_map = {}
    for vp in videos:
        cam_name = vp.stem.split("_")[0]
        cam_idx = int(cam_name[3:]) - 1  # cam01 -> 0
        cam_map[str(vp)] = cam_idx

    # Get image dimensions from first video
    cap = cv2.VideoCapture(str(videos[0]))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"  Image size: {img_w} x {img_h}")

    # Dictionary specs
    dict_specs = [
        {"name": "Custom_A", "type": "custom", "npz_path": str(args.npz_a)},
        {"name": "Custom_B", "type": "custom", "npz_path": str(args.npz_b)},
    ]

    # Build jobs
    jobs = []
    for vp in videos:
        cam_idx = cam_map[str(vp)]
        for ds in dict_specs:
            jobs.append((str(vp), cam_idx, ds, args.sample_every, args.max_frames,
                         args.error_correction_rate))

    n_workers = min(args.workers, len(jobs))
    print(f"\n  Running detection: {len(jobs)} jobs, {n_workers} workers...")
    t0 = time.time()

    # Collect raw detections: {dict_name: {cam_idx: [(mid, cx, cy), ...]}}
    raw_dets = {ds["name"]: {} for ds in dict_specs}

    with mp.Pool(processes=n_workers) as pool:
        for i, (cam_idx, dict_name, dets) in enumerate(
            pool.imap_unordered(detect_one_video, jobs)
        ):
            raw_dets[dict_name][cam_idx] = dets
            print(f"    [{i+1:>2}/{len(jobs)}] cam{cam_idx+1:02d} / {dict_name}: "
                  f"{len(dets)} detections")

    elapsed = time.time() - t0
    print(f"  Detection done in {elapsed:.1f}s")

    # Project to global coordinates
    print("\n  Projecting to global coordinates...")
    global_dets = {}  # {dict_name: [(gx, gy, mid, cam_idx), ...]}

    for dict_name, cam_dets in raw_dets.items():
        projected = []
        for cam_idx, dets in cam_dets.items():
            if not dets:
                continue
            H = H_stack[cam_idx]
            xy = np.array([(cx, cy) for _, cx, cy in dets])
            gxy = apply_homography(xy, H)
            for j, (mid, _, _) in enumerate(dets):
                projected.append((gxy[j, 0], gxy[j, 1], mid, cam_idx))
        global_dets[dict_name] = projected
        print(f"    {dict_name}: {len(projected):,} global detections")

    # Plot 1: Side-by-side scatter maps
    print("\n  Generating plots...")
    plot_global_map(
        global_dets, H_stack, img_w, img_h,
        output_dir / f"global_detection_map_{timestamp}.png",
        title="Custom ArUco Detection - Global Panorama Map",
    )

    # Plot 2: Per-camera heatmap (combined)
    combined_dets = {"combined": []}
    for dets in global_dets.values():
        combined_dets["combined"].extend(dets)
    plot_per_camera_heatmap(
        combined_dets, H_stack, img_w, img_h,
        output_dir / f"global_detection_heatmap_{timestamp}.png",
    )

    # Plot 3: Individual dict maps (larger, more detail)
    for dict_name, dets in global_dets.items():
        plot_global_map(
            {dict_name: dets}, H_stack, img_w, img_h,
            output_dir / f"global_{dict_name}_map_{timestamp}.png",
            title=f"{dict_name} Detection - Global Map",
        )

    print(f"\n  All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
