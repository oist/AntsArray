#!/usr/bin/env python3
"""Frame-by-frame detection consistency analysis for custom ArUco dictionaries.

For each sampled frame, counts detections per camera and globally, then visualizes:
  1. Time-series: detections per frame (total across cameras)
  2. Per-camera detection rate bar chart
  3. Global map colored by detection rate (fraction of frames each camera detects markers)
  4. Per-frame global map snapshots showing spatial consistency

Usage:
    python nn-aruco-detection-test/visualize_frame_consistency.py \
        --video-dir "Z:/ReiterU/Ants/basler/QRcodes_test/custom_ARUCO_4x4_test/20260413" \
        --hmats "Z:/ReiterU/Ants/basler/2025_Sep_no_pertubation/calibration_dataset/set0_patterns_elevated_by_2mm/initial_H_mats.npz" \
        --npz-a nn-aruco-detection-test/results/custom_dicts/custom_4x4_A100_d4_20260410_103938.npz
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from collections import defaultdict
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


def detect_per_frame(args_tuple):
    """Worker: return per-frame detection list for one camera + one dict."""
    video_path, cam_idx, dict_spec, sample_every, max_frames, ecr = args_tuple

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
    # frame_data: list of (sample_idx, [(mid, cx, cy), ...])
    frame_data = []
    frame_idx = 0
    sample_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            dets = []
            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    c = corners[i][0]
                    cx, cy = c.mean(axis=0)
                    dets.append((int(mid), float(cx), float(cy)))
            frame_data.append((sample_idx, dets))
            sample_idx += 1
        frame_idx += 1
        if max_frames > 0 and frame_idx >= max_frames:
            break

    cap.release()
    return cam_idx, dict_spec["name"], frame_data


def main():
    p = argparse.ArgumentParser(description="Frame-by-frame detection consistency")
    p.add_argument("--video-dir", required=True)
    p.add_argument("--hmats", required=True)
    p.add_argument("--npz-a", required=True)
    p.add_argument("--npz-b", default=None)
    p.add_argument("--sample-every", type=int, default=6)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--error-correction-rate", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=25)
    p.add_argument("--output-dir", default="nn-aruco-detection-test/results/custom_dicts")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Frame-by-Frame Detection Consistency")
    print("=" * 60)

    H_stack = np.load(args.hmats)["H"]
    video_dir = Path(args.video_dir)
    videos = sorted(video_dir.glob("cam*.avi"))
    print(f"  {len(videos)} cameras, sample every {args.sample_every} frames")

    cam_map = {}
    for vp in videos:
        cam_name = vp.stem.split("_")[0]
        cam_idx = int(cam_name[3:]) - 1
        cam_map[str(vp)] = cam_idx

    cap = cv2.VideoCapture(str(videos[0]))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    n_samples = (total_frames + args.sample_every - 1) // args.sample_every
    print(f"  {total_frames} frames/video, ~{n_samples} samples, {fps:.0f} fps")

    # Build dict specs
    dict_specs = [{"name": "Custom_A", "type": "custom", "npz_path": str(args.npz_a)}]
    if args.npz_b:
        dict_specs.append({"name": "Custom_B", "type": "custom", "npz_path": str(args.npz_b)})

    # Build jobs
    jobs = []
    for vp in videos:
        cam_idx = cam_map[str(vp)]
        for ds in dict_specs:
            jobs.append((str(vp), cam_idx, ds, args.sample_every,
                         args.max_frames, args.error_correction_rate))

    n_workers = min(args.workers, len(jobs))
    print(f"  {len(jobs)} jobs, {n_workers} workers")

    t0 = time.time()
    # {dict_name: {cam_idx: [(sample_idx, [dets]), ...]}}
    raw = {ds["name"]: {} for ds in dict_specs}

    with mp.Pool(processes=n_workers) as pool:
        for i, (cam_idx, dict_name, frame_data) in enumerate(
            pool.imap_unordered(detect_per_frame, jobs)
        ):
            raw[dict_name][cam_idx] = frame_data
            total_det = sum(len(d) for _, d in frame_data)
            print(f"    [{i+1:>2}/{len(jobs)}] cam{cam_idx+1:02d}/{dict_name}: "
                  f"{len(frame_data)} frames, {total_det} det")

    print(f"  Done in {time.time()-t0:.1f}s\n")

    # -----------------------------------------------------------------------
    # Analysis per dictionary
    # -----------------------------------------------------------------------
    for dict_name in [ds["name"] for ds in dict_specs]:
        cam_frames = raw[dict_name]  # {cam_idx: [(si, [dets]), ...]}
        print(f"{'='*60}")
        print(f"Analysis: {dict_name}")
        print(f"{'='*60}")

        # 1. Build per-frame global detection count
        # Align frames by sample_idx across cameras
        max_si = max(si for cf in cam_frames.values() for si, _ in cf)
        n_frames = max_si + 1

        # Per-frame, per-camera counts
        det_per_frame_cam = np.zeros((n_frames, 25), dtype=int)
        ids_per_frame_cam = [[set() for _ in range(25)] for _ in range(n_frames)]

        for cam_idx, frame_data in cam_frames.items():
            for si, dets in frame_data:
                det_per_frame_cam[si, cam_idx] = len(dets)
                for mid, _, _ in dets:
                    ids_per_frame_cam[si][cam_idx].add(mid)

        # Global per-frame totals
        det_per_frame_global = det_per_frame_cam.sum(axis=1)
        unique_ids_per_frame = []
        for si in range(n_frames):
            all_ids = set()
            for cam_idx in range(25):
                all_ids |= ids_per_frame_cam[si][cam_idx]
            unique_ids_per_frame.append(len(all_ids))
        unique_ids_per_frame = np.array(unique_ids_per_frame)

        # Per-camera detection rate (fraction of frames with >=1 detection)
        cam_det_rate = np.zeros(25)
        cam_total_det = np.zeros(25)
        for cam_idx in range(25):
            n_with_det = np.sum(det_per_frame_cam[:, cam_idx] > 0)
            cam_det_rate[cam_idx] = n_with_det / n_frames
            cam_total_det[cam_idx] = det_per_frame_cam[:, cam_idx].sum()

        time_axis = np.arange(n_frames) * args.sample_every / fps  # seconds

        print(f"  Frames analyzed: {n_frames}")
        print(f"  Global detections/frame: mean={det_per_frame_global.mean():.1f}, "
              f"std={det_per_frame_global.std():.1f}, "
              f"min={det_per_frame_global.min()}, max={det_per_frame_global.max()}")
        print(f"  Unique IDs/frame: mean={unique_ids_per_frame.mean():.1f}, "
              f"std={unique_ids_per_frame.std():.1f}, "
              f"min={unique_ids_per_frame.min()}, max={unique_ids_per_frame.max()}")

        # ---------------------------------------------------------------
        # PLOT 1: Time-series (detections + unique IDs per frame)
        # ---------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        ax1.plot(time_axis, det_per_frame_global, linewidth=0.8, color="steelblue", alpha=0.8)
        ax1.axhline(det_per_frame_global.mean(), color="red", linestyle="--",
                     linewidth=0.8, label=f"mean={det_per_frame_global.mean():.0f}")
        ax1.fill_between(time_axis, det_per_frame_global, alpha=0.2, color="steelblue")
        ax1.set_ylabel("Total detections\n(all 25 cameras)")
        ax1.legend(loc="upper right")
        ax1.set_title(f"{dict_name} - Detection Consistency Over Time", fontsize=13)
        ax1.grid(True, alpha=0.3)

        ax2.plot(time_axis, unique_ids_per_frame, linewidth=0.8, color="darkorange", alpha=0.8)
        ax2.axhline(unique_ids_per_frame.mean(), color="red", linestyle="--",
                     linewidth=0.8, label=f"mean={unique_ids_per_frame.mean():.0f}")
        ax2.fill_between(time_axis, unique_ids_per_frame, alpha=0.2, color="darkorange")
        ax2.set_ylabel("Unique IDs detected\n(global)")
        ax2.set_xlabel("Time (seconds)")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        path1 = output_dir / f"frame_consistency_timeseries_{dict_name}_{timestamp}.png"
        fig.savefig(str(path1), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path1}")

        # ---------------------------------------------------------------
        # PLOT 2: Per-camera detection rate on global map
        # ---------------------------------------------------------------
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        corners_local = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=float)
        cmap = plt.cm.RdYlGn  # Red=low, Green=high

        for cam_idx in range(25):
            H = H_stack[cam_idx]
            cg = apply_homography(corners_local, H)
            rate = cam_det_rate[cam_idx]
            total = int(cam_total_det[cam_idx])

            polygon = plt.Polygon(cg, facecolor=cmap(rate),
                                  edgecolor="black", linewidth=0.8, alpha=0.75)
            ax.add_patch(polygon)

            center = cg.mean(axis=0)
            ax.text(center[0], center[1],
                    f"cam{cam_idx+1:02d}\n{rate*100:.0f}%\n({total})",
                    ha="center", va="center", fontsize=7, fontweight="bold")

        ax.set_xlim(-7500, 11500)
        ax.set_ylim(-5800, 8800)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("Global X (px)")
        ax.set_ylabel("Global Y (px)")
        ax.set_title(f"{dict_name} - Detection Rate per Camera\n"
                     f"(% of frames with >= 1 detection)", fontsize=13)
        ax.grid(True, alpha=0.2)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Detection rate (fraction of frames)", shrink=0.6)

        fig.tight_layout()
        path2 = output_dir / f"frame_consistency_camrate_{dict_name}_{timestamp}.png"
        fig.savefig(str(path2), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path2}")

        # ---------------------------------------------------------------
        # PLOT 3: Per-camera time-series heatmap
        # ---------------------------------------------------------------
        # Show detection count per camera per frame as a 2D heatmap
        active_cams = [c for c in range(25) if cam_det_rate[c] > 0.01]
        if active_cams:
            fig, ax = plt.subplots(1, 1, figsize=(14, max(4, len(active_cams) * 0.35)))

            heatmap_data = det_per_frame_cam[:, active_cams].T  # (n_active, n_frames)
            cam_labels = [f"cam{c+1:02d}" for c in active_cams]

            im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
                           interpolation="nearest",
                           extent=[time_axis[0], time_axis[-1], len(active_cams)-0.5, -0.5])
            ax.set_yticks(range(len(active_cams)))
            ax.set_yticklabels(cam_labels, fontsize=8)
            ax.set_xlabel("Time (seconds)")
            ax.set_title(f"{dict_name} - Detections per Camera per Frame", fontsize=13)
            plt.colorbar(im, ax=ax, label="Detections in frame", shrink=0.8)

            fig.tight_layout()
            path3 = output_dir / f"frame_consistency_heatmap_{dict_name}_{timestamp}.png"
            fig.savefig(str(path3), dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path3}")

        # ---------------------------------------------------------------
        # PLOT 4: Unique IDs per camera per frame heatmap
        # ---------------------------------------------------------------
        if active_cams:
            fig, ax = plt.subplots(1, 1, figsize=(14, max(4, len(active_cams) * 0.35)))

            uid_data = np.zeros((len(active_cams), n_frames), dtype=int)
            for ai, cam_idx in enumerate(active_cams):
                for si in range(n_frames):
                    uid_data[ai, si] = len(ids_per_frame_cam[si][cam_idx])

            im = ax.imshow(uid_data, aspect="auto", cmap="viridis",
                           interpolation="nearest",
                           extent=[time_axis[0], time_axis[-1], len(active_cams)-0.5, -0.5])
            ax.set_yticks(range(len(active_cams)))
            ax.set_yticklabels(cam_labels, fontsize=8)
            ax.set_xlabel("Time (seconds)")
            ax.set_title(f"{dict_name} - Unique IDs per Camera per Frame", fontsize=13)
            plt.colorbar(im, ax=ax, label="Unique IDs in frame", shrink=0.8)

            fig.tight_layout()
            path4 = output_dir / f"frame_consistency_ids_heatmap_{dict_name}_{timestamp}.png"
            fig.savefig(str(path4), dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path4}")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
