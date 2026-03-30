#!/usr/bin/env python3
"""
ArUco detection parameter sweep benchmark.

Loads sample frames from a video (or image directory), runs ArUco detection
across a grid of preprocessing and detector parameter configurations, and
reports detection metrics as a CSV + summary plots.

Example:
    python aruco_benchmark.py --video cam01.avi --sample-frames 100
    python aruco_benchmark.py --image-dir frames/ --sweep-mode full
"""

import argparse
import csv
import itertools
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Baseline: mirrors run_aruco.py lines 116-124
# ---------------------------------------------------------------------------
BASELINE_DETECTOR = {
    "cornerRefinementMethod": "CORNER_REFINE_CONTOUR",
    "adaptiveThreshConstant": 3,
    "adaptiveThreshWinSizeMin": 10,
    "adaptiveThreshWinSizeMax": 40,
    "adaptiveThreshWinSizeStep": 10,
    "minMarkerPerimeterRate": 0.03,
    "maxMarkerPerimeterRate": 4.0,
    "errorCorrectionRate": 1.0,
    "polygonalApproxAccuracyRate": 0.05,
}

BASELINE_PREPROCESS = {
    "clahe": None,  # no CLAHE
    "sharpen": None,  # no sharpening
}

# ---------------------------------------------------------------------------
# Default sweep grid (one-at-a-time values)
# ---------------------------------------------------------------------------
DEFAULT_SWEEP = {
    "preprocessing": {
        "clahe": [None, {"clip_limit": 2.0, "grid": 8}, {"clip_limit": 4.0, "grid": 8}],
        "sharpen": [None, "unsharp"],
    },
    "detector_params": {
        "adaptiveThreshConstant": [3, 5, 7, 10],
        "adaptiveThreshWinSizeMin": [3, 5, 10],
        "adaptiveThreshWinSizeMax": [30, 40, 60],
        "adaptiveThreshWinSizeStep": [5, 10],
        "minMarkerPerimeterRate": [0.005, 0.01, 0.02, 0.03],
        "cornerRefinementMethod": [
            "CORNER_REFINE_CONTOUR",
            "CORNER_REFINE_APRILTAG",
            "CORNER_REFINE_SUBPIX",
        ],
        "errorCorrectionRate": [0.5, 0.8, 1.0],
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Config:
    config_id: int
    param_name: str  # which parameter is being varied (or "full" / "baseline")
    param_value: str  # human-readable value
    preprocess: dict = field(default_factory=dict)
    detector: dict = field(default_factory=dict)


@dataclass
class Metrics:
    total_detections: int = 0
    unique_ids: int = 0
    mean_detections_per_frame: float = 0.0
    std_detections_per_frame: float = 0.0
    rejected_candidates: int = 0
    runtime_sec: float = 0.0


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------
def load_frames_from_video(video_path: str, n_frames: int) -> list[np.ndarray]:
    """Extract n_frames evenly-spaced grayscale frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Cannot determine frame count for: {video_path}")

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames = []
    for idx in tqdm(indices, desc="Loading frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}")
    return frames


def load_frames_from_dir(image_dir: str) -> list[np.ndarray]:
    """Load all images from a directory as grayscale."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(
        p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts
    )
    frames = []
    for p in tqdm(paths, desc="Loading images"):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)
    print(f"Loaded {len(frames)} images from {image_dir}")
    return frames


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_frame(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply preprocessing to a grayscale frame."""
    out = gray

    # CLAHE
    clahe_cfg = cfg.get("clahe")
    if clahe_cfg is not None:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_cfg["clip_limit"],
            tileGridSize=(clahe_cfg["grid"], clahe_cfg["grid"]),
        )
        out = clahe.apply(out)

    # Sharpening (unsharp mask)
    sharpen = cfg.get("sharpen")
    if sharpen == "unsharp":
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=2)
        out = cv2.addWeighted(out, 1.5, blurred, -0.5, 0)

    return out


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
CORNER_REFINE_MAP = {
    "CORNER_REFINE_NONE": aruco.CORNER_REFINE_NONE,
    "CORNER_REFINE_SUBPIX": aruco.CORNER_REFINE_SUBPIX,
    "CORNER_REFINE_CONTOUR": aruco.CORNER_REFINE_CONTOUR,
    "CORNER_REFINE_APRILTAG": aruco.CORNER_REFINE_APRILTAG,
}


def build_detector(det_cfg: dict) -> aruco.ArucoDetector:
    """Build an ArucoDetector from a config dict."""
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    params = aruco.DetectorParameters()

    for key, val in det_cfg.items():
        if key == "cornerRefinementMethod":
            params.cornerRefinementMethod = CORNER_REFINE_MAP[val]
        else:
            setattr(params, key, val)

    return aruco.ArucoDetector(aruco_dict, params)


def run_detection(
    frames: list[np.ndarray], preprocess_cfg: dict, det_cfg: dict
) -> Metrics:
    """Run detection on all frames and return metrics."""
    detector = build_detector(det_cfg)

    total_dets = 0
    all_ids = set()
    total_rejected = 0
    per_frame_counts = []

    t0 = time.perf_counter()
    for gray in frames:
        processed = preprocess_frame(gray, preprocess_cfg)
        corners, ids, rejected = detector.detectMarkers(processed)

        n = 0 if ids is None else len(ids)
        total_dets += n
        per_frame_counts.append(n)
        total_rejected += 0 if rejected is None else len(rejected)

        if ids is not None:
            all_ids.update(ids.flatten().tolist())

    elapsed = time.perf_counter() - t0
    counts = np.array(per_frame_counts)

    return Metrics(
        total_detections=total_dets,
        unique_ids=len(all_ids),
        mean_detections_per_frame=float(counts.mean()) if len(counts) else 0.0,
        std_detections_per_frame=float(counts.std()) if len(counts) else 0.0,
        rejected_candidates=total_rejected,
        runtime_sec=round(elapsed, 3),
    )


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------
def generate_one_at_a_time(sweep: dict) -> list[Config]:
    """Vary one parameter at a time from baseline."""
    configs = []
    cid = 0

    # Baseline
    configs.append(
        Config(cid, "baseline", "baseline", dict(BASELINE_PREPROCESS), dict(BASELINE_DETECTOR))
    )
    cid += 1

    # Preprocessing params
    for pname, values in sweep.get("preprocessing", {}).items():
        for val in values:
            pre = dict(BASELINE_PREPROCESS)
            if pre.get(pname) == val:
                continue  # skip baseline duplicate
            pre[pname] = val
            val_str = json.dumps(val) if not isinstance(val, str) else val
            configs.append(Config(cid, f"pre_{pname}", val_str, pre, dict(BASELINE_DETECTOR)))
            cid += 1

    # Detector params
    for pname, values in sweep.get("detector_params", {}).items():
        for val in values:
            det = dict(BASELINE_DETECTOR)
            if det.get(pname) == val:
                continue  # skip baseline duplicate
            det[pname] = val
            val_str = str(val)
            configs.append(Config(cid, f"det_{pname}", val_str, dict(BASELINE_PREPROCESS), det))
            cid += 1

    return configs


def generate_full_grid(sweep: dict) -> list[Config]:
    """Full combinatorial grid of all parameter values."""
    pre_keys = list(sweep.get("preprocessing", {}).keys())
    pre_vals = [sweep["preprocessing"][k] for k in pre_keys]

    det_keys = list(sweep.get("detector_params", {}).keys())
    det_vals = [sweep["detector_params"][k] for k in det_keys]

    configs = []
    cid = 0
    for pre_combo in itertools.product(*pre_vals):
        pre = dict(BASELINE_PREPROCESS)
        for k, v in zip(pre_keys, pre_combo):
            pre[k] = v

        for det_combo in itertools.product(*det_vals):
            det = dict(BASELINE_DETECTOR)
            for k, v in zip(det_keys, det_combo):
                det[k] = v

            configs.append(Config(cid, "full", f"combo_{cid}", pre, det))
            cid += 1

    return configs


def generate_random_sample(sweep: dict, n_samples: int) -> list[Config]:
    """Random sample from the full grid."""
    full = generate_full_grid(sweep)
    if n_samples >= len(full):
        return full
    sampled = random.sample(full, n_samples)
    for i, c in enumerate(sampled):
        c.config_id = i
    return sampled


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
RESULT_COLUMNS = [
    "config_id",
    "param_name",
    "param_value",
    "preprocess_cfg",
    "detector_cfg",
    "total_detections",
    "unique_ids",
    "mean_detections_per_frame",
    "std_detections_per_frame",
    "rejected_candidates",
    "runtime_sec",
]


def write_results(results: list[tuple[Config, Metrics]], output_path: str):
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        for cfg, met in results:
            w.writerow(
                {
                    "config_id": cfg.config_id,
                    "param_name": cfg.param_name,
                    "param_value": cfg.param_value,
                    "preprocess_cfg": json.dumps(cfg.preprocess),
                    "detector_cfg": json.dumps(cfg.detector),
                    "total_detections": met.total_detections,
                    "unique_ids": met.unique_ids,
                    "mean_detections_per_frame": f"{met.mean_detections_per_frame:.2f}",
                    "std_detections_per_frame": f"{met.std_detections_per_frame:.2f}",
                    "rejected_candidates": met.rejected_candidates,
                    "runtime_sec": met.runtime_sec,
                }
            )
    print(f"Results saved to: {output_path}")


def plot_results(results: list[tuple[Config, Metrics]], output_dir: str):
    """Generate summary plots per swept parameter."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
        return

    # Group by param_name
    groups: dict[str, list[tuple[Config, Metrics]]] = {}
    for cfg, met in results:
        groups.setdefault(cfg.param_name, []).append((cfg, met))

    for pname, entries in groups.items():
        if pname in ("baseline", "full"):
            continue
        if len(entries) < 2:
            continue

        labels = [e[0].param_value for e in entries]
        means = [e[1].mean_detections_per_frame for e in entries]
        stds = [e[1].std_detections_per_frame for e in entries]
        runtimes = [e[1].runtime_sec for e in entries]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(pname)

        x = range(len(labels))
        ax1.bar(x, means, yerr=stds, capsize=4)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("Mean detections / frame")
        ax1.set_title("Detection count")

        ax2.bar(x, runtimes)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Runtime (sec)")
        ax2.set_title("Speed")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"sweep_{pname}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved: {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="ArUco detection parameter benchmark")

    # Input
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--video", type=str, help="Path to input video file")
    inp.add_argument("--image-dir", type=str, help="Directory of images to use as frames")

    p.add_argument(
        "--sample-frames",
        type=int,
        default=100,
        help="Number of frames to sample from video (default: 100)",
    )

    # Sweep mode
    p.add_argument(
        "--sweep-mode",
        choices=["one-at-a-time", "full", "random"],
        default="one-at-a-time",
        help="Sweep strategy (default: one-at-a-time)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of random samples for --sweep-mode random (default: 100)",
    )
    p.add_argument(
        "--sweep-config",
        type=str,
        default=None,
        help="JSON file with custom sweep grid (overrides defaults)",
    )

    # Output
    p.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output CSV and plots (default: current dir)",
    )

    args = p.parse_args()

    # Load frames
    if args.video:
        frames = load_frames_from_video(args.video, args.sample_frames)
    else:
        frames = load_frames_from_dir(args.image_dir)

    if not frames:
        print("[ERROR] No frames loaded")
        return

    # Load sweep grid
    if args.sweep_config:
        with open(args.sweep_config) as f:
            sweep = json.load(f)
    else:
        sweep = DEFAULT_SWEEP

    # Generate configs
    if args.sweep_mode == "one-at-a-time":
        configs = generate_one_at_a_time(sweep)
    elif args.sweep_mode == "full":
        configs = generate_full_grid(sweep)
    else:
        configs = generate_random_sample(sweep, args.n_samples)

    print(f"\nRunning {len(configs)} configurations on {len(frames)} frames\n")

    # Run sweep
    results = []
    for cfg in tqdm(configs, desc="Sweep"):
        met = run_detection(frames, cfg.preprocess, cfg.detector)
        results.append((cfg, met))

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    write_results(results, csv_path)
    plot_results(results, args.output_dir)

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'ID':>4}  {'Parameter':<30}  {'Value':<20}  {'Mean det/f':>10}  {'Unique IDs':>10}  {'Time(s)':>8}")
    print(f"{'-'*90}")
    for cfg, met in results:
        print(
            f"{cfg.config_id:>4}  {cfg.param_name:<30}  {cfg.param_value:<20}  "
            f"{met.mean_detections_per_frame:>10.2f}  {met.unique_ids:>10}  {met.runtime_sec:>8.2f}"
        )
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
