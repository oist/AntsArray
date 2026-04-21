#!/usr/bin/env python3
"""Build a manually-labelable disagreement set between detectors.

Runs two detectors (typically ``opencv`` and ``yolo-cascade``) on sampled
video frames and extracts crops for every disagreement.  The output is a
directory of ``.png`` + ``.json`` pairs that can be reviewed with
``review_gui.py``.

Categories:
    YOLO_ONLY     — YOLO found a marker, OpenCV did not
    OPENCV_ONLY   — OpenCV found a marker, YOLO did not
    ID_DISAGREE   — both found a marker at the same location, different IDs
    UNDECODED     — Hybrid returned id=-1 near a real OpenCV position
    EDGE_CASE     — detection within 100 px of frame edge

Usage:
    python nn-aruco-detection-test/build_disagreement_set.py \\
        --video "Z:\\...\\cam04*.avi" \\
        --yolo-weights runs/detect/.../best.pt \\
        --n-frames 200 \\
        --output-dir nn-aruco-detection-test/disagreement_set
"""

from __future__ import annotations

import argparse
import json
import glob
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TeeLogger:
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


DISTANCE_THRESH = 50.0
EDGE_MARGIN = 100
CROP_SIZE = 128


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _crop_around(frame: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    """Extract a square crop centred at (x, y), clamped to frame bounds."""
    h, w = frame.shape[:2]
    half = size // 2
    x1 = max(0, int(x) - half)
    y1 = max(0, int(y) - half)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    x1 = max(0, x2 - size)
    y1 = max(0, y2 - size)
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] != size or crop.shape[1] != size:
        padded = np.zeros((size, size, 3), dtype=np.uint8) if frame.ndim == 3 else np.zeros((size, size), dtype=np.uint8)
        ph, pw = crop.shape[:2]
        padded[:ph, :pw] = crop
        return padded
    return crop


def _is_edge(x: float, y: float, w: int, h: int, margin: int = EDGE_MARGIN) -> bool:
    return x < margin or y < margin or x > w - margin or y > h - margin


def _nearest_distance(x: float, y: float, dets: list[dict]) -> tuple[float, dict | None]:
    """Distance to nearest detection in *dets*. Returns (dist, det)."""
    if not dets:
        return float("inf"), None
    dists = [((d["x"] - x) ** 2 + (d["y"] - y) ** 2) ** 0.5 for d in dets]
    idx = int(np.argmin(dists))
    return dists[idx], dets[idx]


# ------------------------------------------------------------------
# Core
# ------------------------------------------------------------------

def extract_disagreements(
    video_path: str,
    opencv_detector,
    hybrid_detector,
    n_frames: int = 200,
    output_dir: str = "nn-aruco-detection-test/disagreement_set",
    crop_size: int = CROP_SIZE,
) -> pd.DataFrame:
    """Run both detectors on *n_frames* and save disagreement crops."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    vname = Path(video_path).stem
    records: list[dict] = []
    crop_idx = 0

    for fi in tqdm(indices, desc=f"Disagreements [{vname}]"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue

        fh, fw = frame.shape[:2]

        # Run both detectors
        ocv_dets = opencv_detector.detect(frame)
        hyb_dets = hybrid_detector.detect(frame)

        # Convert to dicts for easier processing
        ocv = [{"id": d.marker_id, "x": d.x, "y": d.y, "conf": d.confidence} for d in ocv_dets]
        hyb = [{"id": d.marker_id, "x": d.x, "y": d.y, "conf": d.confidence} for d in hyb_dets]

        # Track which detections are matched
        ocv_matched = [False] * len(ocv)
        hyb_matched = [False] * len(hyb)

        # Match detections by proximity
        for hi, hd in enumerate(hyb):
            dist, nearest = _nearest_distance(hd["x"], hd["y"], ocv)
            if nearest is None or dist > DISTANCE_THRESH:
                continue
            oi = ocv.index(nearest)
            hyb_matched[hi] = True
            ocv_matched[oi] = True

            if hd["id"] == -1:
                # UNDECODED — YOLO found region, OpenCV decoded nearby
                cat = "UNDECODED"
            elif hd["id"] != nearest["id"]:
                # ID_DISAGREE — different IDs at same location
                cat = "ID_DISAGREE"
            else:
                # Agreement — skip
                continue

            meta = {
                "video": vname,
                "frame": int(fi),
                "category": cat,
                "hybrid_id": hd["id"],
                "opencv_id": nearest["id"],
                "x": hd["x"],
                "y": hd["y"],
                "distance": round(dist, 1),
                "edge": _is_edge(hd["x"], hd["y"], fw, fh),
            }
            fname = f"{vname}_f{fi}_{cat}_{crop_idx}"
            crop = _crop_around(frame, hd["x"], hd["y"], crop_size)
            cv2.imwrite(str(out / f"{fname}.png"), crop)
            with open(out / f"{fname}.json", "w") as f:
                json.dump(meta, f, indent=2)
            records.append(meta)
            crop_idx += 1

        # YOLO_ONLY — hybrid found, opencv didn't
        for hi, hd in enumerate(hyb):
            if hyb_matched[hi]:
                continue
            dist, _ = _nearest_distance(hd["x"], hd["y"], ocv)
            if dist <= DISTANCE_THRESH:
                continue

            cat = "EDGE_CASE" if _is_edge(hd["x"], hd["y"], fw, fh) else "YOLO_ONLY"
            meta = {
                "video": vname,
                "frame": int(fi),
                "category": cat,
                "hybrid_id": hd["id"],
                "opencv_id": None,
                "x": hd["x"],
                "y": hd["y"],
                "distance": round(dist, 1),
                "edge": _is_edge(hd["x"], hd["y"], fw, fh),
            }
            fname = f"{vname}_f{fi}_{cat}_{crop_idx}"
            crop = _crop_around(frame, hd["x"], hd["y"], crop_size)
            cv2.imwrite(str(out / f"{fname}.png"), crop)
            with open(out / f"{fname}.json", "w") as f:
                json.dump(meta, f, indent=2)
            records.append(meta)
            crop_idx += 1

        # OPENCV_ONLY — opencv found, hybrid didn't
        for oi, od in enumerate(ocv):
            if ocv_matched[oi]:
                continue
            dist, _ = _nearest_distance(od["x"], od["y"], [h for h in hyb])
            if dist <= DISTANCE_THRESH:
                continue

            cat = "OPENCV_ONLY"
            meta = {
                "video": vname,
                "frame": int(fi),
                "category": cat,
                "hybrid_id": None,
                "opencv_id": od["id"],
                "x": od["x"],
                "y": od["y"],
                "distance": round(dist, 1),
                "edge": _is_edge(od["x"], od["y"], fw, fh),
            }
            fname = f"{vname}_f{fi}_{cat}_{crop_idx}"
            crop = _crop_around(frame, od["x"], od["y"], crop_size)
            cv2.imwrite(str(out / f"{fname}.png"), crop)
            with open(out / f"{fname}.json", "w") as f:
                json.dump(meta, f, indent=2)
            records.append(meta)
            crop_idx += 1

    cap.release()

    df = pd.DataFrame(records)
    if not df.empty:
        summary_path = out / f"{vname}_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\n  {vname}: {len(df)} disagreements saved to {out}")
        print(f"  Category breakdown:")
        print(df["category"].value_counts().to_string(header=False))
    return df


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Build disagreement set between OpenCV and YOLO-cascade detectors."
    )
    p.add_argument("--video", nargs="+", required=True, help="Video paths (glob OK)")
    p.add_argument("--yolo-weights", required=True, help="YOLO model weights")
    p.add_argument(
        "--n-frames", type=int, default=200, help="Frames to sample per video"
    )
    p.add_argument(
        "--output-dir",
        default="nn-aruco-detection-test/disagreement_set",
        help="Output directory for crops and metadata",
    )
    p.add_argument("--crop-size", type=int, default=CROP_SIZE)
    p.add_argument("--whitelist", type=str, default=None, help="Whitelist JSON path")
    args = p.parse_args()

    # Set up logging — tee all output to a timestamped log file
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"disagreement_log_{timestamp}.txt"
    tee = TeeLogger(log_path)
    sys.stdout = tee

    print(f"=== Disagreement Set Builder ===")
    print(f"Time: {timestamp}")
    print(f"Args: {vars(args)}")
    print()

    from aruco_detection.nn_detection.opencv_baseline import OpenCVArucoDetector
    from aruco_detection.nn_detection.yolo_cascade_hybrid import (
        YOLOCascadeHybridDetector,
    )

    whitelist = None
    if args.whitelist:
        from aruco_detection.nn_detection.whitelist import load_whitelist
        whitelist = load_whitelist(args.whitelist)
        print(f"Whitelist: {len(whitelist)} IDs")

    opencv_det = OpenCVArucoDetector()
    cascade_det = YOLOCascadeHybridDetector(
        yolo_weights=args.yolo_weights,
        whitelist=whitelist,
    )

    # Expand globs
    video_paths: list[str] = []
    for pattern in args.video:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            video_paths.extend(expanded)
        else:
            video_paths.append(pattern)

    all_dfs = []
    for vp in video_paths:
        df = extract_disagreements(
            video_path=vp,
            opencv_detector=opencv_det,
            hybrid_detector=cascade_det,
            n_frames=args.n_frames,
            output_dir=args.output_dir,
            crop_size=args.crop_size,
        )
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(
            Path(args.output_dir) / "all_disagreements.csv", index=False
        )
        print(f"\nTotal: {len(combined)} disagreements across {len(video_paths)} videos")
        print(combined["category"].value_counts().to_string(header=False))

    print(f"\nLog saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
