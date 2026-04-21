#!/usr/bin/env python3
"""SLEAP-crop rescue ablation.

Compares rescue arms for frames where full-frame OpenCV misses a marker
but SLEAP detects an ant body:

  Arm A: SLEAP center crop -> aggressive OpenCV
  Arm B: SLEAP body-axis-aligned rotated crop -> aggressive OpenCV
  Arm C: SLEAP crop + offset/scale bank -> aggressive OpenCV (small sweep)
  Arm D: (optional) YOLO on SLEAP ROI -> aggressive OpenCV

Uses the NEST videos (cam04/05/09/10) where both SLEAP and ArUco data exist.

For each rescue attempt, logs:
  - whether a marker was decoded
  - decoded ID
  - rectification quality (border contrast + cell-mean variance)
  - crop geometry used

Decision rule: if SLEAP rescue recovers a meaningful fraction of misses
with near-zero wrong IDs and without YOLO, it becomes the rescue path.

Usage:
    python nn-aruco-detection-test/sleap_rescue_ablation.py \\
        --video "Z:\\...\\cam04*.avi" \\
        --data-dir "Z:\\...\\data" \\
        --n-frames 200
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
import pandas as pd
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
# OpenCV detector profiles
# ---------------------------------------------------------------------------

def make_baseline_detector() -> aruco.ArucoDetector:
    d = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    p = aruco.DetectorParameters()
    p.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    p.adaptiveThreshConstant = 3
    p.adaptiveThreshWinSizeMin = 10
    p.adaptiveThreshWinSizeMax = 40
    p.adaptiveThreshWinSizeStep = 10
    p.errorCorrectionRate = 1.0
    p.minMarkerPerimeterRate = 0.03
    p.maxMarkerPerimeterRate = 4.0
    return aruco.ArucoDetector(d, p)


def make_aggressive_detector() -> aruco.ArucoDetector:
    d = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    p = aruco.DetectorParameters()
    p.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    p.adaptiveThreshConstant = 3
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 80
    p.adaptiveThreshWinSizeStep = 3
    p.minMarkerPerimeterRate = 0.01
    p.maxMarkerPerimeterRate = 4.0
    p.errorCorrectionRate = 0.8
    p.relativeCornerRefinmentWinSize = 0.5
    p.perspectiveRemovePixelPerCell = 8
    p.perspectiveRemoveIgnoredMarginPerCell = 0.2
    return aruco.ArucoDetector(d, p)


# ---------------------------------------------------------------------------
# Crop strategies
# ---------------------------------------------------------------------------

def crop_center(gray: np.ndarray, cx: float, cy: float, size: int = 200) -> np.ndarray:
    """Square crop centred at (cx, cy)."""
    h, w = gray.shape[:2]
    half = size // 2
    x1 = max(0, int(cx) - half)
    y1 = max(0, int(cy) - half)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    x1 = max(0, x2 - size)
    y1 = max(0, y2 - size)
    return gray[y1:y2, x1:x2]


def crop_oriented(gray: np.ndarray, cx: float, cy: float,
                  angle_deg: float, size: int = 200) -> np.ndarray:
    """Crop aligned to body axis via rotation."""
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return crop_center(rotated, cx, cy, size)


def try_decode(crop: np.ndarray, detector: aruco.ArucoDetector) -> tuple[int, float]:
    """Try to decode a marker from a crop. Returns (id, quality_score).
    quality_score: border contrast metric (-1 if no detection)."""
    corners, ids, rejected = detector.detectMarkers(crop)
    if ids is not None and len(ids) > 0:
        # Compute quality: border contrast of the detected marker
        c = corners[0][0]  # (4,2)
        quality = _rectification_quality(crop, c)
        return int(ids.flatten()[0]), quality

    # Check rejected candidates for quality diagnostic
    if rejected is not None and len(rejected) > 0:
        c = rejected[0][0]
        quality = _rectification_quality(crop, c)
        return -1, quality

    return -1, -1.0


def _rectification_quality(gray: np.ndarray, corners_4x2: np.ndarray) -> float:
    """Quality metric: cell-mean variance + border contrast after rectification."""
    try:
        dst = np.array([[0, 0], [119, 0], [119, 119], [0, 119]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners_4x2.astype(np.float32), dst)
        rect = cv2.warpPerspective(gray, M, (120, 120))
        ch, cw = 20, 20  # 120/6
        means = np.zeros((6, 6))
        for r in range(6):
            for c in range(6):
                means[r, c] = rect[r*ch:(r+1)*ch, c*cw:(c+1)*cw].mean()
        data = means[1:5, 1:5]
        border = np.concatenate([means[0, :], means[5, :], means[1:5, 0], means[1:5, 5]])
        # Quality: high data variance (clear black/white cells) + low border intensity
        data_range = data.max() - data.min()
        border_mean = border.mean()
        # Normalize: good = high data range, low border mean relative to data
        return float(data_range / 255.0 - border_mean / 512.0)
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# SLEAP helpers
# ---------------------------------------------------------------------------

def get_sleap_anchors(sleap_df: pd.DataFrame, frame: int,
                      anchor_bp: int = 0) -> np.ndarray:
    """Get (N, 3) array of [instance_id, x, y] for anchor bodypoints in frame."""
    sub = sleap_df[(sleap_df["Frame"] == frame) & (sleap_df["Bodypoint"] == anchor_bp)]
    sub = sub.dropna(subset=["X", "Y"])
    if sub.empty:
        return np.empty((0, 3))
    return sub[["Instance", "X", "Y"]].values


def get_body_axis(sleap_df: pd.DataFrame, frame: int, instance_id: int) -> float | None:
    """Estimate body axis angle (degrees) from bodypoints.
    Uses BP0 (head) and centroid of BP2-BP6 (body) to define axis."""
    sub = sleap_df[(sleap_df["Frame"] == frame) & (sleap_df["Instance"] == instance_id)]
    sub = sub.dropna(subset=["X", "Y"])
    if len(sub) < 3:
        return None

    bp0 = sub[sub["Bodypoint"] == 0]
    body = sub[sub["Bodypoint"].isin([2, 3, 4, 5, 6])]

    if bp0.empty or body.empty:
        return None

    hx, hy = bp0.iloc[0]["X"], bp0.iloc[0]["Y"]
    bx, by = body["X"].mean(), body["Y"].mean()
    angle = np.degrees(np.arctan2(by - hy, bx - hx))
    return float(angle)


# ---------------------------------------------------------------------------
# Main ablation
# ---------------------------------------------------------------------------

def run_ablation_on_video(
    video_path: str,
    sleap_csv: str,
    aruco_csv: str,
    n_frames: int,
    det_baseline: aruco.ArucoDetector,
    det_aggressive: aruco.ArucoDetector,
    match_distance: float = 50.0,
) -> list[dict]:
    """Run the ablation on one video chunk."""

    vname = Path(video_path).stem
    sleap_df = pd.read_csv(sleap_csv)
    aruco_df = pd.read_csv(aruco_csv)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_frames, total_frames)
    indices = np.linspace(0, total_frames - 1, n, dtype=int)

    records: list[dict] = []

    # Pre-group SLEAP and ArUco by frame
    sleap_by_frame = {f: g for f, g in sleap_df.groupby("Frame")}

    for fi in tqdm(indices, desc=f"  {vname}", leave=False):
        fi = int(fi)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        # 1. Run full-frame baseline OpenCV (ground truth)
        gt_corners, gt_ids, _ = det_baseline.detectMarkers(gray)
        gt_set: set[int] = set()
        gt_positions: list[tuple[int, float, float]] = []
        if gt_ids is not None:
            for j, mid in enumerate(gt_ids.flatten()):
                c = gt_corners[j][0].mean(axis=0)
                gt_set.add(int(mid))
                gt_positions.append((int(mid), float(c[0]), float(c[1])))

        # 2. Get SLEAP anchors for this frame
        if fi not in sleap_by_frame:
            continue
        sleap_frame = sleap_by_frame[fi]
        anchors = get_sleap_anchors(sleap_frame, fi)
        if len(anchors) == 0:
            continue

        # 3. Find SLEAP ants with no nearby ArUco detection (rescue candidates)
        for inst_id, sx, sy in anchors:
            # Check if any GT detection is nearby
            has_gt = False
            nearest_gt_id = -1
            nearest_gt_dist = float("inf")
            for gid, gx, gy in gt_positions:
                d = ((sx - gx)**2 + (sy - gy)**2)**0.5
                if d < nearest_gt_dist:
                    nearest_gt_dist = d
                    nearest_gt_id = gid
                if d < match_distance:
                    has_gt = True

            # We care about BOTH matched and unmatched SLEAP ants:
            # - matched: control (should decode correctly)
            # - unmatched: rescue candidates (the interesting ones)

            rec_base = {
                "video": vname,
                "frame": fi,
                "sleap_inst": int(inst_id),
                "sleap_x": round(sx, 1),
                "sleap_y": round(sy, 1),
                "has_gt_nearby": has_gt,
                "nearest_gt_id": nearest_gt_id,
                "nearest_gt_dist": round(nearest_gt_dist, 1),
            }

            # Get body axis for oriented crop
            body_angle = get_body_axis(sleap_frame, fi, int(inst_id))

            # --- Arm A: center crop -> aggressive decode ---
            crop_a = crop_center(gray, sx, sy, size=200)
            id_a, q_a = try_decode(crop_a, det_aggressive)
            rec_base["arm_a_id"] = id_a
            rec_base["arm_a_quality"] = round(q_a, 4)

            # --- Arm B: body-axis-oriented crop -> aggressive decode ---
            if body_angle is not None:
                crop_b = crop_oriented(gray, sx, sy, body_angle, size=200)
                id_b, q_b = try_decode(crop_b, det_aggressive)
            else:
                id_b, q_b = -1, -1.0
            rec_base["arm_b_id"] = id_b
            rec_base["arm_b_quality"] = round(q_b, 4)
            rec_base["body_angle"] = round(body_angle, 1) if body_angle is not None else None

            # --- Arm C: offset/scale sweep -> aggressive decode ---
            best_c_id, best_c_quality = -1, -1.0
            for size in [150, 200, 280]:
                for dx, dy in [(0, 0), (20, 0), (-20, 0), (0, 20), (0, -20)]:
                    crop_c = crop_center(gray, sx + dx, sy + dy, size=size)
                    id_c, q_c = try_decode(crop_c, det_aggressive)
                    if id_c >= 0 and q_c > best_c_quality:
                        best_c_id, best_c_quality = id_c, q_c
                        if best_c_id >= 0:
                            break  # early exit within scale
                if best_c_id >= 0:
                    break  # early exit across scales
            rec_base["arm_c_id"] = best_c_id
            rec_base["arm_c_quality"] = round(best_c_quality, 4)

            records.append(rec_base)

    cap.release()
    return records


def main():
    p = argparse.ArgumentParser(description="SLEAP-crop rescue ablation")
    p.add_argument("--video", nargs="+", required=True, help="Full video paths (glob OK)")
    p.add_argument("--data-dir", required=True,
                   help="Directory with chunk .avi + _sleap_data.csv + _aruco_detections.csv")
    p.add_argument("--n-frames", type=int, default=200)
    p.add_argument("--output-dir", default="nn-aruco-detection-test/results")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"sleap_rescue_log_{timestamp}.txt"
    tee = _TeeLogger(log_path)
    sys.stdout = tee

    print(f"=== SLEAP Rescue Ablation ===")
    print(f"Time: {timestamp}")
    print(f"Args: {vars(args)}")
    print()

    # Expand video globs
    video_paths = []
    for pattern in args.video:
        expanded = sorted(glob.glob(pattern))
        video_paths.extend(expanded if expanded else [pattern])

    if not video_paths:
        print("[ERROR] No videos found")
        tee.close()
        return

    det_baseline = make_baseline_detector()
    det_aggressive = make_aggressive_detector()
    data_dir = Path(args.data_dir)

    all_records: list[dict] = []

    for vpath in video_paths:
        vname = Path(vpath).stem
        # Find chunk _000 data (use first chunk for the ablation)
        sleap_csv = data_dir / f"{vname}_000_sleap_data.csv"
        aruco_csv = data_dir / f"{vname}_000_aruco_detections.csv"
        chunk_video = data_dir / f"{vname}_000.avi"

        if not sleap_csv.exists():
            print(f"\n[SKIP] No SLEAP data for {vname}: {sleap_csv}")
            continue
        if not chunk_video.exists():
            print(f"\n[SKIP] No chunk video for {vname}: {chunk_video}")
            continue

        print(f"\n{'='*70}")
        print(f"Video: {vname} (chunk _000)")
        print(f"  SLEAP: {sleap_csv.name}")
        print(f"  ArUco: {aruco_csv.name if aruco_csv.exists() else 'N/A'}")
        print(f"{'='*70}")

        records = run_ablation_on_video(
            str(chunk_video), str(sleap_csv),
            str(aruco_csv) if aruco_csv.exists() else "",
            args.n_frames, det_baseline, det_aggressive,
        )
        all_records.extend(records)

        if records:
            df = pd.DataFrame(records)

            # Split into rescue candidates (no GT nearby) vs control
            rescue = df[~df["has_gt_nearby"]]
            control = df[df["has_gt_nearby"]]

            print(f"\n  Total SLEAP ants sampled: {len(df)}")
            print(f"  With GT nearby (control): {len(control)}")
            print(f"  No GT nearby (rescue candidates): {len(rescue)}")

            for label, sub in [("CONTROL", control), ("RESCUE", rescue)]:
                if sub.empty:
                    continue
                n = len(sub)
                print(f"\n  --- {label} ({n} ants) ---")
                for arm in ["a", "b", "c"]:
                    col = f"arm_{arm}_id"
                    decoded = sub[sub[col] >= 0]
                    n_decoded = len(decoded)

                    if label == "CONTROL" and len(decoded) > 0:
                        # Check correctness against nearest GT
                        correct = (decoded[col] == decoded["nearest_gt_id"]).sum()
                        wrong = n_decoded - correct
                        print(f"    Arm {arm.upper()}: decoded {n_decoded}/{n} ({100*n_decoded/n:.1f}%)  "
                              f"correct={correct} wrong={wrong}")
                    else:
                        print(f"    Arm {arm.upper()}: decoded {n_decoded}/{n} ({100*n_decoded/n:.1f}%)")

    # Save CSV
    if all_records:
        csv_path = out_dir / f"sleap_rescue_ablation_{timestamp}.csv"
        df_all = pd.DataFrame(all_records)
        df_all.to_csv(csv_path, index=False)
        print(f"\nCSV saved: {csv_path}")

        # Grand summary
        rescue_all = df_all[~df_all["has_gt_nearby"]]
        control_all = df_all[df_all["has_gt_nearby"]]

        print(f"\n{'='*70}")
        print(f"GRAND SUMMARY")
        print(f"{'='*70}")
        print(f"  Total ants: {len(df_all)}")
        print(f"  Control (GT nearby): {len(control_all)}")
        print(f"  Rescue (no GT): {len(rescue_all)}")

        for label, sub in [("CONTROL", control_all), ("RESCUE", rescue_all)]:
            if sub.empty:
                continue
            n = len(sub)
            print(f"\n  {label} ({n}):")
            for arm in ["a", "b", "c"]:
                col = f"arm_{arm}_id"
                decoded = sub[sub[col] >= 0]
                n_decoded = len(decoded)
                if label == "CONTROL" and n_decoded > 0:
                    correct = (decoded[col] == decoded["nearest_gt_id"]).sum()
                    wrong = n_decoded - correct
                    print(f"    Arm {arm.upper()}: decoded {n_decoded}/{n} ({100*n_decoded/n:.1f}%)  "
                          f"correct={correct} wrong={wrong}")
                else:
                    print(f"    Arm {arm.upper()}: decoded {n_decoded}/{n} ({100*n_decoded/n:.1f}%)")

    print(f"\nLog saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
