#!/usr/bin/env python3
"""Cross-check precision of SLEAP rescue decodes.

For the 897 Arm C rescue decodes from sleap_rescue_ablation.py, estimate
correctness via:

1. **Tracklet consistency**: same SLEAP instance across frames — does it
   decode to the same ID?  Stable = high confidence.
2. **YOLO agreement**: run YOLO+aggressive on the rescue crops — does
   YOLO find the same marker and decode the same ID?
3. **Spatial consistency**: do nearby full-frame OpenCV detections in
   adjacent frames match the rescue ID?

Outputs four confidence buckets:
  - YOLO_AGREE + TRACK_STABLE: highest confidence
  - TRACK_STABLE only
  - YOLO_AGREE only
  - NEITHER: lowest confidence

Usage:
    python nn-aruco-detection-test/rescue_crosscheck.py \\
        --ablation-csv nn-aruco-detection-test/results/sleap_rescue_ablation_*.csv \\
        --data-dir "Z:\\...\\data" \\
        --yolo-weights runs/detect/.../best.pt
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter
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
# Helpers
# ---------------------------------------------------------------------------

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


def crop_center(gray: np.ndarray, cx: float, cy: float, size: int = 200) -> np.ndarray:
    h, w = gray.shape[:2]
    half = size // 2
    x1 = max(0, int(cx) - half)
    y1 = max(0, int(cy) - half)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    x1 = max(0, x2 - size)
    y1 = max(0, y2 - size)
    return gray[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# 1. Tracklet consistency
# ---------------------------------------------------------------------------

def compute_tracklet_consistency(rescue_df: pd.DataFrame) -> pd.DataFrame:
    """For each SLEAP instance, check if the decoded ID is consistent
    across frames.  Adds columns: track_majority_id, track_consistency,
    track_n_frames, track_stable."""

    results = []
    for (video, inst), group in rescue_df.groupby(["video", "sleap_inst"]):
        decoded = group[group["arm_c_id"] >= 0]
        n_total = len(group)
        n_decoded = len(decoded)

        if n_decoded == 0:
            for _, row in group.iterrows():
                results.append({
                    "idx": row.name,
                    "track_majority_id": -1,
                    "track_consistency": 0.0,
                    "track_n_decoded": 0,
                    "track_n_total": n_total,
                    "track_stable": False,
                })
            continue

        counts = Counter(decoded["arm_c_id"])
        majority_id, majority_count = counts.most_common(1)[0]
        consistency = majority_count / n_decoded

        for _, row in group.iterrows():
            results.append({
                "idx": row.name,
                "track_majority_id": int(majority_id),
                "track_consistency": round(consistency, 3),
                "track_n_decoded": n_decoded,
                "track_n_total": n_total,
                "track_stable": consistency >= 0.8 and n_decoded >= 2,
            })

    return pd.DataFrame(results).set_index("idx")


# ---------------------------------------------------------------------------
# 2. YOLO agreement
# ---------------------------------------------------------------------------

def check_yolo_agreement(
    rescue_decoded: pd.DataFrame,
    data_dir: Path,
    yolo_model,
    det_aggressive: aruco.ArucoDetector,
    yolo_conf: float = 0.25,
    match_distance: float = 50.0,
) -> dict[int, dict]:
    """For each rescue decode, check if YOLO finds the same marker nearby.

    Returns {index: {"yolo_id": int, "yolo_agrees": bool, "yolo_dist": float}}
    """
    results = {}

    for video_name, group in rescue_decoded.groupby("video"):
        # Open the chunk video
        chunk_path = data_dir / f"{video_name}.avi"
        if not chunk_path.exists():
            for _, row in group.iterrows():
                results[row.name] = {"yolo_id": -1, "yolo_agrees": False, "yolo_dist": -1}
            continue

        cap = cv2.VideoCapture(str(chunk_path))
        if not cap.isOpened():
            for _, row in group.iterrows():
                results[row.name] = {"yolo_id": -1, "yolo_agrees": False, "yolo_dist": -1}
            continue

        # Group by frame for efficiency
        for frame_idx, frame_group in group.groupby("frame"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                for _, row in frame_group.iterrows():
                    results[row.name] = {"yolo_id": -1, "yolo_agrees": False, "yolo_dist": -1}
                continue

            # Run YOLO on a region around each rescue position
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

            for _, row in frame_group.iterrows():
                sx, sy = row["sleap_x"], row["sleap_y"]
                rescue_id = int(row["arm_c_id"])

                # Extract a larger crop for YOLO (400x400)
                fh, fw = frame.shape[:2]
                half = 200
                x1 = max(0, int(sx) - half)
                y1 = max(0, int(sy) - half)
                x2 = min(fw, x1 + 400)
                y2 = min(fh, y1 + 400)
                roi = frame[y1:y2, x1:x2]

                if roi.size == 0:
                    results[row.name] = {"yolo_id": -1, "yolo_agrees": False, "yolo_dist": -1}
                    continue

                # Run YOLO on the ROI
                yolo_results = yolo_model.predict(roi, conf=yolo_conf, verbose=False)
                best_yolo_id = -1
                best_yolo_dist = float("inf")

                for res in yolo_results:
                    if res.boxes is None:
                        continue
                    for i in range(len(res.boxes)):
                        xyxy = res.boxes.xyxy[i].cpu().numpy()
                        bx1, by1, bx2, by2 = xyxy
                        # Center of YOLO box in ROI coords
                        bcx = (bx1 + bx2) / 2 + x1
                        bcy = (by1 + by2) / 2 + y1
                        d = ((bcx - sx)**2 + (bcy - sy)**2)**0.5

                        if d < best_yolo_dist:
                            # Try to decode from YOLO crop
                            bw, bh = bx2 - bx1, by2 - by1
                            pad = 0.5
                            cx1 = max(0, int(bx1 - bw * pad))
                            cy1 = max(0, int(by1 - bh * pad))
                            cx2 = min(roi.shape[1], int(bx2 + bw * pad))
                            cy2 = min(roi.shape[0], int(by2 + bh * pad))
                            yolo_crop = roi[cy1:cy2, cx1:cx2]
                            if yolo_crop.ndim == 3:
                                yolo_crop = cv2.cvtColor(yolo_crop, cv2.COLOR_BGR2GRAY)
                            corners, ids, _ = det_aggressive.detectMarkers(yolo_crop)
                            if ids is not None and len(ids) > 0:
                                yid = int(ids.flatten()[0])
                                if d < best_yolo_dist:
                                    best_yolo_id = yid
                                    best_yolo_dist = d

                agrees = best_yolo_id >= 0 and best_yolo_id == rescue_id and best_yolo_dist < match_distance
                results[row.name] = {
                    "yolo_id": best_yolo_id,
                    "yolo_agrees": agrees,
                    "yolo_dist": round(best_yolo_dist, 1) if best_yolo_dist < 9999 else -1,
                }

        cap.release()

    return results


# ---------------------------------------------------------------------------
# 3. Spatial consistency (nearby GT in adjacent frames)
# ---------------------------------------------------------------------------

def check_spatial_consistency(
    rescue_decoded: pd.DataFrame,
    all_ablation: pd.DataFrame,
    max_spatial_dist: float = 80.0,
    max_frame_gap: int = 2000,
) -> dict[int, dict]:
    """For each rescue decode, check if control ants nearby in space/time
    have the same ID."""

    control = all_ablation[all_ablation["has_gt_nearby"]].copy()
    results = {}

    for _, row in rescue_decoded.iterrows():
        video = row["video"]
        frame = row["frame"]
        sx, sy = row["sleap_x"], row["sleap_y"]
        rescue_id = int(row["arm_c_id"])

        # Find control ants in same video, nearby in space, close in time
        same_video = control[control["video"] == video]
        nearby_time = same_video[abs(same_video["frame"] - frame) <= max_frame_gap]
        if nearby_time.empty:
            results[row.name] = {"spatial_match": False, "spatial_nearest_id": -1, "spatial_dist": -1}
            continue

        dists = np.sqrt(
            (nearby_time["sleap_x"].values - sx)**2 +
            (nearby_time["sleap_y"].values - sy)**2
        )
        close = nearby_time[dists < max_spatial_dist]

        if close.empty:
            results[row.name] = {"spatial_match": False, "spatial_nearest_id": -1, "spatial_dist": -1}
            continue

        # What IDs do the nearby control ants have?
        nearby_ids = close["nearest_gt_id"].value_counts()
        top_id = int(nearby_ids.index[0])
        spatial_match = top_id == rescue_id

        results[row.name] = {
            "spatial_match": spatial_match,
            "spatial_nearest_id": top_id,
            "spatial_dist": round(float(dists[dists < max_spatial_dist].min()), 1),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Cross-check SLEAP rescue decode precision")
    p.add_argument("--ablation-csv", required=True, help="Path to sleap_rescue_ablation CSV")
    p.add_argument("--data-dir", required=True, help="Directory with chunk videos")
    p.add_argument("--yolo-weights", default=None, help="YOLO weights (optional)")
    p.add_argument("--output-dir", default="nn-aruco-detection-test/results")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"rescue_crosscheck_log_{timestamp}.txt"
    tee = _TeeLogger(log_path)
    sys.stdout = tee

    print(f"=== Rescue Decode Cross-Check ===")
    print(f"Time: {timestamp}")
    print(f"Args: {vars(args)}")
    print()

    # Load ablation data
    df = pd.read_csv(args.ablation_csv)
    rescue = df[~df["has_gt_nearby"]].copy()
    rescue_decoded = rescue[rescue["arm_c_id"] >= 0].copy()

    print(f"Total ablation rows: {len(df)}")
    print(f"Rescue candidates: {len(rescue)}")
    print(f"Arm C decoded (to cross-check): {len(rescue_decoded)}")
    print(f"Unique decoded IDs: {sorted(rescue_decoded['arm_c_id'].unique())}")
    print()

    data_dir = Path(args.data_dir)

    # 1. Tracklet consistency
    print("=" * 60)
    print("1. TRACKLET CONSISTENCY")
    print("=" * 60)
    track_df = compute_tracklet_consistency(rescue)
    rescue_decoded = rescue_decoded.join(track_df, how="left")

    n_stable = rescue_decoded["track_stable"].sum()
    n_decoded = len(rescue_decoded)
    print(f"  Track-stable (>=80% same ID, >=2 frames): {n_stable}/{n_decoded} "
          f"({100*n_stable/n_decoded:.1f}%)")

    # Distribution of consistency scores
    decoded_with_tracks = rescue_decoded[rescue_decoded["track_n_decoded"] >= 2]
    if not decoded_with_tracks.empty:
        print(f"  Multi-decode tracks: {len(decoded_with_tracks)}")
        bins = [0, 0.5, 0.8, 0.95, 1.01]
        labels = ["<50%", "50-80%", "80-95%", "95-100%"]
        cons = pd.cut(decoded_with_tracks["track_consistency"], bins, labels=labels)
        print("  Consistency distribution:")
        for label in labels:
            cnt = (cons == label).sum()
            print(f"    {label}: {cnt}")

    # 2. YOLO agreement (if weights provided)
    yolo_agree_col = None
    if args.yolo_weights:
        print()
        print("=" * 60)
        print("2. YOLO AGREEMENT")
        print("=" * 60)

        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo_weights)
        det_aggressive = make_aggressive_detector()

        yolo_results = check_yolo_agreement(
            rescue_decoded, data_dir, yolo_model, det_aggressive
        )
        yolo_df = pd.DataFrame.from_dict(yolo_results, orient="index")
        rescue_decoded = rescue_decoded.join(yolo_df, how="left")

        n_yolo_agree = rescue_decoded["yolo_agrees"].sum()
        n_yolo_found = (rescue_decoded["yolo_id"] >= 0).sum()
        print(f"  YOLO found marker nearby: {n_yolo_found}/{n_decoded} ({100*n_yolo_found/n_decoded:.1f}%)")
        print(f"  YOLO agrees on ID: {n_yolo_agree}/{n_decoded} ({100*n_yolo_agree/n_decoded:.1f}%)")
        if n_yolo_found > 0:
            n_disagree = n_yolo_found - n_yolo_agree
            print(f"  YOLO disagrees: {n_disagree}")
        yolo_agree_col = "yolo_agrees"
    else:
        print("\n[SKIP] No YOLO weights provided, skipping YOLO agreement check")
        rescue_decoded["yolo_agrees"] = False
        yolo_agree_col = "yolo_agrees"

    # 3. Spatial consistency
    print()
    print("=" * 60)
    print("3. SPATIAL CONSISTENCY")
    print("=" * 60)
    spatial_results = check_spatial_consistency(rescue_decoded, df)
    spatial_df = pd.DataFrame.from_dict(spatial_results, orient="index")
    rescue_decoded = rescue_decoded.join(spatial_df, how="left")

    n_spatial = rescue_decoded["spatial_match"].sum()
    print(f"  Spatial match (nearby control has same ID): {n_spatial}/{n_decoded} "
          f"({100*n_spatial/n_decoded:.1f}%)")

    # 4. Confidence buckets
    print()
    print("=" * 60)
    print("4. CONFIDENCE BUCKETS")
    print("=" * 60)

    def bucket(row):
        yolo = bool(row.get("yolo_agrees", False))
        stable = bool(row.get("track_stable", False))
        spatial = bool(row.get("spatial_match", False))
        if (yolo or spatial) and stable:
            return "HIGH (agree+stable)"
        elif stable:
            return "MEDIUM (stable only)"
        elif yolo or spatial:
            return "LOW (agree only)"
        else:
            return "NONE (neither)"

    rescue_decoded["confidence_bucket"] = rescue_decoded.apply(bucket, axis=1)

    bucket_counts = rescue_decoded["confidence_bucket"].value_counts()
    print(f"\n  Bucket distribution ({n_decoded} rescue decodes):")
    for b in ["HIGH (agree+stable)", "MEDIUM (stable only)", "LOW (agree only)", "NONE (neither)"]:
        cnt = bucket_counts.get(b, 0)
        pct = 100 * cnt / n_decoded if n_decoded else 0
        print(f"    {b}: {cnt} ({pct:.1f}%)")

    # Per-bucket ID distribution
    for b in ["HIGH (agree+stable)", "MEDIUM (stable only)", "LOW (agree only)", "NONE (neither)"]:
        sub = rescue_decoded[rescue_decoded["confidence_bucket"] == b]
        if sub.empty:
            continue
        ids = sub["arm_c_id"].value_counts().head(10)
        print(f"\n  {b} — top IDs:")
        for mid, cnt in ids.items():
            print(f"    ID {int(mid)}: {cnt}")

    # Save enriched CSV
    csv_path = out_dir / f"rescue_crosscheck_{timestamp}.csv"
    rescue_decoded.to_csv(csv_path, index=True)
    print(f"\nCSV saved: {csv_path}")

    # Decision summary
    print()
    print("=" * 60)
    print("DECISION SUMMARY")
    print("=" * 60)
    high = bucket_counts.get("HIGH (agree+stable)", 0)
    medium = bucket_counts.get("MEDIUM (stable only)", 0)
    low = bucket_counts.get("LOW (agree only)", 0)
    none_ = bucket_counts.get("NONE (neither)", 0)

    print(f"  Ship immediately (HIGH): {high}")
    print(f"  Include with stricter thresholds (MEDIUM): {medium}")
    print(f"  Needs manual review (LOW): {low}")
    print(f"  Abstain (NONE): {none_}")

    if high + medium > n_decoded * 0.5:
        print(f"\n  VERDICT: SLEAP rescue is production-viable. "
              f"{high+medium}/{n_decoded} ({100*(high+medium)/n_decoded:.0f}%) are confident.")
    else:
        print(f"\n  VERDICT: SLEAP rescue needs more validation. "
              f"Only {high+medium}/{n_decoded} ({100*(high+medium)/n_decoded:.0f}%) are confident.")

    print(f"\nLog saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
