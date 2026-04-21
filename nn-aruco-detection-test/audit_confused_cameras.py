#!/usr/bin/env python3
"""Bit-level audit for cameras with systematic wrong-ID detections.

For each sampled frame:
1. Run OpenCV ArUco detection
2. For each detected marker:
   - Extract the padded crop
   - Rectify the marker using detected corners
   - Overlay the 6x6 grid (border + 4x4 data)
   - Compute per-cell intensity means
   - Binarize the 4x4 payload
   - Compare against canonical patterns for the "true" ID, the detected ID,
     and all 1000 IDs under all 4 rotations + mirror/transpose
   - Report Hamming distances and margin

Outputs per camera:
  - audit_crops/  — annotated images (original crop, rectified with grid overlay)
  - audit_summary.csv — per-detection breakdown
  - audit_log_YYYYMMDD_HHMMSS.txt — full console output

Usage:
    python nn-aruco-detection-test/audit_confused_cameras.py \\
        --video "Z:\\...\\cam02*.avi" "Z:\\...\\cam06*.avi" ... \\
        --n-frames 30 \\
        --output-dir nn-aruco-detection-test/audit
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
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


# Ground truth from benchmark_single_ant.py
GROUND_TRUTH = {
    "cam01": 25, "cam02": 25, "cam03": 3, "cam04": 3, "cam05": 3,
    "cam06": 3, "cam07": 17, "cam08": 17, "cam09": 3, "cam10": 3,
    "cam11": 25, "cam12": 25, "cam13": 17, "cam14": 17, "cam15": 17,
    "cam16": 17, "cam17": 17, "cam18": 25, "cam19": 25,
}


def get_canonical_patterns():
    """Extract (1000, 4, 4) canonical bit patterns from DICT_4X4_1000."""
    d = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    patterns = np.zeros((1000, 4, 4), dtype=np.uint8)
    for mid in range(1000):
        img = aruco.generateImageMarker(d, mid, 6)
        data = img[1:5, 1:5]
        patterns[mid] = (data < 128).astype(np.uint8)
    return patterns


def rectify_marker(gray: np.ndarray, corners_4x2: np.ndarray, out_size: int = 120) -> np.ndarray:
    """Perspective-rectify a marker patch from its 4 detected corners."""
    dst = np.array([
        [0, 0],
        [out_size - 1, 0],
        [out_size - 1, out_size - 1],
        [0, out_size - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners_4x2.astype(np.float32), dst)
    return cv2.warpPerspective(gray, M, (out_size, out_size))


def extract_cell_means(rectified: np.ndarray, grid_size: int = 6) -> np.ndarray:
    """Extract mean intensity per cell of the 6x6 grid (border + 4x4 data)."""
    h, w = rectified.shape[:2]
    cell_h = h / grid_size
    cell_w = w / grid_size
    means = np.zeros((grid_size, grid_size), dtype=float)
    for r in range(grid_size):
        for c in range(grid_size):
            y1, y2 = int(r * cell_h), int((r + 1) * cell_h)
            x1, x2 = int(c * cell_w), int((c + 1) * cell_w)
            means[r, c] = rectified[y1:y2, x1:x2].mean()
    return means


def binarize_payload(cell_means_6x6: np.ndarray) -> np.ndarray:
    """Binarize the 4x4 data cells using Otsu thresholding on the data region."""
    data = cell_means_6x6[1:5, 1:5]
    # Use Otsu on the 16 cell means
    flat = data.flatten().astype(np.uint8)
    _, thresh = cv2.threshold(flat, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh.reshape(4, 4).astype(np.uint8)


def match_all_rotations(bits: np.ndarray, target_pattern: np.ndarray) -> list[int]:
    """Hamming distance between bits and target under 4 rotations."""
    dists = []
    for r in range(4):
        rotated = np.rot90(target_pattern, k=r)
        dists.append(int(np.sum(bits != rotated)))
    return dists


def match_mirror_transpose(bits: np.ndarray, target_pattern: np.ndarray) -> dict:
    """Check mirror/transpose variants."""
    results = {}
    variants = {
        "flip_h": np.fliplr(bits),
        "flip_v": np.flipud(bits),
        "transpose": bits.T,
        "rot180_flip_h": np.fliplr(np.rot90(bits, 2)),
    }
    for name, variant in variants.items():
        dists = match_all_rotations(variant, target_pattern)
        results[name] = {"dists": dists, "min": min(dists)}
    return results


def draw_grid_overlay(rectified: np.ndarray, cell_means: np.ndarray,
                      bits: np.ndarray, grid_size: int = 6) -> np.ndarray:
    """Draw grid lines, cell means, and binary labels on rectified image."""
    if rectified.ndim == 2:
        vis = cv2.cvtColor(rectified, cv2.COLOR_GRAY2BGR)
    else:
        vis = rectified.copy()

    h, w = vis.shape[:2]
    cell_h = h / grid_size
    cell_w = w / grid_size

    # Grid lines
    for i in range(grid_size + 1):
        y = int(i * cell_h)
        x = int(i * cell_w)
        cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
        cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)

    # Cell means and bit labels
    for r in range(grid_size):
        for c in range(grid_size):
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)
            mean_val = cell_means[r, c]

            # Border cells in blue, data cells with bit label
            if r == 0 or r == 5 or c == 0 or c == 5:
                label = f"{mean_val:.0f}"
                color = (255, 100, 100)  # blue for border
            else:
                bit = bits[r - 1, c - 1]
                label = f"{bit}({mean_val:.0f})"
                color = (0, 255, 255) if bit == 1 else (200, 200, 200)

            cv2.putText(vis, label, (cx - 18, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return vis


def audit_one_frame(
    gray: np.ndarray,
    frame_idx: int,
    true_id: int,
    canonical_patterns: np.ndarray,
    aruco_detector: aruco.ArucoDetector,
    output_dir: Path,
    vname: str,
) -> list[dict]:
    """Audit all detections in a single frame."""
    corners_list, ids, rejected = aruco_detector.detectMarkers(gray)

    records = []

    if ids is None or len(ids) == 0:
        return records

    for j, detected_id in enumerate(ids.flatten()):
        detected_id = int(detected_id)
        corners = corners_list[j][0]  # (4, 2)

        # Rectify
        rectified = rectify_marker(gray, corners, out_size=120)
        cell_means = extract_cell_means(rectified)
        bits = binarize_payload(cell_means)

        # Match against true ID
        true_pattern = canonical_patterns[true_id]
        true_dists = match_all_rotations(bits, true_pattern)

        # Match against detected ID
        det_pattern = canonical_patterns[detected_id]
        det_dists = match_all_rotations(bits, det_pattern)

        # Best match across all 1000 IDs x 4 rotations
        best_id, best_dist, best_rot = -1, 17, 0
        runner_up_id, runner_up_dist = -1, 17
        for mid in range(1000):
            for r in range(4):
                d = int(np.sum(bits != np.rot90(canonical_patterns[mid], k=r)))
                if d < best_dist:
                    runner_up_id, runner_up_dist = best_id, best_dist
                    best_id, best_dist, best_rot = mid, d, r
                elif d < runner_up_dist and mid != best_id:
                    runner_up_id, runner_up_dist = mid, d

        margin = runner_up_dist - best_dist

        # Mirror/transpose checks against true ID
        mirror_results = match_mirror_transpose(bits, true_pattern)

        # Check border consistency (should be all black = high intensity for white bg,
        # or all one value)
        border_cells = np.concatenate([
            cell_means[0, :], cell_means[5, :],
            cell_means[1:5, 0], cell_means[1:5, 5]
        ])
        border_mean = border_cells.mean()
        border_std = border_cells.std()
        data_cells = cell_means[1:5, 1:5].flatten()
        data_mean = data_cells.mean()

        rec = {
            "video": vname,
            "frame": frame_idx,
            "detection_idx": j,
            "true_id": true_id,
            "detected_id": detected_id,
            "match_correct": detected_id == true_id,
            "bits": bits.flatten().tolist(),
            "true_hamming_dists": true_dists,
            "true_min_hamming": min(true_dists),
            "detected_hamming_dists": det_dists,
            "detected_min_hamming": min(det_dists),
            "best_match_id": best_id,
            "best_match_dist": best_dist,
            "best_match_rot": best_rot,
            "runner_up_id": runner_up_id,
            "runner_up_dist": runner_up_dist,
            "margin": margin,
            "border_mean": round(border_mean, 1),
            "border_std": round(border_std, 1),
            "data_mean": round(data_mean, 1),
            "mirror_flip_h_min": mirror_results["flip_h"]["min"],
            "mirror_flip_v_min": mirror_results["flip_v"]["min"],
            "mirror_transpose_min": mirror_results["transpose"]["min"],
        }
        records.append(rec)

        # Save annotated images
        crop_dir = output_dir / "audit_crops" / vname
        crop_dir.mkdir(parents=True, exist_ok=True)

        # Original crop around corners
        cx1 = max(0, int(corners[:, 0].min()) - 20)
        cy1 = max(0, int(corners[:, 1].min()) - 20)
        cx2 = min(gray.shape[1], int(corners[:, 0].max()) + 20)
        cy2 = min(gray.shape[0], int(corners[:, 1].max()) + 20)
        original_crop = gray[cy1:cy2, cx1:cx2]

        # Rectified with grid overlay
        grid_vis = draw_grid_overlay(rectified, cell_means, bits)

        # Side-by-side
        orig_resized = cv2.resize(original_crop, (120, 120))
        if orig_resized.ndim == 2:
            orig_resized = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2BGR)

        # Add labels
        label_bar = np.zeros((30, 240, 3), dtype=np.uint8)
        label = f"det={detected_id} true={true_id} best={best_id}(d={best_dist}) margin={margin}"
        cv2.putText(label_bar, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        combined = np.vstack([
            np.hstack([orig_resized, grid_vis]),
            label_bar,
        ])

        fname = f"f{frame_idx:06d}_det{detected_id}_true{true_id}_best{best_id}.png"
        cv2.imwrite(str(crop_dir / fname), combined)

    return records


def main():
    p = argparse.ArgumentParser(description="Bit-level audit for confused cameras")
    p.add_argument("--video", nargs="+", required=True, help="Video paths (glob OK)")
    p.add_argument("--n-frames", type=int, default=30, help="Frames to sample per video")
    p.add_argument("--output-dir", default="nn-aruco-detection-test/audit")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"audit_log_{timestamp}.txt"
    tee = _TeeLogger(log_path)
    sys.stdout = tee

    print(f"=== Bit-Level Audit ===")
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

    print(f"Found {len(video_paths)} videos")

    # Load canonical patterns
    print("Loading DICT_4X4_1000 canonical patterns...")
    patterns = get_canonical_patterns()

    # Print canonical patterns for session IDs
    session_ids = sorted(set(GROUND_TRUTH.values()))
    print(f"\nSession IDs: {session_ids}")
    for sid in session_ids:
        print(f"\n  ID {sid} canonical pattern:")
        for row in patterns[sid]:
            print(f"    {row.tolist()}")

    # Pairwise Hamming distances
    print(f"\nPairwise Hamming distances (min over 4 rotations):")
    from itertools import combinations
    for a, b in combinations(session_ids, 2):
        dists = match_all_rotations(patterns[a], patterns[b])
        print(f"  {a} <-> {b}: {dists} (min={min(dists)})")

    # Set up OpenCV detector
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshConstant = 3
    params.adaptiveThreshWinSizeMin = 10
    params.adaptiveThreshWinSizeMax = 40
    params.adaptiveThreshWinSizeStep = 10
    params.errorCorrectionRate = 1.0
    detector = aruco.ArucoDetector(aruco_dict, params)

    all_records = []

    for vpath in video_paths:
        vname = Path(vpath).stem
        cam = vname.split("_")[0]
        true_id = GROUND_TRUTH.get(cam)
        if true_id is None:
            print(f"\n[SKIP] Unknown true ID for {cam}")
            continue

        print(f"\n{'='*80}")
        print(f"Camera: {cam}  True ID: {true_id}  Video: {Path(vpath).name}")
        print(f"{'='*80}")

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  [ERROR] Cannot open: {vpath}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = min(args.n_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, n, dtype=int)

        id_counts: dict[int, int] = {}
        frame_records = []

        for fi in tqdm(indices, desc=f"  Auditing {cam}", leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

            recs = audit_one_frame(
                gray, int(fi), true_id, patterns, detector,
                out_dir, vname,
            )
            frame_records.extend(recs)
            for r in recs:
                did = r["detected_id"]
                id_counts[did] = id_counts.get(did, 0) + 1

        cap.release()
        all_records.extend(frame_records)

        # Per-camera summary
        total_dets = len(frame_records)
        correct = sum(1 for r in frame_records if r["match_correct"])
        print(f"\n  Detections: {total_dets}")
        print(f"  Correct ID ({true_id}): {correct} ({100*correct/total_dets:.1f}%)" if total_dets else "  No detections")

        if id_counts:
            print(f"  ID distribution:")
            for did, cnt in sorted(id_counts.items(), key=lambda x: -x[1])[:10]:
                pct = 100 * cnt / total_dets
                ham = min(match_all_rotations(patterns[did], patterns[true_id])) if did != true_id else 0
                flag = " <<<" if did == true_id else f" (Hamming to true={ham})"
                print(f"    ID {did:>4}: {cnt:>4} ({pct:5.1f}%){flag}")

        # Bit-level analysis for the dominant wrong ID
        if frame_records:
            wrong_recs = [r for r in frame_records if not r["match_correct"]]
            if wrong_recs:
                # Most common wrong ID
                wrong_ids = [r["detected_id"] for r in wrong_recs]
                from collections import Counter
                top_wrong = Counter(wrong_ids).most_common(1)[0]
                print(f"\n  Most common wrong ID: {top_wrong[0]} ({top_wrong[1]} times)")

                # Check if bits are closer to wrong ID than true ID
                closer_to_wrong = sum(
                    1 for r in wrong_recs
                    if r["detected_min_hamming"] < r["true_min_hamming"]
                )
                closer_to_true = sum(
                    1 for r in wrong_recs
                    if r["true_min_hamming"] < r["detected_min_hamming"]
                )
                equal = len(wrong_recs) - closer_to_wrong - closer_to_true
                print(f"  Bit analysis (wrong-ID detections):")
                print(f"    Bits closer to detected ID: {closer_to_wrong}")
                print(f"    Bits closer to true ID:     {closer_to_true}")
                print(f"    Equidistant:                {equal}")

                # Mirror/transpose check
                mirror_fixes = {
                    "flip_h": sum(1 for r in wrong_recs if r["mirror_flip_h_min"] <= 2),
                    "flip_v": sum(1 for r in wrong_recs if r["mirror_flip_v_min"] <= 2),
                    "transpose": sum(1 for r in wrong_recs if r["mirror_transpose_min"] <= 2),
                }
                print(f"  Mirror/transpose match to true ID (Hamming <= 2):")
                for name, cnt in mirror_fixes.items():
                    print(f"    {name}: {cnt}/{len(wrong_recs)}")

                # Best-match analysis
                best_match_is_true = sum(1 for r in wrong_recs if r["best_match_id"] == true_id)
                best_match_is_detected = sum(1 for r in wrong_recs if r["best_match_id"] == r["detected_id"])
                best_match_other = len(wrong_recs) - best_match_is_true - best_match_is_detected
                print(f"  Global best-match (brute-force all 1000 IDs × 4 rot):")
                print(f"    Best = true ID ({true_id}):     {best_match_is_true}")
                print(f"    Best = detected ID:       {best_match_is_detected}")
                print(f"    Best = other ID:          {best_match_other}")

    # Save summary CSV
    if all_records:
        import csv
        csv_path = out_dir / f"audit_summary_{timestamp}.csv"
        keys = list(all_records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_records:
                # Convert lists to strings for CSV
                row = {}
                for k, v in r.items():
                    row[k] = str(v) if isinstance(v, list) else v
                w.writerow(row)
        print(f"\nSummary CSV saved: {csv_path}")

    print(f"\nLog saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
