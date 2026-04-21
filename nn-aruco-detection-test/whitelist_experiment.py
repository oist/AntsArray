#!/usr/bin/env python3
"""Offline whitelist matcher experiment.

Upper-bound study: can a session-restricted soft decoder convert undecoded
rescue crops into correct IDs without adding wrong IDs?

First measures whether OpenCV even finds a quad candidate in each crop
(the prerequisite for any ID matching).  Then, for crops with quads,
applies a soft whitelist matcher and compares against brute-force global
matching.

Buckets:
  1. UNDECODED at real marker positions (from disagreement set)
  2. YOLO_ONLY  (YOLO found something, OpenCV missed entirely)
  3. ID_DISAGREE (both decoded, different IDs)
  4. Control: random OpenCV-decoded crops (should score perfectly)

For each crop the script reports:
  - has_accepted_quad: OpenCV decoded at least one marker
  - has_rejected_quad: OpenCV found candidate quads but rejected them
  - no_quad: OpenCV found nothing at all
  - whitelist_id: best session-ID match via soft scorer
  - global_best_id: best match across all 1000 IDs
  - correct: whether whitelist_id matches the OpenCV GT (for UNDECODED
    crops matched to a known position)

Usage:
    python nn-aruco-detection-test/whitelist_experiment.py \\
        --disagreement-dir nn-aruco-detection-test/disagreement_set \\
        --session-ids 3 17 25 \\
        --output-dir nn-aruco-detection-test/results
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
# Soft whitelist matcher
# ---------------------------------------------------------------------------

def get_canonical_patterns() -> np.ndarray:
    """(1000, 4, 4) canonical bit patterns."""
    d = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    patterns = np.zeros((1000, 4, 4), dtype=np.uint8)
    for mid in range(1000):
        img = aruco.generateImageMarker(d, mid, 6)
        patterns[mid] = (img[1:5, 1:5] < 128).astype(np.uint8)
    return patterns


def rectify_from_corners(gray: np.ndarray, corners_4x2: np.ndarray,
                         out_size: int = 120) -> np.ndarray:
    dst = np.array([[0, 0], [out_size-1, 0], [out_size-1, out_size-1],
                    [0, out_size-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners_4x2.astype(np.float32), dst)
    return cv2.warpPerspective(gray, M, (out_size, out_size))


def extract_cell_means_6x6(rectified: np.ndarray) -> np.ndarray:
    """(6, 6) mean intensity per cell."""
    h, w = rectified.shape[:2]
    ch, cw = h / 6, w / 6
    means = np.zeros((6, 6), dtype=float)
    for r in range(6):
        for c in range(6):
            y1, y2 = int(r * ch), int((r + 1) * ch)
            x1, x2 = int(c * cw), int((c + 1) * cw)
            means[r, c] = rectified[y1:y2, x1:x2].mean()
    return means


def soft_score(cell_means_6x6: np.ndarray, pattern_4x4: np.ndarray) -> float:
    """Score a rectified crop against a canonical pattern.

    Higher = better match.  Combines:
    - per-cell agreement (data region)
    - border consistency penalty
    """
    data = cell_means_6x6[1:5, 1:5]  # (4, 4) intensities

    # Normalize data to [0, 1]
    dmin, dmax = data.min(), data.max()
    if dmax - dmin < 10:
        return -999.0  # flat crop, no signal
    data_norm = (data - dmin) / (dmax - dmin)

    # pattern: 1 = black cell (low intensity), 0 = white (high intensity)
    # Expected: bit=1 → low intensity → data_norm near 0
    #           bit=0 → high intensity → data_norm near 1
    expected = 1.0 - pattern_4x4.astype(float)  # 1→0.0, 0→1.0
    agreement = 1.0 - np.abs(data_norm - expected)  # (4,4) in [0,1]
    cell_score = agreement.mean()

    # Border penalty: border cells should all be black (low intensity)
    border = np.concatenate([
        cell_means_6x6[0, :], cell_means_6x6[5, :],
        cell_means_6x6[1:5, 0], cell_means_6x6[1:5, 5],
    ])
    # Border should be darker than data mean
    border_norm = (border - dmin) / (dmax - dmin) if dmax - dmin > 0 else border * 0
    # Good border: most cells near 0 (black)
    border_quality = 1.0 - border_norm.mean()
    border_penalty = max(0.0, 0.5 - border_quality) * 2  # 0 if good, up to 1

    return cell_score - 0.3 * border_penalty


def match_whitelist(
    cell_means_6x6: np.ndarray,
    patterns: np.ndarray,
    whitelist: set[int],
    min_absolute_score: float = 0.6,
    min_margin: float = 0.05,
) -> dict:
    """Score crop against whitelist IDs and global best.

    Returns dict with:
        whitelist_id, whitelist_score, whitelist_rot,
        runner_up_id, runner_up_score, margin,
        global_best_id, global_best_score, global_best_rot,
        accepted, reject_reason
    """
    # Score all 1000 IDs × 4 rotations
    all_scores: list[tuple[int, int, float]] = []  # (id, rot, score)
    for mid in range(1000):
        for r in range(4):
            pat = np.rot90(patterns[mid], k=r)
            s = soft_score(cell_means_6x6, pat)
            all_scores.append((mid, r, s))

    all_scores.sort(key=lambda x: -x[2])
    global_best_id, global_best_rot, global_best_score = all_scores[0]

    # Whitelist-only scores
    wl_scores = [(mid, r, s) for mid, r, s in all_scores if mid in whitelist]
    if not wl_scores:
        return {
            "whitelist_id": -1, "whitelist_score": -999, "whitelist_rot": 0,
            "runner_up_id": -1, "runner_up_score": -999, "margin": 0,
            "global_best_id": global_best_id, "global_best_score": round(global_best_score, 4),
            "global_best_rot": global_best_rot,
            "accepted": False, "reject_reason": "no_whitelist_ids",
        }

    wl_best_id, wl_best_rot, wl_best_score = wl_scores[0]

    # Runner-up: best whitelist score from a DIFFERENT ID
    runner_up_id, runner_up_score = -1, -999.0
    for mid, r, s in wl_scores:
        if mid != wl_best_id:
            runner_up_id, runner_up_score = mid, s
            break

    margin = wl_best_score - runner_up_score if runner_up_id >= 0 else 1.0

    # Acceptance gates
    accepted = True
    reject_reason = ""

    if wl_best_score < min_absolute_score:
        accepted = False
        reject_reason = f"low_quality({wl_best_score:.3f}<{min_absolute_score})"
    elif margin < min_margin:
        accepted = False
        reject_reason = f"thin_margin({margin:.3f}<{min_margin})"
    elif global_best_id not in whitelist and global_best_score > wl_best_score + 0.02:
        accepted = False
        reject_reason = f"global_beats_whitelist(global={global_best_id}@{global_best_score:.3f})"

    return {
        "whitelist_id": wl_best_id if accepted else -1,
        "whitelist_score": round(wl_best_score, 4),
        "whitelist_rot": wl_best_rot,
        "runner_up_id": runner_up_id,
        "runner_up_score": round(runner_up_score, 4),
        "margin": round(margin, 4),
        "global_best_id": global_best_id,
        "global_best_score": round(global_best_score, 4),
        "global_best_rot": global_best_rot,
        "accepted": accepted,
        "reject_reason": reject_reason,
    }


# ---------------------------------------------------------------------------
# Crop analysis
# ---------------------------------------------------------------------------

def analyze_crop(
    crop_gray: np.ndarray,
    detector_standard: aruco.ArucoDetector,
    detector_aggressive: aruco.ArucoDetector,
    patterns: np.ndarray,
    whitelist: set[int],
) -> dict:
    """Analyze a single crop: quad presence, whitelist match."""

    result = {
        "has_accepted_quad": False,
        "has_rejected_quad": False,
        "no_quad": True,
        "accepted_id": -1,
        "n_rejected_quads": 0,
    }

    # Try standard detector first
    corners, ids, rejected = detector_standard.detectMarkers(crop_gray)

    if ids is not None and len(ids) > 0:
        result["has_accepted_quad"] = True
        result["no_quad"] = False
        result["accepted_id"] = int(ids.flatten()[0])
        best_corners = corners[0][0]
    elif rejected is not None and len(rejected) > 0:
        result["has_rejected_quad"] = True
        result["no_quad"] = False
        result["n_rejected_quads"] = len(rejected)
        best_corners = rejected[0][0]  # use first rejected quad
    else:
        # Try aggressive detector
        corners2, ids2, rejected2 = detector_aggressive.detectMarkers(crop_gray)
        if ids2 is not None and len(ids2) > 0:
            result["has_accepted_quad"] = True
            result["no_quad"] = False
            result["accepted_id"] = int(ids2.flatten()[0])
            best_corners = corners2[0][0]
        elif rejected2 is not None and len(rejected2) > 0:
            result["has_rejected_quad"] = True
            result["no_quad"] = False
            result["n_rejected_quads"] = len(rejected2)
            best_corners = rejected2[0][0]
        else:
            result["whitelist_match"] = {
                "whitelist_id": -1, "accepted": False,
                "reject_reason": "no_quad_at_all",
                "whitelist_score": -999, "global_best_id": -1,
                "global_best_score": -999,
            }
            return result

    # Rectify and score
    try:
        rectified = rectify_from_corners(crop_gray, best_corners, out_size=120)
        cell_means = extract_cell_means_6x6(rectified)
        wl_match = match_whitelist(cell_means, patterns, whitelist)
    except Exception:
        wl_match = {
            "whitelist_id": -1, "accepted": False,
            "reject_reason": "rectify_failed",
            "whitelist_score": -999, "global_best_id": -1,
            "global_best_score": -999,
        }

    result["whitelist_match"] = wl_match
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Offline whitelist matcher experiment")
    p.add_argument("--disagreement-dir",
                   default="nn-aruco-detection-test/disagreement_set")
    p.add_argument("--session-ids", type=int, nargs="+", default=[3, 17, 25],
                   help="Valid marker IDs for this session")
    p.add_argument("--output-dir", default="nn-aruco-detection-test/results")
    p.add_argument("--max-crops-per-category", type=int, default=500)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"whitelist_experiment_log_{timestamp}.txt"
    tee = _TeeLogger(log_path)
    sys.stdout = tee

    print(f"=== Whitelist Matcher Experiment ===")
    print(f"Time: {timestamp}")
    print(f"Session IDs: {args.session_ids}")
    print()

    whitelist = set(args.session_ids)
    dis_dir = Path(args.disagreement_dir)

    # Load canonical patterns
    patterns = get_canonical_patterns()

    # Build OpenCV detectors
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

    # Standard
    ps = aruco.DetectorParameters()
    ps.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    ps.adaptiveThreshConstant = 3
    ps.adaptiveThreshWinSizeMin = 5
    ps.adaptiveThreshWinSizeMax = 50
    ps.adaptiveThreshWinSizeStep = 5
    ps.minMarkerPerimeterRate = 0.01
    ps.maxMarkerPerimeterRate = 4.0
    ps.errorCorrectionRate = 1.0
    det_standard = aruco.ArucoDetector(aruco_dict, ps)

    # Aggressive (sweep winner)
    pa = aruco.DetectorParameters()
    pa.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    pa.adaptiveThreshConstant = 3
    pa.adaptiveThreshWinSizeMin = 3
    pa.adaptiveThreshWinSizeMax = 80
    pa.adaptiveThreshWinSizeStep = 3
    pa.minMarkerPerimeterRate = 0.01
    pa.maxMarkerPerimeterRate = 4.0
    pa.errorCorrectionRate = 0.8
    pa.relativeCornerRefinmentWinSize = 0.5
    pa.perspectiveRemovePixelPerCell = 8
    pa.perspectiveRemoveIgnoredMarginPerCell = 0.2
    det_aggressive = aruco.ArucoDetector(aruco_dict, pa)

    # Load crops by category
    jsons = sorted(dis_dir.glob("*.json"))
    print(f"Found {len(jsons)} disagreement crops in {dis_dir}")

    by_category: dict[str, list[Path]] = {}
    for jp in jsons:
        if not jp.with_suffix(".png").exists():
            continue
        with open(jp) as f:
            meta = json.load(f)
        cat = meta.get("category", "UNKNOWN")
        by_category.setdefault(cat, []).append(jp)

    print(f"\nCategories:")
    for cat, paths in sorted(by_category.items()):
        print(f"  {cat}: {len(paths)}")

    # Analyze each category
    all_records = []

    for cat in ["UNDECODED", "YOLO_ONLY", "ID_DISAGREE", "EDGE_CASE", "OPENCV_ONLY"]:
        paths = by_category.get(cat, [])
        if not paths:
            continue

        # Sample if too many
        if len(paths) > args.max_crops_per_category:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(paths), args.max_crops_per_category, replace=False)
            paths = [paths[i] for i in sorted(indices)]

        print(f"\n{'='*70}")
        print(f"Category: {cat} ({len(paths)} crops)")
        print(f"{'='*70}")

        quad_counts = {"has_accepted": 0, "has_rejected": 0, "no_quad": 0}
        wl_accepted = 0
        wl_rejected_reasons: Counter = Counter()
        wl_correct = 0  # whitelist ID matches GT (where GT is known)
        wl_wrong = 0

        for jp in tqdm(paths, desc=f"  {cat}", leave=False):
            with open(jp) as f:
                meta = json.load(f)

            png_path = jp.with_suffix(".png")
            crop = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            if crop is None:
                continue

            result = analyze_crop(crop, det_standard, det_aggressive,
                                  patterns, whitelist)

            # Count quad presence
            if result["has_accepted_quad"]:
                quad_counts["has_accepted"] += 1
            elif result["has_rejected_quad"]:
                quad_counts["has_rejected"] += 1
            else:
                quad_counts["no_quad"] += 1

            wl = result.get("whitelist_match", {})

            if wl.get("accepted"):
                wl_accepted += 1
                # Check correctness against opencv_id (if available)
                opencv_id = meta.get("opencv_id")
                wl_id = wl["whitelist_id"]
                if opencv_id is not None and opencv_id >= 0:
                    if wl_id == opencv_id:
                        wl_correct += 1
                    else:
                        wl_wrong += 1
            else:
                reason = wl.get("reject_reason", "unknown")
                wl_rejected_reasons[reason] += 1

            rec = {
                "category": cat,
                "file": jp.name,
                "video": meta.get("video", ""),
                "frame": meta.get("frame", -1),
                "opencv_id": meta.get("opencv_id"),
                "hybrid_id": meta.get("hybrid_id"),
                "has_accepted_quad": result["has_accepted_quad"],
                "has_rejected_quad": result["has_rejected_quad"],
                "no_quad": result["no_quad"],
                "n_rejected_quads": result.get("n_rejected_quads", 0),
                "accepted_id": result.get("accepted_id", -1),
                "wl_accepted": wl.get("accepted", False),
                "wl_id": wl.get("whitelist_id", -1),
                "wl_score": wl.get("whitelist_score", -999),
                "wl_margin": wl.get("margin", 0),
                "wl_reject_reason": wl.get("reject_reason", ""),
                "global_best_id": wl.get("global_best_id", -1),
                "global_best_score": wl.get("global_best_score", -999),
            }
            all_records.append(rec)

        # Report
        total = len(paths)
        print(f"\n  Quad presence ({total} crops):")
        print(f"    Accepted quad (decoded by OpenCV): {quad_counts['has_accepted']:>5} "
              f"({100*quad_counts['has_accepted']/total:.1f}%)")
        print(f"    Rejected quad (candidate found):   {quad_counts['has_rejected']:>5} "
              f"({100*quad_counts['has_rejected']/total:.1f}%)")
        print(f"    No quad at all:                    {quad_counts['no_quad']:>5} "
              f"({100*quad_counts['no_quad']/total:.1f}%)")

        has_quad = quad_counts["has_accepted"] + quad_counts["has_rejected"]
        if has_quad > 0:
            print(f"\n  Whitelist matcher (on {has_quad} crops with quads):")
            print(f"    Accepted (decoded a session ID):  {wl_accepted:>5} "
                  f"({100*wl_accepted/has_quad:.1f}%)")
            print(f"    Rejected:                         {has_quad - wl_accepted:>5}")
            if wl_rejected_reasons:
                print(f"    Reject reasons:")
                for reason, cnt in wl_rejected_reasons.most_common():
                    print(f"      {reason}: {cnt}")

            if wl_correct + wl_wrong > 0:
                print(f"\n  Correctness (where GT available):")
                print(f"    Correct: {wl_correct}  Wrong: {wl_wrong}  "
                      f"Accuracy: {100*wl_correct/(wl_correct+wl_wrong):.1f}%")

    # Save CSV
    if all_records:
        import csv
        csv_path = out_dir / f"whitelist_experiment_{timestamp}.csv"
        keys = list(all_records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_records)
        print(f"\nCSV saved: {csv_path}")

    # Grand summary
    if all_records:
        total = len(all_records)
        has_any_quad = sum(1 for r in all_records
                          if r["has_accepted_quad"] or r["has_rejected_quad"])
        no_quad = sum(1 for r in all_records if r["no_quad"])
        wl_total_accepted = sum(1 for r in all_records if r["wl_accepted"])

        print(f"\n{'='*70}")
        print(f"GRAND SUMMARY ({total} crops)")
        print(f"{'='*70}")
        print(f"  Has quad (accepted or rejected): {has_any_quad} ({100*has_any_quad/total:.1f}%)")
        print(f"  No quad at all:                  {no_quad} ({100*no_quad/total:.1f}%)")
        print(f"  Whitelist accepted:              {wl_total_accepted} "
              f"({100*wl_total_accepted/total:.1f}% of all, "
              f"{100*wl_total_accepted/has_any_quad:.1f}% of quad-bearing)"
              if has_any_quad else "")
        print(f"\n  Verdict: {'QUAD IS THE BOTTLENECK' if no_quad > has_any_quad else 'ID MATCHING HAS HEADROOM'}")

    print(f"\nLog saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
