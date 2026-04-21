#!/usr/bin/env python3
"""
Run ArUco detection on per-ant videos to compare tag size performance.

Reads each video sequentially (no seeking), processes every Nth frame,
identifies the dominant tag ID and estimates physical size from corner
perimeter. Outputs a summary table and per-size statistics.

Usage:
    python benchmark/run_size_comparison.py \
        --video-dir "Z:/ReiterU/Ants/basler/QRcodes_test/ARUCO_size_comparison_15_20_25/20260331_01" \
        --skip 24 \
        --target-ids 3,17,25
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from tqdm import tqdm


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


DICT_CHOICES = {
    "4x4_1000": aruco.DICT_4X4_1000,
    "4x4_250": aruco.DICT_4X4_250,
    "4x4_100": aruco.DICT_4X4_100,
    "4x4_50": aruco.DICT_4X4_50,
}


def make_detector(dict_type=aruco.DICT_4X4_1000):
    aruco_dict = aruco.getPredefinedDictionary(dict_type)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshConstant = 3
    params.adaptiveThreshWinSizeMin = 10
    params.adaptiveThreshWinSizeMax = 40
    params.adaptiveThreshWinSizeStep = 10
    params.errorCorrectionRate = 1
    return aruco.ArucoDetector(aruco_dict, params)


def process_video(video_path, detector, skip, target_ids, max_read_frames=0):
    """Process a single video, return per-ID detection stats."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}", file=sys.stderr)
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    read_limit = max_read_frames if max_read_frames > 0 else total

    id_frame_count = defaultdict(int)
    id_perimeters = defaultdict(list)
    all_id_count = defaultdict(int)  # includes non-target IDs
    n_processed = 0
    frame_idx = 0

    pbar = tqdm(total=min(read_limit, total), desc=Path(video_path).stem, unit="fr", leave=False)

    while frame_idx < read_limit:
        ret, frame = cap.read()
        if not ret:
            break

        pbar.update(1)

        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        n_processed += 1

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                mid = int(mid)
                all_id_count[mid] += 1

                if target_ids and mid not in target_ids:
                    continue

                c = corners[i][0]
                perim = cv2.arcLength(c, closed=True)
                id_frame_count[mid] += 1
                id_perimeters[mid].append(perim)

        frame_idx += 1

    pbar.close()
    cap.release()

    if not id_frame_count:
        return {
            "n_processed": n_processed,
            "total_frames": total,
            "fps": fps,
            "dominant_id": None,
            "det_rate": 0,
            "mean_perim": 0,
            "std_perim": 0,
            "all_ids": dict(all_id_count),
        }

    dominant_id = max(id_frame_count, key=id_frame_count.get)
    return {
        "n_processed": n_processed,
        "total_frames": total,
        "fps": fps,
        "dominant_id": dominant_id,
        "det_rate": id_frame_count[dominant_id] / n_processed * 100,
        "det_count": id_frame_count[dominant_id],
        "mean_perim": float(np.mean(id_perimeters[dominant_id])),
        "std_perim": float(np.std(id_perimeters[dominant_id])),
        "all_target_ids": dict(id_frame_count),
        "all_ids": dict(all_id_count),
    }


def assign_sizes(results):
    """Cluster perimeters into 3 size groups and assign labels."""
    from scipy.cluster.vq import kmeans

    perims = [r["mean_perim"] for r in results if r.get("dominant_id") is not None and r["mean_perim"] > 0]
    if len(perims) < 3:
        print("[WARN] Not enough data to cluster into 3 sizes")
        return results, None, None

    centroids, _ = kmeans(np.array(perims, dtype=float), 3)
    centroids = sorted(centroids)
    boundaries = [(centroids[i] + centroids[i + 1]) / 2 for i in range(2)]

    for r in results:
        if r.get("dominant_id") is None or r["mean_perim"] == 0:
            r["size"] = "?"
            continue
        p = r["mean_perim"]
        if p < boundaries[0]:
            r["size"] = "1.5mm"
        elif p < boundaries[1]:
            r["size"] = "2.0mm"
        else:
            r["size"] = "2.5mm"

    return results, centroids, boundaries


def main():
    p = argparse.ArgumentParser(description="Compare ArUco tag sizes across per-ant videos")
    p.add_argument("--video-dir", required=True, help="Directory containing per-ant .avi files")
    p.add_argument("--skip", type=int, default=24, help="Process every Nth frame (default: 24, ~1 fps at 24fps)")
    p.add_argument("--target-ids", type=str, default="3,17,25", help="Comma-separated target tag IDs")
    p.add_argument("--max-read-frames", type=int, default=7200,
                   help="Max raw frames to read per video (default: 7200 = 5 min at 24fps, 0=all)")
    p.add_argument("--exclude", type=str, default="global", help="Exclude videos containing this string")
    p.add_argument("--output-csv", type=str, default=None, help="Save results to CSV")
    p.add_argument("--dicts", nargs="+", choices=list(DICT_CHOICES.keys()),
                   default=["4x4_1000"],
                   help="Dictionary sizes to compare (default: 4x4_1000)")
    p.add_argument("--output-dir", default="benchmark/results", help="Output directory for logs")
    args = p.parse_args()

    # Set up logging
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"size_comparison_log_{timestamp}.txt"
    tee = _TeeLogger(log_path)
    sys.stdout = tee

    target_ids = set(int(x) for x in args.target_ids.split(",")) if args.target_ids else set()

    # Find videos
    video_dir = Path(args.video_dir)
    videos = sorted(video_dir.glob("*.avi"))
    if args.exclude:
        videos = [v for v in videos if args.exclude not in v.name]

    print(f"=== Tag Size Comparison ===")
    print(f"Time: {timestamp}")
    print(f"Found {len(videos)} videos in {video_dir}")
    print(f"Target IDs: {target_ids}")
    print(f"Dictionaries: {args.dicts}")
    print(f"Processing every {args.skip}th frame")
    print()
    # Run for each dictionary
    all_dict_results = {}

    for dict_name in args.dicts:
        dict_type = DICT_CHOICES[dict_name]
        print(f"\n{'='*70}")
        print(f"Dictionary: {dict_name}")
        print(f"{'='*70}")

        detector = make_detector(dict_type)
        results = []

        for vpath in videos:
            cam = vpath.stem.split("_")[0]
            r = process_video(str(vpath), detector, args.skip, target_ids, args.max_read_frames)
            if r is None:
                continue
            r["cam"] = cam
            r["filename"] = vpath.name
            r["dict"] = dict_name
            results.append(r)

            if r["dominant_id"] is not None:
                print(f"  {cam}: ID={r['dominant_id']}, rate={r['det_rate']:.1f}%, "
                      f"perim={r['mean_perim']:.0f}+/-{r['std_perim']:.0f}px, "
                      f"processed={r['n_processed']}/{r['total_frames']} frames")
            else:
                print(f"  {cam}: no target IDs detected (processed {r['n_processed']} frames)")
                if r["all_ids"]:
                    top5 = dict(sorted(r["all_ids"].items(), key=lambda x: -x[1])[:5])
                    print(f"         top non-target IDs: {top5}")

        # Cluster into sizes
        print()
        results, centroids, boundaries = assign_sizes(results)

        if centroids is not None:
            print(f"  Perimeter centroids: {[f'{c:.0f}' for c in centroids]}")
            print(f"  Boundaries: <{boundaries[0]:.0f} = 1.5mm, "
                  f"{boundaries[0]:.0f}-{boundaries[1]:.0f} = 2.0mm, "
                  f">{boundaries[1]:.0f} = 2.5mm")
            print()

        # Per-dict table
        print(f"  {'Cam':<7} {'ID':>3} {'Size':>6} {'Det Rate':>9} {'Perim (px)':>14} {'Frames':>10}")
        print(f"  {'-'*58}")

        size_stats = defaultdict(list)
        for r in results:
            if r.get("dominant_id") is None:
                print(f"  {r['cam']:<7} {'?':>3} {'?':>6} {'0.0%':>9} {'N/A':>14} {r['n_processed']:>10}")
                continue

            size = r.get("size", "?")
            size_stats[size].append(r["det_rate"])
            print(f"  {r['cam']:<7} {r['dominant_id']:>3} {size:>6} {r['det_rate']:>8.1f}% "
                  f"{r['mean_perim']:>7.0f}+/-{r['std_perim']:>3.0f}  {r['n_processed']:>10}")

        print()
        print(f"  Summary by Tag Size ({dict_name})")
        print(f"  {'='*50}")
        for size in ["1.5mm", "2.0mm", "2.5mm"]:
            rates = size_stats.get(size, [])
            if rates:
                print(f"    {size}: n={len(rates):>2}, "
                      f"mean={np.mean(rates):5.1f}%, "
                      f"std={np.std(rates):5.1f}%, "
                      f"min={np.min(rates):5.1f}%, "
                      f"max={np.max(rates):5.1f}%")
            else:
                print(f"    {size}: no data")

        all_dict_results[dict_name] = (results, size_stats)

    # Cross-dictionary comparison
    if len(args.dicts) > 1:
        print(f"\n{'='*70}")
        print("CROSS-DICTIONARY COMPARISON")
        print(f"{'='*70}")
        print(f"\n  {'Size':<8}", end="")
        for dn in args.dicts:
            print(f"  {dn:>12}", end="")
        print()
        print(f"  {'-'*8}", end="")
        for _ in args.dicts:
            print(f"  {'-'*12}", end="")
        print()

        for size in ["1.5mm", "2.0mm", "2.5mm"]:
            print(f"  {size:<8}", end="")
            for dn in args.dicts:
                _, ss = all_dict_results[dn]
                rates = ss.get(size, [])
                if rates:
                    print(f"  {np.mean(rates):>11.1f}%", end="")
                else:
                    print(f"  {'N/A':>12}", end="")
            print()

    # Save CSV (all dicts combined)
    csv_path = out_dir / f"size_comparison_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dict", "cam", "filename", "dominant_id", "size", "det_rate",
            "det_count", "n_processed", "total_frames", "fps",
            "mean_perim", "std_perim",
        ])
        w.writeheader()
        for dn in args.dicts:
            results, _ = all_dict_results[dn]
            for r in results:
                w.writerow({
                    "dict": dn,
                    "cam": r["cam"],
                    "filename": r.get("filename", ""),
                    "dominant_id": r.get("dominant_id", ""),
                    "size": r.get("size", ""),
                    "det_rate": f"{r['det_rate']:.1f}",
                    "det_count": r.get("det_count", 0),
                    "n_processed": r["n_processed"],
                    "total_frames": r["total_frames"],
                    "fps": r.get("fps", 0),
                    "mean_perim": f"{r['mean_perim']:.1f}",
                    "std_perim": f"{r['std_perim']:.1f}",
                })
    print(f"\nResults saved to: {csv_path}")
    print(f"Log saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
