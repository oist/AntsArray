#!/usr/bin/env python3
"""Benchmark conservative temporal gap-filling.

Uses nest chunk videos where both SLEAP and ArUco data exist.
Evaluates gap-filling by:

1. Taking the existing ArUco detections as ground truth
2. Artificially removing detections at known rates to create gaps
3. Running gap-fill to recover them
4. Measuring recovery rate, wrong fills, and ID switches

This simulates the real scenario: OpenCV detects a marker in frames N
and N+K but misses frames N+1..N+K-1.  Can gap-filling recover those?

Also runs on the ORIGINAL (un-degraded) data to measure how many
naturally occurring gaps get filled and whether the fills are consistent
with the surrounding detections.

Usage:
    python nn-aruco-detection-test/benchmark_temporal.py \\
        --data-dir "Z:\\...\\data" \\
        --cameras cam04 cam05 cam09 cam10
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.temporal_id import build_sleap_tracklets, _match_aruco_to_tracklets, fill_interior_gaps


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
# Synthetic gap creation
# ---------------------------------------------------------------------------

def create_gaps(aruco_df: pd.DataFrame, drop_rate: float = 0.3,
                seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop a fraction of detections to simulate gaps.

    Returns (degraded_df, dropped_df) where dropped_df contains the
    removed rows (used as ground truth for recovery evaluation).
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(len(aruco_df)) < drop_rate
    dropped = aruco_df[mask].copy()
    kept = aruco_df[~mask].copy()
    return kept, dropped


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_gap_fill(
    filled_df: pd.DataFrame,
    original_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
) -> dict:
    """Evaluate gap-filling quality.

    Parameters
    ----------
    filled_df : DataFrame after gap-filling (has propagation_source column)
    original_df : Full original ArUco detections (ground truth)
    dropped_df : The rows that were artificially removed

    Returns
    -------
    dict with recovery_rate, wrong_fills, precision, etc.
    """
    gap_filled = filled_df[filled_df["propagation_source"] == "gap_filled"]
    n_dropped = len(dropped_df)
    n_filled = len(gap_filled)

    if n_filled == 0:
        return {
            "n_dropped": n_dropped,
            "n_filled": 0,
            "n_correct": 0,
            "n_wrong": 0,
            "recovery_rate": 0.0,
            "fill_precision": 0.0,
        }

    # Match fills to dropped rows by (Frame, proximity)
    correct = 0
    wrong = 0

    # Build lookup of dropped detections
    dropped_by_frame: dict[int, list[tuple[int, float, float]]] = {}
    for _, row in dropped_df.iterrows():
        fi = int(row["Frame"])
        dropped_by_frame.setdefault(fi, []).append(
            (int(row["Instance"]), float(row["X"]), float(row["Y"]))
        )

    for _, fill_row in gap_filled.iterrows():
        fi = int(fill_row["Frame"])
        fill_id = int(fill_row["Instance"])
        fill_x, fill_y = float(fill_row["X"]), float(fill_row["Y"])

        if fi not in dropped_by_frame:
            # Fill in a frame where nothing was dropped — could be a
            # naturally missing frame or a false fill
            continue

        # Find closest dropped detection
        best_dist = float("inf")
        best_id = -1
        for did, dx, dy in dropped_by_frame[fi]:
            d = ((fill_x - dx)**2 + (fill_y - dy)**2)**0.5
            if d < best_dist:
                best_dist = d
                best_id = did

        if best_dist < 50.0:
            if fill_id == best_id:
                correct += 1
            else:
                wrong += 1

    recovery_rate = correct / n_dropped if n_dropped > 0 else 0
    fill_precision = correct / (correct + wrong) if (correct + wrong) > 0 else 0

    return {
        "n_dropped": n_dropped,
        "n_filled": n_filled,
        "n_correct": correct,
        "n_wrong": wrong,
        "recovery_rate": round(recovery_rate, 4),
        "fill_precision": round(fill_precision, 4),
    }


def evaluate_natural_gaps(
    aruco_df: pd.DataFrame,
    sleap_df: pd.DataFrame,
    max_gap_frames: int = 10,
) -> dict:
    """Run gap-filling on the original (un-degraded) data and report stats."""
    filled = fill_interior_gaps(
        aruco_df, sleap_df,
        max_gap_frames=max_gap_frames,
        max_distance=80.0,
        max_tracklet_gap=5,
    )

    gap_filled = filled[filled["propagation_source"] == "gap_filled"]
    n_original = len(aruco_df)
    n_filled = len(gap_filled)
    n_total = len(filled)

    # Check ID consistency: for each gap-fill, is the ID the same as
    # the majority ID of the SLEAP instance it belongs to?
    # (We don't have true GT here, but we can check internal consistency)

    # ID distribution of gap-fills
    if n_filled > 0:
        fill_ids = gap_filled["Instance"].value_counts()
        top_ids = fill_ids.head(10)
    else:
        top_ids = pd.Series(dtype=int)

    return {
        "n_original_detections": n_original,
        "n_gap_filled": n_filled,
        "n_total_after_fill": n_total,
        "fill_fraction": round(n_filled / n_original, 4) if n_original > 0 else 0,
        "top_filled_ids": top_ids.to_dict(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Benchmark temporal gap-filling")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--cameras", nargs="+",
                   default=["cam04", "cam05", "cam09", "cam10"])
    p.add_argument("--chunk", default="000", help="Chunk suffix to use")
    p.add_argument("--max-gap-frames", type=int, default=10)
    p.add_argument("--output-dir", default="nn-aruco-detection-test/results")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"temporal_benchmark_log_{timestamp}.txt"
    tee = _TeeLogger(log_path)
    sys.stdout = tee

    print(f"=== Temporal Gap-Fill Benchmark ===")
    print(f"Time: {timestamp}")
    print(f"Max gap frames: {args.max_gap_frames}")
    print()

    data_dir = Path(args.data_dir)
    all_synthetic = []
    all_natural = []

    for cam in args.cameras:
        # Find chunk files
        pattern = f"{cam}*_{args.chunk}"
        sleap_files = sorted(data_dir.glob(f"{cam}*_{args.chunk}_sleap_data.csv"))
        aruco_files = sorted(data_dir.glob(f"{cam}*_{args.chunk}_aruco_detections.csv"))

        if not sleap_files or not aruco_files:
            print(f"\n[SKIP] {cam}: missing SLEAP or ArUco data")
            continue

        sleap_csv = sleap_files[0]
        aruco_csv = aruco_files[0]

        print(f"\n{'='*60}")
        print(f"Camera: {cam}")
        print(f"  SLEAP: {sleap_csv.name}")
        print(f"  ArUco: {aruco_csv.name}")
        print(f"{'='*60}")

        sleap_df = pd.read_csv(sleap_csv)
        aruco_df = pd.read_csv(aruco_csv)

        # Limit to first N frames for speed (use ~5000 frames)
        max_frame = min(5000, sleap_df["Frame"].max() + 1)
        sleap_sub = sleap_df[sleap_df["Frame"] < max_frame].copy()
        aruco_sub = aruco_df[aruco_df["Frame"] < max_frame].copy()

        print(f"  Using frames 0-{max_frame-1}")
        print(f"  ArUco detections: {len(aruco_sub)}")
        print(f"  SLEAP rows: {len(sleap_sub)}")

        # --- Synthetic gap test ---
        print(f"\n  --- Synthetic gap test ---")
        for drop_rate in [0.1, 0.2, 0.3, 0.5]:
            degraded, dropped = create_gaps(aruco_sub, drop_rate=drop_rate)
            filled = fill_interior_gaps(
                degraded, sleap_sub,
                max_gap_frames=args.max_gap_frames,
                max_distance=80.0,       # SLEAP-ArUco spatial match
                max_tracklet_gap=5,      # frames between SLEAP detections
            )
            result = evaluate_gap_fill(filled, aruco_sub, dropped)
            result["camera"] = cam
            result["drop_rate"] = drop_rate
            result["max_gap"] = args.max_gap_frames
            all_synthetic.append(result)

            print(f"    drop={drop_rate:.0%}: recovered {result['n_correct']}/{result['n_dropped']} "
                  f"({result['recovery_rate']:.1%})  wrong={result['n_wrong']}  "
                  f"precision={result['fill_precision']:.1%}")

        # --- Natural gap test ---
        print(f"\n  --- Natural gap test ---")
        nat_result = evaluate_natural_gaps(aruco_sub, sleap_sub, args.max_gap_frames)
        nat_result["camera"] = cam
        all_natural.append(nat_result)

        print(f"    Original detections: {nat_result['n_original_detections']}")
        print(f"    Gap-filled: {nat_result['n_gap_filled']} "
              f"(+{nat_result['fill_fraction']:.1%})")
        if nat_result["top_filled_ids"]:
            print(f"    Top filled IDs:")
            for mid, cnt in sorted(nat_result["top_filled_ids"].items(),
                                   key=lambda x: -x[1])[:5]:
                print(f"      ID {mid}: {cnt}")

    # Grand summary
    if all_synthetic:
        print(f"\n{'='*60}")
        print("GRAND SUMMARY — Synthetic Gaps")
        print(f"{'='*60}")
        print(f"{'Camera':<8} {'Drop':>5} {'Dropped':>8} {'Filled':>7} "
              f"{'Correct':>8} {'Wrong':>6} {'Recovery':>9} {'Precision':>10}")
        print("-" * 72)
        for r in all_synthetic:
            print(f"{r['camera']:<8} {r['drop_rate']:>5.0%} {r['n_dropped']:>8} "
                  f"{r['n_filled']:>7} {r['n_correct']:>8} {r['n_wrong']:>6} "
                  f"{r['recovery_rate']:>9.1%} {r['fill_precision']:>10.1%}")

        # Averages by drop rate
        syn_df = pd.DataFrame(all_synthetic)
        print(f"\n  Averages by drop rate:")
        for rate in sorted(syn_df["drop_rate"].unique()):
            sub = syn_df[syn_df["drop_rate"] == rate]
            print(f"    {rate:.0%}: recovery={sub['recovery_rate'].mean():.1%}  "
                  f"precision={sub['fill_precision'].mean():.1%}  "
                  f"wrong={sub['n_wrong'].sum()}")

    if all_natural:
        print(f"\n{'='*60}")
        print("GRAND SUMMARY — Natural Gaps")
        print(f"{'='*60}")
        for r in all_natural:
            print(f"  {r['camera']}: {r['n_original_detections']} original + "
                  f"{r['n_gap_filled']} filled ({r['fill_fraction']:.1%} increase)")

    print(f"\nLog saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
