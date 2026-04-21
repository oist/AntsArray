#!/usr/bin/env python3
"""Select max-separation ArUco ID subsets from DICT_4X4_1000.

Computes the full rotation-aware Hamming distance matrix for all 1000 IDs,
then finds subsets that maximize the minimum pairwise distance using a
greedy algorithm.  Also compares against DICT_4X4_50/100/250 baselines.

Outputs:
  - Distance matrix stats (histogram, percentiles)
  - Greedy max-separation subsets at target sizes (50, 100, 150, 200, 250)
  - Comparison with OpenCV's predefined small dictionaries
  - JSON files with the selected ID sets

Usage:
    python nn-aruco-detection-test/select_id_subset.py [--output-dir ...]
    python nn-aruco-detection-test/select_id_subset.py --target-count 100
    python nn-aruco-detection-test/select_id_subset.py --target-min-distance 4
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
# Pattern extraction
# ---------------------------------------------------------------------------

def extract_patterns(dictionary, n_markers: int, grid_size: int) -> np.ndarray:
    """Extract (N, grid_size, grid_size) binary patterns from an OpenCV ArUco dictionary."""
    patterns = np.zeros((n_markers, grid_size, grid_size), dtype=np.uint8)
    render_size = grid_size + 2  # 1px border on each side
    for mid in range(n_markers):
        img = aruco.generateImageMarker(dictionary, mid, render_size)
        data = img[1:render_size - 1, 1:render_size - 1]
        patterns[mid] = (data < 128).astype(np.uint8)
    return patterns


def build_rotation_lookup(patterns: np.ndarray) -> np.ndarray:
    """Build (N*4, bits) flattened lookup with all 4 rotations per marker."""
    n_markers = patterns.shape[0]
    grid = patterns.shape[1]
    n_bits = grid * grid
    lookup = np.zeros((n_markers * 4, n_bits), dtype=np.uint8)
    for mid in range(n_markers):
        for r in range(4):
            lookup[mid * 4 + r] = np.rot90(patterns[mid], k=r).flatten()
    return lookup


# ---------------------------------------------------------------------------
# Distance matrix computation
# ---------------------------------------------------------------------------

def compute_pairwise_distance_matrix(
    patterns: np.ndarray,
) -> np.ndarray:
    """Compute the rotation-aware minimum Hamming distance between all pairs.

    For each pair (i, j), computes min over all 4 rotations of j.

    Returns: (N, N) symmetric int matrix where D[i,j] = min rotational Hamming distance.
    """
    n = patterns.shape[0]
    grid = patterns.shape[1]
    n_bits = grid * grid

    # Flatten all patterns: (N, bits)
    flat = patterns.reshape(n, n_bits).astype(np.int8)

    # Build all 4 rotations for each pattern: (N, 4, bits)
    rotated = np.zeros((n, 4, n_bits), dtype=np.int8)
    for mid in range(n):
        for r in range(4):
            rotated[mid, r] = np.rot90(patterns[mid], k=r).flatten()

    # Distance matrix
    D = np.full((n, n), n_bits, dtype=np.int32)

    # Vectorized: for each rotation r, compute Hamming distance between
    # all pairs (i, j) where j is rotated by r
    for r in range(4):
        # flat: (N, bits), rotated[:, r, :]: (N, bits)
        # XOR and sum -> (N, N) distance matrix for this rotation
        # Using broadcasting: (N, 1, bits) XOR (1, N, bits) -> (N, N, bits) -> sum -> (N, N)
        # Memory: for N=1000, bits=16: 1000*1000*16 = 16M bytes — fine
        xor = flat[:, None, :] ^ rotated[None, :, r, :]
        d_r = xor.sum(axis=2).astype(np.int32)
        D = np.minimum(D, d_r)

    # Diagonal is always 0
    np.fill_diagonal(D, 0)
    return D


# ---------------------------------------------------------------------------
# Subset selection algorithms
# ---------------------------------------------------------------------------

def greedy_max_min_distance(
    D: np.ndarray,
    target_count: int | None = None,
    min_distance: int | None = None,
    seed_id: int = 0,
) -> tuple[list[int], int]:
    """Greedy max-min distance subset selection.

    At each step, adds the ID whose minimum distance to the current set is
    maximized (farthest-first traversal).

    Stops when target_count is reached OR min_distance cannot be maintained.

    Returns: (selected_ids, min_pairwise_distance)
    """
    n = D.shape[0]
    selected = [seed_id]
    remaining = set(range(n)) - {seed_id}

    # min_dist_to_set[i] = min distance from i to any selected ID
    min_dist_to_set = D[seed_id].copy()

    while remaining:
        # Find the remaining ID with max min-distance to current set
        best_id = -1
        best_dist = -1
        for r in remaining:
            if min_dist_to_set[r] > best_dist:
                best_dist = min_dist_to_set[r]
                best_id = r

        # Stop conditions
        if min_distance is not None and best_dist < min_distance:
            break
        if target_count is not None and len(selected) >= target_count:
            break

        selected.append(best_id)
        remaining.remove(best_id)

        # Update min distances
        for r in remaining:
            d = D[best_id, r]
            if d < min_dist_to_set[r]:
                min_dist_to_set[r] = d

    # Compute actual min pairwise distance of selected set
    if len(selected) < 2:
        actual_min = 0
    else:
        sel = np.array(selected)
        sub_D = D[np.ix_(sel, sel)]
        np.fill_diagonal(sub_D, 999)
        actual_min = int(sub_D.min())

    return selected, actual_min


def exhaustive_count_at_distance(
    D: np.ndarray,
    min_distance: int,
) -> int:
    """Count maximum subset size achievable with given minimum distance.

    Uses greedy (not truly exhaustive) but tries multiple seeds for a better bound.
    """
    n = D.shape[0]
    best_count = 0
    best_ids = []

    # Try 20 random seeds + ID 0
    seeds = [0] + list(np.random.default_rng(42).choice(n, size=min(20, n), replace=False))

    for seed in seeds:
        ids, min_d = greedy_max_min_distance(D, min_distance=min_distance, seed_id=seed)
        if len(ids) > best_count:
            best_count = len(ids)
            best_ids = ids

    return best_count


# ---------------------------------------------------------------------------
# Analysis and reporting
# ---------------------------------------------------------------------------

def analyze_distance_matrix(D: np.ndarray, name: str = "DICT_4X4_1000"):
    """Print statistics about the distance matrix."""
    n = D.shape[0]
    # Extract upper triangle (excluding diagonal)
    upper = D[np.triu_indices(n, k=1)]

    print(f"\n{'='*60}")
    print(f"Distance Matrix Analysis: {name} ({n} markers)")
    print(f"{'='*60}")
    print(f"  Total pairs: {len(upper):,}")
    print(f"  Min pairwise distance: {upper.min()}")
    print(f"  Max pairwise distance: {upper.max()}")
    print(f"  Mean pairwise distance: {upper.mean():.2f}")
    print(f"  Median pairwise distance: {np.median(upper):.1f}")

    # Histogram of distances
    n_bits = D.shape[0]  # not used, get from max val
    max_d = int(upper.max())
    print(f"\n  Distance histogram:")
    for d in range(max_d + 1):
        count = int(np.sum(upper == d))
        if count > 0:
            bar = '#' * min(count // max(1, len(upper) // 500), 60)
            print(f"    d={d:2d}: {count:>7,} pairs  {bar}")

    # Per-ID statistics: minimum distance to any other ID
    min_per_id = np.full(n, 999, dtype=np.int32)
    for i in range(n):
        dists_i = np.concatenate([D[i, :i], D[i, i+1:]])
        min_per_id[i] = dists_i.min()

    print(f"\n  Per-ID min-distance distribution:")
    for d in range(int(min_per_id.min()), int(min_per_id.max()) + 1):
        count = int(np.sum(min_per_id == d))
        if count > 0:
            print(f"    min_d={d}: {count} IDs")

    # IDs with the weakest separation
    worst_ids = np.argsort(min_per_id)[:10]
    print(f"\n  10 most vulnerable IDs (lowest min-distance to any other ID):")
    for idx in worst_ids:
        # Find the closest neighbor
        dists_i = D[idx].copy()
        dists_i[idx] = 999
        neighbor = int(np.argmin(dists_i))
        print(f"    ID {idx:4d}: min_d={min_per_id[idx]}, closest neighbor=ID {neighbor}")


def find_subsets(D: np.ndarray):
    """Find and report max-separation subsets at various target sizes."""
    print(f"\n{'='*60}")
    print("Greedy Max-Separation Subset Search")
    print(f"{'='*60}")

    # 1. How many IDs can we fit at each min distance?
    print(f"\n  Max subset size by minimum distance (greedy, multi-seed):")
    distance_vs_count = {}
    for min_d in range(1, 9):
        count = exhaustive_count_at_distance(D, min_d)
        distance_vs_count[min_d] = count
        marker = " <-- sweet spot" if min_d == 4 else ""
        print(f"    min_distance={min_d}: up to {count:>4} IDs{marker}")

    # 2. Find specific target-count subsets
    print(f"\n  Subsets at target counts:")
    targets = [50, 100, 150, 200, 250, 300, 500]
    results = {}
    for target in targets:
        if target > D.shape[0]:
            continue
        ids, min_d = greedy_max_min_distance(D, target_count=target)
        results[target] = (ids, min_d)

        # Also compute mean distance within subset
        sel = np.array(ids[:target])
        sub_D = D[np.ix_(sel, sel)]
        upper_sub = sub_D[np.triu_indices(len(sel), k=1)]

        print(f"    N={target:>3}: min_d={min_d}, mean_d={upper_sub.mean():.2f}, "
              f"selected {len(ids)} IDs")

    return results, distance_vs_count


def compare_predefined_dicts():
    """Compare properties of OpenCV predefined 4x4 dictionaries."""
    print(f"\n{'='*60}")
    print("OpenCV Predefined 4x4 Dictionary Comparison")
    print(f"{'='*60}")

    dicts = {
        "DICT_4X4_50": (aruco.DICT_4X4_50, 50),
        "DICT_4X4_100": (aruco.DICT_4X4_100, 100),
        "DICT_4X4_250": (aruco.DICT_4X4_250, 250),
        "DICT_4X4_1000": (aruco.DICT_4X4_1000, 1000),
    }

    for name, (dict_id, n_markers) in dicts.items():
        d = aruco.getPredefinedDictionary(dict_id)
        patterns = extract_patterns(d, n_markers, 4)
        D = compute_pairwise_distance_matrix(patterns)
        upper = D[np.triu_indices(n_markers, k=1)]

        min_per_id = np.full(n_markers, 999, dtype=np.int32)
        for i in range(n_markers):
            dists_i = np.concatenate([D[i, :i], D[i, i+1:]])
            min_per_id[i] = dists_i.min()

        print(f"\n  {name}:")
        print(f"    Markers: {n_markers}")
        print(f"    Global min distance: {upper.min()}")
        print(f"    Mean distance: {upper.mean():.2f}")
        print(f"    Per-ID min distance: min={min_per_id.min()}, "
              f"mean={min_per_id.mean():.1f}, max={min_per_id.max()}")


def check_subset_in_dict1000(other_dict_id, other_n: int, name: str):
    """Check if IDs from a smaller dictionary match IDs in DICT_4X4_1000.

    OpenCV predefined dicts are NOT subsets of each other — they are
    independently optimized. This function checks pattern-level compatibility.
    """
    d1000 = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    d_other = aruco.getPredefinedDictionary(other_dict_id)

    p1000 = extract_patterns(d1000, 1000, 4)
    p_other = extract_patterns(d_other, other_n, 4)

    # For each pattern in the other dict, find exact match (any rotation) in DICT_4X4_1000
    matches = {}
    for oid in range(other_n):
        for r in range(4):
            rotated = np.rot90(p_other[oid], k=r)
            for mid in range(1000):
                if np.array_equal(rotated, p1000[mid]):
                    matches[oid] = mid
                    break
            if oid in matches:
                break

    n_matched = len(matches)
    print(f"\n  {name} -> DICT_4X4_1000 mapping:")
    print(f"    {n_matched}/{other_n} patterns found in DICT_4X4_1000")
    if n_matched < other_n:
        missing = [i for i in range(other_n) if i not in matches]
        print(f"    Missing: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    return matches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Select max-separation ArUco ID subsets")
    p.add_argument("--output-dir", default="nn-aruco-detection-test/results")
    p.add_argument("--target-count", type=int, default=None,
                   help="Find a subset of exactly this size")
    p.add_argument("--target-min-distance", type=int, default=None,
                   help="Find max subset with at least this min distance")
    p.add_argument("--save-subsets", action="store_true", default=True,
                   help="Save selected ID subsets as JSON")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"id_subset_selection_log_{timestamp}.txt"
    tee = _TeeLogger(log_path)
    sys.stdout = tee

    print(f"=== ArUco ID Subset Selection ===")
    print(f"Time: {timestamp}")
    print()

    # 1. Compare predefined dictionaries
    compare_predefined_dicts()

    # 2. Check cross-dictionary compatibility
    print(f"\n{'='*60}")
    print("Cross-Dictionary Compatibility")
    print(f"{'='*60}")
    check_subset_in_dict1000(aruco.DICT_4X4_50, 50, "DICT_4X4_50")
    check_subset_in_dict1000(aruco.DICT_4X4_100, 100, "DICT_4X4_100")
    check_subset_in_dict1000(aruco.DICT_4X4_250, 250, "DICT_4X4_250")

    # 3. Full distance matrix for DICT_4X4_1000
    print(f"\nComputing DICT_4X4_1000 distance matrix...")
    d1000 = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    patterns = extract_patterns(d1000, 1000, 4)
    D = compute_pairwise_distance_matrix(patterns)
    print("  Done.")

    analyze_distance_matrix(D)

    # 4. Find subsets
    subset_results, dist_vs_count = find_subsets(D)

    # 5. Specific user queries
    if args.target_count:
        print(f"\n{'='*60}")
        print(f"Custom target: {args.target_count} IDs")
        print(f"{'='*60}")
        ids, min_d = greedy_max_min_distance(D, target_count=args.target_count)
        print(f"  Found {len(ids)} IDs with min distance {min_d}")
        print(f"  IDs: {sorted(ids[:args.target_count])}")

    if args.target_min_distance:
        print(f"\n{'='*60}")
        print(f"Custom target: min distance >= {args.target_min_distance}")
        print(f"{'='*60}")
        ids, min_d = greedy_max_min_distance(D, min_distance=args.target_min_distance)
        print(f"  Found {len(ids)} IDs with min distance {min_d}")
        if len(ids) <= 300:
            print(f"  IDs: {sorted(ids)}")

    # 6. Decision table
    print(f"\n{'='*60}")
    print("DECISION TABLE")
    print(f"{'='*60}")
    print()
    print(f"{'Strategy':<40} {'IDs':>5} {'Min d':>6} {'Notes'}")
    print("-" * 80)
    print(f"{'DICT_4X4_1000 (current, all 1000)':<40} {'1000':>5} {'2':>6} {'Baseline -- weakest separation'}")
    print(f"{'DICT_4X4_250 (drop-in replacement)':<40} {'250':>5} {'3':>6} {'Easy swap, backward-compatible codes'}")

    # Add greedy results
    for min_d_target in [3, 4, 5]:
        count = dist_vs_count.get(min_d_target, 0)
        if count > 0:
            note = ""
            if min_d_target == 4:
                note = "*** RECOMMENDED for ~100 ants ***"
            print(f"{'Custom subset (min_d=' + str(min_d_target) + ')':<40} {count:>5} {min_d_target:>6} {note}")

    print()
    print("Recommendation:")
    print("  - <=250 ants: use DICT_4X4_250 (simple, well-tested)")
    print("  - ~100 ants: custom subset with min_d=4 is feasible and safer")
    print("  - Always prefer code-space selection over filming 1000 IDs")

    # 7. Save subsets
    if args.save_subsets:
        for target in [100, 150, 200, 250]:
            if target in subset_results:
                ids, min_d = subset_results[target]
                ids_sorted = sorted(ids[:target])
                subset_path = out_dir / f"id_subset_{target}_mind{min_d}_{timestamp}.json"
                with open(subset_path, "w") as f:
                    json.dump({
                        "description": f"Max-separation {target}-ID subset of DICT_4X4_1000",
                        "n_ids": len(ids_sorted),
                        "min_hamming_distance": min_d,
                        "algorithm": "greedy farthest-first",
                        "source_dictionary": "DICT_4X4_1000",
                        "timestamp": timestamp,
                        "valid_ids": ids_sorted,
                    }, f, indent=2)
                print(f"\n  Saved: {subset_path}")

        # Also save the best min_d=4 subset (multi-seed for best result)
        best_d4_ids = []
        seeds = [0] + list(np.random.default_rng(42).choice(1000, size=20, replace=False))
        for seed in seeds:
            ids_try, _ = greedy_max_min_distance(D, min_distance=4, seed_id=int(seed))
            if len(ids_try) > len(best_d4_ids):
                best_d4_ids = ids_try
        d4_path = out_dir / f"id_subset_mind4_max_{len(best_d4_ids)}_{timestamp}.json"
        with open(d4_path, "w") as f:
            json.dump({
                "description": f"Max subset of DICT_4X4_1000 with min rotational Hamming distance 4",
                "n_ids": len(best_d4_ids),
                "min_hamming_distance": 4,
                "algorithm": "greedy farthest-first, multi-seed",
                "source_dictionary": "DICT_4X4_1000",
                "timestamp": timestamp,
                "valid_ids": sorted(best_d4_ids),
            }, f, indent=2)
        print(f"  Saved: {d4_path}")

    # 8. Check the 3 IDs from single-ant experiments
    print(f"\n{'='*60}")
    print("Current Experiment IDs Check")
    print(f"{'='*60}")
    current_ids = [3, 17, 25]
    print(f"  Single-ant experiment IDs: {current_ids}")
    for i, a in enumerate(current_ids):
        for j, b in enumerate(current_ids):
            if i < j:
                print(f"    d({a}, {b}) = {D[a, b]}")

    print(f"\nLog saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
