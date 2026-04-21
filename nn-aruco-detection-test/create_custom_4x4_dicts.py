#!/usr/bin/env python3
"""Create two custom 4x4 ArUco dictionaries with maximum inter-marker distance.

Searches the FULL 2^16 = 65536 pattern space (not just OpenCV's predefined
dictionaries) to find optimal marker sets. Supports joint optimization so
both dictionaries can coexist in the same video without cross-confusion.

Pipeline:
  Phase 1: Enumerate all 65536 patterns, collapse rotation equivalence classes,
           filter problematic patterns -> ~14000-15000 candidates
  Phase 2: Compute pairwise rotation-aware Hamming distance matrix (chunked uint16)
  Phase 3: Optimize via greedy multi-seed + local swap + simulated annealing
  Phase 4: Output OpenCV-compatible dictionaries, JSON manifests, contact sheets

Usage:
    python nn-aruco-detection-test/create_custom_4x4_dicts.py \
        --dict-a-count 100 --dict-b-count 300 --joint --algorithm full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

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
# Phase 1: Candidate Universe
# ---------------------------------------------------------------------------

def pattern_to_uint16(mat: np.ndarray) -> int:
    """Convert a (4,4) binary matrix to a uint16 integer (row-major, MSB first)."""
    flat = mat.flatten()
    val = 0
    for bit in flat:
        val = (val << 1) | int(bit)
    return val


def uint16_to_pattern(val: int) -> np.ndarray:
    """Convert a uint16 integer to a (4,4) binary matrix."""
    mat = np.zeros((4, 4), dtype=np.uint8)
    for i in range(15, -1, -1):
        row, col = divmod(15 - i, 4)
        mat[row, col] = (val >> i) & 1
    return mat


def rotate_uint16(val: int) -> list[int]:
    """Return all 4 rotations of a 4x4 pattern as uint16 values."""
    mat = uint16_to_pattern(val)
    rotations = []
    for k in range(4):
        rotated = np.rot90(mat, k=k)
        rotations.append(pattern_to_uint16(rotated))
    return rotations


def build_candidate_universe(
    min_self_rotation: int = 2,
    min_bits: int = 3,
    max_bits: int = 13,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build the filtered candidate universe from all 2^16 patterns.

    Returns:
        canonical_codes: (N,) uint16 array of canonical pattern codes
        canonical_patterns: (N, 4, 4) uint8 array of binary patterns
        stats: dictionary with universe statistics
    """
    t0 = time.time()

    # Step 1: enumerate all 65536 patterns, find canonical forms
    seen_canonical = {}  # canonical_code -> (code, self_rot_d, popcount)

    for val in range(65536):
        rotations = rotate_uint16(val)
        canonical = min(rotations)

        if canonical in seen_canonical:
            continue

        # Compute popcount (number of set bits)
        popcount = bin(canonical).count("1")

        # Compute self-rotation distance: min Hamming to any non-identity rotation
        mat = uint16_to_pattern(canonical)
        self_rot_d = 16  # max possible
        for k in range(1, 4):
            rotated = np.rot90(mat, k=k)
            d = int(np.sum(mat != rotated))
            if d < self_rot_d:
                self_rot_d = d

        seen_canonical[canonical] = (canonical, self_rot_d, popcount)

    total_classes = len(seen_canonical)

    # Step 2: filter
    filtered = []
    n_symmetric = 0
    n_near_symmetric = 0
    n_uniform = 0

    for canonical, (code, self_rot_d, popcount) in seen_canonical.items():
        if self_rot_d == 0:
            n_symmetric += 1
            continue
        if self_rot_d < min_self_rotation:
            n_near_symmetric += 1
            continue
        if popcount < min_bits or popcount > max_bits:
            n_uniform += 1
            continue
        filtered.append(code)

    filtered.sort()
    N = len(filtered)

    # Build arrays
    canonical_codes = np.array(filtered, dtype=np.uint16)
    canonical_patterns = np.zeros((N, 4, 4), dtype=np.uint8)
    for i, code in enumerate(filtered):
        canonical_patterns[i] = uint16_to_pattern(code)

    elapsed = time.time() - t0

    stats = {
        "total_raw_patterns": 65536,
        "total_equiv_classes": total_classes,
        "filtered_symmetric": n_symmetric,
        "filtered_near_symmetric": n_near_symmetric,
        "filtered_uniform": n_uniform,
        "candidates_after_filter": N,
        "elapsed_seconds": elapsed,
    }

    return canonical_codes, canonical_patterns, stats


# ---------------------------------------------------------------------------
# Phase 2: Distance Matrix (chunked uint16 + popcount)
# ---------------------------------------------------------------------------

def build_popcount_lut() -> np.ndarray:
    """Build a 65536-entry popcount lookup table for uint16 values."""
    lut = np.zeros(65536, dtype=np.uint8)
    for i in range(65536):
        lut[i] = bin(i).count("1")
    return lut


def compute_distance_matrix_uint16(
    canonical_codes: np.ndarray,
    chunk_size: int = 1000,
) -> np.ndarray:
    """Compute rotation-aware Hamming distance matrix using uint16 packing.

    For each pair (i, j), computes min over all 4 rotations of j:
        d(i, j) = min_r popcount(code_i XOR rot_r(code_j))

    Returns: (N, N) uint8 distance matrix (symmetric).
    """
    t0 = time.time()
    N = len(canonical_codes)
    popcount = build_popcount_lut()

    # Pre-compute all 4 rotations for each pattern
    rotated_codes = np.zeros((N, 4), dtype=np.uint16)
    for i in range(N):
        rots = rotate_uint16(int(canonical_codes[i]))
        for r in range(4):
            rotated_codes[i, r] = rots[r]

    # Distance matrix
    D = np.full((N, N), 16, dtype=np.uint8)

    n_chunks = (N + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        i_start = ci * chunk_size
        i_end = min(i_start + chunk_size, N)
        chunk_codes = canonical_codes[i_start:i_end].astype(np.uint32)

        for r in range(4):
            rot_col = rotated_codes[:, r].astype(np.uint32)
            # XOR: (chunk, 1) ^ (1, N) -> (chunk, N) as uint32
            xor = chunk_codes[:, None] ^ rot_col[None, :]
            # Popcount via lookup on uint16 (xor values fit in uint16)
            d_r = popcount[xor.astype(np.uint16)]
            D[i_start:i_end] = np.minimum(D[i_start:i_end], d_r)

        if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
            pct = 100 * (ci + 1) / n_chunks
            print(f"    Distance matrix: {pct:.0f}% ({ci+1}/{n_chunks} chunks)")

    # Make symmetric (take element-wise min of D and D.T)
    D = np.minimum(D, D.T)
    np.fill_diagonal(D, 0)

    elapsed = time.time() - t0
    print(f"    Distance matrix computed in {elapsed:.1f}s")
    return D


# ---------------------------------------------------------------------------
# Phase 3: Optimization
# ---------------------------------------------------------------------------

def greedy_farthest_first(
    D: np.ndarray,
    target_count: int | None = None,
    min_distance: int | None = None,
    seed_id: int = 0,
    excluded: set[int] | None = None,
    cross_distance: np.ndarray | None = None,
    cross_min_d: int | None = None,
) -> tuple[list[int], int]:
    """Greedy max-min distance subset selection.

    Args:
        D: (N, N) distance matrix
        target_count: stop when this many IDs are selected
        min_distance: stop when the next best candidate has distance < this
        seed_id: starting ID
        excluded: set of indices that cannot be selected (for joint optimization)
        cross_distance: (N,) array of min distances to already-selected dict A
        cross_min_d: minimum required distance to dict A markers

    Returns: (selected_indices, min_pairwise_distance)
    """
    n = D.shape[0]
    if excluded is None:
        excluded = set()

    selected = [seed_id]
    remaining = set(range(n)) - {seed_id} - excluded

    # min_dist_to_set[i] = min distance from i to any selected ID
    min_dist_to_set = D[seed_id].astype(np.int32).copy()

    while remaining:
        best_id = -1
        best_dist = -1
        for r in remaining:
            # Check cross-dictionary constraint
            if cross_distance is not None and cross_min_d is not None:
                if cross_distance[r] < cross_min_d:
                    continue
            if min_dist_to_set[r] > best_dist:
                best_dist = min_dist_to_set[r]
                best_id = r

        if best_id == -1:
            break
        if min_distance is not None and best_dist < min_distance:
            break
        if target_count is not None and len(selected) >= target_count:
            break

        selected.append(best_id)
        remaining.remove(best_id)

        for r in remaining:
            d = int(D[best_id, r])
            if d < min_dist_to_set[r]:
                min_dist_to_set[r] = d

    # Compute actual min pairwise distance
    if len(selected) < 2:
        actual_min = 16
    else:
        sel = np.array(selected)
        sub_D = D[np.ix_(sel, sel)]
        np.fill_diagonal(sub_D, 255)
        actual_min = int(sub_D.min())

    return selected, actual_min


def greedy_multi_seed(
    D: np.ndarray,
    target_count: int,
    n_seeds: int = 100,
    excluded: set[int] | None = None,
    cross_distance: np.ndarray | None = None,
    cross_min_d: int | None = None,
    rng_seed: int = 42,
) -> tuple[list[int], int]:
    """Run greedy with multiple seeds, return best result (highest min_d)."""
    n = D.shape[0]
    if excluded is None:
        excluded = set()

    available = sorted(set(range(n)) - excluded)
    rng = np.random.default_rng(rng_seed)
    seeds = [available[0]] + list(rng.choice(available, size=min(n_seeds - 1, len(available)), replace=False))

    best_ids = []
    best_min_d = -1

    for seed in seeds:
        if seed in excluded:
            continue
        ids, min_d = greedy_farthest_first(
            D, target_count=target_count, seed_id=int(seed),
            excluded=excluded, cross_distance=cross_distance, cross_min_d=cross_min_d,
        )
        if min_d > best_min_d or (min_d == best_min_d and len(ids) > len(best_ids)):
            best_min_d = min_d
            best_ids = ids

    return best_ids, best_min_d


def local_swap_improvement(
    D: np.ndarray,
    selected: list[int],
    excluded: set[int] | None = None,
    cross_distance: np.ndarray | None = None,
    cross_min_d: int | None = None,
    max_rounds: int = 50,
) -> tuple[list[int], int]:
    """Improve a selection via 1-swap local search.

    For each selected marker, try replacing it with every non-selected marker.
    Accept if min_distance improves, or stays same while mean distance improves.
    """
    if excluded is None:
        excluded = set()
    n = D.shape[0]
    selected = list(selected)
    k = len(selected)

    def compute_min_d(sel):
        s = np.array(sel)
        sub = D[np.ix_(s, s)]
        np.fill_diagonal(sub, 255)
        return int(sub.min())

    def compute_mean_d(sel):
        s = np.array(sel)
        sub = D[np.ix_(s, s)].astype(np.float64)
        np.fill_diagonal(sub, 0)
        n_pairs = k * (k - 1)
        return sub.sum() / n_pairs if n_pairs > 0 else 0.0

    current_min_d = compute_min_d(selected)
    current_mean_d = compute_mean_d(selected)
    sel_set = set(selected)

    print(f"    Swap improvement starting: min_d={current_min_d}, mean_d={current_mean_d:.2f}")

    for round_num in range(max_rounds):
        improved = False
        for si in range(k):
            old_id = selected[si]
            candidates = set(range(n)) - sel_set - excluded

            for new_id in candidates:
                # Check cross-dictionary constraint
                if cross_distance is not None and cross_min_d is not None:
                    if cross_distance[new_id] < cross_min_d:
                        continue

                # Quick check: would adding new_id reduce min_d?
                # Check distances from new_id to all other selected (excluding old_id)
                min_new = 16
                for sj in range(k):
                    if sj == si:
                        continue
                    d = int(D[new_id, selected[sj]])
                    if d < min_new:
                        min_new = d
                        if min_new < current_min_d:
                            break

                if min_new < current_min_d:
                    continue

                # Full evaluation with swap
                trial = selected.copy()
                trial[si] = new_id
                trial_min_d = compute_min_d(trial)

                if trial_min_d > current_min_d:
                    selected = trial
                    sel_set = set(selected)
                    current_min_d = trial_min_d
                    current_mean_d = compute_mean_d(selected)
                    improved = True
                    break
                elif trial_min_d == current_min_d:
                    trial_mean_d = compute_mean_d(trial)
                    if trial_mean_d > current_mean_d:
                        selected = trial
                        sel_set = set(selected)
                        current_mean_d = trial_mean_d
                        improved = True
                        break

            if improved:
                break

        print(f"    Round {round_num+1}: min_d={current_min_d}, mean_d={current_mean_d:.2f}, improved={improved}")
        if not improved:
            break

    return selected, current_min_d


def simulated_annealing(
    D: np.ndarray,
    selected: list[int],
    n_iterations: int = 1_000_000,
    excluded: set[int] | None = None,
    cross_distance: np.ndarray | None = None,
    cross_min_d: int | None = None,
    rng_seed: int = 42,
    T_start: float = 2.0,
    T_end: float = 0.01,
) -> tuple[list[int], int]:
    """Simulated annealing to improve dictionary selection.

    Energy = -(min_d * 1000 + percentile_5th_d)
    Lower energy = better dictionary.
    """
    if excluded is None:
        excluded = set()

    n = D.shape[0]
    rng = np.random.default_rng(rng_seed)
    selected = list(selected)
    k = len(selected)
    sel_set = set(selected)
    available = sorted(set(range(n)) - sel_set - excluded)

    if not available:
        sel = np.array(selected)
        sub = D[np.ix_(sel, sel)]
        np.fill_diagonal(sub, 255)
        return selected, int(sub.min())

    # Precompute current pairwise distances in the selected set
    sel_arr = np.array(selected)
    sub_D = D[np.ix_(sel_arr, sel_arr)].copy()
    np.fill_diagonal(sub_D, 255)

    def compute_energy(sub_matrix):
        upper = sub_matrix[np.triu_indices(k, k=1)]
        min_d = int(upper.min())
        p5 = int(np.percentile(upper, 5))
        return -(min_d * 1000 + p5), min_d

    current_energy, current_min_d = compute_energy(sub_D)
    best_selected = selected.copy()
    best_energy = current_energy
    best_min_d = current_min_d

    cooling_rate = (T_end / T_start) ** (1.0 / max(n_iterations, 1))
    T = T_start

    n_accepted = 0
    n_improved = 0
    report_interval = max(n_iterations // 20, 1)

    for it in range(n_iterations):
        # Pick a random selected marker to swap out
        si = rng.integers(0, k)
        old_id = selected[si]

        # Pick a random available candidate
        # Filter by cross-dictionary constraint
        if cross_distance is not None and cross_min_d is not None:
            valid_available = [a for a in available if cross_distance[a] >= cross_min_d]
        else:
            valid_available = available

        if not valid_available:
            continue

        new_id = valid_available[rng.integers(0, len(valid_available))]

        # Compute new sub_D by replacing row/col si
        new_sub_D = sub_D.copy()
        for j in range(k):
            if j == si:
                new_sub_D[si, j] = 255
                new_sub_D[j, si] = 255
            else:
                d = int(D[new_id, selected[j]])
                new_sub_D[si, j] = d
                new_sub_D[j, si] = d

        new_energy, new_min_d = compute_energy(new_sub_D)

        # Accept or reject
        delta = new_energy - current_energy
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            # Accept
            available.remove(new_id)
            available.append(old_id)
            selected[si] = new_id
            sel_set.discard(old_id)
            sel_set.add(new_id)
            sub_D = new_sub_D
            current_energy = new_energy
            current_min_d = new_min_d
            n_accepted += 1

            if current_energy < best_energy:
                best_selected = selected.copy()
                best_energy = current_energy
                best_min_d = current_min_d
                n_improved += 1

        T *= cooling_rate

        if (it + 1) % report_interval == 0:
            pct = 100 * (it + 1) / n_iterations
            print(f"    SA {pct:.0f}%: T={T:.4f}, best_min_d={best_min_d}, "
                  f"current_min_d={current_min_d}, accepted={n_accepted}, improved={n_improved}")

    return best_selected, best_min_d


def find_max_min_distance(
    D: np.ndarray,
    target_count: int,
    n_seeds: int = 100,
    excluded: set[int] | None = None,
    cross_distance: np.ndarray | None = None,
    cross_min_d: int | None = None,
    rng_seed: int = 42,
) -> int:
    """Find the maximum achievable min_d for a given target count using multi-seed greedy."""
    _, min_d = greedy_multi_seed(
        D, target_count, n_seeds=n_seeds,
        excluded=excluded, cross_distance=cross_distance, cross_min_d=cross_min_d,
        rng_seed=rng_seed,
    )
    return min_d


# ---------------------------------------------------------------------------
# Phase 4: Output & Validation
# ---------------------------------------------------------------------------

def create_opencv_dictionary(
    patterns: np.ndarray,
    max_correction_bits: int,
) -> aruco.Dictionary:
    """Create an OpenCV ArUco Dictionary from custom (N, 4, 4) binary patterns."""
    n_markers = len(patterns)
    marker_size = 4

    # Use getByteListFromBits for each pattern
    bytes_list_parts = []
    for i in range(n_markers):
        # OpenCV expects: 0 = white, 1 = black in the bit matrix
        bits = patterns[i].astype(np.uint8)
        bl = aruco.Dictionary.getByteListFromBits(bits)  # shape (1, nbytes, 4)
        bytes_list_parts.append(bl)

    # Concatenate along axis 0 to get (nMarkers, nbytes, 4)
    bytes_list = np.concatenate(bytes_list_parts, axis=0)

    d = aruco.Dictionary()
    d.bytesList = bytes_list
    d.markerSize = marker_size
    d.maxCorrectionBits = max_correction_bits
    return d


def validate_round_trip(
    dictionary: aruco.Dictionary,
    patterns: np.ndarray,
    name: str,
) -> bool:
    """Validate that every marker can be generated and detected correctly."""
    n_markers = len(patterns)
    marker_size = 200  # render size for detection

    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)

    n_ok = 0
    n_fail = 0
    failures = []

    for mid in range(n_markers):
        # Generate marker image
        img = aruco.generateImageMarker(dictionary, mid, marker_size)

        # Add white border for detection
        bordered = np.ones((marker_size + 40, marker_size + 40), dtype=np.uint8) * 255
        bordered[20:20+marker_size, 20:20+marker_size] = img

        # Detect
        corners, ids, rejected = detector.detectMarkers(bordered)

        if ids is not None and len(ids) == 1 and ids[0][0] == mid:
            n_ok += 1
        else:
            n_fail += 1
            detected = int(ids[0][0]) if ids is not None and len(ids) > 0 else None
            failures.append((mid, detected))

    print(f"  Round-trip validation ({name}): {n_ok}/{n_markers} OK, {n_fail} failures")
    if failures:
        for mid, detected in failures[:10]:
            print(f"    ID {mid}: detected as {detected}")
    return n_fail == 0


def save_contact_sheet(
    dictionary: aruco.Dictionary,
    n_markers: int,
    path: Path,
    name: str,
    cols: int = 20,
    marker_px: int = 60,
    border_px: int = 10,
):
    """Save a contact sheet PNG with all markers in a grid."""
    rows = (n_markers + cols - 1) // cols
    cell = marker_px + border_px
    sheet_w = cols * cell + border_px
    sheet_h = rows * cell + border_px + 30  # 30px for header

    sheet = np.ones((sheet_h, sheet_w), dtype=np.uint8) * 255

    # Header
    cv2.putText(sheet, f"{name} ({n_markers} markers)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1)

    for mid in range(n_markers):
        r, c = divmod(mid, cols)
        x = c * cell + border_px
        y = r * cell + border_px + 30

        img = aruco.generateImageMarker(dictionary, mid, marker_px)
        sheet[y:y+marker_px, x:x+marker_px] = img

        # ID label
        cv2.putText(sheet, str(mid), (x + 2, y + marker_px + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, 128, 1)

    cv2.imwrite(str(path), sheet)
    print(f"  Contact sheet saved: {path}")


def save_dictionary_artifacts(
    patterns: np.ndarray,
    canonical_codes: np.ndarray,
    D_internal: np.ndarray,
    min_d: int,
    name: str,
    algorithm: str,
    output_dir: Path,
    timestamp: str,
    cross_min_d: int | None = None,
):
    """Save JSON manifest, NPZ, and contact sheet for a dictionary."""
    n_markers = len(patterns)
    max_corr = (min_d - 1) // 2

    # Create OpenCV dictionary
    opencv_dict = create_opencv_dictionary(patterns, max_corr)

    # Distance stats
    upper = D_internal[np.triu_indices(n_markers, k=1)]
    mean_d = float(upper.mean())
    p5_d = float(np.percentile(upper, 5))
    p25_d = float(np.percentile(upper, 25))

    # Self-rotation distances
    self_rot_dists = []
    for i in range(n_markers):
        mat = patterns[i]
        srd = 16
        for k in range(1, 4):
            rotated = np.rot90(mat, k=k)
            d = int(np.sum(mat != rotated))
            if d < srd:
                srd = d
        self_rot_dists.append(srd)

    # JSON manifest
    manifest = {
        "name": name,
        "description": f"Custom 4x4 ArUco dictionary with {n_markers} markers, min_d={min_d}",
        "n_markers": n_markers,
        "marker_size": 4,
        "min_hamming_distance": min_d,
        "mean_hamming_distance": round(mean_d, 2),
        "p5_hamming_distance": round(p5_d, 2),
        "p25_hamming_distance": round(p25_d, 2),
        "max_correction_bits": max_corr,
        "cross_dictionary_min_d": cross_min_d,
        "min_self_rotation_distance": int(min(self_rot_dists)),
        "algorithm": algorithm,
        "timestamp": timestamp,
        "source": "full_2^16_search",
        "canonical_uint16_codes": [int(c) for c in canonical_codes],
        "bit_patterns": [p.flatten().tolist() for p in patterns],
    }

    json_path = output_dir / f"{name}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  JSON manifest saved: {json_path}")

    # NPZ
    npz_path = output_dir / f"{name}_{timestamp}.npz"
    np.savez_compressed(
        str(npz_path),
        bytesList=opencv_dict.bytesList,
        patterns=patterns,
        canonical_codes=canonical_codes,
        max_correction_bits=np.array(max_corr),
        min_distance=np.array(min_d),
    )
    print(f"  NPZ saved: {npz_path}")

    # Contact sheet
    sheet_path = output_dir / f"{name}_contact_sheet_{timestamp}.png"
    save_contact_sheet(opencv_dict, n_markers, sheet_path, name)

    # Round-trip validation
    ok = validate_round_trip(opencv_dict, patterns, name)

    return opencv_dict, ok


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def analyze_universe(D: np.ndarray, name: str = "Candidate Universe"):
    """Print distance distribution statistics."""
    n = D.shape[0]
    upper = D[np.triu_indices(n, k=1)]

    print(f"\n{'='*60}")
    print(f"Distance Analysis: {name} ({n} candidates)")
    print(f"{'='*60}")
    print(f"  Min pairwise distance: {upper.min()}")
    print(f"  Max pairwise distance: {upper.max()}")
    print(f"  Mean pairwise distance: {upper.mean():.2f}")
    print(f"  Median pairwise distance: {np.median(upper):.1f}")

    max_d = int(upper.max())
    print(f"\n  Distance histogram:")
    for d in range(max_d + 1):
        count = int(np.sum(upper == d))
        if count > 0:
            bar_len = min(count * 60 // max(int(upper.shape[0]), 1), 60)
            bar = "#" * max(bar_len, 1) if count > 0 else ""
            print(f"    d={d:2d}: {count:>10,} pairs  {bar}")


def report_max_subsets(D: np.ndarray, n_seeds: int = 50):
    """Report max achievable subset sizes at each min distance."""
    n = D.shape[0]
    print(f"\n  Max subset size by minimum distance (greedy, {n_seeds} seeds):")
    for min_d in range(2, 10):
        best_count = 0
        rng = np.random.default_rng(42)
        seeds = [0] + list(rng.choice(n, size=min(n_seeds - 1, n), replace=False))
        for seed in seeds:
            ids, actual_d = greedy_farthest_first(D, min_distance=min_d, seed_id=int(seed))
            if len(ids) > best_count:
                best_count = len(ids)
        marker = ""
        if min_d == 4:
            marker = " <-- d1000 subset gave 94"
        if min_d == 5:
            marker = " <-- target for A=100?"
        print(f"    min_d={min_d}: up to {best_count:>5} markers{marker}")
        if best_count < 5:
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Create custom 4x4 ArUco dictionaries with maximum inter-marker distance"
    )
    p.add_argument("--dict-a-count", type=int, default=100,
                   help="Number of markers in dictionary A (default: 100)")
    p.add_argument("--dict-b-count", type=int, default=300,
                   help="Number of markers in dictionary B (default: 300)")
    p.add_argument("--joint", action="store_true", default=False,
                   help="Jointly optimize: enforce cross-dictionary distance")
    p.add_argument("--min-self-rotation", type=int, default=2,
                   help="Min self-rotation Hamming distance (default: 2)")
    p.add_argument("--min-bits", type=int, default=3,
                   help="Min set bits in pattern (default: 3)")
    p.add_argument("--max-bits", type=int, default=13,
                   help="Max set bits in pattern (default: 13)")
    p.add_argument("--algorithm", choices=["greedy", "greedy+swap", "full"], default="full",
                   help="Optimization algorithm (default: full)")
    p.add_argument("--seeds", type=int, default=100,
                   help="Number of random seeds for multi-start greedy (default: 100)")
    p.add_argument("--sa-iterations", type=int, default=1_000_000,
                   help="Simulated annealing iterations (default: 1000000)")
    p.add_argument("--output-dir", default="nn-aruco-detection-test/results/custom_dicts",
                   help="Output directory")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"custom_dict_log_{timestamp}.txt"
    tee = _TeeLogger(log_path)
    sys.stdout = tee

    print("=" * 60)
    print("Custom 4x4 ArUco Dictionary Generator")
    print("=" * 60)
    print(f"  Time: {timestamp}")
    print(f"  Dict A: {args.dict_a_count} markers")
    print(f"  Dict B: {args.dict_b_count} markers")
    print(f"  Joint optimization: {args.joint}")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Seeds: {args.seeds}")
    if args.algorithm == "full":
        print(f"  SA iterations: {args.sa_iterations:,}")
    print(f"  Filters: self_rot >= {args.min_self_rotation}, "
          f"bits in [{args.min_bits}, {args.max_bits}]")
    print(f"  Output: {output_dir}")
    print()

    # -----------------------------------------------------------------------
    # Phase 1: Build candidate universe
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Phase 1: Building candidate universe from 2^16 patterns")
    print("=" * 60)

    canonical_codes, canonical_patterns, stats = build_candidate_universe(
        min_self_rotation=args.min_self_rotation,
        min_bits=args.min_bits,
        max_bits=args.max_bits,
    )
    N = len(canonical_codes)

    print(f"  Raw patterns: {stats['total_raw_patterns']:,}")
    print(f"  Rotation equivalence classes: {stats['total_equiv_classes']:,}")
    print(f"  Filtered (symmetric): {stats['filtered_symmetric']}")
    print(f"  Filtered (near-symmetric, self_rot_d < {args.min_self_rotation}): "
          f"{stats['filtered_near_symmetric']}")
    print(f"  Filtered (too uniform): {stats['filtered_uniform']}")
    print(f"  Candidates after filtering: {N:,}")
    print(f"  Elapsed: {stats['elapsed_seconds']:.2f}s")

    if N < args.dict_a_count + args.dict_b_count:
        print(f"\n  ERROR: Not enough candidates ({N}) for requested "
              f"A={args.dict_a_count} + B={args.dict_b_count} = "
              f"{args.dict_a_count + args.dict_b_count}")
        tee.close()
        return

    # -----------------------------------------------------------------------
    # Phase 2: Distance matrix
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 2: Computing distance matrix")
    print("=" * 60)
    print(f"  Matrix size: {N} x {N} = {N*N:,} entries ({N*N/1e6:.1f} MB)")

    D = compute_distance_matrix_uint16(canonical_codes)

    analyze_universe(D)
    report_max_subsets(D, n_seeds=args.seeds)

    # -----------------------------------------------------------------------
    # Phase 3: Optimization
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 3: Optimization")
    print("=" * 60)

    algo_label = args.algorithm

    # --- Dictionary A ---
    print(f"\n--- Dictionary A ({args.dict_a_count} markers) ---")

    print(f"\n  Step 1: Greedy multi-seed ({args.seeds} seeds)...")
    t0 = time.time()
    sel_a, min_d_a = greedy_multi_seed(
        D, args.dict_a_count, n_seeds=args.seeds, rng_seed=args.seed,
    )
    print(f"  Greedy result: {len(sel_a)} markers, min_d={min_d_a} ({time.time()-t0:.1f}s)")

    if args.algorithm in ("greedy+swap", "full"):
        print(f"\n  Step 2: Local swap improvement...")
        t0 = time.time()
        sel_a, min_d_a = local_swap_improvement(D, sel_a)
        print(f"  Swap result: {len(sel_a)} markers, min_d={min_d_a} ({time.time()-t0:.1f}s)")

    if args.algorithm == "full":
        print(f"\n  Step 3: Simulated annealing ({args.sa_iterations:,} iterations)...")
        t0 = time.time()
        sel_a, min_d_a = simulated_annealing(
            D, sel_a, n_iterations=args.sa_iterations, rng_seed=args.seed,
        )
        print(f"  SA result: {len(sel_a)} markers, min_d={min_d_a} ({time.time()-t0:.1f}s)")

    print(f"\n  >>> Dict A final: {len(sel_a)} markers, min_d={min_d_a}")

    # --- Dictionary B ---
    print(f"\n--- Dictionary B ({args.dict_b_count} markers) ---")

    excluded_b = set()
    cross_dist_b = None
    cross_min_d_b = None

    if args.joint:
        excluded_b = set(sel_a)
        # Compute min distance from each candidate to dict A
        sel_a_arr = np.array(sel_a)
        cross_dist_b = D[:, sel_a_arr].min(axis=1).astype(np.int32)
        cross_min_d_b = min_d_a  # enforce same min_d as A for cross-dictionary
        print(f"  Joint mode: excluding {len(excluded_b)} dict-A markers, "
              f"cross_min_d={cross_min_d_b}")

        # Check feasibility: how many candidates have cross_dist >= cross_min_d?
        n_feasible = int(np.sum(cross_dist_b >= cross_min_d_b)) - len(excluded_b)
        print(f"  Feasible candidates for B (cross_dist >= {cross_min_d_b}): {n_feasible}")

        # If not enough, relax cross_min_d
        while n_feasible < args.dict_b_count and cross_min_d_b > 2:
            cross_min_d_b -= 1
            n_feasible = int(np.sum(cross_dist_b >= cross_min_d_b)) - len(excluded_b)
            print(f"  Relaxed cross_min_d to {cross_min_d_b}, feasible: {n_feasible}")

    print(f"\n  Step 1: Greedy multi-seed ({args.seeds} seeds)...")
    t0 = time.time()
    sel_b, min_d_b = greedy_multi_seed(
        D, args.dict_b_count, n_seeds=args.seeds,
        excluded=excluded_b, cross_distance=cross_dist_b, cross_min_d=cross_min_d_b,
        rng_seed=args.seed + 1,
    )
    print(f"  Greedy result: {len(sel_b)} markers, min_d={min_d_b} ({time.time()-t0:.1f}s)")

    if args.algorithm in ("greedy+swap", "full"):
        print(f"\n  Step 2: Local swap improvement...")
        t0 = time.time()
        sel_b, min_d_b = local_swap_improvement(
            D, sel_b, excluded=excluded_b,
            cross_distance=cross_dist_b, cross_min_d=cross_min_d_b,
        )
        print(f"  Swap result: {len(sel_b)} markers, min_d={min_d_b} ({time.time()-t0:.1f}s)")

    if args.algorithm == "full":
        print(f"\n  Step 3: Simulated annealing ({args.sa_iterations:,} iterations)...")
        t0 = time.time()
        sel_b, min_d_b = simulated_annealing(
            D, sel_b, n_iterations=args.sa_iterations,
            excluded=excluded_b, cross_distance=cross_dist_b, cross_min_d=cross_min_d_b,
            rng_seed=args.seed + 1,
        )
        print(f"  SA result: {len(sel_b)} markers, min_d={min_d_b} ({time.time()-t0:.1f}s)")

    print(f"\n  >>> Dict B final: {len(sel_b)} markers, min_d={min_d_b}")

    # --- Cross-dictionary distance ---
    if args.joint:
        sel_a_arr = np.array(sel_a)
        sel_b_arr = np.array(sel_b)
        cross_D = D[np.ix_(sel_a_arr, sel_b_arr)]
        actual_cross_min_d = int(cross_D.min())
        print(f"\n  Cross-dictionary min distance (A vs B): {actual_cross_min_d}")
    else:
        actual_cross_min_d = None

    # -----------------------------------------------------------------------
    # Phase 4: Output & Validation
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 4: Output & Validation")
    print("=" * 60)

    # Dict A
    patterns_a = canonical_patterns[sel_a]
    codes_a = canonical_codes[sel_a]
    D_a = D[np.ix_(np.array(sel_a), np.array(sel_a))]

    print(f"\n  Saving Dictionary A...")
    dict_a, ok_a = save_dictionary_artifacts(
        patterns_a, codes_a, D_a, min_d_a,
        name=f"custom_4x4_A{args.dict_a_count}_d{min_d_a}",
        algorithm=algo_label,
        output_dir=output_dir,
        timestamp=timestamp,
        cross_min_d=actual_cross_min_d,
    )

    # Dict B
    patterns_b = canonical_patterns[sel_b]
    codes_b = canonical_codes[sel_b]
    D_b = D[np.ix_(np.array(sel_b), np.array(sel_b))]

    print(f"\n  Saving Dictionary B...")
    dict_b, ok_b = save_dictionary_artifacts(
        patterns_b, codes_b, D_b, min_d_b,
        name=f"custom_4x4_B{args.dict_b_count}_d{min_d_b}",
        algorithm=algo_label,
        output_dir=output_dir,
        timestamp=timestamp,
        cross_min_d=actual_cross_min_d,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  Dictionary A: {len(sel_a)} markers, min_d={min_d_a}, "
          f"maxCorrectionBits={(min_d_a-1)//2}")
    print(f"  Dictionary B: {len(sel_b)} markers, min_d={min_d_b}, "
          f"maxCorrectionBits={(min_d_b-1)//2}")
    if actual_cross_min_d is not None:
        print(f"  Cross-dictionary min distance: {actual_cross_min_d}")
    print(f"  Round-trip A: {'PASS' if ok_a else 'FAIL'}")
    print(f"  Round-trip B: {'PASS' if ok_b else 'FAIL'}")
    print()
    print("  Comparison with OpenCV predefined dictionaries:")
    print(f"    DICT_4X4_50:   50 markers, min_d=4")
    print(f"    DICT_4X4_100: 100 markers, min_d=3")
    print(f"    DICT_4X4_250: 250 markers, min_d=3")
    print(f"    DICT_4X4_1000:1000 markers, min_d=2")
    print(f"    Custom A:     {len(sel_a):>3} markers, min_d={min_d_a}  <-- full 2^16 search")
    print(f"    Custom B:     {len(sel_b):>3} markers, min_d={min_d_b}  <-- full 2^16 search")
    print()

    # Verify no overlap
    if args.joint:
        overlap = set(sel_a) & set(sel_b)
        print(f"  Disjoint check: {'PASS (0 overlap)' if len(overlap) == 0 else f'FAIL ({len(overlap)} overlap)'}")

    print(f"\n  Log saved to: {log_path}")
    tee.close()


if __name__ == "__main__":
    main()
