#!/usr/bin/env python3
"""Tests for the evaluation harness matching core.

Runs standalone (there is no pytest in the sleap-nn env):
    /apps/unit/ReiterU/sleap-nn/0.3.1/tools/sleap-nn/bin/python \
        group_labelling/test_eval_match.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import (  # noqa: E402
    BURST_HALF_WIDTH,
    burst_frames,
    in_interior,
    match_lsap,
    sample_anchors,
    single_linkage_clusters,
)

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  PASS  {msg}")
    else:
        print(f"  FAIL  {msg}")
        FAILURES.append(msg)


def greedy_match_counts(anchors: np.ndarray, refs: np.ndarray, radius: float):
    """Faithful reproduction of build_training_inventory's matcher, for the
    regression test below: argmin then radius test, with n_matched counting
    every in-radius anchor while n_unmatched_aruco de-duplicates via set()."""
    d2 = ((anchors[:, None, :] - refs[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1)
    nearest_d2 = d2[np.arange(len(anchors)), nearest]
    in_radius = nearest_d2 <= radius ** 2
    matched = nearest[in_radius]
    n_matched = int(in_radius.sum())
    n_unmatched_ref = len(refs) - len(set(matched.tolist()))
    return n_matched, n_unmatched_ref


def test_greedy_pathology_regression() -> None:
    """The defect that motivates match_lsap.

    Three model detections pile onto one ant carrying one tag; a second tag
    nearby has no detection. The greedy matcher reports n_matched=3 against
    |refs|=2 -- recall 1.5 -- and simultaneously reports the second tag as
    matched, because set() collapsed the three claims onto tag 0.
    """
    print("test_greedy_pathology_regression")
    refs = np.array([[100.0, 100.0], [140.0, 100.0]])
    anchors = np.array([[98.0, 100.0], [102.0, 101.0], [100.0, 97.0]])
    radius = 30.0

    n_matched, n_unmatched_ref = greedy_match_counts(anchors, refs, radius)
    greedy_recall = n_matched / len(refs)
    check(greedy_recall > 1.0,
          f"greedy matcher exhibits recall > 1 (got {greedy_recall:.2f})")
    check(n_matched + n_unmatched_ref != len(refs),
          f"greedy TP+FN != |refs| ({n_matched}+{n_unmatched_ref} != {len(refs)})")

    m = match_lsap(anchors, refs, radius)
    tp, fn, fp = len(m.pairs), len(m.unmatched_b), len(m.unmatched_a)
    check(tp + fn == len(refs), f"LSAP TP+FN == |refs| ({tp}+{fn}=={len(refs)})")
    check(tp + fp == len(anchors),
          f"LSAP TP+FP == |detections| ({tp}+{fp}=={len(anchors)})")
    check(tp / len(refs) <= 1.0, f"LSAP recall <= 1 (got {tp / len(refs):.2f})")
    check(len(set(m.pairs[:, 1].tolist())) == tp,
          "LSAP assigns each reference at most once")


def test_exclusivity_invariants() -> None:
    print("test_exclusivity_invariants")
    rng = np.random.default_rng(0)
    bad = None
    for trial in range(200):
        na, nb = int(rng.integers(0, 12)), int(rng.integers(0, 12))
        a = rng.uniform(0, 200, size=(na, 2))
        b = rng.uniform(0, 200, size=(nb, 2))
        m = match_lsap(a, b, 20.0)
        if len(m.pairs) + len(m.unmatched_a) != na:
            bad = f"trial {trial}: pairs+unmatched_a != na"
        elif len(m.pairs) + len(m.unmatched_b) != nb:
            bad = f"trial {trial}: pairs+unmatched_b != nb"
        elif len(m.pairs) and m.cost.max() > 20.0:
            bad = f"trial {trial}: pair beyond radius"
        elif len(set(m.pairs[:, 0].tolist())) != len(m.pairs):
            bad = f"trial {trial}: duplicate a index"
        elif len(set(m.pairs[:, 1].tolist())) != len(m.pairs):
            bad = f"trial {trial}: duplicate b index"
        if bad:
            break
    check(bad is None,
          f"partition + radius + exclusivity over 200 random cases ({bad or 'ok'})")


def test_optimality_beats_greedy() -> None:
    """A case where greedy nearest-neighbour strands a reference the optimum
    covers: both detections are nearest to ref0."""
    print("test_optimality_beats_greedy")
    refs = np.array([[0.0, 0.0], [10.0, 0.0]])
    dets = np.array([[1.0, 0.0], [4.0, 0.0]])
    m = match_lsap(dets, refs, 12.0)
    check(len(m.pairs) == 2, f"LSAP covers both references (got {len(m.pairs)})")

    both_to_ref0 = (((dets[:, None, :] - refs[None, :, :]) ** 2)
                    .sum(axis=2).argmin(axis=1) == 0).all()
    check(bool(both_to_ref0),
          "greedy would send both detections to the same reference")


def test_radius_gate_is_hard() -> None:
    print("test_radius_gate_is_hard")
    a = np.array([[0.0, 0.0]])
    check(len(match_lsap(a, np.array([[9.99, 0.0]]), 10.0).pairs) == 1,
          "pair just inside radius kept")
    check(len(match_lsap(a, np.array([[10.01, 0.0]]), 10.0).pairs) == 0,
          "pair just outside radius rejected")


def test_ambiguity_flag() -> None:
    print("test_ambiguity_flag")
    dets = np.array([[0.0, 0.0]])
    m = match_lsap(dets, np.array([[3.0, 0.0], [6.0, 0.0]]), 10.0)
    check(bool(m.ambiguous[0]), "runner-up inside radius flags AMBIGUOUS")
    m2 = match_lsap(dets, np.array([[3.0, 0.0], [60.0, 0.0]]), 10.0)
    check(not bool(m2.ambiguous[0]), "runner-up outside radius is unambiguous")


def test_cluster_cancellation_property() -> None:
    """The design's load-bearing property: a cluster in which every detection
    of one model pairs with one of the other contributes exactly zero to both
    dFP and dFN, whatever the true ant count in it is."""
    print("test_cluster_cancellation_property")
    d_a0 = np.array([[10.0, 10.0], [200.0, 200.0]])
    d_all6 = np.array([[11.0, 10.5], [200.5, 199.0]])
    union = np.vstack([d_a0, d_all6])
    labels = single_linkage_clusters(union, 25.0)
    check(len(set(labels.tolist())) == 2, "two well-separated clusters found")

    for lab in sorted(set(labels.tolist())):
        idx = np.where(labels == lab)[0]
        a_idx = [i for i in idx if i < len(d_a0)]
        b_idx = [i - len(d_a0) for i in idx if i >= len(d_a0)]
        m = match_lsap(d_a0[a_idx], d_all6[b_idx], 8.0)
        exclusive = len(m.unmatched_a) + len(m.unmatched_b)
        check(exclusive == 0,
              f"cluster {lab} fully paired -> cancels with coefficient zero")


def test_disagreement_cluster_is_detected() -> None:
    """A cluster with a model-exclusive detection must NOT cancel -- it is the
    adjudication population."""
    print("test_disagreement_cluster_is_detected")
    d_a0 = np.array([[10.0, 10.0], [14.0, 12.0]])   # A0 emits two peaks here
    d_all6 = np.array([[10.5, 10.2]])               # all6 emits one
    m = match_lsap(d_a0, d_all6, 8.0)
    check(len(m.unmatched_a) == 1,
          "the A0-exclusive detection survives pairing and carries signal")
    check(len(m.pairs) == 1, "the agreed detection pairs off")


def test_mega_cluster_chaining() -> None:
    print("test_mega_cluster_chaining")
    chain = np.array([[float(i) * 20.0, 0.0] for i in range(10)])
    check(len(set(single_linkage_clusters(chain, 25.0).tolist())) == 1,
          "single-linkage chains a 20 px-spaced row into one MEGA component")
    check(len(set(single_linkage_clusters(chain, 15.0).tolist())) == 10,
          "a tighter radius separates the same row into singletons")


def test_sampling_is_deterministic_and_in_range() -> None:
    print("test_sampling_is_deterministic_and_in_range")
    a1, r1, _ = sample_anchors("block01", "cam10", 37144)
    a2, r2, _ = sample_anchors("block01", "cam10", 37144)
    check(np.array_equal(a1, a2) and np.array_equal(r1, r2),
          "sampling is reproducible across calls")
    check(len(a1) == len(set(a1.tolist())), "anchors are unique")
    check(a1.min() >= BURST_HALF_WIDTH
          and a1.max() <= 37144 - 1 - BURST_HALF_WIDTH,
          f"anchors leave room for bursts ({a1.min()}..{a1.max()})")
    check(len(a1) == 600, f"600 anchors drawn (got {len(a1)})")
    check(np.bincount(r1, minlength=6).min() > 0, "all 6 replicates populated")

    other, _, _ = sample_anchors("block02", "cam10", 37144)
    check(not np.array_equal(a1, other),
          "a different block yields a different phase")

    spread = a1.max() - a1.min()
    check(spread > 0.9 * 37144,
          f"anchors span the whole recording ({spread}/37144)")

    big, rbig, _ = sample_anchors("block02", "cam10", 5_754_108)
    check(len(big) == 600 and big.max() <= 5_754_108 - 1 - BURST_HALF_WIDTH,
          "sampling scales to the 5.75M-frame block02 recording")
    check(np.bincount(rbig, minlength=6).min() > 0,
          "all replicates populated at block02 scale")


def test_bursts() -> None:
    print("test_bursts")
    frames, owner = burst_frames(np.array([10, 100, 1000]))
    check(len(frames) == 3 * (2 * BURST_HALF_WIDTH + 1),
          f"7 frames per anchor (got {len(frames)})")
    check(len(frames) == len(set(frames.tolist())), "burst frames de-duplicated")
    check(bool((np.diff(frames) > 0).all()), "burst frames sorted")
    check(owner[0] == 10, "anchor ownership recorded")

    overlap, _ = burst_frames(np.array([10, 12]))
    check(len(overlap) == len(set(overlap.tolist())),
          "overlapping bursts decode each frame once")


def test_edge_mask_symmetric() -> None:
    print("test_edge_mask_symmetric")
    pts = np.array([[10.0, 10.0], [2000.0, 1500.0], [4000.0, 3000.0]])
    check(list(in_interior(pts, 4024, 3036, 50.0)) == [False, True, False],
          "50 px band excluded on both borders")


def main() -> int:
    for fn in (
        test_greedy_pathology_regression,
        test_exclusivity_invariants,
        test_optimality_beats_greedy,
        test_radius_gate_is_hard,
        test_ambiguity_flag,
        test_cluster_cancellation_property,
        test_disagreement_cluster_is_detected,
        test_mega_cluster_chaining,
        test_sampling_is_deterministic_and_in_range,
        test_bursts,
        test_edge_mask_symmetric,
    ):
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
