import time

import numpy as np
import pandas as pd

from tracking.colony.skeleton_contacts import SKELETON_EDGES, skeleton_contact_pairs, skeleton_pair_distances
from tracking.colony.interaction_one_chunk import detect_interactions


def test_indexed_worker_matches_bruteforce_viewer():
    rng = np.random.default_rng(19)
    for _ in range(12):
        poses = {i: rng.normal(size=(10, 2))*20 for i in range(9)}
        poses[1][4:7] = np.nan
        poses[2][:] = np.nan
        poses[8] = poses[7].copy()
        reviewed = skeleton_pair_distances(poses, mm_per_pixel=.016, edges=SKELETON_EDGES)
        for threshold in [0, .01, .1, 1]:
            expected = {(p.ant_a, p.ant_b) for p in reviewed if p.is_hit(threshold)}
            computed = skeleton_contact_pairs(poses, distance_mm=threshold, mm_per_pixel=.016)
            assert {(a, b) for a, b, d in computed} == expected
            assert len(computed) == len(expected)


def test_worker_counts_crossing_without_antennae_or_nearby_anchors():
    rows = []
    for frame in [0, 1, 2]:
        for ant, xy in [(1, [[-10, 0], [10, 0]]), (2, [[0, -10], [0, 10]])]:
            for node, (x, y) in enumerate(xy):
                rows.append(dict(Frame=frame, TrackID=ant, Bodypoint=node, X=x, Y=y))
    result, _ = detect_interactions(pd.DataFrame(rows), micro_distance_px=0, mm_per_px=.016,
                                   progress_every_frames=0, run_start_time=time.perf_counter(),
                                   processed_frames_before=0, interactions_before=0)
    assert result.Frame.tolist() == [0, 1, 2]
    assert result.ant_a.tolist() == [1, 1, 1]
    assert result.ant_b.tolist() == [2, 2, 2]
    assert result.distance_mm.eq(0).all()
