import json

import numpy as np

from tracking.single_ant.speed_viewer import SpeedData, speed_window


def test_overview_keeps_peaks_and_breaks_at_short_recording_gap():
    v = np.array([1., 1., 100., 1., np.nan, np.nan, 1., 1., np.nan, np.nan,
                  *([2.]*10)])
    segments = [dict(frame_offset=0, num_frames=8), dict(frame_offset=10, num_frames=10)]
    points, width = speed_window(v, 1, 0, 20/3600, segments, max_bins=2)
    assert width == 10
    assert len(points) == 3 and np.isnan(points[1]).all()
    assert points[0, 1] == 105/6 and points[0, 2] == 1 and points[0, 3] == 100
    assert points[0, 4] == 75  # Missing speeds do not become zero-valued samples.
    assert points[2, 1] == points[2, 2] == points[2, 3] == 2


def test_zoom_resolves_each_original_frame_and_keeps_nan():
    v = np.arange(200, dtype=float); v[102] = np.nan
    points, width = speed_window(v, 24, 100/24/3600, 104/24/3600,
                                 [dict(frame_offset=0, num_frames=len(v))])
    assert width == 1
    np.testing.assert_allclose(points[:, 0]*3600*24, [100, 101, 102, 103])
    np.testing.assert_allclose(points[:, 1], v[100:104], equal_nan=True)
    assert np.isnan(points[2, 2:4]).all() and points[2, 4] == 0


def test_results_use_local_mount_and_keep_compound_and_unassigned_cups(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(output='/unavailable/remote/mount',
        fps=2, n_frames=4, start_datetime='2026-10-27T21:49:46+09:00',
        cups=[dict(cup_id=1),dict(cup_id=2)])))
    (tmp_path/'cup_identities.csv').write_text(
        'CupID,TrackID,ArucoID,Eligible\n1,1,7,True\n1,3,8,True\n2,2,-1,False\n')
    for tid in [1, 3]:
        cache = tmp_path/'analysis/speed_vectors/per_track'/f'TrackID_{tid:04d}_all_single'
        cache.mkdir(parents=True)
        np.save(cache/'speed_mm_s.npy', np.full(4, tid, dtype=np.float32))
        np.save(cache/'speed_unclipped_mm_s.npy', np.full(4, 10*tid, dtype=np.float32))
    d = SpeedData(tmp_path)
    assert [(c['cup'],c['track']) for c in d.choices] == [(1,1),(1,3),(2,None)]
    np.testing.assert_array_equal(d.speed(1), np.full(4,3))
    np.testing.assert_array_equal(d.speed(1,True), np.full(4,30))
    assert d.speed(2) is None
