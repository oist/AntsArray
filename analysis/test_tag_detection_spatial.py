import numpy as np
import h5py
import pandas as pd

from analysis.exploratory.tag_detection_spatial import match_unique, owner_camera, signed_rectangle_distance, process_camera


def test_tag_is_not_reused_and_distant_tags_are_not_counted():
    hit, _, _ = match_unique(np.array([[0., 0.], [.1, 0.], [10., 10.]]), np.array([[0., 0.]]), .5)
    assert hit.tolist() == [True, False, False]


def test_matching_maximizes_number_before_minimizing_distance():
    hit, _, assigned = match_unique(np.array([[0., 0.], [.2, 0.]]), np.array([[.1, 0.], [-.2, 0.]]), .3)
    assert hit.all()
    assert assigned.tolist() == [1, 0]


def test_rectangle_sign_and_euclidean_corners():
    values = signed_rectangle_distance(np.array([[5, 5], [0, 5], [-3, -4], [12, 5]]), [0, 0, 10, 10])
    np.testing.assert_allclose(values, [-5, 0, 5, 2])


def test_fixed_ownership_uses_camera_with_more_edge_margin():
    h = np.array([np.eye(3), [[1, 0, 5], [0, 1, 0], [0, 0, 1]]])
    assert owner_camera(np.array([[4., 5.], [9., 5.]]), h, width=10, height=10).tolist() == [0, 1]


def test_raw_reader_handles_unsigned_slp_counts_and_ignores_zero_filled_tags(tmp_path):
    slp = tmp_path / "cam01_cam0_test_000.slp"
    frames = np.zeros(20, dtype=[("frame_id", "u8"), ("video", "u4"), ("frame_idx", "u8"),
                                 ("instance_id_start", "u8"), ("instance_id_end", "u8")])
    for col in ("frame_id", "frame_idx", "instance_id_start"):
        frames[col] = np.arange(20)
    frames["instance_id_end"] = np.arange(20)+1
    instances = np.zeros(20, dtype=[("instance_type", "u1"), ("point_id_start", "u8")])
    instances["instance_type"], instances["point_id_start"] = 1, np.arange(20)
    points = np.zeros(20, dtype=[("x", "f8"), ("y", "f8"), ("score", "f8")])
    points["x"], points["y"], points["score"] = 10., 10., 1.
    with h5py.File(slp, "w") as f:
        f["frames"], f["instances"], f["pred_points"] = frames, instances, points
    tags = np.zeros((20, 3, 2))
    tags[:, 1] = 10.
    with h5py.File(slp.with_name(slp.stem + "_aruco_tracks.h5"), "w") as f:
        f["aruco_tracks"] = tags
    path, audit = process_camera((str(slp), np.eye(3)[None], 1., 5, str(tmp_path), {}))
    rows = pd.read_parquet(path)
    assert len(rows) == 4
    assert rows["hit_0.5"].all()
    assert rows.tag_id.eq(1).all()
    assert audit["tag_reads"] == 4  # Not 12: absent tags are zero-filled.
