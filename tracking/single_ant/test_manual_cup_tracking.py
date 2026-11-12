import numpy as np
import pandas as pd

from tracking.single_ant.manual_cup_tracking import (
    Settings, assign_cups, census, deduplicate_poses, export_tracks, track_cup, validate_timing, view_groups,
)


def cup(cid=1, x=0):
    return dict(cup_id=cid, name=f"cup_{cid}", center_mm=[x, 0], radius_mm=10.)


def tags(rows):
    return pd.DataFrame(rows, columns=["Frame", "CupID", "Instance", "X", "Y", "Cam"])


def poses(rows):
    p = pd.DataFrame(rows, columns=["Frame", "CupID", "PoseKey", "X", "Y", "Cam"])
    p["Instance"] = p.PoseKey
    p["NNodes"] = 10
    p["MedianScore"] = .9
    p["Strong"] = True
    return p


def test_overlap_views_do_not_add_ants_but_two_same_camera_same_id_do():
    assert len(view_groups(np.array([[0, 0], [.2, 0]]), [0, 1], 2)) == 1
    assert len(view_groups(np.array([[0, 0], [.2, 0]]), [0, 0], 2)) == 2
    assert len(view_groups(np.array([[0, 0], [4, 0]]), [0, 1], 2)) == 2
    # Do not merge a spatial chain through an intermediate camera.
    assert len(view_groups(np.array([[0, 0], [1.5, 0], [3, 0]]), [0, 1, 2], 2)) == 2


def test_same_camera_near_identical_predictions_are_one_pose_not_two_ants():
    p = poses([(0, 1, 0, 0, 0, 0), (0, 1, 1, .1, 0, 0), (0, 1, 2, .2, 0, 0)])
    n = pd.DataFrame([dict(PoseKey=key, Bodypoint=bp, X=bp + offset, Y=0)
                      for key, offset in [(0, 0), (1, .1), (2, 2)] for bp in range(10)])
    result, aliases = deduplicate_poses(p, n, Settings())
    assert set(result.PoseKey) == {0, 2}
    assert aliases.RemovedPoseKey.tolist() == [1]
    assert aliases.SharedNodes.iloc[0] == 10


def test_strict_threshold_all_frames_and_repeated_ids_in_different_cups():
    a = tags([(i, c, 7, x, 0, 0) for c, x in [(1, 0), (2, 25)] for i in range(7)])
    p = poses([(i, c, 100*c+i, x, 0, 0) for c, x in [(1, 0), (2, 25)] for i in range(7)])
    q, _ = census([cup(), cup(2, 25)], a, p, 0, 10, Settings())
    assert q.Eligible.all()
    assert q.ArucoID.tolist() == [7, 7]
    assert np.allclose(q.SingleIDFraction, .7)
    q, _ = census([cup()], a.loc[a.Frame.lt(6) & a.CupID.eq(1)], p, 0, 10, Settings())
    assert not q.Eligible.iloc[0]  # exactly 60% is insufficient


def test_empty_and_multiple_ant_cups_cannot_be_tracked():
    a = tags([(i, 1, 7, 0, 0, 0) for i in range(10)] + [(2, 1, 7, 5, 0, 0)])
    q, _ = census([cup(), cup(2, 25)], a, poses([]), 0, 10, Settings(multiplicity_policy="raw"))
    assert not q.Eligible.any()
    assert "multiple_aruco_instances" in q.Reason.iloc[0]
    assert "no_aruco_ant_detected" in q.Reason.iloc[1]


def test_rare_extra_decode_is_flagged_but_two_supported_ants_are_rejected():
    settings = Settings(multi_hits=3, multi_window=5)
    a = tags([(i, 1, 7, 0, 0, 0) for i in range(10)] + [(2, 1, 99, 5, 0, 0)])
    p = poses([(i, 1, i, 0, 0, 0) for i in range(10)])
    q, c = census([cup()], a, p, 0, 10, Settings())
    assert q.Eligible.iloc[0]
    t = track_cup(cup(), 7, a, p, c, 0, 10, Settings())
    assert t.ExtraTagRead.iloc[2]
    assert t.PositionSource.iloc[2] == "aruco_sleap"
    # Persistent second ID with its own pose is a two-ant cup.
    a = pd.concat([a, tags([(i, 1, 99, 5, 0, 0) for i in [1, 3, 5]])], ignore_index=True)
    p = pd.concat([p, poses([(i, 1, 20 + i, 5, 0, 0) for i in [1, 3, 5]])], ignore_index=True)
    q, _ = census([cup()], a, p, 0, 10, settings)
    assert q.FlaggedMultiAnt.iloc[0]
    assert q.StopFrame.iloc[0] == 1
    assert q.MultipleSupportedArucoFrames.iloc[0] == 3
    t = track_cup(cup(), 7, a, p, census([cup()], a, p, 0, 10, settings)[1], 0, 10, settings)
    assert t.TrackX.iloc[:1].notna().all()
    assert t.TrackX.iloc[1:].isna().all()
    assert t.PositionSource.iloc[1:].eq("stopped_multi_ant").all()


def test_supported_sleap_multiple_instances_reject_cup():
    a = tags([(i, 1, 7, 0, 0, 0) for i in range(10)])
    p = poses([(i, 1, i, 0, 0, 0) for i in range(10)]
              + [(i, 1, 20 + i, 5, 0, 0) for i in [1, 3, 5]])
    q, _ = census([cup()], a, p, 0, 10, Settings(multi_hits=3, multi_window=5))
    assert q.FlaggedMultiAnt.iloc[0]
    assert q.StopFrame.iloc[0] == 1
    assert "multiple_supported_sleap_instances" in q.EventReason.iloc[0]


def test_unsupported_foreign_ids_do_not_block_bridging_and_persistent_supported_ids_stop():
    a = tags([(0, 1, 7, 0, 0, 0)] + [(i, 1, 99, 8, 0, 0) for i in range(1, 10)])
    p = poses([(i, 1, i, 0, 0, 0) for i in range(10)])
    q, c = census([cup()], a, p, 0, 10, Settings())
    t = track_cup(cup(), 7, a, p, c, 0, 10, Settings())
    assert not q.FlaggedMultiAnt.iloc[0]
    assert t.PositionSource.iloc[1:].eq("sleap_bridge").all()
    # A different ID with its own skeleton persists while resident ID is absent.
    a = tags([(i, 1, 7, 0, 0, 0) for i in range(7)] + [(i, 1, 99, 0, 0, 0) for i in range(7, 10)])
    q, c = census([cup()], a, p, 0, 10, Settings(multi_hits=3, multi_window=5))
    assert q.FlaggedMultiAnt.iloc[0]
    assert q.StopFrame.iloc[0] == 7
    assert "foreign_ID" in q.EventReason.iloc[0]


def test_persistence_default_requires_fifteen_supported_frames_in_one_second():
    p = poses([(i, 1, i, 0, 0, 0) for i in range(60)])
    a = tags([(i, 1, 7, 0, 0, 0) for i in range(40)] + [(i, 1, 99, 0, 0, 0) for i in range(40, 60)])
    q, counts = census([cup()], a, p, 0, 60, Settings())
    assert q.Eligible.iloc[0]
    assert q.StopFrame.iloc[0] == 40
    t = track_cup(cup(), 7, a, p, counts, 0, 60, Settings())
    assert t.TrackX.iloc[40:].isna().all()


def test_nearest_sleap_and_long_gap_with_true_missing_frames_and_wrong_id():
    a = tags([(0, 1, 7, 0, 0, 0), (200, 1, 7, 1, 0, 0), (100, 1, 8, 1, 0, 0)])
    p = poses([(i, 1, i, i / 200, 0, 0) for i in range(201) if i != 80]
              + [(0, 1, 1000, .8, 0, 1)])
    # Census here is only the per-frame observation evidence; the test fixes
    # the already-qualified identity explicitly to exercise a long natural gap.
    _, c = census([cup()], a, p, 0, 201, Settings())
    t = track_cup(cup(), 7, a, p, c, 0, 201, Settings())
    assert t.PoseKey.iloc[0] == 0  # closest, not arbitrary pose
    assert t.PositionSource.iloc[150] == "sleap_bridge"  # >120 frames without a tag
    assert np.isclose(t.TrackX.iloc[150], .75)
    assert np.isnan(t.TrackX.iloc[80])  # no fabricated coordinates
    assert t.PositionSource.iloc[100] == "conflicting_tag"
    assert t.PositionSource.iloc[200] == "aruco_sleap"


def test_cup_containment_no_neighbor_switch_and_ambiguous_pose_not_bridged():
    cups = [cup(), cup(2, 25)]
    assert assign_cups([[0, 0], [25, 0], [12, 0]], cups).tolist() == [1, 2, -1]
    assert assign_cups([[7, 0]], [cup(), cup(2, 15)]).tolist() == [-1]
    a = tags([(0, 1, 7, 0, 0, 0)])
    p = poses([(0, 1, 0, 0, 0, 0), (1, 1, 1, 1, 0, 0),
               (1, 1, 2, 5, 0, 0), (2, 2, 3, 25, 0, 0)])
    _, c = census(cups, a, p, 0, 3, Settings())
    t = track_cup(cup(), 7, a, p, c, 0, 3, Settings())
    assert t.TrackX.iloc[0] == 0
    assert not np.isfinite(t.TrackX.iloc[1:]).any()


def test_hidden_tags_do_not_leak_positions_into_gap_validation():
    a = tags([(i, 1, 7, 1, 0, 0) for i in range(10)])
    p = poses([(i, 1, i, 0, 0, 0) for i in range(10)])
    _, counts = census([cup()], a, p, 0, 10, Settings())
    mask = np.zeros(10, bool)
    mask[2:8] = True
    t = track_cup(cup(), 7, a, p, counts, 0, 10, Settings(), mask_tags=mask)
    assert t.loc[mask, "ArucoX"].isna().all()
    assert t.loc[mask, "TrackX"].eq(0).all()  # posture, not withheld tag X=1
    assert t.loc[mask, "PositionSource"].eq("sleap_bridge").all()


def test_export_keeps_actual_nodes_and_colony_units(tmp_path):
    a = tags([(0, 1, 7, 1, 0, 0)])
    p = poses([(0, 1, 1, 1, 0, 0), (1, 1, 2, 2, 0, 0)])
    _, c = census([cup()], a, p, 0, 2, Settings())
    t = track_cup(cup(), 7, a, p, c, 0, 2, Settings())
    nodes = pd.DataFrame(dict(PoseKey=[1, 1, 2, 2], Bodypoint=[0, 1, 0, 1],
                              X=[1., 1.1, 2., 2.1], Y=[0.] * 4, Score_node=[.9] * 4))
    path = tmp_path / "track.parquet"
    export_tracks(t, nodes, path)
    d = pd.read_parquet(path)
    assert len(d) == 4
    assert d.TrackID.unique().tolist() == [1]
    assert np.isclose(d.loc[d.Frame.eq(1), "TrackX"].iloc[0], 2 / .016)
    assert set(["Frame", "TrackID", "Bodypoint", "X", "Y", "TrackX", "ArucoX", "SleapAnchorX", "CameraID"]) <= set(d)


def test_timing_uses_trigger_frames_and_rejects_arming_offsets():
    m = dict(n_frames=20, synchronization=dict(method="common hardware trigger/block IDs",
             diagnostics={"cam01": dict(first_block_id=0, last_block_id=19, frames=20)}),
             cameras={"cam01": dict(offset=0, frames=20)},
             files=[dict(camera="cam01", index=0, frame_start=0, local_start=0, frames=10),
                    dict(camera="cam01", index=1, frame_start=10, local_start=10, frames=10)])
    validate_timing(m)
    m["cameras"]["cam01"]["offset"] = 24
    import pytest
    with pytest.raises(ValueError, match="synchronization"):
        validate_timing(m)
