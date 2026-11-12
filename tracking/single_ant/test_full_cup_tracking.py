from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from tracking.single_ant.full_cup_tracking import (
    identify, native_cup_track, resident_registry, save_parquet, stitch_cup, TRACK_SCHEMA,
)


def detections(frames=40, second_ant=True):
    tags, nodes = [], []
    for frame in range(frames):
        # Resident tag is briefly absent, and returns after an extra tag has
        # persisted long enough to have stopped the previous pilot.
        if not 10 <= frame < 14:
            tags.append((frame, 7, 0., 0., 0))
        if second_ant and frame >= 15:
            tags.append((frame, 8, 5., 0., 0))
        for inst, x in [(100, 0.)] + ([(200, 5.)] if second_ant and frame >= 15 else []):
            for bp in range(10):
                nodes.append((frame, inst, bp, x + bp*.01, 0., 0))
    a = pd.DataFrame(tags, columns=["Frame", "Instance", "X", "Y", "Cam"])
    s = pd.DataFrame(nodes, columns=["Frame", "Instance", "Bodypoint", "X", "Y", "Cam"])
    return a, s


def test_other_persistent_id_never_censors_resident_and_gap_bridges(tmp_path):
    a, s = detections()
    d = native_cup_track(a, s, cup_id=28, dedicated_id=7, frame_start=0,
                         frame_stop=40, output=tmp_path / "chunk000.parquet")
    anchor = d[d.Bodypoint.eq(0)].set_index("Frame")
    assert len(anchor) == 40
    assert anchor.TrackID.eq(28).all() and anchor.ArucoID.eq(7).all()
    assert anchor.PositionSource.loc[10:13].eq("sleap_bridge").all()
    assert anchor.TrackX.loc[30:].eq(0).all()
    # Competing tag 8 gets its own posture inside the unchanged native engine.
    native = pd.read_parquet(tmp_path / "chunk000_native_all_ids.parquet")
    assert set(native[native.Frame.ge(30)].TrackID) == {7, 8}


def test_same_tag_in_two_cups_has_distinct_identity_and_global_frames(tmp_path):
    a, s = detections(10, False)
    for cid in [1, 2]:
        aa, ss = a.copy(), s.copy()
        aa.Frame += 100; ss.Frame += 100
        aa.X += cid*30; ss.X += cid*30
        d = native_cup_track(aa, ss, cup_id=cid, dedicated_id=7, frame_start=100,
                             frame_stop=110, output=tmp_path / f"cup{cid}.parquet")
        assert d.Frame.min() == 100 and d.Frame.max() == 109
        assert set(d.TrackID) == {cid} and set(d.ArucoID) == {7}
        assert np.allclose(d.TrackX * .016, cid*30)


def test_empty_chunks_stitch_with_exact_global_timeline(tmp_path):
    a, s = detections(5, False)
    first = tmp_path / "tracked_chunks/cup001/chunk000.parquet"
    second = first.with_name("chunk001.parquet")
    native_cup_track(a.iloc[:0], s.iloc[:0], cup_id=1, dedicated_id=7,
                     frame_start=0, frame_stop=5, output=first)
    aa, ss = a.copy(), s.copy(); aa.Frame += 5; ss.Frame += 5
    native_cup_track(aa, ss, cup_id=1, dedicated_id=7, frame_start=5,
                     frame_stop=10, output=second)
    for path in [first, second]:
        path.with_suffix(".json").write_text('{}')
        assert pq.ParquetFile(path).schema_arrow.equals(TRACK_SCHEMA, check_metadata=False)
    pd.DataFrame([dict(CupID=1, Eligible=True)]).to_csv(tmp_path / "cup_identities.csv", index=False)
    m = dict(output=str(tmp_path), cups=[dict(cup_id=1)], chunks=[dict(index=0), dict(index=1)], n_frames=10)
    stitch_cup(m, 0)
    d = pd.read_parquet(tmp_path / "per_track/TrackID_0001_all_single.parquet")
    assert d.Frame.unique().tolist() == list(range(5, 10))
    assert pq.ParquetFile(tmp_path / "per_track/TrackID_0001_all_single.parquet").schema_arrow.metadata[b"num_frames"] == b"10"


def test_established_identity_survives_later_tag_loss_and_new_cup_can_qualify(tmp_path):
    folder = tmp_path / "mapped/chunk000"; folder.mkdir(parents=True)
    (folder / "COMPLETE.json").write_text('{}')
    pd.DataFrame([
        dict(CupID=1, ArucoID=7, ObservedFrames=20, SingleInstanceFrames=20, SupportedFrames=20, MultipleInstanceFrames=0),
        dict(CupID=1, ArucoID=8, ObservedFrames=80, SingleInstanceFrames=80, SupportedFrames=80, MultipleInstanceFrames=0),
        dict(CupID=2, ArucoID=7, ObservedFrames=80, SingleInstanceFrames=80, SupportedFrames=80, MultipleInstanceFrames=0),
    ]).to_csv(folder / "id_counts.csv", index=False)
    m = dict(output=str(tmp_path), cups=[dict(cup_id=1, name="cup_1"), dict(cup_id=2, name="cup_2")],
             chunks=[dict(index=0)], initial_identities={"1": 7}, n_frames=100)
    identify(m)
    q = pd.read_csv(tmp_path / "cup_identities.csv")
    assert q.Eligible.all() and q.ArucoID.tolist() == [7, 7]


def test_documented_sparse_id_requires_pose_support_and_untagged_stays_unknown(tmp_path):
    folder = tmp_path / "mapped/chunk000"; folder.mkdir(parents=True)
    (folder / "COMPLETE.json").write_text('{}')
    pd.DataFrame([
        dict(CupID=1, ArucoID=38, ObservedFrames=5, SingleInstanceFrames=5, SupportedFrames=4, MultipleInstanceFrames=0),
        dict(CupID=2, ArucoID=49, ObservedFrames=90, SingleInstanceFrames=90, SupportedFrames=80, MultipleInstanceFrames=0),
        dict(CupID=3, ArucoID=40, ObservedFrames=5, SingleInstanceFrames=5, SupportedFrames=0, MultipleInstanceFrames=0),
    ]).to_csv(folder / "id_counts.csv", index=False)
    m = dict(output=str(tmp_path), cups=[dict(cup_id=i, name=f"cup_{i}") for i in [1,2,3]],
             chunks=[dict(index=0)], initial_identities={}, n_frames=100,
             documented_identities={"1":38,"2":None,"3":40})
    identify(m)
    q = pd.read_csv(tmp_path / "cup_identities.csv")
    assert q.ArucoID.tolist() == [38, -1, -1]
    assert q.Eligible.tolist() == [True, False, False]
    assert q.SingleIDFractionFull.iloc[0] == .05  # Never inflate observed evidence.
    assert q.Reason.iloc[1] == "documented_untagged_cup"
    assert q.Reason.iloc[2] == "documented_ID_without_tag_pose_support"


def test_resident_registry_allows_repeated_tags_but_requires_exact_cup_roster(tmp_path):
    path = tmp_path / "registry.tsv"
    path.write_text("cup\ttag_id\n1\t38\n2\t38\n3\t\n")
    cups = [dict(cup_id=i) for i in [1,2,3]]
    assert resident_registry(path, cups) == {"1":38,"2":38,"3":None}
    path.write_text("cup\ttag_id\n1\t38\n1\t40\n3\t\n")
    with pytest.raises(ValueError, match="unique integers"):
        resident_registry(path, cups)
