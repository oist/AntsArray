import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from tracking.single_ant.combine import offsets, prepare, worker
from tracking.single_ant.full_cup_report import SPEED
from tracking.single_ant.full_cup_tracking import TRACKING, TRACK_SCHEMA, save_parquet
from tracking.single_ant.manual_cup_tracking import write_json


def block(tmp_path, number, first_tag):
    out = tmp_path / f"block{number:02d}"
    out.mkdir()
    cups = [dict(cup_id=cid, name=f"cup_{cid}", center_mm=[25*(cid-1), 0], radius_mm=10)
            for cid in [1, 2]]
    write_json(out / "manifest.json", dict(output=str(out), cups=cups, fps=2, n_frames=4,
               source_block=str(tmp_path / "20261027" / out.name), tracking=TRACKING,
               start_datetime=f"2026-10-27T09:00:{(number-1)*5:02d}+09:00"))
    (out / "manual_regions.csv").write_text("source labels\n")
    write_json(out / "panorama_metadata.json", {})
    pd.DataFrame([dict(CupID=cid, ArucoID=tag, Eligible=True) for cid, tag in [(1, first_tag), (2, 7)]]).to_csv(out / "cup_identities.csv", index=False)
    for cid, tag in [(1, first_tag), (2, 7)]:
        name = f"TrackID_{cid:04d}_all_single.parquet"
        rows = []
        for frame in range(4):
            row = {f.name: 0 for f in TRACK_SCHEMA}
            row.update(Frame=frame, TrackID=cid, CupID=cid, ArucoID=tag, PositionSource="aruco_sleap")
            for col in ["X", "TrackX", "ArucoX", "SleapAnchorX"]:
                row[col] = 25*(cid-1)/.016
            rows.append(row)
        save_parquet(out / "per_track" / name, pd.DataFrame(rows), metadata=dict(num_frames=4), schema=TRACK_SCHEMA)
        cache = out / "analysis/speed_vectors/per_track" / name.removesuffix(".parquet")
        cache.mkdir(parents=True)
        np.save(cache / "speed_mm_s.npy", np.arange(4, dtype=np.float32) + 10*number+cid)
        np.save(cache / "speed_unclipped_mm_s.npy", np.arange(4, dtype=np.float32) + 100*number+cid)
        write_json(cache / "speed_metadata.json", dict(track_name=name, fps=2, mm_per_px=.016,
                   x_col="TrackX", y_col="TrackY", bodypoint_filter=0, frame_min=0, frame_max=3, **SPEED))
    return out / "manifest.json"


def test_real_gaps_cached_speeds_repeated_ids_and_changed_resident(tmp_path):
    manifests = [block(tmp_path, 1, 7), block(tmp_path, 2, 8)]
    path = prepare(manifests, tmp_path / "combined", tmp_path / "published")
    m = json.loads(path.read_text())
    assert m["n_frames"] == 14 and m["recorded_frames"] == 8
    assert m["recording_segments"][1]["gap_before_frames"] == 6
    assert {(a["cup_id"], a["raw_id"]): a["track_id"] for a in m["ants"]} == {(1, 7): 1, (1, 8): 3, (2, 7): 2}
    for index, ant in enumerate(m["ants"]):
        worker(m, index)
        out = tmp_path / "combined"
        track = out / "per_track" / ant["name"]
        data = pd.read_parquet(track)
        assert set(data.TrackID) == {ant["track_id"]}
        assert set(data.CupID) == {ant["cup_id"]}
        assert set(data.ArucoID) == {ant["raw_id"]}
        assert pq.ParquetFile(track).schema_arrow.metadata[b"num_frames"] == b"14"
        cache = out / "analysis/speed_vectors/per_track" / track.stem
        for filename in ["speed_mm_s.npy", "speed_unclipped_mm_s.npy"]:
            merged = np.load(cache / filename)
            assert merged.shape == (14,) and np.isnan(merged[4:10]).all()
            for s in ant["sources"]:
                original = np.load(manifests[int(s["block"][-2:])-1].parent / "analysis/speed_vectors/per_track" / f"TrackID_{ant['cup_id']:04d}_all_single" / filename)
                np.testing.assert_array_equal(merged[s["frame_offset"]:s["frame_offset"]+4], original)
        metrics = json.loads((out / "diagnostics" / m["cups"][index]["name"] / "metrics.json").read_text())
        assert metrics["OutsideCupFrames"] == 0
        assert metrics["ObservedFrames"] == 4*len(ant["sources"])
        assert data.Frame.eq(data.block_frame + data.source_block.map({"block01": 0, "block02": 10})).all()


@pytest.mark.parametrize("change", ["overlap", "fps"])
def test_rejects_ambiguous_block_timeline(change):
    first = dict(fps=2, n_frames=4, start_datetime="2026-10-27T09:00:00+09:00", source_block="block01")
    second = dict(first, source_block="block02", start_datetime="2026-10-27T09:00:01+09:00")
    if change == "fps":
        second["fps"] = 3
    with pytest.raises(ValueError):
        offsets([first, second])


def test_cross_date_repeated_block01_names_and_chronological_sort(tmp_path):
    first, second = block(tmp_path, 1, 7), block(tmp_path, 2, 8)
    later = tmp_path / "later"; later.mkdir()
    third = block(later, 1, 8)
    data = json.loads(third.read_text())
    data.update(source_block=str(tmp_path / "20261028/block01"), start_datetime="2026-10-28T09:00:00+09:00")
    write_json(third, data)
    path = prepare([third, second, first], tmp_path / "combined", tmp_path / "published")
    m = json.loads(path.read_text())
    labels = ["20261027_block01", "20261027_block02", "20261028_block01"]
    assert [s["block"] for s in m["recording_segments"]] == labels
    assert [s["frame_offset"] for s in m["recording_segments"]] == [0, 10, 172800]
    index = next(i for i, ant in enumerate(m["ants"]) if ant["cup_id"] == 2)
    worker(m, index)
    ant = m["ants"][index]
    track = tmp_path / "combined/per_track" / ant["name"]
    d = pd.read_parquet(track)
    assert set(d.source_block) == set(labels)
    shift = dict(zip(labels, [0, 10, 172800]))
    assert d.Frame.eq(d.block_frame+d.source_block.map(shift)).all()
    speed = np.load(tmp_path / "combined/analysis/speed_vectors/per_track" / track.stem / "speed_mm_s.npy")
    assert speed.shape == (172804,) and np.isnan(speed[14:172800]).all()
