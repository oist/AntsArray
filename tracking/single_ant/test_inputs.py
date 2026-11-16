import hashlib
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from tracking.single_ant.inputs import detection_manifest


def exports(tmp_path):
    root = tmp_path / "single_ants"
    labels = root / "20261026/block03"
    labels.mkdir(parents=True)
    calibration = tmp_path / "H.npz"
    np.savez(calibration, H=np.stack([np.eye(3)] * 2))
    (labels / "panorama_from_hmats_metadata.json").write_text(json.dumps(
        dict(mm_per_pixel=1 / 11, homographies=str(calibration))))
    pd.DataFrame([dict(name="cup_1", semantic_label="cup", shape="circle",
                      mm_per_pixel=1 / 11, tracking_center_x_px=0,
                      tracking_center_y_px=0, radius_px=100)]).to_csv(labels / "panorama_regions.csv", index=False)
    block = root / "20261027/block01"
    (block / "data").mkdir(parents=True)
    (block / "sess_start.txt").write_text("2026-10-27 09:18:44.008\tTX\tstart 24\n")
    for cam, lengths in [(1, [2, 3]), (2, [3, 2])]:
        # Different camera arming times must never become frame offsets.
        prefix = f"cam{cam:02d}_cam{cam-1}_2026-10-27-09-18-{30+cam:02d}"
        diag = dict(provenance=dict(firstEncodedFrame=dict(blockId=0, fileFrameIndex=0),
                                   lastEncodedFrame=dict(blockId=4, fileFrameIndex=4)),
                    recorder=dict(framesEncoded=5), context=dict(fps=24))
        (block / (prefix + ".mkv.diag.json")).write_text(json.dumps(diag))
        for i, length in enumerate(lengths):
            with h5py.File(block / "data" / f"{prefix}_{i:03d}_aruco_tracks_.h5", "w") as h:
                h.attrs["num_frames"] = length
                h.create_dataset("aruco_detections", (0,), dtype=[(n, "i8") for n in ["Frame", "Instance", "X", "Y"]])
            with h5py.File(block / "data" / f"{prefix}_{i:03d}_sleap_data.h5", "w") as h:
                h.attrs["frame_count"] = length
                h.create_dataset("sleap_data", (0,), dtype=[(n, "i8") for n in ["Frame", "Instance", "Bodypoint", "X", "Y", "Score_node"]])
    return block, labels


def test_latest_cup_labels_and_actual_camera_chunk_spans(tmp_path):
    block, labels = exports(tmp_path)
    m, paths = detection_manifest(block)
    assert paths[0] == labels / "panorama_regions.csv"
    assert m["origin"] == "2026-10-27T09:18:44.008000+09:00"
    assert m["n_frames"] == 5
    assert [f["frame_start"] for f in m["files"]] == [0, 2, 0, 3]
    assert all(cam["offset"] == 0 for cam in m["cameras"].values())


@pytest.mark.parametrize("problem", ["gap", "missing_pose", "truncated_pose"])
def test_refuses_unsynchronized_or_incomplete_exports(tmp_path, problem):
    block, _ = exports(tmp_path)
    if problem == "gap":
        path = next(block.glob("*.diag.json"))
        d = json.loads(path.read_text())
        d["provenance"]["gaps"] = [dict(firstMissingBlockId=2)]
        path.write_text(json.dumps(d))
    else:
        path = next((block / "data").glob("*_sleap_data.h5"))
        if problem == "missing_pose":
            path.unlink()
        else:
            with h5py.File(path, "r+") as h:
                h.attrs["frame_count"] = 1
    with pytest.raises((ValueError, FileNotFoundError)):
        detection_manifest(block)


def test_interrupted_camera_requires_acknowledgment_and_preserves_its_short_span(tmp_path):
    block, _ = exports(tmp_path)
    diag = next(block.glob("cam02*.diag.json"))
    diag.write_text(json.dumps(dict(synthesized=dict(why="recorder crashed"),
                                   context=dict(fps=24, status="interrupted"),
                                   recorder=dict(framesEncoded=3))))
    for path in (block / "data").glob("cam02*_001*"):
        path.unlink()
    aruco = next((block / "data").glob("cam02*_aruco_tracks*.h5"))
    with h5py.File(aruco, "r+") as h:
        h.attrs["num_frames"] = 4
    with pytest.raises(ValueError):
        detection_manifest(block)
    m, _ = detection_manifest(block, interrupted_cameras=["cam02"])
    assert m["n_frames"] == 5
    assert m["cameras"]["cam02"]["frames"] == 3
    assert m["synchronization"]["alignment_assumed_cameras"] == ["cam02"]
    assert m["synchronization"]["diagnostics"]["cam02"]["first_block_id"] is None
    tail = next(f for f in m["files"] if f["camera"] == "cam02")
    assert tail["frames"] == 3 and tail["aruco_frames"] == 4


def realign(block, rid="r1", chunks=(0, 1), record_gaps=None):
    """cam02 lost trigger pulse 2: its sidecar records the gap, and its exports were re-cut
    onto the pulse grid (tracking/colony/realign_offgrid_camera.py --sidecar-gaps) and tagged."""
    diag = next(block.glob("cam02*.diag.json"))
    prefix = diag.name[:-len(".mkv.diag.json")]
    gap = dict(blockId=3, previousBlockId=1, blocksMissing=1, fileFrameIndex=2,
               discontinuity="forward_gap")
    d = json.loads(diag.read_text())
    d["recorder"]["framesEncoded"] = 4
    d["provenance"].update(gaps=[gap], lastEncodedFrame=dict(blockId=4, fileFrameIndex=3))
    diag.write_text(json.dumps(d))
    for i in chunks:
        for path in (block / "data").glob(f"{prefix}_{i:03d}_*.h5"):
            with h5py.File(path, "r+") as h:
                h.attrs["realign_id"] = rid
    record = dict(schema=1, kind="in_place", id=rid, installed="2026-11-16T00:00:00Z",
                  videos={prefix: dict(camera="cam02",
                                       sidecar_gaps=[gap] if record_gaps is None else record_gaps,
                                       sidecar_sha256=hashlib.sha256(diag.read_bytes()).hexdigest(),
                                       unrecorded_frames=[[2, 3]],
                                       video_frame_map=dict(entries=[[0, 0], [3, 1]]),
                                       chunks=[dict(chunk=i) for i in chunks])})
    (block / "REALIGN.json").write_text(json.dumps(record))
    return prefix


def test_lost_pulses_are_accepted_only_through_an_installed_realignment(tmp_path):
    block, _ = exports(tmp_path)
    realign(block)
    record = block / "REALIGN.json"
    kept = record.read_text()
    record.unlink()
    with pytest.raises(ValueError, match="realignment"):
        detection_manifest(block)
    record.write_text(kept)
    m, _ = detection_manifest(block)
    assert m["n_frames"] == 5
    diag = m["synchronization"]["diagnostics"]["cam02"]
    assert diag["frames"] == 5 and diag["last_block_id"] == 4 and diag["recorded_frames"] == 4
    assert diag["realigned"] is True and diag["unrecorded_frames"] == [[2, 3]]
    assert m["synchronization"]["realigned_cameras"] == ["cam02"]
    assert m["synchronization"]["realignment"]["unrecorded_frames"] == {"cam02": [[2, 3]]}
    assert m["synchronization"]["realignment"]["video_frame_map"] == {"cam02": [[0, 0], [3, 1]]}
    assert "realigned" not in m["synchronization"]["diagnostics"]["cam01"]


def test_clean_blocks_report_no_realignment(tmp_path):
    block, _ = exports(tmp_path)
    m, _ = detection_manifest(block)
    assert m["synchronization"]["realigned_cameras"] == []
    assert m["synchronization"]["realignment"] is None


@pytest.mark.parametrize("problem", ["gaps_differ", "redetected_chunk", "sidecar_changed",
                                     "not_installed", "stray_camera"])
def test_refuses_a_realignment_that_does_not_match_the_exports(tmp_path, problem):
    block, _ = exports(tmp_path)
    if problem == "gaps_differ":
        realign(block, record_gaps=[dict(blockId=4, previousBlockId=1, blocksMissing=2,
                                         fileFrameIndex=2, discontinuity="forward_gap")])
    else:
        prefix = realign(block)
    record = block / "REALIGN.json"
    if problem == "redetected_chunk":
        with h5py.File(next((block / "data").glob(f"{prefix}_001_sleap_data.h5")), "r+") as h:
            del h.attrs["realign_id"]
    elif problem == "sidecar_changed":
        diag = next(block.glob("cam02*.diag.json"))
        d = json.loads(diag.read_text())
        d["recorder"]["note"] = "edited"
        diag.write_text(json.dumps(d))
    elif problem == "not_installed":
        r = json.loads(record.read_text())
        del r["installed"]
        record.write_text(json.dumps(r))
    elif problem == "stray_camera":
        r = json.loads(record.read_text())
        cam01 = next(block.glob("cam01*.diag.json")).name[:-len(".mkv.diag.json")]
        r["videos"][cam01] = dict(r["videos"][prefix], camera="cam01")
        record.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        detection_manifest(block)


def test_interrupted_mode_does_not_accept_normal_sidecars_or_internal_stream_gaps(tmp_path):
    block, _ = exports(tmp_path)
    with pytest.raises(ValueError):
        detection_manifest(block, interrupted_cameras=["cam02"])
    diag = next(block.glob("cam02*.diag.json"))
    diag.write_text(json.dumps(dict(synthesized=dict(why="crash"),
                                   context=dict(fps=24, status="interrupted"),
                                   recorder=dict(framesEncoded=5))))
    aruco = next((block / "data").glob("cam02*_000_aruco_tracks*.h5"))
    with h5py.File(aruco, "r+") as h:
        h.attrs["num_frames"] = 4
    with pytest.raises(ValueError, match="terminal chunk"):
        detection_manifest(block, interrupted_cameras=["cam02"])
