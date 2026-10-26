"""Scientific correctness regressions: one skeleton, real time, no silent gaps."""
import json
import struct
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import pytest

from tracking.single_ant.workflow import (
    Settings, add_time, coverage_intervals, finalize, prepare, probe_video, read_detections,
    select_skeletons, stitch_camera, track_recording,
)


def detections(items):
    rows = []
    for frame, instance, x, y, score in items:
        rows.extend((frame, instance, bp, x + bp, y, score) for bp in range(3))
    return pd.DataFrame(rows, columns=["Frame", "Instance", "Bodypoint", "X", "Y", "Score_node"]).astype(
        {"Frame": "int32", "Instance": "int32", "Bodypoint": "int32", "X": "float32", "Y": "float32", "Score_node": "float32"})


def test_whole_skeleton_mask_and_continuity():
    mask = np.zeros((100, 100), bool)
    mask[10:90, 10:90] = True
    df = detections([(0, 0, 20, 20, .9), (0, 1, 0, 0, 1), (1, 2, 21, 21, .7),
                     (1, 3, 60, 60, 1), (2, 4, 22, 22, .1)])
    out, prev = select_skeletons(df, mask, offset=0, fps=24, settings=Settings(max_distance=10))
    assert out.Instance.unique().tolist() == [0, 2]
    assert out.groupby("Frame").Instance.nunique().max() == 1
    assert out.groupby("Frame").size().tolist() == [3, 3]
    assert prev == (1, 22., 21.)


def test_state_survives_chunk_boundary_but_expires_after_gap():
    mask = np.ones((100, 100), bool)
    df = detections([(0, 4, 25, 25, .6), (0, 9, 70, 70, 1)])
    kwargs = dict(fps=24, settings=Settings(max_distance=10), previous=(9, 25, 25))
    out, _ = select_skeletons(df, mask, offset=10, **kwargs)
    assert out.Instance.unique().tolist() == [4]
    out, _ = select_skeletons(df, mask, offset=200, **kwargs)
    assert out.Instance.unique().tolist() == [9]


def test_missing_head_falls_back_and_does_not_mix_nodes():
    df = detections([(0, 2, 30, 30, .9)])
    df.loc[df.Bodypoint == 1, ["X", "Y"]] = np.nan
    out, _ = select_skeletons(df, np.ones((100, 100), bool), offset=0, fps=24, settings=Settings())
    assert out.AnchorBodypoint.unique().tolist() == [-1]
    assert out.TrackX.unique().tolist() == [31]
    assert out.NodeValid.tolist() == [True, False, True]


def test_complete_pose_beats_two_spurious_high_confidence_nodes():
    good = detections([(0, 1, 20, 20, .8)])
    false = detections([(0, 2, 70, 70, 1)])
    false.loc[false.Bodypoint == 2, ["X", "Y"]] = np.nan
    out, _ = select_skeletons(pd.concat([good, false]), np.ones((100, 100), bool),
                              offset=0, fps=24, settings=Settings())
    assert out.Instance.unique().tolist() == [1]


def test_empty_detections_and_full_missing_coverage():
    out, previous = select_skeletons(detections([]), np.ones((10, 10), bool), offset=0, fps=24, settings=Settings())
    assert out.empty and previous is None
    cov = coverage_intervals(np.array([], dtype=int), {"frame_start": 100, "session": "s", "camera": "cam01"},
                             {"frames": 5, "offset": 10, "index": 1})
    assert cov[["FrameStart", "FrameStop", "Observed"]].values.tolist() == [[110, 115, False]]


def test_time_uses_full_video_length_and_absolute_recording_start():
    rec = dict(start_ns=pd.Timestamp("2025-12-18T16:35:11+09:00").value, frame_start=24, fps=24,
               track=1, camera="cam01", session="s")
    df = detections([(0, 0, 10, 10, 1), (4, 1, 11, 11, 1)])
    result = add_time(df, rec, {"offset": 86500, "index": 1}, rec["start_ns"] - 1_000_000_000)
    assert result.Frame.unique().tolist() == [86524, 86528]
    assert result.FrameInRecording.unique().tolist() == [86500, 86504]
    assert str(result.Timestamp.dt.tz) == "UTC"
    assert result.ElapsedSeconds.iloc[0] == pytest.approx(1 + 86500 / 24)


def make_run(tmp_path):
    root, out = tmp_path / "input", tmp_path / "output"
    root.mkdir()
    mask = np.ones((100, 100, 3), np.uint8) * 255
    cv2.imwrite(str(root / "mask.png"), mask)
    from tracking.single_ant.workflow import digest
    records = []
    for i, start in enumerate((0, 100_000_000_000)):
        record = dict(key=f"session{i}_cam01", session=f"session{i}", camera="cam01", track=1,
                      start_ns=start, end_ns=start + 1_000_000_000, frame_start=i * 2400,
                      frames=24, fps=24, width=100, height=100, mask="mask.png",
                      mask_sha256=digest(root / "mask.png"), gap_before_seconds=None if not i else 99,
                      chunks=[dict(index=0, stem=f"video{i}_000", frames=12, offset=0),
                              dict(index=1, stem=f"video{i}_001", frames=12, offset=12)])
        records.append(record)
        directory = out / "detections" / record["session"]
        directory.mkdir(parents=True)
        for chunk in record["chunks"]:
            df = detections([(0, 0, 20, 20, .9), (4, 1, 21, 21, .9)]) if not chunk["index"] else detections([])
            with h5py.File(directory / f"{chunk['stem']}_sleap_data.h5", "w") as h:
                h.attrs["expected_frames"] = chunk["frames"]
                h.attrs["single_ant_contract"] = "test"
                h.create_dataset("sleap_data", data=df.to_records(index=False))
    return dict(root=str(root), output=str(out), contract_sha256="test", recordings=records,
                cameras=["cam01"], missing_cameras={}, settings=Settings().__dict__, origin_ns=0, fps=24)


def test_streamed_tracking_stitching_preserves_empty_tail_and_recording_gap(tmp_path):
    manifest = make_run(tmp_path)
    for recording in manifest["recordings"]:
        state = track_recording(manifest, recording["key"])
        assert state["selected_frames"] == 2 and state["video_frames"] == 24
    stitched = stitch_camera(manifest, "cam01")
    df = pd.read_parquet(stitched["output"])
    assert df.Frame.unique().tolist() == [0, 4, 2400, 2404]
    assert df.Track.unique().tolist() == [1]
    assert df.groupby(["Frame", "Bodypoint"]).size().max() == 1
    assert finalize(manifest)["chunks"] == 4
    assert (Path(manifest["output"]) / "COMPLETE.json").exists()


def test_stale_inference_and_out_of_range_frames_fail(tmp_path):
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as h:
        h.attrs["expected_frames"] = 10
        h.attrs["single_ant_contract"] = "old-model"
        h.create_dataset("sleap_data", data=detections([(10, 0, 1, 1, 1)]).to_records(index=False))
    with pytest.raises(ValueError, match="different inference"):
        read_detections(path, {"frames": 10}, "new-model")
    with pytest.raises(ValueError, match="outside video"):
        read_detections(path, {"frames": 10}, "old-model")


def test_missing_detection_file_cannot_mark_recording_complete(tmp_path):
    manifest = make_run(tmp_path)
    Path(manifest["output"], "detections/session0/video0_001_sleap_data.h5").unlink()
    with pytest.raises(FileNotFoundError):
        track_recording(manifest, "session0_cam01")
    assert not Path(manifest["output"], "state/session0_cam01.tracked.json").exists()


def test_opendml_uses_video_stream_count_not_first_riff_count(tmp_path):
    def chunk(kind, data):
        return kind + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")
    avih = struct.pack("<14I", 41666, 0, 0, 0, 80038, 0, 1, 0, 4024, 3036, 0, 0, 0, 0)
    strh = bytearray(56)
    strh[:4] = b"vids"
    struct.pack_into("<IIII", strh, 20, 1, 24, 0, 86500)
    hdr = chunk(b"LIST", b"hdrl" + chunk(b"avih", avih)
                + chunk(b"LIST", b"strl" + chunk(b"strh", strh))
                + chunk(b"LIST", b"odml" + chunk(b"dmlh", struct.pack("<I", 86500))))
    path = tmp_path / "chunk.avi"
    path.write_bytes(chunk(b"RIFF", b"AVI " + hdr))
    assert probe_video(path) == dict(frames=86500, fps=24, width=4024, height=3036)


def test_prepare_requires_no_aruco_and_uses_camera_timestamp(tmp_path, monkeypatch):
    from tracking.single_ant import workflow
    root = tmp_path / "input"
    data = root / "20251218-163512" / "data"
    data.mkdir(parents=True)
    masks = root / "arena_seg"
    masks.mkdir()
    cv2.imwrite(str(masks / "cam01_mask.png"), np.ones((10, 10, 3), np.uint8))
    stem = "cam01_cam0_2025-12-18-16-35-11"
    raw = data.parent / f"{stem}.avi"
    raw.touch()
    for i in range(2):
        (data / f"{stem}_{i:03d}.avi").touch()
    (data / f"{stem}_frame_counts.csv").write_text(f"{stem}_000.avi,3\n{stem}_001.avi,7\n")
    monkeypatch.setattr(workflow, "probe_video", lambda p: dict(frames=3 if "_000" in p.name else 7 if "_001" in p.name else 10,
                                                              fps=24, width=10, height=10))
    model = tmp_path / "model"
    model.mkdir()
    (model / "best.ckpt").write_bytes(b"model")
    kwargs = dict(model_centroid=model, model_instance=model, export_dir=model)
    manifest = prepare(root, tmp_path / "out", **kwargs)
    r = manifest["recordings"][0]
    assert r["start_time"] == "2025-12-18T16:35:11+09:00"
    assert r["chunks"][1]["offset"] == 3
    metadata = tmp_path / "export_metadata.json"
    metadata.write_text(json.dumps(dict(checkpoint_path=f"centroid:{model / 'best.ckpt'};centered_instance:{model / 'best.ckpt'}",
                                        peak_threshold=.2, centroid_peak_threshold=.01,
                                        instance_peak_threshold=.2, onnx_sha256="graph-hash")))
    tuned = prepare(root, tmp_path / "tuned", export_metadata=metadata, build_trt_if_missing=True, **kwargs)
    assert tuned["model"]["centroid_peak_threshold"] == .01
    assert tuned["model"]["onnx_sha256"] == "graph-hash"
    assert tuned["model"]["build_trt_if_missing"]
    (data / f"{stem}_001.avi").unlink()
    with pytest.raises(ValueError, match="Incomplete chunks"):
        prepare(root, tmp_path / "missing", **kwargs)


def test_inference_receipt_rejects_stale_or_truncated_outputs(tmp_path):
    from tracking.single_ant.submit import gpu_done
    manifest = make_run(tmp_path)
    directory = Path(manifest["output"], "detections/session0")
    slp = directory / "video0_000.slp"
    slp.write_bytes(b"fresh")
    receipt = directory / "video0_000.done.json"
    state = dict(contract_sha256="test", frames=12, slp_bytes=5,
                 h5_bytes=(directory / "video0_000_sleap_data.h5").stat().st_size)
    receipt.write_text(json.dumps(state))
    assert gpu_done(manifest, [0, 0])
    slp.write_bytes(b"bad")
    assert not gpu_done(manifest, [0, 0])
    slp.write_bytes(b"fresh")
    state["contract_sha256"] = "old-model"
    receipt.write_text(json.dumps(state))
    assert not gpu_done(manifest, [0, 0])


def test_waits_for_all_prerequisite_arrays(monkeypatch):
    from tracking.single_ant import submit
    monkeypatch.setenv("USER", "tester")
    monkeypatch.setattr(submit, "command", lambda *args: "123\n 456 \n")
    assert submit.is_live("123,456", "saion")
    assert submit.is_live("456,789", "saion")
    assert not submit.is_live("789", "saion")
