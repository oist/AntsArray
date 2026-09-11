"""Duplicate-ID unit/integration tests; videos are generated, no dataset needed."""
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import pandas as pd
import pytest

from detection_pipeline.scripts import run_aruco_mp as detector
from detection_pipeline.scripts import aruco_output
import run_aruco as serial_detector
from tracking.colony import map_combine as mapper
from tracking.colony import combine_one_chunk
from tracking.colony.combine_batch import discover_jobs
from tracking.colony.panorama_io import discover_complete_input_chunks
from tracking.core.tracking_utils import get_complete_tracks


NAME = "cam03_cam2_2026-08-10-18-43-25_000"
ROWS = [(0, 75, 3100., 100.), (0, 75, 100., 100.), (1, 10, 110., 120.)]


def write_outputs(path, rows=ROWS, num_frames=5):
    packed = detector.pack_detections(rows, num_frames, 100)
    detector.save_aruco_outputs(path, NAME, *packed, output_format="both")
    return (path / f"{NAME}_aruco_tracks.h5",
            path / f"{NAME}_aruco_detections.h5", packed)


def test_all_exports_keep_duplicates_and_legacy_summary_is_unchanged(tmp_path):
    raw, table, (tracks, confidences, expected) = write_outputs(tmp_path)
    assert tracks[0, 75].tolist() == [100., 100.]
    assert np.count_nonzero(confidences) == 2
    with h5py.File(raw) as f:
        assert len(f["aruco_detections"]) == 3
        assert f.attrs["dense_duplicate_instances_dropped"] == 1
        assert f.attrs["num_frames"] == 5
    pd.testing.assert_frame_equal(pd.read_hdf(table, "detections"), expected)
    assert len(pd.read_csv(tmp_path / f"{NAME}_aruco_detections.csv")) == 3
    for path in (raw, table):
        df, count = mapper._load_aruco_input_to_df_and_num_frames(path)
        pd.testing.assert_frame_equal(df, expected, check_dtype=False)
        assert count == 5
    with h5py.File(raw) as f:
        assert mapper.aruco_h5_to_long_df_full(f, frame_offset=7).Frame.tolist() == [7, 7, 8]


@pytest.mark.parametrize("mode,workers", [("mp", 1), ("mp", 2), ("serial", 1)])
def test_detector_cli_keeps_duplicate_ids_in_real_video(tmp_path, mode, workers):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    image = np.full((300, 800, 3), 255, dtype=np.uint8)
    for x, tag in [(50, 7), (300, 7), (550, 8)]:
        marker = cv2.aruco.generateImageMarker(dictionary, tag, 100)
        image[100:200, x:x+100] = marker[..., None]
    video = tmp_path / "duplicates.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), 24., (800, 300))
    assert writer.isOpened()
    for _ in range(3):
        writer.write(image)
    writer.release()
    # Expected-frame metadata is only a hint: the final worker reads to EOF.
    output = tmp_path / "output"
    # A fresh CLI process also tests main() and avoids forking after this test's
    # video writer has started OpenCV threads in the parent.
    script = detector.__file__ if mode == "mp" else serial_detector.__file__
    extra = ["--workers", str(workers), "--n-frames", "2"] if mode == "mp" else []
    subprocess.run([sys.executable, str(Path(script)),
                    "--video-file", str(video), "--output-path", str(output),
                    "--dictionary-size", "100", *extra], cwd=tmp_path,
                   check=True, capture_output=True, text=True, timeout=45)
    raw = output / "duplicates_aruco_tracks.h5"
    with h5py.File(raw) as f:
        tracks, confidence = f["aruco_tracks"][:], f["aruco_confidences"][:]
    detections, num_frames = mapper._load_aruco_input_to_df_and_num_frames(raw)
    assert num_frames == 3
    assert tracks.shape == (3, 100, 2)
    assert np.count_nonzero(confidence) == 6
    assert detections.groupby("Frame").size().tolist() == [3, 3, 3]
    assert detections.query("Instance == 7").groupby("Frame").size().tolist() == [2, 2, 2]


def test_lossless_records_preserve_valid_origin_coordinate(tmp_path):
    raw, _, _ = write_outputs(tmp_path, rows=[(0, 1, 0., 0.)])
    detections, _ = mapper._load_aruco_input_to_df_and_num_frames(raw)
    assert detections[["X", "Y"]].to_numpy().tolist() == [[0., 0.]]


def test_native_records_do_not_require_legacy_dense_arrays(tmp_path):
    raw, table, _ = write_outputs(tmp_path)
    with h5py.File(raw, "a") as f:
        del f["aruco_tracks"]
        del f["aruco_confidences"]
    for path in (raw, table):
        df, count = mapper._load_aruco_input_to_df_and_num_frames(path)
        assert len(df) == 3 and count == 5


def test_inconsistent_native_frame_span_is_rejected(tmp_path):
    raw, _, _ = write_outputs(tmp_path)
    with h5py.File(raw, "a") as f:
        f.attrs["num_frames"] = 1
    with pytest.raises(ValueError, match="Frame-count metadata disagrees"):
        mapper._load_aruco_input_to_df_and_num_frames(raw)


def test_failed_export_does_not_truncate_existing_file(tmp_path):
    raw, _, packed = write_outputs(tmp_path)
    before = raw.read_bytes()
    with patch.object(aruco_output.h5py.Group, "create_dataset", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            aruco_output.save_aruco_outputs(tmp_path, NAME, *packed)
    assert raw.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_export_without_pytables_keeps_native_records(tmp_path):
    packed = detector.pack_detections(ROWS, 5, 100)
    with patch.dict(sys.modules, {"tables": None}):
        aruco_output.save_aruco_outputs(tmp_path, NAME, *packed)
    raw = tmp_path / f"{NAME}_aruco_tracks.h5"
    df, count = mapper._load_aruco_input_to_df_and_num_frames(raw)
    assert len(df) == 3 and count == 5
    assert not (tmp_path / f"{NAME}_aruco_detections.h5").exists()


@pytest.mark.parametrize("num_frames", [0, 8])
def test_empty_output_and_trailing_empty_frames(tmp_path, num_frames):
    raw, table, _ = write_outputs(tmp_path, rows=[], num_frames=num_frames)
    for path in (raw, table):
        df, count = mapper._load_aruco_input_to_df_and_num_frames(path)
        assert df.empty and count == num_frames
    raw.rename(raw.with_suffix(".hidden"))
    df, count = mapper._load_aruco_input_to_df_and_num_frames(table)
    assert df.empty and count == num_frames


def test_standalone_table_preserves_explicit_frame_span(tmp_path):
    raw, table, _ = write_outputs(tmp_path)
    raw.rename(raw.with_suffix(".hidden"))
    df, count = mapper._load_aruco_input_to_df_and_num_frames(table)
    assert len(df) == 3 and count == 5


def test_native_records_survive_unreadable_or_stale_pandas_table(tmp_path):
    raw, table, _ = write_outputs(tmp_path)
    # An old lossy sidecar must not take priority over refreshed native records.
    pd.DataFrame(ROWS[1:], columns=["Frame", "Instance", "X", "Y"]).to_hdf(
        table, key="detections", mode="w")
    assert len(mapper._load_aruco_input_to_df_and_num_frames(table)[0]) == 3
    with patch.object(mapper.pd, "HDFStore", side_effect=ImportError("PyTables unavailable")):
        df, count = mapper._load_aruco_input_to_df_and_num_frames(table)
    assert len(df) == 3 and count == 5


def test_v2_raw_missing_records_refuses_lossy_fallback(tmp_path):
    raw, table, _ = write_outputs(tmp_path)
    with h5py.File(raw, "a") as f:
        del f["aruco_detections"]
    for path in (raw, table):
        with pytest.raises(ValueError, match="refusing lossy fallback"):
            mapper._load_aruco_input_to_df_and_num_frames(path)


def test_v2_table_unreadable_with_only_legacy_sibling_fails_closed(tmp_path):
    raw, table, _ = write_outputs(tmp_path)
    with h5py.File(raw, "a") as f:
        del f["aruco_detections"]
        del f.attrs["aruco_detection_schema_version"]
    with patch.object(mapper.pd, "HDFStore", side_effect=ValueError("version drift")):
        with pytest.raises(ValueError, match="refusing lossy fallback"):
            mapper._load_aruco_input_to_df_and_num_frames(table)


def test_legacy_dense_still_loads_with_explicit_warning(tmp_path, caplog):
    raw, table, _ = write_outputs(tmp_path)
    with h5py.File(raw, "a") as f:
        del f["aruco_detections"]
        del f.attrs["aruco_detection_schema_version"]
    df, count = mapper._load_aruco_input_to_df_and_num_frames(raw)
    assert len(df) == 2 and count == 5
    assert "duplicates cannot be recovered" in caplog.text
    # Also exercise the older pandas-version-drift fallback.
    pd.DataFrame().to_hdf(table, key="wrong_key", mode="w")
    df, count = mapper._load_aruco_input_to_df_and_num_frames(table)
    assert len(df) == 2 and count == 5


def test_mapping_routes_same_camera_same_id_to_each_colony(tmp_path, monkeypatch):
    monkeypatch.setattr(mapper, "X_THRESHOLD", 2500.)
    source, output = tmp_path / "raw", tmp_path / "mapped"
    output.mkdir()
    write_outputs(source)
    mapper.process_aruco_chunks([np.eye(3)] * 3, source, output, "test",
                               min_instance_frame_frac=0)
    for side, x in [("left", 100.), ("right", 3100.)]:
        data = pd.read_pickle(next(output.glob(f"*_x_{side}*.pkl")))
        hit = data["detections"].query("Frame == 0 and Instance == 75")
        assert hit.X.tolist() == [x]
        assert hit.Cam.tolist() == [2]
        assert data["num_frames"] == 5


@pytest.mark.parametrize("other_camera", [2, 3])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_existing_tracker_selects_nearest_duplicate_regardless_of_camera_or_order(
        other_camera, reverse_order):
    initial = [(0, 75, 100., 100., 2)]
    duplicates = [(1, 75, 140., 100., other_camera), (1, 75, 102., 100., 2)]
    if reverse_order:
        duplicates.reverse()
    aruco = pd.DataFrame(initial + duplicates, columns=["Frame", "Instance", "X", "Y", "Cam"])
    sleap = pd.DataFrame([
        (0, 0, 0, 100., 100., 2),
        (1, 1, 0, 140., 100., other_camera),
        (1, 2, 0, 102., 100., 2),
    ], columns=["Frame", "Instance", "Bodypoint", "X", "Y", "Cam"])
    result = get_complete_tracks(None, aruco, sleap, num_frames=2,
                                 max_distance=100., aruco_sleap_max_distance=5.)
    assert list(result[1]) == [75]
    assert result[1][75][0] == (102., 100.)


@pytest.mark.parametrize("input_format", ["native", "pandas", "both"])
def test_end_to_end_mapping_and_tracking_cli_with_mixed_duplicates(tmp_path, input_format):
    """Two arenas share ID75; duplicates within/across cameras, gap + handoff."""
    source, mapped, output = (tmp_path / name for name in ("data", "panorama", "tracks"))
    hmats = np.repeat(np.eye(3)[None], 4, axis=0)
    hmats[2, 0, 2], hmats[3, 0, 2] = 1000., 900.
    calibration = tmp_path / "hmats.npz"
    np.savez(calibration, H=hmats)
    for camera, prefix in [(3, NAME), (4, NAME.replace("cam03_cam2", "cam04_cam3"))]:
        tag_rows, skeleton_rows = [], []
        for frame in range(5):
            for instance, base in enumerate([1100., 3000.]):
                true_x = base + 3*frame
                offset = hmats[camera-1, 0, 2]
                if camera == 3 and frame in (0, 1, 3):
                    if frame != 0:  # Distractor first; nearest candidate must win.
                        tag_rows.append((frame, 75, true_x + 40-offset, 100.))
                    tag_rows.append((frame, 75, true_x-offset, 100.))
                if camera == 4 and frame in (1, 4):
                    tag_rows.append((frame, 75, true_x + (frame == 1)-offset, 100.))
                if (camera == 3 and frame < 4) or (camera == 4 and frame in (1, 4)):
                    x = true_x + (camera == 4 and frame == 1)-offset
                    # Both cameras intentionally reuse SLEAP Instance values.
                    skeleton_rows.extend([(frame, instance, 0, x, 100.),
                                          (frame, instance, 1, x+1, 100.)])
                    if camera == 3 and frame in (1, 3):
                        skeleton_rows.extend([(frame, instance+2, 0, x+40, 100.),
                                              (frame, instance+2, 1, x+41, 100.)])
        packed = detector.pack_detections(tag_rows, 6, 100)  # trailing empty frame
        detector.save_aruco_outputs(source, prefix, *packed)
        if input_format == "native":
            table = source / f"{prefix}_aruco_detections.h5"
            table.rename(table.with_suffix(".unused"))
        elif input_format == "pandas":
            raw = source / f"{prefix}_aruco_tracks.h5"
            raw.rename(raw.with_suffix(".unused"))
        skeletons = pd.DataFrame(skeleton_rows, columns=["Frame", "Instance", "Bodypoint", "X", "Y"])
        with h5py.File(source / f"{prefix}_sleap_data.h5", "w") as f:
            f["sleap_data"] = skeletons.to_records(index=False)

    complete, _ = discover_complete_input_chunks(source)
    assert complete == ["000"]
    subprocess.run([sys.executable, str(Path(mapper.__file__)), "--data_dir", str(source),
                    "--outdir", str(mapped), "--hmats", str(calibration), "--mode", "both"],
                   cwd=tmp_path, check=True, capture_output=True, text=True, timeout=45)
    jobs = discover_jobs(mapped, ("left", "right"))
    assert len(jobs) == 2
    for job in jobs:
        payload = pd.read_pickle(job.representative_file)
        assert payload["num_frames"] == 6
        tags = payload["detections"]
        assert len(tags.query("Frame == 1")) == 3  # two in Cam2 + one in Cam3
        assert len(tags.query("Frame == 1 and Cam == 2")) == 2
        subprocess.run([sys.executable, str(Path(combine_one_chunk.__file__)),
                        "--input_file", str(job.representative_file), "--output_path", str(output)],
                       cwd=tmp_path, check=True, capture_output=True, text=True, timeout=45)
        tracks = pd.read_parquet(output / f"{job.key}_{job.side}.parquet")
        assert not tracks.duplicated(["Frame", "TrackID", "Bodypoint"]).any()
        anchors = tracks.query("Bodypoint == 0").sort_values("Frame")
        assert anchors.TrackID.tolist() == [75]*5
        assert anchors.Frame.tolist() == list(range(5))
        base = 1100. if job.side == "left" else 3000.
        np.testing.assert_allclose(anchors.TrackX, base + 3*np.arange(5))
        assert anchors.loc[anchors.Frame == 2, "ArucoX"].isna().all()
        assert anchors.loc[anchors.Frame == 4, "ArucoCam"].tolist() == [3]
        assert anchors.loc[anchors.Frame == 4, "SleapCam"].tolist() == [3]
