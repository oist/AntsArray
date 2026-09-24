"""Tracking must partition detections using the selected block's annotations."""
import csv
import logging
import sys

import pandas as pd
import pytest

from camera_cal.region_paths import panorama_tracking_split
from tracking.colony import map_combine, pipeline


def write_regions(path, *, kind="arena", left_end=1900, right_start=2000):
    rows = [dict(semantic_label=f"{kind}_R", shape="rectangle", name="right",
                 tracking_x_min_px=right_start, tracking_x_max_px=11000),
            dict(semantic_label=f"{kind}_L", shape="rectangle", name="left",
                 tracking_x_min_px=-7000, tracking_x_max_px=left_end)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_mapping_reads_block_regions_and_splits_aruco(tmp_path, monkeypatch):
    block = tmp_path / "20260515" / "block03"
    data = block / "data"
    data.mkdir(parents=True)
    write_regions(block.parent / "panorama_regions.csv", left_end=100, right_start=200)
    write_regions(block / "panorama_regions.csv")
    monkeypatch.setattr(map_combine, "X_THRESHOLD", 2500.0)
    monkeypatch.setattr(map_combine, "load_homographies", lambda _: [])
    monkeypatch.setattr(map_combine, "infer_experiment_name", lambda _: "test")

    def map_aruco(hmats, data_dir, out_dir, exp, **kwargs):
        df = pd.DataFrame({"X": [1949.0, 1950.0, 2200.0], "Y": [0.0] * 3})
        map_combine.split_and_write_with_num_frames_flat(df, out_dir, exp, 3)

    monkeypatch.setattr(map_combine, "process_aruco_chunks", map_aruco)
    output = tmp_path / "output"
    pipeline.run_mapping(hmats_path=tmp_path / "H.npz", data_dir=data,
                         panorama_dir=output, map_mode="aruco",
                         min_instance_frame_frac=0.25, x_threshold=None,
                         skip_existing=False)
    left = pd.read_pickle(output / "test_x_left1950.pkl")["detections"]
    right = pd.read_pickle(output / "test_x_right1950.pkl")["detections"]
    assert left.X.tolist() == [1949.0]
    assert right.X.tolist() == [1950.0, 2200.0]


def test_tracking_record_stores_resolved_arena_split_before_mapping(tmp_path, monkeypatch):
    write_regions(tmp_path / "panorama_regions.csv")
    tracks = tmp_path / "tracks"
    hmats = tmp_path / "H.npz"
    hmats.write_bytes(b"test calibration")
    monkeypatch.setattr(map_combine, "X_THRESHOLD", 2500.0)
    monkeypatch.setattr(map_combine, "load_homographies", lambda _: [])
    monkeypatch.setattr(map_combine, "infer_experiment_name", lambda _: "test")

    def check_record(*args, **kwargs):
        state = pipeline.tracking_state.load(tracks)
        assert state["map"]["x_threshold"] == map_combine.X_THRESHOLD == 1950.0
        assert state["map"]["chunks"] == {"first": "000", "last": "001", "count": 2}

    monkeypatch.setattr(map_combine, "process_aruco_chunks", check_record)
    pipeline.run_mapping(hmats_path=hmats, data_dir=tmp_path / "data",
                         panorama_dir=tmp_path / "panorama", map_mode="aruco",
                         min_instance_frame_frac=0.25, x_threshold=None,
                         skip_existing=False, chunks={"000", "001"},
                         tracking_state_dir=tracks)


def test_date_fallback_and_annotation_order(tmp_path):
    block = tmp_path / "20260515" / "block03"
    block.mkdir(parents=True)
    path = block.parent / "panorama_regions.csv"
    write_regions(path)
    assert panorama_tracking_split(block) == (1950.0, path)


@pytest.mark.parametrize("labels", [("colony_L", "colony_R"), ("colonyL", "colonyR")])
def test_nest_rectangles_cannot_define_tracking_split(tmp_path, labels):
    path = tmp_path / "panorama_regions.csv"
    write_regions(path)
    path.write_text(path.read_text().replace("arena_L", labels[0]).replace("arena_R", labels[1]))
    with pytest.raises(ValueError, match="Left arena and Right arena buttons"):
        panorama_tracking_split(tmp_path)


def test_nest_positions_do_not_move_arena_split(tmp_path):
    path = tmp_path / "panorama_regions.csv"
    write_regions(path, left_end=2470, right_start=2480)
    with path.open("a") as handle:
        handle.write("colonyL,rectangle,left nest,291,1372\n")
        handle.write("colonyR,rectangle,right nest,2901,3996\n")
    assert panorama_tracking_split(tmp_path)[0] == 2475


def test_region_reader_rejects_missing_or_invalid_annotations(tmp_path):
    with pytest.raises(FileNotFoundError):
        panorama_tracking_split(tmp_path)
    path = tmp_path / "panorama_regions.csv"
    write_regions(path, left_end=2100)
    with pytest.raises(ValueError, match="overlap"):
        panorama_tracking_split(tmp_path)
    write_regions(path, left_end=float("nan"))
    with pytest.raises(ValueError, match="invalid tracking"):
        panorama_tracking_split(tmp_path)


def test_explicit_override_can_map_without_annotations(tmp_path, monkeypatch):
    monkeypatch.setattr(map_combine, "X_THRESHOLD", 2500.0)
    monkeypatch.setattr(map_combine, "load_homographies", lambda _: [])
    monkeypatch.setattr(map_combine, "infer_experiment_name", lambda _: "test")
    pipeline.run_mapping(hmats_path=tmp_path / "H.npz", data_dir=tmp_path / "data",
                         panorama_dir=tmp_path / "output", map_mode="none",
                         min_instance_frame_frac=0.25, x_threshold=123.5,
                         skip_existing=False)
    assert map_combine.X_THRESHOLD == 123.5


def test_extending_a_window_preserves_its_existing_recording_name(tmp_path, monkeypatch):
    # Full-block directory iteration may find a different camera's start time.
    monkeypatch.setattr(map_combine, "infer_experiment_name", lambda _: "20260726_104312")
    monkeypatch.setattr(map_combine, "load_homographies", lambda _: [])
    names = []
    def capture(hmats, data_dir, out_dir, exp, **kwargs):
        names.append(exp)
    monkeypatch.setattr(map_combine, "process_aruco_chunks", capture)
    monkeypatch.setattr(map_combine, "process_sleap_chunks", capture)
    pipeline.run_mapping(hmats_path=tmp_path / "H.npz", data_dir=tmp_path / "data",
                         panorama_dir=tmp_path / "panorama", map_mode="both",
                         min_instance_frame_frac=0.25, x_threshold=2477.7,
                         skip_existing=True, experiment_name="20260726_104256")
    assert names == ["20260726_104256", "20260726_104256"]


@pytest.mark.parametrize("entrypoint", ["pipeline", "standalone"])
@pytest.mark.parametrize("has_regions", [True, False])
def test_both_mapping_entrypoints_report_boundary_source(
    tmp_path, monkeypatch, caplog, entrypoint, has_regions,
):
    data = tmp_path / "data"
    data.mkdir()
    if has_regions:
        write_regions(tmp_path / "panorama_regions.csv")
    monkeypatch.setattr(map_combine, "X_THRESHOLD", -999.0)
    monkeypatch.setattr(map_combine, "load_homographies", lambda _: [])
    monkeypatch.setattr(map_combine, "infer_experiment_name", lambda _: "test")
    monkeypatch.setattr(map_combine, "process_aruco_chunks", lambda *a, **kw: None)
    caplog.set_level(logging.INFO)
    def run():
        if entrypoint == "pipeline":
            pipeline.run_mapping(hmats_path=tmp_path / "H.npz", data_dir=data,
                                 panorama_dir=tmp_path / "output", map_mode="aruco",
                                 min_instance_frame_frac=0.25, x_threshold=None,
                                 skip_existing=False)
        else:
            monkeypatch.setattr(sys, "argv", ["map_combine.py", "--data_dir", str(data),
                                             "--outdir", str(tmp_path / "output"), "--mode", "aruco"])
            map_combine.main()
    if has_regions:
        run()
        assert map_combine.X_THRESHOLD == 1950.0
        assert "from " + str(tmp_path / "panorama_regions.csv") in caplog.text
    else:
        with pytest.raises(FileNotFoundError, match="any earlier recording date"):
            run()
        assert map_combine.X_THRESHOLD == -999.0
        assert "Annotated boundaries are unavailable" in caplog.text
        assert "--x_threshold override" in caplog.text
        assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_existing_invalid_annotations_do_not_silently_fall_back(tmp_path):
    write_regions(tmp_path / "panorama_regions.csv", left_end=2100)
    with pytest.raises(ValueError, match="overlap"):
        map_combine.resolve_x_threshold(tmp_path / "data")


def test_mapping_uses_and_logs_the_selected_earlier_annotation(tmp_path, caplog):
    block = tmp_path / "20260515/block03"
    (block / "data").mkdir(parents=True)
    earlier = tmp_path / "20260514/block02/panorama_regions.csv"
    earlier.parent.mkdir(parents=True)
    write_regions(earlier, left_end=1700, right_start=1800)
    future = tmp_path / "20260516/block01/panorama_regions.csv"
    future.parent.mkdir(parents=True)
    write_regions(future, left_end=2500, right_start=2600)
    caplog.set_level(logging.INFO)
    assert map_combine.resolve_x_threshold(block / "data") == 1750.0
    assert panorama_tracking_split(block) == (1750.0, earlier)
    assert "most recent earlier recording date" in caplog.text
    assert str(earlier) in caplog.text
    assert "X=1750" in caplog.text


def test_invalid_selected_history_fails_instead_of_using_older_geometry(tmp_path):
    block = tmp_path / "20260515/block03"
    block.mkdir(parents=True)
    older = tmp_path / "20260513/panorama_regions.csv"
    older.parent.mkdir()
    write_regions(older)
    selected = tmp_path / "20260514/panorama_regions.csv"
    selected.parent.mkdir()
    write_regions(selected, left_end=2100)
    with pytest.raises(ValueError, match="overlap"):
        map_combine.resolve_x_threshold(block / "data")


@pytest.mark.parametrize("left,right", [("arenaL", "arenaR"), ("arenaLeft", "arenaRight"),
                                        ("arena_left", "arena_right")])
def test_compact_annotation_side_labels(tmp_path, left, right):
    path = tmp_path / "panorama_regions.csv"
    write_regions(path)
    path.write_text(path.read_text().replace("arena_L", left).replace("arena_R", right))
    assert panorama_tracking_split(tmp_path)[0] == 1950.0
    path.write_text(path.read_text().replace(left, "temporary").replace(right, left).replace("temporary", right))
    with pytest.raises(ValueError, match="label/geometry"):
        panorama_tracking_split(tmp_path)
