"""The tracking pipeline records its homography through the shared module.

The catalog (python 3.6 on the deigo login node) and the tracking pipeline
(the ant_tracking venv) must agree on one file format, so both import
``detection_pipeline/lib/tracking_state.py``. This test pins that the pipeline
entry point resolves that import and writes a record the module reads back.
"""
from pathlib import Path

from detection_pipeline.lib import tracking_state
from tracking.colony import pipeline


def test_pipeline_uses_the_shared_tracking_state_module():
    assert pipeline.tracking_state is tracking_state


def test_record_map_round_trips_through_the_pipeline_import(tmp_path: Path):
    hmats = tmp_path / "cameraArray_calib" / "20260810_calib_elevated_by_2mm" / "aruco_H_mats.npz"
    hmats.parent.mkdir(parents=True)
    hmats.write_bytes(b"H")
    tracks = tmp_path / "tracks"
    pipeline.tracking_state.record_map(
        tracks, hmats=hmats, x_threshold=2475.0, map_mode="both",
        chunks=["000", "001"], code_dir=pipeline.REPO_ROOT, argv=["pipeline.py"],
    )
    state = tracking_state.load(tracks)
    assert state["map"]["hmats_calib_id"] == "20260810_calib_elevated_by_2mm"
    assert state["map"]["chunks"] == {"first": "000", "last": "001", "count": 2}
    assert tracking_state.summary(state)["tracking_x_threshold"] == "2475"
