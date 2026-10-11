"""Shared return/sleep loaders validate track identities and published contacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import interaction_analysis_utils as ia
from analysis import return_sleep_utils as rs
from analysis import sleep_motion_analysis_utils as sma
from tracking.colony.skeleton_contacts import CONTACT_GEOMETRY


def test_sleep_loader_uses_selected_filename_not_duplicate_ant_export(tmp_path):
    for name in ("short_right.parquet", "full_right.parquet"):
        folder = tmp_path / "per_track" / Path(name).stem
        folder.mkdir(parents=True)
        np.save(folder / "state.npy", np.array([0, 1, -1], dtype=np.int8))
        metadata = dict(summary=dict(side="right", track_id=8, track_name=name),
                        frame_min=10, frame_max=12, fps=24, classifier_parameters={},
                        files=dict(sleep_state="state.npy"))
        (folder / "sleep_motion_label_metadata.json").write_text(json.dumps(metadata))
    clusters = pd.DataFrame([dict(side="right", TrackID=8, track_name="full_right.parquet", cluster_id="right_0")])
    selected = sma.load_sleep_label_tracks(tmp_path, clusters)
    assert selected.track_name.tolist() == ["full_right.parquet"]
    assert selected.cluster_id.tolist() == ["right_0"]
    with pytest.raises(ValueError, match="not unique"):
        sma.load_sleep_label_tracks(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing sleep labels"):
        sma.load_sleep_label_tracks(tmp_path, clusters.assign(track_name="missing_right.parquet"))


@pytest.fixture
def published(tmp_path, monkeypatch):
    root, tracks = tmp_path / "contacts", tmp_path / "tracks"
    root.mkdir()
    tracks.mkdir()
    name = "test_left.parquet"
    (root / name).touch()
    (tracks / name).touch()
    manifest = dict(run_id="test", expected_files=[name], parameters=dict(
        geometry=CONTACT_GEOMETRY, micro_interaction_distance_mm=.1, directed=False,
    ))
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    (root / "transfer_complete.ok").write_text(json.dumps(dict(run_id="test", n_chunks=1)))
    chunk = ia.InteractionChunk(root/name, tracks/name, "000", "left", 100, 100, 0, 24, 24)
    monkeypatch.setattr(ia, "resolve_chunks", lambda *args, side, **kwargs: [chunk] if side == "left" else [])
    return root, tracks, manifest


def test_shared_loader_accepts_published_contacts(published):
    root, tracks, manifest = published
    chunks, loaded = rs.load_published_interaction_chunks(root, tracks, fps=24, start_clock_seconds=100)
    assert loaded == manifest and len(chunks) == 1


@pytest.mark.parametrize("field,value", [("micro_interaction_distance_mm", 1), ("directed", True),
                                        ("geometry", "antenna_to_node")])
def test_shared_loader_refuses_incompatible_contacts(published, field, value):
    root, tracks, manifest = published
    manifest["parameters"][field] = value
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="undirected full-skeleton"):
        rs.load_published_interaction_chunks(root, tracks, fps=24, start_clock_seconds=100)


def test_shared_loader_reports_unpublished_and_missing_files(published):
    root, tracks, manifest = published
    path = tracks / manifest["expected_files"][0]
    path.unlink()
    with pytest.raises(FileNotFoundError, match=str(path)):
        rs.load_published_interaction_chunks(root, tracks, fps=24, start_clock_seconds=100)
    (root / "transfer_complete.ok").unlink()
    with pytest.raises(FileNotFoundError, match="not published yet"):
        rs.load_published_interaction_chunks(root, tracks, fps=24, start_clock_seconds=100)


def test_shared_loader_refuses_wrong_clock_or_completion_marker(published):
    root, tracks, _ = published
    with pytest.raises(ValueError, match="clocks differ"):
        rs.load_published_interaction_chunks(root, tracks, fps=24, start_clock_seconds=200)
    (root / "transfer_complete.ok").write_text(json.dumps(dict(run_id="old", n_chunks=1)))
    with pytest.raises(ValueError, match="completion marker"):
        rs.load_published_interaction_chunks(root, tracks, fps=24, start_clock_seconds=100)
