import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis import sleep_analysis_utils as sa
from analysis import sleep_motion_analysis_utils as sma


def _pose(path, frames, *, rotated=False, degenerate=False):
    rows = []
    for frame in frames:
        for bp, (x, y) in {0: (0, 0), 1: (2, 0), 4: (2, 1), 5: (3, 2), 6: (4, 3)}.items():
            if degenerate and bp == 1:
                x, y = 0, 0
            if rotated:
                x, y = -y, x
            rows.append(dict(Frame=frame, Bodypoint=bp, X=x+10, Y=y+20, TrackX=10, TrackY=20))
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_alignment_translation_rotation_and_track_frame_counts(tmp_path):
    samples = []
    for ant in (0, 1):
        path = tmp_path / f"track{ant}.parquet"
        _pose(path, [100], rotated=bool(ant))
        samples.append(dict(track_path=str(path), Frame=100, posture_state="sleep", track_id=ant, side="left"))
    points, summary = sa.aligned_posture_points_from_frame_states(pd.DataFrame(samples), mm_per_px=.5)
    np.testing.assert_allclose(points.loc[points.bodypoint == 1, ["aligned_x_mm", "aligned_y_mm"]], [[1, 0], [1, 0]], atol=1e-12)
    np.testing.assert_allclose(points.loc[points.bodypoint == 6, ["aligned_x_mm", "aligned_y_mm"]], [[2, 1.5], [2, 1.5]], atol=1e-12)
    assert summary.n_frames.item() == 2  # Same global frame in two ants is two poses.


@pytest.fixture
def labelled_tracks(tmp_path):
    per_track = tmp_path / "stitched" / "per_track"
    per_track.mkdir(parents=True)
    rows = []
    for ant, states in [(1, [1, 1, 0, -1]), (2, [1, 0, 0, -1])]:
        name = f"track{ant}.parquet"
        _pose(per_track / name, [100, 101, 102, 103])
        state_path = tmp_path / f"state{ant}.npy"
        np.save(state_path, np.array(states, dtype=np.int8))
        metadata_path = tmp_path / f"label{ant}.json"
        metadata_path.write_text("{}")
        scale_path = tmp_path / "stitched" / "sleep_motion" / "per_track" / Path(name).stem
        scale_path.mkdir(parents=True)
        (scale_path / "sleep_motion_metadata.json").write_text(json.dumps(dict(mm_per_px=.5)))
        rows.append(dict(track_name=name, side="left", track_id=ant, frame_min=100,
                         state_path=str(state_path), metadata_path=str(metadata_path)))
    return pd.DataFrame(rows), tmp_path


def test_cached_labels_unknown_offset_equal_weight_and_cache_invalidation(labelled_tracks, monkeypatch):
    tracks, root = labelled_tracks
    points, audit = sma.load_sleep_posture_points(tracks, root, root / "cache")
    assert set(points.Frame) == {100, 101, 102}
    assert points.loc[(points.track_id == 1) & (points.Frame == 102), "posture_state"].eq("wake").all()
    assert audit.n_aligned_frames.sum() == 6
    weights = points.groupby(["track_id", "posture_state"], observed=True).density_weight.sum()
    np.testing.assert_allclose(weights, 1)
    original = sa.aligned_posture_points_from_frame_states
    monkeypatch.setattr(sa, "aligned_posture_points_from_frame_states", lambda *args, **kwargs: pytest.fail("cache miss"))
    cached, _ = sma.load_sleep_posture_points(tracks, root, root / "cache")
    pd.testing.assert_frame_equal(points, cached)
    monkeypatch.setattr(sa, "aligned_posture_points_from_frame_states", original)
    np.save(tracks.state_path.iloc[0], np.array([0, 1, 0, -1], dtype=np.int8))
    changed, _ = sma.load_sleep_posture_points(tracks, root, root / "cache")
    assert changed.loc[(changed.track_id == 1) & (changed.Frame == 100), "posture_state"].eq("wake").all()


def test_missing_pose_reports_exact_path(labelled_tracks):
    tracks, root = labelled_tracks
    path = root / "stitched" / "per_track" / tracks.track_name.iloc[0]
    path.unlink()
    with pytest.raises(FileNotFoundError, match=str(path)):
        sma.load_sleep_posture_points(tracks, root, root / "cache")


def test_sampling_is_capped_per_state_and_stable_under_track_reordering(labelled_tracks):
    tracks, root = labelled_tracks
    points, audit = sma.load_sleep_posture_points(tracks, root, root / "cache", max_frames_per_ant_state=1)
    assert audit.n_sampled_frames.eq(1).all()
    reordered, _ = sma.load_sleep_posture_points(tracks.iloc[::-1], root, root / "cache", max_frames_per_ant_state=1, force=True)
    order = ["track_id", "posture_state", "Frame", "bodypoint"]
    pd.testing.assert_frame_equal(points.sort_values(order).reset_index(drop=True),
                                  reordered.sort_values(order).reset_index(drop=True))


def test_grid_posture_cell_reports_missing_labels(tmp_path, capsys):
    script = Path(__file__).with_name("grid_occupancy.py")
    cell = next(cell for cell in script.read_text().split("# %%") if "# Sleep/wake posture distribution," in cell)
    root = tmp_path / "labels"
    exec(cell, dict(GRID_ROOT=tmp_path, SLEEP_LABEL_ROOT=root, sleep_label_tracks=None))
    output = capsys.readouterr().out
    assert "MISSING INPUT" in output and str(root) in output
    assert not (tmp_path / "sleep_motion_analysis").exists()


def test_legacy_plot_interface_still_returns_three_tables(labelled_tracks, monkeypatch):
    tracks, root = labelled_tracks
    points, _ = sma.load_sleep_posture_points(tracks, root, root / "cache")
    frames = points.drop_duplicates(["track_id", "Frame"])
    summary = frames.groupby("posture_state", observed=True).agg(n_frames=("Frame", "size"), n_tracks=("track_id", "nunique")).reset_index()
    monkeypatch.setattr(sa, "aligned_posture_points_for_states", lambda *args, **kwargs: (points, summary))
    monkeypatch.setattr(plt, "show", lambda: None)
    original = set(plt.get_fignums())
    try:
        result = sa.plot_aligned_posture_state_heatmaps(pd.DataFrame(), pd.DataFrame(), bins=20, state_order=("sleep", "wake"))
        assert len(result) == 3 and all(isinstance(table, pd.DataFrame) for table in result)
    finally:
        for number in set(plt.get_fignums()) - original:
            plt.close(number)


def test_missing_axis_reported_and_not_plotted_as_zero(labelled_tracks, capsys):
    tracks, root = labelled_tracks
    _pose(root / "stitched" / "per_track" / tracks.track_name.iloc[0], [100, 101, 102, 103], degenerate=True)
    points, audit = sma.load_sleep_posture_points(tracks, root, root / "cache")
    assert points.track_id.eq(2).all()
    assert audit.loc[audit.track_id == 1, "n_aligned_frames"].eq(0).all()
    assert "NO VALID POSTURES" in capsys.readouterr().out


def test_plot_states_common_scale_and_missing_wake_panel(labelled_tracks):
    tracks, root = labelled_tracks
    points, _ = sma.load_sleep_posture_points(tracks, root, root / "cache")
    right = points.loc[points.posture_state == "sleep"].assign(side="right")
    figures, summary, medians = sma.plot_sleep_posture_distributions(pd.concat([points, right]), bins=20, extent_mm=4)
    try:
        assert set(figures) == {"left", "right"}
        assert all(len(fig.axes) == 3 for fig in figures.values())
        assert "No valid classified poses" in [text.get_text() for text in figures["right"].axes[1].texts]
        limits = [ax.collections[0].get_clim() for fig in figures.values() for ax in fig.axes[:2]]
        assert len(set(limits)) == 1
        np.testing.assert_allclose(summary.probability_in_view, 1)
        head = medians[medians.bodypoint == 1]
        np.testing.assert_allclose(head.median_plot_x_mm, 0)
        np.testing.assert_allclose(head.median_plot_y_mm, 1)
    finally:
        for fig in figures.values():
            plt.close(fig)
