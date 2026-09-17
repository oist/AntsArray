"""Regression tests for frame alignment, new contacts, and pre-contact selection."""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis import interaction_analysis_utils as ia
from analysis import return_sleep_utils as rs
from analysis import sleep_motion_analysis_utils as sma


def test_cluster_sleep_equal_ant_weight_and_unknown(tmp_path):
    rows = []
    for side, ant, state, offset in [
        ("left", 1, [1, 1], 2), ("left", 2, [0, 0, 0, 0], 0),
        ("right", 1, [-1, -1, -1, -1], 0),
    ]:
        path = tmp_path / f"{side}_{ant}.npy"
        np.save(path, np.array(state, dtype=np.int8))
        rows.append({"side": side, "track_id": ant, "cluster_id": side+"_0", "fps": 1,
                     "state_path": str(path), "frame_min": offset, "frame_max": offset+len(state)-1})
    summary, bins = sma.cluster_sleep_timeseries(pd.DataFrame(rows), bin_seconds=4, min_classified_fraction=0.5)
    assert summary.loc[summary.side == "left", "mean_sleep_fraction"].item() == 0.5
    assert summary.loc[summary.side == "right", "n_ants_with_data"].item() == 0
    assert np.isnan(summary.loc[summary.side == "right", "mean_sleep_fraction"].item())
    assert bins.loc[(bins.side == "left") & (bins.track_id == 1), "classified_fraction"].item() == 0.5
    np.testing.assert_array_equal(sma.sample_dense(np.array([1, 2]), 10, np.array([9, 10, 11, 12]), missing=-1), [-1, 1, 2, -1])


def test_unknown_position_cannot_manufacture_return():
    settings = rs.ReturnSettings(fps=1, min_outside_seconds=5, min_colony_anchor_seconds=3)
    inside = np.r_[np.ones(5), np.zeros(8), np.ones(5), -1, np.zeros(8), -1, np.ones(5)].astype(np.int8)
    context = {"inside": inside, "resource_seconds": np.zeros(len(inside))}
    clusters = pd.DataFrame({"side": ["left"], "track_id": [1], "cluster_id": ["left_0"]})
    returns = rs.extract_returns({("left", 1): context}, clusters, settings)
    assert len(returns) == 1
    assert returns.return_frame.item() == 13


def test_pair_bout_merges_reciprocal_and_chunk_boundary(tmp_path):
    chunks = []
    for number, frames in enumerate(([0, 1, 7, 8, 9], [0, 1, 5, 6])):
        data = pd.DataFrame({"Frame": np.repeat(frames, 2), "antenna_track_id": [1, 2]*len(frames),
                             "body_track_id": [2, 1]*len(frames)})
        path = tmp_path / f"chunk{number}.parquet"
        data.to_parquet(path)
        chunks.append(ia.InteractionChunk(path, path, str(number), "left", 0, number*10, number*10, 10, 10))
    bouts, _ = rs.load_contact_bouts(chunks, rs.ReturnSettings(fps=1), tmp_path)
    assert bouts.start_frame.tolist() == [0, 7, 15]
    assert bouts.end_frame.tolist() == [1, 11, 16]
    assert bouts.is_new_onset.tolist() == [False, True, True]
    assert bouts.onset_direction.tolist() == [3, 3, 3]
    assert bouts.n_detection_frames.tolist() == [2, 5, 2]
    filtered, _ = rs.load_contact_bouts(chunks, rs.ReturnSettings(fps=1, contact_min_detection_frames=3), tmp_path)
    assert filtered.start_frame.tolist() == [7]


def test_return_uses_first_raw_crossing_not_majority_bin_or_confirmation():
    settings = rs.ReturnSettings(fps=24, min_outside_seconds=5, min_colony_anchor_seconds=3)
    inside = np.r_[np.ones(5), np.zeros(8), np.ones(5)].astype(np.int8)
    context = {"inside": inside, "resource_seconds": np.zeros(len(inside)),
               "crossing_frames": np.array([121, 302, 306, 314]),
               "crossing_previous_frames": np.array([120, 301, 305, 313]),
               "crossing_inside": np.array([0, 1, 0, 1])}
    clusters = pd.DataFrame({"side": ["left"], "track_id": [1], "cluster_id": ["left_0"]})
    result = rs.extract_returns({("left", 1): context}, clusters, settings)
    assert result.return_frame.item() == 302
    assert result.binned_return_frame.item() == 312
    assert result.last_outside_frame.item() == 301
    assert result.crossing_uncertainty_seconds.item() == 1 / 24
    context["crossing_previous_frames"][1] = 290
    assert rs.extract_returns({("left", 1): context}, clusters, settings).empty
    # A much earlier brief entry, followed by another long excursion, cannot
    # manufacture minutes of apparent post-return residence outside the colony.
    context["crossing_frames"] = np.array([121, 180, 182, 314])
    context["crossing_previous_frames"] = np.array([120, 179, 181, 313])
    result = rs.extract_returns({("left", 1): context}, clusters, settings)
    assert result.return_frame.item() == 314
    assert result.exit_frame.item() == 182


def test_undirected_skeleton_bouts_have_no_invented_contact_direction(tmp_path):
    chunks = []
    for number, frames in enumerate(([7, 8, 9], [0, 1, 5])):
        data = pd.DataFrame(dict(Frame=frames, ant_a=1, ant_b=2, distance_mm=.03))
        path = tmp_path / f"skeleton{number}.parquet"
        data.to_parquet(path)
        chunks.append(ia.InteractionChunk(path, path, str(number), "left", 0, number*10, number*10, 10, 10))
    bouts, _ = rs.load_contact_bouts(chunks, rs.ReturnSettings(fps=1), tmp_path)
    assert bouts.start_frame.tolist() == [7, 15]
    assert bouts.end_frame.tolist() == [11, 15]
    assert bouts.n_detection_frames.tolist() == [5, 1]
    assert bouts.onset_direction.eq(0).all()


def test_pool_returns_and_keep_only_three_figure_rows():
    import matplotlib.pyplot as plt

    context = _context(100)
    context.update(contact_onsets=np.zeros(100), interaction_covered=np.ones(100, bool))
    returns = pd.DataFrame([dict(return_id=str(i), side="left", track_id=1, cluster_id="left_0",
                                return_frame=20, inside_stop_frame=100, resource_visit=bool(i)) for i in range(2)])
    curves, latency = rs.return_activity_curves(returns, {("left", 1): context}, rs.ReturnSettings(fps=1),
                                               pre_seconds=10, post_seconds=20)
    assert curves.trip_type.unique().tolist() == ["All returns"]
    assert latency.trip_type.unique().tolist() == ["All returns"]
    fig, _ = rs.plot_return_curves(curves)
    assert len(fig.axes) == 6
    plt.close(fig)


def _context(n=50):
    return {"inside": np.ones(n, dtype=np.int8), "state": np.ones(n, dtype=np.int8),
            "sleep_age_seconds": np.arange(1, n+1, dtype=float), "x_mm": np.zeros(n), "y_mm": np.zeros(n),
            "density": np.ones(n), "body_speed": np.zeros(n), "antenna_speed": np.zeros(n)}


def test_recipient_selection_uses_sleep_before_not_after_contact(tmp_path):
    state_path = tmp_path / "state.npy"
    np.save(state_path, np.r_[np.ones(12), np.zeros(38)].astype(np.int8))
    tracks = pd.DataFrame({"side": ["left"], "track_id": [1], "cluster_id": ["left_0"],
                           "state_path": [str(state_path)], "frame_min": [0], "frame_max": [49]})
    contexts = {("left", 1): _context(), ("left", 2): _context()}
    bouts = pd.DataFrame({"side": ["left"], "ant_a": [1], "ant_b": [2], "start_frame": [12],
                          "end_frame": [15], "onset_direction": [2], "is_new_onset": [True]})
    settings = rs.ReturnSettings(fps=1)
    focal = rs.attach_contact_context(contexts, bouts, {"left": np.ones(50, bool)}, settings)
    returns = pd.DataFrame({"side": ["left"], "track_id": [2], "return_frame": [5],
                            "inside_stop_frame": [50], "return_id": ["r1"], "resource_visit": [True]})
    events = rs.eligible_sleeping_contacts(tracks, contexts, bouts, returns, settings)
    assert len(events) == 1
    assert events.condition.item() == "Recent return contact"
    assert events.recipient_body_contact.item()
    events["match_id"] = events.event_id
    outcomes = rs.waking_outcomes(events, tracks, contexts, focal, settings)
    assert outcomes.wake_observed.item()
    assert outcomes.duration_seconds.item() == 2

    bouts["onset_direction"] = 0
    undirected = rs.eligible_sleeping_contacts(tracks, contexts, bouts, returns, settings)
    assert len(undirected) == 1
    assert undirected.recipient_body_contact.isna().all()


def test_controls_allow_future_contacts_and_exclude_current_contacts():
    settings = rs.ReturnSettings(fps=24)
    context = _context()
    contexts = {("left", 1): context}
    # Candidate t=12 seconds: a contact one frame in the future must not
    # disqualify this baseline time. A contact already spanning t=17 must.
    bouts = pd.DataFrame({"side": ["left", "left"], "ant_a": [1, 1], "ant_b": [2, 3],
                          "start_frame": [12*24+1, 17*24-1], "end_frame": [12*24+2, 17*24+1],
                          "onset_direction": [1, 1], "is_new_onset": [True, True]})
    rs.attach_contact_context(contexts, bouts, {"left": np.ones(50, bool)}, settings)
    pool = rs._no_contact_candidates(context, settings)
    assert 12*24 in pool.frame.to_list()
    assert 17*24 not in pool.frame.to_list()


def test_no_matching_controls_is_an_empty_result():
    events = pd.DataFrame([{"side": "left", "track_id": 1, "cluster_id": "left_0", "frame": 20,
                            "event_id": "event", "condition": "Recent return contact", "resource_visit": False,
                            "sleep_bout_start_frame": 0, "x_mm": 100, "y_mm": 100, "density": 1,
                            "body_speed": 0, "antenna_speed": 0, "sleep_age_seconds": 20}])
    context = _context()
    context.update({"contact_active": np.zeros(50, bool), "interaction_covered": np.ones(50, bool),
                    "contact_start_frames": np.array([], dtype=int), "contact_end_frames": np.array([], dtype=int)})
    triggers, diagnostics = rs.match_sleeping_controls(events, {("left", 1): context}, rs.ReturnSettings(fps=1))
    assert triggers.empty
    assert not diagnostics.matched.item()


def test_next_contact_censors_before_later_wake(tmp_path):
    state_path = tmp_path / "state.npy"
    np.save(state_path, np.r_[np.ones(20), np.zeros(30)].astype(np.int8))
    tracks = pd.DataFrame({"side": ["left"], "track_id": [1], "cluster_id": ["left_0"],
                           "state_path": [str(state_path)], "frame_min": [0], "frame_max": [49]})
    context = _context()
    context["interaction_covered"] = np.ones(50, bool)
    focal = {("left", 1): pd.DataFrame({"start_frame": [12, 15], "is_new_onset": [True, True]})}
    triggers = pd.DataFrame({"side": ["left"], "track_id": [1], "cluster_id": ["left_0"],
                             "frame": [12], "match_id": ["test"], "condition": ["Recent return contact"]})
    result = rs.waking_outcomes(triggers, tracks, {("left", 1): context}, focal, rs.ReturnSettings(fps=1))
    assert not result.wake_observed.item()
    assert result.duration_seconds.item() == 3
    assert result.censor_reason.item() == "next contact"


def test_no_returns_still_produces_exportable_tables():
    context = _context()
    context["resource_seconds"] = np.zeros(50)
    clusters = pd.DataFrame({"side": ["left"], "track_id": [1], "cluster_id": ["left_0"]})
    settings = rs.ReturnSettings(fps=1)
    contexts = {("left", 1): context}
    returns = rs.extract_returns(contexts, clusters, settings)
    assert returns.empty
    curves, latency = rs.return_activity_curves(returns, contexts, settings)
    effects = rs.return_early_late_effects(curves)
    assert curves.empty and latency.empty and effects.empty
    assert {"duration_seconds", "sleep_observed"} <= set(latency)
    assert {"metric", "n_returns", "mean"} <= set(effects)
