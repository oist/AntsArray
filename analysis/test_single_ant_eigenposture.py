"""Single-cup adaptations preserve alignment, duration weights and missing data."""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from analysis.full_behavior_extract import align_body
from analysis.full_behavior_profiles import state_sums
from analysis.full_behavior_states import feature_sets, fit_normalizer, encode
from analysis.grid_occupancy_utils import recording_context, side_from_track
from analysis.single_ant_eigen_report import COMMON_CELLS
from analysis.full_behavior_sleep import select_quiet_states
from analysis.single_ant_timeline import support_mask


def test_large_ants_require_explicit_geometry_without_changing_normalization():
    skeleton = np.array([[0,0],[12,0],[-20,0],[-45,4],[16,5],[24,12],
                         [33,12],[16,-5],[24,-12],[33,-12]], float)[None]
    small, _, good = align_body(skeleton, .32)
    assert good.all()
    _, _, rejected = align_body(skeleton*7, .32*7)
    assert not rejected.any()
    large, _, accepted = align_body(skeleton*7, .32*7,
                                    axis_bounds=(.4, 5), max_landmark_mm=20)
    assert accepted.all()
    np.testing.assert_allclose(large, small, atol=1e-12)


def test_missing_social_features_cannot_affect_single_models(tmp_path):
    (tmp_path/'analysis_protocol.json').write_text(json.dumps({'social_features':False}))
    rng = np.random.default_rng(5)
    raw = rng.uniform(0, 2, (80,31)); raw[:,29:] = np.nan
    for representation, groups in feature_sets(tmp_path).items():
        assert not {29,30}.intersection(sum(groups, []))
        norm = fit_normalizer(raw, representation, groups)
        encoded = encode(raw, norm)
        changed = raw.copy();changed[:,29:] = 1e6
        np.testing.assert_allclose(encode(changed,norm), encoded)
        assert np.isfinite(encoded).all()


def test_posture_means_keep_duration_weights_without_social_data():
    pose = np.repeat(np.array([1,1,1,9,9])[:,None],20,axis=1)
    raw = np.zeros((2,31));raw[:,24:29] = [[2]*5,[7]*5];raw[:,29:] = np.nan
    counts, shapes, motion = state_sums(pose,np.array([0,3]),np.array([3,2]),
                                        np.array([0,0]),raw,1,social_features=False)
    assert counts[0] == 5
    np.testing.assert_allclose(shapes[0]/5, 21/5)
    np.testing.assert_allclose(motion[0,:5]/5, 20/5)
    assert np.isnan(motion[0,5:]).all()
    with pytest.raises(AssertionError):
        state_sums(pose,np.array([0,3]),np.array([3,2]),np.array([0,0]),raw,1)


def test_single_report_reuses_all_applicable_colony_cells():
    source = Path(__file__).with_name('eigenposture_interactive.py').read_text()
    cells = [c for c in re.split(r'(?m)(?=^# %%)',source) if c.strip()]
    assert len(cells) == 11
    assert tuple(i+1 for i in COMMON_CELLS) == (1,2,3,4,5,6,8,9)
    assert 'Full-recording speed' in cells[7]
    assert 'Individual activity/sleep' in cells[8]


def test_no_quiet_candidate_requires_explicit_reporting_mode(tmp_path):
    (tmp_path/'models').mkdir()
    raw=np.zeros((8,31));raw[:,24:28]=1
    np.savez(tmp_path/'models/dictionary.npz',raw=raw,state=np.repeat([0,1],4))
    with pytest.raises(ValueError,match='No state'):
        select_quiet_states(tmp_path)
    candidates,profiles=select_quiet_states(tmp_path,require_candidate=False)
    assert not len(candidates) and not profiles.quiet_candidate.any()
    assert (profiles.body_peak_median_mm_s > .05).all()


def test_recording_support_masks_gap_and_restarts_trailing_window():
    support=dict(intervals=[(0,5),(8,15)])
    np.testing.assert_array_equal(support_mask(support,15),
        [1,1,1,1,1,0,0,0,1,1,1,1,1,1,1])
    expected=[0,0,1,1,1,0,0,0,0,0,1,1,1,1,1]
    np.testing.assert_array_equal(support_mask(support,15,window_frames=3),expected)
    np.testing.assert_array_equal(support_mask(support,9,frame_min=6,window_frames=3),expected[6:])


def test_single_recording_context_keeps_full_gap_and_date(tmp_path):
    (tmp_path/'per_track').mkdir()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(protocol='manual_cups_colony_block_stitch_v1',
        start_datetime='2026-10-27T21:49:46.275000+09:00', n_frames=3036259, fps=24,
        recording_segments=[dict(block='20261027_block02'),dict(block='20261028_block01')])) )
    result = recording_context(tmp_path, pd.DataFrame())
    assert result['frame_stop'] == 3036259
    assert result['recording_date'] == '20261027'
    assert result['start_clock_seconds'] == 21*3600+49*60+46
    assert side_from_track('TrackID_0001_all_214946_single.parquet',tmp_path) == 'single'
