from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from analysis.compare_behavior_blocks import (
    clock_window, state_counts, probabilities, predict_profiles,
    summarize_positions, scan_positions)


def test_match_clock_across_midnight_and_reject_incomplete_baseline():
    before = datetime(2026, 9, 15, 15, 52, 47)
    after = datetime(2026, 9, 16, 22, 54, 36)
    assert clock_window(before, 100020, after, 43200) == (25309, 68509)
    with pytest.raises(ValueError):
        clock_window(before, 60000, after, 43200)


def test_duration_conserved_at_hour_boundaries_missing_is_not_rest():
    counts = state_counts(np.array([0, 8, 25]), np.array([8, 8, 5]),
                          np.array([0, 1, 0]), [0, 10, 20, 30], 2)
    np.testing.assert_equal(counts, [[8, 2], [0, 6], [5, 0]])
    assert counts.sum() == 21
    assert np.isnan(probabilities([0, 0])).all()
    labels, _ = predict_profiles([[np.nan, np.nan], [1., 0.]], [0, 1], np.eye(2))
    np.testing.assert_equal(labels, [-1, 0])


def test_unknown_seconds_do_not_count_as_inside_and_strict_threshold_is_separate():
    counts = dict(observed=np.array([24, 0, 2]), outside=np.array([12, 0, 2]),
                  aruco_observed=np.array([2, 0, 1]), aruco_outside=np.array([0, 0, 1]))
    row = summarize_positions(counts, 0, 3)
    assert row['position_coverage'] == 2/3
    assert row['outside_fraction'] == .75
    assert row['aruco_outside_fraction'] == .5
    strict = summarize_positions(counts, 0, 3, 12)
    assert strict['position_coverage'] == 1/3
    assert strict['outside_fraction'] == .5


def test_direct_tag_positions_are_independent_of_propagated_track(tmp_path):
    path = tmp_path/'track.parquet'
    pd.DataFrame(dict(Frame=[0, 1, 2, 3], Bodypoint=[0, 0, 0, 1],
        TrackX=[20., 20., np.nan, 20.], TrackY=[0., 0., np.nan, 0.],
        ArucoX=[0., np.nan, 20., 20.], ArucoY=[0., np.nan, 0., 0.])).to_parquet(path)
    counts = scan_positions(dict(n_frames=24, track_path=str(path),
        rectangle=dict(x_min_px=-5, x_max_px=5, y_min_px=-5, y_max_px=5)))
    assert counts['observed'].item() == 2
    assert counts['outside'].item() == 2
    assert counts['aruco_observed'].item() == 2
    assert counts['aruco_outside'].item() == 1


def test_role_bootstrap_uses_same_class_reference_and_clock_hours():
    from analysis.report_behavior_transition import bootstrap_pair, spatial_bootstrap
    before = np.tile([100., 0.], (12, 1))
    after = np.tile([0., 100.], (12, 1))
    reference = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    result = bootstrap_pair(before, after, reference, np.array([1, 1, 0, 0]),
                            nurse=1, forager=0, draws=20)
    np.testing.assert_allclose(result, np.tile([0., 1.], (20, 1)))
    # Equal outside share with different amounts of observation is no change.
    intervals = spatial_bootstrap(np.tile([10., 100.], (12, 1)),
                                 np.tile([5., 50.], (12, 1)), draws=20)
    np.testing.assert_allclose(intervals, 0)


def test_end_to_end_roles_do_not_mix_colonies_or_treat_one_class_as_nurses(tmp_path):
    import json
    from analysis.report_behavior_transition import generate_tables
    ants = [('left:001', 1), ('left:002', 1), ('left:003', 0),
            ('left:004', 0), ('right:001', 0), ('right:002', 0)]
    base, identity, profiles, positions = [], [], [], []
    for ant,label in ants:
        side, number = ant.split(':'); number = int(number)
        base.append(dict(ant=ant, side=side, track_id=number, ant_class=label))
        identity.append(dict(ant=ant, side=side, track_id=number, selected_before=True,
            selected_after=True, detection_fraction_before=.8,detection_fraction_after=.8,in_comparison=True,presence='both'))
        for phase in ('before','after'):
            for scope in ('full','matched','hour_00','hour_01'):
                outside = .05 if side=='left' and label==1 else .7
                p = [.9, .1] if label==1 else [.1, .9]
                if ant=='left:001' and phase=='after': p=[.1,.9]; outside=.7
                for variant in ('standard','without_contacts'):
                    profiles.append(dict(ant=ant,side=side,track_id=number,phase=phase,scope=scope,variant=variant,
                        available_seconds=3600,usable_seconds=3000,behavior_coverage=3000/3600,
                        count_S0=p[0]*36000,count_S1=p[1]*36000,S0=p[0],S1=p[1]))
                positions.append(dict(ant=ant,side=side,track_id=number,phase=phase,scope=scope,
                    available_seconds=3600,observed_seconds=3000,position_coverage=3000/3600,
                    outside_seconds=outside*3000,outside_fraction=outside,
                    aruco_outside_fraction=outside,strict_outside_fraction=outside,n_completed_trips=2))
    for name,rows in [('baseline_ant_classes',base),('identity_and_coverage',identity),
                      ('state_profiles',profiles),('position_and_trip_metrics',positions)]:
        pd.DataFrame(rows).to_csv(tmp_path/(name+'.csv'),index=False)
    (tmp_path/'comparison_manifest.json').write_text(json.dumps({}))
    result, *_ = generate_tables(tmp_path,draws=20)
    result=result.set_index('ant')
    assert result.loc['left:001','matched_nurse_to_forager']
    assert result.loc['left:001','corroborated_outside_increase']
    assert not result.loc['right:001','baseline_nurse_like']
    assert not result.loc['right:001','matched_nurse_to_forager']
    assert not result.loc['left:002','matched_nurse_to_forager']
