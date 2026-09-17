import numpy as np
import pandas as pd

from analysis.long_timescale_cluster_switching import neighbor_vote, transition_summary


def test_leave_one_out_does_not_recover_isolated_label_by_self_match():
    d = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
    labels = np.array(["left_1", "left_0", "left_0"])
    assert neighbor_vote(d[:1], labels, k=2)[0]["prediction"] == "left_1"
    result = neighbor_vote(d, labels, k=2, excluded=np.arange(3))[0]
    assert result["prediction"] == "left_0"
    assert 0 not in result["indices"]


def test_exact_matches_share_weight_and_distant_majority_cannot_overrule():
    result = neighbor_vote([[0., 0., 1., 2., 3.]], ["a", "a", "b", "b", "b"])[0]
    assert result["prediction"] == "a" and result["vote"] == 1
    np.testing.assert_allclose(result["weights"], [.5, .5, 0, 0, 0])


def test_distance_weighting_and_deterministic_ties():
    result = neighbor_vote([[.1, 1, 1]], ["a", "b", "b"])[0]
    assert result["prediction"] == "a"
    np.testing.assert_allclose(result["vote"], 10/12)
    assert neighbor_vote([[1., 1.]], ["b", "a"])[0]["prediction"] == "a"


def test_switch_denominator_only_includes_same_labeled_ants_with_assignments():
    common = dict(reference_index=1, side="left", block_index=2, supported=False,
                  in_reference_range=True, supported_switch=False, example_switch=False)
    rows = [dict(common, reference_cluster="left_0", prediction="left_1", switch=True),
            dict(common, reference_cluster="left_0", prediction="left_0", switch=False),
            dict(common, reference_cluster=None, prediction="left_1", switch=False),
            dict(common, reference_cluster="left_0", prediction=None, switch=False)]
    result = transition_summary(pd.DataFrame(rows)).iloc[0]
    assert result.all_future_profiles == 4
    assert result.same_ant_assigned == 2
    assert result.raw_switches == 1


def test_later_cluster_ids_cannot_train_or_change_fixed_reference(monkeypatch):
    from analysis import long_timescale_cluster_switching as cs
    monkeypatch.setattr(cs.go, "umap_embedding", lambda features, **kwargs: features[:, :2])
    rows, maps = [], {}
    for side in ("left", "right"):
        for i in range(12):
            ant = f"{side}:{i:03d}"
            values = [.90 - i*.001, .10 + i*.001] if i < 6 else [.10 + i*.001, .90 - i*.001]
            key = "0|" + ant
            maps[key] = np.array([values])
            rows.append(dict(profile_key=key, block_index=0, side=side, ant=ant,
                             original_selected=True, original_cluster=f"{side}_{int(i>=6)}",
                             occupancy_sum=1., coverage=.9, detected_hours=2., colony_percent=80.))
        # Same ant now exactly matches a different original cluster. Its later
        # independent cluster ID deliberately says the opposite.
        key = "1|" + side + ":000"
        maps[key] = maps["0|" + side + ":007"].copy()
        rows.append(dict(profile_key=key, block_index=1, side=side, ant=side+":000",
                         original_selected=True, original_cluster=side+"_0",
                         occupancy_sum=1., coverage=.9, detected_hours=3., colony_percent=20.))
    profiles = pd.DataFrame(rows)
    a, ref, neighbors, _ = cs.classify_reference(profiles, maps, 0)
    assert a.switch.all()
    assert a.prediction.tolist() == ["left_1", "right_1"]
    assert neighbors.neighbor_ant.str.endswith(":007").any()
    changed = profiles.copy()
    changed.loc[changed.block_index.eq(1), "original_cluster"] = "arbitrary_later_label"
    a2, ref2, _, _ = cs.classify_reference(changed, maps, 0)
    pd.testing.assert_frame_equal(ref, ref2)
    pd.testing.assert_series_equal(a.prediction, a2.prediction)


def test_matched_behavior_weights_shared_clock_slots_equally():
    from analysis.long_timescale_cluster_switching import matched_behavior
    profiles = pd.DataFrame([dict(source_block=str(b), ant="left:001", block_index=b, original_selected=True) for b in (1, 2, 3)])
    rows = []
    for b in (1, 2, 3):
        # One slot has 10 times the observation exposure, but both clock slots
        # must get equal weight after within-slot observation weighting.
        for clock, n, value in [(1., 100., 0.), (2., 10., 100.)]:
            rows.append(dict(source_block=str(b), ant="left:001", clock_hour=clock,
                             bin_minutes=30, n_expected_frames=n, colony_percent=value,
                             mean_speed_mm_s=value, sleep_percent=value, position_coverage=1.,
                             speed_coverage=1., sleep_coverage=1.))
    rows.append(dict(rows[0], source_block="3", clock_hour=3., colony_percent=1000.))
    result = matched_behavior(pd.DataFrame(rows), profiles, [1])
    np.testing.assert_allclose(result.value, 50.)
    assert result.shared_clock_bins.eq(2).all()
