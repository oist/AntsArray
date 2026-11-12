import numpy as np

from analysis.single_ant_sleep_speed_report import (
    ant_weights, bout_confusions, confusion_from_hist, cv_counts, extract, long_bout_mask, metrics,
)


def test_original_state_target_ignores_unknown_even_when_speed_is_low():
    hist = np.zeros((7, 3), int)
    hist[0] = [2, 1, 0]
    hist[1] = [0, 1, 0]
    hist[2] = [1, 2, 0]
    hist[5] = [0, 0, 3]
    hist[6] = [10000, 0, 0]  # Unclassified time cannot become a validation negative.
    counts = confusion_from_hist(hist)
    assert np.array_equal(counts[0], [2, 1, 2, 5])
    assert np.array_equal(counts[-1], [4, 6, 0, 0])
    assert metrics(counts[0])['balanced_accuracy'] == (2 / 4 + 5 / 6) / 2


def test_bouts_break_at_missing_observation_or_recording_gap():
    valid_quiet = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], bool)
    expected = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], bool)
    assert np.array_equal(long_bout_mask(valid_quiet, 3), expected)


def test_ant_weighting_prevents_longer_observations_dominating():
    counts = np.array([[1, 1, 1, 1], [100, 100, 100, 100]])
    assert np.array_equal(ant_weights(counts), np.full((2, 4), .25))


def test_held_out_ant_does_not_select_its_own_threshold():
    counts = np.array([[[9, 1, 1, 9], [1, 9, 9, 1]],
                       [[1, 9, 9, 1], [9, 1, 1, 9]]])
    thresholds, heldout = cv_counts(counts)
    assert np.array_equal(thresholds, [1, 0])
    assert np.array_equal(heldout, [[1, 9, 9, 1], [1, 9, 9, 1]])


def test_s012_target_includes_high_numbered_other_states_and_excludes_unknown():
    hist = np.zeros((9, 2), int)
    hist[0] = [1, 0]
    hist[2] = [2, 1]
    hist[3] = [1, 0]
    hist[6] = [2, 0]
    hist[7] = [0, 3]
    hist[8] = [10000, 0]
    counts = confusion_from_hist(hist, [0, 1, 2])
    assert np.array_equal(counts[0], [3, 3, 1, 3])
    assert np.array_equal(counts[-1], [4, 6, 0, 0])


def test_bout_counts_match_masks_and_split_adjacent_recordings():
    quiet = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], bool)
    positive = np.arange(len(quiet)) % 2 == 0
    negative = ~positive
    for duration, counts in zip([0, 2, 4], bout_confusions(quiet, positive, negative, [0, 2, 4])):
        prediction = long_bout_mask(quiet, duration)
        assert np.array_equal(counts, [(prediction & positive).sum(), (prediction & negative).sum(),
                                      (~prediction & positive).sum(), (~prediction & negative).sum()])
    split = bout_confusions(np.ones(8, bool), np.ones(8, bool), np.zeros(8, bool), [6], boundaries=[4])
    assert np.array_equal(split[0], [0, 0, 8, 0])


def test_extract_relabels_only_selected_states_and_preserves_source_and_gaps(tmp_path):
    import hashlib
    import json
    import pandas as pd
    dataset = tmp_path / 'dataset'; source = dataset / 'analysis/sleep_s01_exclusion'
    source.mkdir(parents=True)
    manifest = dict(n_frames=12, fps=1, start_datetime='2026-11-05T00:00:00+09:00',
                    recording_segments=[dict(block='a', frame_offset=0, num_frames=5),
                                        dict(block='b', frame_offset=7, num_frames=5)])
    (dataset / 'manifest.json').write_text(json.dumps(manifest))
    (source / 'definition.json').write_text(json.dumps(dict(model_selection=dict(n_states=8), model_supported=False)))
    pd.DataFrame([dict(CupID=i, TrackID=i, ArucoID=54, Eligible=True) for i in [1, 2]]).to_csv(dataset / 'cup_identities.csv', index=False)
    pd.DataFrame([dict(CupID=i, TrackID=i, IdentityStatus='matched') for i in [1, 2]]).to_csv(source / 'per_ant_summary.csv', index=False)
    observed = np.array([0, 1, 2, 3, 6, -1, -1, 7, -1, 2, 6, 0], np.int8)
    originals = []
    for i in [1, 2]:
        stem = f'TrackID_{i:04d}_all_single'
        labels = source / 'per_track' / stem; labels.mkdir(parents=True)
        path = labels / 'labels.npz'
        np.savez_compressed(path, observed=observed, category=np.where(observed >= 0, np.isin(observed, [0, 1]), -1),
                            evidence=(observed >= 0).astype(np.uint8))
        originals.append((path, hashlib.sha256(path.read_bytes()).hexdigest()))
        speed = dataset / 'analysis/speed_vectors/per_track' / stem; speed.mkdir(parents=True)
        np.save(speed / 'speed_mm_s.npy', [0, .01, .1, .3, .04, np.nan, .2, .05, .02, .03, .04, .9])
    registry = tmp_path / 'registry.tsv'
    registry.write_text('cup\ttag_id\tcolony\trole\n1\t54\t#14\tnurse\n2\t54\t#10-2\tforager\n')
    output = tmp_path / 'report'
    extract(dataset, source, output, sleep_states=[0, 1, 2], registry=registry, bin_seconds=4)
    summary = pd.read_csv(output / 'per_ant_summary.csv')
    assert summary.CupID.tolist() == [2, 1]
    assert np.allclose(summary.SleepPercent, 50)
    assert np.allclose(summary.NotSleepPercent, 40)
    assert np.allclose(summary.AmbiguousPercent, 10)
    with np.load(output / 'per_track/TrackID_0001_all_single/labels.npz') as labels:
        np.testing.assert_array_equal(labels['observed'], observed)
        assert labels['category'][2] == 1 and labels['category'][7] == 0
        assert (labels['category'][5:7] == -1).all() and labels['category'][8] == -1
    with np.load(output / 'summaries.npz') as z:
        assert z['hist'].shape == (2, 2, 9, 4001)
        counts = confusion_from_hist(z['hist'].sum(axis=1), [0, 1, 2])
        np.testing.assert_array_equal(counts[:, -1], [[5, 4, 0, 0]] * 2)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == old for path, old in originals)
