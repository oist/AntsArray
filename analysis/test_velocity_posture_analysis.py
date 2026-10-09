import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from analysis.velocity_posture_analysis import (
    CommonPhenotypeTransform,
    evaluate,
    matched_accuracy,
)
from analysis.velocity_posture_mixture import mixture, select
from analysis.velocity_posture_report import (
    bootstrap_difference,
    cliffs_delta,
    conditional_posture,
)
from analysis.velocity_posture_features import FEATURES


def test_transform_of_heldout_ant_is_independent_of_other_heldout_ants():
    rng = np.random.default_rng(20)
    x = rng.uniform(0.1, 1, (40, 45))
    t = CommonPhenotypeTransform().fit(x, ("velocity", "posture", "dynamics"))
    center = t.center.copy()
    heldout = rng.uniform(0.1, 2, (9, 45))
    np.testing.assert_allclose(t.transform(heldout[:1]), t.transform(heldout)[:1])
    np.testing.assert_array_equal(t.center, center)
    assert t.transform(heldout).shape == (9, 1)
    assert t.keep.reshape(3, 15)[:, 7:].any()  # Actual posture/dynamics inputs.


def test_unequal_variance_mixture_recovers_separated_synthetic_populations():
    rng = np.random.default_rng(10)
    z = np.r_[rng.normal(-5, 0.2, 30), rng.normal(4, 1, 30)][:, None]
    fitted = mixture(z, 2, covariance="full")
    assert adjusted_rand_score(np.repeat([0, 1], 30), fitted.predict(z)) == 1
    assert fitted.bic(z) < mixture(z, 1, covariance="full").bic(z)


def test_k_is_not_forced_to_two_and_requires_stability():
    table = pd.DataFrame(
        dict(
            k=[1, 2, 3],
            bic=[200, 180, 170],
            smallest_group=[40, 20, 10],
            minimum_group_size=[4] * 3,
            bootstrap_ari_median=[np.nan, 0.95, 0.92],
            split_minute_ari=[np.nan, 0.9, 0.8],
        )
    )
    assert select(table, 0) == 3
    table.loc[2, "bic"] = 179
    assert select(table, 0) == 2  # Parsimony within two BIC units.
    table.loc[1:, "bootstrap_ari_median"] = 0.2
    assert select(table, 0) == 1


def test_spatial_evaluation_aligns_labels_without_changing_assignments(tmp_path):
    assignments = pd.DataFrame(
        dict(
            ant=["a", "b", "c", "d"],
            side=["left"] * 4,
            track_id=[1, 2, 3, 4],
            family=["joint"] * 4,
            activity_cluster=[0, 0, 1, 1],
            selected_k=[2] * 4,
        )
    )
    target = tmp_path / "activity_assignments.csv"
    assignments.to_csv(target, index=False)
    before = target.read_bytes()
    frozen = dict(file_sha256={target.name: hashlib.sha256(before).hexdigest()})
    (tmp_path / "UNSUPERVISED_FROZEN.json").write_text(json.dumps(frozen))
    spatial = tmp_path / "reference.csv"
    pd.DataFrame(
        dict(
            side=["left"] * 4,
            TrackID=[1, 2, 3, 4],
            cluster_id=["left_1", "left_1", "left_0", "left_0"],
        )
    ).to_csv(spatial, index=False)
    evaluate(tmp_path, spatial)
    result = pd.read_csv(tmp_path / "assignments_with_spatial_comparison.csv")
    assert result.spatial_agreement.all()
    assert target.read_bytes() == before
    assert matched_accuracy([0, 0, 1, 1], [1, 1, 0, 0])[0] == 1


def test_conditional_posture_keeps_ant_as_the_observation_unit():
    x = np.full((2, 2880, len(FEATURES)), np.nan)
    names = [f[0] for f in FEATURES]
    for ant, n, value in [(0, 10, 0.4), (1, 100, 0.8)]:
        x[ant, :n, names.index("speed")] = 0.15
        x[ant, :n, names.index("antenna_extension")] = value
    qc = pd.DataFrame(dict(ant=["a", "b"]))
    a = pd.DataFrame(dict(ant=["a", "b"], side=["left"] * 2, spatial_cluster=[0, 1]))
    result, _ = conditional_posture(x, qc, a)
    assert len(result) == 2
    np.testing.assert_allclose(result.value, [0.4, 0.8])
    assert result.n_clips.tolist() == [10, 100]


def test_effect_signs_are_group_one_minus_zero():
    a = np.array([0, 1, 2])
    b = np.array([5, 6, 7])
    delta, lo, hi = bootstrap_difference(a, b)
    assert delta == 5 and lo > 0 and hi >= lo
    assert cliffs_delta(a, b) == 1
