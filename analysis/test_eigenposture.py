import numpy as np
import pandas as pd
import joblib

from analysis.eigenposture_features import (
    coordinates,
    windows,
    clip_moments,
    weighted_basis,
    project,
    hourly,
)
from analysis.eigenposture_analysis import ProfileAxis, select, mixture
from analysis.eigenposture_report import straightness


def fixture():
    angles = np.array([0, np.pi, 0, np.pi / 2, 0, 0, -np.pi / 2, 0])
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    lengths = np.ones((2, 60, 9)) * 0.3
    lengths[..., 0] = 0.6
    return dict(
        shape=np.tile(directions.reshape(1, 1, 16), (2, 60, 1)),
        lengths_mm=lengths,
        cameras=np.zeros((2, 60)),
        position_cameras=np.zeros((2, 60)),
        duplicate_frame=np.zeros((2, 60), bool),
        velocity_mm_s=np.tile([1.0, -0.2], (2, 28, 1)),
    )


def test_coordinates_preserve_bending_and_length_instead_of_inserting_descriptor():
    z = fixture()
    x = coordinates(z["shape"], z["lengths_mm"], 0.6)
    expected = [0.5, 0, 0.5, 0.5, 1, 0.5, 0.5, 0, 0.5, -0.5, 1, -0.5]
    np.testing.assert_allclose(x[0, 0], expected, atol=1e-7)
    z["lengths_mm"][..., 3:] *= 2
    np.testing.assert_allclose(
        coordinates(z["shape"], z["lengths_mm"], 0.6), x * 2, atol=1e-7
    )


def test_local_camera_transition_and_missing_landmark_do_not_reject_whole_clip():
    z = fixture()
    z["cameras"][0, 20] = 1
    z["shape"][1, 20, 4] = np.nan
    x, v, d, c = windows(z, 0.6)
    m, s, ds, vel, counts = clip_moments(x, v, d)
    assert (counts[:, 0] >= 8).all() and (counts[:, 0] < 28).all()
    assert np.isfinite(m).all() and np.isfinite(ds).all()
    np.testing.assert_allclose(vel[:, 0], 1)
    np.testing.assert_allclose(vel[:, 1], -0.2, atol=1e-7)
    np.testing.assert_allclose(vel[:, 3], 0.2, atol=1e-7)


def test_pca_has_equal_colony_weight_despite_unequal_ant_numbers():
    means = np.zeros((12, 12))
    means[:4, 0] = -1
    means[4:, 0] = 1
    moments = np.einsum("ni,nj->nij", means, means) + np.eye(12)[None] * 0.001
    b = weighted_basis(
        means, moments, np.array(["left"] * 4 + ["right"] * 8), np.ones(12, bool)
    )
    np.testing.assert_allclose(b["mean"], 0, atol=1e-12)
    assert b["rank"] == 1
    assert abs(b["components"][0, 0]) > 0.999


def test_projection_moments_match_direct_sample_differences():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 28, 12))
    v = rng.normal(size=(3, 28, 2))
    d = np.diff(x, axis=1) * 12
    m, s, ds, vel, n = clip_moments(x, v, d)
    c = np.linalg.qr(rng.normal(size=(12, 12)))[0].T
    basis = dict(mean=np.zeros(12), components=c, rank=4)
    p = project(m, ds, vel, basis)
    np.testing.assert_allclose(p[:, 4:8], (x @ c[:4].T).mean(axis=1), atol=1e-6)
    np.testing.assert_allclose(
        p[:, 8:], np.sqrt(np.mean((d @ c[:4].T) ** 2, axis=1)), rtol=1e-6
    )


def test_hourly_observation_threshold_and_minute_parity():
    x = np.full((2880, 2), np.nan)
    x[:5] = [1, 2]
    h, n = hourly(x)
    e, _ = hourly(x, 0)
    o, _ = hourly(x, 1)
    np.testing.assert_allclose(h[0], [1, 2])
    np.testing.assert_allclose(e[0], [1, 2])
    assert np.isnan(o).all() and np.isnan(h[1:]).all()


def test_frozen_profile_transform_and_portable_serialization(tmp_path):
    rng = np.random.default_rng(12)
    names = np.array(["forward_mean", "posture_pc1_mean", "posture_pc1_rate"])
    families = np.array(["velocity", "posture", "dynamics"])
    x = rng.normal(size=(30, 9))
    x[:, [2, 5, 8]] = np.abs(x[:, [2, 5, 8]])
    t = ProfileAxis().fit(x, names, families, ("velocity", "posture", "dynamics"))
    path = tmp_path / "axis.joblib"
    joblib.dump(t, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(t.transform(x[:1]), restored.transform(x)[:1])
    assert restored.__class__.__module__ == "analysis.eigenposture_analysis"


def test_k_selection_can_reject_two_classes():
    table = pd.DataFrame(
        dict(
            k=[1, 2, 3],
            bic=[100, 80, 90],
            smallest_group=[40, 20, 10],
            minimum_group_size=[4] * 3,
            bootstrap_median_ari=[np.nan, 0.5, 0.4],
            split_minute_ari=[np.nan, 0.95, 0.9],
        )
    )
    assert select(table) == 1
    table.loc[1, "bootstrap_median_ari"] = 0.9
    assert select(table) == 2


def test_posthoc_straightness_from_coordinates():
    x = np.array([1, 0, 2, 0, 3, 0, 1, 0, 2, 0, 3, 0], float)[None]
    np.testing.assert_allclose(straightness(x), 1.0)
    bent = np.array([1, 0, 1, 1, 2, 1, 1, 0, 1, -1, 2, -1], float)[None]
    np.testing.assert_allclose(straightness(bent), np.sqrt(5) / 3)
