import numpy as np

from analysis.velocity_posture_features import (
    FEATURES,
    clip_measurements,
    hourly_measurements,
    posture_geometry,
)


def straight_ant(n=2):
    angles = np.array([0, np.pi, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5])
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    lengths = np.full((n, 60, 9), 0.3)
    lengths[..., 0] = 0.5
    return dict(
        shape=np.tile(directions.reshape(1, 1, 16), (n, 60, 1)),
        lengths_mm=lengths,
        cameras=np.zeros((n, 60)),
        position_cameras=np.zeros((n, 60)),
        duplicate_frame=np.zeros((n, 60), bool),
        velocity_mm_s=np.tile([1.0, 0.25], (n, 28, 1)),
    )


def test_physical_geometry_and_velocity():
    z = straight_ant()
    geometry = posture_geometry(z["shape"].reshape(2, 60, 8, 2), z["lengths_mm"])
    np.testing.assert_allclose(geometry[..., :2], 0, atol=1e-12)
    np.testing.assert_allclose(geometry[..., 2], np.degrees(1.0))
    np.testing.assert_allclose(geometry[..., 3], 1.0)
    np.testing.assert_allclose(geometry[..., 4], 0.0)
    values, _ = clip_measurements(z)
    assert values.shape == (2, len(FEATURES))
    np.testing.assert_allclose(
        values[:, :3], np.tile([1.0, 0.25, np.hypot(1.0, 0.25)], (2, 1)), rtol=1e-6
    )
    np.testing.assert_allclose(values[:, -3:], 0, atol=1e-12)


def test_local_missingness_and_camera_changes_are_not_whole_clip_rejections():
    z = straight_ant()
    z["shape"][0, 20] = np.nan
    z["cameras"][1, 20] = 1
    values, quality = clip_measurements(z)
    assert np.isfinite(values).all()
    assert 8 < quality["pose_samples"][0] < 28
    assert 8 < quality["pose_samples"][1] < 28
    assert quality["velocity_samples"][1] < 28
    z["shape"][:] = np.nan
    missing, _ = clip_measurements(z)
    assert np.isnan(missing[:, 7:]).all()


def test_hourly_counts_and_missing_hours():
    x = np.full((2880, len(FEATURES)), np.nan)
    x[:5] = 2
    x[60:64] = 10
    hourly, count = hourly_measurements(x)
    np.testing.assert_allclose(hourly[0], 2.0)
    assert np.isnan(hourly[1:]).all()
    np.testing.assert_array_equal(count[:2, 0], [5, 4])
    even, _ = hourly_measurements(x, parity=0)
    odd, _ = hourly_measurements(x, parity=1)
    np.testing.assert_allclose(even[0], 2.0)
    assert np.isnan(odd[0]).all()
