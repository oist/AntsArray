from types import SimpleNamespace

import numpy as np
import pytest

from tracking.gui.interaction_debug_viewer import (
    ArucoPoint, ArucoSleepViewer, InteractionDebugViewer, apply_homography_points,
    finished_track_frame, scale_bar_endpoints,
)


def example_review():
    return SimpleNamespace(
        start=100, stop=102, ids=np.array([41, 7, 56, 99]),
        anchors=np.array([[[20, 30], [70, 40], [np.nan, np.nan], [120, 10]],
                          [[21, 31], [71, 41], [np.nan, np.nan], [121, 11]]]),
        xy=np.full((2, 4, 10, 2), np.nan),
        manifest=dict(mm_per_pixel=.016),
    )


def test_finished_ids_projection_and_missing_nodes():
    review = example_review()
    # Crossed poses retain their final identities; no nearest-anchor reassignment.
    review.xy[1, 0, 0] = [71, 41]
    review.xy[1, 1, 0] = [21, 31]
    inverse_h = np.array([[1, 0, 3], [0, 1, 4], [0, 0, 1]])
    points, poses = finished_track_frame(review, 101, inverse_h, width=100, height=100)
    assert [p.track_id for p in points] == [41, 7]
    np.testing.assert_allclose([[p.x, p.y] for p in points], [[24, 35], [74, 45]])
    np.testing.assert_allclose(poses[41][0], [74, 45])
    np.testing.assert_allclose(poses[7][0], [24, 35])
    assert np.isnan(poses[41][1:]).all()
    assert 56 not in poses and 99 not in poses


def test_frame_offsets_and_camera_bounds():
    review = example_review()
    points, _ = finished_track_frame(review, 100, np.eye(3), width=70, height=100)
    assert [p.track_id for p in points] == [41]
    for frame in (99, 102):
        with pytest.raises(ValueError, match="outside finished-track"):
            finished_track_frame(review, frame, np.eye(3), width=100, height=100)


def test_render_never_reads_raw_labels():
    app = InteractionDebugViewer.__new__(InteractionDebugViewer)
    app.review = example_review()
    app.current_frame = 101
    app.inverse_h = np.eye(3)
    app.panorama_h = np.eye(3)
    app.overlay_scale = 1
    app.video = SimpleNamespace(read_frame=lambda frame: np.zeros((100, 100, 3), dtype=np.uint8))
    spec = SimpleNamespace(chunk=5, to_local=lambda frame: frame-100)

    def forbidden(*args):
        pytest.fail("Finished-track rendering must not read camera detections")

    app.labels = SimpleNamespace(spec_for_frame=lambda frame: spec,
                                aruco_for_frame=forbidden, sleap_for_frame=forbidden)
    app.show_sleap_var = SimpleNamespace(get=lambda: True)
    app.show_sleep_var = SimpleNamespace(get=lambda: False)
    app._draw_text = lambda *args, **kwargs: None
    seen = []

    def draw_labels(image, points, spec):
        seen.extend(p.track_id for p in points)
        return dict(sleep=1, wake=1, unknown=0)

    app.draw_aruco_and_sleep = draw_labels
    app.display_current_image = lambda: None
    status = []
    app.status_var = SimpleNamespace(set=status.append)
    app.render()
    assert seen == [41, 7]
    assert "Frame 101" in status[-1] and "Finished tracks 2" in status[-1]
    assert app.current_annotated_rgb.shape == (100, 100, 3)


@pytest.mark.parametrize("h", [np.eye(3), np.array([[.3, .04, 80], [-.02, .32, 120], [.00001, -.00002, 1]])])
def test_scale_bar_calibration_including_perspective(h):
    points = scale_bar_endpoints(h, origin=(110, 2900), length_mm=1, mm_per_pixel=.016)
    assert points[1, 0] > points[0, 0]
    assert points[1, 1] == pytest.approx(points[0, 1])
    panorama = apply_homography_points(points, h)
    assert np.linalg.norm(panorama[1]-panorama[0])*.016 == pytest.approx(1)


def test_sleep_overlay_off_skips_classification_but_keeps_ids(monkeypatch):
    app = InteractionDebugViewer.__new__(InteractionDebugViewer)
    app.show_sleep_var = SimpleNamespace(get=lambda: False)
    app.show_aruco_var = SimpleNamespace(get=lambda: True)
    app.overlay_scale = 1
    texts = []
    app._draw_text = lambda image, text, *args, **kwargs: texts.append(text)
    classifications = []

    def classify(*args):
        classifications.append(True)
        return dict(sleep=1, wake=0, unknown=0)

    monkeypatch.setattr(ArucoSleepViewer, "draw_aruco_and_sleep", classify)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    point = ArucoPoint(56, 40, 40, np.nan)
    app.draw_track_labels(image, [point], None)
    assert texts == ["T56"] and not classifications
    app.show_sleep_var = SimpleNamespace(get=lambda: True)
    assert app.draw_track_labels(image, [point], None)["sleep"] == 1
    assert classifications == [True]
