"""Synthetic close-up photos of A100 tags for the tag_photos tests.

A tag is the marker plus its white quiet zone, pasted onto a dark, cluttered background with a
rotation and a little perspective, then blurred and noised like a phone photo.
"""
import cv2
import cv2.aruco as aruco
import numpy as np

MARGIN_CELLS = 1.2   # printed tags: 1.5 mm marker + 0.3 mm margin = 6 cells + 1.2 cells


def tag_patch(dictionary, tag_id, marker_side):
    """Marker + white margin, rendered near the target size so pasting does not alias."""
    cell = max(2, int(np.ceil(marker_side / 6)) + 1)
    marker = aruco.generateImageMarker(dictionary, tag_id, 6 * cell, borderBits=1)
    pad = int(round(MARGIN_CELLS * cell))
    patch = cv2.copyMakeBorder(marker, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return patch, cell


def place(scene, patch, cell, center, marker_side, angle_deg=0.0, tilt=0.0):
    """Paste patch so its marker is marker_side px wide at center; returns a new scene."""
    p = patch.shape[0]
    half = marker_side * p / (6 * cell) / 2
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    square = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    square[2] *= 1.0 + tilt
    dst = (square @ rot.T + np.asarray(center, float)).astype(np.float32)
    src = np.float32([[0, 0], [p, 0], [p, p], [0, p]])
    m = cv2.getPerspectiveTransform(src, dst)
    h, w = scene.shape
    warped = cv2.warpPerspective(patch, m, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full_like(patch, 255), m, (w, h), flags=cv2.INTER_LINEAR)
    return np.where(mask > 127, warped, scene).astype(np.uint8)


def background(shape, seed=0):
    """Dark dish with ant-like dark blobs and a few bright reflections."""
    rng = np.random.default_rng(seed)
    h, w = shape
    scene = np.full((h, w), 70, np.uint8)
    for _ in range(12):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        axes = (int(rng.integers(20, 90)), int(rng.integers(10, 40)))
        cv2.ellipse(scene, c, axes, float(rng.uniform(0, 180)), 0, 360, int(rng.integers(15, 45)), -1)
    for _ in range(5):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(scene, p1, p2, int(rng.integers(160, 230)), int(rng.integers(2, 6)))
    return scene


def finish(scene, blur=1.0, noise=5.0, seed=0):
    """Phone-like blur and sensor noise."""
    rng = np.random.default_rng(seed + 1)
    out = cv2.GaussianBlur(scene, (0, 0), blur) if blur > 0 else scene
    out = out.astype(np.float32) + rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def scene_with_tags(dictionary, tags, shape=(1200, 1600), seed=0, blur=1.0, noise=5.0):
    """tags = [(tag_id, (x, y), marker_side, angle_deg, tilt[, mirrored]), ...] -> grayscale photo.
    mirrored = the tag seen from behind through its clear film."""
    scene = background(shape, seed)
    for tag_id, center, side, angle, tilt, *mirrored in tags:
        patch, cell = tag_patch(dictionary, tag_id, side)
        if mirrored and mirrored[0]:
            patch = np.ascontiguousarray(np.fliplr(patch))
        scene = place(scene, patch, cell, center, side, angle, tilt)
    return finish(scene, blur, noise, seed)
