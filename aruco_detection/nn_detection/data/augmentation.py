"""Domain-specific augmentations for ArUco marker detection training.

Reuses the ``simulate_aruco_view`` pattern from ``aruco_detection/aruco_train.py``
and adds augmentations specific to the ant-tracking camera setup (IR lighting,
motion blur, noise).
"""

from __future__ import annotations

import random

import cv2
import numpy as np


def perspective_warp(
    image: np.ndarray,
    degree: float = 0.3,
    border_value: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Apply random perspective distortion to simulate viewing angle changes.

    Mirrors ``aruco_train.py:simulate_aruco_view`` but operates on numpy arrays
    directly instead of PIL images.
    """
    h, w = image.shape[:2]
    max_dx = w * degree
    max_dy = h * degree

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32(
        [
            [random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)],
            [w + random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)],
            [w + random.uniform(-max_dx, max_dx), h + random.uniform(-max_dy, max_dy)],
            [random.uniform(-max_dx, max_dx), h + random.uniform(-max_dy, max_dy)],
        ]
    )

    M = cv2.getPerspectiveTransform(src, dst)

    if border_value is None:
        border_value = tuple(int(c) for c in image.mean(axis=(0, 1)).flat)

    return cv2.warpPerspective(
        image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border_value
    )


def motion_blur(image: np.ndarray, ksize: int | None = None) -> np.ndarray:
    """Apply directional motion blur (random angle)."""
    if ksize is None:
        ksize = random.choice([3, 5, 7, 9])
    angle = random.uniform(0, 360)

    M = cv2.getRotationMatrix2D((ksize // 2, ksize // 2), angle, 1)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    kernel = kernel / kernel.sum()

    return cv2.filter2D(image, -1, kernel)


def adjust_brightness(image: np.ndarray, factor_range: tuple[float, float] = (0.6, 1.4)) -> np.ndarray:
    """Randomly adjust brightness."""
    factor = random.uniform(*factor_range)
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def add_gaussian_noise(image: np.ndarray, sigma_range: tuple[float, float] = (5, 25)) -> np.ndarray:
    """Add Gaussian noise to simulate sensor noise under IR lighting."""
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def simulate_ir_lighting(image: np.ndarray) -> np.ndarray:
    """Simulate IR-like low-contrast grayscale appearance.

    Converts to grayscale (if colour), applies CLAHE with low clip limit,
    and reduces overall contrast to mimic IR camera output.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(
        clipLimit=random.uniform(1.0, 3.0),
        tileGridSize=(8, 8),
    )
    gray = clahe.apply(gray)

    # Reduce contrast
    alpha = random.uniform(0.5, 0.8)
    mean_val = gray.mean()
    gray = np.clip(alpha * gray + (1 - alpha) * mean_val, 0, 255).astype(np.uint8)
    return gray


def augment_marker_image(
    image: np.ndarray,
    perspective: bool = True,
    blur: bool = True,
    brightness: bool = True,
    noise: bool = True,
    ir: bool = False,
) -> np.ndarray:
    """Apply a random combination of augmentations to a marker image."""
    out = image.copy()

    if perspective and random.random() < 0.7:
        out = perspective_warp(out, degree=random.uniform(0.1, 0.35))

    if blur and random.random() < 0.3:
        out = motion_blur(out)

    if brightness and random.random() < 0.5:
        out = adjust_brightness(out)

    if noise and random.random() < 0.4:
        out = add_gaussian_noise(out)

    if ir and random.random() < 0.3:
        out = simulate_ir_lighting(out)

    return out
