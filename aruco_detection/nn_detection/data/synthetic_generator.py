#!/usr/bin/env python3
"""Generate synthetic training data for ArUco marker detection.

Creates images with DICT_4X4_1000 markers composited onto background patches,
with domain-specific augmentations (perspective, blur, IR lighting).

Output formats:
- **YOLO**: ``images/`` + ``labels/`` with one-class ("marker") bounding boxes.
- **Classification**: ``ImageFolder`` layout for ID classification training.

Usage:
    python -m aruco_detection.nn_detection.data.synthetic_generator \\
        --output-dir data/synthetic \\
        --num-images 50000 \\
        --format yolo \\
        --background-dir benchmark/sample_frames
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
from tqdm import tqdm

from aruco_detection.nn_detection.data.augmentation import (
    add_gaussian_noise,
    adjust_brightness,
    motion_blur,
    perspective_warp,
)

DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

# Marker pixel sizes matching real-world observation (~70-80 px in 4K frames)
MARKER_SIZE_RANGE = (40, 120)
NUM_MARKERS_RANGE = (1, 10)


def render_marker(marker_id: int, size: int) -> np.ndarray:
    """Render a single ArUco marker as a grayscale image with white border."""
    # Inner marker is (size - border*2) px, but generateImageMarker adds a 1-cell border
    img = aruco.generateImageMarker(DICT, marker_id, size)
    # Add a small white border so the marker is detectable in composites
    border = max(2, size // 10)
    bordered = cv2.copyMakeBorder(
        img, border, border, border, border,
        cv2.BORDER_CONSTANT, value=255,
    )
    return bordered


def load_backgrounds(bg_dir: str | None, fallback_size: tuple[int, int] = (1280, 1280)) -> list[np.ndarray]:
    """Load background images from a directory, or generate random noise backgrounds."""
    backgrounds: list[np.ndarray] = []

    if bg_dir and os.path.isdir(bg_dir):
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        for p in sorted(Path(bg_dir).iterdir()):
            if p.suffix.lower() in exts:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    backgrounds.append(img)

    if not backgrounds:
        # Generate noise backgrounds as fallback
        for _ in range(20):
            base = np.random.randint(40, 180, dtype=np.uint8)
            bg = np.full(fallback_size, base, dtype=np.uint8)
            noise = np.random.normal(0, 15, fallback_size).astype(np.float32)
            bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            backgrounds.append(bg)

    return backgrounds


def random_crop(bg: np.ndarray, crop_size: int) -> np.ndarray:
    """Take a random crop from a background image."""
    h, w = bg.shape[:2]
    if h < crop_size or w < crop_size:
        return cv2.resize(bg, (crop_size, crop_size))
    y = random.randint(0, h - crop_size)
    x = random.randint(0, w - crop_size)
    crop = bg[y : y + crop_size, x : x + crop_size]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return crop


def composite_marker(
    background: np.ndarray,
    marker: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    """Paste a marker onto a background at (x, y)."""
    mh, mw = marker.shape[:2]
    bh, bw = background.shape[:2]

    # Clip to image bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + mw)
    y2 = min(bh, y + mh)

    mx1 = x1 - x
    my1 = y1 - y
    mx2 = mw - (x + mw - x2)
    my2 = mh - (y + mh - y2)

    if x2 <= x1 or y2 <= y1:
        return background

    out = background.copy()
    out[y1:y2, x1:x2] = marker[my1:my2, mx1:mx2]
    return out


def generate_yolo_image(
    backgrounds: list[np.ndarray],
    image_size: int = 640,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """Generate one training image with markers and YOLO-format labels.

    Returns:
        image: (image_size, image_size) uint8
        labels: list of (class_id, x_center, y_center, w, h) normalised
    """
    bg = random_crop(random.choice(backgrounds), image_size)

    n_markers = random.randint(*NUM_MARKERS_RANGE)
    labels: list[tuple[int, float, float, float, float]] = []

    for _ in range(n_markers):
        marker_id = random.randint(0, 999)
        size = random.randint(*MARKER_SIZE_RANGE)
        marker_img = render_marker(marker_id, size)

        # Apply perspective warp to the marker before compositing
        if random.random() < 0.7:
            marker_img = perspective_warp(marker_img, degree=random.uniform(0.1, 0.3))

        mh, mw = marker_img.shape[:2]

        # Random position (allow partial out-of-bounds)
        x = random.randint(-mw // 4, image_size - mw + mw // 4)
        y = random.randint(-mh // 4, image_size - mh + mh // 4)

        bg = composite_marker(bg, marker_img, x, y)

        # Compute visible bounding box (clipped)
        bx1 = max(0, x)
        by1 = max(0, y)
        bx2 = min(image_size, x + mw)
        by2 = min(image_size, y + mh)
        if bx2 <= bx1 or by2 <= by1:
            continue

        # YOLO normalised coords — class 0 = "marker"
        cx = (bx1 + bx2) / 2 / image_size
        cy = (by1 + by2) / 2 / image_size
        w = (bx2 - bx1) / image_size
        h = (by2 - by1) / image_size
        labels.append((0, cx, cy, w, h))

    # Global augmentations
    if random.random() < 0.3:
        bg = motion_blur(bg)
    if random.random() < 0.5:
        bg = adjust_brightness(bg)
    if random.random() < 0.4:
        bg = add_gaussian_noise(bg)

    return bg, labels


def generate_classification_crop(
    backgrounds: list[np.ndarray],
    marker_id: int,
    crop_size: int = 128,
) -> np.ndarray:
    """Generate a single crop image for the ArUco ID classifier.

    Returns a (crop_size, crop_size) grayscale image with the marker
    centred on a background patch.
    """
    bg = random_crop(random.choice(backgrounds), crop_size)
    marker_size = random.randint(crop_size // 4, crop_size * 3 // 4)
    marker = render_marker(marker_id, marker_size)

    if random.random() < 0.7:
        marker = perspective_warp(marker, degree=random.uniform(0.1, 0.3))

    mh, mw = marker.shape[:2]
    x = (crop_size - mw) // 2 + random.randint(-mw // 6, mw // 6)
    y = (crop_size - mh) // 2 + random.randint(-mh // 6, mh // 6)

    img = composite_marker(bg, marker, x, y)

    if random.random() < 0.3:
        img = motion_blur(img)
    if random.random() < 0.5:
        img = adjust_brightness(img)
    if random.random() < 0.4:
        img = add_gaussian_noise(img)

    return img


def generate_yolo_dataset(
    output_dir: str,
    backgrounds: list[np.ndarray],
    num_images: int = 50000,
    image_size: int = 640,
    train_ratio: float = 0.8,
):
    """Generate a complete YOLO-format dataset.

    Directory layout:
        output_dir/
            images/train/ images/val/
            labels/train/ labels/val/
            data.yaml
    """
    for split in ("train", "val"):
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    train_count = int(num_images * train_ratio)

    for idx in tqdm(range(num_images), desc="Generating YOLO data"):
        split = "train" if idx < train_count else "val"
        img, labels = generate_yolo_image(backgrounds, image_size)

        img_path = os.path.join(output_dir, "images", split, f"{idx:06d}.png")
        lbl_path = os.path.join(output_dir, "labels", split, f"{idx:06d}.txt")

        cv2.imwrite(img_path, img)
        with open(lbl_path, "w") as f:
            for cls, cx, cy, w, h in labels:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # Write data.yaml for Ultralytics
    yaml_path = os.path.join(output_dir, "data.yaml")
    abs_dir = os.path.abspath(output_dir).replace("\\", "/")
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  0: marker\n")

    print(f"YOLO dataset written to {output_dir}  ({train_count} train / {num_images - train_count} val)")


def generate_classification_dataset(
    output_dir: str,
    backgrounds: list[np.ndarray],
    num_ids: int = 300,
    crops_per_id: int = 200,
    crop_size: int = 128,
):
    """Generate ImageFolder dataset for ArUco ID classification.

    Directory layout:
        output_dir/
            0/  1/  2/  ...  {num_ids-1}/
                000.png  001.png  ...
    """
    for marker_id in tqdm(range(num_ids), desc="Generating classifier data"):
        class_dir = os.path.join(output_dir, str(marker_id))
        os.makedirs(class_dir, exist_ok=True)
        for j in range(crops_per_id):
            img = generate_classification_crop(backgrounds, marker_id, crop_size)
            cv2.imwrite(os.path.join(class_dir, f"{j:04d}.png"), img)

    print(f"Classification dataset written to {output_dir}  ({num_ids} classes, {crops_per_id} each)")


def main():
    p = argparse.ArgumentParser(description="Generate synthetic ArUco training data")
    p.add_argument("--output-dir", required=True, help="Root output directory")
    p.add_argument("--background-dir", default=None, help="Directory with background images")
    p.add_argument("--format", choices=["yolo", "classification", "both"], default="yolo")
    p.add_argument("--num-images", type=int, default=50000, help="Number of YOLO images")
    p.add_argument("--image-size", type=int, default=640, help="YOLO image size (px)")
    p.add_argument("--num-ids", type=int, default=300, help="Number of marker IDs for classification")
    p.add_argument("--crops-per-id", type=int, default=200, help="Crops per ID for classification")
    p.add_argument("--crop-size", type=int, default=128, help="Crop size for classification")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    backgrounds = load_backgrounds(args.background_dir)
    print(f"Loaded {len(backgrounds)} background images")

    if args.format in ("yolo", "both"):
        yolo_dir = os.path.join(args.output_dir, "yolo")
        generate_yolo_dataset(yolo_dir, backgrounds, args.num_images, args.image_size)

    if args.format in ("classification", "both"):
        cls_dir = os.path.join(args.output_dir, "classification")
        generate_classification_dataset(
            cls_dir, backgrounds, args.num_ids, args.crops_per_id, args.crop_size
        )


if __name__ == "__main__":
    main()
