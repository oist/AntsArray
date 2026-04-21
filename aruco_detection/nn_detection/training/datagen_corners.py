"""Synthetic training data generator for corner refinement.

Renders DICT_4X4_1000 markers with random perspective warp onto real
background patches, records transformed corner positions, and generates
Gaussian heatmaps as training targets.

Usage:
    python -m aruco_detection.nn_detection.training.datagen_corners \\
        --output-dir nn-aruco-detection-test/training_data/corners \\
        --background-dir benchmark/sample_frames \\
        --num-samples 50000
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from aruco_detection.nn_detection.dict_4x4_1000 import ArucoDictionary
from aruco_detection.nn_detection.models.corner_refiner import make_gaussian_heatmap


CROP_SIZE = 64
HEATMAP_SIGMA = 2.0


def load_background_patches(bg_dir: str | None, patch_size: int = CROP_SIZE) -> list[np.ndarray]:
    """Load random grayscale patches from background images."""
    patches: list[np.ndarray] = []

    if bg_dir and os.path.isdir(bg_dir):
        exts = {".png", ".jpg", ".jpeg"}
        for cam_dir in sorted(Path(bg_dir).iterdir()):
            if not cam_dir.is_dir():
                continue
            for img_path in sorted(cam_dir.iterdir()):
                if img_path.suffix.lower() not in exts:
                    continue
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                h, w = img.shape
                # Extract several random patches per image
                for _ in range(5):
                    y = random.randint(0, h - patch_size)
                    x = random.randint(0, w - patch_size)
                    patches.append(img[y : y + patch_size, x : x + patch_size])

    # Fallback: generate noise patches
    if not patches:
        for _ in range(200):
            base = random.randint(40, 180)
            patch = np.clip(
                np.random.normal(base, 15, (patch_size, patch_size)), 0, 255
            ).astype(np.uint8)
            patches.append(patch)

    return patches


def generate_one_sample(
    aruco_dict: ArucoDictionary,
    backgrounds: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one training sample: (crop, heatmaps, corners).

    Returns
    -------
    crop : (64, 64) uint8 grayscale
    heatmaps : (4, 64, 64) float32 — one Gaussian blob per corner
    corners : (4, 2) float32 — (x, y) corner positions in crop coords
    """
    marker_id = random.randint(0, 999)
    marker_size = random.randint(24, 52)  # marker size within the 64x64 crop
    marker_img = aruco_dict.generate_marker_image(marker_id, marker_size)

    # Add white border (ArUco markers need border for detection)
    border = max(2, marker_size // 8)
    marker_img = cv2.copyMakeBorder(
        marker_img, border, border, border, border,
        cv2.BORDER_CONSTANT, value=255,
    )
    mh, mw = marker_img.shape[:2]

    # Define corners of the marker (with border) in source coords
    src_corners = np.float32([[0, 0], [mw, 0], [mw, mh], [0, mh]])

    # Random perspective warp
    degree = random.uniform(0.05, 0.35)
    max_dx = mw * degree
    max_dy = mh * degree
    dst_corners = np.float32([
        [random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)],
        [mw + random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)],
        [mw + random.uniform(-max_dx, max_dx), mh + random.uniform(-max_dy, max_dy)],
        [random.uniform(-max_dx, max_dx), mh + random.uniform(-max_dy, max_dy)],
    ])

    M = cv2.getPerspectiveTransform(src_corners, dst_corners)

    # Pick a background patch
    bg = random.choice(backgrounds).copy()

    # Place warped marker on background
    # Compute where the marker lands in the crop
    offset_x = random.randint(2, max(3, CROP_SIZE - mw - 2))
    offset_y = random.randint(2, max(3, CROP_SIZE - mh - 2))

    # Adjust the perspective transform to include the offset
    T = np.float32([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]])
    M_full = T @ M

    # Warp marker onto the crop-sized canvas
    warped = cv2.warpPerspective(
        marker_img, M_full, (CROP_SIZE, CROP_SIZE),
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    # Create mask for compositing
    mask = cv2.warpPerspective(
        np.ones_like(marker_img) * 255, M_full, (CROP_SIZE, CROP_SIZE),
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    mask_f = (mask.astype(np.float32) / 255.0)

    # Composite marker onto background
    crop = (bg.astype(np.float32) * (1 - mask_f) + warped.astype(np.float32) * mask_f)

    # Augmentations
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        crop = cv2.GaussianBlur(crop, (ksize, ksize), 0)
    if random.random() < 0.5:
        crop = crop * random.uniform(0.6, 1.4)
    if random.random() < 0.4:
        crop = crop + np.random.normal(0, random.uniform(5, 20), crop.shape)

    crop = np.clip(crop, 0, 255).astype(np.uint8)

    # Transform the 4 inner corners (without border) through M_full
    inner_corners_src = np.float32([
        [border, border],
        [mw - border, border],
        [mw - border, mh - border],
        [border, mh - border],
    ])
    inner_corners_dst = cv2.perspectiveTransform(
        inner_corners_src.reshape(1, -1, 2), M_full
    ).reshape(4, 2)

    # Generate heatmaps
    heatmaps = np.zeros((4, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    corners = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        cx, cy = inner_corners_dst[i]
        cx = np.clip(cx, 0, CROP_SIZE - 1)
        cy = np.clip(cy, 0, CROP_SIZE - 1)
        corners[i] = [cx, cy]
        heatmaps[i] = make_gaussian_heatmap(CROP_SIZE, cx, cy, HEATMAP_SIGMA).numpy()

    return crop, heatmaps, corners


def generate_dataset(
    output_dir: str,
    backgrounds: list[np.ndarray],
    num_samples: int = 50000,
    val_ratio: float = 0.1,
):
    """Generate and save corner refinement dataset as .npz files."""
    aruco_dict = ArucoDictionary()

    crops_list = []
    heatmaps_list = []
    corners_list = []

    for _ in tqdm(range(num_samples), desc="Generating corner data"):
        crop, heatmaps, corners = generate_one_sample(aruco_dict, backgrounds)
        crops_list.append(crop)
        heatmaps_list.append(heatmaps)
        corners_list.append(corners)

    crops = np.array(crops_list, dtype=np.uint8)
    heatmaps = np.array(heatmaps_list, dtype=np.float32)
    corners = np.array(corners_list, dtype=np.float32)

    # Split
    n_val = int(num_samples * val_ratio)
    indices = np.random.RandomState(42).permutation(num_samples)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, "train.npz"),
        crops=crops[train_idx], heatmaps=heatmaps[train_idx], corners=corners[train_idx],
    )
    np.savez_compressed(
        os.path.join(output_dir, "val.npz"),
        crops=crops[val_idx], heatmaps=heatmaps[val_idx], corners=corners[val_idx],
    )
    print(f"Saved {len(train_idx)} train + {len(val_idx)} val to {output_dir}")


def main():
    p = argparse.ArgumentParser(description="Generate corner refinement training data")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--background-dir", default="benchmark/sample_frames")
    p.add_argument("--num-samples", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    backgrounds = load_background_patches(args.background_dir)
    print(f"Loaded {len(backgrounds)} background patches")
    generate_dataset(args.output_dir, backgrounds, args.num_samples)


if __name__ == "__main__":
    main()
