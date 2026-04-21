"""Synthetic training data generator for the bit-pattern decoder.

Renders DICT_4X4_1000 markers, applies mild perspective warp (simulating
imperfect rectification), noise, and blur. Ground truth is the 4x4 bit
pattern.

Usage:
    python -m aruco_detection.nn_detection.training.datagen_decoder \\
        --output-dir nn-aruco-detection-test/training_data/decoder \\
        --num-samples 100000
"""

from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np
from tqdm import tqdm

from aruco_detection.nn_detection.dict_4x4_1000 import ArucoDictionary

RECT_SIZE = 32  # rectified marker size


def generate_one_sample(
    aruco_dict: ArucoDictionary,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one training sample: (rectified_image, bits).

    Returns
    -------
    image : (32, 32) uint8 grayscale
    bits : (16,) uint8 binary pattern (flattened 4x4)
    """
    marker_id = random.randint(0, 999)

    # Render marker at a comfortable size, then perspective-warp to simulate
    # imperfect rectification (the corner refiner won't be perfect)
    render_size = 128
    marker_img = aruco_dict.generate_marker_image(marker_id, render_size)

    # Add border
    border = render_size // 6
    marker_img = cv2.copyMakeBorder(
        marker_img, border, border, border, border,
        cv2.BORDER_CONSTANT, value=255,
    )
    mh, mw = marker_img.shape[:2]

    # Mild residual perspective warp (degree 0.02-0.15)
    degree = random.uniform(0.02, 0.15)
    src = np.float32([[0, 0], [mw, 0], [mw, mh], [0, mh]])
    max_dx = mw * degree
    max_dy = mh * degree
    dst = np.float32([
        [random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)],
        [mw + random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)],
        [mw + random.uniform(-max_dx, max_dx), mh + random.uniform(-max_dy, max_dy)],
        [random.uniform(-max_dx, max_dx), mh + random.uniform(-max_dy, max_dy)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(marker_img, M, (mw, mh), borderValue=128)

    # Resize to 32x32
    img = cv2.resize(warped, (RECT_SIZE, RECT_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)

    # Augmentations
    if random.random() < 0.5:
        img = img * random.uniform(0.5, 1.5)
    if random.random() < 0.4:
        img = img + random.uniform(-30, 30)
    if random.random() < 0.4:
        img = img + np.random.normal(0, random.uniform(5, 25), img.shape)
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    # Simulate salt-and-pepper noise
    if random.random() < 0.2:
        mask = np.random.random(img.shape)
        img[mask < 0.02] = 0
        img[mask > 0.98] = 255

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Ground truth: 4x4 bit pattern (flattened)
    bits = aruco_dict.get_pattern(marker_id).flatten().astype(np.uint8)

    return img, bits


def generate_dataset(
    output_dir: str,
    num_samples: int = 100000,
    val_ratio: float = 0.1,
):
    """Generate and save decoder training dataset."""
    aruco_dict = ArucoDictionary()

    images_list = []
    bits_list = []

    for _ in tqdm(range(num_samples), desc="Generating decoder data"):
        img, bits = generate_one_sample(aruco_dict)
        images_list.append(img)
        bits_list.append(bits)

    images = np.array(images_list, dtype=np.uint8)
    bits = np.array(bits_list, dtype=np.uint8)

    # Split
    n_val = int(num_samples * val_ratio)
    indices = np.random.RandomState(42).permutation(num_samples)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, "train.npz"),
        images=images[train_idx], bits=bits[train_idx],
    )
    np.savez_compressed(
        os.path.join(output_dir, "val.npz"),
        images=images[val_idx], bits=bits[val_idx],
    )
    print(f"Saved {len(train_idx)} train + {len(val_idx)} val to {output_dir}")


def main():
    p = argparse.ArgumentParser(description="Generate bit decoder training data")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=100000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    generate_dataset(args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()
