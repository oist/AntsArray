#!/usr/bin/env python3
"""
Prepare training data for fast local training.

Optimizations:
1. Resize YOLO images from 4K (4024x3036) to training size (640px) — ~20x IO savings
2. Convert PNG to JPEG for faster decode
3. Optionally split YOLO data into train/val (80/20)
4. Print dataset summary

Usage:
    python nn-aruco-detection-test/prep_training.py
    python nn-aruco-detection-test/prep_training.py --imgsz 1280  # larger input
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def resize_yolo_images(
    yolo_dir: str,
    target_size: int = 640,
    use_jpeg: bool = True,
    val_ratio: float = 0.2,
):
    """Resize YOLO images in-place and create train/val split."""
    img_train = os.path.join(yolo_dir, "images", "train")
    lbl_train = os.path.join(yolo_dir, "labels", "train")

    if not os.path.isdir(img_train):
        print(f"[ERROR] No images at {img_train}")
        return

    # Collect all image files
    exts = {".png", ".jpg", ".jpeg"}
    all_images = sorted(
        p for p in Path(img_train).iterdir() if p.suffix.lower() in exts
    )
    print(f"Found {len(all_images)} YOLO images")

    if not all_images:
        return

    # Check current size
    sample = cv2.imread(str(all_images[0]), cv2.IMREAD_UNCHANGED)
    orig_h, orig_w = sample.shape[:2]
    print(f"Original size: {orig_w}x{orig_h}")
    print(f"Target size: {target_size}x{target_size}")

    if orig_w <= target_size and orig_h <= target_size:
        print("Images already at or below target size, skipping resize")
    else:
        # Resize all images
        new_ext = ".jpg" if use_jpeg else ".png"
        print(f"Resizing to {target_size}px, format: {new_ext[1:].upper()}")

        for img_path in tqdm(all_images, desc="Resizing"):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # Resize maintaining aspect ratio, pad to square
            h, w = img.shape[:2]
            scale = target_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Pad to square
            canvas = np.zeros((target_size, target_size), dtype=np.uint8) if resized.ndim == 2 \
                else np.zeros((target_size, target_size, 3), dtype=np.uint8)
            pad_y = (target_size - new_h) // 2
            pad_x = (target_size - new_w) // 2
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

            # Update label coordinates for padding offset
            lbl_path = Path(lbl_train) / (img_path.stem + ".txt")
            if lbl_path.exists():
                with open(lbl_path) as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, bw, bh = parts
                        # Adjust for resize + padding
                        cx_new = (float(cx) * new_w + pad_x) / target_size
                        cy_new = (float(cy) * new_h + pad_y) / target_size
                        bw_new = float(bw) * new_w / target_size
                        bh_new = float(bh) * new_h / target_size
                        new_lines.append(f"{cls} {cx_new:.6f} {cy_new:.6f} {bw_new:.6f} {bh_new:.6f}")
                with open(lbl_path, "w") as f:
                    f.write("\n".join(new_lines) + "\n" if new_lines else "")

            # Save resized image (remove old, write new)
            new_path = img_path.with_suffix(new_ext)
            if new_ext == ".jpg":
                cv2.imwrite(str(new_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(str(new_path), canvas)

            # Rename label to match new extension if changed
            if new_ext != img_path.suffix.lower():
                old_lbl = Path(lbl_train) / (img_path.stem + ".txt")
                # label filename stays the same stem, just image ext changed
                img_path.unlink()  # remove old PNG

        print(f"Resize complete")

    # --- Train/val split ---
    print(f"\nCreating train/val split ({1-val_ratio:.0%}/{val_ratio:.0%})...")

    # Re-scan after resize (extension may have changed)
    all_images = sorted(
        p for p in Path(img_train).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    random.seed(42)
    random.shuffle(all_images)
    n_val = int(len(all_images) * val_ratio)
    val_images = all_images[:n_val]

    img_val = os.path.join(yolo_dir, "images", "val")
    lbl_val = os.path.join(yolo_dir, "labels", "val")
    os.makedirs(img_val, exist_ok=True)
    os.makedirs(lbl_val, exist_ok=True)

    for img_path in val_images:
        lbl_name = img_path.stem + ".txt"
        src_lbl = os.path.join(lbl_train, lbl_name)

        shutil.move(str(img_path), os.path.join(img_val, img_path.name))
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, os.path.join(lbl_val, lbl_name))

    n_train = len(all_images) - n_val
    print(f"Split: {n_train} train / {n_val} val")

    # Update data.yaml with proper val path
    yaml_path = os.path.join(yolo_dir, "data.yaml")
    abs_dir = os.path.abspath(yolo_dir).replace("\\", "/")
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  0: marker\n")
    print(f"Updated: {yaml_path}")


def summarize(training_dir: str):
    """Print dataset summary."""
    print(f"\n{'='*50}")
    print("DATASET SUMMARY")
    print(f"{'='*50}")

    # YOLO
    yolo_dir = os.path.join(training_dir, "yolo")
    for split in ("train", "val"):
        img_dir = os.path.join(yolo_dir, "images", split)
        if os.path.isdir(img_dir):
            imgs = list(Path(img_dir).iterdir())
            if imgs:
                sample = cv2.imread(str(imgs[0]), cv2.IMREAD_UNCHANGED)
                h, w = sample.shape[:2] if sample is not None else (0, 0)
                total_mb = sum(f.stat().st_size for f in imgs) / 1e6
                print(f"YOLO {split}: {len(imgs)} images, {w}x{h}, {total_mb:.0f} MB")

    # Classification
    cls_dir = os.path.join(training_dir, "classification")
    if os.path.isdir(cls_dir):
        classes = [d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))]
        total = sum(
            len(list(Path(os.path.join(cls_dir, c)).iterdir()))
            for c in classes
        )
        print(f"Classifier: {total} crops across {len(classes)} marker IDs")

    # Hard negatives
    hard_dir = os.path.join(training_dir, "hard_negatives")
    if os.path.isdir(hard_dir):
        n = len([f for f in os.listdir(hard_dir) if f.endswith(".png")])
        print(f"Hard negatives: {n} crops")

    print(f"{'='*50}")


def main():
    p = argparse.ArgumentParser(description="Prepare training data for fast local training")
    p.add_argument(
        "--training-dir", type=str,
        default="nn-aruco-detection-test/training_data",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Target YOLO image size")
    p.add_argument("--jpeg", action="store_true", default=True, help="Convert to JPEG (default)")
    p.add_argument("--no-jpeg", dest="jpeg", action="store_false", help="Keep PNG format")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    args = p.parse_args()

    yolo_dir = os.path.join(args.training_dir, "yolo")
    if os.path.isdir(yolo_dir):
        resize_yolo_images(yolo_dir, args.imgsz, args.jpeg, args.val_ratio)

    summarize(args.training_dir)


if __name__ == "__main__":
    main()
