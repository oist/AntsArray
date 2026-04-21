#!/usr/bin/env python3
"""Fine-tune corner refiner and bit decoder on real classification crops.

Uses the existing classification dataset (128x128 crops with known marker IDs)
from ``nn-aruco-detection-test/training_data/classification/``. Runs OpenCV
ArUco detection on each crop to extract corners, then:

1. **Corner refiner fine-tuning**: real crop → resize to 64x64, GT = OpenCV corners
2. **Decoder fine-tuning**: perspective-rectify using OpenCV corners → 32x32,
   GT = known bit pattern from the marker ID

This bridges the sim-to-real gap that caused the DeepArUco-PT decoder to fail
on real data.

Usage:
    python -m aruco_detection.nn_detection.training.finetune_real \\
        --cls-dir nn-aruco-detection-test/training_data/classification \\
        --output-dir nn-aruco-detection-test/models \\
        --corner-refiner-weights nn-aruco-detection-test/models/corner_refiner.pth \\
        --decoder-weights nn-aruco-detection-test/models/bit_decoder.pth \\
        --epochs 50
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aruco_detection.nn_detection.dict_4x4_1000 import ArucoDictionary
from aruco_detection.nn_detection.models.bit_decoder import BitDecoder
from aruco_detection.nn_detection.models.corner_refiner import (
    build_corner_refiner,
    make_gaussian_heatmap,
    soft_argmax_2d,
)


CROP_SIZE = 128       # input classification crop size
CORNER_SIZE = 64      # corner refiner input
DECODER_SIZE = 32     # decoder input
HEATMAP_SIGMA = 2.0


def validate_classes(
    cls_dir: str,
    min_confirm_rate: float = 0.1,
    samples_per_class: int = 30,
) -> set[int]:
    """Identify which class IDs are trustworthy by cross-checking with OpenCV.

    For each class, runs OpenCV detection on a sample of crops. Classes where
    OpenCV never confirms the expected ID are considered FP artifacts from
    the original OpenCV detection pipeline and are excluded.

    Returns set of valid marker IDs.
    """
    aruco_dict_cv = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 50
    params.minMarkerPerimeterRate = 0.01
    detector = aruco.ArucoDetector(aruco_dict_cv, params)

    classes = sorted(d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d)))
    valid_ids: set[int] = set()
    rejected = 0

    for cls_name in tqdm(classes, desc="Validating classes", leave=False):
        marker_id = int(cls_name)
        cls_path = os.path.join(cls_dir, cls_name)
        files = sorted(os.listdir(cls_path))
        sample = random.sample(files, min(samples_per_class, len(files)))

        correct = 0
        for f in sample:
            img = cv2.imread(os.path.join(cls_path, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            corners, ids, _ = detector.detectMarkers(img)
            if ids is not None and marker_id in ids.flatten():
                correct += 1

        rate = correct / len(sample) if sample else 0
        if rate >= min_confirm_rate:
            valid_ids.add(marker_id)
        else:
            rejected += 1

    print(f"Class validation: {len(valid_ids)} valid, {rejected} rejected (confirm rate < {min_confirm_rate:.0%})")
    return valid_ids


def extract_real_training_pairs(
    cls_dir: str,
    max_per_class: int = 100,
    valid_ids: set[int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Extract corner + decoder training pairs from real classification crops.

    Runs OpenCV ArUco on each crop. If successful and the ID matches,
    produces:
    - corner pair: (64x64 crop, 4 corner positions, heatmaps)
    - decoder pair: (32x32 rectified image, 16-bit pattern)

    Parameters
    ----------
    valid_ids : set of int, optional
        If provided, only use crops from these marker IDs.
        Use ``validate_classes()`` to generate this set.

    Returns (corner_pairs, decoder_pairs)
    """
    aruco_dict_cv = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    params.adaptiveThreshConstant = 3
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 50
    params.adaptiveThreshWinSizeStep = 5
    params.minMarkerPerimeterRate = 0.01
    params.errorCorrectionRate = 1.0
    detector = aruco.ArucoDetector(aruco_dict_cv, params)

    bit_dict = ArucoDictionary()

    corner_pairs: list[dict] = []
    decoder_pairs: list[dict] = []

    classes = sorted(d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d)))
    if valid_ids is not None:
        classes = [c for c in classes if int(c) in valid_ids]
    print(f"Scanning {len(classes)} classes for real training pairs...")

    for cls_name in tqdm(classes, desc="Extracting pairs", leave=False):
        marker_id = int(cls_name)
        cls_path = os.path.join(cls_dir, cls_name)
        files = sorted(os.listdir(cls_path))
        if len(files) > max_per_class:
            files = random.sample(files, max_per_class)

        gt_bits = bit_dict.get_pattern(marker_id).flatten().astype(np.float32)

        for fname in files:
            img = cv2.imread(os.path.join(cls_path, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            corners_cv, ids_cv, _ = detector.detectMarkers(img)
            if ids_cv is None or len(ids_cv) == 0:
                continue

            # Find the detection matching the expected ID
            matched_idx = -1
            for j, det_id in enumerate(ids_cv.flatten()):
                if int(det_id) == marker_id:
                    matched_idx = j
                    break

            if matched_idx < 0:
                continue

            corners_4x2 = corners_cv[matched_idx][0]  # (4, 2) in 128x128 space

            # --- Corner refiner pair ---
            # Resize crop to 64x64 and scale corners
            crop_64 = cv2.resize(img, (CORNER_SIZE, CORNER_SIZE))
            scale = CORNER_SIZE / CROP_SIZE
            corners_64 = corners_4x2 * scale

            # Generate heatmaps
            heatmaps = np.zeros((4, CORNER_SIZE, CORNER_SIZE), dtype=np.float32)
            for i in range(4):
                cx, cy = corners_64[i]
                cx = np.clip(cx, 0, CORNER_SIZE - 1)
                cy = np.clip(cy, 0, CORNER_SIZE - 1)
                heatmaps[i] = make_gaussian_heatmap(CORNER_SIZE, float(cx), float(cy), HEATMAP_SIGMA).numpy()

            corner_pairs.append({
                "crop": crop_64,
                "heatmaps": heatmaps,
                "corners": corners_64.astype(np.float32),
            })

            # --- Decoder pair ---
            # Perspective-rectify using OpenCV corners
            src_pts = corners_4x2.astype(np.float32)
            dst_pts = np.float32([
                [0, 0], [DECODER_SIZE, 0],
                [DECODER_SIZE, DECODER_SIZE], [0, DECODER_SIZE],
            ])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            rectified = cv2.warpPerspective(img, M, (DECODER_SIZE, DECODER_SIZE), borderValue=128)

            decoder_pairs.append({
                "image": rectified,
                "bits": gt_bits,
                "marker_id": marker_id,
            })

    print(f"Extracted {len(corner_pairs)} corner pairs, {len(decoder_pairs)} decoder pairs")
    return corner_pairs, decoder_pairs


class CornerFinetuneDataset(Dataset):
    def __init__(self, pairs: list[dict], augment: bool = True):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        crop = p["crop"].astype(np.float32) / 255.0

        if self.augment:
            # Mild augmentations that preserve corner positions
            if random.random() < 0.3:
                crop = crop * random.uniform(0.7, 1.3)
            if random.random() < 0.3:
                noise = np.random.normal(0, 0.02, crop.shape).astype(np.float32)
                crop = crop + noise
            crop = np.clip(crop, 0, 1)

        crop_t = torch.from_numpy(crop).unsqueeze(0)  # (1, 64, 64)
        heatmaps_t = torch.from_numpy(p["heatmaps"])  # (4, 64, 64)
        corners_t = torch.from_numpy(p["corners"])     # (4, 2)
        return crop_t, heatmaps_t, corners_t


class DecoderFinetuneDataset(Dataset):
    def __init__(self, pairs: list[dict], augment: bool = True):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        img = p["image"].astype(np.float32) / 255.0

        if self.augment:
            if random.random() < 0.5:
                img = img * random.uniform(0.6, 1.4)
            if random.random() < 0.4:
                noise = np.random.normal(0, 0.03, img.shape).astype(np.float32)
                img = img + noise
            if random.random() < 0.3:
                ksize = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            img = np.clip(img, 0, 1)

        img_t = torch.from_numpy(img).unsqueeze(0)  # (1, 32, 32)
        bits_t = torch.from_numpy(p["bits"])          # (16,)
        return img_t, bits_t


def finetune_corner_refiner(
    pairs: list[dict],
    pretrained_weights: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
):
    """Fine-tune the corner refiner on real crops."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.1))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_ds = CornerFinetuneDataset(train_pairs, augment=True)
    val_ds = CornerFinetuneDataset(val_pairs, augment=False)
    n_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=True)

    print(f"  Corner refiner: {len(train_pairs)} train / {len(val_pairs)} val")

    # Load pretrained model
    model = build_corner_refiner()
    model.load_state_dict(torch.load(pretrained_weights, map_location=device))
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_err = float("inf")

    for epoch in range(epochs):
        model.train()
        for crops, heatmaps_gt, _ in train_loader:
            crops, heatmaps_gt = crops.to(device), heatmaps_gt.to(device)
            optimizer.zero_grad()
            pred = model(crops)
            loss = criterion(pred, heatmaps_gt)
            loss.backward()
            optimizer.step()

        model.eval()
        val_err = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for crops, _, corners_gt in val_loader:
                crops, corners_gt = crops.to(device), corners_gt.to(device)
                hm = model(crops)
                corners_pred = soft_argmax_2d(hm)
                err = torch.sqrt(((corners_pred - corners_gt) ** 2).sum(dim=-1)).mean()
                val_err += err.item() * crops.size(0)
                n_val_samples += crops.size(0)

        val_err /= max(1, n_val_samples)
        scheduler.step()

        if val_err < best_val_err:
            best_val_err = val_err
            torch.save(model.state_dict(), output_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}  corner_err={val_err:.2f}px")

    print(f"  Best corner error: {best_val_err:.2f}px -> {output_path}")


def finetune_decoder(
    pairs: list[dict],
    pretrained_weights: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 5e-4,
):
    """Fine-tune the bit decoder on real rectified crops."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.1))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_ds = DecoderFinetuneDataset(train_pairs, augment=True)
    val_ds = DecoderFinetuneDataset(val_pairs, augment=False)
    n_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=True)

    print(f"  Decoder: {len(train_pairs)} train / {len(val_pairs)} val")

    model = BitDecoder()
    model.load_state_dict(torch.load(pretrained_weights, map_location=device))
    model.to(device)

    aruco_dict = ArucoDictionary()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_marker_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for images, bits_gt in train_loader:
            images, bits_gt = images.to(device), bits_gt.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, bits_gt)
            loss.backward()
            optimizer.step()

        model.eval()
        bit_correct = 0
        marker_correct = 0
        hamming_correct = 0
        n_val_samples = 0
        with torch.no_grad():
            for images, bits_gt in val_loader:
                images, bits_gt = images.to(device), bits_gt.to(device)
                logits = model(images)
                preds = (torch.sigmoid(logits) > 0.5).float()

                bit_correct += (preds == bits_gt).sum().item()
                marker_correct += (preds == bits_gt).all(dim=1).sum().item()

                # Hamming match
                pred_np = preds.cpu().numpy().astype(np.uint8)
                gt_np = bits_gt.cpu().numpy().astype(np.uint8)
                ids_pred, _, _ = aruco_dict.match_bits_batch(pred_np, max_distance=2)
                ids_gt, _, _ = aruco_dict.match_bits_batch(gt_np, max_distance=0)
                hamming_correct += (ids_pred == ids_gt).sum()

                n_val_samples += images.size(0)

        bit_acc = bit_correct / (n_val_samples * 16)
        marker_acc = marker_correct / max(1, n_val_samples)
        hamming_acc = hamming_correct / max(1, n_val_samples)
        scheduler.step()

        if marker_acc > best_marker_acc:
            best_marker_acc = marker_acc
            torch.save(model.state_dict(), output_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"    Epoch {epoch+1}/{epochs}  bit={bit_acc:.4f}  "
                f"marker={marker_acc:.4f}  hamming={hamming_acc:.4f}"
            )

    print(f"  Best marker accuracy: {best_marker_acc:.4f} -> {output_path}")


def main():
    p = argparse.ArgumentParser(description="Fine-tune DeepArUco-PT on real crops")
    p.add_argument("--cls-dir", required=True, help="Classification crops directory")
    p.add_argument("--output-dir", default="nn-aruco-detection-test/models")
    p.add_argument("--corner-refiner-weights", required=True, help="Pretrained corner refiner")
    p.add_argument("--decoder-weights", required=True, help="Pretrained bit decoder")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--max-per-class", type=int, default=100)
    p.add_argument("--min-confirm-rate", type=float, default=0.1,
                   help="Minimum OpenCV self-confirmation rate to trust a class (0.0-1.0)")
    p.add_argument("--no-filter", action="store_true", help="Skip class validation (use all classes)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validate classes — remove FP IDs that OpenCV can't confirm
    valid_ids = None
    if not args.no_filter:
        valid_ids = validate_classes(args.cls_dir, args.min_confirm_rate)

    # Extract real training pairs
    corner_pairs, decoder_pairs = extract_real_training_pairs(
        args.cls_dir, args.max_per_class, valid_ids=valid_ids
    )

    if not corner_pairs:
        print("[ERROR] No valid training pairs extracted")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Fine-tune corner refiner
    print("\n--- Fine-tuning corner refiner ---")
    finetune_corner_refiner(
        corner_pairs,
        args.corner_refiner_weights,
        os.path.join(args.output_dir, "corner_refiner_ft.pth"),
        epochs=args.epochs,
    )

    # Fine-tune decoder
    print("\n--- Fine-tuning bit decoder ---")
    finetune_decoder(
        decoder_pairs,
        args.decoder_weights,
        os.path.join(args.output_dir, "bit_decoder_ft.pth"),
        epochs=args.epochs,
    )

    print("\nDone! Fine-tuned models saved to:", args.output_dir)


if __name__ == "__main__":
    main()
