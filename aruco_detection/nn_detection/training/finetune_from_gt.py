#!/usr/bin/env python3
"""Fine-tune corner refiner and bit decoder using human-verified GT crops.

Uses crops from ``extract_corner_gt.py`` that passed manual review
(listed in ``_accepted.txt``). Each crop has a paired .json with
OpenCV-detected corners as ground truth.

Usage:
    python -m aruco_detection.nn_detection.training.finetune_from_gt `
        --gt-dir nn-aruco-detection-test/corner_gt_review `
        --output-dir nn-aruco-detection-test/models `
        --corner-refiner-weights nn-aruco-detection-test/models/corner_refiner.pth `
        --decoder-weights nn-aruco-detection-test/models/bit_decoder.pth `
        --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
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


CORNER_SIZE = 64
DECODER_SIZE = 32
CROP_SIZE = 128
HEATMAP_SIGMA = 2.0


def load_verified_gt(gt_dir: str) -> list[dict]:
    """Load crops that passed manual review."""
    accepted_path = os.path.join(gt_dir, "_accepted.txt")
    if not os.path.isfile(accepted_path):
        raise FileNotFoundError(f"No _accepted.txt in {gt_dir}. Run review_gui.py first.")

    with open(accepted_path) as f:
        accepted = set(f.read().strip().splitlines())

    entries: list[dict] = []
    for json_name in accepted:
        json_path = os.path.join(gt_dir, json_name)
        png_path = json_path.replace(".json", ".png")
        if not os.path.isfile(json_path) or not os.path.isfile(png_path):
            continue
        with open(json_path) as f:
            meta = json.load(f)
        meta["_png_path"] = png_path
        entries.append(meta)

    print(f"Loaded {len(entries)} verified GT crops from {gt_dir}")
    return entries


def prepare_corner_pairs(entries: list[dict]) -> list[dict]:
    """Convert verified GT entries into corner refiner training pairs."""
    pairs: list[dict] = []
    for e in entries:
        img = cv2.imread(e["_png_path"], cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (CROP_SIZE, CROP_SIZE):
            continue

        corners_local = np.array(e["corners_local"], dtype=np.float32)  # (4, 2) in 128x128
        if corners_local.shape != (4, 2):
            continue

        # Resize crop to 64x64 and scale corners
        crop_64 = cv2.resize(img, (CORNER_SIZE, CORNER_SIZE))
        scale = CORNER_SIZE / CROP_SIZE
        corners_64 = corners_local * scale

        # Generate heatmaps
        heatmaps = np.zeros((4, CORNER_SIZE, CORNER_SIZE), dtype=np.float32)
        valid = True
        for i in range(4):
            cx, cy = corners_64[i]
            if cx < 0 or cx >= CORNER_SIZE or cy < 0 or cy >= CORNER_SIZE:
                valid = False
                break
            heatmaps[i] = make_gaussian_heatmap(CORNER_SIZE, float(cx), float(cy), HEATMAP_SIGMA).numpy()

        if not valid:
            continue

        pairs.append({
            "crop": crop_64,
            "heatmaps": heatmaps,
            "corners": corners_64,
        })

    print(f"  Prepared {len(pairs)} corner training pairs")
    return pairs


def prepare_decoder_pairs(entries: list[dict]) -> list[dict]:
    """Convert verified GT entries into decoder training pairs."""
    aruco_dict = ArucoDictionary()
    pairs: list[dict] = []

    for e in entries:
        img = cv2.imread(e["_png_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        marker_id = e["marker_id"]
        corners_local = np.array(e["corners_local"], dtype=np.float32)
        if corners_local.shape != (4, 2):
            continue

        # Perspective rectify using the GT corners
        src = corners_local.astype(np.float32)
        dst = np.float32([[0, 0], [DECODER_SIZE, 0], [DECODER_SIZE, DECODER_SIZE], [0, DECODER_SIZE]])
        M = cv2.getPerspectiveTransform(src, dst)
        rectified = cv2.warpPerspective(img, M, (DECODER_SIZE, DECODER_SIZE), borderValue=128)

        gt_bits = aruco_dict.get_pattern(marker_id).flatten().astype(np.float32)

        pairs.append({
            "image": rectified,
            "bits": gt_bits,
            "marker_id": marker_id,
        })

    print(f"  Prepared {len(pairs)} decoder training pairs")
    return pairs


class CornerDataset(Dataset):
    def __init__(self, pairs: list[dict], augment: bool = True):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        crop = p["crop"].astype(np.float32) / 255.0

        if self.augment:
            if random.random() < 0.3:
                crop = crop * random.uniform(0.7, 1.3)
            if random.random() < 0.3:
                crop = crop + np.random.normal(0, 0.02, crop.shape).astype(np.float32)
            crop = np.clip(crop, 0, 1)

        return (
            torch.from_numpy(crop).unsqueeze(0),
            torch.from_numpy(p["heatmaps"]),
            torch.from_numpy(p["corners"]),
        )


class DecoderDataset(Dataset):
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
                img = img + np.random.normal(0, 0.03, img.shape).astype(np.float32)
            if random.random() < 0.3:
                ksize = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            img = np.clip(img, 0, 1)

        return (
            torch.from_numpy(img).unsqueeze(0),
            torch.from_numpy(p["bits"]),
        )


def finetune_corner_refiner(
    pairs: list[dict],
    pretrained_weights: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.1))
    train_ds = CornerDataset(pairs[n_val:], augment=True)
    val_ds = CornerDataset(pairs[:n_val], augment=False)
    n_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=True)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    model = build_corner_refiner()
    model.load_state_dict(torch.load(pretrained_weights, map_location=device))
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_err = float("inf")

    for epoch in range(epochs):
        model.train()
        for crops, hm_gt, _ in train_loader:
            crops, hm_gt = crops.to(device), hm_gt.to(device)
            optimizer.zero_grad()
            loss = criterion(model(crops), hm_gt)
            loss.backward()
            optimizer.step()

        model.eval()
        val_err = 0.0
        n = 0
        with torch.no_grad():
            for crops, _, corners_gt in val_loader:
                crops, corners_gt = crops.to(device), corners_gt.to(device)
                corners_pred = soft_argmax_2d(model(crops))
                err = torch.sqrt(((corners_pred - corners_gt) ** 2).sum(dim=-1)).mean()
                val_err += err.item() * crops.size(0)
                n += crops.size(0)

        val_err /= max(1, n)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.1))
    train_ds = DecoderDataset(pairs[n_val:], augment=True)
    val_ds = DecoderDataset(pairs[:n_val], augment=False)
    n_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=True)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    model = BitDecoder()
    model.load_state_dict(torch.load(pretrained_weights, map_location=device))
    model.to(device)

    aruco_dict = ArucoDictionary()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for images, bits_gt in train_loader:
            images, bits_gt = images.to(device), bits_gt.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), bits_gt)
            loss.backward()
            optimizer.step()

        model.eval()
        bit_correct = marker_correct = hamming_correct = 0
        n = 0
        with torch.no_grad():
            for images, bits_gt in val_loader:
                images, bits_gt = images.to(device), bits_gt.to(device)
                preds = (torch.sigmoid(model(images)) > 0.5).float()
                bit_correct += (preds == bits_gt).sum().item()
                marker_correct += (preds == bits_gt).all(dim=1).sum().item()

                ids_pred, _, _ = aruco_dict.match_bits_batch(
                    preds.cpu().numpy().astype(np.uint8), max_distance=2
                )
                ids_gt, _, _ = aruco_dict.match_bits_batch(
                    bits_gt.cpu().numpy().astype(np.uint8), max_distance=0
                )
                hamming_correct += (ids_pred == ids_gt).sum()
                n += images.size(0)

        bit_acc = bit_correct / (n * 16)
        marker_acc = marker_correct / max(1, n)
        hamming_acc = hamming_correct / max(1, n)
        scheduler.step()

        if marker_acc > best_acc:
            best_acc = marker_acc
            torch.save(model.state_dict(), output_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}  bit={bit_acc:.4f}  marker={marker_acc:.4f}  hamming={hamming_acc:.4f}")

    print(f"  Best marker accuracy: {best_acc:.4f} -> {output_path}")


def main():
    p = argparse.ArgumentParser(description="Fine-tune on human-verified GT crops")
    p.add_argument("--gt-dir", required=True, help="Directory with verified crops + _accepted.txt")
    p.add_argument("--output-dir", default="nn-aruco-detection-test/models")
    p.add_argument("--corner-refiner-weights", required=True)
    p.add_argument("--decoder-weights", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    entries = load_verified_gt(args.gt_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Corner refiner
    print("\n--- Fine-tuning corner refiner ---")
    corner_pairs = prepare_corner_pairs(entries)
    if corner_pairs:
        finetune_corner_refiner(
            corner_pairs,
            args.corner_refiner_weights,
            os.path.join(args.output_dir, "corner_refiner_gt.pth"),
            epochs=args.epochs,
        )

    # Decoder
    print("\n--- Fine-tuning bit decoder ---")
    decoder_pairs = prepare_decoder_pairs(entries)
    if decoder_pairs:
        finetune_decoder(
            decoder_pairs,
            args.decoder_weights,
            os.path.join(args.output_dir, "bit_decoder_gt.pth"),
            epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
