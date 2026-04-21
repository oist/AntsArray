#!/usr/bin/env python3
"""Train the bit-pattern decoder CNN.

Usage:
    python -m aruco_detection.nn_detection.training.train_bit_decoder \\
        --data-dir nn-aruco-detection-test/training_data/decoder \\
        --output-dir nn-aruco-detection-test/models \\
        --epochs 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aruco_detection.nn_detection.models.bit_decoder import BitDecoder
from aruco_detection.nn_detection.dict_4x4_1000 import ArucoDictionary


class DecoderDataset(Dataset):
    """Dataset from .npz file with rectified images and bit labels."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.images = data["images"]  # (N, 32, 32) uint8
        self.bits = data["bits"]      # (N, 16) uint8

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # (1, 32, 32)
        bits = torch.from_numpy(self.bits[idx].astype(np.float32))  # (16,)
        return img, bits


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_ds = DecoderDataset(os.path.join(args.data_dir, "train.npz"))
    val_ds = DecoderDataset(os.path.join(args.data_dir, "val.npz"))
    n_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = BitDecoder().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params / 1e3:.1f}K")

    # Training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Dictionary for full-marker accuracy evaluation
    aruco_dict = ArucoDictionary()

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0
    output_path = os.path.join(args.output_dir, "bit_decoder.pth")

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_train = 0
        for images, bits_gt in train_loader:
            images = images.to(device)
            bits_gt = bits_gt.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, bits_gt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            n_train += images.size(0)

        # Validate
        model.eval()
        val_loss = 0.0
        bit_correct = 0
        marker_correct = 0
        n_val = 0
        with torch.no_grad():
            for images, bits_gt in val_loader:
                images = images.to(device)
                bits_gt = bits_gt.to(device)

                logits = model(images)
                loss = criterion(logits, bits_gt)
                val_loss += loss.item() * images.size(0)

                # Per-bit accuracy
                preds = (torch.sigmoid(logits) > 0.5).float()
                bit_correct += (preds == bits_gt).sum().item()

                # Full-marker accuracy (all 16 bits correct)
                all_correct = (preds == bits_gt).all(dim=1).sum().item()
                marker_correct += all_correct

                n_val += images.size(0)

        train_loss /= n_train
        val_loss /= n_val
        bit_acc = bit_correct / (n_val * 16)
        marker_acc = marker_correct / n_val
        scheduler.step()

        if marker_acc > best_val_acc:
            best_val_acc = marker_acc
            torch.save(model.state_dict(), output_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1}/{args.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"bit_acc={bit_acc:.4f}  marker_acc={marker_acc:.4f}"
            )

    # Final evaluation: Hamming-distance matching on val set
    print(f"\nBest marker accuracy: {best_val_acc:.4f}")
    print(f"Saved: {output_path}")

    # Load best model and evaluate with dictionary matching
    model.load_state_dict(torch.load(output_path, map_location=device))
    model.eval()
    hamming_correct = 0
    n_eval = 0
    with torch.no_grad():
        for images, bits_gt in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
            bits_gt_np = bits_gt.numpy().astype(np.uint8)

            # Match each prediction against dictionary
            ids_pred, dists, _ = aruco_dict.match_bits_batch(preds, max_distance=2)

            # Get ground-truth IDs by matching GT bits
            ids_gt, _, _ = aruco_dict.match_bits_batch(bits_gt_np, max_distance=0)

            hamming_correct += (ids_pred == ids_gt).sum()
            n_eval += len(ids_gt)

    hamming_acc = hamming_correct / n_eval
    print(f"Dictionary match accuracy (Hamming<=2): {hamming_acc:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Directory with train.npz/val.npz")
    p.add_argument("--output-dir", default="nn-aruco-detection-test/models")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    train(p.parse_args())


if __name__ == "__main__":
    main()
