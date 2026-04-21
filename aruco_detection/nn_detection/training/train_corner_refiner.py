#!/usr/bin/env python3
"""Train the corner refinement U-Net.

Usage:
    python -m aruco_detection.nn_detection.training.train_corner_refiner \\
        --data-dir nn-aruco-detection-test/training_data/corners \\
        --output-dir nn-aruco-detection-test/models \\
        --epochs 100
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

from aruco_detection.nn_detection.models.corner_refiner import (
    build_corner_refiner,
    soft_argmax_2d,
)


class CornerDataset(Dataset):
    """Dataset from .npz file with crops, heatmaps, corners."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.crops = data["crops"]        # (N, 64, 64) uint8
        self.heatmaps = data["heatmaps"]  # (N, 4, 64, 64) float32
        self.corners = data["corners"]    # (N, 4, 2) float32

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx].astype(np.float32) / 255.0
        crop = torch.from_numpy(crop).unsqueeze(0)  # (1, 64, 64)
        heatmaps = torch.from_numpy(self.heatmaps[idx])  # (4, 64, 64)
        corners = torch.from_numpy(self.corners[idx])  # (4, 2)
        return crop, heatmaps, corners


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_ds = CornerDataset(os.path.join(args.data_dir, "train.npz"))
    val_ds = CornerDataset(os.path.join(args.data_dir, "val.npz"))
    n_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = build_corner_refiner(encoder=args.encoder).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params / 1e6:.1f}M")

    # Training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    output_path = os.path.join(args.output_dir, "corner_refiner.pth")

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_train = 0
        for crops, heatmaps_gt, corners_gt in train_loader:
            crops = crops.to(device)
            heatmaps_gt = heatmaps_gt.to(device)

            optimizer.zero_grad()
            heatmaps_pred = model(crops)
            loss = criterion(heatmaps_pred, heatmaps_gt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * crops.size(0)
            n_train += crops.size(0)

        # Validate
        model.eval()
        val_loss = 0.0
        val_corner_err = 0.0
        n_val = 0
        with torch.no_grad():
            for crops, heatmaps_gt, corners_gt in val_loader:
                crops = crops.to(device)
                heatmaps_gt = heatmaps_gt.to(device)
                corners_gt = corners_gt.to(device)

                heatmaps_pred = model(crops)
                loss = criterion(heatmaps_pred, heatmaps_gt)
                val_loss += loss.item() * crops.size(0)

                # Corner error (pixels)
                corners_pred = soft_argmax_2d(heatmaps_pred)  # (B, 4, 2)
                err = torch.sqrt(((corners_pred - corners_gt) ** 2).sum(dim=-1)).mean()
                val_corner_err += err.item() * crops.size(0)
                n_val += crops.size(0)

        train_loss /= n_train
        val_loss /= n_val
        val_corner_err /= n_val
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1}/{args.epochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                f"corner_err={val_corner_err:.2f}px"
            )

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Saved: {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Directory with train.npz/val.npz")
    p.add_argument("--output-dir", default="nn-aruco-detection-test/models")
    p.add_argument("--encoder", default="resnet18")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    train(p.parse_args())


if __name__ == "__main__":
    main()
