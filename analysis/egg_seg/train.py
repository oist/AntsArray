#!/usr/bin/env python3

import os
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import segmentation_models_pytorch as smp

from dataset import BroodSegmentationDataset

def get_dataloaders(
    data_root: str,
    batch_size: int = 4,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    dataset = BroodSegmentationDataset(root=data_root)

    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)  # [B, C, H, W]
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

def main():
    data_root = os.path.join("data")  # adjust if needed
    num_classes = 4  # background, egg, larva, cocoon

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders(data_root, batch_size=4)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    ).to(device)

    # CrossEntropyLoss expects raw logits and integer masks
    class_weights = torch.tensor([1.0, 2.0, 2.0, 2.0], device=device)  # upweight eggs/larvae/cocoons
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch:03d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        # simple checkpointing
        torch.save(model.state_dict(), f"checkpoints_epoch_{epoch:03d}.pth")

if __name__ == "__main__":
    main()
