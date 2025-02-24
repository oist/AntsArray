#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:30:16 2025

@author: sam
"""

import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

BATCH_SIZE = 16
data_path = '/bucket/ReiterU/sam/ant_tracking/aruco_imgs/train_dataset/'
output = '/work/ReiterU/sam/antnet.pth'

def simulate_aruco_view(image, degree=0.2):
    """
    Simulates an ArUco tag viewed from different angles by applying a 3D perspective transformation.
    """
    width, height = image.size

    # Define the source points (original corners of the image)
    src_points = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    max_shift_x = width * degree
    max_shift_y = height * degree
    dst_points = np.float32([
        [random.uniform(-max_shift_x, max_shift_x), random.uniform(-max_shift_y, max_shift_y)],  
        [width + random.uniform(-max_shift_x, max_shift_x), random.uniform(-max_shift_y, max_shift_y)],  
        [width + random.uniform(-max_shift_x, max_shift_x), height + random.uniform(-max_shift_y, max_shift_y)],  
        [random.uniform(-max_shift_x, max_shift_x), height + random.uniform(-max_shift_y, max_shift_y)]
    ])

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_np = np.array(image)

    if len(img_np.shape) == 2:  # Grayscale
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    mean_color = tuple(map(int, img_np.mean(axis=(0, 1))))
    transformed_img = cv2.warpPerspective(
        img_np, matrix, (width, height), 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=mean_color
    )
    transformed_img_pil = Image.fromarray(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
    return transformed_img_pil

# ---------- 1) Define train and validation transforms ----------
train_transforms = transforms.Compose([
    transforms.Lambda(lambda img: simulate_aruco_view(img, degree=0.3)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- 2) Load the full dataset (without any transform first) ----------
full_dataset = datasets.ImageFolder(root=data_path)

# Save class names for reference
class_names = full_dataset.classes
np.save(output + '_classnames.npy', class_names)

# ---------- 3) Split indices into train and val subsets ----------
train_percentage = 0.8
train_size = int(train_percentage * len(full_dataset))
val_size = len(full_dataset) - train_size

train_indices, val_indices = random_split(
    range(len(full_dataset)), [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # for reproducibility
)

# ---------- 4) Create Subset objects each with its own transform ----------
train_dataset = Subset(full_dataset, train_indices)
val_dataset   = Subset(full_dataset, val_indices)

# Wrap each Subset with a transform by defining a small helper class
# Alternatively, you could create new ImageFolders with "transform=" each, 
# but here is a simple wrapper:
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y
    def __len__(self):
        return len(self.subset)

train_dataset = TransformSubset(train_dataset, train_transforms)
val_dataset   = TransformSubset(val_dataset, val_transforms)

# ---------- 5) Compute weights for WeightedRandomSampler on the TRAIN SET ----------
# Extract class labels from train_dataset subset
train_labels = [full_dataset.targets[i] for i in train_indices]

class_sample_count = np.array([
    len(np.where(np.array(train_labels) == t)[0]) 
    for t in np.unique(train_labels)
])
weight_per_class = 1.0 / class_sample_count
samples_weight = np.array([weight_per_class[label] for label in train_labels])
samples_weight = torch.from_numpy(samples_weight).float()

train_sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight))

# For validation, we typically do NOT use weighted sampling, 
# so that metrics reflect real-world distribution.
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          sampler=train_sampler, num_workers=4)

# ---------- 6) Model Setup ----------
num_classes = len(full_dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

# Freeze earlier layers
for name, param in model.named_parameters():
    if "layer3" not in name and "layer4" not in name and "fc" not in name:
        param.requires_grad = False

model.to(device)

# ---------- 7) Define Loss and Optimizer ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
num_epochs = 100
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# ---------- 8) Training and Validation Functions ----------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ---------- 9) Training Loop ----------
best_val_acc = 0.0
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), output)

    # Step the scheduler
    scheduler.step()

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

print(f"Best Validation Accuracy: {best_val_acc:.4f}")
