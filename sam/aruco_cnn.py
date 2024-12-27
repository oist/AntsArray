import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE=128

# Visualize a batch of data
def visualize_batch(dataloader):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Un-normalize
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.show()


# 1) Transforms (augmentations for training)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)) ,  # Adjust for ResNet input
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet normalization
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet normalization
])

# 2) Load Dataset
dataset = datasets.ImageFolder(root='/home/sam/bucket/sam/ant_tracking/aruco_imgs/train_dataset/')
train_percentage = 0.8
train_size = int(train_percentage * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

y_train_indices = train_dataset.indices
y_train = [dataset.targets[i] for i in y_train_indices]
class_sample_count = np.array(
    [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in y_train])
samples_weight = torch.from_numpy(samples_weight)
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, sampler=sampler)

y_val_indices = val_dataset.indices
y_val = [dataset.targets[i] for i in y_val_indices]
class_sample_count = np.array(
    [len(np.where(y_val== t)[0]) for t in np.unique(y_val)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in y_val])
samples_weight = torch.from_numpy(samples_weight)
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, sampler=sampler)


visualize_batch(train_loader)

# Number of classes
num_classes = len(dataset.classes)

# 3) Pre-trained ResNet-50 Model
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
# Replace the final fully connected layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Freeze earlier layers
for name, param in model.named_parameters():
    if "layer3" not in name and "layer4" not in name and "fc" not in name:
        param.requires_grad = False
        
# Send the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# 5) Training and Validation Functions
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


# 6) Training Loop
num_epochs = 20
best_val_acc = 0.0

# 4) Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_resnet50_model.pth")

    # Step the learning rate scheduler
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

print(f"Best Validation Accuracy: {best_val_acc:.4f}")
