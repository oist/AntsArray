import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from PIL import Image

BATCH_SIZE=16
#data_path='/home/sam/bucket/sam/ant_tracking/aruco_imgs/train_dataset/'
data_path='/bucket/ReiterU/sam/ant_tracking/aruco_imgs/train_dataset/'

output='/work/ReiterU/sam/antnet.pth'
# Visualize a batch of data
def visualize_batch(dataloader):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # Unnormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    images = images * std[:, None, None] + mean[:, None, None]

    plt.figure(figsize=(16, 16))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).numpy())
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.show()

def simulate_aruco_view(image, degree=0.2):
    """
    Simulates an ArUco tag viewed from different angles by applying a 3D perspective transformation.

    Args:
        image (PIL.Image): Input image of the ArUco tag.
        max_angle (int): Maximum rotation angle in degrees for the simulation.
                         Higher values mean stronger perspective effects.

    Returns:
        PIL.Image: Transformed image simulating the view of an ArUco tag from a different angle.
    """

    width, height = image.size

    # Define the source points (original corners of the image)
    src_points = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    # Generate random perturbations for perspective transformation
    max_shift_x = width * degree # Up to 20% of width
    max_shift_y = height * degree  # Up to 20% of height

    dst_points = np.float32([
        [random.uniform(-max_shift_x, max_shift_x), random.uniform(-max_shift_y, max_shift_y)],  # Top-left
        [width + random.uniform(-max_shift_x, max_shift_x), random.uniform(-max_shift_y, max_shift_y)],  # Top-right
        [width + random.uniform(-max_shift_x, max_shift_x), height + random.uniform(-max_shift_y, max_shift_y)],  # Bottom-right
        [random.uniform(-max_shift_x, max_shift_x), height + random.uniform(-max_shift_y, max_shift_y)]  # Bottom-left
    ])

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Convert the image to a NumPy array with correct channel order (BGR for OpenCV)
    img_np = np.array(image)

    if len(img_np.shape) == 2:  # Grayscale
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    mean_color = tuple(map(int, img_np.mean(axis=(0, 1))))

    # Apply the perspective transformation with the mean color as the border
    transformed_img = cv2.warpPerspective(img_np, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=mean_color)

    # Convert back to PIL Image
    transformed_img_pil = Image.fromarray(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))

    return transformed_img_pil





# 1) Transforms (augmentations for training)
train_transforms = transforms.Compose([
    transforms.Lambda(lambda img: simulate_aruco_view(img,degree=0.3)),
    transforms.Resize((224, 224)) ,  # Adjust for ResNet input
   # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet normalization
   # transforms.ToTensor()
])

# val_transforms = transforms.Compose([
#     transforms.Resize((224, 224)), 
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet normalization
# ])

# 2) Load Dataset
dataset = datasets.ImageFolder(root=data_path)
class_names = dataset.classes
np.save(output + '_classnames.npy',class_names)

train_percentage = 0.8
train_size = int(train_percentage * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset.dataset.transform = train_transforms #changes val_dataloader too

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

#visualize_batch(train_loader)

# Number of classes
num_classes = len(dataset.classes)

# 3) Pre-trained ResNet-50 Model
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
# Replace the final fully connected layer
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)

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
num_epochs = 50
best_val_acc = 0.0

# 4) Loss Function and Optimizer
# Use the weights in the loss function
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), output)

    # Step the learning rate scheduler
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

print(f"Best Validation Accuracy: {best_val_acc:.4f}")
