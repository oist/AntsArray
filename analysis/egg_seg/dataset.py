#!/usr/bin/env python3

import os
from typing import Callable, Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class BroodSegmentationDataset(Dataset):
    """
    PyTorch Dataset for ant brood semantic segmentation.
    Expects:
      root/
        images/
        masks/
    """

    def __init__(
        self,
        root: str,
        image_dir: str = "images",
        mask_dir: str = "masks",
        transforms: Optional[Callable] = None,
    ):
        self.image_dir = os.path.join(root, image_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        self.transforms = transforms

        self.images: List[str] = sorted(os.listdir(self.image_dir))
        self.masks: List[str] = sorted(os.listdir(self.mask_dir))

        assert len(self.images) == len(self.masks), "Image/mask count mismatch"
        for img_name, mask_name in zip(self.images, self.masks):
            assert img_name == mask_name, f"Filename mismatch: {img_name} vs {mask_name}"

        # TODO: set these according to your CVAT export
        self.color_to_class = {
            (0, 0, 0): 0,         # background
            (255, 0, 0): 1,       # egg
            (0, 255, 0): 2,       # larva
            (0, 0, 255): 3,       # cocoon
        }

    def __len__(self) -> int:
        return len(self.images)

    def _mask_rgb_to_indices(self, mask_rgb: np.ndarray) -> np.ndarray:
        """
        Convert HxWx3 RGB mask to HxW integer mask using color_to_class mapping.
        """
        h, w, _ = mask_rgb.shape
        mask_idx = np.zeros((h, w), dtype=np.uint8)

        for rgb, cls in self.color_to_class.items():
            matches = np.all(mask_rgb == np.array(rgb).reshape(1, 1, 3), axis=-1)
            mask_idx[matches] = cls

        return mask_idx

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.images[idx]
        mask_name = self.masks[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")  # segmentation mask is RGB colors

        image_np = np.array(image)
        mask_np = np.array(mask)

        # Convert RGB mask to integer labels
        mask_idx = self._mask_rgb_to_indices(mask_np)

        # Simple tensor conversion (add your own augmentations later if you like)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_idx).long()

        if self.transforms is not None:
            # You can plug Albumentations here if you want more complex aug
            image_tensor, mask_tensor = self.transforms(image_tensor, mask_tensor)

        return image_tensor, mask_tensor
