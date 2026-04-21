"""Bit-pattern decoder CNN for ArUco markers.

Takes a 32x32 perspective-rectified grayscale marker image and predicts
16 logits (4x4 grid), each representing the probability that the cell
is a "1" bit (black). The output is matched against DICT_4X4_1000 with
4-rotation Hamming tolerance.

~200K parameters — very fast inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BitDecoder(nn.Module):
    """Small CNN: 32x32x1 → 16 bit logits."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 → 16x16
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 16x16 → 8x8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 8x8 → 4x4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 16),  # 4x4 = 16 bit logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, 1, 32, 32) grayscale rectified marker

        Returns
        -------
        logits : (B, 16) — one logit per cell
        """
        return self.head(self.features(x))
