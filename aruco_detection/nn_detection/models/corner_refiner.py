"""Corner refinement U-Net for ArUco marker detection.

Predicts 4-channel heatmaps (one Gaussian blob per corner) from a 64x64
grayscale crop. Sub-pixel corner positions are extracted via soft-argmax.

Uses ``segmentation-models-pytorch`` (already a project dependency).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_corner_refiner(encoder: str = "resnet18") -> nn.Module:
    """Build a U-Net corner refinement model.

    Input:  (B, 1, 64, 64) grayscale crop
    Output: (B, 4, 64, 64) heatmaps — one channel per corner
    """
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,  # train from scratch (grayscale input)
        in_channels=1,
        classes=4,  # 4 corner heatmaps
    )
    return model


def soft_argmax_2d(heatmap: torch.Tensor, temperature: float = 100.0) -> torch.Tensor:
    """Extract sub-pixel coordinates from heatmaps via soft-argmax.

    The U-Net outputs heatmaps trained with MSE against Gaussian targets
    (range ~[0, 1]). We scale by temperature and apply softmax to get a
    sharp probability distribution concentrated at the peak.

    Parameters
    ----------
    heatmap : (B, C, H, W) tensor — U-Net output (MSE-trained, ~[0, 1] range)
    temperature : float
        Higher = sharper peaks, more precise.

    Returns
    -------
    coords : (B, C, 2) tensor — (x, y) per channel
    """
    B, C, H, W = heatmap.shape
    # Scale raw heatmap by temperature, then softmax to concentrate at peaks
    flat = (heatmap * temperature).view(B, C, -1)
    weights = torch.softmax(flat, dim=-1).view(B, C, H, W)

    grid_y = torch.arange(H, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, H, 1)
    grid_x = torch.arange(W, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, 1, W)

    x = (weights * grid_x).sum(dim=(2, 3))
    y = (weights * grid_y).sum(dim=(2, 3))

    return torch.stack([x, y], dim=-1)  # (B, 4, 2)


def make_gaussian_heatmap(
    size: int, cx: float, cy: float, sigma: float = 2.0
) -> torch.Tensor:
    """Create a single Gaussian heatmap.

    Parameters
    ----------
    size : int
        Height and width of the heatmap.
    cx, cy : float
        Center of the Gaussian (x, y) in pixel coords.
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    hmap : (size, size) tensor in [0, 1]
    """
    y = torch.arange(size, dtype=torch.float32)
    x = torch.arange(size, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    hmap = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return hmap
