"""SAHI-style image tiling for small-object detection on 4K frames.

ArUco markers on ants are ~70-80 px in 4096-wide frames (~1.8% of the image
width).  Running a detector at 640 or 1280 px input would miss most of them.
Instead, we slice the frame into overlapping tiles, run the detector on each
tile, and merge results with Non-Maximum Suppression across tile boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aruco_detection.nn_detection.base import Detection


@dataclass
class Tile:
    """A rectangular slice of the original frame."""

    x_offset: int
    y_offset: int
    image: np.ndarray  # (tile_h, tile_w, C) or (tile_h, tile_w)


def generate_tiles(
    frame: np.ndarray,
    tile_size: int = 1280,
    overlap_ratio: float = 0.2,
) -> list[Tile]:
    """Slice *frame* into overlapping tiles.

    Parameters
    ----------
    frame : np.ndarray
        Full-resolution image (H, W) or (H, W, 3).
    tile_size : int
        Width and height of each tile (square).
    overlap_ratio : float
        Fractional overlap between adjacent tiles (0.0 – 0.5).

    Returns
    -------
    list[Tile]
        Tiles with their offsets into the original frame.
    """
    h, w = frame.shape[:2]
    step = max(1, int(tile_size * (1 - overlap_ratio)))

    tiles: list[Tile] = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            # Shift back if we're at the edge so tiles are always tile_size
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)

            tile_img = frame[y_start:y_end, x_start:x_end]
            tiles.append(Tile(x_offset=x_start, y_offset=y_start, image=tile_img))

    return tiles


def remap_detections(
    tile: Tile, detections: list[Detection]
) -> list[Detection]:
    """Translate tile-local coordinates back to full-frame coordinates."""
    remapped: list[Detection] = []
    for d in detections:
        corners = None
        if d.corners is not None:
            corners = d.corners.copy()
            corners[:, 0] += tile.x_offset
            corners[:, 1] += tile.y_offset
        remapped.append(
            Detection(
                marker_id=d.marker_id,
                x=d.x + tile.x_offset,
                y=d.y + tile.y_offset,
                confidence=d.confidence,
                corners=corners,
            )
        )
    return remapped


def nms_detections(
    detections: list[Detection],
    distance_threshold: float = 30.0,
) -> list[Detection]:
    """Merge duplicate detections from overlapping tiles.

    For each marker ID, if two detections are within *distance_threshold* px,
    keep the one with higher confidence.  This handles the overlap regions
    where the same marker appears in multiple tiles.
    """
    if not detections:
        return []

    # Group by marker ID
    by_id: dict[int, list[Detection]] = {}
    for d in detections:
        by_id.setdefault(d.marker_id, []).append(d)

    merged: list[Detection] = []
    for _mid, group in by_id.items():
        # Sort by confidence descending
        group.sort(key=lambda d: d.confidence, reverse=True)
        keep: list[Detection] = []
        for d in group:
            is_dup = False
            for kept in keep:
                dist = np.hypot(d.x - kept.x, d.y - kept.y)
                if dist < distance_threshold:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(d)
        merged.extend(keep)

    return merged
