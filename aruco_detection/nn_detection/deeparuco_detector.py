"""DeepArUco++ wrapper implementing the ``ArucoDetector`` interface.

DeepArUco++ is a purpose-built neural network pipeline for ArUco marker
detection that handles challenging lighting, occlusion, and perspective
distortion better than OpenCV's classical detector.

**Setup:**

1. Clone the DeepArUco repository:
       git clone https://github.com/AVAuco/deeparuco.git
       pip install -r deeparuco/requirements.txt

2. Download pre-trained models (or train on Flying-ArUco v2 dataset from
   https://zenodo.org/records/14053985).

3. Set ``deeparuco_path`` to the cloned repo root.

The wrapper handles the three-stage pipeline:
    marker detection → corner refinement → marker decoding

and converts outputs to the standard ``Detection`` format.

Reference:
    https://arxiv.org/abs/2411.05552
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from aruco_detection.nn_detection.base import ArucoDetector, Detection
from aruco_detection.nn_detection.utils.tiling import (
    generate_tiles,
    nms_detections,
    remap_detections,
)


class DeepArucoDetector(ArucoDetector):
    """DeepArUco++ three-stage pipeline wrapper.

    Parameters
    ----------
    deeparuco_path : str
        Path to the cloned ``deeparuco`` repository root.
    detection_model : str
        Path to the marker detection model weights.
    refinement_model : str
        Path to the corner refinement model weights.
    decoding_model : str
        Path to the marker decoding model weights.
    confidence_threshold : float
        Minimum detection confidence.
    tile_size : int
        Tile size for 4K frame processing.
    tile_overlap : float
        Tile overlap ratio.
    tile_threshold : int
        Frame width above which tiling is activated.
    device : str
        Compute device (``"cuda"`` or ``"cpu"``).
    """

    def __init__(
        self,
        deeparuco_path: str,
        detection_model: str,
        refinement_model: str,
        decoding_model: str,
        confidence_threshold: float = 0.5,
        tile_size: int = 1280,
        tile_overlap: float = 0.2,
        tile_threshold: int = 2000,
        device: str = "cuda",
    ):
        self._conf_thresh = confidence_threshold
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._tile_threshold = tile_threshold
        self._device = device

        # Add DeepArUco to Python path
        repo_root = Path(deeparuco_path).resolve()
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        self._detection_model_path = detection_model
        self._refinement_model_path = refinement_model
        self._decoding_model_path = decoding_model

        self._pipeline = None  # lazy-loaded

    def _ensure_loaded(self):
        """Lazy-load the DeepArUco pipeline on first use."""
        if self._pipeline is not None:
            return

        try:
            # DeepArUco++ provides these modules (exact API depends on repo version)
            from deeparuco.detection import MarkerDetector
            from deeparuco.refinement import CornerRefiner
            from deeparuco.decoding import MarkerDecoder

            self._detector = MarkerDetector(
                self._detection_model_path, device=self._device
            )
            self._refiner = CornerRefiner(
                self._refinement_model_path, device=self._device
            )
            self._decoder = MarkerDecoder(
                self._decoding_model_path, device=self._device
            )
            self._pipeline = True
        except ImportError as e:
            raise ImportError(
                f"DeepArUco++ not found. Clone https://github.com/AVAuco/deeparuco "
                f"and set deeparuco_path correctly.\n\n"
                f"Original error: {e}"
            ) from e

    @property
    def name(self) -> str:
        return "DeepArUco++"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._ensure_loaded()

        h, w = frame.shape[:2]
        if max(h, w) > self._tile_threshold:
            return self._detect_tiled(frame)
        return self._detect_single(frame)

    def _detect_single(self, frame: np.ndarray) -> list[Detection]:
        """Run the three-stage DeepArUco pipeline on a single image."""
        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame

        # Stage 1: detect marker regions
        marker_regions = self._detector.detect(frame_bgr)

        detections: list[Detection] = []
        for region in marker_regions:
            # Stage 2: refine corners
            try:
                corners = self._refiner.refine(frame_bgr, region)
            except Exception:
                corners = region.get("corners", None)

            # Stage 3: decode marker ID
            try:
                marker_id, confidence = self._decoder.decode(frame_bgr, region)
            except Exception:
                continue

            if confidence < self._conf_thresh:
                continue

            # Compute centre from corners or bounding box
            if corners is not None and len(corners) == 4:
                corner_arr = np.array(corners, dtype=np.float32)
                centre = corner_arr.mean(axis=0)
            else:
                bbox = region.get("bbox", None)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    centre = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    corner_arr = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                    )
                else:
                    continue

            detections.append(
                Detection(
                    marker_id=int(marker_id),
                    x=float(centre[0]),
                    y=float(centre[1]),
                    confidence=float(confidence),
                    corners=corner_arr if corners is not None else None,
                )
            )

        return detections

    def _detect_tiled(self, frame: np.ndarray) -> list[Detection]:
        """Run pipeline on tiled sub-images, merge results."""
        tiles = generate_tiles(frame, self._tile_size, self._tile_overlap)
        all_dets: list[Detection] = []

        for tile in tiles:
            tile_dets = self._detect_single(tile.image)
            all_dets.extend(remap_detections(tile, tile_dets))

        return nms_detections(all_dets)
