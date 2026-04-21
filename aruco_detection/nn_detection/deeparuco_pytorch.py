"""PyTorch DeepArUco-style detector: YOLO + Corner Refinement + Bit Decoding.

Three-stage pipeline reimplemented in PyTorch for DICT_4X4_1000:
1. YOLO detects marker bounding boxes (existing model)
2. U-Net refines 4 corner positions from 64x64 crops
3. Small CNN decodes the 4x4 bit pattern from 32x32 rectified markers

Falls back to OpenCV decode when the NN decoder is uncertain (Hamming > 2),
so this never performs worse than the YOLO+OpenCV hybrid.

Usage:
    detector = DeepArucoPytorchDetector(
        yolo_weights="runs/.../best.pt",
        corner_refiner_weights="nn-aruco-detection-test/models/corner_refiner.pth",
        decoder_weights="nn-aruco-detection-test/models/bit_decoder.pth",
    )
    detections = detector.detect(frame)
"""

from __future__ import annotations

import cv2
import cv2.aruco as aruco
import numpy as np
import torch

from aruco_detection.nn_detection.base import ArucoDetector, Detection
from aruco_detection.nn_detection.dict_4x4_1000 import ArucoDictionary
from aruco_detection.nn_detection.models.bit_decoder import BitDecoder
from aruco_detection.nn_detection.models.corner_refiner import (
    build_corner_refiner,
    soft_argmax_2d,
)
from aruco_detection.nn_detection.utils.tiling import (
    generate_tiles,
    nms_detections,
    remap_detections,
)


class DeepArucoPytorchDetector(ArucoDetector):
    """YOLO + U-Net corner refinement + CNN bit decoding.

    Parameters
    ----------
    yolo_weights : str
        Path to trained YOLO weights.
    corner_refiner_weights : str
        Path to corner refinement U-Net weights (.pth).
    decoder_weights : str
        Path to bit decoder CNN weights (.pth).
    confidence_threshold : float
        Minimum YOLO detection confidence.
    crop_padding : float
        Fractional padding around YOLO bbox for corner refinement crop.
    max_hamming_distance : int
        Maximum Hamming distance to accept a dictionary match.
    tile_size, tile_overlap, tile_threshold : int/float
        Tiling parameters for 4K frames.
    device : str
        PyTorch device.
    """

    CORNER_INPUT_SIZE = 64
    DECODER_INPUT_SIZE = 32

    def __init__(
        self,
        yolo_weights: str,
        corner_refiner_weights: str,
        decoder_weights: str,
        confidence_threshold: float = 0.25,
        crop_padding: float = 0.5,
        max_hamming_distance: int = 2,
        tile_size: int = 1280,
        tile_overlap: float = 0.2,
        tile_threshold: int = 2000,
        device: str = "cuda",
    ):
        from ultralytics import YOLO

        self._yolo = YOLO(yolo_weights)
        self._conf_thresh = confidence_threshold
        self._crop_padding = crop_padding
        self._max_hamming = max_hamming_distance
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._tile_threshold = tile_threshold
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Corner refinement U-Net
        self._corner_model = build_corner_refiner()
        self._corner_model.load_state_dict(
            torch.load(corner_refiner_weights, map_location=self._device)
        )
        self._corner_model.to(self._device)
        self._corner_model.eval()

        # Bit decoder CNN
        self._decoder_model = BitDecoder()
        self._decoder_model.load_state_dict(
            torch.load(decoder_weights, map_location=self._device)
        )
        self._decoder_model.to(self._device)
        self._decoder_model.eval()

        # Dictionary for Hamming matching
        self._dict = ArucoDictionary()

        # OpenCV fallback decoder
        aruco_dict_cv = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        params.adaptiveThreshConstant = 3
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 50
        params.adaptiveThreshWinSizeStep = 5
        params.minMarkerPerimeterRate = 0.01
        params.errorCorrectionRate = 1.0
        self._opencv_decoder = aruco.ArucoDetector(aruco_dict_cv, params)

    @property
    def name(self) -> str:
        return "DeepArUco-PT"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        if max(h, w) > self._tile_threshold:
            return self._detect_tiled(frame)
        return self._detect_single(frame)

    def _detect_single(self, frame: np.ndarray) -> list[Detection]:
        results = self._yolo.predict(frame, conf=self._conf_thresh, verbose=False)
        return self._process_yolo_results(results, frame)

    def _detect_tiled(self, frame: np.ndarray) -> list[Detection]:
        tiles = generate_tiles(frame, self._tile_size, self._tile_overlap)
        all_dets: list[Detection] = []
        for tile in tiles:
            results = self._yolo.predict(tile.image, conf=self._conf_thresh, verbose=False)
            tile_dets = self._process_yolo_results(results, tile.image)
            all_dets.extend(remap_detections(tile, tile_dets))
        return nms_detections(all_dets)

    def _process_yolo_results(
        self, results, frame: np.ndarray
    ) -> list[Detection]:
        """Process YOLO detections through corner refinement + bit decoding."""
        fh, fw = frame.shape[:2]

        # Collect all YOLO bboxes
        bboxes: list[tuple[float, float, float, float, float]] = []  # x1,y1,x2,y2,conf
        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                xyxy = result.boxes.xyxy[i].cpu().numpy()
                conf = float(result.boxes.conf[i].cpu().numpy())
                bboxes.append((*xyxy, conf))

        if not bboxes:
            return []

        # --- Stage 2: Corner refinement (batched) ---
        crops_64: list[np.ndarray] = []
        crop_offsets: list[tuple[int, int]] = []  # (cx1, cy1) for each crop

        for x1, y1, x2, y2, _ in bboxes:
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = bw * self._crop_padding, bh * self._crop_padding
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(fw, int(x2 + pad_x))
            cy2 = min(fh, int(y2 + pad_y))

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.ndim == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            crop_resized = cv2.resize(crop, (self.CORNER_INPUT_SIZE, self.CORNER_INPUT_SIZE))
            crops_64.append(crop_resized)
            crop_offsets.append((cx1, cy1, cx2 - cx1, cy2 - cy1))

        # Stack and run corner refinement
        batch = np.stack(crops_64).astype(np.float32) / 255.0
        batch_t = torch.from_numpy(batch).unsqueeze(1).to(self._device)  # (N, 1, 64, 64)

        with torch.no_grad():
            heatmaps = self._corner_model(batch_t)  # (N, 4, 64, 64)
            corners_norm = soft_argmax_2d(heatmaps)  # (N, 4, 2) in 64x64 space

        corners_norm = corners_norm.cpu().numpy()  # (N, 4, 2)

        # --- Stage 3: Perspective rectification + bit decoding (batched) ---
        rectified_32: list[np.ndarray] = []
        corners_full: list[np.ndarray] = []  # full-frame corners per detection

        for i, (cx1, cy1, cw, ch) in enumerate(crop_offsets):
            # Map corners from 64x64 space back to crop-local pixel space
            scale_x = cw / self.CORNER_INPUT_SIZE
            scale_y = ch / self.CORNER_INPUT_SIZE
            corners_crop = corners_norm[i].copy()
            corners_crop[:, 0] = corners_crop[:, 0] * scale_x + cx1
            corners_crop[:, 1] = corners_crop[:, 1] * scale_y + cy1
            corners_full.append(corners_crop)

            # Perspective rectification to 32x32
            src_pts = corners_crop.astype(np.float32)
            dst_pts = np.float32([
                [0, 0],
                [self.DECODER_INPUT_SIZE, 0],
                [self.DECODER_INPUT_SIZE, self.DECODER_INPUT_SIZE],
                [0, self.DECODER_INPUT_SIZE],
            ])

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            rectified = cv2.warpPerspective(
                gray, M, (self.DECODER_INPUT_SIZE, self.DECODER_INPUT_SIZE),
                borderValue=128,
            )
            rectified_32.append(rectified)

        # Batch decode
        rect_batch = np.stack(rectified_32).astype(np.float32) / 255.0
        rect_t = torch.from_numpy(rect_batch).unsqueeze(1).to(self._device)

        with torch.no_grad():
            logits = self._decoder_model(rect_t)  # (N, 16)
            bits_pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)

        # Dictionary matching
        ids, dists, rots = self._dict.match_bits_batch(bits_pred, self._max_hamming)

        # --- Assemble detections ---
        detections: list[Detection] = []
        for i, (x1, y1, x2, y2, yolo_conf) in enumerate(bboxes):
            marker_id = int(ids[i])
            hamming_dist = int(dists[i])
            corners = corners_full[i]

            # Fallback to OpenCV if NN decoder is uncertain
            if marker_id == -1:
                marker_id, corners = self._opencv_fallback(
                    frame, fw, fh, x1, y1, x2, y2
                )

            if marker_id == -1:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bbox_corners = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                )
            else:
                cx = float(corners[:, 0].mean())
                cy = float(corners[:, 1].mean())
                bbox_corners = corners

            detections.append(
                Detection(
                    marker_id=marker_id,
                    x=cx,
                    y=cy,
                    confidence=yolo_conf,
                    corners=bbox_corners,
                )
            )

        return detections

    def _opencv_fallback(
        self,
        frame: np.ndarray,
        fw: int, fh: int,
        x1: float, y1: float, x2: float, y2: float,
    ) -> tuple[int, np.ndarray | None]:
        """Try OpenCV ArUco decode on the padded YOLO bbox crop."""
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = bw * self._crop_padding, bh * self._crop_padding
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(fw, int(x2 + pad_x))
        cy2 = min(fh, int(y2 + pad_y))

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return -1, None

        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        corners_cv, ids_cv, _ = self._opencv_decoder.detectMarkers(gray)
        if ids_cv is not None and len(ids_cv) > 0:
            corner = corners_cv[0][0]
            corner_full = corner.copy()
            corner_full[:, 0] += cx1
            corner_full[:, 1] += cy1
            return int(ids_cv[0][0]), corner_full

        return -1, None
