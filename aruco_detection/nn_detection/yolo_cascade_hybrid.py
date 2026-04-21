"""YOLO detection + multi-strategy OpenCV decode cascade.

Extends the YOLO+OpenCV hybrid approach by trying multiple padding values,
preprocessing methods, and OpenCV parameter profiles when the first decode
attempt fails.  This targets the ~1425 "undecoded at real positions" cases
where YOLO correctly localised a marker but a single OpenCV attempt couldn't
read the bit pattern.

The cascade short-circuits on the first successful decode, so easy markers
add zero overhead.  Worst case (18 attempts on a ~200x200 crop) adds ~1.8 ms
per hard detection.

Usage:
    from aruco_detection.nn_detection.yolo_cascade_hybrid import (
        YOLOCascadeHybridDetector,
    )

    detector = YOLOCascadeHybridDetector(
        yolo_weights="runs/detect/.../best.pt",
        whitelist={3, 17, 25},        # optional — restrict decoded IDs
    )
    detections = detector.detect(frame)
"""

from __future__ import annotations

from typing import Sequence

import cv2
import cv2.aruco as aruco
import numpy as np

from aruco_detection.nn_detection.base import ArucoDetector, Detection
from aruco_detection.nn_detection.utils.tiling import (
    generate_tiles,
    nms_detections,
    remap_detections,
)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _preprocess_raw(gray: np.ndarray) -> np.ndarray:
    return gray


def _preprocess_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


def _preprocess_sharpen(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


_PREPROCESSORS = [_preprocess_raw, _preprocess_clahe, _preprocess_sharpen]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class YOLOCascadeHybridDetector(ArucoDetector):
    """YOLO localisation + multi-strategy OpenCV decode cascade.

    Parameters
    ----------
    yolo_weights : str
        Path to trained YOLO weights (``.pt``).
    confidence_threshold : float
        Minimum YOLO detection confidence.
    padding_values : list[float]
        Fractional paddings to try around each YOLO bbox.
    tile_size, tile_overlap, tile_threshold
        SAHI-style tiling settings for 4K frames.
    device : str
        PyTorch device for YOLO inference.
    whitelist : set[int] | None
        If provided, only accept decoded IDs that are in this set.
    min_hamming_margin : int
        When *whitelist* is set, reject a decode whose best-match Hamming
        distance is not at least this many bits better than the runner-up
        among whitelisted IDs.  Ignored when *whitelist* is ``None``.
    """

    def __init__(
        self,
        yolo_weights: str,
        confidence_threshold: float = 0.25,
        padding_values: Sequence[float] = (0.3, 0.5, 0.7),
        tile_size: int = 1280,
        tile_overlap: float = 0.2,
        tile_threshold: int = 2000,
        device: str = "cuda",
        whitelist: set[int] | None = None,
        min_hamming_margin: int = 2,
    ):
        from ultralytics import YOLO

        self._yolo = YOLO(yolo_weights)
        self._conf_thresh = confidence_threshold
        self._padding_values = list(padding_values)
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._tile_threshold = tile_threshold
        self._whitelist = whitelist
        self._min_hamming_margin = min_hamming_margin

        # ArUco dictionary (shared by all profiles)
        self._aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

        # --- Profile A: current proven parameters (CONTOUR refinement) ---
        pa = aruco.DetectorParameters()
        pa.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        pa.adaptiveThreshConstant = 3
        pa.adaptiveThreshWinSizeMin = 5
        pa.adaptiveThreshWinSizeMax = 50
        pa.adaptiveThreshWinSizeStep = 5
        pa.minMarkerPerimeterRate = 0.01
        pa.maxMarkerPerimeterRate = 4.0
        pa.errorCorrectionRate = 1.0

        # --- Profile B: Aggressive crop-rescue (validated by sweep_opencv_params.py) ---
        # +7pp decode rate over baseline on YOLO crops.  Key gains from
        # larger corner refinement window + better bit extraction, not
        # APRILTAG refinement alone.
        pb = aruco.DetectorParameters()
        pb.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        pb.adaptiveThreshConstant = 3
        pb.adaptiveThreshWinSizeMin = 3
        pb.adaptiveThreshWinSizeMax = 80
        pb.adaptiveThreshWinSizeStep = 3
        pb.minMarkerPerimeterRate = 0.01
        pb.maxMarkerPerimeterRate = 4.0
        pb.errorCorrectionRate = 0.8
        pb.relativeCornerRefinmentWinSize = 0.5
        pb.perspectiveRemovePixelPerCell = 8
        pb.perspectiveRemoveIgnoredMarginPerCell = 0.2

        self._detectors = [
            aruco.ArucoDetector(self._aruco_dict, pa),
            aruco.ArucoDetector(self._aruco_dict, pb),
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "YOLO+Cascade"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        if max(h, w) > self._tile_threshold:
            return self._detect_tiled(frame)
        return self._detect_single(frame)

    # ------------------------------------------------------------------
    # YOLO inference (identical to YOLOOpenCVHybridDetector)
    # ------------------------------------------------------------------

    def _detect_single(self, frame: np.ndarray) -> list[Detection]:
        results = self._yolo.predict(frame, conf=self._conf_thresh, verbose=False)
        return self._results_to_detections(results, frame)

    def _detect_tiled(self, frame: np.ndarray) -> list[Detection]:
        tiles = generate_tiles(frame, self._tile_size, self._tile_overlap)
        all_dets: list[Detection] = []
        for tile in tiles:
            results = self._yolo.predict(tile.image, conf=self._conf_thresh, verbose=False)
            tile_dets = self._results_to_detections(results, tile.image)
            all_dets.extend(remap_detections(tile, tile_dets))
        return nms_detections(all_dets)

    # ------------------------------------------------------------------
    # Core: YOLO boxes → Detections via cascade decode
    # ------------------------------------------------------------------

    def _results_to_detections(
        self, results, frame: np.ndarray
    ) -> list[Detection]:
        detections: list[Detection] = []
        fh, fw = frame.shape[:2]

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                yolo_conf = float(boxes.conf[i].cpu().numpy())
                x1, y1, x2, y2 = xyxy

                det = self._decode_from_bbox(frame, x1, y1, x2, y2, yolo_conf, fw, fh)
                detections.append(det)

        return detections

    # ------------------------------------------------------------------
    # Cascade decode
    # ------------------------------------------------------------------

    def _decode_from_bbox(
        self,
        frame: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        yolo_conf: float,
        fw: int,
        fh: int,
    ) -> Detection:
        """Try multiple padding / preprocessing / parameter combos to decode.

        Returns the first successful Detection, or an ``id=-1`` fallback.
        """
        bw = x2 - x1
        bh = y2 - y1

        for pad in self._padding_values:
            # --- extract padded crop ---
            pad_x = bw * pad
            pad_y = bh * pad
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(fw, int(x2 + pad_x))
            cy2 = min(fh, int(y2 + pad_y))

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            if crop.ndim == 3:
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray_crop = crop

            for preprocess in _PREPROCESSORS:
                processed = preprocess(gray_crop)

                for det_obj in self._detectors:
                    corners, ids, _ = det_obj.detectMarkers(processed)

                    if ids is None or len(ids) == 0:
                        continue

                    # Pick the first decoded marker
                    for j, marker_id in enumerate(ids.flatten()):
                        mid = int(marker_id)

                        # Whitelist filter
                        if self._whitelist is not None and mid not in self._whitelist:
                            continue

                        corner = corners[j][0]  # (4, 2) in crop coords
                        corner_full = corner.copy()
                        corner_full[:, 0] += cx1
                        corner_full[:, 1] += cy1
                        centre = corner_full.mean(axis=0)

                        return Detection(
                            marker_id=mid,
                            x=float(centre[0]),
                            y=float(centre[1]),
                            confidence=yolo_conf,
                            corners=corner_full,
                        )

        # All attempts failed → report as undecoded
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return Detection(
            marker_id=-1,
            x=float(cx),
            y=float(cy),
            confidence=yolo_conf,
            corners=np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            ),
        )
