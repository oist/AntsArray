"""YOLO detection + corner refinement + perspective warp + OpenCV decode.

Enhanced hybrid that adds a perspective rectification step when OpenCV
can't decode from the raw YOLO crop. Pipeline:

1. YOLO detects marker bounding box
2. Crop with padding, try OpenCV decode directly (fast path)
3. If decode fails: find corners via contour analysis in the crop,
   perspective-warp to a clean square, optionally enhance contrast,
   retry OpenCV decode on the rectified image

This targets the ~1800 "undecoded" detections from the basic hybrid where
YOLO finds the marker but perspective distortion prevents OpenCV decoding.

Usage:
    detector = YOLOWarpHybridDetector(
        yolo_weights="runs/.../best.pt",
    )
    detections = detector.detect(frame)
"""

from __future__ import annotations

import cv2
import cv2.aruco as aruco
import numpy as np

from aruco_detection.nn_detection.base import ArucoDetector, Detection
from aruco_detection.nn_detection.utils.tiling import (
    generate_tiles,
    nms_detections,
    remap_detections,
)


def _find_marker_corners(gray_crop: np.ndarray) -> np.ndarray | None:
    """Try to find 4 corners of a marker-like quadrilateral in the crop.

    Uses adaptive thresholding + contour detection to find the largest
    quadrilateral, which is likely the ArUco marker border.

    Returns (4, 2) float32 corners or None if not found.
    """
    h, w = gray_crop.shape[:2]

    # Try multiple threshold methods for robustness
    for block_size in [11, 21, 31, 51]:
        thresh = cv2.adaptiveThreshold(
            gray_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 5,
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest quadrilateral contour
        best_quad = None
        best_area = 0
        min_area = h * w * 0.05  # at least 5% of crop area
        max_area = h * w * 0.95

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
                best_quad = approx
                best_area = area

        if best_quad is not None:
            return _order_corners(best_quad.reshape(4, 2).astype(np.float32))

    return None


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    # Sort by y (top to bottom)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    # Top two points: leftmost is TL, rightmost is TR
    top = sorted_by_y[:2]
    top = top[np.argsort(top[:, 0])]
    # Bottom two points: leftmost is BL, rightmost is BR
    bot = sorted_by_y[2:]
    bot = bot[np.argsort(bot[:, 0])]

    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)


def _rectify_and_decode(
    gray_crop: np.ndarray,
    corners: np.ndarray,
    aruco_detector: aruco.ArucoDetector,
    rectify_size: int = 200,
) -> tuple[int, np.ndarray | None, float]:
    """Perspective-warp the crop to a square and try OpenCV ArUco decode.

    Returns (marker_id, corners_in_crop, confidence) or (-1, None, 0).
    """
    dst = np.float32([
        [0, 0], [rectify_size, 0],
        [rectify_size, rectify_size], [0, rectify_size],
    ])
    M = cv2.getPerspectiveTransform(corners, dst)
    rectified = cv2.warpPerspective(gray_crop, M, (rectify_size, rectify_size), borderValue=128)

    # Try decode on rectified image, with optional contrast enhancement
    for img in [rectified, _enhance_contrast(rectified)]:
        rect_corners, rect_ids, _ = aruco_detector.detectMarkers(img)
        if rect_ids is not None and len(rect_ids) > 0:
            # Map the detected corners back to crop coordinates
            inv_M = cv2.getPerspectiveTransform(dst, corners)
            for j, marker_id in enumerate(rect_ids.flatten()):
                rc = rect_corners[j][0]  # (4, 2) in rectified space
                rc_h = np.hstack([rc, np.ones((4, 1))]).T  # (3, 4)
                mapped = (inv_M @ rc_h).T  # (4, 3)
                mapped = mapped[:, :2] / mapped[:, 2:3]
                return int(marker_id), mapped.astype(np.float32), 1.0

    return -1, None, 0.0


def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE for contrast enhancement on low-contrast crops."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    return clahe.apply(gray)


class YOLOWarpHybridDetector(ArucoDetector):
    """YOLO detection + contour corner finding + perspective warp + OpenCV decode.

    Parameters
    ----------
    yolo_weights : str
        Path to trained YOLO weights.
    confidence_threshold : float
        Minimum YOLO detection confidence.
    crop_padding : float
        Fractional padding around YOLO bbox.
    rectify_size : int
        Size of the rectified square image for decoding.
    tile_size, tile_overlap, tile_threshold : int/float
        Tiling parameters for 4K frames.
    device : str
        PyTorch device for YOLO inference.
    """

    def __init__(
        self,
        yolo_weights: str,
        confidence_threshold: float = 0.25,
        crop_padding: float = 0.5,
        rectify_size: int = 200,
        tile_size: int = 1280,
        tile_overlap: float = 0.2,
        tile_threshold: int = 2000,
        device: str = "cuda",
    ):
        from ultralytics import YOLO

        self._yolo = YOLO(yolo_weights)
        self._conf_thresh = confidence_threshold
        self._crop_padding = crop_padding
        self._rectify_size = rectify_size
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._tile_threshold = tile_threshold

        # OpenCV decoder — standard params for direct decode
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        params.adaptiveThreshConstant = 3
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 50
        params.adaptiveThreshWinSizeStep = 5
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 4.0
        params.errorCorrectionRate = 1.0
        self._aruco_detector = aruco.ArucoDetector(aruco_dict, params)

        # Relaxed decoder for rectified images (marker fills the image)
        params_rect = aruco.DetectorParameters()
        params_rect.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        params_rect.adaptiveThreshConstant = 3
        params_rect.adaptiveThreshWinSizeMin = 5
        params_rect.adaptiveThreshWinSizeMax = 80
        params_rect.adaptiveThreshWinSizeStep = 5
        params_rect.minMarkerPerimeterRate = 0.005
        params_rect.maxMarkerPerimeterRate = 4.0
        params_rect.errorCorrectionRate = 1.0
        self._rect_aruco_detector = aruco.ArucoDetector(aruco_dict, params_rect)

    @property
    def name(self) -> str:
        return "YOLO+Warp"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        if max(h, w) > self._tile_threshold:
            return self._detect_tiled(frame)
        return self._detect_single(frame)

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

                # Padded crop
                bw, bh = x2 - x1, y2 - y1
                pad_x, pad_y = bw * self._crop_padding, bh * self._crop_padding
                cx1 = max(0, int(x1 - pad_x))
                cy1 = max(0, int(y1 - pad_y))
                cx2 = min(fw, int(x2 + pad_x))
                cy2 = min(fh, int(y2 + pad_y))

                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

                # --- Fast path: direct OpenCV decode on crop ---
                corners, ids, _rejected = self._aruco_detector.detectMarkers(gray_crop)
                if ids is not None and len(ids) > 0:
                    for j, marker_id in enumerate(ids.flatten()):
                        corner = corners[j][0]
                        corner_full = corner.copy()
                        corner_full[:, 0] += cx1
                        corner_full[:, 1] += cy1
                        centre = corner_full.mean(axis=0)
                        detections.append(Detection(
                            marker_id=int(marker_id),
                            x=float(centre[0]), y=float(centre[1]),
                            confidence=yolo_conf, corners=corner_full,
                        ))
                    continue

                # --- Slow path: corner detection + warp + retry ---
                quad_corners = _find_marker_corners(gray_crop)
                if quad_corners is not None:
                    mid, mapped_corners, _ = _rectify_and_decode(
                        gray_crop, quad_corners, self._rect_aruco_detector, self._rectify_size,
                    )
                    if mid >= 0 and mapped_corners is not None:
                        corner_full = mapped_corners.copy()
                        corner_full[:, 0] += cx1
                        corner_full[:, 1] += cy1
                        centre = corner_full.mean(axis=0)
                        detections.append(Detection(
                            marker_id=mid,
                            x=float(centre[0]), y=float(centre[1]),
                            confidence=yolo_conf * 0.9,  # slight penalty for warp path
                            corners=corner_full,
                        ))
                        continue

                # --- Fallback: undecoded ---
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                detections.append(Detection(
                    marker_id=-1,
                    x=float(cx), y=float(cy),
                    confidence=yolo_conf,
                    corners=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.float32),
                ))

        return detections
