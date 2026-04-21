"""OpenCV ArUco detector wrapped in the ``ArucoDetector`` interface.

This mirrors the detection logic in ``run_aruco.py`` (lines 116-124) so that
the classical detector can be benchmarked alongside NN detectors on an equal
footing.
"""

from __future__ import annotations

import cv2
import cv2.aruco as aruco
import numpy as np

from aruco_detection.nn_detection.base import ArucoDetector, Detection


class OpenCVArucoDetector(ArucoDetector):
    """Classical OpenCV ArUco detector."""

    def __init__(
        self,
        corner_refinement: str = "CORNER_REFINE_CONTOUR",
        adaptive_thresh_constant: int = 3,
        adaptive_thresh_win_min: int = 10,
        adaptive_thresh_win_max: int = 40,
        adaptive_thresh_win_step: int = 10,
        error_correction_rate: float = 1.0,
        min_marker_perimeter_rate: float = 0.03,
        max_marker_perimeter_rate: float = 4.0,
        dict_type: int = aruco.DICT_4X4_1000,
    ):
        self._dict_type = dict_type
        corner_map = {
            "CORNER_REFINE_NONE": aruco.CORNER_REFINE_NONE,
            "CORNER_REFINE_SUBPIX": aruco.CORNER_REFINE_SUBPIX,
            "CORNER_REFINE_CONTOUR": aruco.CORNER_REFINE_CONTOUR,
            "CORNER_REFINE_APRILTAG": aruco.CORNER_REFINE_APRILTAG,
        }

        aruco_dict = aruco.getPredefinedDictionary(dict_type)
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = corner_map[corner_refinement]
        params.adaptiveThreshConstant = adaptive_thresh_constant
        params.adaptiveThreshWinSizeMin = adaptive_thresh_win_min
        params.adaptiveThreshWinSizeMax = adaptive_thresh_win_max
        params.adaptiveThreshWinSizeStep = adaptive_thresh_win_step
        params.errorCorrectionRate = error_correction_rate
        params.minMarkerPerimeterRate = min_marker_perimeter_rate
        params.maxMarkerPerimeterRate = max_marker_perimeter_rate

        self._detector = aruco.ArucoDetector(aruco_dict, params)

    @property
    def name(self) -> str:
        dict_names = {
            aruco.DICT_4X4_50: "OpenCV(4x4_50)",
            aruco.DICT_4X4_100: "OpenCV(4x4_100)",
            aruco.DICT_4X4_250: "OpenCV(4x4_250)",
            aruco.DICT_4X4_1000: "OpenCV(4x4_1000)",
        }
        return dict_names.get(self._dict_type, "OpenCV")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        corners, ids, _rejected = self._detector.detectMarkers(gray)

        detections: list[Detection] = []
        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                corner = corners[i][0]  # (4, 2)
                centre = np.mean(corner, axis=0)
                detections.append(
                    Detection(
                        marker_id=int(marker_id),
                        x=float(centre[0]),
                        y=float(centre[1]),
                        confidence=1.0,
                        corners=corner,
                    )
                )
        return detections
