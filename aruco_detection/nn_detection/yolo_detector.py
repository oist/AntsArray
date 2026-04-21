"""YOLOv8/v11 based ArUco marker detector.

Two-phase approach:
1. **Detection**: Fine-tuned YOLO finds marker bounding boxes (single class).
2. **Classification**: Cropped regions are classified by marker ID using a
   ResNet50 head (reusing the architecture from ``aruco_train.py``).

For 4K frames, SAHI-style tiling is applied automatically when the frame
exceeds ``tile_threshold`` pixels on any side.

Training:
    # Generate synthetic data first
    python -m aruco_detection.nn_detection.data.synthetic_generator \\
        --output-dir data/synthetic --format both

    # Train YOLO detector (single-class "marker")
    yolo detect train data=data/synthetic/yolo/data.yaml model=yolov8n.pt epochs=100 imgsz=640

    # Train classifier using aruco_train.py or the synthetic classification dataset
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from aruco_detection.nn_detection.base import ArucoDetector, Detection
from aruco_detection.nn_detection.utils.tiling import (
    generate_tiles,
    nms_detections,
    remap_detections,
)


class YOLOArucoDetector(ArucoDetector):
    """YOLO + ResNet50 classifier for ArUco detection.

    Parameters
    ----------
    yolo_weights : str
        Path to trained YOLO weights (``.pt``).
    classifier_weights : str | None
        Path to trained ResNet50 classifier (``.pth``).  If None, detection-only
        mode — marker_id will be set to 0 for all detections.
    class_names_path : str | None
        Path to ``.npy`` file mapping class indices to marker IDs.
    confidence_threshold : float
        Minimum YOLO confidence to keep a detection.
    classifier_threshold : float
        Minimum classifier softmax probability to assign an ID.
    tile_size : int
        Tile dimension for SAHI-style tiling on large frames.
    tile_overlap : float
        Overlap ratio between adjacent tiles.
    tile_threshold : int
        Frame width above which tiling is activated.
    device : str
        PyTorch device (``"cuda"`` or ``"cpu"``).
    """

    def __init__(
        self,
        yolo_weights: str,
        classifier_weights: str | None = None,
        class_names_path: str | None = None,
        confidence_threshold: float = 0.25,
        classifier_threshold: float = 0.5,
        tile_size: int = 1280,
        tile_overlap: float = 0.2,
        tile_threshold: int = 2000,
        device: str = "cuda",
    ):
        from ultralytics import YOLO

        self._yolo = YOLO(yolo_weights)
        self._conf_thresh = confidence_threshold
        self._cls_thresh = classifier_threshold
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._tile_threshold = tile_threshold
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Classifier
        self._classifier = None
        self._class_names: np.ndarray | None = None
        if classifier_weights is not None:
            self._classifier = self._load_classifier(classifier_weights)
            if class_names_path is not None:
                self._class_names = np.load(class_names_path, allow_pickle=True)

        self._cls_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_classifier(self, weights_path: str) -> nn.Module:
        state = torch.load(weights_path, map_location=self._device)
        num_classes = state["fc.1.weight"].shape[0]

        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
        model.load_state_dict(state)
        model.to(self._device)
        model.eval()
        return model

    @property
    def name(self) -> str:
        return "YOLO"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]

        if max(h, w) > self._tile_threshold:
            return self._detect_tiled(frame)
        return self._detect_single(frame)

    def _detect_single(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO on a single image (no tiling)."""
        results = self._yolo.predict(frame, conf=self._conf_thresh, verbose=False)
        detections = self._results_to_detections(results, frame)
        return detections

    def _detect_tiled(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO on tiled sub-images, merge results."""
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
        """Convert YOLO results to Detection objects, optionally classifying IDs."""
        detections: list[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                x1, y1, x2, y2 = xyxy

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                corners = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                )

                marker_id = 0
                id_conf = conf

                if self._classifier is not None:
                    marker_id, id_conf = self._classify_crop(
                        frame, int(x1), int(y1), int(x2), int(y2)
                    )
                    if id_conf < self._cls_thresh:
                        continue

                detections.append(
                    Detection(
                        marker_id=marker_id,
                        x=float(cx),
                        y=float(cy),
                        confidence=float(id_conf),
                        corners=corners,
                    )
                )

        return detections

    def _classify_crop(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> tuple[int, float]:
        """Crop, resize, and classify a detected marker region."""
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0, 0.0

        crop = cv2.resize(crop, (224, 224))
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

        from PIL import Image

        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = self._cls_transform(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._classifier(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)

        predicted_idx = int(idx.item())
        predicted_conf = float(conf.item())

        # Map class index to marker ID
        if self._class_names is not None:
            marker_id = int(self._class_names[predicted_idx])
        else:
            marker_id = predicted_idx

        return marker_id, predicted_conf
