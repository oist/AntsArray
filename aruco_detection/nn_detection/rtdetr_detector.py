"""RT-DETR (Real-Time Detection Transformer) ArUco detector.

Uses the same two-phase approach as the YOLO detector:
1. RT-DETR detects marker bounding boxes (single class).
2. ResNet50 classifier identifies the marker ID from the crop.

RT-DETR advantages over YOLO:
- End-to-end transformer architecture — no NMS post-processing
- Better multi-scale feature handling
- Fewer hyperparameters to tune

Available via Ultralytics:
    yolo detect train data=data.yaml model=rtdetr-l.pt epochs=100 imgsz=640

Reference:
    https://arxiv.org/abs/2304.08069
"""

from __future__ import annotations

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


class RTDETRArucoDetector(ArucoDetector):
    """RT-DETR + ResNet50 classifier for ArUco detection.

    Parameters
    ----------
    rtdetr_weights : str
        Path to trained RT-DETR weights (``.pt``).
    classifier_weights : str | None
        Path to trained ResNet50 classifier (``.pth``).
    class_names_path : str | None
        Path to ``.npy`` class names mapping.
    confidence_threshold : float
        Minimum RT-DETR detection confidence.
    classifier_threshold : float
        Minimum classifier softmax probability.
    tile_size : int
        Tile dimension for SAHI-style tiling.
    tile_overlap : float
        Tile overlap ratio.
    tile_threshold : int
        Frame width above which tiling is activated.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        rtdetr_weights: str,
        classifier_weights: str | None = None,
        class_names_path: str | None = None,
        confidence_threshold: float = 0.25,
        classifier_threshold: float = 0.5,
        tile_size: int = 1280,
        tile_overlap: float = 0.2,
        tile_threshold: int = 2000,
        device: str = "cuda",
    ):
        from ultralytics import RTDETR

        self._model = RTDETR(rtdetr_weights)
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
        return "RT-DETR"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]

        if max(h, w) > self._tile_threshold:
            return self._detect_tiled(frame)
        return self._detect_single(frame)

    def _detect_single(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(frame, conf=self._conf_thresh, verbose=False)
        return self._results_to_detections(results, frame)

    def _detect_tiled(self, frame: np.ndarray) -> list[Detection]:
        tiles = generate_tiles(frame, self._tile_size, self._tile_overlap)
        all_dets: list[Detection] = []

        for tile in tiles:
            results = self._model.predict(tile.image, conf=self._conf_thresh, verbose=False)
            tile_dets = self._results_to_detections(results, tile.image)
            all_dets.extend(remap_detections(tile, tile_dets))

        return nms_detections(all_dets)

    def _results_to_detections(
        self, results, frame: np.ndarray
    ) -> list[Detection]:
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

        if self._class_names is not None:
            marker_id = int(self._class_names[predicted_idx])
        else:
            marker_id = predicted_idx

        return marker_id, predicted_conf
