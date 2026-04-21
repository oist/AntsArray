#!/usr/bin/env python3
"""
Hybrid online ArUco detection pipeline.

Three-phase approach that accumulates training data across an experiment set:

Phase 1 — SEED: Run OpenCV on all videos to get reliable detections.
    This generates classification crops and YOLO annotations automatically.

Phase 2 — TRAIN: Fine-tune a YOLO detector + ResNet50 classifier on the
    harvested real data (optionally mixed with synthetic data).

Phase 3 — RECOVER: Run the trained NN on the same videos to find markers
    that OpenCV missed. New detections are validated and can be fed back
    into the next training round.

The model is "broad" — trained on the full experiment set, not per-video.
This matches your setup where a batch of experiments shares the same
physical markers, camera geometry, and lighting conditions.

Usage:
    # Phase 1: Harvest from existing detections
    python nn-aruco-detection-test/hybrid_pipeline.py seed \\
        --data-dir "Z:/ReiterU/Ants/basler/20251020_1_30min_vibration/data" \\
        --max-chunks 3

    # Phase 2: Train models
    python nn-aruco-detection-test/hybrid_pipeline.py train \\
        --training-data nn-aruco-detection-test/training_data

    # Phase 3: Recover missed detections
    python nn-aruco-detection-test/hybrid_pipeline.py recover \\
        --video "Z:/ReiterU/Ants/basler/20251020_1_30min_vibration/cam04*.avi" \\
        --yolo-weights nn-aruco-detection-test/models/yolo_best.pt \\
        --classifier-weights nn-aruco-detection-test/models/classifier_best.pth

    # Or run all phases:
    python nn-aruco-detection-test/hybrid_pipeline.py all \\
        --data-dir "Z:/ReiterU/Ants/basler/20251020_1_30min_vibration/data" \\
        --video "Z:/ReiterU/Ants/basler/20251020_1_30min_vibration/cam04*.avi"
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_DIR = "nn-aruco-detection-test"
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


class _TransformSubset:
    """Dataset wrapper that applies a transform to a Subset.

    Defined at module level so it is picklable by Windows multiprocessing.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset.dataset[self.subset.indices[idx]]
        return self.transform(x), y

    def __len__(self):
        return len(self.subset)


# =========================================================================
# Phase 1: SEED — Harvest training data from OpenCV detections
# =========================================================================
def phase_seed(args):
    """Harvest training data from existing OpenCV detections."""
    print("\n" + "=" * 60)
    print("PHASE 1: SEED — Harvesting training data from OpenCV")
    print("=" * 60)

    # Delegate to the harvester
    from harvest_training_data import harvest_from_chunk, harvest_from_raw_video, write_yolo_yaml

    output_dir = args.training_data_dir or TRAINING_DATA_DIR

    if args.data_dir and os.path.isdir(args.data_dir):
        chunk_videos = sorted(glob.glob(os.path.join(args.data_dir, "*_???.avi")))
        print(f"Found {len(chunk_videos)} chunk videos")

        for vpath in chunk_videos[: args.max_chunks]:
            stem = Path(vpath).stem
            aruco_csv = os.path.join(args.data_dir, f"{stem}_aruco_detections.csv")
            sleap_csv = os.path.join(args.data_dir, f"{stem}_sleap_data.csv")

            if not os.path.isfile(aruco_csv):
                continue

            print(f"\n  Processing: {Path(vpath).name}")
            harvest_from_chunk(
                vpath, aruco_csv,
                sleap_csv if os.path.isfile(sleap_csv) else None,
                output_dir, crop_size=128, sample_every=args.sample_every,
            )

    if args.video:
        video_paths = []
        for pattern in args.video:
            video_paths.extend(glob.glob(pattern))
        for vpath in video_paths:
            print(f"\n  Processing raw: {Path(vpath).name}")
            harvest_from_raw_video(vpath, output_dir, crop_size=128, n_frames=args.n_frames)

    write_yolo_yaml(output_dir)
    print("\nSeed phase complete.")


# =========================================================================
# Phase 2: TRAIN — Fine-tune YOLO detector + classifier
# =========================================================================
def phase_train(args):
    """Train YOLO detector and ResNet50 classifier on harvested data."""
    print("\n" + "=" * 60)
    print("PHASE 2: TRAIN — Fine-tuning models on real data")
    print("=" * 60)

    training_dir = args.training_data_dir or TRAINING_DATA_DIR
    model_dir = args.model_output_dir or MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    # --- YOLO detector training ---
    yolo_yaml = os.path.join(training_dir, "yolo", "data.yaml")
    if os.path.isfile(yolo_yaml):
        print("\n--- Training YOLO detector ---")
        try:
            from ultralytics import YOLO

            model = YOLO(args.yolo_base or "yolov8n.pt")
            results = model.train(
                data=yolo_yaml,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch_size,
                project=model_dir,
                name="yolo_aruco",
                exist_ok=True,
                cache="disk",   # cache resized images to disk (NVMe-friendly)
                workers=4,
            )
            best_path = os.path.join(model_dir, "yolo_aruco", "weights", "best.pt")
            print(f"YOLO model saved: {best_path}")
        except ImportError:
            print("[WARN] ultralytics not installed. Run: pip install ultralytics")
            print("  Or train manually:")
            print(f"  yolo detect train data={yolo_yaml} model=yolov8n.pt epochs=100 imgsz=640")
    else:
        print(f"[SKIP] No YOLO dataset at {yolo_yaml}")

    # --- Classifier training ---
    cls_dir = os.path.join(training_dir, "classification")
    if os.path.isdir(cls_dir):
        n_classes = len([d for d in os.listdir(cls_dir) if os.path.isdir(os.path.join(cls_dir, d))])
        if n_classes > 0:
            print(f"\n--- Training classifier ({n_classes} classes) ---")
            train_classifier(cls_dir, model_dir, args.epochs, args.batch_size)
        else:
            print("[SKIP] No classification data")
    else:
        print(f"[SKIP] No classification directory at {cls_dir}")

    print("\nTrain phase complete.")


def train_classifier(
    data_dir: str, model_dir: str, epochs: int = 50, batch_size: int = 16
):
    """Train a ResNet50 classifier on ImageFolder data.

    Reuses the architecture and training pattern from aruco_train.py.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split, Subset
    from torch.utils.data.sampler import WeightedRandomSampler
    from torchvision import datasets, models, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    # 80/20 split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_idx, val_idx = random_split(
        range(len(full_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataset = _TransformSubset(Subset(full_dataset, train_idx), train_transforms)
    val_dataset = _TransformSubset(Subset(full_dataset, val_idx), val_transforms)

    # Weighted sampling for class balance
    train_labels = [full_dataset.targets[i] for i in train_idx]
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    class_counts[class_counts == 0] = 1  # avoid division by zero
    weights = 1.0 / class_counts
    sample_weights = torch.tensor([weights[l] for l in train_labels], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # num_workers=0 on Windows to avoid multiprocessing pickle issues
    n_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)

    # Model (same architecture as aruco_train.py)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))

    # Freeze early layers
    for name, param in model.named_parameters():
        if "layer3" not in name and "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    output_path = os.path.join(model_dir, "classifier_best.pth")

    for epoch in range(epochs):
        # Train
        model.train()
        correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        # Val
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    # Save class names
    np.save(os.path.join(model_dir, "classifier_classnames.npy"), np.array(class_names))
    print(f"  Best val acc: {best_val_acc:.4f}")
    print(f"  Classifier saved: {output_path}")


# =========================================================================
# Phase 3: RECOVER — Find missed detections with the trained NN
# =========================================================================
def phase_recover(args):
    """Run the trained NN on videos to find detections OpenCV missed."""
    print("\n" + "=" * 60)
    print("PHASE 3: RECOVER — Finding missed detections with NN")
    print("=" * 60)

    if not args.yolo_weights:
        model_dir = args.model_output_dir or MODEL_DIR
        candidate = os.path.join(model_dir, "yolo_aruco", "weights", "best.pt")
        if os.path.isfile(candidate):
            args.yolo_weights = candidate
        else:
            print("[ERROR] No YOLO weights. Run 'train' phase first or provide --yolo-weights")
            return

    cls_weights = args.classifier_weights
    if not cls_weights:
        model_dir = args.model_output_dir or MODEL_DIR
        candidate = os.path.join(model_dir, "classifier_best.pth")
        if os.path.isfile(candidate):
            cls_weights = candidate

    cls_names = args.class_names
    if not cls_names:
        model_dir = args.model_output_dir or MODEL_DIR
        candidate = os.path.join(model_dir, "classifier_classnames.npy")
        if os.path.isfile(candidate):
            cls_names = candidate

    # Build NN detector
    from aruco_detection.nn_detection.yolo_detector import YOLOArucoDetector
    from aruco_detection.nn_detection.opencv_baseline import OpenCVArucoDetector

    nn_detector = YOLOArucoDetector(
        yolo_weights=args.yolo_weights,
        classifier_weights=cls_weights,
        class_names_path=cls_names,
        device=args.device,
    )
    opencv_detector = OpenCVArucoDetector()

    # Process videos
    video_paths = []
    if args.video:
        for pattern in args.video:
            video_paths.extend(glob.glob(pattern))

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for vpath in video_paths:
        vname = Path(vpath).stem
        print(f"\n  Processing: {Path(vpath).name}")

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  [SKIP] Cannot open: {vpath}")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = min(args.n_frames, total)
        indices = np.linspace(0, total - 1, n, dtype=int)

        recovery_rows = []

        for idx in tqdm(indices, desc=f"    Recover", leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            # Run both detectors
            opencv_dets = opencv_detector.detect(frame)
            nn_dets = nn_detector.detect(frame)

            # Find NN detections that OpenCV missed
            opencv_positions = {
                (d.marker_id, round(d.x), round(d.y)) for d in opencv_dets
            }

            for nd in nn_dets:
                is_new = True
                for od in opencv_dets:
                    if nd.marker_id == od.marker_id:
                        dist = np.hypot(nd.x - od.x, nd.y - od.y)
                        if dist < 50:
                            is_new = False
                            break

                recovery_rows.append({
                    "frame": int(idx),
                    "marker_id": nd.marker_id,
                    "x": round(nd.x, 1),
                    "y": round(nd.y, 1),
                    "confidence": round(nd.confidence, 3),
                    "status": "RECOVERED" if is_new else "CONFIRMED",
                    "video": Path(vpath).name,
                })

        cap.release()

        if recovery_rows:
            df = pd.DataFrame(recovery_rows)
            n_recovered = (df["status"] == "RECOVERED").sum()
            n_confirmed = (df["status"] == "CONFIRMED").sum()
            print(f"    Confirmed: {n_confirmed}, Recovered (new): {n_recovered}")

            out_path = os.path.join(RESULTS_DIR, f"recovery_{vname}.csv")
            df.to_csv(out_path, index=False)
            print(f"    Saved: {out_path}")

    print("\nRecover phase complete.")


# =========================================================================
# Main
# =========================================================================
def main():
    p = argparse.ArgumentParser(
        description="Hybrid online ArUco detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="phase", help="Pipeline phase")

    # Shared args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=str, help="Chunk data directory")
    common.add_argument("--video", type=str, nargs="*", help="Video paths/globs")
    common.add_argument("--n-frames", type=int, default=500)
    common.add_argument("--sample-every", type=int, default=10)
    common.add_argument("--max-chunks", type=int, default=3)
    common.add_argument("--training-data-dir", type=str)
    common.add_argument("--model-output-dir", type=str)
    common.add_argument("--device", type=str, default="cuda")

    # Seed
    sub.add_parser("seed", parents=[common], help="Phase 1: Harvest training data")

    # Train
    train_p = sub.add_parser("train", parents=[common], help="Phase 2: Train models")
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--batch-size", type=int, default=16)
    train_p.add_argument("--imgsz", type=int, default=640)
    train_p.add_argument("--yolo-base", type=str, default="yolov8n.pt", help="Base YOLO model")

    # Recover
    recover_p = sub.add_parser("recover", parents=[common], help="Phase 3: Recover missed detections")
    recover_p.add_argument("--yolo-weights", type=str)
    recover_p.add_argument("--classifier-weights", type=str)
    recover_p.add_argument("--class-names", type=str)

    # All phases
    all_p = sub.add_parser("all", parents=[common], help="Run all phases sequentially")
    all_p.add_argument("--epochs", type=int, default=50)
    all_p.add_argument("--batch-size", type=int, default=16)
    all_p.add_argument("--imgsz", type=int, default=640)
    all_p.add_argument("--yolo-base", type=str, default="yolov8n.pt")
    all_p.add_argument("--yolo-weights", type=str)
    all_p.add_argument("--classifier-weights", type=str)
    all_p.add_argument("--class-names", type=str)

    args = p.parse_args()

    if args.phase is None:
        p.print_help()
        return

    if args.phase in ("seed", "all"):
        phase_seed(args)
    if args.phase in ("train", "all"):
        phase_train(args)
    if args.phase in ("recover", "all"):
        phase_recover(args)


if __name__ == "__main__":
    main()
