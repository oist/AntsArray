#!/usr/bin/env python3
"""
Panorama stitching under **similarity transforms** using **SIFT key‑points**
===========================================================================
This single script:
1. Loads all `camXX*` still images in a 5 x 5 camera grid, or samples the first
   frame from all `camXX*` videos in a folder.
2. Detects SIFT features for every image.
3. Matches only 8‑connected neighbour cameras, keeps pairs with ≥ `MATCH_THRESHOLD` RANSAC inliers.
4. Builds an initial global pose graph from pairwise similarities.
5. Saves `initial_H_mats.npz`, a diagnostic mosaic, and per-pair match visualisations to `debug/`.

Requires **OpenCV ≥ 4.4** compiled with non‑free modules (for SIFT), plus `numpy`, `scipy`, and `matplotlib`.

```bash
pip install opencv-contrib-python numpy scipy matplotlib
python camera_cal/similarity_panorama.py --input /path/to/images_or_videos
```
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
from pathlib import Path
import re
from typing import List, Tuple, Dict
from collections import deque, defaultdict
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

###############################################################################
# CONFIG                                                                      #
###############################################################################

IM_PATH = "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260414_20260417_CustomAruco/calibration_dataset/set0_patterns_elevated_by_2mm/frame0/"  # folder with camXX images or videos
ARRAY_SIZE = (5, 5)          # camera grid (rows, cols)
MATCH_THRESHOLD = 20          # discard edges with fewer inliers
DEBUG = True                 # write debug artefacts
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v"}

###############################################################################
# Helper I/O                                                                  #
###############################################################################

CAM_RE = re.compile(r"cam(\d{2})")


def _cam_num(file: Path) -> int | None:
    m = CAM_RE.search(file.name)
    return int(m.group(1)) if m else None


def _cam_sort_key(file: Path):
    cam_num = _cam_num(file)
    return cam_num if cam_num is not None else 1e9


def _one_file_per_camera(files: List[Path]) -> List[Path]:
    """Choose one input per camXX, preferring the first lexicographic file."""
    grouped: Dict[int, List[Path]] = defaultdict(list)
    no_cam: List[Path] = []
    for f in files:
        cam_num = _cam_num(f)
        if cam_num is None:
            no_cam.append(f)
        else:
            grouped[cam_num].append(f)

    if not grouped:
        return sorted(files)

    chosen = []
    for cam_num in sorted(grouped):
        matches = sorted(grouped[cam_num])
        if len(matches) > 1:
            print(
                f"cam{cam_num:02d}: found {len(matches)} inputs; using {matches[0].name}"
            )
        chosen.append(matches[0])

    if no_cam:
        print(f"Ignoring {len(no_cam)} files without camXX in the name.")
    return chosen


def _first_video_frame_gray(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise IOError(f"Could not read first frame from video: {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def list_images(path: str) -> Tuple[List[Path], List[np.ndarray]]:
    """Return sorted filenames + grayscale calibration images.

    If still images are present, they are used. Otherwise, first frames are
    extracted from videos. This avoids creating intermediate frame files.
    """
    p = Path(path)
    if not p.is_dir():
        raise NotADirectoryError(p)

    all_files = [f for f in p.iterdir() if f.is_file() and not f.name.startswith(".")]
    files = [f for f in all_files if f.suffix.lower() in IMAGE_SUFFIXES]
    source = "image"

    if not files:
        files = [f for f in all_files if f.suffix.lower() in VIDEO_SUFFIXES]
        source = "video"

    files = _one_file_per_camera(files)
    files.sort(key=_cam_sort_key)
    if not files:
        raise RuntimeError(
            f"No image or video files found in {p}. "
            f"Image suffixes: {sorted(IMAGE_SUFFIXES)}; video suffixes: {sorted(VIDEO_SUFFIXES)}"
        )

    if source == "image":
        imgs = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
    else:
        imgs = [_first_video_frame_gray(f) for f in files]

    if any(i is None for i in imgs):
        raise IOError(f"Some {source} inputs failed to load - check paths/permissions.")
    return files, imgs


def neighbours(array_size: Tuple[int, int]):
    rows, cols = array_size
    layout = np.arange(rows * cols).reshape(rows, cols)
    edges = []
    for r in range(rows):
        for c in range(cols):
            me = layout[r, c]
            for rr in range(max(r - 1, 0), min(r + 2, rows)):
                for cc in range(max(c - 1, 0), min(c + 2, cols)):
                    other = layout[rr, cc]
                    if other > me:  # undirected without duplicates
                        edges.append((me, other))
    return edges

###############################################################################
# Feature detection & matching                                                #
###############################################################################

def sift_detector(n=3000):
    """Create a SIFT detector (fallback to ORB if SIFT unavailable)."""
    try:
        return cv2.SIFT_create(nfeatures=n)
    except AttributeError:
        print("⚠️  SIFT missing – falling back to ORB (accuracy ↓).")
        return cv2.ORB_create(nfeatures=n)


def detect_all(imgs: List[np.ndarray]):
    det = sift_detector()
    keypoints, descriptors = [], []
    for im in imgs:
        kp, d = det.detectAndCompute(im, None)
        keypoints.append(kp)
        descriptors.append(d)
    return keypoints, descriptors


def match_features(d1, d2, ratio=0.75):
    if d1 is None or d2 is None:
        return []
    if d1.dtype != np.float32:
        d1, d2 = map(np.float32, (d1, d2))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(d1, d2, k=2)
    return [m for m, n in raw if m.distance < ratio * n.distance]


def estimate_similarity(kp1, kp2, matches):
    if len(matches) < 4:
        return None, np.empty((0, 2)), np.empty((0, 2))
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, inl = cv2.estimateAffinePartial2D(
        pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=4,
        maxIters=2000, confidence=0.99
    )
    if H is None:
        return None, np.empty((0, 2)), np.empty((0, 2))
    inl = inl.ravel().astype(bool)
    return np.vstack([H, [0, 0, 1]]), pts1[inl], pts2[inl]


def draw_matches(im1, im2, pts1, pts2, out_path: Path):
    h1, w1 = im1.shape
    h2, w2 = im2.shape
    canvas = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    canvas[:h1, :w1] = im1
    canvas[:h2, w1:] = im2
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for p1, p2 in zip(pts1, pts2):
        p1i = tuple(np.int32(p1))
        p2i = tuple(np.int32(p2 + np.array([w1, 0])))
        cv2.circle(canvas, p1i, 4, (0, 255, 0), -1)
        cv2.circle(canvas, p2i, 4, (0, 255, 0), -1)
        cv2.line(canvas, p1i, p2i, (255, 0, 0), 1)
    cv2.imwrite(str(out_path), canvas)


def compute_pairwise_similarities(
    imgs, kps, desc, edge_all, out_root: Path, match_thresh: int
):
    """Return edges, inlier‑pairs and per‑edge similarity matrices."""
    debug_dir = out_root / "debug"
    debug_dir.mkdir(exist_ok=True)

    edges: List[Tuple[int, int]] = []
    pairs: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    edges_H: Dict[Tuple[int, int], np.ndarray] = {}

    for i, j in edge_all:
        matches = match_features(desc[i], desc[j])
        H, Pi, Pj = estimate_similarity(kps[i], kps[j], matches)
        if H is None or Pi.shape[0] < match_thresh:
            print(f"Edge {i}-{j}: skipped – {Pi.shape[0]} inliers")
            continue
        edges.append((i, j))
        pairs[(i, j)] = (Pi, Pj)
        edges_H[(i, j)] = H
        print(f"Edge {i}-{j}: {Pi.shape[0]} inliers")

        if DEBUG:
            out_img = debug_dir / f"match_cam{i:02d}_cam{j:02d}.png"
            draw_matches(imgs[i], imgs[j], Pi, Pj, out_img)

    return edges, pairs, edges_H


###############################################################################
# Global transforms                                                           #
###############################################################################

def global_transforms(n: int, edges_H: Dict[Tuple[int, int], np.ndarray],CENTER=12):
    Hs = [None] * n
    Hs[CENTER] = np.eye(3)
    adj: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for (i, j), H_ij in edges_H.items():
        adj[i].append((j, H_ij))
        adj[j].append((i, np.linalg.inv(H_ij)))
    q = deque([CENTER])
    while q:
        i = q.popleft()
        for j, Hji in adj[i]:
            if Hs[j] is None and Hs[i] is not None:
                Hs[j] = Hs[i] @ Hji
                q.append(j)
    for k in range(n):
        if Hs[k] is None:
            Hs[k] = np.eye(3)
    return Hs

###############################################################################
# Warping & Blending                                                          #
###############################################################################

def warp_and_blend(imgs, Hs):
    all_xy = []
    for im, H in zip(imgs, Hs):
        h, w = im.shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
        ch = np.column_stack([corners, np.ones(4)])
        warped = (H @ ch.T)[:2].T
        warped /= (H @ ch.T)[2].reshape(-1, 1)
        all_xy.append(warped)
    all_xy = np.vstack(all_xy)
    xmin, ymin = all_xy.min(0)
    xmax, ymax = all_xy.max(0)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
    out_w, out_h = int(np.ceil(xmax - xmin)), int(np.ceil(ymax - ymin))

    mosaic = np.zeros((out_h, out_w), float)
    weight = np.zeros_like(mosaic)

    for im, H in zip(imgs, Hs):
        Htot = T @ H
        warp = cv2.warpPerspective(im.astype(float), Htot, (out_w, out_h))
        m = (warp > 0).astype(float)
        mosaic += warp
        weight += m

    mosaic /= np.maximum(weight, 1)
    return np.clip(mosaic, 0, 255).astype(np.uint8)

###############################################################################
# Main                                                                        #
###############################################################################


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build initial similarity panorama transforms from camXX images or videos."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(IM_PATH),
        help="Folder containing camXX still images or videos. Videos use their first frame.",
    )
    return parser.parse_args()


def main():
    """Run the full stitching pipeline."""
    args = parse_args()
    input_path = Path(args.input)
    out_root = input_path
    debug_dir = out_root / "debug"
    if DEBUG:
        debug_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Load images
    # ---------------------------------------------------------------------
    files, imgs = list_images(str(input_path))
    n = len(imgs)
    print(f"Loaded {n} calibration images from {input_path}")
    if n == 0:
        raise RuntimeError("No inputs found - check --input and file pattern.")

    # ---------------------------------------------------------------------
    # 2. Build neighbour graph (8‑connected grid)
    # ---------------------------------------------------------------------
    edge_all = neighbours(ARRAY_SIZE)

    # ---------------------------------------------------------------------
    # 3. Detect SIFT features for every image
    # ---------------------------------------------------------------------
    print("Detecting SIFT features …")
    kps, desc = detect_all(imgs)

    # ---------------------------------------------------------------------
    # 4. Pairwise similarity estimation for neighbour edges
    # ---------------------------------------------------------------------
    print("Matching neighbour pairs …")
    edges, pairs, edges_H = compute_pairwise_similarities(
        imgs, kps, desc, edge_all, out_root, MATCH_THRESHOLD
    )
    if not edges:
        raise RuntimeError("No valid edges after SIFT matching – abort.")

    # ---------------------------------------------------------------------
    # 5. Initial global transforms from the pairwise similarity graph
    # ---------------------------------------------------------------------
    Hs_initial = global_transforms(n, edges_H)
    
    # Save homography matrices in the same format as refine_homographies.py
    np.savez_compressed(out_root/"initial_H_mats.npz", H=np.stack(Hs_initial))
    print(f"Saved initial homography matrices to: {out_root/'initial_H_mats.npz'}")
    
    mosaic_initial = warp_and_blend(imgs, Hs_initial)
    if DEBUG:
        cv2.imwrite(str(debug_dir / "mosaic_initial.png"), mosaic_initial)

   

if __name__ == "__main__":
    main()
