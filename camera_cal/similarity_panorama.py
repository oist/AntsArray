#!/usr/bin/env python3
"""
Panorama stitching **under similarity transforms** with **ArUco‑marker matching**
==========================================================================
This revision replaces SIFT with `cv2.aruco` (DICT_4X4_1000) and adds rich
**debug artefacts**:

* **Pair‑wise match visualisations** – every neighbour pair that survives
  RANSAC is written as ``debug/match_camXX_camYY.png``.
* **Mosaics before/after bundle‑adjustment** – ``debug/mosaic_pre.png`` and
  ``debug/mosaic_post.png`` let you see BA’s effect immediately.

OpenCV ≥ 4.7 is required for the updated ArUco interface.

```
pip install opencv-contrib-python numpy scipy matplotlib
python similarity_panorama.py
```
"""

from __future__ import annotations

import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path
import re
from typing import List, Tuple, Dict
from collections import deque, defaultdict
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

############### CONFIG ########################################################

IM_PATH = (
    "/home/sam-reiter/bucket/ReiterU/sam/ant_tracking/frame_1"
)  # folder with camXX*.tiff
ARRAY_SIZE = (5, 5)                    # grid layout (rows, cols)
MIN_COMMON_MARKERS = 1                 # discard edges with < N shared IDs
DEBUG = True                           # write debug PNGs

ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
###############################################################################
# Helper I/O                                                                   #
###############################################################################

def list_images(path: str) -> Tuple[List[Path], List[np.ndarray]]:
    """Load *.tiff images and sort by cam‑index."""
    p = Path(path)
    files = [f for f in p.glob("*.tiff") if not f.name.startswith(".")]
    cam_re = re.compile(r"cam(\d{2})")

    def key(file):
        m = cam_re.search(file.name)
        return int(m.group(1)) if m else 1e9

    files.sort(key=key)
    imgs = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
    if any(i is None for i in imgs):
        raise IOError("Some images failed to load – check paths & codecs.")
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
                    if other > me:
                        edges.append((me, other))
    return edges


###############################################################################
# ArUco detection & matching                                                   #
###############################################################################

def detect_aruco_all(imgs: List[np.ndarray]):
    """Return per‑image dict *id → centre*(x,y)."""
    id_maps: List[Dict[int, np.ndarray]] = []
    for idx, im in enumerate(imgs):
        corners, ids, _ = aruco.detectMarkers(im, ARUCO_DICT)
        d: Dict[int, np.ndarray] = {}
        if ids is not None:
            ids = ids.flatten()
            for cid, c in zip(ids, corners):
                # c shape (4,1,2) or (4,2)
                pts = c.reshape(-1, 2)
                centre = pts.mean(axis=0)
                d[int(cid)] = centre
        id_maps.append(d)
        print(f"Image {idx}: {len(d)} markers")
    return id_maps


def match_markers(id_maps, i: int, j: int):
    """Return common‑ID centres for images *i* and *j*."""
    ids_i, ids_j = id_maps[i], id_maps[j]
    common = set(ids_i) & set(ids_j)
    if len(common) < MIN_COMMON_MARKERS:
        return None, None
    Pi = np.array([ids_i[k] for k in common], float)
    Pj = np.array([ids_j[k] for k in common], float)
    return Pi, Pj


def estimate_similarity_pts(Pi: np.ndarray, Pj: np.ndarray):
    """Return H (3×3) + inlier subsets via RANSAC."""
    if Pi.shape[0] < 2:
        return None, np.empty((0, 2)), np.empty((0, 2))
    H, inl = cv2.estimateAffinePartial2D(
        Pj, Pi, method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=2000, confidence=0.995
    )
    if H is None:
        return None, np.empty((0, 2)), np.empty((0, 2))
    inl = inl.ravel().astype(bool)
    return np.vstack([H, [0, 0, 1]]), Pi[inl], Pj[inl]


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


###############################################################################
# Bundle adjustment                                                            #
###############################################################################

def params_to_H(p):
    a, b, tx, ty = p
    a += 1.0  # keep matrix non‑singular at p=0
    return np.array([[a, b, tx], [b, a, ty], [0.0, 0.0, 1.0]], float)


def unpack_params(p, n):
    Hs = [np.eye(3)]
    for i in range(n - 1):
        Hs.append(params_to_H(p[4 * i : 4 * (i + 1)]))
    return Hs


def residuals(p, n, edges, pairs, sigma):
    Hs = unpack_params(p, n)
    res = []
    for (i, j), (Pi, Pj) in pairs.items():
        H = Hs[i] @ np.linalg.inv(Hs[j])  # j → i
        Pj_h = np.vstack((Pj.T, np.ones(Pj.shape[0])))
        pred = (H @ Pj_h)[:2] / (H @ Pj_h)[2]
        diff = (pred.T - Pi).ravel() / sigma
        res.append(diff)
    return np.concatenate(res)


def bundle_adjust(n, edges, pairs, sigmas=(1000, 100, 10)):
    p = np.zeros(4 * (n - 1))
    for s in sigmas:
        fun = lambda x: residuals(x, n, edges, pairs, s)
        p = least_squares(fun, p, method="lm", max_nfev=3000).x
    return unpack_params(p, n)


###############################################################################
# Build initial global transforms (pre‑BA)                                     #
###############################################################################

def global_transforms(n: int, edges_H: Dict[Tuple[int, int], np.ndarray]):
    Hs = [None] * n
    Hs[0] = np.eye(3)
    adj: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for (i, j), H_ij in edges_H.items():
        adj[i].append((j, H_ij))           # j→i
        adj[j].append((i, np.linalg.inv(H_ij)))  # i→j
    q = deque([0])
    while q:
        i = q.popleft()
        for j, Hji in adj[i]:
            if Hs[j] is None and Hs[i] is not None:
                Hs[j] = Hs[i] @ Hji
                q.append(j)
    # fallbacks
    for k in range(n):
        if Hs[k] is None:
            Hs[k] = np.eye(3)
    return Hs


###############################################################################
# Warping & blending                                                           #
###############################################################################

def warp_and_blend(imgs, Hs):
    all_xy = []
    for im, H in zip(imgs, Hs):
        h, w = im.shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
        corners_h = np.column_stack([corners, np.ones(4)])
        warped = (H @ corners_h.T)[:2].T
        warped /= (H @ corners_h.T)[2].reshape(-1, 1)
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
        mask = (warp > 0).astype(float)
        mosaic += warp
        weight += mask
    mosaic /= np.maximum(weight, 1)
    return np.clip(mosaic, 0, 255).astype(np.uint8)


###############################################################################
# Main                                                                         #
###############################################################################

def main():
    out_root = Path(IM_PATH)
    debug_dir = out_root / "debug"
    if DEBUG:
        debug_dir.mkdir(exist_ok=True)

    files, imgs = list_images(IM_PATH)
    n = len(imgs)
    print(f"Loaded {n} images from {IM_PATH}")

    # 1 neighbours
    edge_all = neighbours(ARRAY_SIZE)

    # 2 detect markers
    id_maps = detect_aruco_all(imgs)

    # 3 pairwise similarities
    edges: List[Tuple[int, int]] = []
    pairs: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    edges_H: Dict[Tuple[int, int], np.ndarray] = {}

    for i, j in edge_all:
        Pi, Pj = match_markers(id_maps, i, j)
        if Pi is None:
            print(f"Edge {i}-{j}: skipped – {MIN_COMMON_MARKERS} markers not found")
            continue
        H, Pi_in, Pj_in = estimate_similarity_pts(Pi, Pj)
        if H is None or Pi_in.shape[0] < MIN_COMMON_MARKERS:
            print(f"Edge {i}-{j}: RANSAC failed or too few inliers")
            continue
        edges.append((i, j))
        pairs[(i, j)] = (Pi_in, Pj_in)
        edges_H[(i, j)] = H
        print(f"Edge {i}-{j}: {Pi_in.shape[0]} inliers")

        if DEBUG:
            out_img = debug_dir / f"match_cam{i:02d}_cam{j:02d}.png"
            draw_matches(imgs[i], imgs[j], Pi_in, Pj_in, out_img)

    if not edges:
        raise RuntimeError("No valid edges after ArUco matching – abort.")

    # 4 initial global transforms & pre‑BA mosaic
    Hs_pre = global_transforms(n, edges_H)
    mosaic_pre = warp_and_blend(imgs, Hs_pre)
    if DEBUG:
        cv2.imwrite(str(debug_dir / "mosaic_pre.png"), mosaic_pre)

    # 5 bundle adjustment
    Hs_post = bundle_adjust(n, edges, pairs)

    mosaic_post = warp_and_blend(imgs, Hs_post)

    out = out_root / "mosaic_similarity.png"
    cv2.imwrite(str(out), mosaic_post)
    if DEBUG:
        cv2.imwrite(str(debug_dir / "mosaic_post.png"), mosaic_post)

    print("--- Summary ---")
    print(f"Pre‑BA mosaic → {debug_dir/'mosaic_pre.png'}")
    print(f"Post‑BA mosaic → {debug_dir/'mosaic_post.png'}")
    print(f"Panorama saved → {out}")

    # quick visualisation
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Before BA")
    plt.imshow(mosaic_pre, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("After BA")
    plt.imshow(mosaic_post, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
