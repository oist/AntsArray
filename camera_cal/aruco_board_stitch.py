#!/usr/bin/env python3
"""
ArUco board-grid stitching for the AntsArray 5x5 camera array.

This is the ArUco-based counterpart to ``similarity_panorama.py`` (which uses
SIFT). Because the calibration target is a single planar ArUco board, every
camera can be registered **directly to a canonical board grid** instead of
chaining pairwise homographies, so there is no accumulated drift and overlapping
cameras agree automatically.

Pipeline
--------
1. Detect DICT_APRILTAG_36h10 markers in each ``camXX`` image.
2. Map marker IDs to board grid coordinates (``col = id % BOARD_W``,
   ``row = id // BOARD_W``) and solve one image-pixel -> board homography per
   camera (RANSAC over marker centres).
3. Warp + blend all cameras into a single orthorectified global map.
4. Save per-camera homographies, the global map, a detection montage, an
   overlap-consistency report, and a timestamped log.

The saved ``aruco_H_mats.npz`` (key ``H``, shape (N,3,3), cam01..camN ->
index 0..N-1) is drop-in compatible with ``panorama_from_hmats.py``:

    python camera_cal/panorama_from_hmats.py \
        --hfile <input>/aruco_stitch/aruco_H_mats.npz \
        --images <input> --out pano.png

Requires OpenCV (with aruco) and numpy.

    python camera_cal/aruco_board_stitch.py --input /path/to/frame0
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

DEFAULT_INPUT = r"Z:\ReiterU\Ants\basler\cameraArray_calib\20260414_calibration_dataset\set0_patterns_elevated_by_2mm\frame0"
BOARD_W = 46                       # markers_x; matches GenerateArucoGrid.py and the recovered board
ARUCO_DICT = aruco.DICT_APRILTAG_36h10
CAM_RE = re.compile(r"cam(\d{2})", re.IGNORECASE)
MIN_MARKERS = 4                   # minimum markers to solve a homography
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def cam_num(path: Path) -> int | None:
    m = CAM_RE.search(path.name)
    return int(m.group(1)) if m else None


def one_per_cam(files: list[Path]) -> dict[int, Path]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for f in files:
        n = cam_num(f)
        if n is not None:
            grouped[n].append(f)
    return {n: sorted(v)[0] for n, v in grouped.items()}


def find_camera_images(in_dir: Path) -> list[Path]:
    files: list[Path] = []
    for suf in IMAGE_SUFFIXES:
        files += [Path(p) for p in glob.glob(str(in_dir / f"cam*{suf}"))]
    return files


def build_detector(args) -> aruco.ArucoDetector:
    params = aruco.DetectorParameters()
    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    # Robustness knobs (defaults reproduce the legacy behaviour). Small,
    # ink-bled markers (e.g. the 2.5 mm board) need permissive decoding:
    # sample only each cell centre and tolerate more bit/border errors.
    params.errorCorrectionRate = args.error_correction_rate
    params.maxErroneousBitsInBorderRate = args.max_border_err
    params.perspectiveRemoveIgnoredMarginPerCell = args.ignored_margin
    params.minMarkerPerimeterRate = args.min_perimeter_rate
    if args.wide_thresh:
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 51
        params.adaptiveThreshWinSizeStep = 4
    dict_id = getattr(aruco, args.dict) if isinstance(args.dict, str) else args.dict
    return aruco.ArucoDetector(aruco.getPredefinedDictionary(dict_id), params)


def detect(img: np.ndarray, det: aruco.ArucoDetector):
    """Return {id: (center_xy, corners_4x2)}."""
    corners, ids, _ = det.detectMarkers(img)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if ids is None:
        return out
    for c, i in zip(corners, ids.ravel()):
        c4 = c.reshape(4, 2).astype(np.float64)
        out[int(i)] = (c4.mean(0), c4)
    return out


def grid_coords(ids: np.ndarray, board_w: int) -> np.ndarray:
    return np.column_stack([ids % board_w, ids // board_w]).astype(np.float64)


def homography_img_to_board(dets: dict, board_w: int):
    """Solve image-pixel -> board-grid homography from marker centers."""
    ids = np.array(sorted(dets), dtype=np.int64)
    if len(ids) < MIN_MARKERS:
        return None, ids, 0, np.nan
    pix = np.array([dets[i][0] for i in ids], np.float64)
    grid = grid_coords(ids, board_w)
    H, mask = cv2.findHomography(pix, grid, cv2.RANSAC, 3.0 / 400.0)  # thresh in grid units (~3px @400px/unit)
    if H is None:
        return None, ids, 0, np.nan
    inl = int(mask.sum())
    proj = cv2.perspectiveTransform(pix.reshape(-1, 1, 2), H).reshape(-1, 2)
    resid_grid = np.sqrt(((proj - grid) ** 2).sum(1))
    return H, ids, inl, float(np.median(resid_grid[mask.ravel().astype(bool)]))


def warp_and_blend(imgs: list[np.ndarray], Hs: list[np.ndarray]):
    """Hs[i] maps image i into a common (board-pixel) frame. Average blend."""
    all_xy = []
    for im, H in zip(imgs, Hs):
        if im is None or H is None:
            continue
        h, w = im.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
        warped = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
        all_xy.append(warped)
    all_xy = np.vstack(all_xy)
    xmin, ymin = all_xy.min(0)
    xmax, ymax = all_xy.max(0)
    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], np.float64)
    out_w = int(np.ceil(xmax - xmin))
    out_h = int(np.ceil(ymax - ymin))
    print(f"Mosaic canvas: {out_w} x {out_h} px")

    mosaic = np.zeros((out_h, out_w), np.float32)
    weight = np.zeros((out_h, out_w), np.float32)
    ones = None
    for im, H in zip(imgs, Hs):
        if im is None or H is None:
            continue
        Htot = T @ H
        warped = cv2.warpPerspective(im.astype(np.float32), Htot, (out_w, out_h),
                                     flags=cv2.INTER_LINEAR)
        if ones is None or ones.shape != im.shape:
            ones = np.ones(im.shape[:2], np.float32)
        cov = cv2.warpPerspective(ones, Htot, (out_w, out_h), flags=cv2.INTER_NEAREST)
        m = cov > 0.5
        mosaic[m] += warped[m]
        weight[m] += 1.0
    weight[weight == 0] = 1.0
    mosaic /= weight
    return np.clip(mosaic, 0, 255).astype(np.uint8), T


def make_montage(imgs_color: dict[int, np.ndarray], n_cams: int, cols: int = 5, tile_w: int = 620):
    rows = int(np.ceil(n_cams / cols))
    tiles = []
    tile_h = None
    for i in range(1, n_cams + 1):
        im = imgs_color.get(i)
        if im is None:
            im = np.zeros((10, 10, 3), np.uint8)
        h, w = im.shape[:2]
        scale = tile_w / w
        t = cv2.resize(im, (tile_w, int(round(h * scale))))
        tile_h = t.shape[0] if tile_h is None else tile_h
        if t.shape[0] != tile_h:
            t = cv2.resize(t, (tile_w, tile_h))
        tiles.append(t)
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((tile_h, tile_w, 3), np.uint8))
    grid_rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
    return np.vstack(grid_rows)


def overlap_consistency(per_cam_dets, per_cam_H, board_w):
    """Disagreement (in board-pixel units of H) on shared markers between cams."""
    cams = sorted(per_cam_H)
    results = []
    for a_idx in range(len(cams)):
        for b_idx in range(a_idx + 1, len(cams)):
            a, b = cams[a_idx], cams[b_idx]
            Ha, Hb = per_cam_H[a], per_cam_H[b]
            if Ha is None or Hb is None:
                continue
            shared = sorted(set(per_cam_dets[a]) & set(per_cam_dets[b]))
            if len(shared) < 3:
                continue
            pa = np.array([per_cam_dets[a][i][0] for i in shared], np.float64)
            pb = np.array([per_cam_dets[b][i][0] for i in shared], np.float64)
            ba = cv2.perspectiveTransform(pa.reshape(-1, 1, 2), Ha).reshape(-1, 2)
            bb = cv2.perspectiveTransform(pb.reshape(-1, 1, 2), Hb).reshape(-1, 2)
            d = np.sqrt(((ba - bb) ** 2).sum(1))
            results.append((a, b, len(shared), float(np.median(d)), float(d.max())))
    return results


def run(args) -> None:
    in_dir = Path(args.input)
    out_dir = Path(args.out) if args.out else in_dir / "aruco_stitch"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== ArUco board stitch ===")
    print(f"Input : {in_dir}")
    print(f"Output: {out_dir}")
    print(f"Board width (cols): {args.board_w}   output scale: {args.scale} px/unit")

    files = find_camera_images(in_dir)
    if not files:
        raise SystemExit(f"No camXX images ({', '.join(IMAGE_SUFFIXES)}) found in {in_dir}")
    cam_files = one_per_cam(files)
    n_cams = max(cam_files)
    print(f"Found {len(cam_files)} cameras (max cam number {n_cams})")

    det = build_detector(args)
    per_cam_dets: dict[int, dict] = {}
    per_cam_img: dict[int, np.ndarray] = {}
    montage_color: dict[int, np.ndarray] = {}

    if args.dilate > 0:
        print(f"Pre-detection white dilation: {args.dilate}px ellipse "
              f"(counters black ink-bleed on tiny markers; centres unbiased)")
    dilate_kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate, args.dilate))
        if args.dilate > 0 else None
    )

    print("\n--- Detection ---")
    for cam in sorted(cam_files):
        img = cv2.imread(str(cam_files[cam]), cv2.IMREAD_GRAYSCALE)
        per_cam_img[cam] = img
        # Detect on a dilated copy when requested; keep the original for warping.
        det_img = cv2.dilate(img, dilate_kernel) if dilate_kernel is not None else img
        dets = detect(det_img, det)
        per_cam_dets[cam] = dets
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if dets:
            cs = [v[1].reshape(1, 4, 2).astype(np.float32) for v in dets.values()]
            ids = np.array(list(dets.keys())).reshape(-1, 1)
            aruco.drawDetectedMarkers(color, cs, ids)
        cv2.putText(color, f"cam{cam:02d}: {len(dets)} tags", (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)
        montage_color[cam] = color
        print(f"cam{cam:02d}: {len(dets):3d} markers")

    print("\n--- Per-camera homography to board grid ---")
    scale_mat = np.array([[args.scale, 0, 0], [0, args.scale, 0], [0, 0, 1]], np.float64)
    per_cam_H: dict[int, np.ndarray] = {}
    for cam in sorted(per_cam_dets):
        H, ids, inl, resid = homography_img_to_board(per_cam_dets[cam], args.board_w)
        if H is None:
            per_cam_H[cam] = None
            print(f"cam{cam:02d}: FAILED (only {len(per_cam_dets[cam])} markers)")
            continue
        per_cam_H[cam] = scale_mat @ H  # image-pixel -> board-pixel
        print(f"cam{cam:02d}: {inl:3d}/{len(ids):3d} inliers, median residual {resid * args.scale:.2f} board-px")

    print("\n--- Overlap consistency (shared markers, board-px disagreement) ---")
    oc = overlap_consistency(per_cam_dets, per_cam_H, args.board_w)
    oc_strong = [r for r in oc if r[2] >= 10]
    for a, b, ns, med, mx in sorted(oc_strong, key=lambda r: -r[2])[:30]:
        print(f"cam{a:02d}-cam{b:02d}: {ns:3d} shared, median {med:.2f} px, max {mx:.2f} px")
    if oc_strong:
        meds = np.array([r[3] for r in oc_strong])
        print(f"\nOverlap median disagreement across {len(oc_strong)} pairs: "
              f"median {np.median(meds):.2f} px, p90 {np.percentile(meds, 90):.2f} px")

    print("\n--- Stitching ---")
    cams_sorted = sorted(per_cam_H)
    imgs = [per_cam_img[c] for c in cams_sorted]
    Hs = [per_cam_H[c] for c in cams_sorted]
    mosaic, _T = warp_and_blend(imgs, Hs)

    mosaic_path = out_dir / "mosaic_board.png"
    cv2.imwrite(str(mosaic_path), mosaic)
    print(f"Saved global map: {mosaic_path}  ({mosaic.shape[1]}x{mosaic.shape[0]})")

    if mosaic.shape[1] > args.max_preview:
        s = args.max_preview / mosaic.shape[1]
        prev = cv2.resize(mosaic, (args.max_preview, int(mosaic.shape[0] * s)))
        cv2.imwrite(str(out_dir / "mosaic_board_preview.png"), prev)
        print(f"Saved preview: {out_dir / 'mosaic_board_preview.png'} ({prev.shape[1]}x{prev.shape[0]})")

    # Save homographies (cam01..camN -> index 0..N-1), board-pixel frame (pre-translation)
    H_stack = np.zeros((n_cams, 3, 3), np.float64)
    for i in range(n_cams):
        H = per_cam_H.get(i + 1)
        H_stack[i] = H if H is not None else np.eye(3)
    npz_path = out_dir / "aruco_H_mats.npz"
    np.savez_compressed(npz_path, H=H_stack, board_w=args.board_w, scale=args.scale,
                        note="H[i] maps cam(i+1) image pixels -> board-pixel frame (col*scale,row*scale)")
    print(f"Saved homographies: {npz_path}  shape {H_stack.shape}")

    montage = make_montage(montage_color, n_cams)
    mont_path = out_dir / "detections_montage.png"
    cv2.imwrite(str(mont_path), montage)
    print(f"Saved detection montage: {mont_path} ({montage.shape[1]}x{montage.shape[0]})")
    print("\nDONE.")


def main() -> None:
    ap = argparse.ArgumentParser(description="ArUco board-grid stitching for the 5x5 ant camera array.")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Folder with camXX images.")
    ap.add_argument("--out", default=None, help="Output dir (default: <input>/aruco_stitch).")
    ap.add_argument("--board-w", type=int, default=BOARD_W, dest="board_w", help="Board width in markers (cols). Default 46.")
    ap.add_argument("--scale", type=float, default=110.0, help="Output px per board grid unit. Default 110.")
    ap.add_argument("--max-preview", type=int, default=4000, dest="max_preview", help="Downscale preview if wider than this.")
    ap.add_argument("--dict", default="DICT_APRILTAG_36h10", help="ArUco dictionary name. Default DICT_APRILTAG_36h10.")
    # Detection robustness (defaults = OpenCV/legacy behaviour).
    ap.add_argument("--dilate", type=int, default=0,
                    help="White-dilation kernel (px) applied before detection to counter "
                         "black ink-bleed on tiny markers. 0=off. Try 7 for the 2.5mm board.")
    ap.add_argument("--error-correction-rate", type=float, default=0.6, dest="error_correction_rate")
    ap.add_argument("--max-border-err", type=float, default=0.35, dest="max_border_err",
                    help="maxErroneousBitsInBorderRate. Default 0.35.")
    ap.add_argument("--ignored-margin", type=float, default=0.13, dest="ignored_margin",
                    help="perspectiveRemoveIgnoredMarginPerCell. Raise (~0.33) to skip bled cell edges.")
    ap.add_argument("--min-perimeter-rate", type=float, default=0.03, dest="min_perimeter_rate")
    ap.add_argument("--wide-thresh", action="store_true",
                    help="Use a wide adaptive-threshold window range (5..51).")
    ap.add_argument("--small-markers", action="store_true",
                    help="Preset for tiny ink-bled markers: dilate=7, ecr=2.0, max-border-err=0.6, "
                         "ignored-margin=0.33, min-perimeter-rate=0.005, wide-thresh.")
    args = ap.parse_args()

    if args.small_markers:
        if args.dilate == 0:
            args.dilate = 7
        args.error_correction_rate = max(args.error_correction_rate, 2.0)
        args.max_border_err = max(args.max_border_err, 0.6)
        args.ignored_margin = max(args.ignored_margin, 0.33)
        args.min_perimeter_rate = min(args.min_perimeter_rate, 0.005)
        args.wide_thresh = True

    out_dir = Path(args.out) if args.out else Path(args.input) / "aruco_stitch"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_fh = open(out_dir / f"aruco_stitch_{stamp}.log", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)
    try:
        run(args)
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_fh.close()


if __name__ == "__main__":
    main()
