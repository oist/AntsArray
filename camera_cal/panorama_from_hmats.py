#!/usr/bin/env python3
"""
Build a panorama from pre-computed similarity/homography matrices.

Usage examples
--------------

# From images (same ordering as camXX*.png / camXX*.tiff / etc.)
python panorama_from_hmats.py \
    --hfile /path/to/initial_H_mats.npz \
    --images /path/to/image_dir \
    --out /path/to/panorama_from_images.png

# From videos (one video per camera; a single frame is sampled from each)
python panorama_from_hmats.py \
    --hfile /path/to/refined_H_mats.npz \
    --videos /path/to/video_dir \
    --out /path/to/panorama_from_videos.png

The H file is expected to contain an array of shape (N, 3, 3),
typically under the key 'H' if using np.savez.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# I/O helpers (UPDATED)
# ---------------------------------------------------------------------------

CAM_RE = re.compile(r"^cam(\d{2})", re.IGNORECASE)


def _is_camxx_file(p: Path) -> bool:
    return CAM_RE.match(p.stem) is not None


def _cam_num_from_path(p: Path) -> int | None:
    """
    Extract cam number XX from filename stem 'camXX...'
    Returns integer XX, or None if not matching.
    """
    m = CAM_RE.match(p.stem)
    if not m:
        return None
    return int(m.group(1))#-1 #1 based to 0 based


def _collect_cam_files_by_number(
    d: Path,
    exts: set[str],
) -> dict[int, list[Path]]:
    """
    Return mapping: cam_num (int) -> list of matching files (sorted by name).
    Only includes files whose stem starts with camXX and whose suffix is in exts.
    """
    files = [
        f for f in d.iterdir()
        if f.is_file()
        and f.suffix.lower() in exts
        and not f.name.startswith(".")
        and _is_camxx_file(f)
    ]

    by_num: dict[int, list[Path]] = {}
    for f in files:
        cam_num = _cam_num_from_path(f)
        if cam_num is None:
            continue
        by_num.setdefault(cam_num, []).append(f)

    # Deterministic selection if duplicates exist
    for k in by_num:
        by_num[k].sort(key=lambda p: p.name.lower())

    return by_num


def _white_like(shape: tuple[int, ...]) -> np.ndarray:
    """
    Create a white uint8 image with the same (H,W) shape as others.
    Assumes grayscale images.
    """
    if len(shape) < 2:
        raise ValueError(f"Invalid image shape for placeholder: {shape}")
    h, w = shape[:2]
    return np.full((h, w), 255, dtype=np.uint8)


def load_images_from_dir_aligned(
    img_dir: str | Path,
    n_expected: int,
) -> tuple[list[Path | None], list[np.ndarray]]:
    """
    Load grayscale images aligned to H_mats order.

    Mapping rule (per your request):
      - camXX corresponds to H index (XX - 1)
        e.g. cam03 -> H_mats[2]

    For each i in [0..n_expected-1], we look for cam{(i+1):02d}*.
    If missing, fill with a white image of the same size as the others.

    Returns
    -------
    files_aligned : list[Path|None]
        Length n_expected; None where placeholder used.
    imgs_aligned : list[np.ndarray]
        Length n_expected; grayscale uint8 images.
    """
    img_dir = Path(img_dir)
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    by_num = _collect_cam_files_by_number(img_dir, exts)

    # Warn about out-of-range cameras (e.g., cam99 when n_expected=4)
    out_of_range = sorted([k for k in by_num.keys() if not (1 <= k <= n_expected)])
    if out_of_range:
        print(f"Warning: ignoring camXX images out of H range 1..{n_expected}: {out_of_range}")

    # Determine reference size from any in-range image (preferred), otherwise any image at all.
    ref_img = None
    for cam_num in range(1, n_expected + 1):
        if cam_num in by_num and by_num[cam_num]:
            ref_img = cv2.imread(str(by_num[cam_num][0]), cv2.IMREAD_GRAYSCALE)
            if ref_img is not None:
                break
    if ref_img is None:
        # fallback: any camXX file (even if out of range) to get size
        any_cam = next((by_num[k][0] for k in sorted(by_num.keys()) if by_num[k]), None)
        if any_cam is not None:
            ref_img = cv2.imread(str(any_cam), cv2.IMREAD_GRAYSCALE)

    if ref_img is None:
        raise RuntimeError(f"No readable camXX image files found in {img_dir}")

    placeholder = _white_like(ref_img.shape)

    files_aligned: list[Path | None] = []
    imgs_aligned: list[np.ndarray] = []

    for i in range(n_expected):
        cam_num = i + 1  # cam01 -> H[0], cam02 -> H[1], ...
        candidates = by_num.get(cam_num, [])
        if not candidates:
            files_aligned.append(None)
            imgs_aligned.append(placeholder.copy())
            continue

        chosen = candidates[0]
        im = cv2.imread(str(chosen), cv2.IMREAD_GRAYSCALE)
        if im is None:
            # Treat unreadable as missing
            print(f"Warning: failed to load {chosen}; using white placeholder for cam{cam_num:02d}")
            files_aligned.append(None)
            imgs_aligned.append(placeholder.copy())
            continue

        # Optional: enforce same size (since you said they will all be the same size)
        if im.shape[:2] != placeholder.shape[:2]:
            raise ValueError(
                f"Image size mismatch for {chosen}: got {im.shape[:2]}, expected {placeholder.shape[:2]}"
            )

        files_aligned.append(chosen)
        imgs_aligned.append(im)

        if len(candidates) > 1:
            print(f"Warning: multiple files for cam{cam_num:02d}; using {chosen.name}")

    return files_aligned, imgs_aligned


def load_frames_from_videos_dir_aligned(
    vid_dir: str | Path,
    n_expected: int,
    frame_index: int | None = None,
) -> tuple[list[Path | None], list[np.ndarray]]:
    """
    Load one grayscale frame per camera video, aligned to H_mats order.

    Mapping rule:
      - camXX corresponds to H index (XX - 1)

    Missing videos are filled with a white frame of the same size as others.
    """
    vid_dir = Path(vid_dir)
    exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

    by_num = _collect_cam_files_by_number(vid_dir, exts)

    out_of_range = sorted([k for k in by_num.keys() if not (1 <= k <= n_expected)])
    if out_of_range:
        print(f"Warning: ignoring camXX videos out of H range 1..{n_expected}: {out_of_range}")

    # Determine reference size from any in-range video frame (preferred), otherwise any camXX video frame
    ref_frame = None
    for cam_num in range(1, n_expected + 1):
        if cam_num in by_num and by_num[cam_num]:
            try:
                ref_frame = extract_frame_from_video(by_num[cam_num][0], frame_index=frame_index)
                if ref_frame is not None:
                    break
            except Exception:
                pass

    if ref_frame is None:
        any_cam = next((by_num[k][0] for k in sorted(by_num.keys()) if by_num[k]), None)
        if any_cam is not None:
            ref_frame = extract_frame_from_video(any_cam, frame_index=frame_index)

    if ref_frame is None:
        raise RuntimeError(f"No readable camXX video files found in {vid_dir}")

    placeholder = _white_like(ref_frame.shape)

    files_aligned: list[Path | None] = []
    imgs_aligned: list[np.ndarray] = []

    for i in range(n_expected):
        cam_num = i + 1
        candidates = by_num.get(cam_num, [])
        if not candidates:
            files_aligned.append(None)
            imgs_aligned.append(placeholder.copy())
            continue

        chosen = candidates[0]
        try:
            frame = extract_frame_from_video(chosen, frame_index=frame_index)
        except Exception as e:
            print(f"Warning: failed to read {chosen} ({e}); using white placeholder for cam{cam_num:02d}")
            files_aligned.append(None)
            imgs_aligned.append(placeholder.copy())
            continue

        if frame.shape[:2] != placeholder.shape[:2]:
            raise ValueError(
                f"Frame size mismatch for {chosen}: got {frame.shape[:2]}, expected {placeholder.shape[:2]}"
            )

        files_aligned.append(chosen)
        imgs_aligned.append(frame)

        if len(candidates) > 1:
            print(f"Warning: multiple files for cam{cam_num:02d}; using {chosen.name}")

    return files_aligned, imgs_aligned


def extract_frame_from_video(
    video_path: Path,
    frame_index: int | None = None,
) -> np.ndarray:
    """
    Extract a single grayscale frame from a video.

    If frame_index is None, the middle frame is used when the container exposes
    a frame count. Some MKV files report an unknown count; for those, the first
    readable frame is used.
    """
    video_path = Path(video_path)

    sidecar_frame_count, sidecar_fps = _read_video_sidecar_metadata(video_path)

    cap = cv2.VideoCapture(str(video_path))
    cv2_error: Exception | None = None
    fps = sidecar_fps
    target_frame = _choose_video_frame_index(
        frame_index=frame_index,
        cv2_frame_count=None,
        sidecar_frame_count=sidecar_frame_count,
    )

    if cap.isOpened():
        try:
            cv2_frame_count = _positive_int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cv2_fps = _positive_float(cap.get(cv2.CAP_PROP_FPS))
            fps = cv2_fps if cv2_fps is not None else sidecar_fps

            target_frame = _choose_video_frame_index(
                frame_index=frame_index,
                cv2_frame_count=cv2_frame_count,
                sidecar_frame_count=sidecar_frame_count,
            )

            if target_frame > 0:
                seek_ok = cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                reported_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if not seek_ok or reported_pos < max(1, target_frame - 1):
                    raise RuntimeError(
                        f"OpenCV could not seek to frame {target_frame}; "
                        f"reported position {reported_pos}"
                    )

            ok, frame = cap.read()
            if ok and frame is not None:
                return _as_grayscale(frame)

            cv2_error = RuntimeError(
                f"Failed to read frame {target_frame} with OpenCV"
            )
        except Exception as e:
            cv2_error = e
        finally:
            cap.release()
    else:
        fps = sidecar_fps
        cv2_error = IOError(f"Could not open video with OpenCV")

    try:
        return _extract_frame_from_video_ffmpeg(
            video_path,
            frame_index=target_frame,
            fps=fps,
        )
    except Exception as ffmpeg_error:
        raise RuntimeError(
            f"Failed to read frame {target_frame} from {video_path}; "
            f"OpenCV error: {cv2_error}; ffmpeg error: {ffmpeg_error}"
        ) from ffmpeg_error


def _positive_int(value: float | int | None) -> int | None:
    if value is None:
        return None
    try:
        i = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return i if i > 0 else None


def _positive_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if f > 0 else None


def _choose_video_frame_index(
    frame_index: int | None,
    cv2_frame_count: int | None,
    sidecar_frame_count: int | None,
) -> int:
    if frame_index is None:
        if cv2_frame_count is not None:
            return cv2_frame_count // 2
        return 0

    frame_index = max(0, int(frame_index))
    frame_count = cv2_frame_count if cv2_frame_count is not None else sidecar_frame_count
    if frame_count is not None:
        frame_index = min(frame_index, frame_count - 1)
    return frame_index


def _read_video_sidecar_metadata(video_path: Path) -> tuple[int | None, float | None]:
    """
    Read optional Basler recorder diagnostics next to an MKV.

    The MKV container may omit duration/frame-count metadata, but the recorder
    writes camXX...mkv.diag.json with frame and fps fields.
    """
    diag_path = video_path.with_name(f"{video_path.name}.diag.json")
    if not diag_path.exists():
        return None, None

    try:
        with diag_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None

    recorder = data.get("recorder", {})
    capture = data.get("capture", {})
    context = data.get("context", {})

    frame_count = (
        _positive_int(recorder.get("framesEncoded"))
        or _positive_int(capture.get("framesEmitted"))
    )

    fps = _positive_float(context.get("fps"))
    duration_ms = _positive_float(context.get("durationMs"))
    if fps is None and frame_count is not None and duration_ms is not None:
        fps = frame_count / (duration_ms / 1000.0)

    return frame_count, fps


def _extract_frame_from_video_ffmpeg(
    video_path: Path,
    frame_index: int,
    fps: float | None,
) -> np.ndarray:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

    if frame_index > 0 and fps is not None:
        cmd += ["-ss", f"{frame_index / fps:.6f}"]

    cmd += ["-i", str(video_path)]

    if frame_index > 0 and fps is None:
        cmd += ["-vf", f"select=eq(n\\,{frame_index})"]

    cmd += [
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-pix_fmt",
        "gray",
        "-vcodec",
        "png",
        "-",
    ]

    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"ffmpeg exited with code {proc.returncode}")

    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise RuntimeError("ffmpeg produced an unreadable frame image")

    return frame


def _as_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame


# ---------------------------------------------------------------------------
# Dedicated core function
# ---------------------------------------------------------------------------

def make_panorama_from_hmats(
    H_mats: np.ndarray,
    imgs: List[np.ndarray],
) -> np.ndarray:
    """
    Dedicated function: image list + H matrices -> panorama.

    Parameters
    ----------
    H_mats : np.ndarray
        Shape (N, 3, 3), one transform per image.
    imgs : list of np.ndarray
        N grayscale images.

    Returns
    -------
    pano : np.ndarray
        Stitched panorama (uint8).
    """
    H_mats = np.asarray(H_mats, dtype=float)
    if H_mats.ndim != 3 or H_mats.shape[1:] != (3, 3):
        raise ValueError(
            f"H_mats must have shape (N, 3, 3), got {H_mats.shape}"
        )
    if H_mats.shape[0] != len(imgs):
        raise ValueError(
            f"Number of H matrices ({H_mats.shape[0]}) "
            f"must match number of images ({len(imgs)})"
        )

    Hs_list = [H_mats[i] for i in range(H_mats.shape[0])]
    pano = warp_and_blend(imgs, Hs_list)
    return pano

def warp_and_blend(
    imgs: list[np.ndarray],
    Hs: list[np.ndarray],
) -> np.ndarray:
    """
    Warp all images using their homographies and blend into a single panorama.

    Assumptions:
      - imgs[i] is grayscale uint8
      - Hs[i] maps image i into a common world/mosaic frame
      - len(imgs) == len(Hs)
    """
    if len(imgs) == 0:
        raise ValueError("warp_and_blend: empty image list")
    if len(imgs) != len(Hs):
        raise ValueError(
            f"warp_and_blend: {len(imgs)} images but {len(Hs)} homographies"
        )

    # ------------------------------------------------------------------
    # 1. Compute bounding box of all warped image corners
    # ------------------------------------------------------------------
    all_xy = []

    for im, H in zip(imgs, Hs):
        h, w = im.shape[:2]

        corners = np.array(
            [[0, 0],
             [w, 0],
             [w, h],
             [0, h]],
            dtype=float,
        )
        corners_h = np.c_[corners, np.ones(4)]

        warped_h = (H @ corners_h.T).T
        warped_xy = warped_h[:, :2] / warped_h[:, 2:3]

        all_xy.append(warped_xy)

    all_xy = np.vstack(all_xy)
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)

    # Translation so panorama starts at (0,0)
    T = np.array(
        [[1, 0, -xmin],
         [0, 1, -ymin],
         [0, 0, 1]],
        dtype=float,
    )

    out_w = int(np.ceil(xmax - xmin))
    out_h = int(np.ceil(ymax - ymin))

    # ------------------------------------------------------------------
    # 2. Accumulate warped images + weights
    # ------------------------------------------------------------------
    mosaic = np.zeros((out_h, out_w), dtype=float)
    weight = np.zeros((out_h, out_w), dtype=float)

    for im, H in zip(imgs, Hs):
        Htot = T @ H
        warped = cv2.warpPerspective(
            im.astype(float),
            Htot,
            (out_w, out_h),
        )

        mask = warped > 0
        mosaic[mask] += warped[mask]
        weight[mask] += 1.0

    # ------------------------------------------------------------------
    # 3. Normalize
    # ------------------------------------------------------------------
    weight[weight == 0] = 1.0
    mosaic /= weight

    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    return mosaic

# ---------------------------------------------------------------------------
# H matrix loading helper
# ---------------------------------------------------------------------------

def load_h_mats(hfile: str | Path) -> np.ndarray:
    """
    Load H matrices from .npz or .npy.

    - If .npz: tries key 'H' first, otherwise takes the first array.
    - If .npy: loads directly.
    """
    hfile = Path(hfile)
    if not hfile.exists():
        raise FileNotFoundError(f"H matrix file not found: {hfile}")

    if hfile.suffix == ".npz":
        data = np.load(hfile)
        if "H" in data:
            H = data["H"]
        else:
            # Take the first array in the archive
            first_key = list(data.keys())[0]
            H = data[first_key]
    else:
        # .npy or other single-array formats
        H = np.load(hfile)

    H = np.asarray(H, dtype=float)
    if H.ndim != 3 or H.shape[1:] != (3, 3):
        raise ValueError(
            f"Loaded H has shape {H.shape}, expected (N, 3, 3)"
        )
    return H


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a panorama from pre-computed H matrices "
                    "and either images or videos."
    )
    parser.add_argument(
        "--hfile",
        required=True,
        type=str,
        help="Path to .npz/.npy file containing H matrices (shape: N x 3 x 3).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--images",
        type=str,
        help="Directory containing still images (camXX*.png/jpg/etc.).",
    )
    group.add_argument(
        "--videos",
        type=str,
        help="Directory containing videos (camXX*.mp4/avi/etc.). "
             "One video per camera.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Output panorama image path (e.g. pano.png).",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame index to sample from each video. Default: middle frame "
             "when frame count is known, otherwise first readable frame. "
             "Only used with --videos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load H matrices
    H_mats = load_h_mats(args.hfile)
    n_expected = H_mats.shape[0]

    # Load images either from stills or from videos, aligned to H_mats
    if args.images is not None:
        files, imgs = load_images_from_dir_aligned(args.images, n_expected=n_expected)
        n_real = sum(f is not None for f in files)
        n_pad = n_expected - n_real
        print(f"Loaded {n_real} images from {args.images}; padded {n_pad} missing with white placeholders.")
    else:
        files, imgs = load_frames_from_videos_dir_aligned(
            args.videos,
            n_expected=n_expected,
            frame_index=args.frame_index,
        )
        n_real = sum(f is not None for f in files)
        n_pad = n_expected - n_real
        print(f"Loaded {n_real} video frames from {args.videos}; padded {n_pad} missing with white placeholders.")

    # Build panorama (lengths now guaranteed to match)
    pano = make_panorama_from_hmats(H_mats, imgs)

    # Save output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), pano)
    if not ok:
        raise IOError(f"Failed to write panorama to {out_path}")
    print(f"Panorama written to {out_path}")



if __name__ == "__main__":
    main()
