"""Shared pieces for scoring printed Custom_A tag sheets in videos (see README.md).

A sheet is located in a frame by fitting a homography from each read ID's position on the 10x10
print grid to the image. Grid coordinates are mm with tag 0's centre at the origin.
"""

from __future__ import annotations

import io
import pickle
import sys
import traceback
from contextlib import redirect_stdout
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aruco_detection" / "custom_dicts"))
from generate_cuttable_sheet import load_custom_dictionary  # noqa: E402

PITCH_X, PITCH_Y = 3.3, 3.9      # tag pitch on the sheet (2.1 mm tag + 1.2 / 1.8 mm gaps)
MARKER_MM = 1.5
MIN_INLIERS = 10
EDGE_PX = 10                     # a visible marker stays this far inside the frame
MATCH_MM = 0.8                   # a read sits this close to its predicted centre; a wrong ID is >= 3.3 mm off
GRID = np.array([[(k % 10) * PITCH_X, (k // 10) * PITCH_Y] for k in range(100)], np.float64)
GRID_CENTRE = GRID.mean(axis=0)
HALF = MARKER_MM / 2
CORNER_OFF = np.array([[-HALF, -HALF], [HALF, -HALF], [HALF, HALF], [-HALF, HALF]])
CONFIGS = ("prod_ecr1", "prod_ecr0")
CONFIG_NAMES = {"prod_ecr1": "1-bit correction, as production", "prod_ecr0": "no bit correction"}

_detectors = {}


def detector(dictionary, ecr):
    # same settings as detection_pipeline/scripts/run_aruco_mp.py::_build_detector, ecr varied
    p = aruco.DetectorParameters()
    p.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    p.adaptiveThreshConstant = 3
    p.adaptiveThreshWinSizeMin = 10
    p.adaptiveThreshWinSizeMax = 40
    p.adaptiveThreshWinSizeStep = 10
    p.errorCorrectionRate = ecr
    return aruco.ArucoDetector(dictionary, p)


def init_worker(npz):
    cv2.setNumThreads(1)
    dictionary = load_custom_dictionary(npz)[0]
    _detectors["prod_ecr1"] = detector(dictionary, 1.0)
    _detectors["prod_ecr0"] = detector(dictionary, 0.0)


def detect(job):
    """(frame index, {config: (ids, corners)}) for one grey frame."""
    idx, gray = job
    out = {}
    for name in CONFIGS:
        corners, ids, _ = _detectors[name].detectMarkers(gray)
        if ids is None:
            out[name] = (np.zeros(0, int), np.zeros((0, 4, 2), np.float32))
        else:
            out[name] = (ids.ravel().astype(int), np.array([c[0] for c in corners], np.float32))
    return idx, out


def frames(video):
    cap = cv2.VideoCapture(str(video))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield i, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        i += 1


def frame_shape(video):
    cap = cv2.VideoCapture(str(video))
    return int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


def video_fps(video):
    fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)
    if not fps > 0:
        raise SystemExit(f"{video}: no frame rate (missing or unreadable video?)")
    return fps


def detect_video(video, npz, workers, cache_dir=None):
    """Every frame's reads, in parallel. Run from a script file: spawned workers re-import __main__.

    With cache_dir the reads are kept as <video stem>_detections.pkl, together with the dictionary
    they were read with; delete the file after changing the detector settings.
    """
    cache = None if cache_dir is None else Path(cache_dir) / f"{Path(video).stem}_detections.pkl"
    if cache is not None and cache.exists():
        saved = pickle.loads(cache.read_bytes())
        if isinstance(saved, dict) and saved.get("npz") == Path(npz).name:
            return saved["results"]
    with Pool(workers, initializer=init_worker, initargs=(str(npz),)) as pool:
        results = dict(pool.imap(detect, frames(video), chunksize=2))
    if cache is not None:
        cache.write_bytes(pickle.dumps({"npz": Path(npz).name, "results": results}))
    return results


def project(H, pts):
    return cv2.perspectiveTransform(np.asarray(pts, np.float64)[None], H)[0]


def norm(H):
    return H / H[2, 2]


def pose(H):
    """(image position of the grid centre, angle of the grid x axis in degrees)."""
    c = project(H, GRID_CENTRE[None])[0]
    a, b = project(H, [[0.0, 0.0], [10.0, 0.0]])
    return c, np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))


def match_threshold_px(corners):
    side = np.median(np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)) if len(corners) else 1.0
    return MATCH_MM * side / MARKER_MM


def fit_sheets(ids, corners):
    """Up to two (H grid->image, inlier mask) fits, largest first."""
    if len(ids) < MIN_INLIERS:
        return []
    centres = corners.mean(axis=1).astype(np.float64)
    thresh = match_threshold_px(corners)
    remaining = np.ones(len(ids), bool)
    fits = []
    for _ in range(2):
        idx = np.flatnonzero(remaining)
        if len(idx) < MIN_INLIERS:
            break
        H, mask = cv2.findHomography(GRID[ids[idx]], centres[idx], cv2.RANSAC, thresh, maxIters=5000)
        if H is None or mask.sum() < MIN_INLIERS:
            break
        inl = idx[mask.ravel() > 0]
        refit, _ = cv2.findHomography(GRID[ids[inl]], centres[inl], 0)   # refit on all inliers
        H = H if refit is None else refit                # None: degenerate (e.g. all in one row)
        inl = idx[np.linalg.norm(project(H, GRID[ids[idx]]) - centres[idx], axis=1) < thresh]
        if len(set(ids[inl])) != len(inl):              # one ID twice in one sheet: drop the fit
            break
        mask_all = np.zeros(len(ids), bool)
        mask_all[inl] = True
        fits.append((H, mask_all))
        remaining &= ~mask_all
    return fits


def match_sheet(H, ids, corners, candidates):
    """Reads among `candidates` that sit on sheet H's grid (refit once on them), one per ID."""
    centres = corners.mean(axis=1).astype(np.float64)
    thresh = match_threshold_px(corners)
    for _ in range(2):
        mask = np.zeros(len(ids), bool)
        idx = np.flatnonzero(candidates)
        if len(idx):
            d = np.linalg.norm(project(H, GRID[ids[idx]]) - centres[idx], axis=1)
            best = {}
            for j, dist in zip(idx, d):
                if dist < thresh and (ids[j] not in best or dist < best[ids[j]][1]):
                    best[ids[j]] = (j, dist)
            mask[[j for j, _ in best.values()]] = True
        if mask.sum() < 8:
            break
        refit, _ = cv2.findHomography(GRID[ids[mask]], centres[mask], 0)
        if refit is None:                                # degenerate (e.g. one row): keep H
            break
        H = refit
    return H, mask


def wrong_reads(H, ids, corners, stray, visible):
    """(wrong IDs, right ID but off position) among the stray reads that sit on a visible tag of sheet H.

    A stray read is on no sheet's grid within MATCH_MM. On a tag means within 1.2 mm of its centre.
    """
    idx = np.flatnonzero(stray)
    at = tag_at(H, corners[idx].mean(axis=1))
    on = at >= 0
    on[on] = visible[at[on]]
    return int(np.sum(on & (ids[idx] != at))), int(np.sum(on & (ids[idx] == at)))


def tag_at(H, points):
    """Print-grid tag under each image point, -1 if none: where a stray read sits on a sheet."""
    if not len(points):
        return np.zeros(0, int)
    g = project(np.linalg.inv(H), points)
    c, r = np.rint(g[:, 0] / PITCH_X).astype(int), np.rint(g[:, 1] / PITCH_Y).astype(int)
    on = ((0 <= c) & (c < 10) & (0 <= r) & (r < 10)
          & (np.abs(g[:, 0] - c * PITCH_X) < 1.2) & (np.abs(g[:, 1] - r * PITCH_Y) < 1.2))
    return np.where(on, r * 10 + c, -1)


def marker_quads(H):
    """(100, 4, 2) image corners of every marker on sheet H."""
    return project(H, (GRID[:, None, :] + CORNER_OFF[None]).reshape(-1, 2)).reshape(100, 4, 2)


def in_frame(quads, shape):
    h, w = shape
    return np.all((quads[..., 0] > EDGE_PX) & (quads[..., 0] < w - EDGE_PX)
                  & (quads[..., 1] > EDGE_PX) & (quads[..., 1] < h - EDGE_PX), axis=1)


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)
        return len(s)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_logged(out_dir, name, fn, *args):
    """Run fn(*args) with stdout also saved to OUT_DIR/<name>_log_<timestamp>.txt."""
    log = io.StringIO()
    try:
        with redirect_stdout(_Tee(sys.stdout, log)):
            return fn(*args)
    except BaseException:
        log.write(traceback.format_exc())                # the log must say why the run stopped
        raise
    finally:
        path = Path(out_dir) / f"{name}_log_{datetime.now().astimezone():%Y%m%d_%H%M%S}.txt"
        path.write_text(log.getvalue(), encoding="utf-8")
