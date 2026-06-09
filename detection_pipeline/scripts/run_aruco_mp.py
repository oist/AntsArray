#!/usr/bin/env python3
"""
run_aruco_mp.py — multiprocessing ArUco detection (frame-range parallel).

Faster drop-in for the serial run_aruco.py used by the detection pipeline.
Splits the video into <workers> contiguous frame ranges, each decoded +
detected in its own process (cv2.setNumThreads(1)), then merges into the same
(num_frames, dict_size, 2) tracks / (num_frames, dict_size) confidences arrays
the serial version produces.

Verified frame-exact vs the serial detector (benchmark 2026-06-09: 0 mismatch,
maxcoord=0.0 at P=4/8/16) and ~3.9x faster at -c 16 (9.5 -> 36.5 it/s) on
block02 chunks. OpenCV's internal thread pool is ~4x less core-efficient than
independent single-thread processes, so this beats setNumThreads().

OpenCV + fork note: the worker Pool is created BEFORE any detectMarkers call in
the parent. OpenCV's thread pool does not survive fork once spun, so ALL
detection happens in the cold-forked children only.

Frame count: range splitting uses --n-frames (the worklist's expected_frames,
column 3) for good load balance; the final range always reads to EOF, so an
under- or over-count only affects balance, never correctness.
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import cv2.aruco as aruco
import h5py
import numpy as np
import pandas as pd


def load_custom_aruco_dict(npz_path: str):
    """Load custom 4x4 ArUco dictionary from NPZ. Returns (aruco.Dictionary, n_markers)."""
    data = np.load(npz_path, allow_pickle=True)
    d = aruco.Dictionary()
    d.bytesList = data["bytesList"]
    d.markerSize = 4
    d.maxCorrectionBits = int(data["max_correction_bits"])
    return d, int(d.bytesList.shape[0])


def _build_dict(spec):
    """spec is ('custom', npz_path) or ('predef', dictionary_size)."""
    kind, val = spec
    if kind == "custom":
        return load_custom_aruco_dict(val)
    return aruco.getPredefinedDictionary(aruco.DICT_4X4_1000), int(val)


def _build_detector(aruco_dict):
    # Identical to the serial production detector.
    p = aruco.DetectorParameters()
    p.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
    p.adaptiveThreshConstant = 3
    p.adaptiveThreshWinSizeMin = 10
    p.adaptiveThreshWinSizeMax = 40
    p.adaptiveThreshWinSizeStep = 10
    p.errorCorrectionRate = 1
    return aruco.ArucoDetector(aruco_dict, p)


def _worker(args):
    """Detect ArUco over [start, start+count) (or to EOF if to_eof). Rebuilds the
    detector from the dict spec so it works under both fork and spawn."""
    video, dict_spec, start, count, to_eof = args
    cv2.setNumThreads(1)
    aruco_dict, nmark = _build_dict(dict_spec)
    det = _build_detector(aruco_dict)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video}")
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = []
    n = 0
    while to_eof or n < count:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = det.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                mid = int(marker_id)
                if 0 <= mid < nmark:
                    c = np.mean(corners[i][0], axis=0)
                    out.append((start + n, mid, float(c[0]), float(c[1])))
        n += 1
    cap.release()
    return out, start + n  # end index (exclusive) this worker reached


def _probe_frame_count(video: str) -> int:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def detect_aruco_mp(video, dict_spec, dictionary_size, *, n_frames=0, workers=None):
    """Returns (tracks, confidences) identical to the serial detector."""
    if workers is None or workers <= 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 4)
    workers = max(1, workers)

    if not n_frames or n_frames <= 0:
        n_frames = _probe_frame_count(video)
    if n_frames <= 0:
        n_frames = 1  # unknown: single worker reads to EOF

    per = (n_frames + workers - 1) // workers
    tasks = []
    for i in range(workers):
        start = i * per
        if i != 0 and start >= n_frames:
            break
        is_last = (i == workers - 1) or ((i + 1) * per >= n_frames)
        tasks.append((str(video), dict_spec, start, per, is_last))
        if is_last:
            break

    # COLD fork: no detectMarkers has run in the parent yet.
    with Pool(len(tasks)) as pool:
        results = pool.map(_worker, tasks)

    dets = [d for r in results for d in r[0]]
    num_frames = max((r[1] for r in results), default=0)
    tracks = np.zeros((num_frames, dictionary_size, 2), dtype=np.float32)
    confidences = np.zeros((num_frames, dictionary_size), dtype=np.float32)
    for fidx, mid, x, y in dets:
        tracks[fidx, mid, 0] = x
        tracks[fidx, mid, 1] = y
        confidences[fidx, mid] = 1.0
    return tracks, confidences


def main():
    ap = argparse.ArgumentParser("aruco-track-mp")
    ap.add_argument("--video-file", required=True)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--dictionary-size", type=int, default=300,
                    help="Marker ID cap for DICT_4X4_1000 (ignored when --custom-dict is given).")
    ap.add_argument("--custom-dict", default=None, help="Path to custom 4x4 ArUco NPZ.")
    ap.add_argument("--output-format", choices=["csv", "h5", "both"], default="h5")
    ap.add_argument("--n-frames", type=int, default=0,
                    help="Expected frame count (worklist col 3) for range splitting; 0 = probe.")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel workers; 0 = SLURM_CPUS_PER_TASK or cpu_count.")
    args = ap.parse_args()

    if not os.path.isfile(args.video_file):
        raise FileNotFoundError(f"Video file not found: {args.video_file}")
    os.makedirs(args.output_path, exist_ok=True)

    name = os.path.splitext(os.path.basename(args.video_file))[0]
    out_dir = Path(args.output_path)

    if args.custom_dict:
        if not os.path.isfile(args.custom_dict):
            raise FileNotFoundError(f"--custom-dict not found: {args.custom_dict}")
        _, nmark = load_custom_aruco_dict(args.custom_dict)
        dict_spec, dict_size = ("custom", args.custom_dict), nmark
        print(f"[INFO] Using custom dict {args.custom_dict} ({nmark} markers)", flush=True)
    else:
        dict_spec, dict_size = ("predef", args.dictionary_size), args.dictionary_size

    print(f"[INFO] MP aruco on {name} "
          f"(workers={args.workers or 'auto'}, n_frames={args.n_frames or 'probe'})", flush=True)
    tracks, confidences = detect_aruco_mp(
        args.video_file, dict_spec, dict_size, n_frames=args.n_frames, workers=args.workers
    )
    print(f"[INFO] processed {tracks.shape[0]} frames", flush=True)

    raw_h5 = out_dir / f"{name}_aruco_tracks.h5"
    with h5py.File(raw_h5, "w") as h:
        h.create_dataset("aruco_tracks", data=tracks, compression="gzip", shuffle=True, chunks=True)
        h.create_dataset("aruco_confidences", data=confidences, compression="gzip", shuffle=True, chunks=True)
    print(f"[INFO] Saved raw arrays to: {raw_h5}", flush=True)

    fr, inst = np.where(confidences > 0)
    if len(fr):
        df = pd.DataFrame({
            "Frame": fr.astype(np.int32),
            "Instance": inst.astype(np.int32),
            "X": tracks[fr, inst, 0].astype(np.float32),
            "Y": tracks[fr, inst, 1].astype(np.float32),
            "Confidence": confidences[fr, inst].astype(np.float32),
        })
    else:
        df = pd.DataFrame(columns=["Frame", "Instance", "X", "Y", "Confidence"])

    if args.output_format in ("csv", "both"):
        df.to_csv(out_dir / f"{name}_aruco_detections.csv", index=False, float_format="%.1f")
    if args.output_format in ("h5", "both"):
        try:
            import tables  # noqa: F401
            df.to_hdf(out_dir / f"{name}_aruco_detections.h5", key="detections",
                      mode="w", format="table", complevel=4, complib="zlib")
        except ImportError:
            print("[WARN] 'tables' not found; skipping detections H5 export.", flush=True)


if __name__ == "__main__":
    main()
