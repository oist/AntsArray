#!/usr/bin/env python3
"""Stage A: extract the sampled frames to a lossless clip and build the ragged
two-pass ArUco reference on exactly those pixels.

Why a clip rather than predicting straight off the source .avi: the paired
argument in EVAL_HARNESS_DESIGN.md section 2 rests on both models seeing
*identical* pixels. Decoding once into an FFV1 clip and pointing everything --
ArUco, model A0, model all6, and the human adjudication crops -- at that one
file makes the identity structural instead of assumed. It also decodes the
5.75M-frame block02 sources once instead of three times.

Frame indices are renumbered 0..N-1 inside the clip. The mapping back to source
frames lives in the sidecar index JSON; never infer it.

Two detectors run on every burst frame:
  DET_EXACT  errorCorrectionRate = 0.0   -> bit_err 0
  DET_CORR   errorCorrectionRate = 1.0   -> bit_err 1 where it alone decodes

Measured on this footage (EVAL_HARNESS_DESIGN.md section 3.1): ECR=0 is NOT a
purity win. It removes ~4-7% of detections, and they are overwhelmingly real
roster tags with one misread bit -- the blurred/tilted/occluded ants where the
two models differ. Because min_distance=4 > 2*max_correction_bits=2, a one-bit
correction decodes uniquely and provably correctly. So we keep both passes and
tier by measured bit margin instead of pre-filtering.

Contract: group_labelling/EVAL_HARNESS_DESIGN.md sections 3 and 5.

    python group_labelling/eval_extract_and_aruco.py \
        --out-dir /work/ReiterU/centroid_eval --key block01_cam10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_common import (  # noqa: E402
    BIT_MARGIN_TOL,
    build_detectors,
    detector_configs,
    setup_logging,
    write_json,
)



def _detect(detector, gray):
    corners, ids, rejected = detector.detectMarkers(gray)
    n_rej = 0 if rejected is None else len(rejected)
    if ids is None or len(ids) == 0:
        return (np.empty(0, np.int16), np.empty((0, 2), np.float32),
                np.empty((0, 4, 2), np.float32), n_rej)
    ids_flat = np.asarray(ids, np.int32).ravel().astype(np.int16)
    quads = np.stack([np.asarray(c, np.float32).reshape(4, 2) for c in corners])
    return ids_flat, quads.mean(axis=1).astype(np.float32), quads, n_rej


def bit_margin(ids_exact, ctr_exact, ids_corr, ctr_corr, tol=BIT_MARGIN_TOL):
    """Label each DET_CORR detection with its bit margin.

    A CORR detection matched by id AND position (within `tol`) to an EXACT
    detection was decoded with zero corrections -> bit_err 0. Otherwise the one
    permitted correction was used -> bit_err 1.

    DET_CORR is a superset of DET_EXACT in every frame measured on this
    footage, so CORR is the right base set: taking EXACT as the base would
    silently discard the hard population this study needs.
    """
    bit_err = np.ones(len(ids_corr), np.uint8)
    if len(ids_exact) == 0 or len(ids_corr) == 0:
        return bit_err
    for k in range(len(ids_corr)):
        same_id = np.where(ids_exact == ids_corr[k])[0]
        if len(same_id) == 0:
            continue
        if np.linalg.norm(ctr_exact[same_id] - ctr_corr[k], axis=1).min() <= tol:
            bit_err[k] = 0
    return bit_err


def extract_and_detect(manifest: dict, out_dir: Path, write_clip: bool,
                       ffmpeg: str = "") -> dict:
    key = manifest["key"]
    video = Path(manifest["video"])
    burst = np.asarray(manifest["burst_frames"], np.int64)
    owner = np.asarray(manifest["anchor_of_frame"], np.int64)
    anchors = {int(a) for a in manifest["anchors"]}
    width, height = int(manifest["width"]), int(manifest["height"])

    det_exact, det_corr = build_detectors()
    cfg_exact, cfg_corr = detector_configs()

    clips = out_dir / "clips"
    aruco_dir = out_dir / "aruco"
    clips.mkdir(parents=True, exist_ok=True)
    aruco_dir.mkdir(parents=True, exist_ok=True)

    clip_path = clips / f"{key}.mkv"
    writer = None
    if write_clip:
        # FFV1 is mathematically lossless, so the clip re-presents the exact
        # decoded source frames. A lossy re-encode would break the paired
        # argument in a way no checksum on the written bytes would catch.
        # OpenCV's FFV1 muxer is used rather than an ffmpeg subprocess because
        # ffmpeg is not on PATH on the login node; the round trip is verified
        # bit-exact by eval_preflight gate 0a2.
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"FFV1"),
                                 24.0, (width, height), False)
        if not writer.isOpened():
            raise RuntimeError(f"cannot open FFV1 writer at {clip_path}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")

    rows_frame, rows_id, rows_ctr, rows_quad, rows_bit = [], [], [], [], []
    n_rejected = np.zeros(len(burst), np.int32)
    clip_index: list[dict] = []
    n_seek_fail = 0
    n_decoded = 0
    t0 = time.time()

    for i, f in enumerate(burst.tolist()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            n_seek_fail += 1
            logging.warning("[%s] read failed at source frame %d", key, f)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ie, ce, _qe, _ = _detect(det_exact, gray)
        ic, cc, qc, nrej = _detect(det_corr, gray)
        n_rejected[i] = nrej
        be = bit_margin(ie, ce, ic, cc)

        if len(ic):
            rows_frame.append(np.full(len(ic), f, np.int64))
            rows_id.append(ic)
            rows_ctr.append(cc)
            rows_quad.append(qc)
            rows_bit.append(be)

        # Only anchor frames go into the clip. Model inference runs on anchors
        # only, and at ~1.5 MB/frame writing all 4,200 burst frames would cost
        # ~6 GB per camera (~460 GB overall) to store pixels nothing reads.
        # Adjudication crops are re-seeked from the source, which gate 0c
        # verified is bit-exact.
        is_anchor = int(f) in anchors
        if writer is not None and is_anchor:
            writer.write(gray)
            clip_index.append({
                "clip_frame": len(clip_index),
                "source_frame": int(f),
                "anchor": int(owner[i]),
            })
        n_decoded += 1
        if (i + 1) % 500 == 0:
            logging.info("[%s] %d/%d burst frames (%.2f s/frame)",
                         key, i + 1, len(burst), (time.time() - t0) / (i + 1))

    cap.release()
    if writer is not None:
        writer.release()
        if not clip_path.exists() or clip_path.stat().st_size == 0:
            raise RuntimeError(f"FFV1 writer produced nothing at {clip_path}")

    def cat(parts, empty):
        return np.concatenate(parts) if parts else empty

    det_frame = cat(rows_frame, np.empty(0, np.int64))
    det_id = cat(rows_id, np.empty(0, np.int16))
    det_center = cat(rows_ctr, np.empty((0, 2), np.float32))
    det_corners = cat(rows_quad, np.empty((0, 4, 2), np.float32))
    det_bit_err = cat(rows_bit, np.empty(0, np.uint8))

    ref_path = aruco_dir / f"{key}_ref.h5"
    with h5py.File(ref_path, "w") as h:
        # Ragged, never (F,100,2): the dense slot layout used by both existing
        # writers is last-write-wins, so two ants carrying the same id in one
        # frame silently become one. Here they are two rows.
        h.create_dataset("det_frame", data=det_frame)
        h.create_dataset("det_id", data=det_id)
        h.create_dataset("det_center", data=det_center)
        h.create_dataset("det_corners", data=det_corners)
        h.create_dataset("det_bit_err", data=det_bit_err)
        h.create_dataset("det_side", data=np.zeros(len(det_frame), np.int8))
        h.create_dataset("det_tier", data=np.zeros(len(det_frame), np.uint8))
        h.create_dataset("frame_n_rejected", data=n_rejected)
        h.create_dataset("burst_frame", data=burst)
        h.create_dataset("anchor_of_frame", data=owner)
        h.attrs["key"] = key
        h.attrs["width"] = width
        h.attrs["height"] = height
        h.attrs["cfg_exact"] = json.dumps(vars(cfg_exact), default=str)
        h.attrs["cfg_corr"] = json.dumps(vars(cfg_corr), default=str)
        h.attrs["bit_margin_tol"] = BIT_MARGIN_TOL

    n_exact = int((det_bit_err == 0).sum())
    n_corr = int((det_bit_err == 1).sum())
    n_dec = max(n_decoded, 1)
    stats = {
        "key": key,
        "n_burst_frames": int(len(burst)),
        "n_frames_decoded": n_decoded,
        "n_anchor_frames_in_clip": len(clip_index),
        "n_seek_fail": n_seek_fail,
        "n_detections": int(len(det_frame)),
        "n_exact": n_exact,
        "n_corr_only": n_corr,
        "corr_only_fraction": (n_corr / len(det_frame)) if len(det_frame) else 0.0,
        "detections_per_frame": len(det_frame) / n_dec,
        "mean_rejected_per_frame": float(n_rejected.mean()) if len(n_rejected) else 0.0,
        "ref_h5": str(ref_path),
        "clip": str(clip_path) if write_clip else None,
        "seconds": round(time.time() - t0, 1),
        "low_reference": False,
    }
    write_json(clip_index, clips / f"{key}_index.json")
    logging.info("[%s] %d detections (%d exact, %d corr-only = %.1f%%), "
                 "%.2f/frame, %.1f rejected/frame, %.1f min",
                 key, len(det_frame), n_exact, n_corr,
                 100 * stats["corr_only_fraction"], stats["detections_per_frame"],
                 stats["mean_rejected_per_frame"], stats["seconds"] / 60)
    if stats["detections_per_frame"] < 1.0:
        logging.warning("[%s] LOW-REFERENCE camera (%.2f tags/frame). It stays in "
                        "the primary adjudication arm (which needs no reference) "
                        "but is gated out of the corroborative TagRecall arm.",
                        key, stats["detections_per_frame"])
        stats["low_reference"] = True
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--key", type=str, default=None,
                   help="single camera key, e.g. block01_cam10")
    p.add_argument("--array-index", type=int, default=None,
                   help="SLURM_ARRAY_TASK_ID; picks the Nth key in strata.json")
    p.add_argument("--no-clip", action="store_true",
                   help="skip FFV1 clip writing (ArUco reference only)")
    p.add_argument("--ffmpeg", type=str, default="ffmpeg")
    args = p.parse_args()

    out: Path = args.out_dir.resolve()
    setup_logging(out / "logs", "eval_extract_and_aruco")

    strata = json.loads((out / "strata.json").read_text(encoding="utf-8"))
    keys = sorted(k for group in strata["strata"].values() for k in group)
    if args.key:
        selected = [args.key]
    elif args.array_index is not None:
        if args.array_index >= len(keys):
            logging.error("array index %d out of range (%d keys)",
                          args.array_index, len(keys))
            return 1
        selected = [keys[args.array_index]]
    else:
        selected = keys

    results = []
    for key in selected:
        mpath = out / "anchors" / f"{key}.json"
        if not mpath.exists():
            logging.error("[%s] no manifest at %s", key, mpath)
            continue
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        try:
            results.append(extract_and_detect(
                manifest, out, write_clip=not args.no_clip, ffmpeg=args.ffmpeg))
        except Exception:
            logging.exception("[%s] stage A failed", key)
    if not results:
        return 1
    tag = args.key or (f"idx{args.array_index}" if args.array_index is not None
                       else "all")
    write_json(results, out / "aruco" / f"stage_a_stats_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
