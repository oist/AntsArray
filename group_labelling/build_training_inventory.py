#!/usr/bin/env python3
"""Build per-frame inventory of ArUco + SLEAP signals for training-frame selection.

For each (camera, frame) row, computes objective signals that downstream
selection (Stage 2) uses to stratify and quota training frames:

  n_aruco            - visible ArUco tags
  n_sleap            - SLEAP predicted instances
  n_matched          - SLEAP instances NN-matched to an ArUco tag within radius
  n_unmatched_sleap  - SLEAP instances with no ArUco tag in radius
                       (queen / tagless worker / false positive candidates)
  n_unmatched_aruco  - ArUco tags with no SLEAP anchor in radius (missed-ant)
  n_duplicate_sleap  - SLEAP instances >1 mapped to same ArUco tag (duplicates)
  n_near_edge_aruco  - ArUco tags within EDGE_MARGIN px of any image boundary
  mean_kp_score      - mean of per-keypoint scores across all SLEAP instances
                       (NaN if no instances)
  min_pair_dist      - min pairwise distance among present ArUco tags (px)
                       (NaN if <2 tags)
  mean_speed         - mean per-tag ||pos[t]-pos[t-1]|| over tags present in
                       both frames (px / frame). NaN at frame 0.
  n_low_score_inst   - SLEAP instances with mean point score < LOW_SCORE_THR

Per-camera files: <out-dir>/<cam>_inventory.parquet (CSV fallback if pyarrow
absent). Concatenated master: <out-dir>/inventory_master.parquet.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sleap_io as sio
from scipy.spatial.distance import cdist, pdist

CAM_RE = re.compile(r"^(cam\d{2})_")
LOW_SCORE_THR = 0.5
DEFAULT_MATCH_RADIUS = 30.0  # pixels — SLEAP anchor must be within this of ArUco
DEFAULT_EDGE_MARGIN = 50.0   # pixels from image boundary


def discover_camera_pairs(data_dir: Path) -> dict[str, dict[str, Path]]:
    """Return {cam_id: {'slp': Path, 'aruco': Path}} for every camNN with both files."""
    pairs: dict[str, dict[str, Path]] = {}
    for slp in sorted(data_dir.glob("cam*.slp")):
        m = CAM_RE.match(slp.name)
        if not m:
            continue
        cam = m.group(1)
        aruco = slp.with_name(slp.stem + "_aruco_tracks.h5")
        if not aruco.exists():
            logging.warning("[%s] missing aruco_tracks file: %s", cam, aruco.name)
            continue
        pairs[cam] = {"slp": slp, "aruco": aruco}
    return pairs


def compute_aruco_metrics(
    tracks: np.ndarray,  # (F, S, 2) float
    conf: np.ndarray,    # (F, S) float
    img_w: int,
    img_h: int,
    edge_margin: float,
) -> dict[str, np.ndarray]:
    """Per-frame ArUco-only signals."""
    n_frames = tracks.shape[0]
    present = conf > 0
    n_aruco = present.sum(axis=1).astype(np.int32)

    min_pair_dist = np.full(n_frames, np.nan, dtype=np.float32)
    n_near_edge = np.zeros(n_frames, dtype=np.int32)

    delta = np.zeros_like(tracks)
    delta[1:] = tracks[1:] - tracks[:-1]
    step = np.linalg.norm(delta, axis=2)  # (F, S)
    valid_step = np.zeros((n_frames, tracks.shape[1]), dtype=bool)
    valid_step[1:] = present[1:] & present[:-1]
    step_masked = np.where(valid_step, step, np.nan)
    with np.errstate(invalid="ignore"):
        mean_speed = np.nanmean(step_masked, axis=1).astype(np.float32)

    edge_lo_x, edge_hi_x = edge_margin, img_w - edge_margin
    edge_lo_y, edge_hi_y = edge_margin, img_h - edge_margin

    for f in range(n_frames):
        present_idx = np.where(present[f])[0]
        if len(present_idx) >= 2:
            pts = tracks[f, present_idx]
            min_pair_dist[f] = float(pdist(pts).min())
        if len(present_idx):
            pts = tracks[f, present_idx]
            x, y = pts[:, 0], pts[:, 1]
            near = (
                (x < edge_lo_x) | (x > edge_hi_x)
                | (y < edge_lo_y) | (y > edge_hi_y)
            )
            n_near_edge[f] = int(near.sum())

    return {
        "n_aruco": n_aruco,
        "min_pair_dist": min_pair_dist,
        "mean_speed": mean_speed,
        "n_near_edge_aruco": n_near_edge,
    }


def compute_sleap_match_metrics(
    labels: sio.Labels,
    tracks: np.ndarray,
    conf: np.ndarray,
    match_radius: float,
) -> dict[str, np.ndarray]:
    """Per-frame SLEAP signals + matching against ArUco."""
    n_frames = tracks.shape[0]
    present = conf > 0
    radius_sq = match_radius ** 2

    n_sleap = np.zeros(n_frames, dtype=np.int32)
    n_matched = np.zeros(n_frames, dtype=np.int32)
    n_unmatched_sleap = np.zeros(n_frames, dtype=np.int32)
    n_unmatched_aruco = np.zeros(n_frames, dtype=np.int32)
    n_duplicate_sleap = np.zeros(n_frames, dtype=np.int32)
    mean_kp_score = np.full(n_frames, np.nan, dtype=np.float32)
    n_low_score_inst = np.zeros(n_frames, dtype=np.int32)
    frame_in_slp = np.zeros(n_frames, dtype=bool)

    frames_by_idx = {lf.frame_idx: lf for lf in labels.labeled_frames}

    for f in range(n_frames):
        n_present_aruco = int(present[f].sum())
        lf = frames_by_idx.get(f)
        frame_in_slp[f] = lf is not None
        if lf is None or not lf.instances:
            n_unmatched_aruco[f] = n_present_aruco
            continue

        anchors: list[tuple[float, float]] = []
        inst_scores: list[float] = []
        all_scores: list[float] = []
        for ins in lf.instances:
            pts = ins.points
            xy = pts["xy"]
            sc = pts["score"]
            vis = pts["visible"]
            anchor_xy = xy[0]
            if vis[0] and np.isfinite(anchor_xy).all():
                anchors.append((float(anchor_xy[0]), float(anchor_xy[1])))
            else:
                anchors.append((np.nan, np.nan))
            visible_scores = sc[vis]
            if visible_scores.size:
                inst_scores.append(float(visible_scores.mean()))
                all_scores.extend(visible_scores.tolist())
            else:
                inst_scores.append(np.nan)

        n_sleap[f] = len(lf.instances)
        if all_scores:
            mean_kp_score[f] = float(np.mean(all_scores))
        n_low_score_inst[f] = int(
            sum(1 for s in inst_scores if np.isfinite(s) and s < LOW_SCORE_THR)
        )

        anchors_arr = np.array(anchors, dtype=np.float32)
        aruco_pts = tracks[f, present[f]] if n_present_aruco else np.empty((0, 2),
                                                                            dtype=np.float32)

        valid_anchor_mask = np.isfinite(anchors_arr[:, 0])
        n_invalid_anchors = int((~valid_anchor_mask).sum())

        if n_present_aruco == 0 or not valid_anchor_mask.any():
            n_unmatched_sleap[f] = len(lf.instances) - n_invalid_anchors
            n_unmatched_aruco[f] = n_present_aruco
            continue

        anchors_valid = anchors_arr[valid_anchor_mask]
        d2 = cdist(anchors_valid, aruco_pts, "sqeuclidean")
        nearest = d2.argmin(axis=1)
        nearest_d2 = d2[np.arange(len(anchors_valid)), nearest]
        in_radius = nearest_d2 <= radius_sq

        matched_aruco = nearest[in_radius]
        n_matched[f] = int(in_radius.sum())
        n_unmatched_sleap[f] = int((~in_radius).sum()) + n_invalid_anchors
        n_unmatched_aruco[f] = n_present_aruco - len(set(matched_aruco.tolist()))
        if matched_aruco.size:
            _, counts = np.unique(matched_aruco, return_counts=True)
            n_duplicate_sleap[f] = int((counts - 1).sum())

    return {
        "n_sleap": n_sleap,
        "n_matched": n_matched,
        "n_unmatched_sleap": n_unmatched_sleap,
        "n_unmatched_aruco": n_unmatched_aruco,
        "n_duplicate_sleap": n_duplicate_sleap,
        "mean_kp_score": mean_kp_score,
        "n_low_score_inst": n_low_score_inst,
        "frame_in_slp": frame_in_slp,
    }


def process_camera(cam: str, slp_path: Path, aruco_path: Path,
                   match_radius: float, edge_margin: float) -> pd.DataFrame:
    logging.info("[%s] reading aruco %s", cam, aruco_path.name)
    with h5py.File(aruco_path, "r") as h:
        tracks = h["aruco_tracks"][:]
        conf = h["aruco_confidences"][:]

    logging.info("[%s] reading slp %s", cam, slp_path.name)
    t0 = time.time()
    labels = sio.load_slp(str(slp_path))
    logging.info("[%s] loaded slp in %.1fs (%d labeled frames)",
                 cam, time.time() - t0, len(labels.labeled_frames))

    n_frames_aruco = tracks.shape[0]
    n_frames_slp = max((lf.frame_idx for lf in labels.labeled_frames), default=-1) + 1
    if n_frames_slp > n_frames_aruco:
        logging.warning("[%s] slp has %d frames, aruco has %d — using aruco range",
                        cam, n_frames_slp, n_frames_aruco)
    n_frames = n_frames_aruco

    vid_shape = labels.videos[0].shape  # (F, H, W, C)
    img_h = int(vid_shape[1])
    img_w = int(vid_shape[2])
    logging.info("[%s] image %dx%d, %d frames", cam, img_w, img_h, n_frames)

    t0 = time.time()
    aruco_metrics = compute_aruco_metrics(tracks, conf, img_w, img_h, edge_margin)
    logging.info("[%s] aruco metrics in %.1fs", cam, time.time() - t0)

    t0 = time.time()
    sleap_metrics = compute_sleap_match_metrics(labels, tracks, conf, match_radius)
    logging.info("[%s] sleap+match metrics in %.1fs", cam, time.time() - t0)

    coverage = float(sleap_metrics["frame_in_slp"].mean())
    if coverage < 0.99:
        logging.warning(
            "[%s] sparse .slp coverage: only %.1f%% of %d frames are present in the slp "
            "(rest will have frame_in_slp=False and must be excluded from frame selection)",
            cam, coverage * 100, n_frames,
        )

    df = pd.DataFrame({"frame_idx": np.arange(n_frames, dtype=np.int32),
                       **aruco_metrics, **sleap_metrics})
    df.insert(0, "camera", cam)
    return df


def save_df(df: pd.DataFrame, out_path: Path) -> Path:
    try:
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception as exc:
        csv_path = out_path.with_suffix(".csv")
        logging.warning("parquet write failed (%s) — falling back to CSV: %s",
                        exc, csv_path.name)
        df.to_csv(csv_path, index=False)
        return csv_path


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_path, encoding="utf-8")],
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory containing cam*.slp and cam*_aruco_tracks.h5")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output dir (default: <data-dir>/../training_inventory)")
    p.add_argument("--cameras", type=str, default=None,
                   help="Comma-separated camera ids to process (e.g. cam01,cam02)")
    p.add_argument("--match-radius", type=float, default=DEFAULT_MATCH_RADIUS)
    p.add_argument("--edge-margin", type=float, default=DEFAULT_EDGE_MARGIN)
    args = p.parse_args()

    data_dir: Path = args.data_dir.resolve()
    out_dir: Path = (args.out_dir or data_dir.parent / "training_inventory").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(out_dir / f"build_training_inventory_{ts}.log")

    logging.info("data_dir=%s", data_dir)
    logging.info("out_dir=%s", out_dir)
    logging.info("match_radius=%.1f edge_margin=%.1f",
                 args.match_radius, args.edge_margin)

    pairs = discover_camera_pairs(data_dir)
    if args.cameras:
        wanted = {c.strip() for c in args.cameras.split(",") if c.strip()}
        pairs = {c: v for c, v in pairs.items() if c in wanted}
    if not pairs:
        logging.error("No camera pairs found")
        return 1
    logging.info("Processing %d cameras: %s", len(pairs), ",".join(sorted(pairs)))

    master_parts: list[pd.DataFrame] = []
    for cam in sorted(pairs):
        try:
            df = process_camera(
                cam, pairs[cam]["slp"], pairs[cam]["aruco"],
                args.match_radius, args.edge_margin,
            )
        except Exception:
            logging.exception("[%s] failed - skipping", cam)
            continue
        out_path = out_dir / f"{cam}_inventory.parquet"
        saved = save_df(df, out_path)
        logging.info("[%s] saved %s (%d rows)", cam, saved.name, len(df))
        master_parts.append(df)

    if master_parts:
        master = pd.concat(master_parts, ignore_index=True)
        saved = save_df(master, out_dir / "inventory_master.parquet")
        logging.info("master saved: %s (%d rows, %d cams)",
                     saved.name, len(master), master["camera"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
