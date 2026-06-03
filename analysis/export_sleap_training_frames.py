#!/usr/bin/env python3
"""Export suspect SLEAP frames for additional labeling.

This scans existing tracks or per-camera SLEAP outputs, then exports:
  - raw video frames for import into SLEAP
  - overlay PNGs showing the current SLEAP labels
  - a CSV summary of why each frame was selected

With --tracks_dir, candidate ranking uses only existing track parquets and then
exports raw camera frames from the matching chunk videos. It does not load ArUco
files; SLEAP files are loaded only for the selected exported frames when writing
annotated overlays.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd


def log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class Candidate:
    score: float
    reason: str
    video_path: Path
    sleap_path: Path
    frame: int
    instance: int
    bodypoint: int | None = None
    aruco_id: int | None = None
    aruco_x: float | None = None
    aruco_y: float | None = None


@dataclass(frozen=True)
class TrackCandidate:
    score: float
    reason: str
    track_path: Path
    chunk_idx: int
    side: str
    frame: int
    track_id: int
    bodypoint: int | None = None


def dedupe_track_candidates(candidates: list[TrackCandidate]) -> list[TrackCandidate]:
    """Keep one strongest example per tracked ant and frame."""
    best: dict[tuple[Path, int, int], TrackCandidate] = {}
    for candidate in candidates:
        key = (candidate.track_path, candidate.frame, candidate.track_id)
        previous = best.get(key)
        if previous is None or candidate.score > previous.score:
            best[key] = candidate
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


def dedupe_raw_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Keep one strongest example per raw SLEAP instance and frame."""
    best: dict[tuple[Path, int, int], Candidate] = {}
    for candidate in candidates:
        key = (candidate.video_path, candidate.frame, candidate.instance)
        previous = best.get(key)
        if previous is None or candidate.score > previous.score:
            best[key] = candidate
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


def normalize_sleap_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    required = {"Frame", "Instance", "Bodypoint", "X", "Y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"SLEAP table missing columns: {sorted(missing)}")

    cols = ["Frame", "Instance", "Bodypoint", "X", "Y"]
    for optional in ["Score_node", "Score"]:
        if optional in df.columns:
            cols.append(optional)
    out = df[cols].copy()

    for col in ["Frame", "Instance", "Bodypoint"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["X"] = pd.to_numeric(out["X"], errors="coerce")
    out["Y"] = pd.to_numeric(out["Y"], errors="coerce")
    out = out.dropna(subset=["Frame", "Instance", "Bodypoint", "X", "Y"])
    out[["Frame", "Instance", "Bodypoint"]] = out[["Frame", "Instance", "Bodypoint"]].astype(int)
    return out


def load_sleap(path: Path) -> pd.DataFrame:
    t0 = time.perf_counter()
    log(f"Loading SLEAP {path.name}")
    try:
        df = normalize_sleap_df(pd.read_hdf(path, key="sleap_data"))
        log(f"  SLEAP rows={len(df):,} time={time.perf_counter() - t0:.1f}s")
        return df
    except Exception:
        pass

    with h5py.File(path, "r") as f:
        if "sleap_data" in f:
            arr = f["sleap_data"][:]
            if arr.dtype.names:
                df = normalize_sleap_df(pd.DataFrame.from_records(arr))
                log(f"  SLEAP rows={len(df):,} time={time.perf_counter() - t0:.1f}s")
                return df

        required = ["Frame", "Instance", "Bodypoint", "X", "Y"]
        missing = [name for name in required if name not in f]
        if missing:
            raise ValueError(f"{path} missing datasets: {missing}")
        data = {name: np.squeeze(f[name][:]) for name in required}
        if "Score_node" in f:
            data["Score_node"] = np.squeeze(f["Score_node"][:])
        df = normalize_sleap_df(pd.DataFrame(data))
        log(f"  SLEAP rows={len(df):,} time={time.perf_counter() - t0:.1f}s")
        return df


def video_for_sleap(sleap_path: Path) -> Path | None:
    stem = re.sub(r"_sleap_data$", "", sleap_path.stem)
    for suffix in [".avi", ".mp4", ".mov", ".mkv"]:
        candidate = sleap_path.with_name(stem + suffix)
        if candidate.exists():
            return candidate
    return None


def aruco_for_sleap(sleap_path: Path) -> Path | None:
    stem = re.sub(r"_sleap_data$", "", sleap_path.stem)
    for suffix in ["_aruco_detections.h5", "_aruco_detections.csv", "_aruco.csv"]:
        candidate = sleap_path.with_name(stem + suffix)
        if candidate.exists():
            return candidate
    return None


def load_aruco(path: Path) -> pd.DataFrame:
    t0 = time.perf_counter()
    log(f"Loading ArUco {path.name}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_hdf(path, key="detections")
    required = {"Frame", "Instance", "X", "Y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"ArUco table missing columns: {sorted(missing)}")
    out = df[["Frame", "Instance", "X", "Y"]].copy()
    for col in ["Frame", "Instance"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["X"] = pd.to_numeric(out["X"], errors="coerce")
    out["Y"] = pd.to_numeric(out["Y"], errors="coerce")
    out = out.dropna(subset=["Frame", "Instance", "X", "Y"])
    out[["Frame", "Instance"]] = out[["Frame", "Instance"]].astype(int)
    log(f"  ArUco rows={len(out):,} time={time.perf_counter() - t0:.1f}s")
    return out


def parse_track_file(path: Path) -> tuple[int, str]:
    match = re.search(r"chunk(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse chunk index from {path.name}")
    side = "right" if "right" in path.stem.lower() else "left"
    return int(match.group(1)), side


def score_track_file(
    track_path: Path,
    *,
    expected_bodypoints: int | None,
    track_jump: float,
    max_candidates_per_file: int,
) -> list[TrackCandidate]:
    t0 = time.perf_counter()
    chunk_idx, side = parse_track_file(track_path)
    cols = ["Frame", "TrackID", "Bodypoint", "X", "Y"]
    log(f"Scoring track file {track_path.name}")
    df = pd.read_parquet(track_path, columns=cols)
    if df.empty:
        log("  empty file")
        return []

    candidates: list[TrackCandidate] = []
    expected = expected_bodypoints if expected_bodypoints is not None else int(df["Bodypoint"].nunique())

    counts = df.groupby(["Frame", "TrackID"])["Bodypoint"].nunique().reset_index(name="n_bodypoints")
    counts["missing"] = np.maximum(0, expected - counts["n_bodypoints"])
    for row in counts[counts["missing"] > 0].itertuples(index=False):
        candidates.append(
            TrackCandidate(
                score=float(row.missing),
                reason=f"track_missing_{int(row.missing)}_bodypoints",
                track_path=track_path,
                chunk_idx=chunk_idx,
                side=side,
                frame=int(row.Frame),
                track_id=int(row.TrackID),
            )
        )

    jumps = df.sort_values(["TrackID", "Bodypoint", "Frame"]).copy()
    grouped = jumps.groupby(["TrackID", "Bodypoint"])
    jumps["prev_x"] = grouped["X"].shift()
    jumps["prev_y"] = grouped["Y"].shift()
    jumps["prev_frame"] = grouped["Frame"].shift()
    jumps["jump"] = np.hypot(jumps["X"] - jumps["prev_x"], jumps["Y"] - jumps["prev_y"])
    jumps = jumps[(jumps["prev_frame"] == jumps["Frame"] - 1) & (jumps["jump"] >= track_jump)]
    for row in jumps.itertuples(index=False):
        candidates.append(
            TrackCandidate(
                score=float(row.jump),
                reason=f"track_jump_bp{int(row.Bodypoint)}_{float(row.jump):.1f}",
                track_path=track_path,
                chunk_idx=chunk_idx,
                side=side,
                frame=int(row.Frame),
                track_id=int(row.TrackID),
                bodypoint=int(row.Bodypoint),
            )
        )

    raw_count = len(candidates)
    candidates = dedupe_track_candidates(candidates)
    out = candidates[:max_candidates_per_file]
    log(
        f"  rows={len(df):,} raw_candidates={raw_count:,} "
        f"unique_frame_track_candidates={len(candidates):,} "
        f"kept={len(out):,} time={time.perf_counter() - t0:.1f}s"
    )
    return out


def score_tracks_dir(
    tracks_dir: Path,
    *,
    expected_bodypoints: int | None,
    track_jump: float,
    max_candidates_per_file: int,
    pattern: str,
) -> list[TrackCandidate]:
    candidates: list[TrackCandidate] = []
    track_files = sorted(tracks_dir.glob(pattern))
    log(f"Found {len(track_files)} track parquet files in {tracks_dir}")
    for i, track_path in enumerate(track_files, start=1):
        log(f"[tracks {i}/{len(track_files)}]")
        candidates.extend(
            score_track_file(
                track_path,
                expected_bodypoints=expected_bodypoints,
                track_jump=track_jump,
                max_candidates_per_file=max_candidates_per_file,
            )
        )
    candidates = dedupe_track_candidates(candidates)
    log(f"Total track candidates after ranking: {len(candidates):,}")
    return candidates


def sleap_files_for_chunk(data_dir: Path, chunk_idx: int) -> list[Path]:
    return sorted(data_dir.glob(f"*_{chunk_idx:03d}_sleap_data.h5"))


def video_files_for_chunk(data_dir: Path, chunk_idx: int) -> list[Path]:
    videos: list[Path] = []
    for suffix in ("avi", "mp4", "mov", "mkv"):
        videos.extend(data_dir.glob(f"*_{chunk_idx:03d}.{suffix}"))
    return sorted(videos)


def raw_candidates_from_track_candidates(
    track_candidates: list[TrackCandidate],
    *,
    data_dir: Path,
    max_cameras_per_candidate: int,
    aruco_cache: dict[Path, pd.DataFrame],
    sleap_cache: dict[Path, pd.DataFrame],
    anchor_bodypoint: int,
) -> list[Candidate]:
    raw_candidates: list[Candidate] = []
    log(f"Resolving {len(track_candidates):,} track candidates to raw camera frames")
    for n, tc in enumerate(track_candidates, start=1):
        if n == 1 or n % 10 == 0 or n == len(track_candidates):
            log(
                f"[resolve {n}/{len(track_candidates)}] "
                f"chunk={tc.chunk_idx:03d} side={tc.side} frame={tc.frame} "
                f"track={tc.track_id} score={tc.score:.1f}"
            )
        exported_for_tc = 0
        sleap_files = sleap_files_for_chunk(data_dir, tc.chunk_idx)
        for sleap_path in sleap_files:
            if exported_for_tc >= max_cameras_per_candidate:
                break
            video_path = video_for_sleap(sleap_path)
            aruco_path = aruco_for_sleap(sleap_path)
            if video_path is None or aruco_path is None:
                continue

            aruco = aruco_cache.setdefault(aruco_path, load_aruco(aruco_path))
            a = aruco[(aruco["Frame"] == tc.frame) & (aruco["Instance"] == tc.track_id)]
            if a.empty:
                continue
            ax = float(a.iloc[0]["X"])
            ay = float(a.iloc[0]["Y"])

            sleap = sleap_cache.setdefault(sleap_path, load_sleap(sleap_path))
            anchors = sleap[(sleap["Frame"] == tc.frame) & (sleap["Bodypoint"] == anchor_bodypoint)]
            nearest_inst = -1
            if not anchors.empty:
                dists = np.hypot(anchors["X"].to_numpy(float) - ax, anchors["Y"].to_numpy(float) - ay)
                nearest_inst = int(anchors.iloc[int(np.argmin(dists))]["Instance"])

            raw_candidates.append(
                Candidate(
                    score=tc.score,
                    reason=tc.reason,
                    video_path=video_path,
                    sleap_path=sleap_path,
                    frame=tc.frame,
                    instance=nearest_inst,
                    bodypoint=tc.bodypoint,
                    aruco_id=tc.track_id,
                    aruco_x=ax,
                    aruco_y=ay,
                )
            )
            exported_for_tc += 1
        if exported_for_tc == 0:
            log(
                f"  no raw camera match for chunk={tc.chunk_idx:03d} "
                f"frame={tc.frame} track={tc.track_id}"
            )
    log(f"Resolved raw export candidates: {len(raw_candidates):,}")
    return raw_candidates


def draw_track_diagnostic(
    df: pd.DataFrame,
    candidate: TrackCandidate,
    *,
    context_frames: int,
    width: int = 1200,
    height: int = 900,
) -> np.ndarray:
    context = df[
        (df["TrackID"] == candidate.track_id)
        & (df["Frame"] >= candidate.frame - context_frames)
        & (df["Frame"] <= candidate.frame + context_frames)
    ].copy()
    current = context[context["Frame"] == candidate.frame]
    if context.empty:
        context = df[(df["TrackID"] == candidate.track_id) & (df["Frame"] == candidate.frame)].copy()
        current = context

    xs = context["X"].to_numpy(float)
    ys = context["Y"].to_numpy(float)
    x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
    y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
    if abs(x_max - x_min) < 1e-6:
        x_min -= 1.0
        x_max += 1.0
    if abs(y_max - y_min) < 1e-6:
        y_min -= 1.0
        y_max += 1.0
    pad_x = 0.08 * (x_max - x_min)
    pad_y = 0.08 * (y_max - y_min)
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    image = np.full((height, width, 3), 24, dtype=np.uint8)
    left, top = 72, 64
    right, bottom = width - 44, height - 72
    plot_w = max(1, right - left)
    plot_h = max(1, bottom - top)

    def to_px(x: float, y: float) -> tuple[int, int]:
        px = left + int(round((x - x_min) / max(1e-6, x_max - x_min) * plot_w))
        py = top + int(round((y - y_min) / max(1e-6, y_max - y_min) * plot_h))
        return int(np.clip(px, 0, width - 1)), int(np.clip(py, 0, height - 1))

    cv2.rectangle(image, (left, top), (right, bottom), (80, 80, 80), 1)

    anchor = context[context["Bodypoint"] == 0].sort_values("Frame")
    last_pt = None
    for row in anchor.itertuples(index=False):
        pt = to_px(float(row.X), float(row.Y))
        if last_pt is not None:
            cv2.line(image, last_pt, pt, (95, 95, 95), 1, cv2.LINE_AA)
        cv2.circle(image, pt, 2, (130, 130, 130), -1, cv2.LINE_AA)
        last_pt = pt

    for row in current.itertuples(index=False):
        is_suspect = candidate.bodypoint is None or int(row.Bodypoint) == candidate.bodypoint
        color = (0, 0, 255) if is_suspect else (0, 255, 255)
        radius = 10 if is_suspect else 7
        pt = to_px(float(row.X), float(row.Y))
        cv2.circle(image, pt, radius, color, -1, cv2.LINE_AA)
        draw_text_with_outline(
            image,
            str(int(row.Bodypoint)),
            (pt[0] + radius + 4, pt[1] - radius - 2),
            color,
            scale=1.0 if is_suspect else 0.75,
            thickness=2,
        )

    draw_text_with_outline(
        image,
        f"{candidate.track_path.name} | frame {candidate.frame} | TrackID {candidate.track_id}",
        (24, 32),
        (255, 255, 255),
        scale=0.75,
        thickness=2,
    )
    draw_text_with_outline(
        image,
        f"{candidate.reason} | score {candidate.score:.1f} | context +/-{context_frames} frames",
        (24, height - 28),
        (255, 255, 255),
        scale=0.65,
        thickness=2,
    )
    return image


def export_track_candidate(
    candidate: TrackCandidate,
    out_dir: Path,
    track_cache: dict[Path, pd.DataFrame],
    *,
    context_frames: int,
) -> dict[str, object]:
    df = track_cache.setdefault(
        candidate.track_path,
        pd.read_parquet(candidate.track_path, columns=["Frame", "TrackID", "Bodypoint", "X", "Y"]),
    )
    image = draw_track_diagnostic(df, candidate, context_frames=context_frames)
    base = (
        f"{candidate.track_path.stem}_frame{candidate.frame:06d}"
        f"_track{candidate.track_id:04d}_{candidate.reason}"
    )
    out_path = out_dir / "track_diagnostics" / f"{base}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)
    return {
        "score": candidate.score,
        "reason": candidate.reason,
        "track_file": str(candidate.track_path),
        "chunk": candidate.chunk_idx,
        "side": candidate.side,
        "frame": candidate.frame,
        "track_id": candidate.track_id,
        "bodypoint": candidate.bodypoint,
        "diagnostic_png": str(out_path),
    }


def export_track_candidate_raw_frames(
    candidate: TrackCandidate,
    *,
    data_dir: Path,
    out_dir: Path,
    max_cameras: int,
    sleap_cache: dict[Path, pd.DataFrame],
    write_annotated: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    videos = video_files_for_chunk(data_dir, candidate.chunk_idx)
    if max_cameras > 0:
        videos = videos[:max_cameras]
    if not videos:
        log(f"  no videos found for chunk {candidate.chunk_idx:03d}")
        return rows

    for video_path in videos:
        frame = read_frame(video_path, candidate.frame)
        if frame is None:
            log(f"  could not read {video_path.name} frame={candidate.frame}")
            continue

        annotated_path: Path | None = None
        suspect_instance: int | None = None
        annotation_status = "not_requested"
        annotated: np.ndarray | None = None
        if write_annotated:
            sleap_path = video_path.with_name(video_path.stem + "_sleap_data.h5")
            if not sleap_path.exists():
                annotation_status = "missing_sleap_file"
                log(f"  no SLEAP file for annotation: {sleap_path.name}")
            else:
                labels = sleap_cache.setdefault(sleap_path, load_sleap(sleap_path))
                frame_labels = labels[labels["Frame"] == candidate.frame]
                suspect_instance = infer_suspect_instance(
                    labels,
                    frame=candidate.frame,
                    suspect_bodypoint=candidate.bodypoint,
                )
                if suspect_instance is None:
                    log(
                        f"  skipping unresolved annotation for {video_path.name} "
                        f"frame={candidate.frame} track={candidate.track_id}"
                    )
                    continue
                annotation_status = "ok"
                annotated = draw_sleap_frame_overlay(
                    frame,
                    frame_labels,
                    reason=candidate.reason,
                    suspect_instance=suspect_instance,
                    suspect_bodypoint=candidate.bodypoint,
                )

        base = (
            f"{video_path.stem}_frame{candidate.frame:06d}"
            f"_track{candidate.track_id:04d}_{candidate.reason}"
        )
        raw_path = out_dir / "raw_frames" / f"{base}.png"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(raw_path), frame)
        if annotated is not None:
            annotated_path = out_dir / "annotated_frames" / f"{base}_annotated.png"
            annotated_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(annotated_path), annotated)
        rows.append(
            {
                "score": candidate.score,
                "reason": candidate.reason,
                "track_file": str(candidate.track_path),
                "chunk": candidate.chunk_idx,
                "side": candidate.side,
                "frame": candidate.frame,
                "track_id": candidate.track_id,
                "bodypoint": candidate.bodypoint,
                "suspect_instance": suspect_instance if write_annotated else None,
                "annotation_status": annotation_status,
                "video": str(video_path),
                "raw_png": str(raw_path),
                "annotated_png": str(annotated_path) if annotated_path is not None else "",
            }
        )
    return rows


def score_sleap_file(
    sleap_path: Path,
    *,
    expected_bodypoints: int | None,
    jump_px: float,
    aruco_match_px: float,
    aruco_history_frames: int,
    anchor_bodypoint: int,
    max_candidates_per_file: int,
) -> list[Candidate]:
    t0 = time.perf_counter()
    log(f"Scoring raw SLEAP file {sleap_path.name}")
    video_path = video_for_sleap(sleap_path)
    if video_path is None:
        log(f"  skipping: no matching video")
        return []

    df = load_sleap(sleap_path)
    if df.empty:
        return []

    expected = expected_bodypoints if expected_bodypoints is not None else int(df["Bodypoint"].nunique())
    candidates: list[Candidate] = []

    counts = df.groupby(["Frame", "Instance"])["Bodypoint"].nunique().reset_index(name="n_bodypoints")
    counts["missing"] = np.maximum(0, expected - counts["n_bodypoints"])
    for row in counts[counts["missing"] > 0].itertuples(index=False):
        candidates.append(
            Candidate(
                score=float(row.missing),
                reason=f"missing_{int(row.missing)}_bodypoints",
                video_path=video_path,
                sleap_path=sleap_path,
                frame=int(row.Frame),
                instance=int(row.Instance),
                bodypoint=None,
            )
        )

    jumps = df.sort_values(["Instance", "Bodypoint", "Frame"]).copy()
    grouped = jumps.groupby(["Instance", "Bodypoint"])
    jumps["prev_x"] = grouped["X"].shift()
    jumps["prev_y"] = grouped["Y"].shift()
    jumps["prev_frame"] = grouped["Frame"].shift()
    jumps["jump"] = np.hypot(jumps["X"] - jumps["prev_x"], jumps["Y"] - jumps["prev_y"])
    jumps = jumps[(jumps["prev_frame"] == jumps["Frame"] - 1) & (jumps["jump"] >= jump_px)]
    for row in jumps.itertuples(index=False):
        candidates.append(
            Candidate(
                score=float(row.jump),
                reason=f"jump_bp{int(row.Bodypoint)}_{float(row.jump):.1f}px",
                video_path=video_path,
                sleap_path=sleap_path,
                frame=int(row.Frame),
                instance=int(row.Instance),
                bodypoint=int(row.Bodypoint),
            )
        )

    aruco_path = aruco_for_sleap(sleap_path)
    if aruco_path is not None:
        aruco = load_aruco(aruco_path)
        anchor = df[df["Bodypoint"] == anchor_bodypoint][["Frame", "Instance", "X", "Y"]]
        anchor_by_frame = {int(f): g for f, g in anchor.groupby("Frame")}
        recent_supported: dict[int, int] = {}

        for frame_idx, a_frame in aruco.groupby("Frame"):
            frame_idx = int(frame_idx)
            s_frame = anchor_by_frame.get(frame_idx)
            for row in a_frame.itertuples(index=False):
                tag_id = int(row.Instance)
                ax, ay = float(row.X), float(row.Y)
                supported = False
                nearest_dist = float("inf")
                nearest_inst = -1

                if s_frame is not None and not s_frame.empty:
                    dx = s_frame["X"].to_numpy(float) - ax
                    dy = s_frame["Y"].to_numpy(float) - ay
                    dists = np.hypot(dx, dy)
                    nearest_idx = int(np.argmin(dists))
                    nearest_dist = float(dists[nearest_idx])
                    nearest_inst = int(s_frame.iloc[nearest_idx]["Instance"])
                    supported = nearest_dist <= aruco_match_px

                if supported:
                    recent_supported[tag_id] = frame_idx
                    continue

                last_supported = recent_supported.get(tag_id)
                if last_supported is None:
                    continue
                if frame_idx - last_supported > aruco_history_frames:
                    continue

                dist_score = nearest_dist if np.isfinite(nearest_dist) else aruco_match_px * 4.0
                candidates.append(
                    Candidate(
                        score=jump_px * 4.0 + dist_score,
                        reason=f"aruco_tag{tag_id}_lost_sleap_nearest_{dist_score:.1f}px",
                        video_path=video_path,
                        sleap_path=sleap_path,
                        frame=frame_idx,
                        instance=nearest_inst,
                        bodypoint=anchor_bodypoint,
                        aruco_id=tag_id,
                        aruco_x=ax,
                        aruco_y=ay,
                    )
                )

    raw_count = len(candidates)
    candidates = dedupe_raw_candidates(candidates)
    out = candidates[:max_candidates_per_file]
    log(
        f"  raw_candidates={raw_count:,} "
        f"unique_frame_instance_candidates={len(candidates):,} kept={len(out):,} "
        f"time={time.perf_counter() - t0:.1f}s"
    )
    return out


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def draw_text_with_outline(
    image: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float,
    thickness: int,
) -> None:
    cv2.putText(
        image,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    labels: pd.DataFrame,
    highlight_instance: int,
    reason: str,
    suspect_bodypoint: int | None,
    aruco_id: int | None = None,
    aruco_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    image = frame.copy()
    for inst, inst_df in labels.groupby("Instance"):
        is_suspect_instance = int(inst) == highlight_instance
        for row in inst_df.itertuples(index=False):
            is_suspect_point = is_suspect_instance and (
                suspect_bodypoint is None or int(row.Bodypoint) == suspect_bodypoint
            )
            if is_suspect_point:
                color = (0, 0, 255)
                radius = 10
                scale = 1.2
                thickness = 3
            elif is_suspect_instance:
                color = (0, 255, 255)
                radius = 7
                scale = 0.9
                thickness = 2
            else:
                color = (180, 180, 180)
                radius = 4
                scale = 0.7
                thickness = 2
            x, y = int(round(row.X)), int(round(row.Y))
            cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            draw_text_with_outline(
                image,
                f"{int(row.Bodypoint)}",
                (x + radius + 4, y - radius - 2),
                color,
                scale=scale,
                thickness=thickness,
            )

    draw_text_with_outline(
        image,
        f"instance {highlight_instance} | {reason}",
        (20, 35),
        (255, 255, 255),
        scale=1.0,
        thickness=2,
    )
    if aruco_id is not None and aruco_xy is not None:
        ax, ay = int(round(aruco_xy[0])), int(round(aruco_xy[1]))
        cv2.drawMarker(
            image,
            (ax, ay),
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=36,
            thickness=4,
            line_type=cv2.LINE_AA,
        )
        draw_text_with_outline(
            image,
            f"aruco {aruco_id}",
            (ax + 18, ay + 18),
            (255, 0, 255),
            scale=1.2,
            thickness=3,
        )
    return image


def draw_sleap_frame_overlay(
    frame: np.ndarray,
    labels: pd.DataFrame,
    *,
    reason: str,
    suspect_instance: int | None,
    suspect_bodypoint: int | None,
) -> np.ndarray:
    image = frame.copy()
    for inst, inst_df in labels.groupby("Instance"):
        is_suspect_instance = suspect_instance is not None and int(inst) == suspect_instance
        for row in inst_df.itertuples(index=False):
            is_problem_point = is_suspect_instance and (
                suspect_bodypoint is None or int(row.Bodypoint) == suspect_bodypoint
            )
            color = (0, 0, 255) if is_problem_point else (0, 255, 255)
            radius = 10 if is_problem_point else 7
            x, y = int(round(row.X)), int(round(row.Y))
            cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return image


def infer_suspect_instance(
    labels: pd.DataFrame,
    *,
    frame: int,
    suspect_bodypoint: int | None,
) -> int | None:
    frame_labels = labels[labels["Frame"] == frame]
    if frame_labels.empty:
        return None

    if suspect_bodypoint is None:
        counts = frame_labels.groupby("Instance")["Bodypoint"].nunique()
        if counts.empty:
            return None
        min_count = int(counts.min())
        sparse = counts[counts == min_count]
        return int(sparse.index[0]) if len(sparse) == 1 else None

    current = frame_labels[frame_labels["Bodypoint"] == suspect_bodypoint]
    if current.empty:
        return None

    previous = labels[(labels["Frame"] == frame - 1) & (labels["Bodypoint"] == suspect_bodypoint)]
    if not previous.empty:
        merged = current.merge(
            previous[["Instance", "Bodypoint", "X", "Y"]],
            on=["Instance", "Bodypoint"],
            suffixes=("", "_prev"),
        )
        if not merged.empty:
            jumps = np.hypot(
                merged["X"].to_numpy(float) - merged["X_prev"].to_numpy(float),
                merged["Y"].to_numpy(float) - merged["Y_prev"].to_numpy(float),
            )
            return int(merged.iloc[int(np.argmax(jumps))]["Instance"])

    return int(current.iloc[0]["Instance"]) if len(current) == 1 else None


def export_candidate(candidate: Candidate, out_dir: Path, sleap_cache: dict[Path, pd.DataFrame]) -> dict[str, object] | None:
    t0 = time.perf_counter()
    frame = read_frame(candidate.video_path, candidate.frame)
    if frame is None:
        log(f"Could not read frame {candidate.frame} from {candidate.video_path.name}")
        return None

    labels = sleap_cache.setdefault(candidate.sleap_path, load_sleap(candidate.sleap_path))
    frame_labels = labels[labels["Frame"] == candidate.frame]
    overlay = draw_overlay(
        frame,
        frame_labels,
        candidate.instance,
        candidate.reason,
        candidate.bodypoint,
        aruco_id=candidate.aruco_id,
        aruco_xy=(
            (candidate.aruco_x, candidate.aruco_y)
            if candidate.aruco_x is not None and candidate.aruco_y is not None
            else None
        ),
    )

    base = (
        f"{candidate.video_path.stem}_frame{candidate.frame:06d}"
        f"_inst{candidate.instance:03d}_{candidate.reason}"
    )
    raw_path = out_dir / "raw_frames" / f"{base}.png"
    overlay_path = out_dir / "overlays" / f"{base}_overlay.png"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(raw_path), frame)
    cv2.imwrite(str(overlay_path), overlay)
    log(
        f"  exported {candidate.video_path.name} frame={candidate.frame} "
        f"reason={candidate.reason} time={time.perf_counter() - t0:.1f}s"
    )
    return {
        "score": candidate.score,
        "reason": candidate.reason,
        "video": str(candidate.video_path),
        "sleap": str(candidate.sleap_path),
        "frame": candidate.frame,
        "instance": candidate.instance,
        "bodypoint": candidate.bodypoint,
        "aruco_id": candidate.aruco_id,
        "aruco_x": candidate.aruco_x,
        "aruco_y": candidate.aruco_y,
        "raw_png": str(raw_path),
        "overlay_png": str(overlay_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True, help="Folder with videos and *_sleap_data.h5 files.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--tracks_dir", type=Path, default=None, help="Fast path: rank suspect frames from existing track parquets first.")
    parser.add_argument("--limit", type=int, default=200, help="Total number of candidates to export.")
    parser.add_argument("--per_file", type=int, default=20, help="Max candidates kept per SLEAP file before global ranking.")
    parser.add_argument("--jump_px", type=float, default=120.0, help="Bodypoint jump threshold in camera pixels.")
    parser.add_argument("--track_jump", type=float, default=1000.0, help="Track-space jump threshold for --tracks_dir mode.")
    parser.add_argument("--aruco_match_px", type=float, default=80.0, help="Max ArUco-to-SLEAP anchor distance for support.")
    parser.add_argument("--aruco_history_frames", type=int, default=5, help="Frames after support where lost SLEAP support is suspicious.")
    parser.add_argument("--anchor_bodypoint", type=int, default=0)
    parser.add_argument("--expected_bodypoints", type=int, default=None)
    parser.add_argument("--pattern", default="*_sleap_data.h5")
    parser.add_argument("--track_pattern", default="*.parquet")
    parser.add_argument(
        "--max_cameras_per_candidate",
        type=int,
        default=25,
        help="In --tracks_dir mode, export up to this many camera frames per suspect tracked frame. Use 0 for all.",
    )
    parser.add_argument(
        "--no_annotated",
        action="store_true",
        help="In --tracks_dir mode, skip annotated SLEAP overlay PNGs and write raw frames only.",
    )
    parser.add_argument("--track_context_frames", type=int, default=60)
    args = parser.parse_args()

    t_total = time.perf_counter()
    log("Starting SLEAP training-frame export")
    log(f"  data_dir={args.data_dir}")
    log(f"  out_dir={args.out_dir}")
    if args.tracks_dir is not None:
        log(f"  tracks_dir={args.tracks_dir}")

    sleap_cache: dict[Path, pd.DataFrame] = {}
    aruco_cache: dict[Path, pd.DataFrame] = {}
    if args.tracks_dir is not None:
        t0 = time.perf_counter()
        track_candidates = score_tracks_dir(
            args.tracks_dir,
            expected_bodypoints=args.expected_bodypoints,
            track_jump=args.track_jump,
            max_candidates_per_file=args.per_file,
            pattern=args.track_pattern,
        )
        track_candidates = track_candidates[: args.limit]
        log(f"Selected top {len(track_candidates):,} track candidates in {time.perf_counter() - t0:.1f}s")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, candidate in enumerate(track_candidates, start=1):
            log(f"[export raw frames {i}/{len(track_candidates)}]")
            rows.extend(
                export_track_candidate_raw_frames(
                    candidate,
                    data_dir=args.data_dir,
                    out_dir=args.out_dir,
                    max_cameras=args.max_cameras_per_candidate,
                    sleap_cache=sleap_cache,
                    write_annotated=not args.no_annotated,
                )
            )
        summary = pd.DataFrame(rows)
        summary_path = args.out_dir / "exported_raw_frames_from_tracks.csv"
        summary.to_csv(summary_path, index=False)
        log(f"Exported {len(summary)} raw camera frames for SLEAP import")
        log(f"Summary: {summary_path}")
        if args.no_annotated:
            log("No ArUco or SLEAP files were loaded in --tracks_dir mode.")
        else:
            log("No ArUco files were loaded. SLEAP files were loaded only for exported annotated frames.")
        log(f"Total time={time.perf_counter() - t_total:.1f}s")
        return
    else:
        sleap_files = sorted(args.data_dir.glob(args.pattern))
        if not sleap_files:
            raise SystemExit(f"No SLEAP files matching {args.pattern} in {args.data_dir}")

        log(f"Found {len(sleap_files)} raw SLEAP files")
        candidates = []
        for i, sleap_path in enumerate(sleap_files, start=1):
            log(f"[raw {i}/{len(sleap_files)}]")
            candidates.extend(
                score_sleap_file(
                    sleap_path,
                    expected_bodypoints=args.expected_bodypoints,
                    jump_px=args.jump_px,
                    aruco_match_px=args.aruco_match_px,
                    aruco_history_frames=args.aruco_history_frames,
                    anchor_bodypoint=args.anchor_bodypoint,
                    max_candidates_per_file=args.per_file,
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[: args.limit]
    if not candidates:
        raise SystemExit("No suspect frames found.")
    log(f"Selected {len(candidates):,} final export candidates")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, candidate in enumerate(candidates, start=1):
        log(f"[export {i}/{len(candidates)}]")
        row = export_candidate(candidate, args.out_dir, sleap_cache)
        if row is not None:
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = args.out_dir / "exported_frames.csv"
    summary.to_csv(summary_path, index=False)
    log(f"Exported {len(summary)} frame pairs")
    log(f"Summary: {summary_path}")
    log(f"Total time={time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
