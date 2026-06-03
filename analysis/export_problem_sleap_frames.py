#!/usr/bin/env python3
"""Export diverse problem frames from one camera's SLEAP output.

The script scores each video frame from the camera-level SLEAP table. A high
score means the frame has incomplete skeletons and/or bodypoints that jumped
too far from the previous frame. The selected frames are exported as raw PNGs
for SLEAP labeling plus annotated PNGs showing the current model output.
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
class ProblemPoint:
    instance: int
    bodypoint: int | None
    kind: str
    value: float


@dataclass(frozen=True)
class FrameScore:
    frame: int
    score: float
    missing_points: int
    jump_points: int
    max_jump: float
    problems: tuple[ProblemPoint, ...]


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
    log(f"Loading SLEAP: {path}")
    try:
        df = normalize_sleap_df(pd.read_hdf(path, key="sleap_data"))
        log(f"  rows={len(df):,} time={time.perf_counter() - t0:.1f}s")
        return df
    except Exception:
        pass

    with h5py.File(path, "r") as f:
        if "sleap_data" in f:
            arr = f["sleap_data"][:]
            if arr.dtype.names:
                df = normalize_sleap_df(pd.DataFrame.from_records(arr))
                log(f"  rows={len(df):,} time={time.perf_counter() - t0:.1f}s")
                return df

        required = ["Frame", "Instance", "Bodypoint", "X", "Y"]
        missing = [name for name in required if name not in f]
        if missing:
            raise ValueError(f"{path} missing datasets: {missing}")
        data = {name: np.squeeze(f[name][:]) for name in required}
        if "Score_node" in f:
            data["Score_node"] = np.squeeze(f["Score_node"][:])
        df = normalize_sleap_df(pd.DataFrame(data))
        log(f"  rows={len(df):,} time={time.perf_counter() - t0:.1f}s")
        return df


def infer_video_path(sleap_path: Path) -> Path | None:
    stem = re.sub(r"_sleap_data$", "", sleap_path.stem)
    for suffix in [".avi", ".mp4", ".mov", ".mkv"]:
        candidate = sleap_path.with_name(stem + suffix)
        if candidate.exists():
            return candidate
    return None


def sleap_folder_sort_key(path: Path) -> tuple[int, int, str]:
    chunk_match = re.search(r"_(\d{3})_sleap_data$", path.stem)
    cam_match = re.search(r"cam(\d+)", path.stem)
    chunk_idx = int(chunk_match.group(1)) if chunk_match else 10**9
    cam_idx = int(cam_match.group(1)) if cam_match else 10**9
    return chunk_idx, cam_idx, path.name


def expected_bodypoint_count(df: pd.DataFrame, requested: int | None) -> int:
    if requested is not None:
        return requested
    counts = df.groupby(["Frame", "Instance"])["Bodypoint"].nunique()
    if counts.empty:
        raise ValueError("No SLEAP labels found.")
    return int(counts.max())


def score_frames(
    df: pd.DataFrame,
    *,
    expected_bodypoints: int,
    jump_px: float,
    jump_weight: float,
    missing_weight: float,
) -> list[FrameScore]:
    t0 = time.perf_counter()
    log("Scoring incomplete skeletons and frame-to-frame jumps")

    problems_by_frame: dict[int, list[ProblemPoint]] = {}
    missing_by_frame: dict[int, int] = {}
    jump_by_frame: dict[int, int] = {}
    max_jump_by_frame: dict[int, float] = {}

    counts = df.groupby(["Frame", "Instance"])["Bodypoint"].nunique().reset_index(name="n_bodypoints")
    counts["missing"] = np.maximum(0, expected_bodypoints - counts["n_bodypoints"])
    for row in counts[counts["missing"] > 0].itertuples(index=False):
        frame = int(row.Frame)
        instance = int(row.Instance)
        missing_count = int(row.missing)
        missing_by_frame[frame] = missing_by_frame.get(frame, 0) + missing_count
        problems_by_frame.setdefault(frame, []).append(
            ProblemPoint(instance=instance, bodypoint=None, kind="incomplete", value=float(missing_count))
        )

    jumps = df.sort_values(["Instance", "Bodypoint", "Frame"]).copy()
    grouped = jumps.groupby(["Instance", "Bodypoint"])
    jumps["prev_x"] = grouped["X"].shift()
    jumps["prev_y"] = grouped["Y"].shift()
    jumps["prev_frame"] = grouped["Frame"].shift()
    jumps["jump"] = np.hypot(jumps["X"] - jumps["prev_x"], jumps["Y"] - jumps["prev_y"])
    jumps = jumps[(jumps["prev_frame"] == jumps["Frame"] - 1) & (jumps["jump"] >= jump_px)]
    for row in jumps.itertuples(index=False):
        frame = int(row.Frame)
        jump = float(row.jump)
        jump_by_frame[frame] = jump_by_frame.get(frame, 0) + 1
        max_jump_by_frame[frame] = max(max_jump_by_frame.get(frame, 0.0), jump)
        problems_by_frame.setdefault(frame, []).append(
            ProblemPoint(
                instance=int(row.Instance),
                bodypoint=int(row.Bodypoint),
                kind="jump",
                value=jump,
            )
        )

    scores: list[FrameScore] = []
    for frame, problems in problems_by_frame.items():
        missing_points = missing_by_frame.get(frame, 0)
        jump_points = jump_by_frame.get(frame, 0)
        max_jump = max_jump_by_frame.get(frame, 0.0)
        score = missing_weight * missing_points + jump_weight * jump_points + max_jump / max(jump_px, 1.0)
        scores.append(
            FrameScore(
                frame=frame,
                score=float(score),
                missing_points=missing_points,
                jump_points=jump_points,
                max_jump=float(max_jump),
                problems=tuple(problems),
            )
        )

    scores.sort(key=lambda s: (s.score, s.max_jump, s.missing_points + s.jump_points), reverse=True)
    log(f"  problem_frames={len(scores):,} time={time.perf_counter() - t0:.1f}s")
    return scores


def select_diverse_frames(scores: list[FrameScore], *, limit: int, min_frame_gap: int) -> list[FrameScore]:
    selected: list[FrameScore] = []
    for score in scores:
        if all(abs(score.frame - prev.frame) >= min_frame_gap for prev in selected):
            selected.append(score)
            if len(selected) >= limit:
                break
    selected.sort(key=lambda s: s.frame)
    log(f"Selected {len(selected):,} diverse frames with min_frame_gap={min_frame_gap}")
    return selected


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"OpenCV failed to write PNG: {path}")


def draw_overlay(frame: np.ndarray, frame_labels: pd.DataFrame, score: FrameScore) -> np.ndarray:
    image = frame.copy()
    problem_keys: set[tuple[int, int | None]] = {
        (problem.instance, problem.bodypoint) for problem in score.problems
    }
    problem_instances = {problem.instance for problem in score.problems}

    for instance, inst_df in frame_labels.groupby("Instance"):
        instance = int(instance)
        is_problem_instance = instance in problem_instances
        for row in inst_df.itertuples(index=False):
            bodypoint = int(row.Bodypoint)
            is_problem_point = (instance, bodypoint) in problem_keys or (instance, None) in problem_keys
            if is_problem_point:
                color = (0, 0, 255)
                radius = 10
            elif is_problem_instance:
                color = (0, 255, 255)
                radius = 7
            else:
                color = (180, 180, 180)
                radius = 4
            x, y = int(round(row.X)), int(round(row.Y))
            cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return image


def safe_reason(score: FrameScore) -> str:
    parts = []
    if score.missing_points:
        parts.append(f"missing{score.missing_points}")
    if score.jump_points:
        parts.append(f"jumps{score.jump_points}")
    return "_".join(parts) if parts else "problem"


def export_frames(
    *,
    df: pd.DataFrame,
    sleap_path: Path,
    video_path: Path,
    out_dir: Path,
    selected: list[FrameScore],
) -> pd.DataFrame:
    raw_dir = out_dir / "raw_frames"
    annotated_dir = out_dir / "annotated_frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    labels_by_frame = {int(frame): group for frame, group in df.groupby("Frame")}
    for i, score in enumerate(selected, start=1):
        log(f"[export {i}/{len(selected)}] frame={score.frame} score={score.score:.2f}")
        frame = read_frame(video_path, score.frame)
        if frame is None:
            log(f"  could not read frame {score.frame}")
            continue

        reason = safe_reason(score)
        base = f"{video_path.stem}_frame{score.frame:06d}_score{score.score:.2f}_{reason}"
        raw_path = raw_dir / f"{base}.png"
        annotated_path = annotated_dir / f"{base}_annotated.png"

        write_png(raw_path, frame)
        overlay = draw_overlay(frame, labels_by_frame.get(score.frame, pd.DataFrame()), score)
        write_png(annotated_path, overlay)
        log(f"  raw={raw_path}")
        log(f"  annotated={annotated_path}")

        rows.append(
            {
                "sleap": str(sleap_path),
                "video": str(video_path),
                "frame": score.frame,
                "score": score.score,
                "missing_points": score.missing_points,
                "jump_points": score.jump_points,
                "max_jump": score.max_jump,
                "problem_instances": ",".join(str(p.instance) for p in score.problems),
                "problem_bodypoints": ",".join("" if p.bodypoint is None else str(p.bodypoint) for p in score.problems),
                "problem_types": ",".join(p.kind for p in score.problems),
                "problem_values": ",".join(f"{p.value:.2f}" for p in score.problems),
                "raw_png": str(raw_path),
                "annotated_png": str(annotated_path),
            }
        )

    return pd.DataFrame(rows)


def process_sleap_file(
    *,
    sleap_path: Path,
    video_path: Path | None,
    out_dir: Path,
    frames_per_video: int,
    min_frame_gap: int,
    jump_px: float,
    expected_bodypoints_arg: int | None,
    jump_weight: float,
    missing_weight: float,
) -> pd.DataFrame:
    resolved_video = video_path or infer_video_path(sleap_path)
    if resolved_video is None:
        log(f"Skipping {sleap_path.name}: could not infer video path")
        return pd.DataFrame()
    if not resolved_video.exists():
        log(f"Skipping {sleap_path.name}: video does not exist: {resolved_video}")
        return pd.DataFrame()

    log("Processing camera SLEAP file")
    log(f"  sleap={sleap_path}")
    log(f"  video={resolved_video}")

    df = load_sleap(sleap_path)
    expected = expected_bodypoint_count(df, expected_bodypoints_arg)
    log(f"Expected bodypoints per complete skeleton: {expected}")

    scores = score_frames(
        df,
        expected_bodypoints=expected,
        jump_px=jump_px,
        jump_weight=jump_weight,
        missing_weight=missing_weight,
    )
    if not scores:
        log(f"No problematic frames found for {sleap_path.name}")
        return pd.DataFrame()

    selected = select_diverse_frames(scores, limit=frames_per_video, min_frame_gap=min_frame_gap)
    return export_frames(df=df, sleap_path=sleap_path, video_path=resolved_video, out_dir=out_dir, selected=selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleap", type=Path, default=None, help="Camera-level *_sleap_data.h5 file.")
    parser.add_argument("--sleap_dir", type=Path, default=None, help="Folder with camera-level SLEAP H5 files.")
    parser.add_argument("--pattern", default="*_sleap_data.h5", help="SLEAP filename pattern for --sleap_dir.")
    parser.add_argument("--video", type=Path, default=None, help="Matching camera video. Inferred when omitted.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Deprecated alias for --frames_per_video in single-file mode.")
    parser.add_argument("--frames_per_video", type=int, default=15, help="Number of diverse problem frames to export per video.")
    parser.add_argument("--min_frame_gap", type=int, default=240, help="Minimum spacing between exported frames.")
    parser.add_argument("--jump_px", type=float, default=80.0, help="Current-frame bodypoint jump threshold in pixels.")
    parser.add_argument("--expected_bodypoints", type=int, default=None)
    parser.add_argument("--jump_weight", type=float, default=1.0)
    parser.add_argument("--missing_weight", type=float, default=2.0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    if (args.sleap is None) == (args.sleap_dir is None):
        raise SystemExit("Pass exactly one of --sleap or --sleap_dir.")
    if args.video is not None and args.sleap_dir is not None:
        raise SystemExit("--video can only be used with --sleap, not --sleap_dir.")

    log("Starting camera SLEAP problem-frame export")
    log(f"  out_dir={args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames_per_video = args.limit if args.limit is not None else args.frames_per_video
    if frames_per_video <= 0:
        raise SystemExit("--frames_per_video must be positive.")

    if args.sleap is not None:
        summaries = [
            process_sleap_file(
                sleap_path=args.sleap,
                video_path=args.video,
                out_dir=args.out_dir,
                frames_per_video=frames_per_video,
                min_frame_gap=args.min_frame_gap,
                jump_px=args.jump_px,
                expected_bodypoints_arg=args.expected_bodypoints,
                jump_weight=args.jump_weight,
                missing_weight=args.missing_weight,
            )
        ]
    else:
        sleap_files = sorted(args.sleap_dir.glob(args.pattern), key=sleap_folder_sort_key)
        if not sleap_files:
            raise SystemExit(f"No SLEAP files matching {args.pattern} in {args.sleap_dir}")
        log(f"Found {len(sleap_files):,} SLEAP files")
        summaries = []
        for i, sleap_path in enumerate(sleap_files, start=1):
            log(f"[file {i}/{len(sleap_files)}]")
            summaries.append(
                process_sleap_file(
                    sleap_path=sleap_path,
                    video_path=None,
                    out_dir=args.out_dir,
                    frames_per_video=frames_per_video,
                    min_frame_gap=args.min_frame_gap,
                    jump_px=args.jump_px,
                    expected_bodypoints_arg=args.expected_bodypoints,
                    jump_weight=args.jump_weight,
                    missing_weight=args.missing_weight,
                )
            )

    summary = pd.concat([s for s in summaries if not s.empty], ignore_index=True) if summaries else pd.DataFrame()
    summary_path = args.out_dir / "problem_sleap_frames.csv"
    summary.to_csv(summary_path, index=False)

    log(f"Exported {len(summary):,} frame pairs")
    log(f"Summary: {summary_path}")
    log(f"Total time={time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
