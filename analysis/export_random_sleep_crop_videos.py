#!/usr/bin/env python3
"""Export random sleep crop videos filtered by projected track visibility."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from analysis.export_sleep_crop_videos import (
    DEFAULT_CHUNK_FRAMES,
    DEFAULT_HMATS,
    ChunkTrackIndex,
    build_track_specs,
    export_batch,
    export_h264_stream_batch,
    aruco_curation_tools,
    log,
    make_writer,
    multicam_tracking_tools,
    open_video_reader,
    read_window_points,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block_root",
        type=Path,
        default=Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02"),
        help="Block folder containing cam*.avi and tracks/.",
    )
    parser.add_argument("--out_dir", type=Path, default=None, help="Output folder.")
    parser.add_argument("--n_videos", type=int, default=100, help="Number of crop videos to export.")
    parser.add_argument("--duration_sec", type=float, default=60.0, help="Clip duration in seconds.")
    parser.add_argument("--crop_size_px", type=int, default=480, help="Square crop size.")
    parser.add_argument("--min_coverage", type=float, default=1.0, help="Minimum visible fraction in the window.")
    parser.add_argument(
        "--max_gap_frames",
        type=int,
        default=0,
        help="Interpolate track-center gaps up to this many frames after coverage filtering.",
    )
    parser.add_argument("--seed", type=int, default=480100, help="Random seed.")
    parser.add_argument("--max_attempts", type=int, default=2000, help="Maximum random windows to test.")
    parser.add_argument(
        "--max_tracks_per_window",
        type=int,
        default=5,
        help="Maximum passing tracks to export from one random time window.",
    )
    parser.add_argument("--hmats", type=Path, default=DEFAULT_HMATS, help="Homography .npz with key H.")
    parser.add_argument("--video_backend", choices=["h264", "opencv", "auto"], default="h264")
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc for output videos.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    total_start = time.perf_counter()

    block_root = Path(args.block_root)
    tracks_dir = block_root / "tracks"
    videos = sorted(block_root.glob("cam*.avi"))
    if not videos:
        raise FileNotFoundError(f"No cam*.avi files found in {block_root}")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else block_root
        / "stitched"
        / "sleep_crop_videos"
        / (
            f"random_{int(args.n_videos)}_{int(round(float(args.duration_sec)))}s_"
            f"min{int(round(100 * float(args.min_coverage)))}pct_"
            f"{int(args.crop_size_px)}px_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    track_index = ChunkTrackIndex(tracks_dir, chunk_frames=DEFAULT_CHUNK_FRAMES, chunk_offset_mode="metadata")
    estimated_frame_count = int(track_index.frame_count_estimate())
    AviH264ElementaryReader, _OpenCvVideoReader, _resolve_frame_count = aruco_curation_tools()
    _apply_homography_points, _discover_media, load_homography_stack, parse_camera_index = multicam_tracking_tools()
    homographies = load_homography_stack(args.hmats)

    probe_reader = open_video_reader(videos[0], estimated_frame_count, backend=args.video_backend)
    try:
        fps = float(getattr(probe_reader, "fps", 24.0) or 24.0)
        video_width = int(getattr(probe_reader, "width", 0)) or 1
        video_height = int(getattr(probe_reader, "height", 0)) or 1
    finally:
        probe_reader.close()

    duration_frames = max(1, int(round(float(args.duration_sec) * fps)))
    if duration_frames >= estimated_frame_count:
        raise ValueError("duration_sec is longer than the estimated recording")

    log(f"block_root={block_root}")
    log(f"out_dir={out_dir}")
    log(
        f"target={int(args.n_videos)} videos, duration={duration_frames} frames "
        f"({duration_frames / fps:.1f}s), crop={int(args.crop_size_px)}px, "
        f"min_coverage={float(args.min_coverage):.3f}, max_gap_frames={int(args.max_gap_frames)}, "
        f"seed={int(args.seed)}"
    )

    selected_rows: list[dict[str, object]] = []
    selected_groups: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    used_keys: set[tuple[str, int, str, int]] = set()

    for attempt in range(int(args.max_attempts)):
        if len(selected_rows) >= int(args.n_videos):
            break

        video_path = videos[int(rng.integers(0, len(videos)))]
        frame_start = int(rng.integers(0, estimated_frame_count - duration_frames))
        frame_stop = int(frame_start + duration_frames)
        cam_index = parse_camera_index(video_path)
        if cam_index < 0 or cam_index >= len(homographies):
            failure_rows.append(
                {
                    "attempt": attempt,
                    "video": str(video_path),
                    "frame_start": frame_start,
                    "reason": "camera_index_out_of_range",
                }
            )
            continue
        inv_h = np.linalg.inv(homographies[cam_index])

        points = read_window_points(
            track_index,
            frame_start=frame_start,
            frame_stop=frame_stop,
            sides={"left", "right"},
        )
        specs = build_track_specs(
            points,
            inv_h=inv_h,
            video_width=video_width,
            video_height=video_height,
            frame_start=frame_start,
            frame_stop=frame_stop,
            out_dir=out_dir,
            video_stem=video_path.stem,
            crop_size=int(args.crop_size_px),
            min_coverage=float(args.min_coverage),
            max_gap_frames=int(args.max_gap_frames),
            track_ids=None,
            max_tracks=0,
        )
        specs = [
            spec
            for spec in specs
            if (str(video_path), frame_start, spec.side, int(spec.track_id)) not in used_keys
        ]
        if not specs:
            failure_rows.append(
                {
                    "attempt": attempt,
                    "video": str(video_path),
                    "frame_start": frame_start,
                    "reason": "no_track_met_coverage",
                }
            )
            continue

        remaining = int(args.n_videos) - len(selected_rows)
        n_take = min(int(args.max_tracks_per_window), remaining, len(specs))
        chosen_specs = [specs[int(i)] for i in rng.permutation(len(specs))[:n_take]]
        log(
            f"selected {len(selected_rows) + len(chosen_specs)}/{int(args.n_videos)}: "
            f"{video_path.name} frame={frame_start:,}, {len(chosen_specs)} passing tracks"
        )

        rows = []
        for spec in chosen_specs:
            key = (str(video_path), frame_start, spec.side, int(spec.track_id))
            used_keys.add(key)
            row = {
                "selection_index": len(selected_rows) + 1,
                "attempt": attempt,
                "seed": int(args.seed),
                "camera_video": str(video_path),
                "frame_start_requested": int(frame_start),
                "start_time_seconds": float(frame_start / fps),
                "side": spec.side,
                "track_id": int(spec.track_id),
                "output_path": str(spec.output_path),
                "video_path": str(video_path),
                "tracks_dir": str(tracks_dir),
                "frame_start": int(frame_start),
                "frame_stop_exclusive": int(frame_stop),
                "fps": float(fps),
                "duration_seconds": float(duration_frames / fps),
                "crop_size_px": int(args.crop_size_px),
                "min_coverage_required": float(args.min_coverage),
                "max_gap_frames": int(args.max_gap_frames),
                "observed_frames": int(spec.observed.sum()),
                "window_frames": int(duration_frames),
                "coverage": float(spec.coverage),
            }
            rows.append(row)
            selected_rows.append(row)
        selected_groups.append(
            {
                "video_path": video_path,
                "frame_start": frame_start,
                "frame_stop": frame_stop,
                "specs": chosen_specs,
                "rows": rows,
            }
        )
        pd.DataFrame(selected_rows).to_csv(out_dir / "random_crop_selected.csv", index=False)
        if failure_rows:
            pd.DataFrame(failure_rows).to_csv(out_dir / "random_crop_failures.csv", index=False)

    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(out_dir / "random_crop_failures.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out_dir / "random_crop_selected.csv", index=False)

    if len(selected_rows) < int(args.n_videos):
        raise RuntimeError(
            f"Only selected {len(selected_rows)} of {int(args.n_videos)} passing videos after "
            f"{int(args.max_attempts)} attempts"
        )

    log(f"selected {len(selected_rows)} passing crops across {len(selected_groups)} random windows")

    manifest_rows: list[dict[str, object]] = []
    groups_by_video: dict[Path, list[dict[str, object]]] = {}
    for group in selected_groups:
        groups_by_video.setdefault(Path(group["video_path"]), []).append(group)

    for video_path in sorted(groups_by_video):
        video_groups = sorted(groups_by_video[video_path], key=lambda item: int(item["frame_start"]))
        log(f"exporting {sum(len(group['specs']) for group in video_groups)} crops from {video_path.name}")
        reader = None
        try:
            if args.video_backend == "h264":
                reader = open_video_reader(video_path, estimated_frame_count, backend=args.video_backend)
            for group in video_groups:
                specs = list(group["specs"])
                frame_start = int(group["frame_start"])
                frame_stop = int(group["frame_stop"])
                log(f"  window frame={frame_start:,}-{frame_stop - 1:,}, {len(specs)} crops")
                try:
                    if reader is not None and isinstance(reader, AviH264ElementaryReader):
                        writers = []
                        try:
                            writers = [
                                make_writer(
                                    spec.output_path,
                                    fps=fps,
                                    crop_size=int(args.crop_size_px),
                                    codec=str(args.codec),
                                )
                                for spec in specs
                            ]
                            export_h264_stream_batch(
                                reader=reader,
                                frame_start=frame_start,
                                frame_stop=frame_stop,
                                specs=specs,
                                writers=writers,
                                crop_size=int(args.crop_size_px),
                            )
                        finally:
                            for writer in writers:
                                writer.release()
                    else:
                        export_batch(
                            video_path=video_path,
                            backend=args.video_backend,
                            estimated_frame_count=estimated_frame_count,
                            frame_start=frame_start,
                            frame_stop=frame_stop,
                            specs=specs,
                            fps=fps,
                            crop_size=int(args.crop_size_px),
                            codec=str(args.codec),
                        )
                except Exception as exc:
                    for spec in specs:
                        try:
                            spec.output_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                    failure_rows.append(
                        {
                            "attempt": group["rows"][0]["attempt"],
                            "video": str(video_path),
                            "frame_start": frame_start,
                            "reason": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    pd.DataFrame(failure_rows).to_csv(out_dir / "random_crop_failures.csv", index=False)
                    continue

                manifest_rows.extend(group["rows"])
                manifest = pd.DataFrame(manifest_rows).sort_values("selection_index", kind="mergesort")
                manifest.insert(0, "sample_index", np.arange(1, len(manifest) + 1, dtype=int))
                manifest.to_csv(out_dir / "random_crop_manifest.csv", index=False)
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass

    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(out_dir / "random_crop_failures.csv", index=False)
    manifest = pd.DataFrame(manifest_rows).sort_values("selection_index", kind="mergesort")
    if not manifest.empty:
        manifest.insert(0, "sample_index", np.arange(1, len(manifest) + 1, dtype=int))
    manifest.to_csv(out_dir / "random_crop_manifest.csv", index=False)

    if len(manifest_rows) < int(args.n_videos):
        raise RuntimeError(
            f"Only exported {len(manifest_rows)} of {int(args.n_videos)} videos after "
            f"{len(selected_groups)} selected windows"
        )

    log(f"wrote manifest: {out_dir / 'random_crop_manifest.csv'}")
    log(f"wrote {len(manifest_rows)} videos to {out_dir}")
    log(f"done in {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
