#!/usr/bin/env python3
"""Export small per-ant crop videos for fast offline sleep/wake labeling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

DEFAULT_HMATS = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/cameraArray_calib/"
    "20260414_calibration_dataset/set0_patterns_elevated_by_2mm/"
    "frame0/initial_H_mats.npz"
)
DEFAULT_CHUNK_FRAMES = 172_800


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required to write crop videos; install the cv2 package.") from exc
    return cv2


def aruco_curation_tools():
    from tracking.gui.aruco_curation import AviH264ElementaryReader, OpenCvVideoReader, resolve_frame_count

    return AviH264ElementaryReader, OpenCvVideoReader, resolve_frame_count


def multicam_tracking_tools():
    from tracking.gui.multicam_tracking_viewer import (
        apply_homography_points,
        discover_media,
        load_homography_stack,
        parse_camera_index,
    )

    return apply_homography_points, discover_media, load_homography_stack, parse_camera_index


def project_xy(xy: np.ndarray, inv_h: np.ndarray) -> np.ndarray:
    if xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    finite = np.isfinite(xy).all(axis=1)
    out = np.full((len(xy), 2), np.nan, dtype=np.float64)
    if np.any(finite):
        apply_homography_points, _discover_media, _load_homography_stack, _parse_camera_index = multicam_tracking_tools()
        out[finite] = apply_homography_points(xy[finite], inv_h)
    return out


def parse_track_file(path: Path) -> tuple[int, str] | None:
    match = re.search(r"_chunk(\d+)_(left|right)\.parquet$", Path(path).name)
    if not match:
        return None
    return int(match.group(1)), str(match.group(2))


@dataclass(frozen=True)
class ChunkSpec:
    chunk: int
    start: int
    stop: int

    def contains(self, frame: int) -> bool:
        return int(self.start) <= int(frame) < int(self.stop)

    def to_local(self, frame: int) -> int:
        return int(frame) - int(self.start)

    @property
    def frame_count(self) -> int:
        return int(self.stop) - int(self.start)


def parquet_frame_count(path: Path) -> int | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        metadata = pf.schema_arrow.metadata or {}
        raw_count = metadata.get(b"num_frames")
        if raw_count is not None:
            try:
                value = int(raw_count.decode("utf-8") if isinstance(raw_count, bytes) else raw_count)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                return int(value)
        names = pf.schema.names
        if "Frame" not in names:
            return None
        idx = names.index("Frame")
        max_frame = None
        for row_group in range(pf.num_row_groups):
            stats = pf.metadata.row_group(row_group).column(idx).statistics
            if stats is None or stats.max is None:
                continue
            value = int(stats.max)
            max_frame = value if max_frame is None else max(max_frame, value)
        if max_frame is not None:
            return int(max_frame) + 1
    except Exception:
        return None
    return None


class ChunkTrackIndex:
    def __init__(
        self,
        tracks_dir: Path,
        *,
        chunk_frames: int,
        chunk_offset_mode: str = "metadata",
        sides: tuple[str, ...] = ("left", "right"),
    ) -> None:
        self.tracks_dir = Path(tracks_dir)
        self.chunk_frames = int(chunk_frames)
        self.chunk_offset_mode = str(chunk_offset_mode).lower()
        self.files: dict[tuple[int, str], Path] = {}
        for name in sorted(os.listdir(self.tracks_dir)):
            if not name.endswith(".parquet"):
                continue
            path = self.tracks_dir / name
            parsed = parse_track_file(path)
            if parsed is None:
                continue
            chunk, side = parsed
            if side not in sides:
                continue
            self.files[(chunk, side)] = path
        if not self.files:
            raise FileNotFoundError(f"No chunk track parquet files found in {self.tracks_dir}")
        self.specs = self.resolve_specs()

    @property
    def chunks(self) -> list[int]:
        return sorted({chunk for chunk, _side in self.files})

    @property
    def sides(self) -> list[str]:
        return sorted({side for _chunk, side in self.files})

    def frame_count_estimate(self) -> int:
        return int(self.specs[-1].stop)

    def resolve_specs(self) -> list[ChunkSpec]:
        specs: list[ChunkSpec] = []
        if self.chunk_offset_mode == "fixed":
            for chunk in self.chunks:
                start = int(chunk) * int(self.chunk_frames)
                specs.append(ChunkSpec(chunk=int(chunk), start=start, stop=start + int(self.chunk_frames)))
            return specs

        offset = 0
        for chunk in self.chunks:
            counts = []
            for side in self.sides:
                path = self.path_for(chunk, side)
                if path is None:
                    continue
                count = parquet_frame_count(path)
                if count is not None and count > 0:
                    counts.append(int(count))
            n_frames = max(counts) if counts else self.chunk_frames
            specs.append(ChunkSpec(chunk=int(chunk), start=int(offset), stop=int(offset + n_frames)))
            offset += int(n_frames)
        return specs

    def path_for(self, chunk: int, side: str) -> Path | None:
        return self.files.get((int(chunk), str(side)))


def open_video_reader(path: Path, frame_count: int, *, backend: str = "opencv"):
    AviH264ElementaryReader, OpenCvVideoReader, _resolve_frame_count = aruco_curation_tools()
    path = Path(path)
    backend = str(backend).lower()
    if backend in {"opencv", "auto"}:
        try:
            return OpenCvVideoReader(path)
        except Exception as exc:
            if backend == "opencv":
                log(f"OpenCV video reader failed, falling back to H264 reader: {exc}")
            else:
                log(f"OpenCV video reader failed in auto mode: {exc}")
    if path.suffix.lower() == ".avi" and backend in {"opencv", "h264", "auto"}:
        try:
            return AviH264ElementaryReader(path, frame_count=int(frame_count))
        except Exception as exc:
            log(f"H264 AVI reader failed, falling back to OpenCV: {exc}")
    return OpenCvVideoReader(path)


def looks_like_chunk_tracks_dir(path: Path) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False
    return any(path.glob("*_chunk*_left.parquet")) or any(path.glob("*_chunk*_right.parquet"))


def infer_block_root(video: Path | None, tracks_dir: Path | None, video_dir: Path | None) -> Path:
    if tracks_dir is not None:
        tracks_dir = Path(tracks_dir).resolve()
        if tracks_dir.name == "tracks":
            return tracks_dir.parent
        return tracks_dir
    if video is not None:
        video_parent = Path(video).resolve().parent
        if looks_like_chunk_tracks_dir(video_parent / "tracks"):
            return video_parent
        if looks_like_chunk_tracks_dir(video_parent.parent / "tracks"):
            return video_parent.parent
        return video_parent
    if video_dir is not None:
        video_dir = Path(video_dir).resolve()
        if looks_like_chunk_tracks_dir(video_dir / "tracks"):
            return video_dir
        if looks_like_chunk_tracks_dir(video_dir.parent / "tracks"):
            return video_dir.parent
        return video_dir
    return Path.cwd()


def infer_tracks_dir(video: Path | None, video_dir: Path | None, block_root: Path) -> Path:
    candidates: list[Path] = []
    bases = [
        Path(video).resolve().parent if video is not None else None,
        Path(video_dir).resolve() if video_dir is not None else None,
    ]
    for base in bases:
        if base is None:
            continue
        candidates.append(base / "tracks")
        candidates.append(base.parent / "tracks")
    candidates.append(Path(block_root).resolve() / "tracks")

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if looks_like_chunk_tracks_dir(candidate):
            return candidate
    for candidate in unique_candidates:
        if candidate.is_dir():
            return candidate
    return unique_candidates[0]


def choose_video(video: Path | None, video_dir: Path, cameras: list[int] | None) -> Path:
    if video is not None:
        return Path(video)
    _apply_homography_points, discover_media, _load_homography_stack, _parse_camera_index = multicam_tracking_tools()
    media = discover_media(video_dir, cameras)
    if not media:
        raise FileNotFoundError(f"No camera videos found in {video_dir}")
    return sorted(media)[0]


@dataclass(frozen=True)
class TrackCropSpec:
    side: str
    track_id: int
    centers_x: np.ndarray
    centers_y: np.ndarray
    observed: np.ndarray
    coverage: float
    output_path: Path


def parse_time_seconds(raw: str | float | int | None) -> float:
    if raw is None:
        return 0.0
    text = str(raw).strip()
    if not text:
        return 0.0
    if ":" not in text:
        return float(text)
    parts = [float(part) for part in text.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60.0 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600.0 + minutes * 60.0 + seconds
    raise ValueError(f"Could not parse time: {raw!r}")


def parse_int_csv(raw: str | None) -> set[int] | None:
    if raw is None or not str(raw).strip():
        return None
    out: set[int] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            step = 1 if hi >= lo else -1
            out.update(range(lo, hi + step, step))
        else:
            out.add(int(part))
    return out


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "NA"


def read_window_points(
    track_index: ChunkTrackIndex,
    *,
    frame_start: int,
    frame_stop: int,
    sides: set[str],
) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    tables = []
    for spec in track_index.specs:
        overlap_start = max(int(frame_start), int(spec.start))
        overlap_stop = min(int(frame_stop), int(spec.stop))
        if overlap_stop <= overlap_start:
            continue
        local_start = int(overlap_start - spec.start)
        local_stop = int(overlap_stop - spec.start)
        for side in track_index.sides:
            if side not in sides:
                continue
            path = track_index.path_for(spec.chunk, side)
            if path is None:
                continue
            filt = (
                (pc.field("Frame") >= int(local_start))
                & (pc.field("Frame") < int(local_stop))
                & (pc.field("Bodypoint") == 0)
            )
            table = ds.dataset(path, format="parquet").to_table(
                columns=["Frame", "TrackID", "Bodypoint", "TrackX", "TrackY"],
                filter=filt,
            )
            df = table.to_pandas()
            if df.empty:
                continue
            df["side"] = side
            df["chunk"] = int(spec.chunk)
            df["global_frame"] = pd.to_numeric(df["Frame"], errors="coerce").astype(np.float64) + int(spec.start)
            tables.append(df)

    if not tables:
        return pd.DataFrame(columns=["global_frame", "side", "TrackID", "TrackX", "TrackY"])

    out = pd.concat(tables, ignore_index=True)
    for col in ["global_frame", "TrackID", "TrackX", "TrackY"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["global_frame", "TrackID", "TrackX", "TrackY"]).copy()
    out["global_frame"] = out["global_frame"].round().astype(np.int64)
    out["TrackID"] = out["TrackID"].round().astype(np.int64)
    out = (
        out.groupby(["side", "TrackID", "global_frame"], sort=True, as_index=False)
        .agg(TrackX=("TrackX", "mean"), TrackY=("TrackY", "mean"))
    )
    return out


def fill_short_gaps(values: np.ndarray, observed: np.ndarray, *, max_gap_frames: int) -> np.ndarray:
    filled = np.full(values.shape, np.nan, dtype=np.float32)
    obs_idx = np.flatnonzero(observed)
    if len(obs_idx) == 0:
        return filled
    filled[obs_idx] = values[obs_idx]
    if len(obs_idx) == 1:
        return filled
    for left, right in zip(obs_idx[:-1], obs_idx[1:]):
        gap = int(right - left - 1)
        if gap <= 0:
            continue
        if max_gap_frames >= 0 and gap > int(max_gap_frames):
            continue
        xs = np.arange(left, right + 1)
        filled[left : right + 1] = np.interp(xs, [left, right], [values[left], values[right]]).astype(np.float32)
    return filled


def build_track_specs(
    points: pd.DataFrame,
    *,
    inv_h: np.ndarray,
    video_width: int,
    video_height: int,
    frame_start: int,
    frame_stop: int,
    out_dir: Path,
    video_stem: str,
    crop_size: int,
    min_coverage: float,
    max_gap_frames: int,
    track_ids: set[int] | None,
    max_tracks: int,
) -> list[TrackCropSpec]:
    n_frames = int(frame_stop - frame_start)
    if points.empty or n_frames <= 0:
        return []
    if track_ids is not None:
        points = points[points["TrackID"].isin(track_ids)].copy()
        if points.empty:
            return []

    xy = points[["TrackX", "TrackY"]].to_numpy(np.float64)
    projected = project_xy(xy, inv_h)
    points = points.copy()
    points["CamX"] = projected[:, 0]
    points["CamY"] = projected[:, 1]
    visible = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= 0)
        & (projected[:, 1] >= 0)
        & (projected[:, 0] < float(video_width))
        & (projected[:, 1] < float(video_height))
    )
    points = points.loc[visible].copy()
    if points.empty:
        return []

    specs: list[TrackCropSpec] = []
    for (side, track_id), group in points.groupby(["side", "TrackID"], sort=True):
        idx = (group["global_frame"].to_numpy(np.int64) - int(frame_start)).astype(np.int64)
        valid_idx = (idx >= 0) & (idx < n_frames)
        if not np.any(valid_idx):
            continue
        idx = idx[valid_idx]
        group = group.iloc[np.flatnonzero(valid_idx)]

        observed = np.zeros(n_frames, dtype=bool)
        xs = np.full(n_frames, np.nan, dtype=np.float32)
        ys = np.full(n_frames, np.nan, dtype=np.float32)
        xs[idx] = group["CamX"].to_numpy(np.float32)
        ys[idx] = group["CamY"].to_numpy(np.float32)
        observed[idx] = True
        coverage = float(observed.mean())
        if coverage < float(min_coverage):
            continue

        centers_x = fill_short_gaps(xs, observed, max_gap_frames=max_gap_frames)
        centers_y = fill_short_gaps(ys, observed, max_gap_frames=max_gap_frames)
        name = (
            f"{safe_name(video_stem)}_frame{int(frame_start):07d}_"
            f"{side}_track{int(track_id):04d}_{int(crop_size)}px.mp4"
        )
        specs.append(
            TrackCropSpec(
                side=str(side),
                track_id=int(track_id),
                centers_x=centers_x,
                centers_y=centers_y,
                observed=observed,
                coverage=coverage,
                output_path=out_dir / name,
            )
        )

    specs.sort(key=lambda spec: (-spec.coverage, spec.side, spec.track_id))
    if max_tracks > 0:
        specs = specs[: int(max_tracks)]
    return specs


def crop_centered(frame: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    size = int(size)
    half = size / 2.0
    x0 = int(round(float(cx) - half))
    y0 = int(round(float(cy) - half))
    x1 = x0 + size
    y1 = y0 + size

    out = np.zeros((size, size, 3), dtype=np.uint8)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(frame.shape[1], x1)
    src_y1 = min(frame.shape[0], y1)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = frame[
        src_y0:src_y1, src_x0:src_x1
    ]
    return out


def make_writer(path: Path, fps: float, crop_size: int, codec: str) -> cv2.VideoWriter:
    cv2 = require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(crop_size), int(crop_size)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def read_exact(stream, n_bytes: int) -> bytes:
    chunks = []
    remaining = int(n_bytes)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def export_h264_stream_batch(
    *,
    reader: AviH264ElementaryReader,
    frame_start: int,
    frame_stop: int,
    specs: list[TrackCropSpec],
    writers: list[cv2.VideoWriter],
    crop_size: int,
) -> None:
    n_frames = int(frame_stop - frame_start)
    batch_stop = int(frame_stop) - 1
    start_packet = reader._decode_start_packet(int(frame_start))
    end_packet = min(reader.frame_count() - 1, batch_stop + int(reader.packet_margin))
    packets = reader._read_packet_window(start_packet, end_packet)
    relative_start = int(frame_start) - int(start_packet)
    relative_stop = int(batch_stop) - int(start_packet)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "h264",
        "-i",
        "pipe:0",
        "-vf",
        f"select=between(n\\,{relative_start}\\,{relative_stop})",
        "-frames:v",
        str(n_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    def feed_packets() -> None:
        try:
            assert proc.stdin is not None
            proc.stdin.write(packets)
        except BrokenPipeError:
            pass
        finally:
            try:
                assert proc.stdin is not None
                proc.stdin.close()
            except Exception:
                pass

    feeder = threading.Thread(target=feed_packets, daemon=True)
    feeder.start()

    assert proc.stdout is not None
    blank = np.zeros((int(crop_size), int(crop_size), 3), dtype=np.uint8)
    try:
        for offset in range(n_frames):
            raw = read_exact(proc.stdout, reader.frame_bytes)
            if len(raw) != reader.frame_bytes:
                raise RuntimeError(
                    f"ffmpeg produced {offset:,}/{n_frames:,} frames for H264 window "
                    f"{frame_start}-{frame_stop - 1}"
                )
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((reader.height, reader.width, 3))
            for spec, writer in zip(specs, writers):
                cx = spec.centers_x[offset]
                cy = spec.centers_y[offset]
                if np.isfinite(cx) and np.isfinite(cy):
                    writer.write(crop_centered(frame, float(cx), float(cy), int(crop_size)))
                else:
                    writer.write(blank)
            if (offset + 1) % 1000 == 0 or offset + 1 == n_frames:
                log(f"  wrote {offset + 1:,}/{n_frames:,} frames for current batch")
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        feeder.join(timeout=5)
    returncode = proc.wait(timeout=30)
    if returncode != 0:
        stderr = b""
        if proc.stderr is not None:
            try:
                stderr = proc.stderr.read()
            except Exception:
                stderr = b""
        raise RuntimeError(f"ffmpeg H264 stream failed with code {returncode}: {stderr.decode(errors='replace')}")


def export_batch(
    *,
    video_path: Path,
    backend: str,
    estimated_frame_count: int,
    frame_start: int,
    frame_stop: int,
    specs: list[TrackCropSpec],
    fps: float,
    crop_size: int,
    codec: str,
) -> None:
    AviH264ElementaryReader, _OpenCvVideoReader, _resolve_frame_count = aruco_curation_tools()
    reader = open_video_reader(video_path, estimated_frame_count, backend=backend)
    writers: list[cv2.VideoWriter] = []
    try:
        writers = [make_writer(spec.output_path, fps=fps, crop_size=crop_size, codec=codec) for spec in specs]
        n_frames = int(frame_stop - frame_start)
        blank = np.zeros((int(crop_size), int(crop_size), 3), dtype=np.uint8)
        if isinstance(reader, AviH264ElementaryReader):
            export_h264_stream_batch(
                reader=reader,
                frame_start=int(frame_start),
                frame_stop=int(frame_stop),
                specs=specs,
                writers=writers,
                crop_size=int(crop_size),
            )
            return
        for offset, frame_idx in enumerate(range(int(frame_start), int(frame_stop))):
            frame = reader.read_frame(frame_idx)
            for spec, writer in zip(specs, writers):
                cx = spec.centers_x[offset]
                cy = spec.centers_y[offset]
                if np.isfinite(cx) and np.isfinite(cy):
                    writer.write(crop_centered(frame, float(cx), float(cy), int(crop_size)))
                else:
                    writer.write(blank)
            if (offset + 1) % 1000 == 0 or offset + 1 == n_frames:
                log(f"  wrote {offset + 1:,}/{n_frames:,} frames for current batch")
    finally:
        for writer in writers:
            writer.release()
        try:
            reader.close()
        except Exception:
            pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=None, help="Camera video to crop. If omitted, choose from --video_dir.")
    parser.add_argument("--video_dir", type=Path, default=None, help="Block folder containing camNN AVI files.")
    parser.add_argument("--camera", type=int, default=None, help="One-based camera number to choose from --video_dir.")
    parser.add_argument("--tracks_dir", type=Path, default=None, help="Chunk track parquet folder. Default: infer from video block.")
    parser.add_argument("--hmats", type=Path, default=DEFAULT_HMATS, help="Homography .npz with key H.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output folder. Default: block/stitched/sleep_crop_videos/<window>.")
    parser.add_argument("--start_time", default="0", help="Window start as seconds, MM:SS, or HH:MM:SS.")
    parser.add_argument("--start_frame", type=int, default=None, help="Window start frame. Overrides --start_time.")
    parser.add_argument("--duration_min", type=float, default=10.0, help="Window duration in minutes.")
    parser.add_argument("--crop_size_px", type=int, default=480, help="Square crop size.")
    parser.add_argument("--video_backend", choices=["opencv", "h264", "auto"], default="opencv")
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--track_ids", default=None, help="Comma/range list such as 12,15,20-30. Default: all visible tracks.")
    parser.add_argument("--min_coverage", type=float, default=0.25, help="Minimum visible fraction in the window.")
    parser.add_argument("--max_tracks", type=int, default=0, help="Limit exported tracks after sorting by coverage. 0 means no limit.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of crop videos to write per video pass.")
    parser.add_argument("--max_gap_frames", type=int, default=120, help="Interpolate track-center gaps up to this many frames.")
    parser.add_argument("--codec", default="mp4v", help="OpenCV fourcc for output videos.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    total_start = time.perf_counter()

    block_root = infer_block_root(args.video, args.tracks_dir, args.video_dir)
    video_dir = Path(args.video_dir) if args.video_dir is not None else block_root
    camera_list = [int(args.camera)] if args.camera is not None else None
    video_path = choose_video(args.video, video_dir, camera_list)
    tracks_dir = Path(args.tracks_dir) if args.tracks_dir is not None else infer_tracks_dir(video_path, video_dir, block_root)

    log(f"video={video_path}")
    log(f"tracks_dir={tracks_dir}")
    track_index = ChunkTrackIndex(tracks_dir, chunk_frames=DEFAULT_CHUNK_FRAMES, chunk_offset_mode="metadata")
    estimated_frame_count = int(track_index.frame_count_estimate())

    reader = open_video_reader(video_path, estimated_frame_count, backend=args.video_backend)
    try:
        _AviH264ElementaryReader, _OpenCvVideoReader, resolve_frame_count = aruco_curation_tools()
        frame_count = int(resolve_frame_count(video_path, reader) or estimated_frame_count)
        fps = float(getattr(reader, "fps", 24.0) or 24.0)
        video_width = int(getattr(reader, "width", 0)) or 1
        video_height = int(getattr(reader, "height", 0)) or 1
    finally:
        try:
            reader.close()
        except Exception:
            pass

    if args.start_frame is None:
        frame_start = int(round(parse_time_seconds(args.start_time) * fps))
    else:
        frame_start = int(args.start_frame)
    frame_start = max(0, min(frame_start, max(0, frame_count - 1)))
    duration_frames = max(1, int(round(float(args.duration_min) * 60.0 * fps)))
    frame_stop = min(frame_count, frame_start + duration_frames)
    if frame_stop <= frame_start:
        raise ValueError(f"Empty frame window: {frame_start}-{frame_stop}")

    crop_size = max(32, int(args.crop_size_px))
    if crop_size % 2:
        crop_size += 1
    window_name = f"{safe_name(video_path.stem)}_frame{frame_start:07d}_{frame_stop - frame_start:06d}frames"
    out_dir = Path(args.out_dir) if args.out_dir is not None else block_root / "stitched" / "sleep_crop_videos" / window_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sides = {"left", "right"} if args.side == "both" else {str(args.side)}
    track_ids = parse_int_csv(args.track_ids)

    _apply_homography_points, _discover_media, load_homography_stack, parse_camera_index = multicam_tracking_tools()
    cam_index = parse_camera_index(video_path)
    homographies = load_homography_stack(args.hmats)
    if cam_index < 0 or cam_index >= len(homographies):
        raise ValueError(f"Could not map {video_path.name} to homography index")
    inv_h = np.linalg.inv(homographies[cam_index])

    log(
        f"window frames {frame_start:,}-{frame_stop - 1:,} "
        f"({(frame_stop - frame_start) / fps:.1f}s at {fps:.3f} fps), crop={crop_size}px"
    )
    points = read_window_points(track_index, frame_start=frame_start, frame_stop=frame_stop, sides=sides)
    log(f"loaded {len(points):,} track center rows in window")
    specs = build_track_specs(
        points,
        inv_h=inv_h,
        video_width=video_width,
        video_height=video_height,
        frame_start=frame_start,
        frame_stop=frame_stop,
        out_dir=out_dir,
        video_stem=video_path.stem,
        crop_size=crop_size,
        min_coverage=float(args.min_coverage),
        max_gap_frames=int(args.max_gap_frames),
        track_ids=track_ids,
        max_tracks=int(args.max_tracks),
    )
    if not specs:
        raise RuntimeError("No visible tracks met the export filters")

    log(f"exporting {len(specs):,} track crop videos to {out_dir}")
    batch_size = max(1, int(args.batch_size))
    for batch_start in range(0, len(specs), batch_size):
        batch = specs[batch_start : batch_start + batch_size]
        log(f"batch {batch_start // batch_size + 1}: tracks {batch_start + 1}-{batch_start + len(batch)}")
        export_batch(
            video_path=video_path,
            backend=args.video_backend,
            estimated_frame_count=estimated_frame_count,
            frame_start=frame_start,
            frame_stop=frame_stop,
            specs=batch,
            fps=fps,
            crop_size=crop_size,
            codec=str(args.codec),
        )

    manifest = pd.DataFrame(
        [
            {
                "side": spec.side,
                "track_id": int(spec.track_id),
                "output_path": str(spec.output_path),
                "video_path": str(video_path),
                "tracks_dir": str(tracks_dir),
                "frame_start": int(frame_start),
                "frame_stop_exclusive": int(frame_stop),
                "fps": float(fps),
                "crop_size_px": int(crop_size),
                "observed_frames": int(spec.observed.sum()),
                "window_frames": int(frame_stop - frame_start),
                "coverage": float(spec.coverage),
            }
            for spec in specs
        ]
    )
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    log(f"wrote manifest: {manifest_path}")
    log(f"done in {time.perf_counter() - total_start:.1f}s")


if __name__ == "__main__":
    main()
