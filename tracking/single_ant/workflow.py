"""SLEAP-only, camera-local tracking with a recording-time coordinate system.

The manifest is the input contract: video frame counts (never detection counts),
camera-specific recording timestamps, masks, and model provenance are frozen
before inference. Outputs are streamed one chunk at a time.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import struct
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

VIDEO_RE = re.compile(r"^(cam\d+)_cam\d+_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})(?:_(\d+))?\.(avi|mkv|mp4)$")
SESSION_RE = re.compile(r"^\d{8}-\d{6}$")
REQUIRED = ("Frame", "Instance", "Bodypoint", "X", "Y", "Score_node")


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probe_video(path: Path) -> dict:
    # OpenCV scans large OpenDML AVI indexes on open (minutes per recording).
    # The video stream header and OpenDML header contain the same frame count.
    # Read the RIFF header tree only; do not mistake avih's first-RIFF count for
    # the total in a multi-RIFF (>2 GB) recording.
    if path.suffix.lower() == ".avi":
        with path.open("rb") as f:
            header = f.read(64 * 1024)
        if header[:4] == b"RIFF" and header[8:12] == b"AVI ":
            fields = {}
            streams = []

            def walk(start, stop):
                pos = start
                while pos + 8 <= min(stop, len(header)):
                    kind = header[pos:pos + 4]
                    size = struct.unpack_from("<I", header, pos + 4)[0]
                    data, end = pos + 8, pos + 8 + size
                    if kind == b"LIST" and header[data:data + 4] in (b"hdrl", b"strl", b"odml"):
                        walk(data + 4, end)
                    elif end <= len(header):
                        if kind == b"avih" and size >= 40:
                            fields["width"], fields["height"] = struct.unpack_from("<II", header, data + 32)
                        elif kind == b"strh" and size >= 36 and header[data:data + 4] == b"vids":
                            scale, rate, _, length = struct.unpack_from("<IIII", header, data + 20)
                            streams.append((scale, rate, length))
                        elif kind == b"dmlh" and size >= 4:
                            fields["opendml_frames"] = struct.unpack_from("<I", header, data)[0]
                    pos = end + (size & 1)

            walk(12, len(header))
            if len(streams) == 1 and "width" in fields:
                scale, rate, length = streams[0]
                odml = fields.get("opendml_frames", 0)
                if scale > 0 and rate > 0 and length > 0 and (not odml or odml == length):
                    return dict(frames=length, fps=rate / scale, width=fields["width"], height=fields["height"])
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        result = dict(frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                      fps=float(cap.get(cv2.CAP_PROP_FPS)),
                      width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if min(result.values()) <= 0:
            raise ValueError(f"Invalid video metadata: {path}: {result}")
        return result
    finally:
        cap.release()


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != shape:
        raise ValueError(f"Mask missing or wrong dimensions: {path}; expected {shape}")
    mask = np.any(image != 0, axis=2)
    if not mask.any():
        raise ValueError(f"Empty arena mask: {path}")
    return mask


@dataclass(frozen=True)
class Settings:
    anchor_bodypoint: int = 1
    min_score: float = 0.2
    min_nodes: int = 2
    max_distance: float = 100.0
    reacquire_seconds: float = 5.0

    def __post_init__(self):
        if not 0 <= self.min_score <= 1 or self.min_nodes < 1:
            raise ValueError("Invalid confidence/node threshold")
        if self.max_distance <= 0 or self.reacquire_seconds <= 0:
            raise ValueError("Distance and reacquisition time must be positive")


def prepare(root: Path, output: Path, *, model_centroid: Path, model_instance: Path,
            export_dir: Path, timezone: str = "Asia/Tokyo", module: str = "sleap-nn/0.3.3",
            settings: Settings = Settings(), export_metadata: Path | None = None,
            build_trt_if_missing: bool = False) -> dict:
    """Discover existing video chunks; verify their counts against the sidecars.

    A missing intermediate chunk is an error, as its duration cannot be inferred
    safely. A camera absent for a whole recording is recorded in the manifest.
    Global overview videos are excluded: they have no individual-ant arena mask.
    """
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Run already prepared: {output}; resume it or use a new output directory")
    recordings, cameras, excluded = [], set(), []
    for session in sorted(root.iterdir()):
        if not session.is_dir() or not SESSION_RE.fullmatch(session.name):
            continue
        raw_videos = [p for p in sorted(session.iterdir()) if VIDEO_RE.fullmatch(p.name)]
        excluded.extend(str(p.relative_to(root)) for p in session.glob("global*.avi"))
        if not raw_videos:
            raise ValueError(f"No camera videos in {session}")
        seen = set()
        for raw in raw_videos:
            match = VIDEO_RE.fullmatch(raw.name)
            camera, stamp, chunk, _ = match.groups()
            if chunk is not None:
                continue
            if camera in seen:
                raise ValueError(f"Multiple recordings for {session.name}/{camera}")
            seen.add(camera)
            cameras.add(camera)
            masks = sorted((root / "arena_seg").glob(f"{camera}_*.png"))
            if len(masks) != 1:
                raise ValueError(f"Expected exactly one arena mask for {camera}, found {masks}")
            meta = probe_video(raw)
            read_mask(masks[0], (meta["height"], meta["width"]))
            sidecar = session / "data" / f"{raw.stem}_frame_counts.csv"
            recorded_counts = {}
            if sidecar.exists():
                with sidecar.open() as f:
                    for row in csv.reader(f):
                        if len(row) != 2 or row[0] in recorded_counts:
                            raise ValueError(f"Invalid/duplicate frame-count row in {sidecar}: {row}")
                        recorded_counts[row[0]] = int(row[1])
            chunks, offset = [], 0
            for video in sorted((session / "data").glob(f"{raw.stem}_*{raw.suffix}")):
                cm = VIDEO_RE.fullmatch(video.name)
                if not cm or cm[3] is None:
                    continue
                index = int(cm[3])
                if index != len(chunks):
                    raise ValueError(f"Missing/duplicate chunk before {video}")
                info = probe_video(video)
                if any(info[k] != meta[k] for k in ("fps", "width", "height")):
                    raise ValueError(f"Chunk metadata disagrees with source: {video}")
                if recorded_counts and recorded_counts.get(video.name) != info["frames"]:
                    raise ValueError(f"Video/sidecar frame counts disagree: {video}: {info}")
                chunks.append(dict(index=index, stem=video.stem, video=str(video.relative_to(root)),
                                   frames=info["frames"], offset=offset, bytes=video.stat().st_size))
                offset += info["frames"]
            if not chunks or (recorded_counts and len(recorded_counts) != len(chunks)):
                raise ValueError(f"Incomplete chunks in {session / 'data'} for {camera}")
            if offset != meta["frames"]:
                raise ValueError(f"Chunk total {offset} != source frames {meta['frames']}: {raw}")
            start = datetime.strptime(stamp, "%Y-%m-%d-%H-%M-%S").replace(tzinfo=ZoneInfo(timezone))
            recordings.append(dict(key=f"{session.name}_{camera}", session=session.name,
                                   camera=camera, track=int(camera[3:]), start_ns=int(pd.Timestamp(start).value),
                                   start_time=start.isoformat(), video=str(raw.relative_to(root)),
                                   mask=str(masks[0].relative_to(root)), mask_sha256=digest(masks[0]),
                                   chunks=chunks, **meta))
            print(f"[PLAN] {session.name}/{camera}: {len(chunks)} chunks, {offset} frames", flush=True)
    if not recordings:
        raise ValueError(f"No dated recording folders in {root}")
    fps_set = {r["fps"] for r in recordings}
    if len(fps_set) != 1:
        raise ValueError(f"A shared Frame axis requires a common FPS, found {fps_set}")
    origin_ns = min(r["start_ns"] for r in recordings)
    for camera in sorted(cameras):
        previous = None
        for r in sorted((r for r in recordings if r["camera"] == camera), key=lambda r: r["start_ns"]):
            r["frame_start"] = int(round((r["start_ns"] - origin_ns) / 1e9 * r["fps"]))
            r["end_ns"] = r["start_ns"] + int(round(r["frames"] / r["fps"] * 1e9))
            r["gap_before_seconds"] = None if previous is None else (r["start_ns"] - previous["end_ns"]) / 1e9
            if previous and r["gap_before_seconds"] < 0:
                raise ValueError(f"Overlapping recordings for {camera}: {previous['key']}, {r['key']}")
            previous = r
    model = dict(centroid=str(model_centroid), instance=str(model_instance), export_dir=str(export_dir),
                 module=module, runtime="tensorrt", batch_size=8)
    if export_metadata is not None:
        exported = json.loads(export_metadata.read_text())
        expected = f"centroid:{model_centroid / 'best.ckpt'};centered_instance:{model_instance / 'best.ckpt'}"
        if exported.get("checkpoint_path") != expected:
            raise ValueError("Export metadata does not name the requested model pair")
        model.update(centroid_peak_threshold=exported.get("centroid_peak_threshold", exported["peak_threshold"]),
                     instance_peak_threshold=exported.get("instance_peak_threshold", exported["peak_threshold"]))
        if exported.get("onnx_sha256"):
            model["onnx_sha256"] = exported["onnx_sha256"]
    if build_trt_if_missing:
        if "onnx_sha256" not in model or len({(r["height"], r["width"]) for r in recordings}) != 1:
            raise ValueError("Lazy TensorRT compilation requires validated export metadata and a shared video resolution")
        model["build_trt_if_missing"] = True
    for role, directory in (("centroid", model_centroid), ("instance", model_instance)):
        model[role + "_sha256"] = digest(directory / "best.ckpt")
    sessions = sorted({r["session"] for r in recordings})
    manifest = dict(version=1, root=str(root.resolve()), output=str(output.resolve()), model=model,
                    settings=asdict(settings), timezone=timezone, timestamp_source="per-camera video filename; 1 second resolution",
                    origin_ns=origin_ns, fps=next(iter(fps_set)), recordings=recordings,
                    cameras=sorted(cameras), excluded_overview_videos=excluded,
                    missing_cameras={s: sorted(cameras - {r['camera'] for r in recordings if r['session'] == s}) for s in sessions})
    contract = json.dumps(manifest, sort_keys=True).encode()
    manifest["contract_sha256"] = hashlib.sha256(contract).hexdigest()
    atomic_json(output / "manifest.json", manifest)
    return manifest


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def detection_path(manifest: dict, recording: dict, chunk: dict) -> Path:
    return Path(manifest.get("detections_root", manifest["output"])) / "detections" / recording["session"] / f"{chunk['stem']}_sleap_data.h5"


def read_detections(path: Path, chunk: dict, contract: str) -> pd.DataFrame:
    with h5py.File(path, "r") as h:
        if int(h.attrs.get("expected_frames", -1)) != chunk["frames"]:
            raise ValueError(f"Missing/mismatched video frame count: {path}")
        if h.attrs.get("single_ant_contract", "") != contract:
            raise ValueError(f"Detections belong to a different inference run: {path}")
        df = pd.DataFrame.from_records(h["sleap_data"][...])
    if not set(REQUIRED).issubset(df.columns):
        raise ValueError(f"Invalid SLEAP schema in {path}")
    if not df.empty:
        frames = df["Frame"].to_numpy()
        if np.any(frames < 0) or np.any(frames >= chunk["frames"]):
            raise ValueError(f"Detection frames outside video bounds: {path}")
        if df.duplicated(["Frame", "Instance", "Bodypoint"]).any():
            raise ValueError(f"Duplicate skeleton nodes: {path}")
    return df


def select_skeletons(df: pd.DataFrame, mask: np.ndarray, *, offset: int, fps: float,
                     settings: Settings, previous: tuple[int, float, float] | None = None
                     ) -> tuple[pd.DataFrame, tuple[int, float, float] | None]:
    """Choose a whole skeleton, never independent nodes from different instances.

    Score-valid nodes define candidate quality. The preferred anchor is the
    occiput (bodypoint 1), avoiding dependence on the model's tag node; a median
    of valid nodes is used when absent.
    Short gaps retain continuity; after reacquire_seconds the highest-confidence
    candidate wins. The identity remains the camera's single ant.
    """
    valid = df[np.isfinite(df.X) & np.isfinite(df.Y) & (df.Score_node >= settings.min_score)].copy()
    grouped = valid.groupby(["Frame", "Instance"], sort=True)
    candidates = grouped.agg(TrackX=("X", "median"), TrackY=("Y", "median"),
                             DetectionScore=("Score_node", "sum"), ValidNodes=("Bodypoint", "size"))
    # Missing/weak nodes contribute zero: two confident spurious points should
    # not outrank a well-supported complete ant skeleton on the first frame.
    node_count = max(1, int(df.Bodypoint.max()) + 1) if len(df) else 1
    candidates["DetectionScore"] /= node_count
    anchors = valid[valid.Bodypoint == settings.anchor_bodypoint].set_index(["Frame", "Instance"])
    candidates["AnchorBodypoint"] = np.int32(-1)
    common = candidates.index.intersection(anchors.index)
    candidates.loc[common, ["TrackX", "TrackY"]] = anchors.loc[common, ["X", "Y"]].to_numpy()
    candidates.loc[common, "AnchorBodypoint"] = settings.anchor_bodypoint
    candidates = candidates[candidates.ValidNodes >= settings.min_nodes].reset_index()
    x = np.rint(candidates.TrackX).to_numpy(dtype=np.int64)
    y = np.rint(candidates.TrackY).to_numpy(dtype=np.int64)
    inside = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
    inside[inside] &= mask[y[inside], x[inside]]
    candidates = candidates[inside]
    selected = []
    # Avoid a pandas group allocation for every video frame (millions per camera).
    values = candidates.to_numpy()
    columns = {name: i for i, name in enumerate(candidates.columns)}
    fcol, xcol, ycol, scol = (columns[k] for k in ("Frame", "TrackX", "TrackY", "DetectionScore"))
    if len(values):
        boundaries = np.r_[0, np.flatnonzero(np.diff(values[:, fcol])) + 1, len(values)]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            rows = values[lo:hi]
            frame = int(rows[0, fcol]) + offset
            costs = 1 - rows[:, scol]
            if previous is not None and frame - previous[0] <= settings.reacquire_seconds * fps:
                radius = settings.max_distance * max(1, frame - previous[0])
                distance = np.hypot(rows[:, xcol] - previous[1], rows[:, ycol] - previous[2])
                costs = costs + distance / radius
                costs[distance > radius] = np.inf
            chosen = int(np.argmin(costs))
            if not np.isfinite(costs[chosen]):
                continue
            selected.append(lo + chosen)
            previous = frame, float(rows[chosen, xcol]), float(rows[chosen, ycol])
    chosen = candidates.iloc[selected].copy()
    out = df.merge(chosen, on=["Frame", "Instance"], how="inner", validate="many_to_one")
    out["NodeValid"] = np.isfinite(out.X) & np.isfinite(out.Y) & (out.Score_node >= settings.min_score)
    # Low-confidence nodes are preserved for QC but never presented as valid XY.
    out.loc[~out.NodeValid, ["X", "Y"]] = np.nan
    return out.sort_values(["Frame", "Bodypoint"]).reset_index(drop=True), previous


def add_time(df: pd.DataFrame, recording: dict, chunk: dict, origin_ns: int) -> pd.DataFrame:
    df = df.copy()
    df["FrameLocal"] = df["Frame"].astype("int64")
    df["FrameInRecording"] = df.FrameLocal + chunk["offset"]
    df["Frame"] = df.FrameInRecording + recording["frame_start"]
    ns = recording["start_ns"] + np.rint(df.FrameInRecording.to_numpy() / recording["fps"] * 1e9).astype("int64")
    df["Timestamp"] = pd.to_datetime(ns, utc=True)
    df["ElapsedSeconds"] = (ns - origin_ns) / 1e9
    df["Track"] = np.int64(recording["track"])
    df["Camera"] = pd.Series(recording["camera"], index=df.index, dtype="string")
    df["Recording"] = pd.Series(recording["session"], index=df.index, dtype="string")
    df["Chunk"] = np.int32(chunk["index"])
    return df


def coverage_intervals(selected: np.ndarray, recording: dict, chunk: dict) -> pd.DataFrame:
    observed = np.zeros(chunk["frames"], dtype=bool)
    observed[selected.astype(int)] = True
    bounds = np.r_[0, np.flatnonzero(observed[1:] != observed[:-1]) + 1, len(observed)]
    start = bounds[:-1] + chunk["offset"]
    stop = bounds[1:] + chunk["offset"]
    return pd.DataFrame(dict(FrameStart=start + recording["frame_start"],
                             FrameStop=stop + recording["frame_start"], Observed=observed[bounds[:-1]],
                             Recording=recording["session"], Camera=recording["camera"], Chunk=chunk["index"]))


def track_recording(manifest: dict, key: str) -> dict:
    recording = next(r for r in manifest["recordings"] if r["key"] == key)
    root, output = Path(manifest["root"]), Path(manifest["output"])
    mask_path = root / recording["mask"]
    if digest(mask_path) != recording["mask_sha256"]:
        raise ValueError(f"Arena mask changed since preparation: {mask_path}")
    mask = read_mask(mask_path, (recording["height"], recording["width"]))
    settings, previous = Settings(**manifest["settings"]), None
    target = output / "tracks" / f"{key}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".parquet.tmp")
    marker = output / "state" / f"{key}.tracked.json"
    marker.unlink(missing_ok=True)
    writer, intervals, summaries = None, [], []
    try:
        for chunk in recording["chunks"]:
            path = detection_path(manifest, recording, chunk)
            detections = read_detections(path, chunk, manifest["contract_sha256"])
            selected, previous = select_skeletons(detections, mask, offset=chunk["offset"],
                                                   fps=recording["fps"], settings=settings, previous=previous)
            frames = selected.Frame.unique()
            intervals.append(coverage_intervals(frames, recording, chunk))
            selected = add_time(selected, recording, chunk, manifest["origin_ns"])
            table = pa.Table.from_pandas(selected, preserve_index=False)
            if writer is None:
                metadata = dict(table.schema.metadata or {})
                metadata[b"single_ant_contract"] = manifest["contract_sha256"].encode()
                writer = pq.ParquetWriter(tmp, table.schema.with_metadata(metadata), compression="zstd")
            writer.write_table(table)
            summaries.append(dict(chunk=chunk["index"], video_frames=chunk["frames"],
                                  detected_frames=int(detections.Frame.nunique()), selected_frames=len(frames), rows=len(selected)))
            print(f"[TRACK] {key} chunk {chunk['index']}: {len(frames)}/{chunk['frames']} frames", flush=True)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(target)
    coverage = output / "coverage" / f"{key}.parquet"
    coverage.parent.mkdir(parents=True, exist_ok=True)
    coverage_tmp = coverage.with_suffix(".parquet.tmp")
    pd.concat(intervals, ignore_index=True).to_parquet(coverage_tmp, index=False, compression="zstd")
    coverage_tmp.replace(coverage)
    summary = dict(contract_sha256=manifest["contract_sha256"], recording=key, chunks=summaries,
                   rows=sum(s["rows"] for s in summaries), selected_frames=sum(s["selected_frames"] for s in summaries),
                   video_frames=recording["frames"], output=str(target))
    atomic_json(marker, summary)
    return summary


def stitch_camera(manifest: dict, camera: str) -> dict:
    """Concatenate time-anchored recordings without closing any recording gap."""
    output = Path(manifest["output"])
    recordings = sorted((r for r in manifest["recordings"] if r["camera"] == camera), key=lambda r: r["start_ns"])
    if not recordings:
        raise ValueError(f"Unknown camera: {camera}")
    target = output / "stitched" / "per_track" / f"TrackID_{recordings[0]['track']}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp, writer, last_frame, rows = target.with_suffix(".parquet.tmp"), None, -1, 0
    marker = output / "state" / f"{camera}.stitched.json"
    marker.unlink(missing_ok=True)
    try:
        for r in recordings:
            state = load_manifest(output / "state" / f"{r['key']}.tracked.json")
            if state["contract_sha256"] != manifest["contract_sha256"]:
                raise ValueError(f"Stale tracking: {r['key']}")
            parquet = pq.ParquetFile(output / "tracks" / f"{r['key']}.parquet")
            if writer is None:
                writer = pq.ParquetWriter(tmp, parquet.schema_arrow, compression="zstd")
            for batch in parquet.iter_batches(batch_size=250_000):
                table = pa.Table.from_batches([batch])
                frames = table["Frame"].to_numpy()
                if len(frames):
                    if frames[0] < last_frame or np.any(np.diff(frames) < 0):
                        raise ValueError(f"Non-monotonic/overlapping frame axis for {camera}")
                    last_frame = frames[-1]
                writer.write_table(table)
                rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(target)
    summary = dict(contract_sha256=manifest["contract_sha256"], camera=camera, track=recordings[0]["track"],
                   recordings=[r["key"] for r in recordings], rows=rows, output=str(target))
    atomic_json(marker, summary)
    return summary


def finalize(manifest: dict) -> dict:
    output = Path(manifest["output"])
    tracks = []
    for camera in manifest["cameras"]:
        state = load_manifest(output / "state" / f"{camera}.stitched.json")
        if state["contract_sha256"] != manifest["contract_sha256"]:
            raise ValueError(f"Stale stitched result for {camera}")
        published = output / "stitched" / "per_track" / f"TrackID_{state['track']}.parquet"
        p = pq.ParquetFile(published)
        if p.metadata.num_rows != state["rows"]:
            raise ValueError(f"Wrong row count for {camera}")
        tracks.append(dict(state, output=str(published)))
    records = []
    for r in manifest["recordings"]:
        qc = load_manifest(output / "state" / f"{r['key']}.tracked.json")
        if qc["contract_sha256"] != manifest["contract_sha256"]:
            raise ValueError(f"Stale recording QC: {r['key']}")
        record = {k: v for k, v in r.items() if k != "chunks"}
        record.update(end_time=pd.Timestamp(r["end_ns"], tz="UTC").tz_convert(manifest.get("timezone", "UTC")).isoformat(),
                      duration_seconds=r["frames"] / r["fps"], selected_frames=qc["selected_frames"],
                      tracked_frame_fraction=qc["selected_frames"] / r["frames"])
        records.append(record)
    pd.DataFrame(records).to_parquet(output / "recordings.parquet", index=False)
    pd.DataFrame(records).to_csv(output / "recordings.csv", index=False)
    pd.DataFrame(tracks).to_parquet(output / "stitched" / "track_manifest.parquet", index=False)
    result = dict(contract_sha256=manifest["contract_sha256"], cameras=len(tracks),
                  recordings=len(records), chunks=sum(len(r["chunks"]) for r in manifest["recordings"]),
                  rows=sum(t["rows"] for t in tracks), missing_cameras=manifest["missing_cameras"])
    atomic_json(output / "COMPLETE.json", result)
    return result
