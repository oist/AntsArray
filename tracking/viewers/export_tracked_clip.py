#!/usr/bin/env python3
"""Export camera video with finished SLEAP poses, ArUco track IDs and a scale bar."""

from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
import cv2
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

BODY_EDGES = ((0, 1), (0, 2), (2, 3))
# The antennae branch from the occiput (1), not the tag center (0).
ANTENNA_EDGES = ((1, 4), (4, 5), (5, 6), (1, 7), (7, 8), (8, 9))
BODY_BGR = (240, 205, 25)
ANTENNA_BGR = (0, 155, 255)


def project(points, h):
    shape = points.shape
    flat = np.asarray(points, float).reshape(-1, 2)
    mapped = np.column_stack([flat, np.ones(len(flat))]) @ h.T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = mapped[:, :2] / mapped[:, 2:3]
    return out.reshape(shape)


def scale_bar(h, origin, length_mm, mm_per_pixel):
    origin = np.asarray(origin, float)
    a, b = project(np.array([origin, origin + [1, 0]]), h)
    v = (b - a) / np.linalg.norm(b - a)
    end = project((a + v * length_mm / mm_per_pixel)[None], np.linalg.inv(h))[0]
    if not np.isfinite(end).all() or end[0] <= origin[0]:
        raise ValueError("Invalid scale-bar projection")
    if not np.isclose(
        np.linalg.norm(project(np.array([origin, end]), h)[1] - a) * mm_per_pixel,
        length_mm,
        atol=1e-8,
    ):
        raise ValueError("Scale calibration failed")
    return np.array([origin, end])


def load_tracks(block, start, stop, stride):
    pieces = []
    sources = []
    for chunk in range(start // stride, (stop - 1) // stride + 1):
        files = sorted((block / "tracks").glob(f"*_chunk{chunk:03d}_*.parquet"))
        if len(files) != 2:
            raise ValueError(f"Expected both colony track files for chunk {chunk}")
        for path in files:
            offset = chunk * stride
            predicate = (ds.field("Frame") >= start - offset) & (
                ds.field("Frame") < stop - offset
            )
            columns = [
                "Frame",
                "TrackID",
                "Bodypoint",
                "X",
                "Y",
                "TrackX",
                "TrackY",
                "ArucoX",
                "ArucoY",
                "CameraID",
                "ArucoCam",
                "SleapCam",
            ]
            part = (
                ds.dataset(path, format="parquet")
                .to_table(columns=columns, filter=predicate)
                .to_pandas()
            )
            part["Frame"] += offset
            part["side"] = path.stem.rsplit("_", 1)[1]
            pieces.append(part)
            stat = path.stat()
            sources.append(
                dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            )
    table = pd.concat(pieces, ignore_index=True)
    if table.duplicated(["Frame", "side", "TrackID", "Bodypoint"]).any():
        raise ValueError("Duplicate finished poses")
    return table, sources


def annotate(frame, rows, inverse, h, camera_index, width, scale_mm, bar_mm):
    native_h, native_w = frame.shape[:2]
    height = int(round(native_h * width / native_w / 2)) * 2
    sx = width / native_w
    sy = height / native_h
    image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    poses = []
    identities = []
    for (side, track_id), part in rows.groupby(["side", "TrackID"], sort=True):
        first = part.iloc[0]
        xy = np.full((10, 2), np.nan)
        keep = part[(part.SleapCam == camera_index) & part.Bodypoint.between(0, 9)]
        xy[keep.Bodypoint.to_numpy(int)] = keep[["X", "Y"]].to_numpy()
        xy = project(xy, inverse) * [sx, sy]
        # Pose node 0 is the tag center. Use this camera's ArUco measurement when no pose is available.
        anchor = xy[0].copy()
        if not np.isfinite(anchor).all() and first.ArucoCam == camera_index:
            anchor = project(np.array([[first.ArucoX, first.ArucoY]]), inverse)[0] * [
                sx,
                sy,
            ]
        if not np.isfinite(anchor).all() or not (
            0 <= anchor[0] < width and 0 <= anchor[1] < height
        ):
            continue
        valid = np.isfinite(xy).all(axis=1)
        if valid.sum() >= 2:
            poses.append((int(track_id), xy))
        identities.append((side, int(track_id), anchor))
    line = max(1, round(width / 1100))
    radius = max(2, round(width / 850))
    for track_id, xy in poses:
        nodes = {
            i: tuple(np.round(p).astype(int))
            for i, p in enumerate(xy)
            if np.isfinite(p).all()
            and -30 <= p[0] < width + 30
            and -30 <= p[1] < height + 30
        }
        for edges, color in [(BODY_EDGES, BODY_BGR), (ANTENNA_EDGES, ANTENNA_BGR)]:
            for a, b in edges:
                if a in nodes and b in nodes:
                    cv2.line(image, nodes[a], nodes[b], color, line, cv2.LINE_AA)
        for node, p in nodes.items():
            cv2.circle(
                image, p, radius, BODY_BGR if node < 4 else ANTENNA_BGR, -1, cv2.LINE_AA
            )
    font = max(0.55, width / 2800)
    label_boxes = []
    for side, track_id, anchor in identities:
        text = str(track_id)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
        ax, ay = np.round(anchor).astype(int)
        options = []
        for dx, dy in [
            (14, -16),
            (14, th + 14),
            (-tw - 14, -16),
            (-tw - 14, th + 14),
            (14, -th - 28),
        ]:
            x = int(np.clip(ax + dx, 4, width - tw - 5))
            y = int(np.clip(ay + dy, th + 4, height - baseline - 5))
            box = (x - 3, y - th - 3, x + tw + 3, y + baseline + 3)
            overlap = sum(
                max(0, min(box[2], b[2]) - max(box[0], b[0]))
                * max(0, min(box[3], b[3]) - max(box[1], b[1]))
                for b in label_boxes
            )
            options.append((overlap, dx * dx + dy * dy, x, y, box))
        _, _, x, y, box = min(options)
        label_boxes.append(box)
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    # Only numeric IDs and this physical-length label are added; no title, timestamp or legend.
    bar = scale_bar(h, (60 / sx, 95 / sy), bar_mm, scale_mm) * [sx, sy]
    a, b = np.round(bar).astype(int)
    for color, thick in [((0, 0, 0), 8), ((255, 255, 255), 4)]:
        cv2.line(image, tuple(a), tuple(b), color, thick, cv2.LINE_AA)
        for p in (a, b):
            cv2.line(
                image, tuple(p - [0, 6]), tuple(p + [0, 6]), color, thick, cv2.LINE_AA
            )
    text = f"{bar_mm:g} mm"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
    origin = (int((a[0] + b[0] - tw) / 2), a[1] - 15)
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image, dict(
        ids=len(identities),
        poses=len(poses),
        visible_ids=[f"{s}:{i}" for s, i, _ in identities],
        scale_bar_pixels=bar.tolist(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("block", "video", "homographies", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--start-frame", type=int, default=7200)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--width", type=int, default=2012)
    p.add_argument("--mm-per-pixel", type=float, default=0.016)
    p.add_argument("--scale-bar-mm", type=float, default=1)
    p.add_argument("--frames-per-chunk", type=int, default=43200)
    p.add_argument("--preview-only", action="store_true")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    match = re.match(r"cam(\d+)_", a.video.name)
    if not match:
        p.error("Video filename must start with its one-based camNN prefix")
    camera_index = int(match.group(1)) - 1
    with np.load(a.homographies) as z:
        h = z["H"][camera_index]
    inverse = np.linalg.inv(h)
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate",
                "-of",
                "json",
                str(a.video),
            ],
            text=True,
        )
    )["streams"][0]
    fps = float(Fraction(probe["avg_frame_rate"]))
    count = round(a.seconds * fps)
    if Fraction(probe["r_frame_rate"]) != Fraction(probe["avg_frame_rate"]):
        raise ValueError("This exporter requires constant-frame-rate input")
    if fps <= 0 or count <= 0 or a.start_frame < 0:
        raise ValueError("Invalid clip interval")
    if a.width % 2:
        raise ValueError("Output width must be even")
    stop = a.start_frame + (1 if a.preview_only else count)
    rows, sources = load_tracks(a.block, a.start_frame, stop, a.frames_per_chunk)
    grouped = {int(f): g for f, g in rows.groupby("Frame")}
    empty = rows.iloc[:0]
    # These long MKVs have no duration index; OpenCV cannot seek them reliably.
    # FFmpeg accurate seeking decodes/discards to the requested CFR frame time.
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "4",
            "-ss",
            f"{a.start_frame/fps:.9f}",
            "-i",
            str(a.video),
            "-an",
            "-sn",
            "-dn",
            "-frames:v",
            str(stop - a.start_frame),
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
    )
    native_width, native_height = int(probe["width"]), int(probe["height"])
    frame_bytes = native_width * native_height * 3
    records = []
    encoder = None
    for i in range(stop - a.start_frame):
        data = decoder.stdout.read(frame_bytes)
        if len(data) != frame_bytes:
            raise RuntimeError(f"Video ended at frame {a.start_frame+i}")
        frame = np.frombuffer(data, dtype=np.uint8).reshape(
            native_height, native_width, 3
        )
        image, record = annotate(
            frame,
            grouped.get(a.start_frame + i, empty),
            inverse,
            h,
            camera_index,
            a.width,
            a.mm_per_pixel,
            a.scale_bar_mm,
        )
        record["frame"] = a.start_frame + i
        records.append(record)
        if i in (0, (stop - a.start_frame) // 2, stop - a.start_frame - 1):
            cv2.imwrite(
                str(a.output.with_name(a.output.stem + f"_frame{i:03d}.jpg")),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 96],
            )
        if not a.preview_only:
            if encoder is None:
                height, width = image.shape[:2]
                temporary = a.output.with_name(a.output.stem + ".partial.mp4")
                command = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "slow",
                    "-crf",
                    "17",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-threads",
                    "4",
                    str(temporary),
                ]
                encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
            encoder.stdin.write(image.tobytes())
        if i % 48 == 0:
            print("RENDER", i, record, flush=True)
    decoder.stdout.close()
    if decoder.wait() != 0:
        raise RuntimeError("Video decoding failed")
    if encoder is not None:
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
        temporary.replace(a.output)
    ids = [x["ids"] for x in records]
    poses = [x["poses"] for x in records]
    manifest = dict(
        video=str(a.video),
        block=str(a.block),
        camera_one_based=camera_index + 1,
        start_frame=a.start_frame,
        stop_frame_exclusive=stop,
        frames=len(records),
        fps=fps,
        duration_seconds=len(records) / fps,
        output=str(a.output),
        width=a.width,
        body_rgb=list(BODY_BGR[::-1]),
        antenna_rgb=list(ANTENNA_BGR[::-1]),
        text_overlays=["ArUco track ID numbers", f"{a.scale_bar_mm:g} mm"],
        calibration=dict(
            path=str(a.homographies),
            sha256=hashlib.sha256(a.homographies.read_bytes()).hexdigest(),
            panorama_mm_per_pixel=a.mm_per_pixel,
            scale_bar_mm=a.scale_bar_mm,
            output_endpoints=records[0]["scale_bar_pixels"],
        ),
        pose_source="Finished assigned SLEAP poses from this camera only; no synthetic interpolation",
        track_sources=sources,
        ids_per_frame=dict(min=min(ids), median=float(np.median(ids)), max=max(ids)),
        poses_per_frame=dict(
            min=min(poses), median=float(np.median(poses)), max=max(poses)
        ),
        visible_ids=sorted(set(x for r in records for x in r["visible_ids"])),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    a.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("COMPLETE", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
