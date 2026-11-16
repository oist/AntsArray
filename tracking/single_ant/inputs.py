"""Discover ordinary detection exports and verify their common-trigger timeline."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import h5py
import numpy as np

from camera_cal.region_paths import panorama_regions_path
from tracking.colony.combine_blocks import fingerprint
from tracking.single_ant.manual_cup_tracking import load_cups, sha256, validate_timing

ARUCO = re.compile(r"^(cam\d+)_cam\d+_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_(\d+)_aruco_tracks_?\.h5$")
REALIGN = "REALIGN.json"


def realignment(block):
    """The block's installed in-place frame-drop realignment, or None.

    Written by tracking/colony/realign_offgrid_camera.py --sidecar-gaps: a camera that lost
    trigger pulses on the way to the PC had its exports re-cut onto the pulse grid, with the
    lost pulses as empty frames. Its sidecar still records the gap, so it is accepted only
    through this record (checked against the sidecar and the exports' realign_id attrs).
    """
    path = Path(block) / REALIGN
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record.get("kind") != "in_place":
        return None     # the view-mode marker of an off-grid re-cut: no lost pulses involved
    if record.get("schema") != 1 or not record.get("id") or not record.get("installed"):
        raise ValueError(f"Not an installed in-place realignment record: {path}")
    return dict(record, path=str(path), sha256=sha256(path))


def lost_pulse_spans(gaps):
    """Pulse spans [lo, hi) that a camera's sidecar gaps leave without a frame."""
    spans, lost = [], 0
    for g in sorted(gaps, key=lambda g: int(g["fileFrameIndex"])):
        start = int(g["fileFrameIndex"]) + lost
        lost += int(g["blocksMissing"])
        spans.append([start, start + int(g["blocksMissing"])])
    return spans


def realigned_entry(record, prefix, diag, diag_sha256, gaps):
    """The record's entry for a camera whose sidecar shows lost pulses; refuse any mismatch."""
    entry = (record or {}).get("videos", {}).get(prefix)
    if (entry is None or entry.get("sidecar_gaps") != gaps or entry.get("sidecar_sha256") != diag_sha256
            or entry.get("unrecorded_frames") != lost_pulse_spans(gaps)):
        raise ValueError(f"Lost trigger pulses need a matching installed realignment record ({REALIGN}): {diag}")
    return entry


def check_realigned_exports(record, realigned, tags, files, total):
    """Exactly the chunks from the first lost pulse on were re-cut, and each still carries the
    record's id (a later re-detection would not)."""
    if int(record.get("total_frames", -1)) != total:
        raise ValueError(f"{REALIGN} describes {record.get('total_frames')} frames, the block has {total}")
    for camera, (prefix, entry) in realigned.items():
        first = entry["unrecorded_frames"][0][0]
        camera_files = [f for f in files if f["camera"] == camera]
        expected = {f["index"] for f in camera_files if f["frame_start"] + f["frames"] > first}
        listed = {int(c["chunk"]) for c in entry["chunks"]}
        if listed != expected:
            raise ValueError(f"{REALIGN} re-cut {camera} chunks {sorted(listed)}, "
                             f"but pulses are lost from chunk {min(expected):03d} on")
        for f in camera_files:
            if tags[camera, f["index"]] != ((record["id"],) * 2 if f["index"] in listed else (None, None)):
                raise ValueError(f"Exports differ from realignment {record['id']} ({REALIGN}): "
                                 f"{camera} chunk {f['index']:03d}")
    stray = set(record["videos"]) - {prefix for prefix, _ in realigned.values()}
    if stray:
        raise ValueError(f"{REALIGN} lists cameras without lost pulses: {sorted(stray)}")


def annotations(block, regions=None, homography=None):
    """Use the colony annotation fallback: local, same date, latest earlier date."""
    regions = Path(regions) if regions else panorama_regions_path(Path(block))
    load_cups(regions)  # Refuse a colony annotation masquerading as cup labels.
    metadata = regions.with_name("panorama_from_hmats_metadata.json")
    meta = json.loads(metadata.read_text())
    h = Path(homography or meta["homographies"])
    if not h.is_file() and "cameraArray_calib" in h.parts:
        # Metadata can have been saved through a workstation's bucket mount.
        h = Path(block).parents[2].joinpath(*h.parts[h.parts.index("cameraArray_calib"):])
    if not h.is_file():
        raise FileNotFoundError(f"Calibration unavailable; specify --homography: {h}")
    return regions, metadata, h


def detection_manifest(block, regions=None, homography=None, timezone="Asia/Tokyo", interrupted_cameras=()):
    block = Path(block)
    interrupted_cameras = set(interrupted_cameras)
    regions, metadata, homography = annotations(block, regions, homography)
    matrices = np.load(homography)["H"]
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or not np.isfinite(matrices).all():
        raise ValueError("Invalid camera homographies")
    files, cameras, diagnostics, rates = [], {}, {}, set()
    record, realigned, tags = realignment(block), {}, {}
    for path in sorted((block / "data").glob("cam*_aruco_tracks*.h5")):
        match = ARUCO.fullmatch(path.name)
        if not match:
            continue
        camera, index = match[1], int(match[2])
        prefix = path.name[:match.start(2)-1]
        sleap = path.with_name(f"{prefix}_{index:03d}_sleap_data.h5")
        if not sleap.is_file():
            raise FileNotFoundError(f"Missing SLEAP export: {sleap}")
        with h5py.File(path) as a, h5py.File(sleap) as s:
            if "aruco_detections" not in a or not {"Frame", "Instance", "X", "Y"} <= set(a["aruco_detections"].dtype.names or ()):
                raise ValueError(f"Lossless ArUco records required: {path}")
            aruco_count = int(a.attrs["num_frames"])
            count = int(s.attrs["frame_count"])
            if (count <= 0 or aruco_count < count
                    or (aruco_count != count and camera not in interrupted_cameras)
                    or int(s.attrs.get("expected_frames", count)) < count):
                raise ValueError(f"Incomplete/mismatched detection frame spans: {path}")
            if not {"Frame", "Instance", "Bodypoint", "X", "Y", "Score_node"} <= set(s["sleap_data"].dtype.names or ()):
                raise ValueError(f"Invalid SLEAP export schema: {sleap}")
            tags[camera, index] = tuple(None if h.attrs.get("realign_id") is None else str(h.attrs["realign_id"])
                                        for h in (a, s))
        info = cameras.setdefault(camera, dict(chunks=[], offset=0, frames=0))
        if index != len(info["chunks"]):
            raise ValueError(f"Missing or duplicate chunk before {path}")
        start = info["frames"]
        files.append(dict(camera=camera, camera_id=int(camera[3:])-1, index=index,
                          frame_start=start, local_start=start, frames=count,
                          aruco=str(path), sleap=str(sleap),
                          aruco_frames=aruco_count,
                          fingerprints=[fingerprint(path), fingerprint(sleap)]))
        info["chunks"].append(count)
        info["frames"] += count
        if index == 0:
            diag = block / (prefix + ".mkv.diag.json")
            d = json.loads(diag.read_text())
            diag_sha256 = sha256(diag)
            total = int(d["recorder"]["framesEncoded"])
            recorded, entry = total, None
            interrupted = camera in interrupted_cameras
            if interrupted:
                if (not d.get("synthesized") or "provenance" in d
                        or d["context"].get("status") != "interrupted"):
                    raise ValueError(f"Expected an explicitly acknowledged interrupted sidecar: {diag}")
            else:
                if "provenance" not in d:
                    raise ValueError(f"Missing recorder telemetry; explicitly acknowledge interrupted cameras: {diag}")
                p = d["provenance"]
                first, last = p["firstEncodedFrame"], p["lastEncodedFrame"]
                if p.get("gaps"):
                    # Lost trigger pulses: the exports were re-cut onto the pulse grid.
                    entry = realigned_entry(record, prefix, diag, diag_sha256, p["gaps"])
                    realigned[camera] = (prefix, entry)
                    total += sum(int(g["blocksMissing"]) for g in p["gaps"])
                if (first["blockId"] != 0 or first["fileFrameIndex"] != 0
                        or last["blockId"] != total-1 or last["fileFrameIndex"] != recorded-1
                        or any(p.get(key) for key in ["resetEvents", "invalidIdEvents", "wrapEvents"])
                        or p.get("eventOverflowCount", 0)):
                    raise ValueError(f"Cannot establish common-trigger alignment: {diag}")
            rates.add(float(d["context"]["fps"]))
            diagnostics[camera] = dict(path=str(diag), sha256=diag_sha256,
                                       first_block_id=None if interrupted else 0,
                                       last_block_id=None if interrupted else total-1, frames=total,
                                       interrupted=interrupted, telemetry_available=not interrupted)
            if entry is not None:
                diagnostics[camera].update(realigned=True, recorded_frames=recorded,
                                           unrecorded_frames=entry["unrecorded_frames"])
    expected = {f"cam{i:02d}" for i in range(1, len(matrices)+1)}
    if set(cameras) != expected or not interrupted_cameras <= expected or len(rates) != 1:
        raise ValueError(f"Incomplete camera roster or incompatible FPS: {block}")
    for camera, info in cameras.items():
        if info["frames"] != diagnostics[camera]["frames"]:
            raise ValueError(f"Incomplete detection recording: {camera} in {block}")
        camera_files = [f for f in files if f["camera"] == camera]
        if any(f["aruco_frames"] != f["frames"] for f in camera_files[:-1]):
            raise ValueError(f"Only an interrupted terminal chunk may have an ArUco-only tail: {camera}")
    lengths = {v["frames"] for v in cameras.values()}
    total = max(lengths)
    if any(v["frames"] != total and camera not in interrupted_cameras for camera, v in cameras.items()):
        raise ValueError("Common-trigger cameras have different frame counts")
    if record is not None:
        check_realigned_exports(record, realigned, tags, files, total)
    log = next(block.glob("sess_*.txt"))
    trigger = [line for line in log.read_text().splitlines() if "\tTX\tstart " in line]
    if len(trigger) != 1:
        raise ValueError(f"Expected exactly one hardware-trigger start: {log}")
    origin = datetime.fromisoformat(trigger[0].split("\t")[0]).replace(tzinfo=ZoneInfo(timezone))
    fps = rates.pop()
    timing = block / "block_combination_timing.json"
    if timing.is_file():
        override = json.loads(timing.read_text())
        if float(override.get("fps", fps)) != fps:
            raise ValueError(f"Timing override has incompatible FPS: {timing}")
        origin = datetime.fromisoformat(override["start_datetime"])
        if origin.tzinfo is None:
            origin = origin.replace(tzinfo=ZoneInfo(timezone))
    stride = int(round(fps * 1800))
    meta = json.loads(metadata.read_text())
    limitations = []
    if interrupted_cameras:
        limitations.append("Interrupted cameras have no recorder telemetry; common-trigger alignment is assumed, "
                           "and dropped frames cannot be ruled out. Their terminal ArUco-only tails are excluded.")
    if realigned:
        limitations.append("Realigned cameras lost trigger pulses on the way to the PC; their exports were re-cut "
                           "onto the pulse grid and the lost pulses are empty frames (unrecorded_frames). "
                           "Their raw video and .slp frame numbers follow video_frame_map, not the pulse.")
    realignment_info = None if not realigned else dict(
        path=record["path"], sha256=record["sha256"], id=record["id"],
        unrecorded_frames={camera: entry["unrecorded_frames"] for camera, (_, entry) in sorted(realigned.items())},
        video_frame_map={camera: entry["video_frame_map"]["entries"]
                         for camera, (_, entry) in sorted(realigned.items())})
    result = dict(block=str(block), fps=fps, n_frames=total, cameras=cameras, files=files,
                  chunks=[dict(index=i, start=s, stop=min(s+stride, total)) for i, s in enumerate(range(0, total, stride))],
                  board_px_per_mm=1/float(meta["mm_per_pixel"]), homographies=matrices.tolist(),
                  homography_source=str(homography), homography_sha256=sha256(homography),
                  origin=origin.isoformat(), timezone=timezone,
                  synchronization=dict(method="common hardware trigger/block IDs", diagnostics=diagnostics,
                                       alignment_assumed_cameras=sorted(interrupted_cameras),
                                       realigned_cameras=sorted(realigned), realignment=realignment_info,
                                       limitation=" ".join(limitations) or None,
                                       session_log=str(log), session_sha256=sha256(log), trigger_start_line=trigger[0]),
                  annotation_source=str(regions))
    validate_timing(result)
    return result, (regions, metadata, homography)
