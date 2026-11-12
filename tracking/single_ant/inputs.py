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
            total = int(d["recorder"]["framesEncoded"])
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
                if (first["blockId"] != 0 or first["fileFrameIndex"] != 0
                        or last["blockId"] != total-1 or last["fileFrameIndex"] != total-1
                        or any(p.get(key) for key in ["gaps", "resetEvents", "invalidIdEvents", "wrapEvents"])
                        or p.get("eventOverflowCount", 0)):
                    raise ValueError(f"Cannot establish common-trigger alignment: {diag}")
            rates.add(float(d["context"]["fps"]))
            diagnostics[camera] = dict(path=str(diag), sha256=sha256(diag),
                                       first_block_id=None if interrupted else 0,
                                       last_block_id=None if interrupted else total-1, frames=total,
                                       interrupted=interrupted, telemetry_available=not interrupted)
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
    result = dict(block=str(block), fps=fps, n_frames=total, cameras=cameras, files=files,
                  chunks=[dict(index=i, start=s, stop=min(s+stride, total)) for i, s in enumerate(range(0, total, stride))],
                  board_px_per_mm=1/float(meta["mm_per_pixel"]), homographies=matrices.tolist(),
                  homography_source=str(homography), homography_sha256=sha256(homography),
                  origin=origin.isoformat(), timezone=timezone,
                  synchronization=dict(method="common hardware trigger/block IDs", diagnostics=diagnostics,
                                       alignment_assumed_cameras=sorted(interrupted_cameras),
                                       limitation=("Interrupted cameras have no recorder telemetry; common-trigger alignment is assumed, "
                                                   "and dropped frames cannot be ruled out. Their terminal ArUco-only tails are excluded."
                                                   if interrupted_cameras else None),
                                       session_log=str(log), session_sha256=sha256(log), trigger_start_line=trigger[0]),
                  annotation_source=str(regions))
    validate_timing(result)
    return result, (regions, metadata, homography)
