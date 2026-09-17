#!/usr/bin/env python3
"""Combine consecutive tracked blocks; run on a Deigo login node.

    python tracking/colony/combine_blocks.py /bucket/.../20260515 --wait

Uses the existing per_track_slurm_fanout.sh wrapper. Each ant is identified by
(colony side, tag ID), not its timestamped filename. Compute jobs write flash;
the login-side publisher installs the complete result in continous_stitched.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.sleep_motion_utils import track_id_from_name, side_from_name, _atomic_write_json
from analysis.compute_track_grid_occupancy import DEFAULT_GRID_SIZE_MM
from camera_cal.region_paths import panorama_regions_path, panorama_tracking_split
from tracking.stitch_tracks import parquet_num_frames, stitched_parquet_path, concatenate_shifted_parquets
from tracking.colony.block_caches import CACHE_METADATA, DENSE, cache_segments, require_compatible, combine_caches

VERSION = 1
BLOCK_RE = re.compile(r"block(\d+)$")
STAMP_RE = re.compile(r"(\d{8})[_-](\d{6})(?:_|\.)")


def fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def check_fingerprint(record):
    if fingerprint(record["path"]) != record:
        raise ValueError(f"Input changed since planning: {record['path']}")


def consecutive_groups(blocks):
    groups = []
    for block in sorted(blocks, key=lambda item: item["number"]):
        if not groups or block["number"] != groups[-1][-1]["number"] + 1:
            groups.append([])
        groups[-1].append(block)
    return groups


def recording_start(block, tracks):
    """Prefer the exact timestamp used by the chunk stitcher, not date-folder names.

    A block_combination_timing.json can supply start_datetime and fps when the
    original chunk files have been archived. Never guess the date from HHMMSS.
    """
    explicit = block / "block_combination_timing.json"
    if explicit.is_file():
        timing = json.loads(explicit.read_text())
        return datetime.fromisoformat(timing["start_datetime"]), fingerprint(explicit), timing.get("fps")
    stamps = set()
    for path in (block / "tracks").glob("*.parquet"):
        match = STAMP_RE.search(path.name)
        if match:
            stamps.add(datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S"))
    if not stamps:
        for path in tracks[:1]:
            pf = pq.ParquetFile(path)
            if "source_file" in pf.schema_arrow.names:
                batch = next(pf.iter_batches(batch_size=1, columns=["source_file"]), None)
                if batch and batch.num_rows:
                    match = STAMP_RE.search(str(batch.column(0)[0].as_py()))
                    if match:
                        stamps.add(datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S"))
    if len(stamps) != 1:
        raise ValueError(f"{block}: need one recording timestamp, found {sorted(stamps)}. "
                         "Provide block_combination_timing.json with start_datetime and fps.")
    return stamps.pop(), None, None


def discover_block(block, fps_override=None):
    paths = sorted((block / "stitched/per_track").glob("TrackID_*.parquet"))
    if not paths:
        return None
    start, timing_source, timing_fps = recording_start(block, paths)
    tracks, lengths, rates = {}, set(), set()
    ignored_tracks, fps_sources = [], []
    if timing_fps is not None:
        rates.add(float(timing_fps))
    for path in paths:
        stamp = re.search(r"_all_(\d{6})_", path.name)
        if stamp is None:
            raise ValueError(f"Missing stitched recording timestamp: {path}")
        if timing_source is None and stamp[1] != start.strftime("%H%M%S"):
            ignored_tracks.append(dict(path=str(path), reason="timestamp differs from the completed chunk recording"))
            continue
        identity = (side_from_name(path), track_id_from_name(path))
        if None in identity or identity in tracks:
            raise ValueError(f"Ambiguous/duplicate ant identity in {block}: {path.name}")
        pf = pq.ParquetFile(path)
        if not pf.metadata.num_rows or not {"Frame", "TrackID"} <= set(pf.schema_arrow.names):
            raise ValueError(f"Incomplete tracking parquet: {path}")
        length = parquet_num_frames(path)
        if length is None:
            raise ValueError(f"{path}: missing num_frames metadata; restitch this block first")
        lengths.add(length)
        raw_fps = (pf.schema_arrow.metadata or {}).get(b"fps")
        if raw_fps is not None:
            rates.add(float(raw_fps))
        caches = {}
        for kind, filename in CACHE_METADATA.items():
            directory = block / "stitched" / kind / "per_track" / path.stem
            metadata = directory / filename
            if not metadata.is_file():
                continue
            meta = json.loads(metadata.read_text())
            if "fps" in meta:
                rates.add(float(meta["fps"]))
            caches[kind] = dict(metadata=fingerprint(metadata),
                                files=[fingerprint(p) for p in sorted(directory.iterdir())
                                       if p.is_file() and p.suffix in {".json", ".npy", ".npz", ".parquet"}])
        tracks[identity] = dict(track=fingerprint(path), caches=caches)
    if len(ignored_tracks) > len(tracks):
        raise ValueError(f"Most tracks in {block} disagree with recording timestamp; check block_combination_timing.json")
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent recording lengths in {block}: {sorted(lengths)}")
    if fps_override is not None:
        rates.add(float(fps_override))
    if not rates:
        # Tracking-only blocks may have no analysis metadata yet. The camera
        # diagnostics record the acquisition rate without decoding any video.
        for path in sorted(block.glob("cam*.diag.json")):
            context = json.loads(path.read_text()).get("context", {})
            if context.get("fps") is not None:
                rates.add(float(context["fps"]))
                fps_sources.append(fingerprint(path))
    if len(rates) != 1 or not np.isfinite(next(iter(rates), np.nan)) or next(iter(rates), 0) <= 0:
        raise ValueError(f"Missing or incompatible frame rates in {block}: {sorted(rates)}; use --fps only when metadata is absent")
    extras = [str(path) for path in sorted((block / "stitched").iterdir())
              if path.is_dir() and path.name not in {*CACHE_METADATA, "per_track", "track_pngs"}]
    if (block / "interactions").is_dir():
        extras.append(str(block / "interactions"))
    return dict(name=block.name, path=str(block), number=int(BLOCK_RE.fullmatch(block.name)[1]),
                start_datetime=start.isoformat(), timing_source=timing_source,
                fps=rates.pop(), num_frames=lengths.pop(), tracks=tracks,
                source_only_artifacts=extras, ignored_tracks=ignored_tracks, fps_sources=fps_sources)


def shared_geometry(blocks, grid_size_mm, grid_pad_mm):
    sources, geometries = [], []
    for block in blocks:
        path = panorama_regions_path(Path(block["path"]), search_earlier_dates=True)
        if not path.is_file():
            return None
        split, _ = panorama_tracking_split(Path(block["path"]), regions_path=path)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        arenas = [row for row in rows if re.fullmatch(r"arena(?:[_\s-]*(?:left|right|l|r))?", row["semantic_label"], re.I)]
        colonies = [row for row in rows if re.fullmatch(r"colony(?:[_\s-]*(?:left|right|l|r))?", row["semantic_label"], re.I)]
        if len(arenas) != 2 or len(colonies) != 2:
            return None
        arenas.sort(key=lambda row: float(row["tracking_x_min_px"]))
        colonies.sort(key=lambda row: float(row["tracking_x_min_px"]))
        scales = {float(row["mm_per_pixel"]) for row in arenas + colonies}
        if len(scales) != 1:
            raise ValueError(f"Inconsistent annotated scales: {path}")
        scale = scales.pop()
        geometry = dict(mm_per_px=scale,
                        colony_boxes_mm=[[float(row[f"tracking_{key}_px"]) * scale
                                          for key in ("x_min", "x_max", "y_min", "y_max")] for row in colonies],
                        bounds=dict(x_min_px=float(arenas[0]["tracking_x_min_px"]), x_split_px=split,
                                    x_max_px=float(arenas[1]["tracking_x_max_px"]),
                                    y_min_px=min(float(r["tracking_y_min_px"]) for r in arenas),
                                    y_max_px=max(float(r["tracking_y_max_px"]) for r in arenas)))
        geometries.append(geometry)
        sources.append(fingerprint(path))
    if any(geometry != geometries[0] for geometry in geometries[1:]):
        raise ValueError("Panorama geometry differs between blocks; align their coordinate systems before combining")
    return dict(geometries[0], sources=sources, grid_size_mm=grid_size_mm, grid_pad_mm=grid_pad_mm)


def build_plan(data_folder, *, fps=None, grid_size_mm=DEFAULT_GRID_SIZE_MM, grid_pad_mm=10.0):
    data_folder = Path(data_folder).resolve()
    tracked, skipped = [], []
    block_paths = sorted(p for p in data_folder.iterdir() if p.is_dir() and BLOCK_RE.fullmatch(p.name))
    if not block_paths:
        raise ValueError(f"{data_folder}: expected a data folder containing blockNN directories")
    for path in block_paths:
        block = discover_block(path, fps)
        if block:
            tracked.append(block)
        else:
            skipped.append(dict(block=path.name, reason="no stitched per-ant tracks"))
    groups = []
    for chain in consecutive_groups(tracked):
        if len(chain) < 2:
            skipped.append(dict(block=chain[0]["name"], reason="no adjacent tracked block"))
            continue
        rates = {block["fps"] for block in chain}
        if len(rates) != 1:
            raise ValueError("Cannot combine different frame rates without resampling")
        rate = rates.pop()
        origin = datetime.fromisoformat(chain[0]["start_datetime"])
        previous_stop = 0
        ants = {}
        for block in chain:
            offset = int(round((datetime.fromisoformat(block["start_datetime"]) - origin).total_seconds() * rate))
            if offset < previous_stop:
                raise ValueError(f"Overlapping or reversed recording intervals at {block['name']}; check timing metadata")
            block["frame_offset"] = offset
            block["gap_before_frames"] = offset - previous_stop
            previous_stop = offset + block["num_frames"]
            for (side, track_id), source in block.pop("tracks").items():
                if (side, track_id) not in ants:
                    ants[(side, track_id)] = dict(side=side, track_id=track_id, sources=[])
                ants[(side, track_id)]["sources"].append(dict(source, block=block["name"],
                                                              frame_offset=offset, num_frames=block["num_frames"]))
        group = dict(name=chain[0]["name"] + "_" + chain[-1]["name"], blocks=chain,
                     start_datetime=origin.isoformat(), fps=rate, num_frames=previous_stop,
                     geometry=shared_geometry(chain, grid_size_mm, grid_pad_mm), ants=[])
        for identity, ant in sorted(ants.items()):
            ant["name"] = stitched_parquet_path(Path(), ant["track_id"], origin.strftime("%H%M%S") + "_" + ant["side"]).name
            for kind in CACHE_METADATA:
                segments = cache_segments(ant, kind)
                if group["geometry"]:
                    for _, path, metadata in segments:
                        if "mm_per_px" in metadata and metadata["mm_per_px"] != group["geometry"]["mm_per_px"]:
                            raise ValueError(f"Cache and panorama annotation scales differ: {path}")
                if group["geometry"] and kind in {"colony_presence_vectors", "grid_occupancy_histograms"}:
                    continue
                require_compatible(segments, kind)
            group["ants"].append(ant)
        groups.append(group)
    if not groups:
        raise ValueError(f"No consecutive tracked blocks can be combined: {skipped}")
    for group in groups:
        group["relative_output"] = "" if len(groups) == 1 else group["name"]
    return dict(format_version=VERSION, data_folder=str(data_folder), skipped_blocks=skipped, groups=groups,
                time_policy="nominal-FPS elapsed timeline from recording timestamps; gaps stay unknown",
                identity_policy="colony side plus tag ID; no inferred cross-tag identity matches",
                source_only_policy="Exploratory caches, models, manual labels and interactions retain their source-block semantics; indexed, not concatenated")


def source_records(ant):
    yield from (source["track"] for source in ant["sources"])
    for source in ant["sources"]:
        for cache in source["caches"].values():
            yield from cache["files"]


def check_sources(plan):
    for group in plan["groups"]:
        if group["geometry"]:
            for source in group["geometry"]["sources"]:
                check_fingerprint(source)
        for block in group["blocks"]:
            if block["timing_source"]:
                check_fingerprint(block["timing_source"])
            for source in block.get("fps_sources", []):
                check_fingerprint(source)
        for ant in group["ants"]:
            for source in source_records(ant):
                check_fingerprint(source)


def worker(manifest, group_name, track, out):
    pa.set_cpu_count(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    plan = json.loads(Path(manifest).read_text())
    group = next(group for group in plan["groups"] if group["name"] == group_name)
    ant = next(ant for ant in group["ants"] if ant["name"] == Path(track).name)
    for record in source_records(ant):
        check_fingerprint(record)
    root = Path(plan["stage"]) / group["relative_output"]
    final_root = Path(plan["output"]) / group["relative_output"]
    inputs = [(Path(s["track"]["path"]), s["frame_offset"], s["block"]) for s in ant["sources"]]
    result = concatenate_shifted_parquets(inputs, root / "per_track" / ant["name"],
                                          metadata={"num_frames": str(group["num_frames"]), "fps": str(group["fps"]),
                                                    "start_datetime": group["start_datetime"], "block_combination_version": str(VERSION)},
                                          expected_track_id=ant["track_id"], keep_block_frame=True,
                                          frame_limits={s["block"]: s["num_frames"] for s in ant["sources"]})
    if result["frame_max"] is None or result["frame_max"] >= group["num_frames"]:
        raise ValueError("Combined track extends beyond the recording timeline")
    print(f"Combined {ant['name']}: {result['rows']:,} rows", flush=True)
    result["caches"] = combine_caches(ant, group, root, final_root)
    for record in source_records(ant):
        check_fingerprint(record)
    result.update(name=ant["name"], side=ant["side"], track_id=ant["track_id"], group=group_name)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out / "result.json", result)
    print(json.dumps(result), flush=True)


def finalize(manifest):
    plan = json.loads(Path(manifest).read_text())
    check_sources(plan)
    reports = []
    for group in plan["groups"]:
        root = Path(plan["stage"]) / group["relative_output"]
        (root / "analysis_cache/block_combination/per_track").mkdir(parents=True, exist_ok=True)
        summaries = []
        group_reports = []
        for ant in group["ants"]:
            task = Path(plan["run"]) / group["name"] / "tasks/per_track" / Path(ant["name"]).stem
            if not (task / "_SUCCESS").is_file():
                raise ValueError(f"Ant worker incomplete: {task}")
            report = json.loads((task / "result.json").read_text())
            path = root / "per_track" / ant["name"]
            pf = pq.ParquetFile(path)
            if pf.metadata.num_rows != report["rows"] or parquet_num_frames(path) != group["num_frames"]:
                raise ValueError(f"Combined parquet validation failed: {path}")
            for kind, info in report["caches"].items():
                if info["action"] == "unavailable":
                    continue
                cache = root / kind / "per_track" / Path(ant["name"]).stem
                meta = json.loads((cache / CACHE_METADATA[kind]).read_text())
                if kind in DENSE:
                    for filename, _, _ in DENSE[kind]:
                        array = np.load(cache / filename, mmap_mode="r")
                        if array.shape != (int(meta["frame_max"]) - int(meta["frame_min"]) + 1,):
                            raise ValueError(f"Combined cache span mismatch: {cache / filename}")
                if kind == "sleep_motion_labels":
                    summaries.append(meta["summary"])
            group_reports.append(report)
            _atomic_write_json(root / "analysis_cache/block_combination/per_track" / (Path(ant["name"]).stem + ".json"),
                               dict(report, source_tracks=[dict(block=s["block"], frame_offset=s["frame_offset"],
                                                                 track=s["track"]) for s in ant["sources"]]))
        if summaries:
            labels = root / "sleep_motion_labels"
            pd.DataFrame(summaries).to_parquet(labels / "sleep_motion_label_summary.parquet", index=False)
            _atomic_write_json(labels / "sleep_motion_label_summary.json",
                               dict(n_tracks=len(summaries), combination=True,
                                    n_frames=sum(row["n_frames"] for row in summaries),
                                    n_sleep_frames=sum(row["n_sleep_frames"] for row in summaries),
                                    n_wake_frames=sum(row["n_wake_frames"] for row in summaries),
                                    n_unknown_frames=sum(row["n_unknown_frames"] for row in summaries)))
        _atomic_write_json(root / "block_combination.json", {key: value for key, value in group.items() if key != "ants"})
        _atomic_write_json(root / "ant_combination_report.json", {"ants": group_reports})
        for kind in CACHE_METADATA:
            counts = Counter(report["caches"][kind]["action"] for report in group_reports)
            _atomic_write_json(root / (kind + "_combination_status.json"), dict(counts))
            if not counts.get("unavailable"):
                marker_name = {"speed_vectors": "speed_vector", "colony_presence_vectors": "colony_presence",
                               "grid_occupancy_histograms": "grid_occupancy", "sleep_motion": "sleep_motion",
                               "sleep_motion_labels": "sleep_motion_labels", "sleep_predictions": "sleep_prediction"}[kind]
                (root / kind / (marker_name + "_complete.ok")).write_text("block combination validated\n")
        # Retain access to source-specific analyses and their own metadata. A
        # relative link is portable between /bucket and a mounted bucket root.
        links = root / "source_blocks"
        links.mkdir(exist_ok=True)
        final_links = Path(plan["output"]) / group["relative_output"] / "source_blocks"
        for block in group["blocks"]:
            link = links / block["name"]
            if not link.is_symlink():
                link.symlink_to(os.path.relpath(block["path"], final_links), target_is_directory=True)
        reports.append(dict(group=group["name"], ants=len(group_reports), rows=sum(r["rows"] for r in group_reports)))
    _atomic_write_json(Path(plan["stage"]) / "combination_manifest.json", plan)
    _atomic_write_json(Path(plan["stage"]) / "_SUCCESS.json", {"run": plan["run"], "groups": reports})
    _atomic_write_json(Path(plan["run"]) / "READY_TO_PUBLISH.json", {"groups": reports})
    print("Validated all combined tracks and caches:", json.dumps(reports), flush=True)


def publish(manifest, *, wait=True):
    run = Path(json.loads(Path(manifest).read_text())["run"])
    with (run / "publisher.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return _publish(manifest, wait=wait)


def copy_result_tree(source, destination, run, workers=4):
    """Copy disjoint file lists concurrently; resume completed files via rsync.

    Balance by bytes so large pose parquets are distributed across transfers.
    All writes remain in this run's private incoming directory. Never combine
    --delete with these partial lists: another transfer owns the omitted files.
    """
    files = [path for path in source.rglob("*") if path.is_symlink() or path.is_file()]
    expected = {path.relative_to(source) for path in files}
    for path in destination.rglob("*"):
        if (path.is_symlink() or path.is_file()) and path.relative_to(destination) not in expected:
            path.unlink()
    buckets = [[] for _ in range(workers)]
    sizes = [0] * workers
    for path in sorted(files, key=lambda p: 0 if p.is_symlink() else p.stat().st_size, reverse=True):
        target = min(range(workers), key=sizes.__getitem__)
        buckets[target].append(path.relative_to(source))
        sizes[target] += 0 if path.is_symlink() else path.stat().st_size
    lists = run / "transfer_lists"
    lists.mkdir(exist_ok=True)
    commands = []
    for index, paths in enumerate(buckets):
        if not paths:
            continue
        file_list = lists / f"part{index}.txt"
        file_list.write_bytes(b"".join(os.fsencode(str(path)) + b"\0" for path in paths))
        commands.append(["rsync", "-a", "--relative", "--from0", "--files-from=" + str(file_list),
                         "--partial", str(source) + "/", str(destination) + "/"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(subprocess.run, command, check=True) for command in commands]
        for future in futures:
            future.result()


def _publish(manifest, *, wait=True):
    plan = json.loads(Path(manifest).read_text())
    run = Path(plan["run"])
    if (run / "PUBLISHED.json").is_file():
        print((run / "PUBLISHED.json").read_text())
        return
    while not (run / "READY_TO_PUBLISH.json").is_file():
        if not wait:
            raise ValueError("Final validation has not completed")
        submission = json.loads((run / "submission.json").read_text())
        result = subprocess.run(["squeue", "-h", "-j", submission["finalizer"], "-o", "%T|%R"],
                                check=True, capture_output=True, text=True)
        if "DependencyNeverSatisfied" in result.stdout or not result.stdout.strip():
            # Allow marker visibility after a just-completed job.
            if not (run / "READY_TO_PUBLISH.json").is_file():
                raise RuntimeError(f"Block combination failed; inspect {run}/finalize.err and ant worker logs")
        print("Waiting for ant jobs and validation:", result.stdout.strip(), flush=True)
        time.sleep(30)
    check_sources(plan)
    destination = Path(plan["output"])
    incoming = destination.with_name("." + destination.name + "." + run.name + ".partial")
    incoming.mkdir(parents=True, exist_ok=True)
    print(f"Copying validated outputs to {incoming}", flush=True)
    copy_result_tree(Path(plan["stage"]), incoming, run)
    print("Copy complete; verifying file sizes and parquet footers", flush=True)
    # Verify every published file's size and parquet footer before installation.
    count = 0
    for path in Path(plan["stage"]).rglob("*"):
        copied = incoming / path.relative_to(plan["stage"])
        if path.is_symlink():
            if not copied.is_symlink() or os.readlink(path) != os.readlink(copied):
                raise ValueError(f"Source reference failed to copy: {copied}")
        elif path.is_file():
            if not copied.is_file() or path.stat().st_size != copied.stat().st_size:
                raise ValueError(f"Incomplete transfer: {copied}")
            if path.suffix == ".parquet":
                if pq.ParquetFile(path).metadata.num_rows != pq.ParquetFile(copied).metadata.num_rows:
                    raise ValueError(f"Parquet transfer validation failed: {copied}")
            count += 1
    previous = None
    if destination.exists():
        if not plan["overwrite"]:
            raise FileExistsError(f"Output already exists: {destination}; use --overwrite for a new generation")
        previous = destination.with_name(destination.name + ".previous_" + run.name)
        destination.rename(previous)
    try:
        incoming.rename(destination)
    except BaseException:
        if previous is not None:
            previous.rename(destination)
        raise
    result = dict(output=str(destination), files=count, previous_output=str(previous) if previous else None,
                  groups=json.loads((run / "READY_TO_PUBLISH.json").read_text())["groups"])
    _atomic_write_json(run / "PUBLISHED.json", result)
    lock = Path(plan["lock"])
    if (lock / "run").read_text().strip() == str(run):
        (lock / "run").unlink()
        lock.rmdir()
    print("PUBLISHED", json.dumps(result), flush=True)


def start_publisher(plan, submission, wait):
    run = Path(plan["run"])
    if wait:
        print("SUBMITTED", json.dumps(submission), flush=True)
        publish(run / "plan.json")
    else:
        with (run / "publish.log").open("a") as log:
            process = subprocess.Popen([sys.executable, str(run / "code/tracking/colony/combine_blocks.py"),
                                        "publish", "--manifest", str(run / "plan.json")], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, start_new_session=True)
        submission.update(publisher_pid=process.pid, publisher_host=os.uname().nodename)
        _atomic_write_json(run / "submission.json", submission)
        print("SUBMITTED", json.dumps(submission), flush=True)


def retry(manifest, *, wait=False):
    """Resubmit unfinished ant workers without replacing a running generation."""
    plan = json.loads(Path(manifest).read_text())
    run = Path(plan["run"])
    if (run / "PUBLISHED.json").is_file() or (run / "READY_TO_PUBLISH.json").is_file():
        publish(manifest)
        return
    check_sources(plan)
    submission = json.loads((run / "submission.json").read_text())
    listing = subprocess.check_output(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T"], text=True)
    live = dict(line.split("|", 1) for line in listing.splitlines())
    worker_ids = set(submission.get("retry_workers", []))
    for group in plan["groups"]:
        ids = run / group["name"] / "tasks/jobs/block_combine_job_ids.tsv"
        if ids.exists():
            worker_ids.update(line.split("\t")[1] for line in ids.read_text().splitlines())
    if worker_ids & live.keys():
        raise RuntimeError("Ant workers are still active; wait for them before retrying")
    parent_ids = [item["completion_job"] for item in submission["groups"]]
    if submission.get("finalizer"):
        parent_ids.append(submission["finalizer"])
    for jid in parent_ids:
        if jid in live:
            if live[jid] != "PENDING":
                raise RuntimeError(f"Completion/validation job {jid} is still active")
            subprocess.run(["scancel", jid], check=True)
    new_jobs = []
    for group in plan["groups"]:
        jobs = run / group["name"] / "tasks/jobs"
        for row in (jobs / "block_combine_worklist.tsv").read_text().splitlines():
            index, _, _, _, _, task = row.split("\t")
            if (Path(task) / "_SUCCESS").is_file():
                continue
            script = jobs / "workers" / f"block_combine_task{index}.sbatch"
            result = subprocess.check_output(["sbatch", "--parsable", str(script)], text=True)
            jid = result.strip().split(";")[0]
            new_jobs.append(jid)
            submission.setdefault("retry_workers", []).append(jid)
            _atomic_write_json(run / "submission.json", submission)
    command = ["sbatch", "--parsable"]
    if new_jobs:
        command.append("--dependency=afterok:" + ":".join(new_jobs))
    command.append(str(run / "finalize.sbatch"))
    submission["finalizer"] = subprocess.check_output(command, text=True).strip().split(";")[0]
    _atomic_write_json(run / "submission.json", submission)
    start_publisher(plan, submission, wait)


def submit(args):
    plan = build_plan(args.data_folder, fps=args.fps, grid_size_mm=args.grid_size_mm, grid_pad_mm=args.grid_pad_mm)
    destination = (args.output or Path(args.data_folder) / "continous_stitched").resolve()
    for group in plan["groups"]:
        for block in group["blocks"]:
            source = Path(block["path"])
            if destination == source or destination in source.parents or source in destination.parents:
                raise ValueError("The combined output must be separate from all input block directories")
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"{destination} already exists; use --overwrite to replace it after validation")
    work_parent = args.work_root or Path(f"/flash/ReiterU/ant_tmp/{getpass.getuser()}/block_combination")
    work_parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=Path(args.data_folder).name + "_", dir=work_parent)).resolve()
    lock = destination.with_name("." + destination.name + ".lock")
    if not args.dry_run:
        lock.mkdir()  # Fail before submission if another run owns the output.
        (lock / "run").write_text(str(run) + "\n")
    plan.update(run=str(run), stage=str(run / "result"), output=str(destination),
                overwrite=args.overwrite, lock=str(lock), python=sys.executable)
    code = run / "code"
    for directory in ("tracking", "analysis", "scripts", "camera_cal"):
        shutil.copytree(REPO / directory, code / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.ipynb", ".pytest_cache"))
    plan["code_sha256"] = {str(path.relative_to(code)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(code.rglob("*")) if path.is_file()}
    manifest = run / "plan.json"
    _atomic_write_json(manifest, plan)
    final_dependencies = []
    submission = {"run": str(run), "groups": [], "manifest": str(manifest)}
    _atomic_write_json(run / "submission.json", submission)
    for group in plan["groups"]:
        group_run = run / group["name"]
        links = group_run / "inputs"
        links.mkdir(parents=True)
        root = Path(plan["stage"]) / group["relative_output"]
        root.mkdir(parents=True, exist_ok=True)
        if group["geometry"]:
            _atomic_write_json(root / "grid_bounds_from_panorama_regions.json", group["geometry"]["bounds"])
            shutil.copy2(group["geometry"]["sources"][0]["path"], root / "panorama_regions.csv")
        for ant in group["ants"]:
            (links / ant["name"]).symlink_to(ant["sources"][0]["track"]["path"])
        operation = shlex.join([sys.executable, str(code / "tracking/colony/combine_blocks.py"), "worker",
                                "--manifest", str(manifest), "--group", group["name"]])
        operation += ' --track "$TRACK_PATH" --out "$TASK_OUTPUT_DIR"'
        command = ["bash", str(code / "scripts/per_track_slurm_fanout.sh"), "--per_track_dir", str(links),
                   "--flash_output_dir", str(group_run / "tasks"), "--bucket_output_dir", str(destination),
                   "--operation_name", "block_combine", "--operation_cmd", operation, "--run_workdir", str(code),
                   "--python_bin", sys.executable, "--worker_python_bin", sys.executable, "--no_conda",
                   "--worker_setup", f"export OMP_NUM_THREADS={args.cpus}; export OPENBLAS_NUM_THREADS=1",
                   "--no_transfer_to_bucket", "--cpus", str(args.cpus), "--mem", args.mem, "--time", args.time,
                   "--partition", args.partition]
        if args.dry_run:
            command.append("--dry_run")
        print(f"{group['name']}: {len(group['ants'])} ants, {group['num_frames']:,} frames at {group['fps']:g} FPS", flush=True)
        with (group_run / "submit.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        if not args.dry_run:
            done_id = (group_run / "tasks/jobs/block_combine_complete_job_id.txt").read_text().strip()
            final_dependencies.append(done_id)
            submission["groups"].append(dict(group=group["name"], completion_job=done_id))
            _atomic_write_json(run / "submission.json", submission)
    script = run / "finalize.sbatch"
    script.write_text("\n".join(["#!/bin/bash", "#SBATCH -J block_combine_validate", f"#SBATCH -p {args.partition}",
                                 "#SBATCH -c 1", "#SBATCH --mem=4G", "#SBATCH -t 0-01:00:00",
                                 f"#SBATCH -o {run}/finalize.out", f"#SBATCH -e {run}/finalize.err",
                                 "set -euo pipefail", "umask 0002", "export PYTHONNOUSERSITE=1",
                                 shlex.join([sys.executable, str(code / "tracking/colony/combine_blocks.py"),
                                             "finalize", "--manifest", str(manifest)]), ""]))
    subprocess.run(["bash", "-n", str(script)], check=True)
    if args.dry_run:
        print("DRY RUN", str(manifest), flush=True)
        return
    result = subprocess.run(["sbatch", "--parsable", "--dependency=afterok:" + ":".join(final_dependencies), str(script)],
                            check=True, capture_output=True, text=True)
    submission["finalizer"] = result.stdout.strip().split(";")[0]
    _atomic_write_json(run / "submission.json", submission)
    start_publisher(plan, submission, args.wait)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in {"submit", "plan", "worker", "finalize", "publish", "retry", "-h", "--help"}:
        argv.insert(0, "submit")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "submit"):
        sub = commands.add_parser(command)
        sub.add_argument("data_folder", type=Path)
        sub.add_argument("--fps", type=float, help="Required only if frame rate is absent from source metadata")
        sub.add_argument("--grid-size-mm", type=float, default=DEFAULT_GRID_SIZE_MM)
        sub.add_argument("--grid-pad-mm", type=float, default=10.0)
        if command == "submit":
            sub.add_argument("--output", type=Path, help="Default: DATA_FOLDER/continous_stitched")
            sub.add_argument("--work-root", type=Path, help="Scratch root; defaults to /flash/ReiterU/ant_tmp/USER/block_combination")
            sub.add_argument("--cpus", type=int, default=2)
            sub.add_argument("--mem", default="16G")
            sub.add_argument("--time", default="0-04:00:00")
            sub.add_argument("--partition", default="compute")
            sub.add_argument("--dry-run", action="store_true")
            sub.add_argument("--overwrite", action="store_true", help="Archive the previous combined folder when installing the validated replacement")
            sub.add_argument("--wait", action="store_true", help="Wait for jobs and publication instead of starting a detached publisher")
    for command in ("worker", "finalize", "publish", "retry"):
        sub = commands.add_parser(command)
        sub.add_argument("--manifest", type=Path, required=True)
        if command == "worker":
            sub.add_argument("--group", required=True)
            sub.add_argument("--track", type=Path, required=True)
            sub.add_argument("--out", type=Path, required=True)
        if command == "retry":
            sub.add_argument("--wait", action="store_true")
    args = parser.parse_args(argv)
    os.umask(0o002)
    if args.command in {"plan", "submit"}:
        if args.grid_size_mm <= 0 or not np.isfinite(args.grid_size_mm) or args.grid_pad_mm < 0 or not np.isfinite(args.grid_pad_mm):
            parser.error("Grid size must be positive; padding must be nonnegative")
    if args.command == "plan":
        print(json.dumps(build_plan(args.data_folder, fps=args.fps, grid_size_mm=args.grid_size_mm,
                                    grid_pad_mm=args.grid_pad_mm), indent=2))
    elif args.command == "submit":
        submit(args)
    elif args.command == "worker":
        worker(args.manifest, args.group, args.track, args.out)
    elif args.command == "finalize":
        finalize(args.manifest)
    elif args.command == "publish":
        publish(args.manifest)
    elif args.command == "retry":
        retry(args.manifest, wait=args.wait)


if __name__ == "__main__":
    main()
