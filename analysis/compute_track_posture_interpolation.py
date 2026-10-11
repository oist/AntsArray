#!/usr/bin/env python3
"""Fill short, bounded SLEAP node gaps in one stitched per-track parquet.

The original tracking is never modified. Whole-pose/identity losses, camera
changes and recording boundaries remain unknown. A gap is measured by its
number of missing samples (24 missing frames at 24 Hz is one second).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


VERSION = 1
FLAGS = ("PostureInterpolated", "PostureGapFrames")
# Only frame-level metadata may be inherited by a newly created node row.
FRAME_FIELDS = {"Frame", "TrackID", "TrackX", "TrackY", "ArucoX", "ArucoY",
                "SleapAnchorX", "SleapAnchorY", "CameraID", "ArucoCam", "SleapCam",
                "source_file", "source_block", "recording_id", "Timestamp", "timestamp"}


def values(table, name):
    return table[name].to_numpy(zero_copy_only=False)


def interpolate_table(table: pa.Table, max_gap_frames: int):
    """Return a sorted, schema-preserving table plus auditable coverage counts."""
    required = {"Frame", "TrackID", "Bodypoint", "X", "Y", "SleapCam", "TrackX", "TrackY"}
    if not required <= set(table.column_names):
        raise ValueError(f"Missing required columns: {sorted(required - set(table.column_names))}")
    if set(FLAGS) & set(table.column_names):
        raise ValueError("Input already contains interpolation provenance; use the raw track")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be nonnegative")
    frame, node = values(table, "Frame"), values(table, "Bodypoint")
    if np.any(frame < 0) or np.any((node < 0) | (node > 9)):
        raise ValueError("Expected nonnegative frames and the colony's bodypoints 0..9")
    if len(np.unique(values(table, "TrackID"))) > 1:
        raise ValueError("Expected exactly one ant per input")
    # Reject ambiguous rows rather than silently selecting one detection.
    key = frame * 10 + node
    if len(key) > 1 and np.any(np.diff(key) <= 0):
        table = table.take(pc.sort_indices(table, sort_keys=[("Frame", "ascending"), ("Bodypoint", "ascending")]))
        frame, node = values(table, "Frame"), values(table, "Bodypoint")
        if np.any(np.diff(frame * 10 + node) == 0):
            raise ValueError("Duplicate Frame/Bodypoint rows")
    x, y = values(table, "X"), values(table, "Y")
    finite = np.isfinite(x) & np.isfinite(y)
    anchors = np.flatnonzero(node == 0)
    af = frame[anchors]
    cam = values(table, "SleapCam")
    ac = cam[anchors]
    good = (finite[anchors] & np.isfinite(ac) & (ac >= 0)
            & np.isfinite(values(table, "TrackX")[anchors])
            & np.isfinite(values(table, "TrackY")[anchors]))
    cuts = ((np.diff(af) != 1) | (np.diff(ac) != 0) | ~good[1:] | ~good[:-1])
    for name in ("source_block", "recording_id", "source_file"):
        if name in table.column_names:
            source = values(table.select([name]).take(pa.array(anchors)), name)
            if name == "source_file":
                # Adjacent chunks of one recording are continuous; different
                # timestamped recordings must never be bridged.
                source = np.array([str(s).split("_chunk", 1)[0] for s in source])
            cuts |= source[1:] != source[:-1]
    segments = np.r_[0, np.cumsum(cuts)] if len(anchors) else np.array([], dtype=int)
    segments[~good] = -1
    imputed = np.zeros(len(table), dtype=bool)
    gap_frames = np.zeros(len(table), dtype=np.int32)
    new_tables = []
    before_counts = np.bincount(frame[finite], minlength=int(frame.max()) + 1 if len(frame) else 0)
    after_counts = before_counts.copy()
    stats = []
    changed_x, changed_y = None, None
    for bp in range(1, 10):
        rows = np.flatnonzero(node == bp)
        observed = rows[finite[rows]]
        lengths = np.diff(frame[observed]) - 1
        candidates = np.flatnonzero((lengths > 0) & (lengths <= max_gap_frames))
        left, right = observed[candidates], observed[candidates + 1]
        if len(anchors) and len(left):
            ai = np.searchsorted(af, frame[left])
            bi = np.searchsorted(af, frame[right])
            inside = (ai < len(af)) & (bi < len(af))
            ai, bi = np.minimum(ai, len(af) - 1), np.minimum(bi, len(af) - 1)
            eligible = (inside & (af[ai] == frame[left]) & (af[bi] == frame[right])
                        & (segments[ai] >= 0) & (segments[ai] == segments[bi])
                        & (cam[left] == ac[ai]) & (cam[right] == ac[bi]))
            left, right = left[eligible], right[eligible]
        else:
            left, right = left[:0], right[:0]
        lengths = frame[right] - frame[left] - 1
        repeats = lengths.astype(np.int64)
        starts = np.repeat(frame[left], repeats)
        offset = np.arange(int(repeats.sum())) - np.repeat(np.cumsum(repeats) - repeats, repeats) + 1
        ff = starts + offset
        gg = np.repeat(lengths, repeats).astype(np.int32)
        weight = offset / (gg + 1)
        xx = (1 - weight) * np.repeat(x[left], repeats) + weight * np.repeat(x[right], repeats)
        yy = (1 - weight) * np.repeat(y[left], repeats) + weight * np.repeat(y[right], repeats)
        existing = np.searchsorted(frame[rows], ff)
        present = existing < len(rows)
        if len(rows):
            existing = np.minimum(existing, len(rows) - 1)
            present &= frame[rows[existing]] == ff
        if present.any():
            rr = rows[existing[present]]
            if changed_x is None:
                changed_x, changed_y = x.copy(), y.copy()
            # Even a partially observed X/Y pair retains its finite coordinate.
            changed_x[rr] = np.where(np.isfinite(x[rr]), x[rr], xx[present])
            changed_y[rr] = np.where(np.isfinite(y[rr]), y[rr], yy[present])
            imputed[rr], gap_frames[rr] = True, gg[present]
        absent = ~present
        if absent.any():
            template = table.take(pa.array(anchors[np.searchsorted(af, ff[absent])]))
            columns = []
            for field in table.schema:
                if field.name in FRAME_FIELDS:
                    col = template[field.name]
                elif field.name in ("X", "Y", "Bodypoint"):
                    val = {"X": xx[absent], "Y": yy[absent], "Bodypoint": np.full(absent.sum(), bp)}[field.name]
                    col = pa.array(val, type=field.type)
                else:
                    col = pa.nulls(int(absent.sum()), type=field.type)
                columns.append(col)
            added = pa.Table.from_arrays(columns, schema=table.schema)
            added = added.append_column(FLAGS[0], pa.array(np.ones(absent.sum(), dtype=bool)))
            added = added.append_column(FLAGS[1], pa.array(gg[absent]))
            new_tables.append(added)
        np.add.at(after_counts, ff, 1)
        stats.append(dict(bodypoint=bp, observed_points=int(len(observed)), filled_points=int(len(ff)),
                          filled_gaps=int(len(lengths)),
                          gap_length_counts={str(k): int(v) for k, v in zip(*np.unique(lengths, return_counts=True))}))
    if changed_x is not None:
        for name, arr in (("X", changed_x), ("Y", changed_y)):
            table = table.set_column(table.schema.get_field_index(name), table.schema.field(name), pa.array(arr))
    table = table.append_column(FLAGS[0], pa.array(imputed)).append_column(FLAGS[1], pa.array(gap_frames))
    if new_tables:
        table = pa.concat_tables([table, *new_tables])
        table = table.take(pc.sort_indices(table, sort_keys=[("Frame", "ascending"), ("Bodypoint", "ascending")]))
    stats = dict(bodypoints=stats, filled_points=sum(s["filled_points"] for s in stats),
                 filled_gaps=sum(s["filled_gaps"] for s in stats),
                 complete_frames_before=int((before_counts == 10).sum()),
                 complete_frames_after=int((after_counts == 10).sum()),
                 anchor_frames=int(len(anchors)), matched_anchor_frames=int(good.sum()))
    return table, stats


def run(track: Path, out: Path, fps: float = 24., max_gap_seconds: float = 1.):
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(max_gap_seconds) or max_gap_seconds < 0:
        raise ValueError("fps must be positive and gap duration nonnegative")
    out.mkdir(parents=True, exist_ok=True)
    output = out / track.name
    if output.resolve() == track.resolve():
        raise ValueError("Output must be separate from original tracking")
    marker = out / "_SUCCESS"
    marker.unlink(missing_ok=True)
    start = time.monotonic()
    source_stat = track.stat()
    source = dict(path=str(track.resolve()), size=source_stat.st_size, mtime_ns=source_stat.st_mtime_ns)
    pf = pq.ParquetFile(track)
    original_meta = pf.schema_arrow.metadata or {}
    if b"fps" in original_meta and not math.isclose(float(original_meta[b"fps"]), fps):
        raise ValueError("--fps differs from the source parquet metadata")
    max_frames = math.floor(fps * max_gap_seconds + 1e-9)
    parameters = dict(method="bounded_linear", version=VERSION, fps=fps,
                      max_gap_seconds=max_gap_seconds, max_gap_frames=max_frames,
                      gap_definition="number of missing native frames",
                      continuity="matched anchor, tracked position, same SLEAP camera and recording")
    table, summary = interpolate_table(pf.read(), max_frames)
    summary.update(source=source, parameters=parameters, input_rows=pf.metadata.num_rows,
                   output_rows=len(table), num_frames=int(original_meta.get(b"num_frames", 0)),
                   code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    metadata = {**original_meta, b"posture_interpolation": json.dumps(parameters).encode()}
    table = table.replace_schema_metadata(metadata)
    temporary = output.with_suffix(f".parquet.{os.getpid()}.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd", row_group_size=1_000_000)
        current = track.stat()
        if (current.st_size, current.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise RuntimeError("Source tracking changed during interpolation")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    summary["elapsed_seconds"] = time.monotonic() - start
    summary["output_bytes"] = output.stat().st_size
    report = out / "posture_interpolation_metadata.json"
    tmp_report = report.with_suffix(".json.tmp")
    tmp_report.write_text(json.dumps(summary, indent=2) + "\n")
    tmp_report.replace(report)
    marker.write_text("completed bounded linear posture interpolation\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--max_gap_seconds", type=float, default=1.)
    args = parser.parse_args()
    print(json.dumps(run(args.track, args.out, args.fps, args.max_gap_seconds)), flush=True)


if __name__ == "__main__":
    main()
