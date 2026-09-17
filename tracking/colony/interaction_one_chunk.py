#!/usr/bin/env python3
"""Compute undirected distance-threshold skeleton contacts for one track chunk."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tracking.colony.skeleton_contacts import CONTACT_GEOMETRY, SKELETON_EDGES, skeleton_contact_pairs, validate_distance, validate_scale

REQUIRED_COLUMNS = {"Frame", "TrackID", "Bodypoint", "X", "Y"}
DEFAULT_ANTENNA_BODYPOINTS = (4, 5, 6, 7, 8, 9)
DEFAULT_MICRO_DISTANCE_MM = 0.1


def interaction_parameters(*, mm_per_px, interaction_radius_mm, micro_interaction_distance_mm,
                           antenna_bodypoints, frame_start, max_frames, frame_step):
    return dict(schema_version=2, geometry=CONTACT_GEOMETRY, directed=False,
                mm_per_px=validate_scale(mm_per_px),
                micro_interaction_distance_mm=validate_distance(micro_interaction_distance_mm),
                skeleton_edges=[list(edge) for edge in SKELETON_EDGES],
                frame_start=int(frame_start), max_frames=max_frames, frame_step=int(frame_step))


def cache_matches(output_path: Path, chunk_file: Path, parameters: dict) -> bool:
    try:
        metadata = json.loads(output_path.with_suffix(".metadata.json").read_text())
        stat = chunk_file.stat()
        return (output_path.is_file() and metadata["parameters"] == parameters
                and metadata["input_size"] == stat.st_size and metadata["input_mtime_ns"] == stat.st_mtime_ns
                and metadata["output_size"] == output_path.stat().st_size)
    except (OSError, ValueError, KeyError):
        return False


def infer_side(path: Path) -> str | None:
    match = re.search(r"_(left|right)(?:\.parquet)?$", path.name)
    return match.group(1) if match else None


def parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).schema.names


def parquet_frame_bounds(path: Path) -> tuple[int, int]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    frame_col_idx = parquet_file.schema.names.index("Frame")
    mins = []
    maxs = []
    for row_group_idx in range(parquet_file.metadata.num_row_groups):
        stats = parquet_file.metadata.row_group(row_group_idx).column(frame_col_idx).statistics
        if stats is not None:
            mins.append(int(stats.min))
            maxs.append(int(stats.max))
    if mins and maxs:
        return min(mins), max(maxs)

    frame = pd.read_parquet(path, columns=["Frame"])["Frame"]
    return int(frame.min()), int(frame.max())


def frame_windows(frame_start: int, frame_stop: int, batch_size: int) -> list[tuple[int, int]]:
    if int(batch_size) <= 0:
        raise ValueError("frame_batch_size must be positive")
    return [
        (start, min(start + int(batch_size), int(frame_stop)))
        for start in range(int(frame_start), int(frame_stop), int(batch_size))
    ]


def load_chunk_window(path: Path, *, frame_start: int, frame_stop: int, frame_step: int) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    missing = REQUIRED_COLUMNS.difference(parquet_columns(path))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    table = ds.dataset(path, format="parquet").to_table(
        columns=sorted(REQUIRED_COLUMNS),
        filter=(pc.field("Frame") >= int(frame_start)) & (pc.field("Frame") < int(frame_stop)),
        use_threads=True,
    )
    df = table.to_pandas()
    if df.empty:
        return df

    for col in ["Frame", "TrackID", "Bodypoint"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["X", "Y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    df[["Frame", "TrackID", "Bodypoint"]] = df[["Frame", "TrackID", "Bodypoint"]].astype(np.int64)
    df = df[df.Bodypoint.between(0, 9)].drop_duplicates()
    if df.duplicated(["Frame", "TrackID", "Bodypoint"]).any():
        raise ValueError(f"Conflicting finished poses in {path}")
    if int(frame_step) > 1:
        df = df[((df["Frame"] - int(frame_start)) % int(frame_step)) == 0].copy()
    return df.sort_values(["Frame", "TrackID", "Bodypoint"], kind="mergesort").reset_index(drop=True)


def frame_track_arrays(frame_df: pd.DataFrame) -> dict[int, np.ndarray]:
    ids, indices = np.unique(frame_df.TrackID.to_numpy(np.int64), return_inverse=True)
    xy = np.full((len(ids), 10, 2), np.nan)
    xy[indices, frame_df.Bodypoint.to_numpy(np.int64)] = frame_df[["X", "Y"]].to_numpy(float)
    return {int(ant): pose for ant, pose in zip(ids, xy)}


def detect_interactions(
    tracks_df: pd.DataFrame,
    *,
    micro_distance_px: float,
    mm_per_px: float,
    progress_every_frames: int,
    run_start_time: float,
    processed_frames_before: int,
    interactions_before: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interaction_rows = []
    summary_rows = []

    grouped = tracks_df.groupby("Frame", sort=True)
    n_frames = len(grouped)
    for i, (frame, frame_df) in enumerate(grouped, start=1):
        tracks = frame_track_arrays(frame_df)
        pairs = skeleton_contact_pairs(tracks, distance_mm=micro_distance_px*mm_per_px, mm_per_pixel=mm_per_px)
        interaction_rows.extend((int(frame), a, b, distance) for a, b, distance in pairs)

        summary_rows.append(
            {
                "Frame": int(frame),
                "n_tracks": len(tracks),
                "n_candidate_pairs": len(tracks)*(len(tracks)-1)//2,
                "n_interactions": len(pairs),
            }
        )
        should_print = i == 1 or i == n_frames or (progress_every_frames and i % int(progress_every_frames) == 0)
        if should_print:
            elapsed = max(time.perf_counter() - run_start_time, 1e-9)
            total_frames = int(processed_frames_before) + i
            print(
                f"frame={int(frame)} total_frames={total_frames} "
                f"elapsed={elapsed:.1f}s speed={total_frames / elapsed:.2f} frames/s "
                f"interactions={int(interactions_before) + len(interaction_rows)}",
                flush=True,
            )

    interactions = pd.DataFrame(interaction_rows, columns=["Frame", "ant_a", "ant_b", "distance_mm"])
    return interactions, pd.DataFrame(summary_rows)


def interaction_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("Frame", pa.int64()),
            ("ant_a", pa.int64()),
            ("ant_b", pa.int64()),
            ("distance_mm", pa.float64()),
        ]
    )


def append_interactions_parquet(writer, interactions: pd.DataFrame, path: Path, *, force_write: bool = False):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = interaction_schema()
    if writer is None:
        writer = pq.ParquetWriter(path, schema=schema, compression="zstd")

    if force_write or not interactions.empty:
        table = pa.Table.from_pandas(
            interactions[["Frame", "ant_a", "ant_b", "distance_mm"]].astype(
                {"Frame": "int64", "ant_a": "int64", "ant_b": "int64", "distance_mm": "float64"}),
            schema=schema,
            preserve_index=False,
        )
        writer.write_table(table)
    return writer


def process_chunk(
    *,
    chunk_file: Path,
    output_path: Path,
    mm_per_px: float,
    interaction_radius_mm: float,
    micro_interaction_distance_mm: float,
    antenna_bodypoints: tuple[int, ...],
    frame_start: int,
    max_frames: int | None,
    frame_step: int,
    frame_batch_size: int,
    progress_every_frames: int,
    skip_existing: bool,
) -> Path:
    interactions_path = (
        output_path
        if output_path.suffix == ".parquet"
        else output_path / f"{chunk_file.stem}.parquet"
    )
    parameters = interaction_parameters(mm_per_px=mm_per_px, interaction_radius_mm=interaction_radius_mm,
                                        micro_interaction_distance_mm=micro_interaction_distance_mm,
                                        antenna_bodypoints=antenna_bodypoints, frame_start=frame_start,
                                        max_frames=max_frames, frame_step=frame_step)
    if skip_existing and cache_matches(interactions_path, chunk_file, parameters):
        print(f"Skipping existing {interactions_path}", flush=True)
        return interactions_path

    interactions_path.parent.mkdir(parents=True, exist_ok=True)
    working_path = interactions_path.with_name(f".{interactions_path.stem}.{os.getpid()}.partial.parquet")

    chunk_min_frame, chunk_max_frame = parquet_frame_bounds(chunk_file)
    start = max(int(frame_start), int(chunk_min_frame))
    chunk_stop = int(chunk_max_frame) + 1
    stop = chunk_stop if max_frames is None else min(start + int(max_frames), chunk_stop)
    windows = frame_windows(start, stop, int(frame_batch_size))

    micro_distance_px = float(micro_interaction_distance_mm) / float(mm_per_px)

    print(f"Chunk: {chunk_file}", flush=True)
    print(f"Output file: {interactions_path}", flush=True)
    print(f"Processing frames: {start}-{stop - 1} in {len(windows)} windows", flush=True)
    print(f"Geometry: {CONTACT_GEOMETRY}; undirected, no center cutoff or temporal filters", flush=True)
    print(f"Micro-interaction distance: {micro_interaction_distance_mm:g} mm = {micro_distance_px:.1f} track units", flush=True)

    run_start = time.perf_counter()
    writer = None
    processed_frames = 0
    total_interactions = 0
    total_candidate_pairs = 0

    for batch_idx, (window_start, window_stop) in enumerate(windows, start=1):
        read_start = time.perf_counter()
        print(f"read window {batch_idx}/{len(windows)} frames={window_start}-{window_stop - 1}", flush=True)
        chunk_tracks = load_chunk_window(
            chunk_file,
            frame_start=window_start,
            frame_stop=window_stop,
            frame_step=frame_step,
        )
        read_elapsed = time.perf_counter() - read_start
        if chunk_tracks.empty:
            print(f"window {batch_idx}: no rows read in {read_elapsed:.1f}s", flush=True)
            continue

        print(
            f"window {batch_idx}: loaded {len(chunk_tracks):,} rows, "
            f"{chunk_tracks['Frame'].nunique()} frames, "
            f"{chunk_tracks['TrackID'].nunique()} tracks in {read_elapsed:.1f}s",
            flush=True,
        )
        interactions, frame_summary = detect_interactions(
            chunk_tracks,
            micro_distance_px=micro_distance_px,
            mm_per_px=mm_per_px,
            progress_every_frames=int(progress_every_frames),
            run_start_time=run_start,
            processed_frames_before=processed_frames,
            interactions_before=total_interactions,
        )
        if not interactions.empty:
            writer = append_interactions_parquet(writer, interactions, working_path)

        processed_frames += int(len(frame_summary))
        total_interactions += int(len(interactions))
        total_candidate_pairs += int(frame_summary["n_candidate_pairs"].sum()) if not frame_summary.empty else 0
        elapsed = time.perf_counter() - run_start
        print(
            f"window {batch_idx} done: total_frames={processed_frames}, "
            f"total_interactions={total_interactions:,}, "
            f"elapsed={elapsed:.1f}s, speed={processed_frames / max(elapsed, 1e-9):.2f} frames/s",
            flush=True,
        )

    if writer is None:
        writer = append_interactions_parquet(
            writer,
            pd.DataFrame(columns=["Frame", "ant_a", "ant_b", "distance_mm"]),
            working_path,
            force_write=True,
        )
    writer.close()
    working_path.replace(interactions_path)
    stat = chunk_file.stat()
    metadata = dict(parameters=parameters, input_file=str(chunk_file), input_size=stat.st_size,
                    input_mtime_ns=stat.st_mtime_ns, output_size=interactions_path.stat().st_size,
                    processed_frames=processed_frames, n_pair_detections=total_interactions,
                    first_frame=start, stop_frame=stop)
    metadata_path = interactions_path.with_suffix(".metadata.json")
    metadata_temp = metadata_path.with_suffix(".json.tmp")
    metadata_temp.write_text(json.dumps(metadata, indent=2) + "\n")
    metadata_temp.replace(metadata_path)

    elapsed = time.perf_counter() - run_start
    print(
        f"Finished chunk={chunk_file.name}: "
        f"{processed_frames:,} frames, {total_interactions:,} pair detections, "
        f"{elapsed:.1f}s, {processed_frames / max(elapsed, 1e-9):.2f} frames/s",
        flush=True,
    )
    print(f"Wrote {interactions_path}", flush=True)
    return interactions_path


def parse_antenna_bodypoints(values: list[int] | None) -> tuple[int, ...]:
    if not values:
        return tuple(int(bp) for bp in DEFAULT_ANTENNA_BODYPOINTS)
    return tuple(int(bp) for bp in values)


def parse_optional_int(value: str | None) -> int | None:
    if value is None or str(value).lower() in {"none", "off", "all"}:
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk_file", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--mm_per_px", type=float, default=0.016)
    parser.add_argument("--interaction_radius_mm", type=float, default=8.0, help="Deprecated; ignored by skeleton-distance detection")
    parser.add_argument("--micro_interaction_distance_mm", type=float, default=DEFAULT_MICRO_DISTANCE_MM)
    parser.add_argument("--antenna_bodypoint", action="append", type=int, default=None, help="Deprecated; all skeleton nodes are used")
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--max_frames", default=None, help="None/all means process the full chunk.")
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--frame_batch_size", type=int, default=3000)
    parser.add_argument("--progress_every_frames", type=int, default=500)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    process_chunk(
        chunk_file=args.chunk_file,
        output_path=args.output_path,
        mm_per_px=float(args.mm_per_px),
        interaction_radius_mm=float(args.interaction_radius_mm),
        micro_interaction_distance_mm=float(args.micro_interaction_distance_mm),
        antenna_bodypoints=parse_antenna_bodypoints(args.antenna_bodypoint),
        frame_start=int(args.frame_start),
        max_frames=parse_optional_int(args.max_frames),
        frame_step=int(args.frame_step),
        frame_batch_size=int(args.frame_batch_size),
        progress_every_frames=int(args.progress_every_frames),
        skip_existing=bool(args.skip_existing),
    )


if __name__ == "__main__":
    main()
