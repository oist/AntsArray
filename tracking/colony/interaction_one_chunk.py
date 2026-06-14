#!/usr/bin/env python3
"""Compute directed antenna-to-body interactions for one chunk track parquet."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"Frame", "TrackID", "Bodypoint", "X", "Y", "TrackX", "TrackY"}
DEFAULT_ANTENNA_BODYPOINTS = (4, 5, 6, 7, 8, 9)


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
    for col in ["X", "Y", "TrackX", "TrackY"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    df[["Frame", "TrackID", "Bodypoint"]] = df[["Frame", "TrackID", "Bodypoint"]].astype(np.int64)
    if int(frame_step) > 1:
        df = df[((df["Frame"] - int(frame_start)) % int(frame_step)) == 0].copy()
    return df.sort_values(["Frame", "TrackID", "Bodypoint"], kind="mergesort").reset_index(drop=True)


def frame_track_arrays(frame_df: pd.DataFrame, antenna_bodypoints: tuple[int, ...]) -> dict[int, dict]:
    antenna_bodypoints_array = np.array([int(bp) for bp in antenna_bodypoints], dtype=np.int64)
    tracks = {}
    for track_id, group in frame_df.groupby("TrackID", sort=False):
        anchor = group[["TrackX", "TrackY"]].dropna()
        if anchor.empty:
            continue
        bodypoints = group["Bodypoint"].to_numpy(np.int64, copy=False)
        xy = group[["X", "Y"]].to_numpy(np.float64, copy=True)
        if xy.size == 0:
            continue
        tracks[int(track_id)] = {
            "center": anchor.iloc[0].to_numpy(np.float64),
            "body_xy": xy,
            "antenna_xy": xy[np.isin(bodypoints, antenna_bodypoints_array)],
        }
    return tracks


def min_antenna_to_body_distance_sq(antenna_track: dict, body_track: dict) -> float:
    antenna_xy = antenna_track["antenna_xy"]
    body_xy = body_track["body_xy"]
    if antenna_xy.size == 0 or body_xy.size == 0:
        return np.inf
    delta = antenna_xy[:, None, :] - body_xy[None, :, :]
    distance_sq = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
    return float(np.min(distance_sq))


def detect_interactions(
    tracks_df: pd.DataFrame,
    *,
    interaction_radius_px: float,
    micro_distance_px: float,
    antenna_bodypoints: tuple[int, ...],
    progress_every_frames: int,
    run_start_time: float,
    processed_frames_before: int,
    interactions_before: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interaction_rows = []
    summary_rows = []
    interaction_radius_sq = float(interaction_radius_px) * float(interaction_radius_px)
    micro_distance_sq = float(micro_distance_px) * float(micro_distance_px)

    grouped = tracks_df.groupby("Frame", sort=True)
    n_frames = len(grouped)
    for i, (frame, frame_df) in enumerate(grouped, start=1):
        tracks = frame_track_arrays(frame_df, antenna_bodypoints)
        track_ids = np.array(sorted(tracks), dtype=np.int64)
        if len(track_ids) < 2:
            summary_rows.append(
                {"Frame": int(frame), "n_tracks": int(len(track_ids)), "n_candidate_pairs": 0, "n_interactions": 0}
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
            continue

        centers = np.stack([tracks[int(track_id)]["center"] for track_id in track_ids], axis=0)
        delta = centers[:, None, :] - centers[None, :, :]
        distance_sq = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
        pair_i, pair_j = np.where(np.triu(distance_sq <= interaction_radius_sq, k=1))

        n_interacting = 0
        for idx_i, idx_j in zip(pair_i, pair_j):
            tid_a = int(track_ids[idx_i])
            tid_b = int(track_ids[idx_j])
            for antenna_track_id, body_track_id in ((tid_a, tid_b), (tid_b, tid_a)):
                if min_antenna_to_body_distance_sq(tracks[antenna_track_id], tracks[body_track_id]) <= micro_distance_sq:
                    n_interacting += 1
                    interaction_rows.append(
                        {
                            "Frame": int(frame),
                            "antenna_track_id": int(antenna_track_id),
                            "body_track_id": int(body_track_id),
                        }
                    )

        summary_rows.append(
            {
                "Frame": int(frame),
                "n_tracks": int(len(track_ids)),
                "n_candidate_pairs": int(len(pair_i)),
                "n_interactions": int(n_interacting),
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

    interactions = pd.DataFrame(interaction_rows, columns=["Frame", "antenna_track_id", "body_track_id"])
    return interactions, pd.DataFrame(summary_rows)


def interaction_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("Frame", pa.int64()),
            ("antenna_track_id", pa.int64()),
            ("body_track_id", pa.int64()),
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
            interactions[["Frame", "antenna_track_id", "body_track_id"]].astype("int64"),
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
    if skip_existing and interactions_path.exists():
        print(f"Skipping existing {interactions_path}", flush=True)
        return interactions_path

    interactions_path.parent.mkdir(parents=True, exist_ok=True)
    if interactions_path.exists():
        interactions_path.unlink()

    chunk_min_frame, chunk_max_frame = parquet_frame_bounds(chunk_file)
    start = max(int(frame_start), int(chunk_min_frame))
    chunk_stop = int(chunk_max_frame) + 1
    stop = chunk_stop if max_frames is None else min(start + int(max_frames), chunk_stop)
    windows = frame_windows(start, stop, int(frame_batch_size))

    interaction_radius_px = float(interaction_radius_mm) / float(mm_per_px)
    micro_distance_px = float(micro_interaction_distance_mm) / float(mm_per_px)

    print(f"Chunk: {chunk_file}", flush=True)
    print(f"Output file: {interactions_path}", flush=True)
    print(f"Processing frames: {start}-{stop - 1} in {len(windows)} windows", flush=True)
    print(f"Interaction radius: {interaction_radius_mm:g} mm = {interaction_radius_px:.1f} track units", flush=True)
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
            interaction_radius_px=interaction_radius_px,
            micro_distance_px=micro_distance_px,
            antenna_bodypoints=antenna_bodypoints,
            progress_every_frames=int(progress_every_frames),
            run_start_time=run_start,
            processed_frames_before=processed_frames,
            interactions_before=total_interactions,
        )
        if not interactions.empty:
            writer = append_interactions_parquet(writer, interactions, interactions_path)

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
            pd.DataFrame(columns=["Frame", "antenna_track_id", "body_track_id"]),
            interactions_path,
            force_write=True,
        )
    writer.close()

    elapsed = time.perf_counter() - run_start
    print(
        f"Finished chunk={chunk_file.name}: "
        f"{processed_frames:,} frames, {total_interactions:,} directed interactions, "
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
    parser.add_argument("--interaction_radius_mm", type=float, default=8.0)
    parser.add_argument("--micro_interaction_distance_mm", type=float, default=1.0)
    parser.add_argument("--antenna_bodypoint", action="append", type=int, default=None)
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
