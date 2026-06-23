# %%
# Lightweight ant-ant interaction test for one chunk.
#
# This uses the stitched track coordinate space directly:
# - candidate ant pairs are selected by TrackX/TrackY distance
# - micro-interactions are antenna bodypoints from one ant near any X/Y bodypoint
#   from the other ant
# - debug images are blank-coordinate plots in the same units used for detection

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = print


# %%
# Edit these settings first.
CHUNK_PATH = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/tracks/"
    "20260515_142047_chunk000_left.parquet"
)
OUT_ROOT = CHUNK_PATH.parents[1] / "interaction_debug" / CHUNK_PATH.stem

FRAME_START = 0
MAX_FRAMES = None  # None means process through the end of the chunk.
FRAME_STEP = 1
FRAME_BATCH_SIZE = 3000
PROGRESS_EVERY_FRAMES = 500

MM_PER_PX = 0.016
INTERACTION_RADIUS_MM = 8.0
MICRO_INTERACTION_DISTANCE_MM = 1.0

# The parquet does not preserve node names. These defaults are the two longer
# skeleton branches from the local viewer skeleton; tune from labeled debug PNGs.
ANTENNA_BODYPOINTS = (4, 5, 6, 7, 8, 9)
SKELETON_EDGES = (
    (0, 1),
    (0, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
)

ENABLE_DEBUG_PNGS = False
DEBUG_FRAME_COUNT = 12
DEBUG_FRAMES = None  # Example: [0, 10, 100]. If None, use frames with interactions.
DEBUG_CONTEXT_RADIUS_MM = 20.0
DEBUG_DPI = 180


# %%
def load_chunk_window(
    path: Path,
    *,
    frame_start: int,
    max_frames: int,
    frame_step: int,
) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    frame_stop = int(frame_start) + int(max_frames)
    columns = pq.ParquetFile(path).schema.names
    required = {"Frame", "TrackID", "Bodypoint", "X", "Y", "TrackX", "TrackY"}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    table = ds.dataset(path, format="parquet").to_table(
        columns=sorted(required),
        filter=(pc.field("Frame") >= int(frame_start)) & (pc.field("Frame") < int(frame_stop)),
        use_threads=True,
    )
    df = table.to_pandas()
    for col in ["Frame", "TrackID", "Bodypoint"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["X", "Y", "TrackX", "TrackY"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Frame", "TrackID", "Bodypoint", "X", "Y", "TrackX", "TrackY"]).copy()
    df[["Frame", "TrackID", "Bodypoint"]] = df[["Frame", "TrackID", "Bodypoint"]].astype(np.int64)
    if int(frame_step) > 1:
        df = df[((df["Frame"] - int(frame_start)) % int(frame_step)) == 0].copy()
    return df.sort_values(["Frame", "TrackID", "Bodypoint"], kind="mergesort").reset_index(drop=True)


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
        raise ValueError("FRAME_BATCH_SIZE must be positive")
    return [
        (start, min(start + int(batch_size), int(frame_stop)))
        for start in range(int(frame_start), int(frame_stop), int(batch_size))
    ]


def append_interactions_parquet(writer, interactions: pd.DataFrame, path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("Frame", pa.int64()),
            ("antenna_track_id", pa.int64()),
            ("body_track_id", pa.int64()),
        ]
    )
    if writer is None:
        writer = pq.ParquetWriter(path, schema=schema, compression="zstd")

    if not interactions.empty:
        table = pa.Table.from_pandas(
            interactions[["Frame", "antenna_track_id", "body_track_id"]].astype("int64"),
            schema=schema,
            preserve_index=False,
        )
        writer.write_table(table)
    return writer


def frame_track_dict(frame_df: pd.DataFrame) -> dict[int, dict]:
    tracks = {}
    for track_id, group in frame_df.groupby("TrackID", sort=False):
        group = group.sort_values("Bodypoint", kind="mergesort")
        anchor = group[["TrackX", "TrackY"]].dropna()
        if anchor.empty:
            continue
        tracks[int(track_id)] = {
            "center": anchor.iloc[0].to_numpy(np.float64),
            "bodypoints": {
                int(row.Bodypoint): np.array([float(row.X), float(row.Y)], dtype=np.float64)
                for row in group.itertuples(index=False)
                if np.isfinite(row.X) and np.isfinite(row.Y)
            },
        }
    return tracks


def frame_track_arrays(frame_df: pd.DataFrame, antenna_bodypoints: tuple[int, ...]) -> dict[int, dict]:
    antenna_bodypoints = np.array([int(bp) for bp in antenna_bodypoints], dtype=np.int64)
    tracks = {}
    for track_id, group in frame_df.groupby("TrackID", sort=False):
        anchor = group[["TrackX", "TrackY"]].dropna()
        if anchor.empty:
            continue
        bodypoints = group["Bodypoint"].to_numpy(np.int64, copy=False)
        xy = group[["X", "Y"]].to_numpy(np.float64, copy=True)
        if xy.size == 0:
            continue
        antenna_mask = np.isin(bodypoints, antenna_bodypoints)
        tracks[int(track_id)] = {
            "center": anchor.iloc[0].to_numpy(np.float64),
            "body_xy": xy,
            "antenna_xy": xy[antenna_mask],
        }
    return tracks


def nearest_antenna_to_body_contact_fast(
    antenna_ant: dict,
    body_ant: dict,
    *,
    collect_points: bool = False,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    antenna_xy = antenna_ant["antenna_xy"]
    body_xy = body_ant["body_xy"]
    if antenna_xy.size == 0 or body_xy.size == 0:
        return np.inf, None, None

    delta = antenna_xy[:, None, :] - body_xy[None, :, :]
    distance_sq = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
    flat_idx = int(np.argmin(distance_sq))
    best_distance_sq = float(distance_sq.reshape(-1)[flat_idx])
    if not collect_points:
        return best_distance_sq, None, None

    antenna_idx, body_idx = np.unravel_index(flat_idx, distance_sq.shape)
    return best_distance_sq, antenna_xy[antenna_idx].copy(), body_xy[body_idx].copy()


def detect_interactions(
    tracks_df: pd.DataFrame,
    *,
    interaction_radius_px: float,
    micro_distance_px: float,
    antenna_bodypoints: tuple[int, ...],
    collect_debug: bool = False,
    progress_every_frames: int = 500,
    run_start_time: float | None = None,
    processed_frames_before: int = 0,
    interactions_before: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interaction_rows = []
    debug_rows = []
    summary_rows = []
    grouped = tracks_df.groupby("Frame", sort=True)
    n_frames = len(grouped)
    run_start_time = time.perf_counter() if run_start_time is None else float(run_start_time)
    interaction_radius_sq = float(interaction_radius_px) * float(interaction_radius_px)
    micro_distance_sq = float(micro_distance_px) * float(micro_distance_px)

    for i, (frame, frame_df) in enumerate(grouped, start=1):
        tracks = frame_track_arrays(frame_df, antenna_bodypoints)
        track_ids = np.array(sorted(tracks), dtype=np.int64)
        if len(track_ids) < 2:
            summary_rows.append(
                {"Frame": int(frame), "n_tracks": len(track_ids), "n_candidate_pairs": 0, "n_interactions": 0}
            )
            if i == 1 or i == n_frames or (progress_every_frames and i % int(progress_every_frames) == 0):
                elapsed = max(time.perf_counter() - run_start_time, 1e-9)
                total_frames = int(processed_frames_before) + i
                print(
                    f"frame={int(frame)} total_frames={total_frames} "
                    f"elapsed={elapsed:.1f}s speed={total_frames / elapsed:.2f} frames/s "
                    f"interactions={int(interactions_before) + len(interaction_rows)}"
                )
            continue

        centers = np.stack([tracks[int(tid)]["center"] for tid in track_ids], axis=0)
        delta = centers[:, None, :] - centers[None, :, :]
        distance_sq = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
        pair_i, pair_j = np.where(np.triu(distance_sq <= interaction_radius_sq, k=1))

        n_interacting = 0
        for idx_i, idx_j in zip(pair_i, pair_j):
            tid_a = int(track_ids[idx_i])
            tid_b = int(track_ids[idx_j])
            for antenna_track_id, body_track_id in ((tid_a, tid_b), (tid_b, tid_a)):
                min_distance_sq, antenna_xy, body_xy = nearest_antenna_to_body_contact_fast(
                    tracks[antenna_track_id],
                    tracks[body_track_id],
                    collect_points=bool(collect_debug),
                )
                if min_distance_sq <= micro_distance_sq:
                    n_interacting += 1
                    interaction_rows.append(
                        {
                            "Frame": int(frame),
                            "antenna_track_id": int(antenna_track_id),
                            "body_track_id": int(body_track_id),
                        }
                    )
                    if collect_debug:
                        debug_rows.append(
                            {
                                "Frame": int(frame),
                                "antenna_track_id": int(antenna_track_id),
                                "body_track_id": int(body_track_id),
                                "min_distance_mm": float(np.sqrt(min_distance_sq)) * MM_PER_PX,
                                "antenna_x": float(antenna_xy[0]) if antenna_xy is not None else np.nan,
                                "antenna_y": float(antenna_xy[1]) if antenna_xy is not None else np.nan,
                                "body_x": float(body_xy[0]) if body_xy is not None else np.nan,
                                "body_y": float(body_xy[1]) if body_xy is not None else np.nan,
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
        if i == 1 or i == n_frames or (progress_every_frames and i % int(progress_every_frames) == 0):
            elapsed = max(time.perf_counter() - run_start_time, 1e-9)
            total_frames = int(processed_frames_before) + i
            print(
                f"frame={int(frame)} total_frames={total_frames} "
                f"elapsed={elapsed:.1f}s speed={total_frames / elapsed:.2f} frames/s "
                f"interactions={int(interactions_before) + len(interaction_rows)}"
            )

    interactions = pd.DataFrame(interaction_rows, columns=["Frame", "antenna_track_id", "body_track_id"])
    debug_interactions = pd.DataFrame(
        debug_rows,
        columns=[
            "Frame",
            "antenna_track_id",
            "body_track_id",
            "min_distance_mm",
            "antenna_x",
            "antenna_y",
            "body_x",
            "body_y",
        ],
    )
    return interactions, debug_interactions, pd.DataFrame(summary_rows)


def color_for_track(track_id: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(int(track_id) + 31)
    return tuple((rng.integers(40, 230, size=3) / 255.0).tolist())


def choose_debug_frames(summary: pd.DataFrame, *, debug_frames, debug_frame_count: int) -> list[int]:
    if debug_frames is not None:
        return [int(frame) for frame in debug_frames]
    interacting = summary[summary["n_interactions"] > 0]["Frame"].drop_duplicates().head(int(debug_frame_count)).to_list()
    if interacting:
        return [int(frame) for frame in interacting]
    return [int(frame) for frame in summary["Frame"].drop_duplicates().head(int(debug_frame_count)).to_list()]


def draw_debug_frame(
    frame_df: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    frame: int,
    out_path: Path,
    interaction_radius_px: float,
    micro_distance_px: float,
    antenna_bodypoints: tuple[int, ...],
    context_radius_px: float,
) -> None:
    tracks = frame_track_dict(frame_df)
    interacting_ids = (
        set(interactions["antenna_track_id"].to_list()) | set(interactions["body_track_id"].to_list())
        if not interactions.empty
        else set()
    )
    if interactions.empty:
        centers = np.stack([track["center"] for track in tracks.values()], axis=0)
        plot_center = centers.mean(axis=0)
    else:
        pts = interactions[["antenna_x", "antenna_y", "body_x", "body_y"]].to_numpy(np.float64).reshape(-1, 2)
        plot_center = np.nanmean(pts, axis=0)

    x0, x1 = plot_center[0] - context_radius_px, plot_center[0] + context_radius_px
    y0, y1 = plot_center[1] - context_radius_px, plot_center[1] + context_radius_px

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")

    for tid, track in tracks.items():
        nodes = track["bodypoints"]
        if not nodes:
            continue
        color = color_for_track(tid)
        is_focus = tid in interacting_ids
        lw = 2.2 if is_focus else 0.8
        alpha = 1.0 if is_focus else 0.35
        for a, b in SKELETON_EDGES:
            if a in nodes and b in nodes:
                xy = np.vstack([nodes[a], nodes[b]])
                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, alpha=alpha, zorder=2)
        for bp, xy in nodes.items():
            is_antenna = int(bp) in antenna_bodypoints
            ax.scatter(
                xy[0],
                xy[1],
                s=40 if is_focus or is_antenna else 18,
                color="crimson" if is_antenna else color,
                edgecolor="black" if is_focus or is_antenna else "none",
                linewidth=0.5,
                alpha=alpha,
                zorder=3,
            )
            ax.text(
                xy[0] + 5,
                xy[1] + 5,
                str(bp),
                fontsize=6,
                color="black",
                alpha=0.95 if is_focus else 0.55,
                zorder=4,
            )
        center = track["center"]
        ax.scatter(center[0], center[1], s=55, marker="x", color=color, linewidth=1.5, zorder=4)
        ax.text(center[0] + 8, center[1] - 8, f"T{tid}", fontsize=8, color=color, weight="bold", zorder=5)
        if is_focus:
            ax.add_patch(plt.Circle(center, interaction_radius_px, fill=False, color=color, lw=1.0, alpha=0.35, zorder=1))

    for row in interactions.itertuples(index=False):
        antenna_xy = np.array([row.antenna_x, row.antenna_y], dtype=float)
        other_xy = np.array([row.body_x, row.body_y], dtype=float)
        ax.plot(
            [antenna_xy[0], other_xy[0]],
            [antenna_xy[1], other_xy[1]],
            color="red",
            lw=2.5,
            alpha=0.9,
            zorder=6,
        )
        ax.add_patch(plt.Circle(antenna_xy, micro_distance_px, fill=False, color="red", lw=1.5, alpha=0.85, zorder=5))
        ax.text(
            np.nanmean([antenna_xy[0], other_xy[0]]),
            np.nanmean([antenna_xy[1], other_xy[1]]),
            f"{row.antenna_track_id}->{row.body_track_id} {row.min_distance_mm:.2f} mm",
            fontsize=8,
            color="red",
            weight="bold",
            zorder=7,
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_title(
        f"{CHUNK_PATH.name} frame {frame} | "
        f"radius={interaction_radius_px * MM_PER_PX:.1f} mm, micro={micro_distance_px * MM_PER_PX:.2f} mm"
    )
    ax.set_xlabel("Track coordinate X")
    ax.set_ylabel("Track coordinate Y")
    ax.grid(True, color="0.9", lw=0.5)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DEBUG_DPI)
    plt.close(fig)


def write_metadata(
    path: Path,
    *,
    frame_stop: int,
    interaction_radius_px: float,
    micro_distance_px: float,
    elapsed_seconds: float,
    processed_frames: int,
    total_interactions: int,
) -> None:
    path.write_text(
        json.dumps(
            {
                "chunk_path": str(CHUNK_PATH),
                "frame_start": int(FRAME_START),
                "frame_stop_exclusive": int(frame_stop),
                "max_frames": None if MAX_FRAMES is None else int(MAX_FRAMES),
                "frame_step": int(FRAME_STEP),
                "frame_batch_size": int(FRAME_BATCH_SIZE),
                "progress_every_frames": int(PROGRESS_EVERY_FRAMES),
                "mm_per_px": float(MM_PER_PX),
                "interaction_radius_mm": float(INTERACTION_RADIUS_MM),
                "interaction_radius_px": float(interaction_radius_px),
                "micro_interaction_distance_mm": float(MICRO_INTERACTION_DISTANCE_MM),
                "micro_interaction_distance_px": float(micro_distance_px),
                "antenna_bodypoints": [int(bp) for bp in ANTENNA_BODYPOINTS],
                "skeleton_edges": [[int(a), int(b)] for a, b in SKELETON_EDGES],
                "debug_pngs_enabled": bool(ENABLE_DEBUG_PNGS),
                "elapsed_seconds": float(elapsed_seconds),
                "processed_frames": int(processed_frames),
                "total_interactions": int(total_interactions),
                "frames_per_second": float(processed_frames / elapsed_seconds) if elapsed_seconds > 0 else None,
            },
            indent=2,
        )
        + "\n"
    )


def process_chunk() -> tuple[pd.DataFrame, Path, Path]:
    interaction_radius_px = float(INTERACTION_RADIUS_MM) / float(MM_PER_PX)
    micro_distance_px = float(MICRO_INTERACTION_DISTANCE_MM) / float(MM_PER_PX)

    chunk_min_frame, chunk_max_frame = parquet_frame_bounds(CHUNK_PATH)
    frame_start = max(int(FRAME_START), int(chunk_min_frame))
    chunk_stop = int(chunk_max_frame) + 1
    frame_stop = chunk_stop if MAX_FRAMES is None else min(frame_start + int(MAX_FRAMES), chunk_stop)
    windows = frame_windows(frame_start, frame_stop, int(FRAME_BATCH_SIZE))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    interactions_path = OUT_ROOT / "interactions.parquet"
    summary_path = OUT_ROOT / "frame_summary.csv"
    metadata_path = OUT_ROOT / "metadata.json"
    old_verbose_csv = OUT_ROOT / "interactions.csv"
    if old_verbose_csv.exists():
        old_verbose_csv.unlink()
    if interactions_path.exists():
        interactions_path.unlink()
    if summary_path.exists():
        summary_path.unlink()

    print(f"Chunk: {CHUNK_PATH}")
    print(f"Chunk frame bounds: {chunk_min_frame}-{chunk_max_frame}")
    print(f"Processing frames: {frame_start}-{frame_stop - 1} in {len(windows)} windows")
    print(f"Interaction radius: {INTERACTION_RADIUS_MM:g} mm = {interaction_radius_px:.1f} track units")
    print(f"Micro-interaction distance: {MICRO_INTERACTION_DISTANCE_MM:g} mm = {micro_distance_px:.1f} track units")
    print(f"Debug PNGs enabled: {ENABLE_DEBUG_PNGS}")

    run_start = time.perf_counter()
    writer = None
    summary_chunks = []
    processed_frames = 0
    total_interactions = 0

    for batch_idx, (window_start, window_stop) in enumerate(windows, start=1):
        read_start = time.perf_counter()
        print(f"read window {batch_idx}/{len(windows)} frames={window_start}-{window_stop - 1}")
        chunk_tracks = load_chunk_window(
            CHUNK_PATH,
            frame_start=window_start,
            max_frames=window_stop - window_start,
            frame_step=FRAME_STEP,
        )
        read_elapsed = time.perf_counter() - read_start
        if chunk_tracks.empty:
            print(f"window {batch_idx}: no rows read in {read_elapsed:.1f}s")
            continue
        print(
            f"window {batch_idx}: loaded {len(chunk_tracks):,} rows, "
            f"{chunk_tracks['Frame'].nunique()} frames, "
            f"{chunk_tracks['TrackID'].nunique()} tracks in {read_elapsed:.1f}s"
        )

        interactions, _debug_interactions, frame_summary = detect_interactions(
            chunk_tracks,
            interaction_radius_px=interaction_radius_px,
            micro_distance_px=micro_distance_px,
            antenna_bodypoints=tuple(int(bp) for bp in ANTENNA_BODYPOINTS),
            collect_debug=bool(ENABLE_DEBUG_PNGS),
            progress_every_frames=int(PROGRESS_EVERY_FRAMES),
            run_start_time=run_start,
            processed_frames_before=processed_frames,
            interactions_before=total_interactions,
        )

        writer = append_interactions_parquet(writer, interactions, interactions_path)
        frame_summary.to_csv(
            summary_path,
            mode="a",
            header=not summary_path.exists(),
            index=False,
        )
        summary_chunks.append(frame_summary)
        processed_frames += int(len(frame_summary))
        total_interactions += int(len(interactions))
        elapsed = time.perf_counter() - run_start
        print(
            f"window {batch_idx} done: total_frames={processed_frames}, "
            f"total_interactions={total_interactions:,}, "
            f"elapsed={elapsed:.1f}s, speed={processed_frames / max(elapsed, 1e-9):.2f} frames/s"
        )

        del chunk_tracks, interactions, _debug_interactions, frame_summary

    if writer is None:
        writer = append_interactions_parquet(
            writer,
            pd.DataFrame(columns=["Frame", "antenna_track_id", "body_track_id"]),
            interactions_path,
        )
    writer.close()

    elapsed = time.perf_counter() - run_start
    write_metadata(
        metadata_path,
        frame_stop=frame_stop,
        interaction_radius_px=interaction_radius_px,
        micro_distance_px=micro_distance_px,
        elapsed_seconds=elapsed,
        processed_frames=processed_frames,
        total_interactions=total_interactions,
    )

    frame_summary_all = pd.concat(summary_chunks, ignore_index=True) if summary_chunks else pd.DataFrame()
    print(f"Wrote {interactions_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metadata_path}")
    print(
        f"Finished: {processed_frames:,} frames, {total_interactions:,} directed interactions, "
        f"{elapsed:.1f}s, {processed_frames / max(elapsed, 1e-9):.2f} frames/s"
    )
    return frame_summary_all, interactions_path, summary_path


# %%
# Run the full-chunk interaction pass.
frame_summary, interactions_path, summary_path = process_chunk()
display(frame_summary.describe())
