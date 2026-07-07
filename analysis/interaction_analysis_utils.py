"""Utilities for interactive directed ant-ant interaction analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIDES = ("left", "right")


def _save_current_figures_if_enabled() -> None:
    try:
        from analysis.figure_saving import save_new_figures

        save_new_figures(plt)
    except Exception as exc:
        print(f"Warning: failed to save interaction analysis figure: {exc}")


@dataclass(frozen=True)
class InteractionChunk:
    interaction_path: Path
    track_path: Path
    chunk: str
    side: str
    recording_start_clock_seconds: int
    chunk_start_clock_seconds: int
    chunk_global_frame_offset: int
    chunk_frame_count: int
    chunk_stride_frames: int


def chunk_label(chunk: int | str) -> str:
    text = str(chunk)
    if text.startswith("chunk"):
        text = text.removeprefix("chunk")
    return text.zfill(3)


def parse_side(path: Path) -> str:
    match = re.search(r"_(left|right)(?:\.parquet)?$", path.name)
    if match is None:
        raise ValueError(f"Could not infer side from {path}")
    return match.group(1)


def parse_clock_seconds_from_name(name: str) -> int:
    match = re.search(r"_(\d{6})(?:_|$)", name)
    if match is None:
        raise ValueError(f"Could not parse HHMMSS start time from {name}")
    clock = match.group(1)
    hour = int(clock[0:2])
    minute = int(clock[2:4])
    second = int(clock[4:6])
    return hour * 3600 + minute * 60 + second


def format_clock_time(clock_seconds: float) -> str:
    seconds = int(round(float(clock_seconds))) % (24 * 3600)
    hour = seconds // 3600
    minute = (seconds % 3600) // 60
    second = seconds % 60
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def find_chunk_file(root: Path, *, chunk: int | str, side: str) -> Path:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    chunk_id = chunk_label(chunk)
    matches = sorted(Path(root).glob(f"*_chunk{chunk_id}_{side}.parquet"))
    if not matches:
        raise FileNotFoundError(f"No chunk{chunk_id} {side} parquet found in {root}")
    if len(matches) > 1:
        raise ValueError(f"Multiple chunk{chunk_id} {side} parquets found in {root}: {matches}")
    return matches[0]


def available_chunks(interaction_root: Path, *, side: str) -> list[str]:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    pattern = re.compile(rf"_chunk(\d{{3}})_{re.escape(side)}\.parquet$")
    chunks = []
    for path in sorted(Path(interaction_root).glob(f"*_chunk???_{side}.parquet")):
        match = pattern.search(path.name)
        if match is not None:
            chunks.append(match.group(1))
    if not chunks:
        raise FileNotFoundError(f"No interaction chunk parquets found in {interaction_root} for side={side!r}")
    return sorted(set(chunks))


def normalize_chunk_selection(
    chunks: str | int | list[str | int] | tuple[str | int, ...],
    interaction_root: Path,
    *,
    side: str,
    max_chunks: int | None = None,
) -> list[str]:
    if isinstance(chunks, str) and chunks.lower() == "all":
        labels = available_chunks(interaction_root, side=side)
    elif isinstance(chunks, (str, int)):
        labels = [chunk_label(chunks)]
    else:
        labels = [chunk_label(chunk) for chunk in chunks]

    ordered = []
    seen = set()
    for label in labels:
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    if max_chunks is not None:
        ordered = ordered[: int(max_chunks)]
    return ordered


def parquet_num_frames(path: Path) -> int | None:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    raw = metadata.get(b"num_frames")
    if raw is None:
        return None
    try:
        value = int(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def chunk_frame_count(path: Path) -> int:
    count = parquet_num_frames(path)
    if count is not None:
        return int(count)
    frame_min, frame_max = frame_bounds(path)
    return int(frame_max - frame_min + 1)


def infer_chunk_stride_frames(tracks_root: Path, *, side: str) -> int:
    paths = sorted(Path(tracks_root).glob(f"*_chunk???_{side}.parquet"))
    if not paths:
        raise FileNotFoundError(f"No track chunk parquets found in {tracks_root} for side={side!r}")

    counts = []
    for path in paths:
        count = parquet_num_frames(path)
        if count is not None:
            counts.append(int(count))
    if counts:
        return int(max(counts))
    return chunk_frame_count(paths[0])


def frame_bounds(path: Path) -> tuple[int, int]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    names = parquet_file.schema.names
    frame_idx = names.index("Frame")
    mins = []
    maxs = []
    for row_group_idx in range(parquet_file.metadata.num_row_groups):
        stats = parquet_file.metadata.row_group(row_group_idx).column(frame_idx).statistics
        if stats is not None:
            mins.append(int(stats.min))
            maxs.append(int(stats.max))
    if mins and maxs:
        return min(mins), max(maxs)

    frame = pd.read_parquet(path, columns=["Frame"])["Frame"]
    return int(frame.min()), int(frame.max())


def resolve_chunk(
    interaction_root: Path,
    tracks_root: Path,
    *,
    chunk: int | str,
    side: str,
    fps: float,
    chunk_stride_frames: int | None = None,
) -> InteractionChunk:
    interaction_path = find_chunk_file(interaction_root, chunk=chunk, side=side)
    track_path = find_chunk_file(tracks_root, chunk=chunk, side=side)
    chunk_id = chunk_label(chunk)
    recording_start = parse_clock_seconds_from_name(interaction_path.name)
    chunk_frame_count_value = chunk_frame_count(track_path)
    chunk_stride_value = int(chunk_stride_frames) if chunk_stride_frames is not None else int(chunk_frame_count_value)
    chunk_global_frame_offset = int(chunk_id) * chunk_stride_value
    chunk_start = int(round(recording_start + chunk_global_frame_offset / float(fps))) % (24 * 3600)
    return InteractionChunk(
        interaction_path=interaction_path,
        track_path=track_path,
        chunk=chunk_id,
        side=side,
        recording_start_clock_seconds=recording_start,
        chunk_start_clock_seconds=chunk_start,
        chunk_global_frame_offset=int(chunk_global_frame_offset),
        chunk_frame_count=int(chunk_frame_count_value),
        chunk_stride_frames=int(chunk_stride_value),
    )


def resolve_chunks(
    interaction_root: Path,
    tracks_root: Path,
    *,
    chunks: str | int | list[str | int] | tuple[str | int, ...],
    side: str,
    fps: float,
    max_chunks: int | None = None,
) -> list[InteractionChunk]:
    labels = normalize_chunk_selection(chunks, interaction_root, side=side, max_chunks=max_chunks)
    stride = infer_chunk_stride_frames(tracks_root, side=side)
    return [
        resolve_chunk(
            interaction_root,
            tracks_root,
            chunk=label,
            side=side,
            fps=fps,
            chunk_stride_frames=stride,
        )
        for label in labels
    ]


def describe_chunks(chunks: list[InteractionChunk]) -> str:
    if not chunks:
        return "no chunks"
    if len(chunks) == 1:
        return f"chunk{chunks[0].chunk}"
    return f"{len(chunks)} chunks {chunks[0].chunk}-{chunks[-1].chunk}"


def file_fingerprint(path: Path) -> dict[str, object]:
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def interaction_analysis_cache_key(chunks: list[InteractionChunk], settings: dict[str, object]) -> str:
    payload = {
        "settings": settings,
        "chunks": [
            {
                "chunk": chunk.chunk,
                "side": chunk.side,
                "interaction": file_fingerprint(chunk.interaction_path),
                "track": file_fingerprint(chunk.track_path),
                "frame_offset": int(chunk.chunk_global_frame_offset),
                "n_frames": int(chunk.chunk_frame_count),
            }
            for chunk in chunks
        ],
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def parquet_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object:
            continue
        sample = out[col].dropna().head(1000)
        if sample.empty:
            continue
        sample_types = {type(value) for value in sample}
        needs_string = len(sample_types) > 1 or any(isinstance(value, Path) for value in sample)
        if needs_string:
            out[col] = out[col].map(lambda value: None if pd.isna(value) else str(value))
    return out


def load_or_build_table(
    path: Path,
    builder,
    *,
    use_cache: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    path = Path(path)
    if use_cache and not force and path.exists():
        print(f"cache hit: {path}")
        return pd.read_parquet(path)

    print(f"cache build: {path}")
    table = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = parquet_safe_dataframe(table)
    table.to_parquet(path, index=False)
    return table


def load_or_build_pickle(
    path: Path,
    builder,
    *,
    use_cache: bool = True,
    force: bool = False,
):
    path = Path(path)
    if use_cache and not force and path.exists():
        print(f"cache hit: {path}")
        with path.open("rb") as f:
            return pickle.load(f)

    print(f"cache build: {path}")
    value = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    return value


def load_interactions(path: Path, *, max_interactions: int | None = None, sample: bool = False, random_state: int = 0) -> pd.DataFrame:
    interactions = pd.read_parquet(path, columns=["Frame", "antenna_track_id", "body_track_id"])
    for col in ["Frame", "antenna_track_id", "body_track_id"]:
        interactions[col] = pd.to_numeric(interactions[col], errors="coerce").astype("Int64")
    interactions = interactions.dropna().astype(
        {"Frame": "int64", "antenna_track_id": "int64", "body_track_id": "int64"}
    )
    if max_interactions is not None and len(interactions) > int(max_interactions):
        if sample:
            interactions = interactions.sample(n=int(max_interactions), random_state=int(random_state))
            interactions = interactions.sort_values("Frame", kind="mergesort")
        else:
            interactions = interactions.iloc[: int(max_interactions)].copy()
    return interactions.reset_index(drop=True)


def load_interactions_for_chunks(
    chunks: list[InteractionChunk],
    *,
    max_interactions_per_chunk: int | None = None,
    sample: bool = False,
    random_state: int = 0,
) -> pd.DataFrame:
    frames = []
    for chunk in chunks:
        interactions = load_interactions(
            chunk.interaction_path,
            max_interactions=max_interactions_per_chunk,
            sample=sample,
            random_state=random_state,
        )
        interactions["chunk"] = chunk.chunk
        interactions["side"] = chunk.side
        interactions["chunk_global_frame_offset"] = int(chunk.chunk_global_frame_offset)
        interactions["chunk_start_clock_seconds"] = int(chunk.chunk_start_clock_seconds)
        interactions["global_frame"] = interactions["Frame"].astype(np.int64) + int(chunk.chunk_global_frame_offset)
        frames.append(interactions)
    if not frames:
        return pd.DataFrame(
            columns=[
                "Frame",
                "antenna_track_id",
                "body_track_id",
                "chunk",
                "side",
                "chunk_global_frame_offset",
                "chunk_start_clock_seconds",
                "global_frame",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def load_track_positions(path: Path, *, bodypoint: int = 0) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    required = {"Frame", "TrackID", "Bodypoint", "TrackX", "TrackY"}
    columns = set(pq.ParquetFile(path).schema.names)
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    table = ds.dataset(path, format="parquet").to_table(
        columns=["Frame", "TrackID", "Bodypoint", "TrackX", "TrackY"],
        filter=pc.field("Bodypoint") == int(bodypoint),
        use_threads=True,
    )
    positions = table.to_pandas()
    positions = positions.rename(columns={"TrackX": "x_px", "TrackY": "y_px"})
    for col in ["Frame", "TrackID"]:
        positions[col] = pd.to_numeric(positions[col], errors="coerce").astype("Int64")
    positions["x_px"] = pd.to_numeric(positions["x_px"], errors="coerce")
    positions["y_px"] = pd.to_numeric(positions["y_px"], errors="coerce")
    positions = positions.dropna(subset=["Frame", "TrackID", "x_px", "y_px"])
    positions = positions.astype({"Frame": "int64", "TrackID": "int64"})
    if positions.duplicated(["Frame", "TrackID"]).any():
        positions = positions.groupby(["Frame", "TrackID"], sort=True, as_index=False).agg(
            x_px=("x_px", "mean"),
            y_px=("y_px", "mean"),
        )
    else:
        positions = positions[["Frame", "TrackID", "x_px", "y_px"]]
    return positions.sort_values(["Frame", "TrackID"], kind="mergesort").reset_index(drop=True)


def load_cluster_table(path: Path, *, side: str | None = None) -> pd.DataFrame:
    clusters = pd.read_csv(path)
    required = {"TrackID", "side", "cluster_id", "leiden_cluster_id"}
    missing = required.difference(clusters.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    clusters = clusters.copy()
    clusters["TrackID"] = pd.to_numeric(clusters["TrackID"], errors="coerce").astype("Int64")
    clusters = clusters.dropna(subset=["TrackID"]).astype({"TrackID": "int64"})
    if side is not None:
        clusters = clusters[clusters["side"] == side].copy()
    return clusters.reset_index(drop=True)


def cluster_ant_counts(
    clusters: pd.DataFrame,
    *,
    cluster_col: str = "cluster_id",
    track_col: str = "TrackID",
) -> pd.DataFrame:
    required = {cluster_col, track_col}
    missing = required.difference(clusters.columns)
    if missing:
        raise ValueError(f"clusters is missing required columns: {sorted(missing)}")

    counts = clusters[[cluster_col, track_col]].dropna().copy()
    counts[cluster_col] = counts[cluster_col].astype(str)
    return (
        counts.groupby(cluster_col, sort=True)[track_col]
        .nunique()
        .rename("n_ants")
        .reset_index()
    )


def cluster_ant_count_map(
    clusters: pd.DataFrame,
    *,
    cluster_col: str = "cluster_id",
    track_col: str = "TrackID",
) -> dict[str, int]:
    counts = cluster_ant_counts(clusters, cluster_col=cluster_col, track_col=track_col)
    return dict(zip(counts[cluster_col].astype(str), counts["n_ants"].astype(int)))


def add_cluster_size_weights(
    events: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    cluster_col: str = "cluster_id",
    count_col: str = "interaction_count",
    output_col: str = "interaction_count_per_ant",
    cluster_size_col: str = "cluster_n_ants",
) -> pd.DataFrame:
    required = {cluster_col, count_col}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"events is missing required columns: {sorted(missing)}")

    out = events.copy()
    ant_counts = cluster_ant_count_map(clusters)
    n_ants = out[cluster_col].astype(str).map(ant_counts).astype("float64")
    counts = pd.to_numeric(out[count_col], errors="coerce").astype("float64")
    out[cluster_size_col] = n_ants
    out[output_col] = np.where(n_ants > 0, counts / n_ants, np.nan)
    return out


def attach_cluster_labels(
    interactions: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    drop_unclustered: bool,
) -> pd.DataFrame:
    cluster_id = clusters.set_index("TrackID")["cluster_id"].to_dict()
    leiden = clusters.set_index("TrackID")["leiden_cluster_id"].to_dict()
    out = interactions.copy()
    out["antenna_cluster_id"] = out["antenna_track_id"].map(cluster_id)
    out["body_cluster_id"] = out["body_track_id"].map(cluster_id)
    out["antenna_leiden_cluster_id"] = out["antenna_track_id"].map(leiden)
    out["body_leiden_cluster_id"] = out["body_track_id"].map(leiden)
    if drop_unclustered:
        out = out.dropna(subset=["antenna_cluster_id", "body_cluster_id"]).copy()
    for col in ["antenna_cluster_id", "body_cluster_id"]:
        out[col] = out[col].fillna("unclustered")
    return out.reset_index(drop=True)


def add_time_columns(
    df: pd.DataFrame,
    *,
    frame_col: str,
    fps: float,
    chunk_start_clock_seconds: int,
    light_on_hour: float,
    time_bin_minutes: float,
    absolute_frame_col: str | None = None,
    recording_start_clock_seconds: int | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if absolute_frame_col is not None and recording_start_clock_seconds is not None:
        absolute_seconds = float(recording_start_clock_seconds) + out[absolute_frame_col].to_numpy(np.float64) / float(fps)
    else:
        absolute_seconds = float(chunk_start_clock_seconds) + out[frame_col].to_numpy(np.float64) / float(fps)

    clock_seconds = absolute_seconds % (24 * 3600)
    light_on_seconds = float(light_on_hour) * 3600.0
    rel_seconds = (clock_seconds - light_on_seconds) % (24 * 3600)
    bin_seconds = float(time_bin_minutes) * 60.0
    time_bin = np.floor(rel_seconds / bin_seconds).astype(np.int64)
    out["clock_hour"] = clock_seconds / 3600.0
    out["light_cycle_day"] = np.floor((absolute_seconds - light_on_seconds) / (24 * 3600)).astype(np.int64)
    out["hours_since_light_on"] = rel_seconds / 3600.0
    out["time_bin"] = time_bin
    out["time_bin_start_h"] = time_bin.astype(np.float64) * bin_seconds / 3600.0
    out["time_bin_label"] = [
        f"{start:.1f}-{start + bin_seconds / 3600.0:.1f}h"
        for start in out["time_bin_start_h"].to_numpy(np.float64)
    ]
    return out


def role_event_positions(
    interactions: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    role: str,
    side: str,
    mm_per_px: float,
    fps: float,
    chunk_start_clock_seconds: int,
    light_on_hour: float,
    time_bin_minutes: float,
    chunk: str | None = None,
    chunk_global_frame_offset: int = 0,
    recording_start_clock_seconds: int | None = None,
) -> pd.DataFrame:
    if role not in {"antenna", "body"}:
        raise ValueError("role must be 'antenna' or 'body'")
    track_col = f"{role}_track_id"
    cluster_col = f"{role}_cluster_id"
    leiden_col = f"{role}_leiden_cluster_id"
    grouped = (
        interactions.groupby(["Frame", track_col, cluster_col, leiden_col], dropna=False, sort=True)
        .size()
        .rename("interaction_count")
        .reset_index()
        .rename(columns={track_col: "TrackID", cluster_col: "cluster_id", leiden_col: "leiden_cluster_id"})
    )
    grouped["cluster_id"] = grouped["cluster_id"].fillna("unclustered")
    merged = grouped.merge(positions, on=["Frame", "TrackID"], how="inner", validate="many_to_one")
    merged["role"] = role
    merged["side"] = side
    if chunk is not None:
        merged["chunk"] = str(chunk)
    merged["global_frame"] = merged["Frame"].astype(np.int64) + int(chunk_global_frame_offset)
    merged["x_mm"] = merged["x_px"].to_numpy(np.float64) * float(mm_per_px)
    merged["y_mm"] = merged["y_px"].to_numpy(np.float64) * float(mm_per_px)
    return add_time_columns(
        merged,
        frame_col="Frame",
        fps=float(fps),
        chunk_start_clock_seconds=int(chunk_start_clock_seconds),
        light_on_hour=float(light_on_hour),
        time_bin_minutes=float(time_bin_minutes),
        absolute_frame_col="global_frame",
        recording_start_clock_seconds=recording_start_clock_seconds,
    )


def role_event_positions_for_chunks(
    interactions: pd.DataFrame,
    chunks: list[InteractionChunk],
    *,
    role: str,
    side: str,
    mm_per_px: float,
    fps: float,
    light_on_hour: float,
    time_bin_minutes: float,
    bodypoint: int = 0,
) -> pd.DataFrame:
    if len(chunks) > 1 and "chunk" not in interactions.columns:
        raise ValueError("Multi-chunk event building requires interactions loaded with load_interactions_for_chunks")
    rows = []
    for i, chunk in enumerate(chunks, start=1):
        if "chunk" in interactions.columns:
            subset = interactions[interactions["chunk"].astype(str) == chunk.chunk].copy()
        else:
            subset = interactions.copy()
        if subset.empty:
            continue
        print(f"{role}: loading positions for chunk{chunk.chunk} ({i}/{len(chunks)})")
        positions = load_track_positions(chunk.track_path, bodypoint=bodypoint)
        events = role_event_positions(
            subset,
            positions,
            role=role,
            side=side,
            mm_per_px=mm_per_px,
            fps=fps,
            chunk_start_clock_seconds=chunk.chunk_start_clock_seconds,
            light_on_hour=light_on_hour,
            time_bin_minutes=time_bin_minutes,
            chunk=chunk.chunk,
            chunk_global_frame_offset=chunk.chunk_global_frame_offset,
            recording_start_clock_seconds=chunk.recording_start_clock_seconds,
        )
        rows.append(events)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _is_per_ant_weight(weight_col: str) -> bool:
    return str(weight_col).endswith("_per_ant")


def cluster_pair_counts(
    interactions: pd.DataFrame,
    *,
    clusters: pd.DataFrame | None = None,
    normalize_by_n_ants: bool = False,
) -> pd.DataFrame:
    pairs = (
        interactions.groupby(["antenna_cluster_id", "body_cluster_id"], dropna=False)
        .size()
        .rename("n_interactions")
        .reset_index()
    )

    if clusters is not None:
        ant_counts = cluster_ant_count_map(clusters)
        pairs["n_antenna_ants"] = pairs["antenna_cluster_id"].astype(str).map(ant_counts).astype("float64")
        pairs["n_body_ants"] = pairs["body_cluster_id"].astype(str).map(ant_counts).astype("float64")
        same_cluster = pairs["antenna_cluster_id"].astype(str) == pairs["body_cluster_id"].astype(str)
        cross_cluster_pairs = pairs["n_antenna_ants"] * pairs["n_body_ants"]
        same_cluster_pairs = pairs["n_antenna_ants"] * np.maximum(pairs["n_antenna_ants"] - 1.0, 0.0)
        pairs["n_possible_directed_ant_pairs"] = np.where(same_cluster, same_cluster_pairs, cross_cluster_pairs)
        denominator = pairs["n_possible_directed_ant_pairs"]
        pairs["interactions_per_directed_ant_pair"] = np.where(
            denominator > 0,
            pairs["n_interactions"].astype("float64") / denominator,
            np.nan,
        )
    elif normalize_by_n_ants:
        raise ValueError("clusters is required when normalize_by_n_ants=True")

    sort_col = "interactions_per_directed_ant_pair" if normalize_by_n_ants else "n_interactions"
    return pairs.sort_values(sort_col, ascending=False, kind="mergesort", na_position="last").reset_index(drop=True)


def cluster_order(
    events: pd.DataFrame,
    *,
    cluster_col: str = "cluster_id",
    weight_col: str = "interaction_count",
    max_clusters: int | None = None,
) -> list[str]:
    if events.empty:
        return []
    if weight_col not in events.columns:
        raise ValueError(f"events is missing weight column: {weight_col!r}")
    values = events[[cluster_col, weight_col]].copy()
    values["_cluster_label"] = values[cluster_col].astype(str)
    values[weight_col] = pd.to_numeric(values[weight_col], errors="coerce").fillna(0.0)
    counts = values.groupby("_cluster_label")[weight_col].sum().sort_values(ascending=False)
    labels = [str(label) for label in counts.index]
    if max_clusters is not None:
        labels = labels[: int(max_clusters)]
    return labels


def spatial_edges(
    events: pd.DataFrame,
    *,
    bin_size_mm: float,
    quantile: tuple[float, float] = (0.005, 0.995),
    pad_bins: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if events.empty:
        raise ValueError("Cannot compute spatial edges from an empty event table")
    q0, q1 = quantile
    x0, x1 = events["x_mm"].quantile([q0, q1]).to_numpy(np.float64)
    y0, y1 = events["y_mm"].quantile([q0, q1]).to_numpy(np.float64)
    step = float(bin_size_mm)
    x0 = math.floor(x0 / step) * step - pad_bins * step
    x1 = math.ceil(x1 / step) * step + pad_bins * step
    y0 = math.floor(y0 / step) * step - pad_bins * step
    y1 = math.ceil(y1 / step) * step + pad_bins * step
    x_edges = np.arange(x0, x1 + step, step, dtype=np.float64)
    y_edges = np.arange(y0, y1 + step, step, dtype=np.float64)
    return x_edges, y_edges


def weighted_hist2d(
    events: pd.DataFrame,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    normalize: bool,
    weight_col: str = "interaction_count",
) -> np.ndarray:
    if weight_col not in events.columns:
        raise ValueError(f"events is missing weight column: {weight_col!r}")
    hist, _, _ = np.histogram2d(
        events["y_mm"].to_numpy(np.float64),
        events["x_mm"].to_numpy(np.float64),
        bins=[y_edges, x_edges],
        weights=pd.to_numeric(events[weight_col], errors="coerce").fillna(0.0).to_numpy(np.float64),
    )
    if normalize and hist.sum() > 0:
        hist = hist / hist.sum()
    return hist


def day_averaged_weighted_hist2d(
    events: pd.DataFrame,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    normalize: bool,
    weight_col: str = "interaction_count",
) -> np.ndarray:
    if events.empty:
        return np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)
    if "light_cycle_day" not in events.columns:
        return weighted_hist2d(events, x_edges, y_edges, normalize=normalize, weight_col=weight_col)

    hists = [
        weighted_hist2d(day_events, x_edges, y_edges, normalize=False, weight_col=weight_col)
        for _, day_events in events.groupby("light_cycle_day", sort=True)
    ]
    if not hists:
        return np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)
    hist = np.mean(np.stack(hists, axis=0), axis=0)
    if normalize and hist.sum() > 0:
        hist = hist / hist.sum()
    return hist


def cluster_pair_matrix_from_table(
    pairs: pd.DataFrame,
    *,
    value_col: str = "n_interactions",
) -> pd.DataFrame:
    if value_col not in pairs.columns:
        raise ValueError(f"pairs is missing value column: {value_col!r}")
    antenna_labels = sorted(str(v) for v in pairs["antenna_cluster_id"].dropna().unique())
    body_labels = sorted(str(v) for v in pairs["body_cluster_id"].dropna().unique())
    matrix = pd.DataFrame(0.0, index=antenna_labels, columns=body_labels)
    for _, row in pairs.iterrows():
        value = row[value_col]
        matrix.loc[str(row["antenna_cluster_id"]), str(row["body_cluster_id"])] = float(value) if pd.notna(value) else np.nan
    return matrix


def plot_cluster_pair_matrix_from_table(
    pairs: pd.DataFrame,
    *,
    value_col: str = "n_interactions",
    title: str = "Directed interaction counts by cluster pair",
    colorbar_label: str | None = None,
) -> pd.DataFrame:
    matrix = cluster_pair_matrix_from_table(pairs, value_col=value_col)
    antenna_labels = matrix.index.tolist()
    body_labels = matrix.columns.tolist()
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(body_labels)), max(4, 0.5 * len(antenna_labels))))
    image = ax.imshow(matrix.to_numpy(), origin="upper", interpolation="none", cmap="magma")
    ax.set_xticks(np.arange(len(body_labels)))
    ax.set_xticklabels(body_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(antenna_labels)))
    ax.set_yticklabels(antenna_labels)
    ax.set_xlabel("body/receiver cluster")
    ax.set_ylabel("antenna/source cluster")
    ax.set_title(title)
    if colorbar_label is None:
        colorbar_label = "directed interactions"
        if value_col == "interactions_per_directed_ant_pair":
            colorbar_label = "interactions / possible directed ant-pair"
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    _save_current_figures_if_enabled()
    return matrix


def plot_cluster_pair_matrix(
    interactions: pd.DataFrame,
    *,
    clusters: pd.DataFrame | None = None,
    normalize_by_n_ants: bool = False,
    value_col: str | None = None,
    title: str = "Directed interaction counts by cluster pair",
    colorbar_label: str | None = None,
) -> pd.DataFrame:
    pairs = cluster_pair_counts(interactions, clusters=clusters, normalize_by_n_ants=normalize_by_n_ants)
    if value_col is None:
        value_col = "interactions_per_directed_ant_pair" if normalize_by_n_ants else "n_interactions"
    return plot_cluster_pair_matrix_from_table(
        pairs,
        value_col=value_col,
        title=title,
        colorbar_label=colorbar_label,
    )


def spatial_heatmaps_by_cluster(
    events: pd.DataFrame,
    *,
    bin_size_mm: float = 10.0,
    max_clusters: int | None = None,
    normalize: bool = False,
    weight_col: str = "interaction_count",
) -> dict[str, object]:
    labels = cluster_order(events, max_clusters=max_clusters, weight_col=weight_col)
    if not labels:
        raise ValueError("No clusters to plot")
    x_edges, y_edges = spatial_edges(events, bin_size_mm=float(bin_size_mm))
    histograms = {}
    for label in labels:
        hist = weighted_hist2d(
            events[events["cluster_id"].astype(str) == label],
            x_edges,
            y_edges,
            normalize=normalize,
            weight_col=weight_col,
        )
        histograms[label] = hist
    return {
        "labels": labels,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "histograms": histograms,
        "bin_size_mm": float(bin_size_mm),
        "normalize": bool(normalize),
        "weight_col": str(weight_col),
    }


def plot_spatial_heatmaps_by_cluster_result(
    result: dict[str, object],
    *,
    ncols: int = 3,
    vmax_percentile: float = 99.0,
    cmap: str = "magma",
    title: str = "Interaction locations by cluster",
    colorbar_label: str | None = None,
):
    labels = [str(label) for label in result["labels"]]
    x_edges = np.asarray(result["x_edges"], dtype=np.float64)
    y_edges = np.asarray(result["y_edges"], dtype=np.float64)
    histograms = result["histograms"]
    weight_col = str(result.get("weight_col", "interaction_count"))
    normalize = bool(result.get("normalize", False))
    nonzero_values = []
    for label in labels:
        hist = np.asarray(histograms[label], dtype=np.float64)
        nonzero_values.extend(hist[hist > 0].ravel().tolist())
    vmax = None
    if nonzero_values:
        vmax = float(np.percentile(np.asarray(nonzero_values), float(vmax_percentile)))
    if colorbar_label is None:
        if normalize:
            colorbar_label = "fraction of cluster total"
        elif _is_per_ant_weight(weight_col):
            colorbar_label = "directed interactions / ant"
        else:
            colorbar_label = "directed interactions"

    ncols = max(1, int(ncols))
    nrows = math.ceil(len(labels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)
    for ax, label in zip(axes.ravel(), labels):
        hist = np.asarray(histograms[label], dtype=np.float64)
        image = ax.imshow(
            hist,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            origin="lower",
            interpolation="none",
            aspect="auto",
            cmap=cmap,
            vmax=vmax,
        )
        total = float(hist.sum())
        total_name = "strength" if _is_per_ant_weight(weight_col) else "n"
        total_fmt = ".3g" if (_is_per_ant_weight(weight_col) or normalize) else ".0f"
        ax.set_title(f"{label} {total_name}={total:{total_fmt}}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    for ax in axes.ravel()[len(labels):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    _save_current_figures_if_enabled()
    return histograms


def plot_spatial_heatmaps_by_cluster(
    events: pd.DataFrame,
    *,
    bin_size_mm: float = 10.0,
    max_clusters: int | None = None,
    ncols: int = 3,
    normalize: bool = False,
    weight_col: str = "interaction_count",
    vmax_percentile: float = 99.0,
    cmap: str = "magma",
    title: str = "Interaction locations by cluster",
):
    result = spatial_heatmaps_by_cluster(
        events,
        bin_size_mm=bin_size_mm,
        max_clusters=max_clusters,
        normalize=normalize,
        weight_col=weight_col,
    )
    plot_spatial_heatmaps_by_cluster_result(
        result,
        ncols=ncols,
        vmax_percentile=vmax_percentile,
        cmap=cmap,
        title=title,
    )
    return result["histograms"]


def time_counts_by_cluster(
    events: pd.DataFrame,
    *,
    max_clusters: int | None = None,
    average_over_days: bool = True,
    weight_col: str = "interaction_count",
) -> pd.DataFrame:
    labels = cluster_order(events, max_clusters=max_clusters, weight_col=weight_col)
    filtered = events[events["cluster_id"].astype(str).isin(labels)].copy()
    filtered["_cluster_label"] = filtered["cluster_id"].astype(str)
    filtered[weight_col] = pd.to_numeric(filtered[weight_col], errors="coerce").fillna(0.0)
    if average_over_days and "light_cycle_day" in filtered.columns:
        values = (
            filtered.groupby(
                ["_cluster_label", "light_cycle_day", "time_bin", "time_bin_label", "time_bin_start_h"],
                sort=True,
            )[weight_col]
            .sum()
            .rename("daily_interaction_value")
            .reset_index()
        )
        values = (
            values.groupby(["_cluster_label", "time_bin", "time_bin_label", "time_bin_start_h"], sort=True)[
                "daily_interaction_value"
            ]
            .mean()
            .rename("interaction_value")
            .reset_index()
        )
    else:
        values = (
            filtered.groupby(["_cluster_label", "time_bin", "time_bin_label", "time_bin_start_h"], sort=True)[
                weight_col
            ]
            .sum()
            .rename("interaction_value")
            .reset_index()
        )
    values = values.rename(columns={"_cluster_label": "cluster_id"})

    pivot = values.pivot_table(
        index="cluster_id",
        columns="time_bin_label",
        values="interaction_value",
        aggfunc="sum",
        fill_value=0.0,
    )
    pivot = pivot.reindex(labels)
    time_order = (
        values[["time_bin_label", "time_bin_start_h"]]
        .drop_duplicates()
        .sort_values("time_bin_start_h", kind="mergesort")["time_bin_label"]
        .tolist()
    )
    pivot = pivot.reindex(columns=time_order)
    return pivot


def plot_time_counts_table(
    pivot: pd.DataFrame,
    *,
    title: str = "Interaction counts by time since light on",
    cmap: str = "magma",
    color_label: str = "directed interactions",
) -> pd.DataFrame:
    labels = [str(label) for label in pivot.index.tolist()]
    time_order = [str(label) for label in pivot.columns.tolist()]

    fig, ax = plt.subplots(figsize=(max(5, 0.65 * len(time_order)), max(3, 0.45 * len(labels))))
    image = ax.imshow(pivot.to_numpy(dtype=float), origin="upper", interpolation="none", aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(time_order)))
    ax.set_xticklabels(time_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("hours since light on")
    ax.set_ylabel("cluster")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=color_label)
    fig.tight_layout()
    _save_current_figures_if_enabled()
    return pivot


def plot_time_counts_by_cluster(
    events: pd.DataFrame,
    *,
    max_clusters: int | None = None,
    average_over_days: bool = True,
    weight_col: str = "interaction_count",
    title: str = "Interaction counts by time since light on",
    cmap: str = "magma",
) -> pd.DataFrame:
    pivot = time_counts_by_cluster(
        events,
        max_clusters=max_clusters,
        average_over_days=average_over_days,
        weight_col=weight_col,
    )
    if average_over_days:
        color_label = "mean directed interactions / light-cycle day"
        if _is_per_ant_weight(weight_col):
            color_label = "mean directed interactions / ant / light-cycle day"
    else:
        color_label = "directed interactions / ant" if _is_per_ant_weight(weight_col) else "directed interactions"
    return plot_time_counts_table(pivot, title=title, cmap=cmap, color_label=color_label)


def smooth_series(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if sigma_bins <= 0 or len(values) < 3:
        return values
    try:
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(values, sigma=float(sigma_bins), mode="nearest")
    except Exception:
        window = int(max(3, round(6 * float(sigma_bins) + 1)))
        if window % 2 == 0:
            window += 1
        return (
            pd.Series(values)
            .rolling(window, center=True, min_periods=1)
            .mean()
            .to_numpy(np.float64)
        )


def interaction_timeseries_by_cluster(
    events: pd.DataFrame,
    *,
    bin_minutes: float,
    cluster_col: str = "cluster_id",
    average_over_days: bool = True,
    weight_col: str = "interaction_count",
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                cluster_col,
                "time_bin",
                "time_bin_start_h",
                "time_bin_mid_h",
                "n_interactions",
                "interaction_rate_per_min",
            ]
        )
    if weight_col not in events.columns:
        raise ValueError(f"events is missing weight column: {weight_col!r}")
    values = events.copy()
    values["_cluster_label"] = values[cluster_col].astype(str)
    values[weight_col] = pd.to_numeric(values[weight_col], errors="coerce").fillna(0.0)
    if average_over_days and "light_cycle_day" in events.columns:
        grouped = (
            values.groupby(["_cluster_label", "light_cycle_day", "time_bin", "time_bin_start_h"], sort=True)[
                weight_col
            ]
            .sum()
            .rename("daily_interaction_value")
            .reset_index()
        )
        grouped = (
            grouped.groupby(["_cluster_label", "time_bin", "time_bin_start_h"], sort=True)["daily_interaction_value"]
            .mean()
            .rename("n_interactions")
            .reset_index()
        )
    else:
        grouped = (
            values.groupby(["_cluster_label", "time_bin", "time_bin_start_h"], sort=True)[weight_col]
            .sum()
            .rename("n_interactions")
            .reset_index()
        )
    grouped = grouped.rename(columns={"_cluster_label": cluster_col})
    grouped["time_bin_mid_h"] = grouped["time_bin_start_h"] + float(bin_minutes) / 120.0
    grouped["interaction_rate_per_min"] = grouped["n_interactions"] / float(bin_minutes)
    return grouped


def interaction_timeseries_plot_table_by_cluster(
    events: pd.DataFrame,
    *,
    bin_minutes: float,
    smooth_bins: float = 0.0,
    max_clusters: int | None = None,
    average_over_days: bool = True,
    weight_col: str = "interaction_count",
) -> pd.DataFrame:
    table = interaction_timeseries_by_cluster(
        events,
        bin_minutes=bin_minutes,
        average_over_days=average_over_days,
        weight_col=weight_col,
    )
    labels = cluster_order(events, max_clusters=max_clusters, weight_col=weight_col)
    if not labels:
        raise ValueError("No clusters to plot")

    time_bins = (
        events[["time_bin", "time_bin_start_h"]]
        .drop_duplicates()
        .sort_values("time_bin_start_h", kind="mergesort")
        .reset_index(drop=True)
    )
    full_index = pd.MultiIndex.from_product(
        [labels, time_bins["time_bin"].tolist()],
        names=["cluster_id", "time_bin"],
    )
    complete = (
        table.set_index(["cluster_id", "time_bin"])
        .reindex(full_index)
        .reset_index()
        .merge(time_bins, on="time_bin", how="left", suffixes=("", "_filled"))
    )
    if "time_bin_start_h_filled" in complete.columns:
        complete["time_bin_start_h"] = complete["time_bin_start_h"].fillna(complete["time_bin_start_h_filled"])
        complete = complete.drop(columns=["time_bin_start_h_filled"])
    complete["n_interactions"] = complete["n_interactions"].fillna(0.0)
    complete["time_bin_mid_h"] = complete["time_bin_start_h"] + float(bin_minutes) / 120.0
    complete["interaction_rate_per_min"] = complete["n_interactions"] / float(bin_minutes)

    plot_rows = []
    for cluster_id, cluster_df in complete.groupby("cluster_id", sort=False):
        cluster_df = cluster_df.sort_values("time_bin_start_h", kind="mergesort").copy()
        cluster_df["smoothed_rate_per_min"] = smooth_series(
            cluster_df["interaction_rate_per_min"].to_numpy(np.float64),
            smooth_bins,
        )
        plot_rows.append(cluster_df)
    plot_df = pd.concat(plot_rows, ignore_index=True)
    plot_df["weight_col"] = str(weight_col)
    plot_df["average_over_days"] = bool(average_over_days)
    return plot_df


def plot_interaction_timeseries_table(
    plot_df: pd.DataFrame,
    *,
    smooth_bins: float = 0.0,
    average_over_days: bool = True,
    ylim: tuple[float, float] | None = None,
    title: str = "Interaction time series by cluster",
    cmap: str = "tab10",
) -> pd.DataFrame:
    labels = [str(label) for label in plot_df["cluster_id"].drop_duplicates().tolist()]
    weight_col = str(plot_df["weight_col"].dropna().iloc[0]) if "weight_col" in plot_df and not plot_df.empty else "interaction_count"
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = plt.get_cmap(cmap, len(labels))
    y_col = "smoothed_rate_per_min" if smooth_bins > 0 else "interaction_rate_per_min"
    for i, label in enumerate(labels):
        cluster_df = plot_df[plot_df["cluster_id"] == label]
        ax.plot(
            cluster_df["time_bin_mid_h"],
            cluster_df[y_col],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=str(label),
            color=colors(i),
        )
    ax.set_xlabel("hours since light on")
    if average_over_days:
        ylabel = "mean directed interactions / min / light-cycle day"
        if _is_per_ant_weight(weight_col):
            ylabel = "mean directed interactions / min / ant / light-cycle day"
    else:
        ylabel = "directed interactions / min / ant" if _is_per_ant_weight(weight_col) else "directed interactions / min"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(title="cluster", loc="best", fontsize="small")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_current_figures_if_enabled()
    return plot_df


def plot_interaction_timeseries_by_cluster(
    events: pd.DataFrame,
    *,
    bin_minutes: float,
    smooth_bins: float = 0.0,
    max_clusters: int | None = None,
    average_over_days: bool = True,
    weight_col: str = "interaction_count",
    ylim: tuple[float, float] | None = None,
    title: str = "Interaction time series by cluster",
    cmap: str = "tab10",
) -> pd.DataFrame:
    plot_df = interaction_timeseries_plot_table_by_cluster(
        events,
        bin_minutes=bin_minutes,
        smooth_bins=smooth_bins,
        max_clusters=max_clusters,
        average_over_days=average_over_days,
        weight_col=weight_col,
    )
    return plot_interaction_timeseries_table(
        plot_df,
        smooth_bins=smooth_bins,
        average_over_days=average_over_days,
        ylim=ylim,
        title=title,
        cmap=cmap,
    )


def spatial_heatmaps_by_cluster_and_time(
    events: pd.DataFrame,
    *,
    bin_size_mm: float = 10.0,
    max_clusters: int | None = 4,
    max_time_bins: int | None = None,
    average_over_days: bool = True,
    normalize: bool = False,
    weight_col: str = "interaction_count",
) -> dict[str, object]:
    labels = cluster_order(events, max_clusters=max_clusters, weight_col=weight_col)
    if not labels:
        raise ValueError("No clusters to plot")
    time_bins = (
        events[["time_bin", "time_bin_label", "time_bin_start_h"]]
        .drop_duplicates()
        .sort_values("time_bin_start_h", kind="mergesort")
    )
    if max_time_bins is not None:
        time_bins = time_bins.iloc[: int(max_time_bins)]
    time_labels = time_bins["time_bin_label"].tolist()
    time_ids = time_bins["time_bin"].tolist()
    x_edges, y_edges = spatial_edges(events, bin_size_mm=float(bin_size_mm))

    hists = {}
    for label in labels:
        for time_id, time_label in zip(time_ids, time_labels):
            subset = events[(events["cluster_id"].astype(str) == label) & (events["time_bin"] == time_id)]
            if average_over_days:
                hist = day_averaged_weighted_hist2d(
                    subset,
                    x_edges,
                    y_edges,
                    normalize=normalize,
                    weight_col=weight_col,
                )
            else:
                hist = (
                    weighted_hist2d(subset, x_edges, y_edges, normalize=normalize, weight_col=weight_col)
                    if not subset.empty
                    else np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=float)
                )
            hists[(label, time_label)] = hist
    return {
        "labels": labels,
        "time_labels": time_labels,
        "time_ids": time_ids,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "hists": hists,
        "bin_size_mm": float(bin_size_mm),
        "average_over_days": bool(average_over_days),
        "normalize": bool(normalize),
        "weight_col": str(weight_col),
    }


def plot_spatial_heatmaps_by_cluster_and_time_result(
    result: dict[str, object],
    *,
    vmax_percentile: float = 99.0,
    cmap: str = "magma",
    title: str = "Interaction locations by cluster and time",
    colorbar_label: str | None = None,
):
    labels = [str(label) for label in result["labels"]]
    time_labels = [str(label) for label in result["time_labels"]]
    x_edges = np.asarray(result["x_edges"], dtype=np.float64)
    y_edges = np.asarray(result["y_edges"], dtype=np.float64)
    hists = result["hists"]
    average_over_days = bool(result.get("average_over_days", True))
    normalize = bool(result.get("normalize", False))
    weight_col = str(result.get("weight_col", "interaction_count"))
    nonzero_values = []
    for hist in hists.values():
        hist = np.asarray(hist, dtype=np.float64)
        nonzero_values.extend(hist[hist > 0].ravel().tolist())
    vmax = None
    if nonzero_values:
        vmax = float(np.percentile(np.asarray(nonzero_values), float(vmax_percentile)))
    if colorbar_label is None:
        if normalize:
            colorbar_label = "fraction of cluster/time total"
        elif average_over_days and _is_per_ant_weight(weight_col):
            colorbar_label = "mean interactions / ant / day"
        elif average_over_days:
            colorbar_label = "mean interactions / day"
        elif _is_per_ant_weight(weight_col):
            colorbar_label = "directed interactions / ant"
        else:
            colorbar_label = "directed interactions"

    fig, axes = plt.subplots(
        len(labels),
        len(time_labels),
        figsize=(4.0 * len(time_labels), 3.7 * len(labels)),
        squeeze=False,
    )
    for row_idx, label in enumerate(labels):
        for col_idx, time_label in enumerate(time_labels):
            ax = axes[row_idx, col_idx]
            hist = np.asarray(hists[(label, time_label)], dtype=np.float64)
            image = ax.imshow(
                hist,
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                origin="lower",
                interpolation="none",
                aspect="auto",
                cmap=cmap,
                vmax=vmax,
            )
            total = float(hist.sum())
            total_name = "strength" if _is_per_ant_weight(weight_col) else "n"
            total_fmt = ".3g" if (_is_per_ant_weight(weight_col) or normalize) else ".0f"
            ax.set_title(f"{label} {time_label} {total_name}={total:{total_fmt}}")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.suptitle(title)
    fig.tight_layout()
    _save_current_figures_if_enabled()
    return hists


def plot_spatial_heatmaps_by_cluster_and_time(
    events: pd.DataFrame,
    *,
    bin_size_mm: float = 10.0,
    max_clusters: int | None = 4,
    max_time_bins: int | None = None,
    average_over_days: bool = True,
    normalize: bool = False,
    weight_col: str = "interaction_count",
    vmax_percentile: float = 99.0,
    cmap: str = "magma",
    title: str = "Interaction locations by cluster and time",
):
    result = spatial_heatmaps_by_cluster_and_time(
        events,
        bin_size_mm=bin_size_mm,
        max_clusters=max_clusters,
        max_time_bins=max_time_bins,
        average_over_days=average_over_days,
        normalize=normalize,
        weight_col=weight_col,
    )
    return plot_spatial_heatmaps_by_cluster_and_time_result(
        result,
        vmax_percentile=vmax_percentile,
        cmap=cmap,
        title=title,
    )


def speed_metadata_paths(speed_root: Path) -> list[Path]:
    paths = sorted((Path(speed_root) / "per_track").glob("*/speed_metadata.json"))
    if not paths:
        paths = sorted(Path(speed_root).glob("*/speed_metadata.json"))
    return paths


def side_from_track_name(track_name: str, track_dir: Path) -> str | None:
    if track_name.endswith("_left.parquet") or track_dir.name.endswith("_left"):
        return "left"
    if track_name.endswith("_right.parquet") or track_dir.name.endswith("_right"):
        return "right"
    return None


def load_speed_tracks(speed_root: Path, *, side: str | None = None) -> pd.DataFrame:
    import json

    rows = []
    for metadata_path in speed_metadata_paths(Path(speed_root)):
        meta = json.loads(metadata_path.read_text())
        track_name = str(meta.get("track_name", metadata_path.parent.name))
        track_side = side_from_track_name(track_name, metadata_path.parent)
        if track_side is None:
            continue
        if side is not None and track_side != side:
            continue
        n_frames = int(meta["n_frames"])
        n_observed = int(meta["n_observed_frames"])
        rows.append(
            {
                "track_name": track_name,
                "track_id": int(meta["track_id"]) if meta.get("track_id") is not None else None,
                "side": track_side,
                "speed_metadata_path": metadata_path,
                "speed_path": metadata_path.parent / "speed_mm_s.npy",
                "frame_min": int(meta["frame_min"]),
                "frame_max": int(meta.get("frame_max", int(meta["frame_min"]) + n_frames - 1)),
                "n_frames": n_frames,
                "n_observed_frames": n_observed,
                "present_frac": n_observed / n_frames if n_frames else np.nan,
                "fps": float(meta["fps"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise FileNotFoundError(f"No speed_metadata.json files found under {speed_root}")
    return out.sort_values(["side", "track_id", "track_name"], kind="mergesort").reset_index(drop=True)


def contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(changes[0::2], changes[1::2])]


def immobile_bouts_for_speed(
    speed: np.ndarray,
    *,
    track_id: int,
    side: str,
    frame_min: int,
    fps: float,
    speed_threshold_mm_s: float,
    min_immobile_seconds: float,
    chunk_global_start_frame: int,
    chunk_global_stop_frame: int,
) -> pd.DataFrame:
    global_start = max(int(frame_min), int(chunk_global_start_frame))
    global_stop = min(int(frame_min) + len(speed), int(chunk_global_stop_frame))
    if global_stop <= global_start:
        return pd.DataFrame()

    local_start = global_start - int(frame_min)
    local_stop = global_stop - int(frame_min)
    chunk_speed = np.asarray(speed[local_start:local_stop], dtype=np.float32)
    finite = np.isfinite(chunk_speed)
    immobile = finite & (chunk_speed <= float(speed_threshold_mm_s))
    min_frames = max(1, int(round(float(min_immobile_seconds) * float(fps))))

    rows = []
    for run_start, run_stop in contiguous_true_runs(immobile):
        duration_frames = int(run_stop - run_start)
        if duration_frames < min_frames:
            continue
        bout_start = int(global_start + run_start)
        bout_end = int(global_start + run_stop - 1)
        next_local = local_start + run_stop
        next_speed = float(speed[next_local]) if next_local < len(speed) else np.nan
        mobility_observed = bool(np.isfinite(next_speed) and next_speed > float(speed_threshold_mm_s))
        mobility_frame = int(frame_min + next_local) if mobility_observed else np.nan
        rows.append(
            {
                "track_id": int(track_id),
                "side": side,
                "bout_start_frame": bout_start,
                "bout_end_frame": bout_end,
                "mobility_frame": mobility_frame,
                "mobility_observed": mobility_observed,
                "immobile_duration_frames": duration_frames,
                "immobile_duration_seconds": duration_frames / float(fps),
                "time_to_mobility_seconds": ((int(mobility_frame) - bout_start) / float(fps)) if mobility_observed else np.nan,
                "mean_speed_mm_s": float(np.nanmean(chunk_speed[run_start:run_stop])),
            }
        )
    return pd.DataFrame(rows)


def immobile_bouts_for_tracks(
    speed_tracks: pd.DataFrame,
    *,
    track_ids: list[int] | np.ndarray | pd.Series,
    side: str,
    fps: float,
    speed_threshold_mm_s: float,
    min_immobile_seconds: float,
    chunk_global_start_frame: int,
    chunk_global_stop_frame: int,
) -> pd.DataFrame:
    track_id_set = {int(track_id) for track_id in track_ids}
    chosen = speed_tracks[(speed_tracks["side"] == side) & (speed_tracks["track_id"].isin(track_id_set))].copy()
    rows = []
    for row in chosen.itertuples(index=False):
        speed_path = Path(row.speed_path)
        if not speed_path.exists():
            continue
        speed = np.load(speed_path, mmap_mode="r")
        bouts = immobile_bouts_for_speed(
            speed,
            track_id=int(row.track_id),
            side=side,
            frame_min=int(row.frame_min),
            fps=float(fps),
            speed_threshold_mm_s=float(speed_threshold_mm_s),
            min_immobile_seconds=float(min_immobile_seconds),
            chunk_global_start_frame=int(chunk_global_start_frame),
            chunk_global_stop_frame=int(chunk_global_stop_frame),
        )
        if not bouts.empty:
            bouts["speed_path"] = speed_path
            rows.append(bouts)
    if not rows:
        return pd.DataFrame(
            columns=[
                "track_id",
                "side",
                "bout_start_frame",
                "bout_end_frame",
                "mobility_frame",
                "mobility_observed",
                "immobile_duration_frames",
                "immobile_duration_seconds",
                "time_to_mobility_seconds",
                "mean_speed_mm_s",
                "speed_path",
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(["track_id", "bout_start_frame"], kind="mergesort").reset_index(drop=True)


def interaction_frame_counts_by_track(
    interactions: pd.DataFrame,
    *,
    chunk_global_frame_offset: int,
) -> dict[int, pd.DataFrame]:
    if interactions.empty:
        return {}
    if "global_frame" in interactions.columns:
        work = interactions[["global_frame", "antenna_track_id", "body_track_id"]].copy()
        work["global_frame"] = work["global_frame"].astype(np.int64)
    else:
        work = interactions[["Frame", "antenna_track_id", "body_track_id"]].copy()
        work["global_frame"] = work["Frame"].astype(np.int64) + int(chunk_global_frame_offset)
    antenna_counts = (
        work.groupby(["antenna_track_id", "global_frame"], sort=True)
        .size()
        .rename("n_interactions_as_antenna")
        .reset_index()
        .rename(columns={"antenna_track_id": "track_id"})
    )
    body_counts = (
        work.groupby(["body_track_id", "global_frame"], sort=True)
        .size()
        .rename("n_interactions_as_body")
        .reset_index()
        .rename(columns={"body_track_id": "track_id"})
    )
    counts = antenna_counts.merge(body_counts, on=["track_id", "global_frame"], how="outer")
    counts["n_interactions_as_antenna"] = counts["n_interactions_as_antenna"].fillna(0).astype(np.int64)
    counts["n_interactions_as_body"] = counts["n_interactions_as_body"].fillna(0).astype(np.int64)
    counts["n_interactions_total"] = counts["n_interactions_as_antenna"] + counts["n_interactions_as_body"]
    return {
        int(track_id): group.sort_values("global_frame", kind="mergesort").reset_index(drop=True)
        for track_id, group in counts.groupby("track_id", sort=False)
    }


def interaction_bout_counts_by_track(
    interactions: pd.DataFrame,
    *,
    chunk_global_frame_offset: int,
    fps: float = 24.0,
    event_gap_seconds: float = 2.0,
) -> dict[int, pd.DataFrame]:
    """Count directed interaction bouts instead of every frame-level detection.

    A directed pair that remains in contact for consecutive frames contributes
    one count at the first frame. A new bout starts after a gap longer than
    `event_gap_seconds`.
    """
    if interactions.empty:
        return {}
    if "global_frame" in interactions.columns:
        work = interactions[["global_frame", "antenna_track_id", "body_track_id"]].copy()
    else:
        work = interactions[["Frame", "antenna_track_id", "body_track_id"]].copy()
        work["global_frame"] = pd.to_numeric(work["Frame"], errors="coerce") + int(chunk_global_frame_offset)
    for col in ["global_frame", "antenna_track_id", "body_track_id"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["global_frame", "antenna_track_id", "body_track_id"]).copy()
    if work.empty:
        return {}
    work["global_frame"] = np.rint(work["global_frame"]).astype(np.int64)
    work["antenna_track_id"] = work["antenna_track_id"].astype(int)
    work["body_track_id"] = work["body_track_id"].astype(int)
    work = work[work["antenna_track_id"] != work["body_track_id"]].drop_duplicates(
        ["antenna_track_id", "body_track_id", "global_frame"]
    )
    if work.empty:
        return {}

    gap_frames = max(1, int(round(float(event_gap_seconds) * float(fps))))
    work = work.sort_values(["antenna_track_id", "body_track_id", "global_frame"], kind="mergesort")
    frame_gap = work.groupby(["antenna_track_id", "body_track_id"], sort=False)["global_frame"].diff()
    onsets = work[frame_gap.isna() | (frame_gap > gap_frames)].copy()
    if onsets.empty:
        return {}

    antenna_counts = (
        onsets.groupby(["antenna_track_id", "global_frame"], sort=True)
        .size()
        .rename("n_interactions_as_antenna")
        .reset_index()
        .rename(columns={"antenna_track_id": "track_id"})
    )
    body_counts = (
        onsets.groupby(["body_track_id", "global_frame"], sort=True)
        .size()
        .rename("n_interactions_as_body")
        .reset_index()
        .rename(columns={"body_track_id": "track_id"})
    )
    counts = antenna_counts.merge(body_counts, on=["track_id", "global_frame"], how="outer")
    counts["n_interactions_as_antenna"] = counts["n_interactions_as_antenna"].fillna(0).astype(np.int64)
    counts["n_interactions_as_body"] = counts["n_interactions_as_body"].fillna(0).astype(np.int64)
    counts["n_interactions_total"] = counts["n_interactions_as_antenna"] + counts["n_interactions_as_body"]
    counts["interaction_count_measure"] = "directed_bout_onset"
    counts["interaction_bout_gap_seconds"] = float(event_gap_seconds)
    return {
        int(track_id): group.sort_values("global_frame", kind="mergesort").reset_index(drop=True)
        for track_id, group in counts.groupby("track_id", sort=False)
    }


def add_interaction_counts_to_bouts(
    bouts: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    chunk_global_frame_offset: int,
    clusters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if bouts.empty:
        return bouts.copy()
    counts_by_track = interaction_frame_counts_by_track(
        interactions,
        chunk_global_frame_offset=int(chunk_global_frame_offset),
    )
    rows = []
    for bout in bouts.itertuples(index=False):
        track_counts = counts_by_track.get(int(bout.track_id))
        if track_counts is None or track_counts.empty:
            n_antenna = 0
            n_body = 0
            n_total = 0
            first_interaction_frame = np.nan
            last_interaction_frame = np.nan
        else:
            mask = (
                (track_counts["global_frame"] >= int(bout.bout_start_frame))
                & (track_counts["global_frame"] <= int(bout.bout_end_frame))
            )
            selected = track_counts.loc[mask]
            n_antenna = int(selected["n_interactions_as_antenna"].sum())
            n_body = int(selected["n_interactions_as_body"].sum())
            n_total = int(selected["n_interactions_total"].sum())
            if selected.empty:
                first_interaction_frame = np.nan
                last_interaction_frame = np.nan
            else:
                first_interaction_frame = int(selected["global_frame"].min())
                last_interaction_frame = int(selected["global_frame"].max())
        row = bout._asdict()
        row.update(
            {
                "n_interactions_as_antenna": n_antenna,
                "n_interactions_as_body": n_body,
                "n_interactions_total": n_total,
                "first_interaction_frame": first_interaction_frame,
                "last_interaction_frame": last_interaction_frame,
                "has_interaction": n_total > 0,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    seconds_per_frame = out["immobile_duration_seconds"] / out["immobile_duration_frames"]
    out["time_from_first_interaction_to_mobility_seconds"] = np.where(
        out["mobility_observed"] & out["first_interaction_frame"].notna(),
        (out["mobility_frame"].astype(float) - out["first_interaction_frame"].astype(float)) * seconds_per_frame,
        np.nan,
    )
    out["interaction_rate_per_min_immobile"] = out["n_interactions_total"] / (out["immobile_duration_seconds"] / 60.0)
    if clusters is not None and not clusters.empty:
        cluster_map = clusters.set_index("TrackID")["cluster_id"].to_dict()
        leiden_map = clusters.set_index("TrackID")["leiden_cluster_id"].to_dict()
        out["cluster_id"] = out["track_id"].map(cluster_map)
        out["leiden_cluster_id"] = out["track_id"].map(leiden_map)
    return out


def immobility_interaction_analysis(
    *,
    speed_root: Path,
    interactions: pd.DataFrame,
    clusters: pd.DataFrame,
    chunk: InteractionChunk,
    speed_threshold_mm_s: float,
    min_immobile_seconds: float,
    fps: float,
) -> pd.DataFrame:
    speed_tracks = load_speed_tracks(speed_root, side=chunk.side)
    track_ids = pd.unique(
        pd.concat(
            [
                interactions["antenna_track_id"],
                interactions["body_track_id"],
                clusters["TrackID"],
            ],
            ignore_index=True,
        ).dropna()
    )
    chunk_start = int(chunk.chunk_global_frame_offset)
    chunk_stop = int(chunk.chunk_global_frame_offset + chunk.chunk_frame_count)
    bouts = immobile_bouts_for_tracks(
        speed_tracks,
        track_ids=track_ids,
        side=chunk.side,
        fps=float(fps),
        speed_threshold_mm_s=float(speed_threshold_mm_s),
        min_immobile_seconds=float(min_immobile_seconds),
        chunk_global_start_frame=chunk_start,
        chunk_global_stop_frame=chunk_stop,
    )
    return add_interaction_counts_to_bouts(
        bouts,
        interactions,
        chunk_global_frame_offset=int(chunk.chunk_global_frame_offset),
        clusters=clusters,
    )


def immobility_interaction_analysis_for_chunks(
    *,
    speed_root: Path,
    interactions: pd.DataFrame,
    clusters: pd.DataFrame,
    chunks: list[InteractionChunk],
    speed_threshold_mm_s: float,
    min_immobile_seconds: float,
    fps: float,
) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame()
    sides = {chunk.side for chunk in chunks}
    if len(sides) != 1:
        raise ValueError(f"Expected chunks from one side, got {sorted(sides)}")

    side = chunks[0].side
    speed_tracks = load_speed_tracks(speed_root, side=side)
    track_ids = pd.unique(
        pd.concat(
            [
                interactions["antenna_track_id"],
                interactions["body_track_id"],
                clusters["TrackID"],
            ],
            ignore_index=True,
        ).dropna()
    )

    rows = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"immobility: scanning speed vectors for chunk{chunk.chunk} ({i}/{len(chunks)})")
        chunk_start = int(chunk.chunk_global_frame_offset)
        chunk_stop = int(chunk.chunk_global_frame_offset + chunk.chunk_frame_count)
        bouts = immobile_bouts_for_tracks(
            speed_tracks,
            track_ids=track_ids,
            side=side,
            fps=float(fps),
            speed_threshold_mm_s=float(speed_threshold_mm_s),
            min_immobile_seconds=float(min_immobile_seconds),
            chunk_global_start_frame=chunk_start,
            chunk_global_stop_frame=chunk_stop,
        )
        if not bouts.empty:
            bouts["chunk"] = chunk.chunk
            rows.append(bouts)

    if rows:
        bouts = pd.concat(rows, ignore_index=True).sort_values(
            ["track_id", "bout_start_frame"],
            kind="mergesort",
        )
    else:
        bouts = pd.DataFrame()

    return add_interaction_counts_to_bouts(
        bouts,
        interactions,
        chunk_global_frame_offset=0,
        clusters=clusters,
    )


def immobility_wake_prediction_table(
    bouts: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    fps: float,
    bin_seconds: float = 60.0,
    include_censored: bool = False,
    max_bins_per_bout: int | None = None,
) -> pd.DataFrame:
    if bouts.empty:
        return pd.DataFrame()

    bin_frames = max(1, int(round(float(bin_seconds) * float(fps))))
    counts_by_track = interaction_frame_counts_by_track(interactions, chunk_global_frame_offset=0)
    rows = []
    for bout_index, bout in enumerate(bouts.itertuples(index=False)):
        mobility_observed = bool(getattr(bout, "mobility_observed", False))
        if not mobility_observed and not include_censored:
            continue

        track_id = int(getattr(bout, "track_id"))
        bout_start = int(getattr(bout, "bout_start_frame"))
        bout_end = int(getattr(bout, "bout_end_frame"))
        if bout_end < bout_start:
            continue

        track_counts = counts_by_track.get(track_id)
        if track_counts is None or track_counts.empty:
            frames = np.asarray([], dtype=np.int64)
            antenna_counts = np.asarray([], dtype=np.int64)
            body_counts = np.asarray([], dtype=np.int64)
            total_counts = np.asarray([], dtype=np.int64)
        else:
            track_counts = track_counts.sort_values("global_frame", kind="mergesort")
            frames = track_counts["global_frame"].to_numpy(np.int64)
            antenna_counts = track_counts["n_interactions_as_antenna"].to_numpy(np.int64)
            body_counts = track_counts["n_interactions_as_body"].to_numpy(np.int64)
            total_counts = track_counts["n_interactions_total"].to_numpy(np.int64)

        cumulative_antenna = 0
        cumulative_body = 0
        cumulative_total = 0
        n_bins = int(math.ceil((bout_end - bout_start + 1) / bin_frames))
        if max_bins_per_bout is not None:
            n_bins = min(n_bins, int(max_bins_per_bout))

        for bin_index in range(n_bins):
            bin_start = bout_start + bin_index * bin_frames
            bin_end = min(bout_end, bin_start + bin_frames - 1)
            left = int(np.searchsorted(frames, bin_start, side="left"))
            right = int(np.searchsorted(frames, bin_end, side="right"))
            n_antenna = int(antenna_counts[left:right].sum()) if right > left else 0
            n_body = int(body_counts[left:right].sum()) if right > left else 0
            n_total = int(total_counts[left:right].sum()) if right > left else 0
            cumulative_antenna += n_antenna
            cumulative_body += n_body
            cumulative_total += n_total

            elapsed_seconds = (bin_end - bout_start + 1) / float(fps)
            woke = int(mobility_observed and bin_end >= bout_end)
            rows.append(
                {
                    "bout_index": int(bout_index),
                    "track_id": track_id,
                    "side": getattr(bout, "side", None),
                    "chunk": getattr(bout, "chunk", None),
                    "cluster_id": getattr(bout, "cluster_id", np.nan),
                    "bin_index": int(bin_index),
                    "bin_start_frame": int(bin_start),
                    "bin_end_frame": int(bin_end),
                    "time_since_immobile_seconds": float(elapsed_seconds),
                    "time_since_immobile_minutes": float(elapsed_seconds / 60.0),
                    "n_interactions_bin": n_total,
                    "n_interactions_as_antenna_bin": n_antenna,
                    "n_interactions_as_body_bin": n_body,
                    "n_interactions_cumulative": int(cumulative_total),
                    "n_interactions_as_antenna_cumulative": int(cumulative_antenna),
                    "n_interactions_as_body_cumulative": int(cumulative_body),
                    "woke": woke,
                    "mobility_observed": mobility_observed,
                }
            )
            if woke:
                break

    return pd.DataFrame(rows)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _fit_logistic_model(
    design: pd.DataFrame,
    *,
    predictors: list[str],
    outcome_col: str,
    ridge: float,
    max_iter: int,
    tol: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    cols = [outcome_col, *predictors]
    data = design[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if data.empty:
        raise ValueError("No rows available after dropping missing values")

    y = data[outcome_col].to_numpy(np.float64)
    if np.unique(y).size < 2:
        raise ValueError("Outcome has only one class")

    x_raw = data[predictors].to_numpy(np.float64)
    means = x_raw.mean(axis=0)
    stds = x_raw.std(axis=0)
    stds[stds == 0] = 1.0
    x = (x_raw - means) / stds
    x_design = np.column_stack([np.ones(len(x), dtype=np.float64), x])

    beta = np.zeros(x_design.shape[1], dtype=np.float64)
    ridge_vector = np.full_like(beta, float(ridge), dtype=np.float64)
    ridge_vector[0] = 0.0
    for _ in range(int(max_iter)):
        p = _sigmoid(x_design @ beta)
        weights = np.clip(p * (1.0 - p), 1e-9, None)
        gradient = x_design.T @ (y - p) - ridge_vector * beta
        hessian = (x_design.T * weights) @ x_design + np.diag(ridge_vector)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta = beta + step
        if float(np.max(np.abs(step))) < float(tol):
            break

    p = np.clip(_sigmoid(x_design @ beta), 1e-9, 1.0 - 1e-9)
    log_likelihood = float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    null_p = float(np.clip(y.mean(), 1e-9, 1.0 - 1e-9))
    null_ll = float(np.sum(y * np.log(null_p) + (1.0 - y) * np.log(1.0 - null_p)))
    n = int(len(y))
    k = int(len(beta))

    coef_rows = [
        {
            "term": "intercept",
            "coef_standardized": float(beta[0]),
            "coef_original_units": float(beta[0] - np.sum(beta[1:] * means / stds)),
            "odds_ratio_per_sd": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }
    ]
    for predictor, coef, mean, std in zip(predictors, beta[1:], means, stds):
        coef_rows.append(
            {
                "term": predictor,
                "coef_standardized": float(coef),
                "coef_original_units": float(coef / std),
                "odds_ratio_per_sd": float(np.exp(coef)),
                "mean": float(mean),
                "std": float(std),
            }
        )

    model = {
        "n_rows": n,
        "n_wake_bins": int(y.sum()),
        "n_predictors": len(predictors),
        "log_likelihood": log_likelihood,
        "log_loss": float(-log_likelihood / n),
        "aic": float(2 * k - 2 * log_likelihood),
        "bic": float(k * math.log(n) - 2 * log_likelihood),
        "mcfadden_r2": float(1.0 - log_likelihood / null_ll) if null_ll != 0 else np.nan,
        "formula_standardized": " + ".join(
            [f"{beta[0]:.3g}"] + [f"{coef:.3g}*z({name})" for name, coef in zip(predictors, beta[1:])]
        ),
    }
    if len(predictors) == 2 and beta[1] != 0:
        model["second_to_first_weight"] = float(beta[2] / beta[1])
    else:
        model["second_to_first_weight"] = np.nan
    return model, pd.DataFrame(coef_rows)


def fit_wake_logistic_regressions(
    design: pd.DataFrame,
    *,
    outcome_col: str = "woke",
    time_col: str = "time_since_immobile_seconds",
    interaction_col: str = "n_interactions_cumulative",
    ridge: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = [
        ("time_since_immobility", [time_col]),
        ("cumulative_interactions", [interaction_col]),
        ("time_plus_interactions", [time_col, interaction_col]),
    ]
    model_rows = []
    coef_tables = []
    for model_name, predictors in specs:
        try:
            model, coefs = _fit_logistic_model(
                design,
                predictors=predictors,
                outcome_col=outcome_col,
                ridge=float(ridge),
                max_iter=int(max_iter),
                tol=float(tol),
            )
        except ValueError as exc:
            model = {
                "n_rows": 0,
                "n_wake_bins": 0,
                "n_predictors": len(predictors),
                "log_likelihood": np.nan,
                "log_loss": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "mcfadden_r2": np.nan,
                "formula_standardized": str(exc),
                "second_to_first_weight": np.nan,
            }
            coefs = pd.DataFrame()
        model["model"] = model_name
        model["predictors"] = ", ".join(predictors)
        model_rows.append(model)
        if not coefs.empty:
            coefs["model"] = model_name
            coef_tables.append(coefs)

    model_table = pd.DataFrame(model_rows).sort_values("aic", na_position="last", kind="mergesort").reset_index(drop=True)
    coef_table = pd.concat(coef_tables, ignore_index=True) if coef_tables else pd.DataFrame()
    return model_table, coef_table


def plot_wake_regression_model_comparison(model_table: pd.DataFrame, *, metric: str = "aic"):
    plot_df = model_table.dropna(subset=[metric]).sort_values(metric, kind="mergesort")
    if plot_df.empty:
        raise ValueError(f"No finite {metric} values to plot")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(plot_df["model"], plot_df[metric], color="0.35")
    ax.set_ylabel(metric)
    ax.set_title("Wake prediction model comparison")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save_current_figures_if_enabled()
    return plot_df


def correlation_summary(df: pd.DataFrame, *, x_col: str, y_col: str) -> pd.Series:
    valid = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3:
        return pd.Series({"n": len(valid), "pearson_r": np.nan, "spearman_r": np.nan, "pearson_p": np.nan, "spearman_p": np.nan})
    pearson_r = valid[x_col].corr(valid[y_col], method="pearson")
    spearman_r = valid[x_col].corr(valid[y_col], method="spearman")
    pearson_p = np.nan
    spearman_p = np.nan
    try:
        from scipy.stats import pearsonr, spearmanr

        pearson_r, pearson_p = pearsonr(valid[x_col], valid[y_col])
        spearman_r, spearman_p = spearmanr(valid[x_col], valid[y_col])
    except Exception:
        pass
    return pd.Series(
        {
            "n": int(len(valid)),
            "pearson_r": float(pearson_r),
            "spearman_r": float(spearman_r),
            "pearson_p": float(pearson_p) if np.isfinite(pearson_p) else np.nan,
            "spearman_p": float(spearman_p) if np.isfinite(spearman_p) else np.nan,
        }
    )


def plot_immobility_interaction_correlation(
    bouts: pd.DataFrame,
    *,
    x_col: str = "n_interactions_total",
    y_col: str = "time_to_mobility_seconds",
    complete_only: bool = True,
    color_col: str = "cluster_id",
    log_x: bool = False,
    log_y: bool = False,
    title: str = "Interactions during immobility vs time to mobility",
) -> tuple[pd.DataFrame, pd.Series]:
    plot_df = bouts.copy()
    if complete_only and "mobility_observed" in plot_df.columns:
        plot_df = plot_df[plot_df["mobility_observed"]].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
    if plot_df.empty:
        raise ValueError("No bouts available for correlation plot after filtering")
    stats = correlation_summary(plot_df, x_col=x_col, y_col=y_col)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    if color_col in plot_df.columns:
        labels = sorted(plot_df[color_col].dropna().astype(str).unique())
        cmap = plt.get_cmap("tab10", max(1, len(labels)))
        for i, label in enumerate(labels):
            subset = plot_df[plot_df[color_col].astype(str) == label]
            ax.scatter(subset[x_col], subset[y_col], s=32, alpha=0.75, label=label, color=cmap(i))
        ax.legend(title=color_col, fontsize="small", loc="best")
    else:
        ax.scatter(plot_df[x_col], plot_df[y_col], s=32, alpha=0.75)

    if log_x:
        ax.set_xscale("symlog", linthresh=1)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(
        f"{title}\n"
        f"n={int(stats['n'])}, Spearman r={stats['spearman_r']:.3f}, Pearson r={stats['pearson_r']:.3f}"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save_current_figures_if_enabled()
    return plot_df, stats
