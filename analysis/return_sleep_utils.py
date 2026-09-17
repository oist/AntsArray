"""Return-event and sleeping-recipient analyses using cached sleep labels.

Times are global tracking frames. Position bins are one second; contact onsets
and sleep-state selection retain frame precision. An excursion is an observed
colony -> outside -> colony sequence, not an assumed task identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from camera_cal.region_paths import panorama_regions_path

from analysis import interaction_analysis_utils as ia
from analysis import sleep_motion_analysis_utils as sma
from analysis import sleep_motion_utils as sm
from analysis.trip_phenotyping_utils import _remove_short_state_flicker, _run_length_encoding


@dataclass(frozen=True)
class ReturnSettings:
    fps: float = 24.0
    position_bin_seconds: float = 1.0
    min_position_fraction: float = 0.25
    min_outside_seconds: float = 30.0
    min_colony_anchor_seconds: float = 5.0
    border_flicker_seconds: float = 3.0
    min_resource_seconds: float = 2.0
    recent_return_seconds: float = 300.0
    contact_gap_seconds: float = 2.0
    contact_min_detection_frames: int = 1
    crossing_max_gap_frames: int = 5
    prior_sleep_seconds: float = 10.0
    prior_contact_free_seconds: float = 2.0
    wake_sustain_seconds: float = 2.0
    wake_followup_seconds: float = 120.0
    match_clock_seconds: float = 1800.0
    match_distance_mm: float = 5.0
    density_radius_mm: float = 5.0
    match_density_difference: float = 2.0
    match_body_speed_mm_s: float = 0.2
    match_antenna_speed_mm_s: float = 0.3
    match_sleep_age_ratio: float = 2.0
    min_control_separation_seconds: float = 120.0
    random_state: int = 0

    @property
    def bin_frames(self) -> int:
        return round(self.fps * self.position_bin_seconds)

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if key != "random_state" and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be positive")
        if self.min_position_fraction > 1 or self.bin_frames < 1:
            raise ValueError("Invalid position coverage or bin duration")
        if not np.isclose(self.bin_frames, self.fps * self.position_bin_seconds):
            raise ValueError("Position bins must contain an integer number of frames")
        if self.match_sleep_age_ratio <= 1:
            raise ValueError("match_sleep_age_ratio must be greater than one")
        for name in ("contact_min_detection_frames", "crossing_max_gap_frames"):
            if int(getattr(self, name)) != getattr(self, name):
                raise ValueError(f"{name} must be an integer")


def fingerprint(path: Path) -> dict:
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cache_key(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def resolve_interaction_root(block_dir: Path) -> Path:
    """Use the published pipeline output, allowing older separately named runs."""
    canonical = Path(block_dir) / "interactions"
    legacy = Path(block_dir) / "interactions_skeleton_0p1mm"
    for path in (canonical, legacy):
        if (path / "run_manifest.json").is_file():
            return path
    return legacy if legacy.is_dir() and not canonical.is_dir() else canonical


def load_published_interaction_chunks(interaction_root: Path, tracks_root: Path, *,
                                      fps: float, start_clock_seconds: float) -> tuple[list[ia.InteractionChunk], dict]:
    """Require a complete, clock-aligned 0.1 mm full-skeleton fanout."""
    from tracking.colony.skeleton_contacts import CONTACT_GEOMETRY

    ready_path = interaction_root / "transfer_complete.ok"
    if not ready_path.is_file():
        raise FileNotFoundError(f"0.1 mm interaction rebuild is not published yet: {interaction_root}")
    ready = json.loads(ready_path.read_text())
    run = json.loads((interaction_root / "run_manifest.json").read_text())
    parameters = run["parameters"]
    if (parameters.get("geometry") != CONTACT_GEOMETRY
            or parameters.get("micro_interaction_distance_mm") != .1
            or parameters.get("directed") is not False):
        raise ValueError("Expected undirected full-skeleton contacts at 0.1 mm")
    if ready.get("run_id") != run["run_id"] or ready.get("n_chunks") != len(run["expected_files"]):
        raise ValueError("Interaction completion marker does not match the fanout manifest")
    missing = [root / name for name in run["expected_files"] for root in (interaction_root, tracks_root)
               if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing interaction/finished-track files:\n" + "\n".join(map(str, missing)))
    chunks = []
    for side in ("left", "right"):
        chunks.extend(ia.resolve_chunks(interaction_root, tracks_root, chunks="all", side=side, fps=fps))
    if any(chunk.recording_start_clock_seconds != start_clock_seconds for chunk in chunks):
        raise ValueError("Contact and sleep-label recording clocks differ")
    if not chunks or {c.interaction_path.name for c in chunks} != set(run["expected_files"]):
        raise ValueError("Interaction files do not match the completed fanout manifest")
    return chunks, run


def _bin_mean(values: np.ndarray, frame_min: int, n_bins: int, width: int) -> np.ndarray:
    starts = np.arange(n_bins, dtype=np.int64) * width
    lo = np.clip(starts - frame_min, 0, len(values))
    hi = np.clip(starts + width - frame_min, 0, len(values))
    valid = np.isfinite(values)
    sums = np.r_[0.0, np.cumsum(np.where(valid, values, 0.0), dtype=np.float64)]
    count = np.r_[0, np.cumsum(valid, dtype=np.int64)]
    number = count[hi] - count[lo]
    return np.divide(sums[hi] - sums[lo], number,
                     out=np.full(n_bins, np.nan), where=number > 0).astype(np.float32)


def _region_mask(x, y, region) -> np.ndarray:
    if region.shape == "rectangle":
        return ((x >= region.tracking_x_min_px) & (x <= region.tracking_x_max_px)
                & (y >= region.tracking_y_min_px) & (y <= region.tracking_y_max_px))
    return np.hypot(x - region.tracking_center_x_px, y - region.tracking_center_y_px) <= region.radius_px


def build_track_context(row, block_dir: Path, regions: pd.DataFrame,
                        settings: ReturnSettings, n_bins: int) -> dict[str, np.ndarray]:
    import pyarrow.dataset as ds

    width = settings.bin_frames
    track_path = block_dir / "stitched" / "per_track" / row.track_name
    dataset = ds.dataset(track_path, format="parquet")
    table = dataset.to_table(columns=["Frame", "TrackX", "TrackY"], filter=ds.field("Bodypoint") == 0)
    positions = table.to_pandas().dropna().drop_duplicates("Frame").sort_values("Frame")
    frames = positions.Frame.to_numpy(np.int64)
    x = positions.TrackX.to_numpy(float)
    y = positions.TrackY.to_numpy(float)
    bins = frames // width
    valid = (bins >= 0) & (bins < n_bins) & np.isfinite(x) & np.isfinite(y)
    bins, x, y = bins[valid], x[valid], y[valid]
    counts = np.bincount(bins, minlength=n_bins)
    has_position = counts >= max(1, int(np.ceil(width * settings.min_position_fraction)))
    selected = regions[regions.side == row.side]
    colony_rows = selected[selected.region_type == "colony"]
    if len(colony_rows) != 1:
        raise ValueError(f"Expected one colony annotation for {row.side}")
    colony = next(colony_rows.itertuples())
    inside_counts = np.bincount(bins, weights=_region_mask(x, y, colony), minlength=n_bins)
    inside = np.full(n_bins, -1, dtype=np.int8)
    inside[has_position] = (inside_counts[has_position] > counts[has_position] / 2).astype(np.int8)
    resource = np.zeros(len(bins), dtype=bool)
    for region in selected[selected.region_type.isin(["food", "water"])].itertuples():
        resource |= _region_mask(x, y, region)
    resource_counts = np.bincount(bins, weights=resource, minlength=n_bins)
    scale = float(colony.mm_per_pixel)
    context = {"inside": inside, "position_count": counts.astype(np.int16),
               "resource_seconds": (resource_counts / settings.fps).astype(np.float32)}
    for name, vector in (("x_mm", x), ("y_mm", y)):
        sums = np.bincount(bins, weights=vector * scale, minlength=n_bins)
        context[name] = np.divide(sums, counts, out=np.full(n_bins, np.nan), where=has_position).astype(np.float32)

    state = np.load(row.state_path, mmap_mode="r")
    sample_frames = np.arange(n_bins, dtype=np.int64) * width + width - 1
    context["state"] = sma.sample_dense(state, row.frame_min, sample_frames, missing=-1).astype(np.int8)
    last_not_sleep = np.maximum.accumulate(np.where(state != 1, np.arange(len(state)), -1))
    age = np.where(state == 1, (np.arange(len(state)) - last_not_sleep) / settings.fps, 0).astype(np.float32)
    context["sleep_age_seconds"] = sma.sample_dense(age, row.frame_min, sample_frames).astype(np.float32)
    motion_cache = sm.load_sleep_motion_cache(block_dir / "stitched" / "sleep_motion" / "per_track" / Path(row.track_name).stem)
    parameters = json.loads(row.classifier_parameters)
    for name, ids, percentile, fraction in (
        ("body_speed", sm.BODY_BODYPOINT_IDS, "body_percentile", "min_valid_bodypoint_fraction"),
        ("antenna_speed", sm.ANTENNA_BODYPOINT_IDS, "antenna_percentile", "min_valid_antenna_fraction"),
    ):
        vector = sm.aggregate_bodypoint_group_slice(
            motion_cache, 0, int(motion_cache.metadata["n_frames"]), bodypoint_ids=ids,
            bodypoint_percentile=parameters[percentile], min_valid_bodypoint_fraction=parameters[fraction],
            max_gap_frames=parameters["max_gap_frames"],
            max_bodypoint_speed_mm_s=parameters["max_bodypoint_speed_mm_s"],
        )
        context[name] = _bin_mean(vector, row.frame_min, n_bins, width)
    return context


def context_cache_paths(row, block_dir: Path, regions: pd.DataFrame,
                        settings: ReturnSettings, n_bins: int, cache_root: Path) -> tuple[Path, Path]:
    """Resolve the exact existing context cache without rebuilding raw motion."""
    source_settings = {"version": 2, "fps": settings.fps, "bin_frames": settings.bin_frames,
                       "min_position_fraction": settings.min_position_fraction, "n_bins": n_bins,
                       "regions": fingerprint(panorama_regions_path(block_dir)),
                       "region_sides": regions[["region_id", "region_type", "side"]].to_dict("records")}
    track_path = block_dir / "stitched" / "per_track" / row.track_name
    motion_root = block_dir / "stitched" / "sleep_motion" / "per_track" / Path(row.track_name).stem
    sources = {**source_settings, "track": fingerprint(track_path), "labels": fingerprint(Path(row.metadata_path)),
               "motion": fingerprint(motion_root / sm.METADATA_FILENAME)}
    key = cache_key(sources)
    stem = f"{Path(row.track_name).stem}_{key}"
    return cache_root / "context" / f"{stem}.npz", cache_root / "crossings" / f"{stem}_v1.npz"


def load_contexts(tracks: pd.DataFrame, block_dir: Path, regions: pd.DataFrame,
                  settings: ReturnSettings, cache_root: Path, *, force: bool = False) -> dict:
    settings.validate()
    n_bins = int(tracks.frame_max.max()) // settings.bin_frames + 1
    contexts = {}
    for index, row in enumerate(tracks.itertuples(), start=1):
        track_path = block_dir / "stitched" / "per_track" / row.track_name
        path, crossing_path = context_cache_paths(row, block_dir, regions, settings, n_bins, cache_root)
        if path.is_file() and not force:
            with np.load(path) as data:
                context = {name: data[name] for name in data.files}
        else:
            context = build_track_context(row, block_dir, regions, settings, n_bins)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **context)
        # Store sparse frame-resolved crossings separately so motion caches need
        # not be recomputed when refining the event timestamp.
        if crossing_path.is_file() and not force:
            with np.load(crossing_path) as data:
                context.update({name: data[name] for name in data.files})
        else:
            import pyarrow.dataset as ds

            positions = ds.dataset(track_path, format="parquet").to_table(
                columns=["Frame", "TrackX", "TrackY"], filter=ds.field("Bodypoint") == 0,
            ).to_pandas().dropna().drop_duplicates("Frame").sort_values("Frame")
            colony = next(regions[(regions.side == row.side) & (regions.region_type == "colony")].itertuples())
            crossing = position_crossings(positions.Frame.to_numpy(np.int64),
                                          _region_mask(positions.TrackX.to_numpy(), positions.TrackY.to_numpy(), colony))
            crossing_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(crossing_path, **crossing)
            context.update(crossing)
        contexts[(row.side, row.track_id)] = context
        if index % 10 == 0 or index == len(tracks):
            print(f"Position/motion context: {index}/{len(tracks)} tracks", flush=True)

    # Distances are computed separately by colony, with missing positions excluded.
    for side in tracks.side.unique():
        keys = [key for key in contexts if key[0] == side]
        x = np.stack([contexts[key]["x_mm"] for key in keys])
        y = np.stack([contexts[key]["y_mm"] for key in keys])
        density = np.full(x.shape, np.nan, dtype=np.float32)
        for start in range(0, n_bins, 1000):
            stop = min(n_bins, start + 1000)
            dx = x[:, None, start:stop] - x[None, :, start:stop]
            dy = y[:, None, start:stop] - y[None, :, start:stop]
            count = ((dx * dx + dy * dy) <= settings.density_radius_mm ** 2).sum(axis=1) - 1
            density[:, start:stop] = np.where(np.isfinite(x[:, start:stop]), count, np.nan)
        for index, key in enumerate(keys):
            contexts[key]["density"] = density[index]
    return contexts


def position_crossings(frames: np.ndarray, inside: np.ndarray) -> dict:
    """First observed anchor position on the other side, with its last observation."""
    changed = np.flatnonzero(inside[1:] != inside[:-1]) + 1
    return {"crossing_frames": frames[changed], "crossing_previous_frames": frames[changed - 1],
            "crossing_inside": inside[changed].astype(np.int8)}


def extract_returns(contexts: dict, clusters: pd.DataFrame, settings: ReturnSettings) -> pd.DataFrame:
    columns = ["return_id", "side", "track_id", "cluster_id", "exit_frame", "return_frame",
               "inside_stop_frame", "outside_seconds", "resource_seconds", "resource_visit",
               "binned_return_frame", "last_outside_frame", "crossing_uncertainty_seconds"]
    rows = []
    width = settings.bin_frames
    dt = settings.position_bin_seconds
    for ant in sma.normalize_clusters(clusters).itertuples():
        context = contexts[(ant.side, ant.track_id)]
        state = _remove_short_state_flicker(context["inside"], max(1, round(settings.border_flicker_seconds / dt)))
        starts, ends, values = _run_length_encoding(state)
        for j in range(1, len(values) - 1):
            if tuple(values[j-1:j+2]) != (1, 0, 1):
                continue
            outside_seconds = (ends[j] - starts[j] + 1) * dt
            anchors = min(ends[j-1] - starts[j-1] + 1, ends[j+1] - starts[j+1] + 1) * dt
            if outside_seconds < settings.min_outside_seconds or anchors < settings.min_colony_anchor_seconds:
                continue
            resource_seconds = float(context["resource_seconds"][starts[j]:ends[j]+1].sum())
            return_frame = int(starts[j+1] * width)
            binned_return_frame = return_frame
            exit_frame = int(starts[j] * width)
            last_outside_frame = return_frame - width
            if "crossing_frames" in context:
                frames = context["crossing_frames"]
                direction = context["crossing_inside"]
                exits = np.flatnonzero((direction == 0) & (frames >= exit_frame - settings.border_flicker_seconds * settings.fps)
                                      & (frames < exit_frame + width))
                if not len(exits):
                    continue
                exit_frame = int(frames[exits[-1]])
                # Find the first entry into the confirmed residence. Earlier
                # boundary touches followed by a long outside run are separate
                # visits, not the beginning of this residence.
                entries = np.flatnonzero((direction == 1) & (frames > exit_frame)
                                        & (frames < binned_return_frame + width))
                if not len(entries):
                    continue
                prior_exits = entries - 1
                valid = prior_exits >= 0
                separated = valid & (frames[entries] - frames[np.maximum(prior_exits, 0)] >= settings.border_flicker_seconds * settings.fps)
                if not separated.any():
                    continue
                entry = entries[np.flatnonzero(separated)[-1]]
                exit_frame = int(frames[entry - 1])
                return_frame = int(frames[entry])
                last_outside_frame = int(context["crossing_previous_frames"][entry])
                if return_frame - last_outside_frame > settings.crossing_max_gap_frames:
                    continue
                outside_seconds = (return_frame - exit_frame) / settings.fps
                if outside_seconds < settings.min_outside_seconds:
                    continue
                resource_seconds = float(context["resource_seconds"][exit_frame // width:(return_frame + width - 1) // width].sum())
            rows.append({"return_id": f"{ant.side}:{ant.track_id}:{return_frame}", "side": ant.side,
                         "track_id": ant.track_id, "cluster_id": ant.cluster_id,
                         "exit_frame": exit_frame, "return_frame": return_frame,
                         "inside_stop_frame": int((ends[j+1] + 1) * width),
                         "outside_seconds": float(outside_seconds), "resource_seconds": resource_seconds,
                         "resource_visit": resource_seconds >= settings.min_resource_seconds,
                         "binned_return_frame": binned_return_frame, "last_outside_frame": last_outside_frame,
                         "crossing_uncertainty_seconds": (return_frame - last_outside_frame) / settings.fps})
    return pd.DataFrame(rows, columns=columns)


def load_contact_bouts(chunks: list[ia.InteractionChunk], settings: ReturnSettings,
                       cache_root: Path, *, force: bool = False) -> tuple[pd.DataFrame, dict]:
    key = cache_key({"version": 3, "gap": settings.contact_gap_seconds, "fps": settings.fps,
                     "min_detection_frames": settings.contact_min_detection_frames,
                     "chunks": [{"offset": c.chunk_global_frame_offset, "n_frames": c.chunk_frame_count,
                                 **fingerprint(c.interaction_path)} for c in chunks]})
    path = cache_root / f"pair_contact_bouts_{key}.parquet"
    n_frames = max(c.chunk_global_frame_offset + c.chunk_frame_count for c in chunks)
    n_bins = (n_frames + settings.bin_frames - 1) // settings.bin_frames
    coverage = {side: np.zeros(n_bins, dtype=bool) for side in {c.side for c in chunks}}
    for c in chunks:
        start = c.chunk_global_frame_offset // settings.bin_frames
        stop = (c.chunk_global_frame_offset + c.chunk_frame_count) // settings.bin_frames
        coverage[c.side][start:stop] = True
    if path.is_file() and not force:
        return pd.read_parquet(path), coverage
    gap = round(settings.contact_gap_seconds * settings.fps)
    pieces = []
    for index, chunk in enumerate(chunks, start=1):
        import pyarrow.parquet as pq
        schema = set(pq.ParquetFile(chunk.interaction_path).schema_arrow.names)
        undirected = {"ant_a", "ant_b"} <= schema
        id_columns = ["ant_a", "ant_b"] if undirected else ["antenna_track_id", "body_track_id"]
        data = pd.read_parquet(chunk.interaction_path, columns=["Frame", *id_columns])
        data = data.dropna()
        f = data.Frame.to_numpy(np.int64) + chunk.chunk_global_frame_offset
        antenna = data[id_columns[0]].to_numpy(np.int64)
        body = data[id_columns[1]].to_numpy(np.int64)
        keep = antenna != body
        f, antenna, body = f[keep], antenna[keep], body[keep]
        if len(f):
            if f.min() < chunk.chunk_global_frame_offset or f.max() >= chunk.chunk_global_frame_offset + chunk.chunk_frame_count:
                raise ValueError(f"Contact frames outside chunk bounds: {chunk.interaction_path}")
            a, b = np.minimum(antenna, body), np.maximum(antenna, body)
            direction = np.zeros(len(a), np.int8) if undirected else np.where(antenna == a, 1, 2).astype(np.int8)
            order = np.lexsort((f, b, a))
            a, b, f, direction = a[order], b[order], f[order], direction[order]
            unique = np.r_[0, np.flatnonzero((a[1:] != a[:-1]) | (b[1:] != b[:-1]) | (f[1:] != f[:-1])) + 1]
            direction = np.bitwise_or.reduceat(direction, unique)
            a, b, f = a[unique], b[unique], f[unique]
            starts = np.r_[0, np.flatnonzero((a[1:] != a[:-1]) | (b[1:] != b[:-1]) | (np.diff(f) > gap)) + 1]
            ends = np.r_[starts[1:] - 1, len(f) - 1]
            pieces.append(pd.DataFrame({"side": chunk.side, "ant_a": a[starts], "ant_b": b[starts],
                                        "start_frame": f[starts], "end_frame": f[ends],
                                        "n_detection_frames": ends - starts + 1,
                                        "onset_direction": direction[starts]}))
        print(f"Contact bouts: {index}/{len(chunks)} chunks", flush=True)
    if not pieces:
        raise ValueError("No pair contacts in selected chunks")
    partial = pd.concat(pieces, ignore_index=True).sort_values(["side", "ant_a", "ant_b", "start_frame"])
    merged = []
    for (side, a, b), group in partial.groupby(["side", "ant_a", "ant_b"], sort=False):
        current = None
        for row in group.itertuples():
            lo = row.start_frame // settings.bin_frames
            connected = current is not None and row.start_frame - current["end_frame"] <= gap
            if connected:
                prev_bin = current["end_frame"] // settings.bin_frames
                connected = coverage[side][prev_bin:lo+1].all()
            if connected:
                current["end_frame"] = max(current["end_frame"], int(row.end_frame))
                current["n_detection_frames"] += int(row.n_detection_frames)
            else:
                if current is not None:
                    merged.append(current)
                current = {"side": side, "ant_a": int(a), "ant_b": int(b),
                           "start_frame": int(row.start_frame), "end_frame": int(row.end_frame),
                           "n_detection_frames": int(row.n_detection_frames),
                           "onset_direction": int(row.onset_direction)}
        if current is not None:
            merged.append(current)
    bouts = pd.DataFrame(merged)
    bouts = bouts[bouts.n_detection_frames >= settings.contact_min_detection_frames].reset_index(drop=True)
    bouts["is_new_onset"] = False
    for side, indices in bouts.groupby("side").groups.items():
        # The start of an observed segment is left-censored, not a known new contact.
        starts = bouts.loc[indices, "start_frame"].to_numpy(np.int64)
        lo = (starts - gap) // settings.bin_frames
        hi = starts // settings.bin_frames
        missing = np.r_[0, np.cumsum(~coverage[side])]
        valid = lo >= 0
        valid &= missing[hi+1] - missing[np.maximum(lo, 0)] == 0
        bouts.loc[indices, "is_new_onset"] = valid
    path.parent.mkdir(parents=True, exist_ok=True)
    bouts.to_parquet(path, index=False)
    return bouts, coverage


def attach_contact_context(contexts: dict, bouts: pd.DataFrame, coverage: dict,
                           settings: ReturnSettings) -> dict:
    focal = {}
    for key, context in contexts.items():
        side, ant = key
        selected = bouts[(bouts.side == side) & ((bouts.ant_a == ant) | (bouts.ant_b == ant))].copy()
        selected["partner_id"] = np.where(selected.ant_a == ant, selected.ant_b, selected.ant_a)
        selected = selected.sort_values("start_frame")
        focal[key] = selected
        n_bins = len(context["inside"])
        counts = np.zeros(n_bins, dtype=np.int32)
        starts = selected.loc[selected.is_new_onset, "start_frame"].to_numpy(np.int64)
        start_bins = starts // settings.bin_frames
        np.add.at(counts, start_bins[start_bins < n_bins], 1)
        activity_delta = np.zeros(n_bins + 1, dtype=np.int32)
        lo = selected.start_frame.to_numpy(np.int64) // settings.bin_frames
        hi = selected.end_frame.to_numpy(np.int64) // settings.bin_frames + 1
        np.add.at(activity_delta, np.clip(lo, 0, n_bins), 1)
        np.add.at(activity_delta, np.clip(hi, 0, n_bins), -1)
        context["contact_active"] = np.cumsum(activity_delta[:-1]) > 0
        context["contact_start_frames"] = np.sort(selected.start_frame.to_numpy(np.int64))
        context["contact_new_start_frames"] = np.sort(starts)
        context["contact_end_frames"] = np.sort(selected.end_frame.to_numpy(np.int64))
        context["contact_onsets"] = counts
        context["interaction_covered"] = np.zeros(n_bins, dtype=bool)
        count = min(n_bins, len(coverage[side]))
        context["interaction_covered"][:count] = coverage[side][:count]
    return focal


def _recent_return(returns: pd.DataFrame, side: str, ant: int, frame: int, settings: ReturnSettings):
    events = returns[(returns.side == side) & (returns.track_id == ant)]
    if events.empty:
        return None
    events = events.sort_values("return_frame")
    index = np.searchsorted(events.return_frame.to_numpy(), frame, side="right") - 1
    if index < 0:
        return None
    event = events.iloc[index]
    if frame >= event.inside_stop_frame or frame - event.return_frame > settings.recent_return_seconds * settings.fps:
        return None
    return event


def eligible_sleeping_contacts(tracks: pd.DataFrame, contexts: dict, bouts: pd.DataFrame,
                               returns: pd.DataFrame, settings: ReturnSettings) -> pd.DataFrame:
    rows = []
    lookup = {(row.side, row.track_id): row for row in tracks.itertuples()}
    prior = round(settings.prior_sleep_seconds * settings.fps)
    for key, row in lookup.items():
        context = contexts[key]
        state = np.load(row.state_path, mmap_mode="r")
        non_sleep_frames = np.flatnonzero(state != 1) + row.frame_min
        contacts = bouts[(bouts.side == row.side) & ((bouts.ant_a == row.track_id) | (bouts.ant_b == row.track_id)) & bouts.is_new_onset]
        frames = contacts.start_frame.to_numpy(np.int64)
        sleep_counts = sma.interval_counts(state == 1, row.frame_min, frames - prior, frames)
        for event, n_sleep in zip(contacts.itertuples(), sleep_counts):
            if n_sleep != prior:
                continue
            frame = int(event.start_frame)
            previous_bin = frame // settings.bin_frames - 1
            if previous_bin < 0 or context["inside"][previous_bin] != 1:
                continue
            pre_bins = max(1, int(np.ceil(settings.prior_contact_free_seconds / settings.position_bin_seconds)))
            lo = previous_bin - pre_bins + 1
            if lo < 0 or context["contact_active"][lo:previous_bin+1].any():
                continue
            if not context["interaction_covered"][lo:previous_bin+1].all():
                continue
            partner = int(event.ant_b if event.ant_a == row.track_id else event.ant_a)
            partner_context = contexts.get((row.side, partner))
            if partner_context is None or partner_context["inside"][previous_bin] != 1:
                continue
            returning = _recent_return(returns, row.side, partner, frame, settings)
            direction_bit = 1 if partner == event.ant_a else 2
            record = {"event_id": f"{row.side}:{row.track_id}:{partner}:{frame}",
                      "side": row.side, "track_id": row.track_id, "cluster_id": row.cluster_id,
                      "partner_id": partner, "frame": frame,
                      "condition": "Recent return contact" if returning is not None else "Other contact",
                      "return_id": returning.return_id if returning is not None else "",
                      "resource_visit": bool(returning.resource_visit) if returning is not None else False,
                      "seconds_since_return": (frame - returning.return_frame) / settings.fps if returning is not None else np.nan,
                      "recipient_body_contact": (bool(event.onset_direction & direction_bit)
                                                 if event.onset_direction else pd.NA)}
            last_non_sleep = np.searchsorted(non_sleep_frames, frame, side="left") - 1
            record["sleep_bout_start_frame"] = int(non_sleep_frames[last_non_sleep] + 1) if last_non_sleep >= 0 else row.frame_min
            for column in ("x_mm", "y_mm", "density", "body_speed", "antenna_speed", "sleep_age_seconds"):
                record[column] = float(context[column][previous_bin])
            rows.append(record)
    columns = ["event_id", "side", "track_id", "cluster_id", "partner_id", "frame", "condition", "return_id",
               "resource_visit", "seconds_since_return", "recipient_body_contact", "x_mm", "y_mm", "density",
               "body_speed", "antenna_speed", "sleep_age_seconds", "sleep_bout_start_frame"]
    return pd.DataFrame(rows, columns=columns).sort_values(["side", "track_id", "frame"])


def _match_scores(event: pd.Series, candidates: pd.DataFrame, settings: ReturnSettings) -> np.ndarray:
    distance = np.hypot(candidates.x_mm - event.x_mm, candidates.y_mm - event.y_mm).to_numpy()
    clock = np.abs(candidates.frame.to_numpy() - event.frame) / settings.fps
    density = np.abs(candidates.density.to_numpy() - event.density)
    body = np.abs(candidates.body_speed.to_numpy() - event.body_speed)
    antenna = np.abs(candidates.antenna_speed.to_numpy() - event.antenna_speed)
    age = np.abs(np.log(np.maximum(1, np.minimum(candidates.sleep_age_seconds, 300)) /
                        max(1, min(event.sleep_age_seconds, 300)))).to_numpy()
    scaled = np.column_stack((distance / settings.match_distance_mm, clock / settings.match_clock_seconds,
                              density / settings.match_density_difference, body / settings.match_body_speed_mm_s,
                              antenna / settings.match_antenna_speed_mm_s, age / np.log(settings.match_sleep_age_ratio)))
    valid = np.isfinite(scaled).all(axis=1) & (scaled <= 1).all(axis=1)
    valid &= clock >= settings.min_control_separation_seconds
    return np.where(valid, np.square(scaled).sum(axis=1), np.inf)


def _no_contact_candidates(context: dict, settings: ReturnSettings) -> pd.DataFrame:
    n = len(context["state"])
    pre = max(1, int(np.ceil(settings.prior_contact_free_seconds / settings.position_bin_seconds)))
    prior_contact = np.r_[0, np.cumsum(context["contact_active"], dtype=np.int64)]
    prior_missing = np.r_[0, np.cumsum(~context["interaction_covered"], dtype=np.int64)]
    bins = np.arange(pre, n - 1, max(1, round(5 / settings.position_bin_seconds)))
    before = bins - 1
    valid = (context["state"][before] == 1) & (context["inside"][before] == 1)
    valid &= context["sleep_age_seconds"][before] >= settings.prior_sleep_seconds
    valid &= prior_contact[bins] - prior_contact[bins-pre] == 0
    valid &= prior_missing[bins+1] - prior_missing[bins-pre] == 0
    frames = bins * settings.bin_frames
    active_at_trigger = (np.searchsorted(context["contact_start_frames"], frames, side="right")
                         - np.searchsorted(context["contact_end_frames"], frames, side="left"))
    valid &= active_at_trigger == 0
    bins, before = bins[valid], before[valid]
    table = pd.DataFrame({"frame": bins * settings.bin_frames})
    for column in ("x_mm", "y_mm", "density", "body_speed", "antenna_speed", "sleep_age_seconds"):
        table[column] = context[column][before]
    return table


def match_sleeping_controls(events: pd.DataFrame, contexts: dict, settings: ReturnSettings,
                            *, resource_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Risk-set matching uses only pre-trigger state and covariates.

    Control times may have future contacts; follow-up is censored at the next
    onset for both groups. No future wake/sleep outcome enters matching.
    """
    returning = events[events.condition == "Recent return contact"]
    if resource_only:
        returning = returning[returning.resource_visit]
    returning = returning.drop_duplicates(["side", "track_id", "sleep_bout_start_frame"])
    triggers, diagnostics = [], []
    rng = np.random.default_rng(settings.random_state)
    for key, ant_events in returning.groupby(["side", "track_id"]):
        context = contexts[key]
        pool = _no_contact_candidates(context, settings)
        other = events[(events.side == key[0]) & (events.track_id == key[1]) & (events.condition == "Other contact")].copy()
        used_controls, used_others = [], set()
        # Random order prevents systematic preference for early-night events.
        for event_index in rng.permutation(ant_events.index.to_numpy()):
            event = ant_events.loc[event_index]
            scores = _match_scores(event, pool, settings)
            for previous in used_controls:
                scores[np.abs(pool.frame.to_numpy() - previous) < settings.min_control_separation_seconds * settings.fps] = np.inf
            eligible = np.isfinite(scores)
            diagnostic = {"event_id": event.event_id, "side": key[0], "track_id": key[1],
                          "frame": event.frame, "n_eligible_controls": int(eligible.sum()), "matched": bool(eligible.any())}
            diagnostics.append(diagnostic)
            if not eligible.any():
                continue
            control = pool.iloc[int(np.argmin(scores))]
            used_controls.append(int(control.frame))
            exposed = event.to_dict()
            exposed["match_id"] = event.event_id
            exposed["match_score"] = 0.0
            triggers.append(exposed)
            control_record = event.to_dict()
            control_record.update(control.to_dict())
            control_record.update({"condition": "No contact control", "partner_id": -1,
                                   "return_id": "", "match_id": event.event_id,
                                   "match_score": float(np.min(scores)), "seconds_since_return": np.nan,
                                   "recipient_body_contact": False})
            triggers.append(control_record)
            if len(other):
                other_scores = _match_scores(event, other, settings)
                other_scores[other.event_id.isin(used_others).to_numpy()] = np.inf
                if np.isfinite(other_scores).any():
                    comparator = other.iloc[int(np.argmin(other_scores))]
                    used_others.add(comparator.event_id)
                    record = comparator.to_dict()
                    record.update({"match_id": event.event_id, "match_score": float(np.min(other_scores))})
                    triggers.append(record)
    trigger_columns = [*events.columns, "match_id", "match_score"]
    diagnostic_columns = ["event_id", "side", "track_id", "frame", "n_eligible_controls", "matched"]
    return pd.DataFrame(triggers, columns=trigger_columns), pd.DataFrame(diagnostics, columns=diagnostic_columns)


def trigger_state_curves(triggers: pd.DataFrame, tracks: pd.DataFrame, contexts: dict,
                         settings: ReturnSettings, *, pre_seconds: int = 30, post_seconds: int = 120) -> pd.DataFrame:
    rows = []
    lags = np.arange(-pre_seconds, post_seconds, settings.position_bin_seconds)
    track_lookup = {(row.side, row.track_id): row for row in tracks.itertuples()}
    for key, group in triggers.groupby(["side", "track_id"]):
        row = track_lookup[key]
        context = contexts[key]
        state = np.load(row.state_path, mmap_mode="r")
        known_prefix = np.r_[0, np.cumsum(state >= 0, dtype=np.int64)]
        awake_prefix = np.r_[0, np.cumsum(state == 0, dtype=np.int64)]
        for event in group.itertuples():
            starts = np.rint(event.frame + lags * settings.fps).astype(np.int64)
            stops = starts + settings.bin_frames
            lo = np.clip(starts - row.frame_min, 0, len(state))
            hi = np.clip(stops - row.frame_min, 0, len(state))
            known = known_prefix[hi] - known_prefix[lo]
            awake = awake_prefix[hi] - awake_prefix[lo]
            fraction = np.divide(awake, known, out=np.full(len(lags), np.nan), where=known >= settings.bin_frames / 2)
            bins = starts // settings.bin_frames
            valid = (bins >= 0) & (bins < len(context["state"]))
            data = {"side": key[0], "track_id": key[1], "cluster_id": event.cluster_id,
                    "match_id": event.match_id, "condition": event.condition,
                    "lag_seconds": lags + settings.position_bin_seconds / 2,
                    "awake_fraction": fraction, "classified_fraction": known / settings.bin_frames}
            for metric in ("body_speed", "antenna_speed"):
                vector = np.full(len(lags), np.nan)
                vector[valid] = context[metric][bins[valid]]
                data[metric] = vector
            rows.append(pd.DataFrame(data))
    columns = ["side", "track_id", "cluster_id", "match_id", "condition", "lag_seconds", "awake_fraction",
               "classified_fraction", "body_speed", "antenna_speed"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def waking_outcomes(triggers: pd.DataFrame, tracks: pd.DataFrame, contexts: dict,
                    focal_contacts: dict, settings: ReturnSettings) -> pd.DataFrame:
    """Time to sustained wake; censor on lost labels, exit, or a subsequent contact."""
    lookup = {(r.side, r.track_id): r for r in tracks.itertuples()}
    rows = []
    horizon = round(settings.wake_followup_seconds * settings.fps)
    sustain = max(1, round(settings.wake_sustain_seconds * settings.fps))
    for key, group in triggers.groupby(["side", "track_id"]):
        row = lookup[key]
        state = np.load(row.state_path, mmap_mode="r")
        context = contexts[key]
        onsets = focal_contacts[key].loc[lambda x: x.is_new_onset, "start_frame"].to_numpy(np.int64)
        for trigger in group.itertuples():
            frame = int(trigger.frame)
            end = min(frame + horizon, row.frame_max + 1)
            reason = "horizon" if end == frame + horizon else "end of labels"
            next_index = np.searchsorted(onsets, frame, side="right")
            if next_index < len(onsets) and onsets[next_index] < end:
                end, reason = int(onsets[next_index]), "next contact"
            start_bin = frame // settings.bin_frames
            stop_bin = min(len(context["inside"]), (end + settings.bin_frames - 1) // settings.bin_frames)
            valid = (context["inside"][start_bin:stop_bin] == 1) & context["interaction_covered"][start_bin:stop_bin]
            missing_bin = np.flatnonzero(~valid)
            if len(missing_bin):
                end, reason = min(end, max(frame, (start_bin + int(missing_bin[0])) * settings.bin_frames)), "position/interaction coverage"
            values = np.asarray(state[max(0, frame-row.frame_min):max(0, end-row.frame_min)])
            unknown = np.flatnonzero(values < 0)
            if len(unknown):
                end, reason = frame + int(unknown[0]), "unknown sleep label"
                values = values[:unknown[0]]
            starts, ends, states = _run_length_encoding(values)
            wakes = np.flatnonzero((states == 0) & (ends - starts + 1 >= sustain))
            observed = bool(len(wakes))
            duration = (int(starts[wakes[0]]) + sustain) / settings.fps if observed else max(0, end-frame) / settings.fps
            rows.append({"match_id": trigger.match_id, "side": key[0], "track_id": key[1],
                         "cluster_id": trigger.cluster_id, "condition": trigger.condition,
                         "frame": frame, "duration_seconds": duration, "wake_observed": observed,
                         "censor_reason": "wake" if observed else reason})
    columns = ["match_id", "side", "track_id", "cluster_id", "condition", "frame", "duration_seconds",
               "wake_observed", "censor_reason"]
    return pd.DataFrame(rows, columns=columns)


def bootstrap_ant_mean(table: pd.DataFrame, *, value: str, group_cols: list[str],
                       n_bootstrap: int = 500, random_state: int = 0) -> pd.DataFrame:
    """Average repeated events within ant first, then bootstrap whole ants."""
    ant = table.groupby([*group_cols, "track_id"], as_index=False)[value].mean()
    rng = np.random.default_rng(random_state)
    rows = []
    for key, group in ant.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        values = group[value].dropna().to_numpy(float)
        record = dict(zip(group_cols, key))
        record.update({"mean": float(values.mean()) if len(values) else np.nan, "n_ants": len(values),
                       "lower": np.nan, "upper": np.nan})
        if len(values) >= 5:
            draws = rng.choice(values, (n_bootstrap, len(values)), replace=True).mean(axis=1)
            record["lower"], record["upper"] = np.quantile(draws, [0.025, 0.975])
        rows.append(record)
    return pd.DataFrame(rows, columns=[*group_cols, "mean", "n_ants", "lower", "upper"])


def paired_wake_effects(outcomes: pd.DataFrame, *, horizons=(5, 10, 30, 60, 120),
                        random_state: int = 0) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        data = outcomes.copy()
        observed = data.wake_observed & (data.duration_seconds <= horizon)
        known_no_wake = data.duration_seconds >= horizon
        data["outcome"] = np.where(observed, 1.0, np.where(known_no_wake, 0.0, np.nan))
        wide = data.pivot(index=["side", "track_id", "match_id"], columns="condition", values="outcome")
        for comparator in ("No contact control", "Other contact"):
            if comparator not in wide or "Recent return contact" not in wide:
                continue
            pairs = wide[["Recent return contact", comparator]].dropna().reset_index()
            pairs["difference"] = pairs["Recent return contact"] - pairs[comparator]
            summary = bootstrap_ant_mean(pairs, value="difference", group_cols=["side"], random_state=random_state)
            for row in summary.to_dict("records"):
                row.update({"horizon_seconds": horizon, "comparator": comparator,
                            "n_complete_pairs": int((pairs.side == row["side"]).sum())})
                rows.append(row)
    return pd.DataFrame(rows, columns=["side", "mean", "n_ants", "lower", "upper", "horizon_seconds", "comparator", "n_complete_pairs"])


def return_activity_curves(returns: pd.DataFrame, contexts: dict, settings: ReturnSettings,
                           *, pre_seconds: int = 120, post_seconds: int = 600,
                           curve_bin_seconds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, latency = [], []
    dt = settings.position_bin_seconds
    span = max(1, round(curve_bin_seconds / dt))
    sleep_sustain = max(1, round(10 / dt))
    for event in returns.itertuples():
        context = contexts[(event.side, event.track_id)]
        trigger_bin = event.return_frame // settings.bin_frames
        inside_stop = min(len(context["inside"]), event.inside_stop_frame // settings.bin_frames)
        for lag in range(-pre_seconds, post_seconds, curve_bin_seconds):
            lo = int(np.ceil((event.return_frame + lag * settings.fps) / settings.bin_frames - 0.5))
            hi = lo + span
            if lo < 0 or hi > len(context["inside"]):
                continue
            # Stop post-return exposure at the next exit or unobserved state run.
            if lag >= 0 and hi > inside_stop:
                continue
            row = {"return_id": event.return_id, "side": event.side, "track_id": event.track_id,
                   "cluster_id": event.cluster_id, "trip_type": "All returns",
                   "lag_seconds": lag + curve_bin_seconds / 2}
            position_valid = context["inside"][lo:hi] >= 0
            for name in ("body_speed", "antenna_speed"):
                values = context[name][lo:hi]
                valid = np.isfinite(values) & position_valid
                row[name] = float(values[valid].mean()) if valid.sum() >= span / 2 else np.nan
            state = context["state"][lo:hi]
            known = (state >= 0) & position_valid
            row["awake_fraction"] = float((state[known] == 0).mean()) if known.sum() >= span / 2 else np.nan
            covered = context["interaction_covered"][lo:hi] & position_valid
            if "contact_new_start_frames" in context:
                onsets = context["contact_new_start_frames"]
                start = event.return_frame + lag * settings.fps
                stop = start + curve_bin_seconds * settings.fps
                selected = onsets[np.searchsorted(onsets, start):np.searchsorted(onsets, stop)]
                bins = selected // settings.bin_frames
                count = (context["interaction_covered"][bins] & (context["inside"][bins] >= 0)).sum()
            else:
                count = context["contact_onsets"][lo:hi][covered].sum()
            row["contacts_per_minute"] = float(count * 60 / (covered.sum() * dt)) if covered.sum() >= span / 2 else np.nan
            rows.append(row)
        stop = min(inside_stop, trigger_bin + round(post_seconds / dt))
        states = context["state"][trigger_bin:stop]
        unknown = np.flatnonzero(states < 0)
        reason = "exit/end of position coverage" if stop == inside_stop else "horizon"
        if len(unknown):
            states = states[:unknown[0]]
            reason = "unknown sleep label"
        starts, ends, values = _run_length_encoding(states)
        sleep = np.flatnonzero((values == 1) & (ends - starts + 1 >= sleep_sustain))
        observed = bool(len(sleep))
        time = (int(starts[sleep[0]]) + sleep_sustain) * dt if observed else len(states) * dt
        latency.append({"return_id": event.return_id, "side": event.side, "track_id": event.track_id,
                        "trip_type": "All returns",
                        "duration_seconds": time, "sleep_observed": observed,
                        "censor_reason": "sleep" if observed else reason})
    columns = ["return_id", "side", "track_id", "cluster_id", "trip_type", "lag_seconds", "body_speed",
               "antenna_speed", "awake_fraction", "contacts_per_minute"]
    latency_columns = ["return_id", "side", "track_id", "trip_type", "duration_seconds", "sleep_observed", "censor_reason"]
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(latency, columns=latency_columns)


def return_early_late_effects(curves: pd.DataFrame, *, random_state: int = 0) -> pd.DataFrame:
    """Pair early and late resident windows from the same return before averaging ants."""
    columns = ["side", "trip_type", "mean", "n_ants", "lower", "upper", "metric", "n_returns"]
    if curves.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for metric in ("body_speed", "antenna_speed", "contacts_per_minute", "awake_fraction"):
        data = curves.copy()
        data["period"] = np.where(data.lag_seconds.between(0, 60), "early", np.where(data.lag_seconds.between(300, 600), "late", "exclude"))
        data = data[data.period != "exclude"]
        wide = data.groupby(["side", "track_id", "return_id", "trip_type", "period"])[metric].agg(["mean", "count"]).unstack("period")
        if "early" not in wide["mean"] or "late" not in wide["mean"]:
            continue
        # At least half of each nominal interval must be observed.
        keep = (wide["count"]["early"] >= 6) & (wide["count"]["late"] >= 30)
        pairs = wide.loc[keep, "mean"].dropna().reset_index()
        pairs["difference"] = pairs.early - pairs.late
        summary = bootstrap_ant_mean(pairs, value="difference", group_cols=["side", "trip_type"], random_state=random_state)
        for row in summary.to_dict("records"):
            row.update({"metric": metric, "n_returns": int(((pairs.side == row["side"]) & (pairs.trip_type == row["trip_type"])).sum())})
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def plot_return_curves(curves: pd.DataFrame, *, random_state: int = 0, side: str | None = None):
    """Plot pooled returns; selecting one colony gives a horizontal 1 × 3 figure."""
    if side not in (None, "left", "right"):
        raise ValueError("side must be left or right")
    metrics = [("body_speed", "Body motion (mm/s)"), ("antenna_speed", "Antenna motion (mm/s)"),
               ("contacts_per_minute", "New pair contacts / min")]
    sides = (side,) if side is not None else ("left", "right")
    single_colony = side is not None
    prefix = f"{side.capitalize()} colony — " if single_colony else ""
    if single_colony:
        fig, panels = plt.subplots(1, len(metrics), figsize=(15, 4.5), sharex=True, layout="constrained")
        axes = panels.reshape(-1, 1)
    else:
        fig, axes = plt.subplots(len(metrics), 2, figsize=(13, 9), sharex=True, layout="constrained")
    summaries = []
    colors = {"All returns": "#087e8b"}
    for j, side in enumerate(sides):
        for i, (metric, label) in enumerate(metrics):
            ax = axes[i, j]
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            if not single_colony and i == 0:
                ax.set_title(f"{side.capitalize()} colony")
            if curves.empty or not (curves.side == side).any():
                ax.text(0.5, 0.5, "No qualifying returns", transform=ax.transAxes, ha="center")
                continue
            summary = bootstrap_ant_mean(curves[curves.side == side], value=metric,
                                         group_cols=["side", "trip_type", "lag_seconds"], random_state=random_state)
            summaries.append(summary.assign(metric=metric))
            for trip_type, group in summary.groupby("trip_type"):
                ax.plot(group.lag_seconds, group["mean"], label=trip_type, color=colors[trip_type])
                ax.fill_between(group.lag_seconds, group.lower, group.upper, color=colors[trip_type], alpha=0.17)
            if i == 0:
                ax.legend(fontsize=8)
    for ax in (axes.flat if single_colony else axes[-1]):
        ax.set_xlabel("Time since first observed\noutside-to-inside crossing (s)")
    fig.suptitle(prefix + "All colony returns pooled; 95% ant bootstrap where n >= 5")
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    return fig, summary


def plot_recipient_curves(curves: pd.DataFrame, *, random_state: int = 0, side: str | None = None,
                          xlim: tuple[float, float] | None = None):
    """Selecting a colony shows three responses in one row; counts remain in the summary.

    ``xlim`` crops the display only, retaining the complete follow-up in the table.
    """
    if side not in (None, "left", "right"):
        raise ValueError("side must be left or right")
    metrics = [("awake_fraction", "Awake fraction"), ("body_speed", "Body motion (mm/s)"),
               ("antenna_speed", "Antenna motion (mm/s)"), ("awake_fraction", "Contributing ants")]
    sides = (side,) if side is not None else ("left", "right")
    single_colony = side is not None
    prefix = f"{side.capitalize()} colony — " if single_colony else ""
    if single_colony:
        metrics = metrics[:3]
        fig, panels = plt.subplots(1, len(metrics), figsize=(15, 4.5), sharex=True, layout="constrained")
        axes = panels.reshape(-1, 1)
    else:
        fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True, layout="constrained")
    colors = {"Recent return contact": "#c04c45", "No contact control": "#267f9a", "Other contact": "#7a7044"}
    summaries = []
    for j, side in enumerate(sides):
        for i, (metric, label) in enumerate(metrics):
            ax = axes[i, j]
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.grid(alpha=0.2)
            ax.set_ylabel(label)
            if xlim is not None:
                ax.set_xlim(*xlim)
            if i == 0:
                if not single_colony:
                    ax.set_title(f"{side.capitalize()} colony")
                ax.set_ylim(0, 1)
            if i == 3:
                ax.set_ylim(0, None)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if curves.empty or not (curves.side == side).any():
                ax.text(0.5, 0.5, "No matched sleeping recipients", transform=ax.transAxes, ha="center")
                continue
            summary = bootstrap_ant_mean(curves[curves.side == side], value=metric,
                                         group_cols=["side", "condition", "lag_seconds"], random_state=random_state)
            if i < 3:
                summaries.append(summary.assign(metric=metric))
            for condition, group in summary.groupby("condition"):
                ax.plot(group.lag_seconds, group.n_ants if i == 3 else group["mean"], label=condition, color=colors[condition])
                if i < 3:
                    ax.fill_between(group.lag_seconds, group.lower, group.upper, color=colors[condition], alpha=0.17)
            if i == 0:
                ax.legend(fontsize=8)
    for ax in (axes.flat if single_colony else axes[-1]):
        ax.set_xlabel("Time since new contact / matched control (s)")
    fig.suptitle(prefix + "Previously sleeping recipients: observed response, including later contacts")
    return fig, pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def plot_survival(outcomes: pd.DataFrame, *, event_column: str, group_column: str, title: str,
                  ylabel: str, horizon: float):
    from statsmodels.duration.survfunc import SurvfuncRight

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, layout="constrained")
    for j, side in enumerate(("left", "right")):
        if outcomes.empty:
            axes[0, j].text(0.5, 0.5, "No qualifying events", ha="center", transform=axes[0, j].transAxes)
            continue
        for name, group in outcomes[outcomes.side == side].groupby(group_column):
            group = group[group.duration_seconds > 0]
            if group.empty:
                continue
            sf = SurvfuncRight(group.duration_seconds.to_numpy(), group[event_column].to_numpy(int))
            event_risk = (group.duration_seconds.to_numpy()[:, None] >= sf.surv_times).sum(axis=0)
            supported = event_risk >= 5
            t = np.r_[0, sf.surv_times[supported]]
            probability = np.r_[0, 1 - sf.surv_prob[supported]]
            line, = axes[0, j].step(t, probability, where="post", label=f"{name} (n={len(group)})")
            grid = np.linspace(0, horizon, 121)
            at_risk = (group.duration_seconds.to_numpy()[:, None] >= grid).sum(axis=0)
            axes[1, j].plot(grid, at_risk, color=line.get_color())
        axes[0, j].set(title=f"{side.capitalize()} colony", ylabel=ylabel, ylim=(0, 1))
        axes[0, j].legend(fontsize=8)
        risk_label = "Returns awaiting sustained sleep\nand still followed" if event_column == "sleep_observed" else "Contacts awaiting sustained wake\nand still followed"
        axes[1, j].set(ylabel=risk_label, xlabel="Time (s)", xlim=(0, horizon), ylim=(0, None))
        for ax in axes[:, j]:
            ax.grid(alpha=0.2)
    fig.suptitle(title + "\nCurves stop below 5 still-followed events without the outcome", fontsize=12)
    return fig


def plot_matching_diagnostics(triggers: pd.DataFrame, diagnostics: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    balance = []
    if not triggers.empty:
        event = triggers[triggers.condition == "Recent return contact"].set_index("match_id")
        control = triggers[triggers.condition == "No contact control"].set_index("match_id")
        for name in ("body_speed", "antenna_speed", "density", "sleep_age_seconds", "x_mm", "y_mm"):
            paired = event[["side", name]].join(control[[name]], rsuffix="_control")
            for match_id, row in paired.iterrows():
                balance.append({"match_id": match_id, "side": row.side, "feature": name,
                                "exposed": row[name], "control": row[name + "_control"]})
    balance_table = pd.DataFrame(balance, columns=["match_id", "side", "feature", "exposed", "control"])
    for ax, feature in zip(axes.ravel(), ("body_speed", "antenna_speed", "density", "sleep_age_seconds", "x_mm", "y_mm")):
        for side, group in balance_table[balance_table.feature == feature].groupby("side"):
            ax.scatter(group.control, group.exposed, s=12, alpha=0.6, label=side)
        if not balance_table.empty:
            limits = ax.get_xlim()
            ax.plot(limits, limits, color="gray", linestyle="--", linewidth=0.8)
        ax.set(title=feature, xlabel="No-contact control", ylabel="Recent-return contact")
    axes[0, 0].legend(fontsize=8)
    total = len(diagnostics)
    matched = int(diagnostics.matched.sum()) if total else 0
    fig.suptitle(f"Pre-contact matching balance: {matched}/{total} eligible events matched")
    return fig, balance_table


def plot_paired_wake_effects(effects: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, side in zip(axes, ("left", "right")):
        if not effects.empty:
            for comparator, group in effects[effects.side == side].groupby("comparator"):
                ax.plot(group.horizon_seconds, group["mean"], marker="o", label=comparator)
                ax.fill_between(group.horizon_seconds, group.lower, group.upper, alpha=0.2)
                for row in group.itertuples():
                    ax.annotate(f"n={row.n_ants}", (row.horizon_seconds, row.mean),
                                xytext=(0, 6 if comparator == "No contact control" else -12),
                                textcoords="offset points", fontsize=7, ha="center")
            ax.legend(fontsize=8)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set(title=f"{side.capitalize()} colony", xlabel="Follow-up (s)", ylabel="Wake risk: return contact minus control")
        ax.grid(alpha=0.2)
    fig.suptitle("Paired wake-risk difference; n = recipient ants, intervals only when n >= 5")
    return fig


def return_contact_summary(returns: pd.DataFrame, focal_contacts: dict, sleeping_events: pd.DataFrame,
                           settings: ReturnSettings) -> pd.DataFrame:
    rows = []
    for event in returns.itertuples():
        contacts = focal_contacts[(event.side, event.track_id)]
        stop = min(event.inside_stop_frame, event.return_frame + round(60 * settings.fps))
        selected = contacts[contacts.is_new_onset & (contacts.start_frame >= event.return_frame) & (contacts.start_frame < stop)]
        rows.append({"return_id": event.return_id, "side": event.side, "track_id": event.track_id,
                     "resource_visit": event.resource_visit, "resident_followup_seconds": (stop-event.return_frame) / settings.fps,
                     "new_contacts_first_minute": len(selected), "distinct_partners_first_minute": selected.partner_id.nunique(),
                     "eligible_sleeping_recipients_first_five_minutes": int((sleeping_events.return_id == event.return_id).sum())})
    return pd.DataFrame(rows, columns=["return_id", "side", "track_id", "resource_visit", "resident_followup_seconds",
                                       "new_contacts_first_minute", "distinct_partners_first_minute",
                                       "eligible_sleeping_recipients_first_five_minutes"])


def plot_return_examples(returns: pd.DataFrame, contexts: dict, regions: pd.DataFrame, settings: ReturnSettings):
    from matplotlib.patches import Circle, Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout="constrained")
    example_rows = []
    for ax, side in zip(axes, ("left", "right")):
        for region in regions[regions.side == side].itertuples():
            scale = region.mm_per_pixel
            if region.shape == "rectangle":
                patch = Rectangle((region.tracking_x_min_px * scale, region.tracking_y_min_px * scale),
                                  (region.tracking_x_max_px-region.tracking_x_min_px)*scale,
                                  (region.tracking_y_max_px-region.tracking_y_min_px)*scale,
                                  fill=False, edgecolor="black", linewidth=1.4)
            else:
                patch = Circle((region.tracking_center_x_px*scale, region.tracking_center_y_px*scale),
                               region.radius_px*scale, facecolor="#d7e6e2", edgecolor="gray", alpha=0.7)
            ax.add_patch(patch)
        subset = returns[(returns.side == side) & returns.resource_visit].sort_values("outside_seconds")
        if subset.empty:
            subset = returns[returns.side == side].sort_values("outside_seconds")
        indices = np.unique(np.linspace(0, max(0, len(subset)-1), min(3, len(subset))).astype(int))
        for index in indices:
            event = subset.iloc[index]
            context = contexts[(side, int(event.track_id))]
            lo = int(event.exit_frame) // settings.bin_frames
            hi = min(len(context["inside"]), int(event.return_frame) // settings.bin_frames + 60)
            x, y = context["x_mm"][lo:hi], context["y_mm"][lo:hi]
            line, = ax.plot(x, y, alpha=0.7, linewidth=0.8, label=f"T{event.track_id}, return {event.return_frame}")
            return_bin = int(event.return_frame) // settings.bin_frames
            ax.scatter(context["x_mm"][return_bin], context["y_mm"][return_bin], marker="x", color=line.get_color(), s=60)
            example_rows.append(event.to_dict())
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.autoscale_view()
        ax.set(title=f"{side.capitalize()} colony", xlabel="Tracking x (mm)", ylabel="Tracking y (mm)")
        ax.legend(fontsize=7)
    fig.suptitle("Return examples and annotated regions; crosses mark return bins")
    return fig, pd.DataFrame(example_rows)
