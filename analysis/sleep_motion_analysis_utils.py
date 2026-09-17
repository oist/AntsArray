"""Read cached motion-based sleep labels for cluster and event analyses."""

from __future__ import annotations

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from analysis import grid_occupancy_utils as go


def normalize_clusters(clusters: pd.DataFrame) -> pd.DataFrame:
    out = clusters.rename(columns={"TrackID": "track_id"}).copy()
    if "cluster_id" not in out:
        column = "leiden_cluster" if "leiden_cluster" in out else "leiden_cluster_id"
        out["cluster_id"] = out["side"].astype(str) + "_" + out[column].astype(str)
    out = out[["side", "track_id", "cluster_id"]].copy()
    out["track_id"] = pd.to_numeric(out["track_id"], errors="raise").astype(int)
    if out.isna().any().any() or out.duplicated(["side", "track_id"]).any():
        raise ValueError("Cluster assignments must be nonmissing and unique by side and track_id")
    return out


def load_sleep_label_tracks(root: Path, clusters: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for path in sorted((Path(root) / "per_track").glob("*/sleep_motion_label_metadata.json")):
        metadata = json.loads(path.read_text())
        summary = metadata["summary"]
        state_path = path.parent / metadata["files"]["sleep_state"]
        state = np.load(state_path, mmap_mode="r")
        n_frames = int(metadata["frame_max"]) - int(metadata["frame_min"]) + 1
        if state.shape != (n_frames,) or state.dtype != np.int8:
            raise ValueError(f"Invalid sleep-state array: {state_path}")
        rows.append({
            "side": summary["side"], "track_id": int(summary["track_id"]),
            "track_name": summary["track_name"], "state_path": str(state_path),
            "metadata_path": str(path), "frame_min": int(metadata["frame_min"]),
            "frame_max": int(metadata["frame_max"]), "fps": float(metadata["fps"]),
            "classifier_parameters": json.dumps(metadata["classifier_parameters"], sort_keys=True),
        })
    if not rows:
        raise FileNotFoundError(f"No sleep_motion_labels found under {root}")
    tracks = pd.DataFrame(rows)
    # A dataset can retain a short/exported version of an otherwise complete
    # track. When clustering explicitly selected filenames, honor that choice
    # before checking ant identity: never merge the other version by ID alone.
    if clusters is not None and "track_name" in clusters:
        tracks = tracks[tracks["track_name"].isin(clusters["track_name"])].copy()
    if tracks.duplicated(["side", "track_id"]).any():
        raise ValueError("Sleep-label identities are not unique by side and track_id")
    if not tracks.empty and (tracks["fps"].nunique() != 1 or tracks["classifier_parameters"].nunique() != 1):
        raise ValueError("Sleep-label caches mix frame rates or classifier parameters")
    if clusters is not None:
        assignments = normalize_clusters(clusters)
        if "track_name" in clusters:
            named = clusters.rename(columns={"TrackID": "track_id"})
            named = named.assign(track_id=pd.to_numeric(named["track_id"], errors="raise").astype(int))
            assignments = assignments.merge(named[["side", "track_id", "track_name"]],
                                            on=["side", "track_id"], validate="one_to_one")
            tracks = assignments.merge(tracks, on=["side", "track_id", "track_name"],
                                       how="left", validate="one_to_one")
        else:
            tracks = assignments.merge(tracks, on=["side", "track_id"], how="left", validate="one_to_one")
        if tracks["state_path"].isna().any():
            missing = tracks.loc[tracks["state_path"].isna(), ["side", "track_id"]]
            raise FileNotFoundError(f"Missing sleep labels for clustered ants:\n{missing}")
    return tracks.sort_values(["side", "track_id"]).reset_index(drop=True)


def sample_dense(values: np.ndarray, frame_min: int, frames: np.ndarray, missing=np.nan) -> np.ndarray:
    """Sample global frames without treating frames before a track as negative indices."""
    frames = np.asarray(frames, dtype=np.int64)
    index = frames - int(frame_min)
    out = np.full(frames.shape, missing, dtype=np.result_type(values.dtype, type(missing)))
    valid = (index >= 0) & (index < len(values))
    out[valid] = values[index[valid]]
    return out


def interval_counts(mask: np.ndarray, frame_min: int, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    prefix = np.r_[0, np.cumsum(mask, dtype=np.int64)]
    lo = np.clip(np.asarray(starts) - frame_min, 0, len(mask)).astype(np.int64)
    hi = np.clip(np.asarray(stops) - frame_min, 0, len(mask)).astype(np.int64)
    return prefix[hi] - prefix[lo]


def cluster_sleep_timeseries(
    tracks: pd.DataFrame, *, bin_seconds: float = 600.0, min_classified_fraction: float = 0.5,
    recording_start_frame: int = 0, recording_stop_frame: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average per-ant sleep fractions equally; unknown frames never count as wake."""
    if bin_seconds <= 0 or not 0 <= min_classified_fraction <= 1:
        raise ValueError("Invalid bin duration or minimum classified fraction")
    fps = float(tracks["fps"].iloc[0])
    width = max(1, round(fps * bin_seconds))
    recording_stop = (int(tracks["frame_max"].max()) + 1 if recording_stop_frame is None
                      else int(recording_stop_frame))
    if not 0 <= recording_start_frame < recording_stop:
        raise ValueError("Invalid recording frame window")
    starts = np.arange(recording_start_frame, recording_stop, width, dtype=np.int64)
    stops = np.minimum(starts + width, recording_stop)
    rows = []
    for row in tracks.itertuples():
        state = np.load(row.state_path, mmap_mode="r")
        known = interval_counts(state >= 0, row.frame_min, starts, stops)
        sleep = interval_counts(state == 1, row.frame_min, starts, stops)
        coverage = known / (stops - starts)
        fraction = np.divide(sleep, known, out=np.full(len(starts), np.nan), where=known > 0)
        fraction[coverage < min_classified_fraction] = np.nan
        rows.append(pd.DataFrame({
            "side": row.side, "track_id": row.track_id, "cluster_id": row.cluster_id,
            "bin_start_frame": starts, "time_h": (starts + stops) / (2 * fps * 3600),
            "sleep_fraction": fraction, "classified_fraction": coverage,
            "n_sleep_frames": sleep, "n_classified_frames": known,
        }))
    ant_bins = pd.concat(rows, ignore_index=True)
    summary = ant_bins.groupby(["side", "cluster_id", "time_h"], as_index=False).agg(
        mean_sleep_fraction=("sleep_fraction", "mean"),
        n_ants_with_data=("sleep_fraction", "count"),
        n_ants_total=("track_id", "size"),
        mean_classified_fraction=("classified_fraction", "mean"),
    )
    return summary, ant_bins


def clock_axis(ax, start_clock_seconds: float, max_time_h: float,
               light_off_hour: float = 19.5, light_on_hour: float = 5.5,
               min_time_h: float = 0.0) -> None:
    go.add_light_shading(ax, start_clock_seconds, max_time_h,
                         light_off_hour=light_off_hour, light_on_hour=light_on_hour,
                         min_time_h=min_time_h)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda hours, _: go.format_clock_time(start_clock_seconds + hours * 3600)))
    ax.set_xlabel("Recording clock time (HH:MM)")


def plot_cluster_sleep_timeseries(
    summary: pd.DataFrame, *, start_clock_seconds: float,
    light_off_hour: float = 19.5, light_on_hour: float = 5.5,
):
    sides = list(summary["side"].unique())
    fig, axes = plt.subplots(len(sides), 2, figsize=(13, 3.5 * len(sides)), squeeze=False,
                             sharex=True, layout="constrained")
    for row, side in enumerate(sides):
        for index, (cluster, group) in enumerate(summary[summary.side == side].groupby("cluster_id")):
            color = plt.get_cmap("tab10")(index)
            label = f"{cluster} (n={int(group.n_ants_total.max())})"
            axes[row, 0].plot(group.time_h, group.mean_sleep_fraction, color=color, label=label)
            axes[row, 1].plot(group.time_h, group.n_ants_with_data, color=color, label=label)
        for ax in axes[row]:
            clock_axis(ax, start_clock_seconds, float(summary.time_h.max()), light_off_hour, light_on_hour,
                       min_time_h=float(summary.time_h.min()))
            ax.set_title(f"{side.capitalize()} colony")
            ax.grid(alpha=0.2)
        axes[row, 0].set(ylabel="Mean sleep fraction across ants", ylim=(0, 1))
        axes[row, 1].set(ylabel="Ants with sufficient classified data", ylim=(0, None))
        axes[row, 0].legend(fontsize=9)
    fig.suptitle("Sleep by occupancy cluster")
    return fig


def plot_ant_sleep_heatmap(ant_bins: pd.DataFrame, *, start_clock_seconds: float):
    sides = list(ant_bins.side.unique())
    fig, axes = plt.subplots(len(sides), 1, figsize=(13, 4 * len(sides)), squeeze=False,
                             layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d7d7d7")
    for ax, side in zip(axes.ravel(), sides):
        data = ant_bins[ant_bins.side == side]
        wide = data.pivot(index=["cluster_id", "track_id"], columns="time_h", values="sleep_fraction").sort_index()
        times = wide.columns.to_numpy(float)
        step = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 6
        im = ax.imshow(wide.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap,
                       vmin=0, vmax=1, extent=(times[0] - step / 2, times[-1] + step / 2, len(wide), 0))
        clusters = wide.index.get_level_values(0).to_numpy()
        boundaries = np.r_[0, np.flatnonzero(clusters[1:] != clusters[:-1]) + 1, len(clusters)]
        for boundary in boundaries[1:-1]:
            ax.axhline(boundary, color="white", linewidth=1)
        ax.set_yticks((boundaries[:-1] + boundaries[1:]) / 2,
                      [clusters[start] for start in boundaries[:-1]])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda hours, _: go.format_clock_time(start_clock_seconds + hours * 3600)))
        ax.set(title=f"{side.capitalize()} colony", xlabel="Recording clock time (HH:MM)", ylabel="Ants by cluster")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Sleep fraction; gray = insufficient data", shrink=0.8)
    return fig


def load_sleep_posture_points(
    tracks: pd.DataFrame, block_dir: Path, cache_root: Path, *,
    max_frames_per_ant_state: int = 1000, random_state: int = 0, force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample cached sleep/wake labels and align finished poses, cached per ant.

    Sampling is uniform within each ant/state, excludes unknown labels, and uses
    global frames including each label array's frame_min offset. Density weights
    give every contributing ant equal mass within each colony/state.
    """
    from analysis import sleep_analysis_utils as sa
    from analysis.interaction_analysis_utils import file_fingerprint, load_or_build_pickle

    if max_frames_per_ant_state < 1 or int(max_frames_per_ant_state) != max_frames_per_ant_state:
        raise ValueError("max_frames_per_ant_state must be a positive integer")
    if tracks.empty:
        raise ValueError("No labelled ants selected for posture analysis")
    stitched_root = go.resolve_stitched_root(block_dir)
    sources = []
    for row in tracks.itertuples():
        pose_path = stitched_root / "per_track" / row.track_name
        scale_path = stitched_root / "sleep_motion" / "per_track" / Path(row.track_name).stem / "sleep_motion_metadata.json"
        sources.append((row, pose_path, scale_path))
    missing = [str(path) for row, pose_path, scale_path in sources
               for path in (pose_path, scale_path, Path(row.state_path), Path(row.metadata_path)) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing sleep-posture inputs:\n" + "\n".join(missing))

    points, counts = [], []
    for index, (row, pose_path, scale_path) in enumerate(sources, start=1):
        scale = float(json.loads(scale_path.read_text())["mm_per_px"])
        key = {"version": 1, "max_frames_per_ant_state": int(max_frames_per_ant_state),
               "random_state": int(random_state), "side": row.side, "track_id": int(row.track_id),
               "frame_min": int(row.frame_min), "mm_per_px": scale,
               "sources": [file_fingerprint(path) for path in
                           (pose_path, scale_path, Path(row.state_path), Path(row.metadata_path))]}
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]

        def build():
            state = np.load(row.state_path, mmap_mode="r")
            samples, audit = [], []
            for value, name in ((1, "sleep"), (0, "wake")):
                frames = np.flatnonzero(state == value) + int(row.frame_min)
                n_available = len(frames)
                seed = np.random.SeedSequence([int(random_state), 0 if row.side == "left" else 1, int(row.track_id), value])
                rng = np.random.default_rng(seed)
                if len(frames) > max_frames_per_ant_state:
                    frames = np.sort(rng.choice(frames, int(max_frames_per_ant_state), replace=False))
                samples.append(pd.DataFrame(dict(Frame=frames, posture_state=name, track_path=str(pose_path),
                                                  side=row.side, track_id=row.track_id, track_name=row.track_name)))
                audit.append(dict(side=row.side, track_id=row.track_id, posture_state=name,
                                  n_labelled_frames=n_available, n_sampled_frames=len(frames), n_aligned_frames=0))
            sample_table = pd.concat(samples, ignore_index=True)
            aligned = pd.DataFrame(columns=["Frame", "posture_state", "track_path", "side", "track_id",
                                            "track_name", "bodypoint", "aligned_x_mm", "aligned_y_mm", "axis_length_mm"])
            if not sample_table.empty:
                try:
                    aligned, _ = sa.aligned_posture_points_from_frame_states(sample_table, mm_per_px=scale)
                except ValueError as error:
                    if str(error) != "No aligned posture points could be computed from the sampled frames":
                        raise
            for record in audit:
                record["n_aligned_frames"] = aligned.loc[aligned.posture_state == record["posture_state"], "Frame"].nunique()
            return aligned, pd.DataFrame(audit)

        aligned, audit = load_or_build_pickle(cache_root / f"{pose_path.stem}_{digest}.pkl", build, force=force)
        if not aligned.empty:
            points.append(aligned)
        counts.append(audit)
        print(f"Sleep posture: {index}/{len(sources)} ants", flush=True)
    audit = pd.concat(counts, ignore_index=True)
    missing_states = audit[audit.n_aligned_frames == 0]
    if not missing_states.empty:
        print("NO VALID POSTURES for these ant/states (no labels or missing/degenerate body axis):", flush=True)
        print(missing_states.to_string(index=False), flush=True)
    if not points:
        raise ValueError("No valid sleep/wake postures; inspect label coverage and bodypoints 0 and 1")
    points = pd.concat(points, ignore_index=True)
    points["density_weight"] = 1.0 / points.groupby(["side", "track_id", "posture_state"], observed=True).Frame.transform("size")
    return points, audit


def plot_sleep_posture_distributions(points: pd.DataFrame, *, bins: int = 160, extent_mm: float | None = None):
    """One sleep/wake comparison per colony, in the legacy aligned-density style."""
    from analysis import sleep_analysis_utils as sa

    # A common physical extent across colonies; probabilities retain out-of-view mass.
    if extent_mm is None:
        extent_mm = max(.25, float(np.quantile(np.abs(points[["aligned_x_mm", "aligned_y_mm"]].to_numpy()), .995))*1.08)
        extent_mm = float(np.ceil(extent_mm * 2) / 2)
    figures, summaries, medians = {}, [], []
    for side, group in points.groupby("side", sort=True):
        summary = group.groupby("posture_state", observed=True).agg(
            n_aligned_points=("Frame", "size"), n_tracks=("track_id", "nunique"),
        ).reset_index()
        frames = group.drop_duplicates(["track_id", "Frame"]).groupby("posture_state", observed=True).size().rename("n_frames")
        summary = summary.merge(frames, on="posture_state", validate="one_to_one")
        fig, plotted, median, summary = sa.plot_aligned_posture_points(
            group, summary, state_order=("sleep", "wake"), weight_col="density_weight", bins=bins, extent_mm=extent_mm,
            title=f"{side.capitalize()} colony: sleep/wake posture density\nBodypoint 0 at origin; 0 -> 1 upward; equal ant weight",
        )
        for state, selected in plotted.groupby("posture_state", observed=True):
            visible = (selected.plot_x_mm.abs() <= extent_mm) & (selected.plot_y_mm.abs() <= extent_mm)
            summary.loc[summary.posture_state == state, "probability_in_view"] = selected.loc[visible, "density_weight"].sum() / selected.density_weight.sum()
        figures[side] = fig
        summaries.append(summary.assign(side=side))
        medians.append(median.assign(side=side))
    images = [ax.collections[0] for fig in figures.values() for ax in fig.axes[:2]]
    vmin = min(im.norm.vmin for im in images)
    vmax = max(im.norm.vmax for im in images)
    for im in images:
        im.set_clim(vmin, vmax)
    return figures, pd.concat(summaries, ignore_index=True), pd.concat(medians, ignore_index=True)


def _clock_profile_ant(task):
    """Read existing vectors; unknown frames are never interpreted as inactivity."""
    bins = task["bins"].copy()
    audit = dict(task["identity"])
    for source, value_column in (("speed", "mean_speed_mm_s"), ("sleep", "sleep_percent")):
        metadata = task[source]
        path = None if metadata is None else Path(metadata[f"{source}_path"])
        counts = np.zeros(len(bins), dtype=np.int64)
        sums = np.zeros(len(bins), dtype=float)
        audit[f"{source}_status"] = "missing metadata" if path is None else "missing vector"
        if path is not None and path.is_file():
            values = np.load(path, mmap_mode="r")
            span = int(metadata["frame_max"]) - int(metadata["frame_min"]) + 1
            if values.shape != (span,):
                raise ValueError(f"Invalid {source} vector span: {path}")
            if source == "sleep" and values.dtype != np.int8:
                raise ValueError(f"Invalid sleep-state dtype: {path}")
            for i, row in enumerate(bins.itertuples(index=False)):
                lo = max(0, min(span, max(row.bin_start_frame, task["recording_start_frame"])
                                - int(metadata["frame_min"])))
                hi = max(0, min(span, min(row.bin_stop_frame, task["recording_stop_frame"])
                                - int(metadata["frame_min"])))
                lo = min(lo, hi)
                part = values[lo:hi]
                if source == "speed":
                    valid = np.isfinite(part) & (part >= 0)
                    counts[i] = valid.sum()
                    sums[i] = part[valid].sum(dtype=np.float64)
                else:
                    if not np.isin(part, [-1, 0, 1]).all():
                        raise ValueError(f"Unrecognized sleep-state values: {path}")
                    counts[i] = (part >= 0).sum()
                    sums[i] = (part == 1).sum()
            audit[f"{source}_status"] = "ok"
        coverage = counts / bins["n_expected_frames"].to_numpy(float)
        value = np.divide(sums, counts, out=np.full(len(bins), np.nan), where=counts > 0)
        value[coverage < task["min_bin_coverage"]] = np.nan
        if source == "sleep":
            value *= 100
            bins["n_sleep_frames"] = sums.astype(np.int64)
        bins[f"n_{source}_frames"] = counts
        bins[f"{source}_coverage"] = coverage
        bins[value_column] = value
        audit[f"{source}_usable_bins"] = int(np.isfinite(value).sum())
    return bins.assign(**task["identity"]), audit


def compute_activity_sleep_clock_profiles(
    clusters: pd.DataFrame,
    speed_tracks: pd.DataFrame,
    sleep_tracks: pd.DataFrame | None,
    *,
    fps: float,
    recording_stop_frame: int,
    recording_start_frame: int = 0,
    start_clock_seconds: float,
    light_on_hour: float,
    light_off_hour: float,
    bin_minutes: float = 30.0,
    min_bin_coverage: float = 0.50,
    min_cycles: int = 2,
    include_partial_cycles: bool = False,
    recording_date: str | None = None,
    max_workers: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Activity and sleep by ant, full light–dark cycle, and time since lights-on.

    Uses the existing gap-aware speed and motion-based sleep vectors, never raw
    position interpolation or a new sleep classifier. Coverage is measured
    separately for valid speed and classified sleep frames within each bin.
    Complete experiment-wide cycles by default; optionally show partial cycles
    with unrecorded hours missing, never zero. The folded profile averages usable cycles
    equally, requiring min_cycles for EACH ant/time/measure. No row scaling,
    smoothing, peak-based sorting, or selection for apparent rhythmicity.
    """
    if fps <= 0 or not 0 <= recording_start_frame < recording_stop_frame or bin_minutes <= 0:
        raise ValueError("FPS, recording span, and bin duration must be positive")
    if not np.isclose(1440 / bin_minutes, round(1440 / bin_minutes)):
        raise ValueError("bin_minutes must divide 24 hours")
    if not 0 <= min_bin_coverage <= 1 or min_cycles < 1:
        raise ValueError("Require coverage in [0, 1] and at least one cycle")
    if not (0 <= start_clock_seconds < 86400 and 0 <= light_on_hour < 24
            and 0 <= light_off_hour < 24 and light_on_hour != light_off_hour):
        raise ValueError("Invalid start clock or light schedule")
    base = clusters.rename(columns={"TrackID": "track_id"}).copy()
    identities = normalize_clusters(clusters)
    if "track_name" not in base:
        # Older saved cluster handoffs contain only colony/ID, while the
        # interactive table includes names. Resolve names without guessing.
        names = sleep_tracks if sleep_tracks is not None else speed_tracks
        base = base.merge(names[["side", "track_id", "track_name"]],
                          on=["side", "track_id"], how="left", validate="one_to_one")
        if base.track_name.isna().any():
            raise ValueError("Cannot resolve track names for all cluster identities")
    base = base[["side", "track_id", "track_name"]].merge(
        identities, on=["side", "track_id"], validate="one_to_one"
    ).sort_values(["side", "cluster_id", "track_id"])
    if base.empty or base.track_name.duplicated().any():
        raise ValueError("Supply unique, nonempty ant identities")
    speeds = speed_tracks[speed_tracks.track_name.isin(base.track_name)].copy()
    sleeps = (pd.DataFrame(columns=["side", "track_id", "track_name", "fps", "state_path"])
              if sleep_tracks is None else sleep_tracks.copy())
    sleeps = sleeps.merge(base[["side", "track_id"]], on=["side", "track_id"], validate="one_to_one")
    if speeds.track_name.duplicated().any():
        raise ValueError("Speed metadata contains duplicate tracks")
    for table in (speeds, sleeps):
        if len(table) and not np.allclose(table["fps"], fps):
            raise ValueError("Activity/sleep caches have inconsistent frame rates")
    speeds = speeds.set_index("track_name")
    sleeps = sleeps.rename(columns={"state_path": "sleep_path"}).set_index(["side", "track_id"])

    offset = light_on_hour * 3600 - start_clock_seconds
    first = int(np.ceil((recording_start_frame / fps - offset) / 86400))
    stop = int(np.floor((recording_stop_frame / fps - offset) / 86400))
    if include_partial_cycles:
        first = int(np.floor((recording_start_frame / fps - offset) / 86400))
        stop = int(np.ceil((recording_stop_frame / fps - offset) / 86400))
    if stop <= first:
        raise ValueError("No complete lights-on-to-lights-on cycle in this recording")
    n_bins = int(round(1440 / bin_minutes))
    date = pd.to_datetime(recording_date, format="%Y%m%d", errors="coerce") if recording_date else pd.NaT
    parts = []
    for cycle in range(first, stop):
        edges = np.rint((offset + cycle * 86400 + np.arange(n_bins + 1) * bin_minutes * 60) * fps).astype(np.int64)
        label = ((date + pd.Timedelta(days=cycle)).strftime("%b %d") if pd.notna(date)
                 else f"Day {cycle + 1}")
        parts.append(pd.DataFrame(dict(
            cycle_index=cycle, cycle_label=label, zt_bin=np.arange(n_bins),
            complete_cycle=bool(edges[0] >= recording_start_frame and edges[-1] <= recording_stop_frame),
            zt_hour=(np.arange(n_bins) + 0.5) * bin_minutes / 60,
            bin_start_frame=edges[:-1], bin_stop_frame=edges[1:], n_expected_frames=np.diff(edges),
        )))
    template = pd.concat(parts, ignore_index=True)
    if (template.n_expected_frames <= 0).any():
        raise ValueError("Time bins must contain at least one frame")
    template["clock_hour"] = (light_on_hour + template.zt_hour) % 24
    template["light_on_hour"], template["light_off_hour"] = light_on_hour, light_off_hour
    template["bin_minutes"] = bin_minutes
    template["min_bin_coverage"] = min_bin_coverage
    template["recording_hours"] = (recording_stop_frame - recording_start_frame) / fps / 3600
    tasks = []
    for ant in base.itertuples(index=False):
        speed = speeds.loc[ant.track_name].to_dict() if ant.track_name in speeds.index else None
        sleep = sleeps.loc[(ant.side, ant.track_id)].to_dict() if (ant.side, ant.track_id) in sleeps.index else None
        if speed is not None and (speed["side"] != ant.side or int(speed["track_id"]) != ant.track_id):
            raise ValueError(f"Speed identity mismatch: {ant.track_name}")
        if sleep is not None and sleep["track_name"] != ant.track_name:
            raise ValueError(f"Sleep identity mismatch: {ant.track_name}")
        tasks.append(dict(bins=template, speed=speed, sleep=sleep, min_bin_coverage=min_bin_coverage,
                          recording_stop_frame=recording_stop_frame, recording_start_frame=recording_start_frame,
                          identity=dict(side=ant.side, track_id=int(ant.track_id),
                                        track_name=ant.track_name, cluster_id=ant.cluster_id)))
    tables, audit = [], []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = [pool.submit(_clock_profile_ant, task) for task in tasks]
        for number, future in enumerate(as_completed(futures), 1):
            table, status = future.result()
            tables.append(table)
            audit.append(status)
            if number == 1 or number % 16 == 0 or number == len(tasks):
                print(f"Activity/sleep clock profiles: {number}/{len(tasks)} ants", flush=True)
    cycle_bins = pd.concat(tables, ignore_index=True).sort_values(
        ["side", "cluster_id", "track_id", "cycle_index", "zt_bin"]
    ).reset_index(drop=True)
    profiles = cycle_bins.groupby(
        ["side", "track_id", "track_name", "cluster_id", "zt_bin", "zt_hour", "clock_hour"], as_index=False
    ).agg(mean_speed_mm_s=("mean_speed_mm_s", "mean"), sleep_percent=("sleep_percent", "mean"),
          n_speed_cycles=("mean_speed_mm_s", "count"), n_sleep_cycles=("sleep_percent", "count"))
    profiles.loc[profiles.n_speed_cycles < min_cycles, "mean_speed_mm_s"] = np.nan
    profiles.loc[profiles.n_sleep_cycles < min_cycles, "sleep_percent"] = np.nan
    profiles["min_cycles"] = min_cycles
    profiles["light_on_hour"], profiles["light_off_hour"] = light_on_hour, light_off_hour
    return cycle_bins, profiles, pd.DataFrame(audit).sort_values(["side", "track_id"]).reset_index(drop=True)


def plot_activity_sleep_clock_matrices(cycle_bins: pd.DataFrame, profiles: pd.DataFrame, *, speed_vmax=None):
    """One figure per colony: activity/sleep rows, cycles and folded-mean columns.

    Ant order is fixed by spatial cluster then ID, never by peak time. Color
    limits are shared across colonies and cycles; no per-ant normalization.
    Returns figures for the interactive caller's usual plt.show/auto-save.
    """
    on = float(cycle_bins.light_on_hour.iloc[0])
    off = float(cycle_bins.light_off_hour.iloc[0])
    dark_start = (off - on) % 24
    cycles = cycle_bins[["cycle_index", "cycle_label", "complete_cycle"]].drop_duplicates().sort_values("cycle_index")
    n_bins = int(cycle_bins.zt_bin.max()) + 1
    valid_speed = cycle_bins.mean_speed_mm_s.dropna()
    if speed_vmax is None:
        speed_vmax = max(0.01, float(valid_speed.quantile(0.99))) if len(valid_speed) else 1.0
    if not np.isfinite(speed_vmax) or speed_vmax <= 0:
        raise ValueError("speed_vmax must be positive")
    figures = []
    for side in sorted(cycle_bins.side.unique()):
        ants = cycle_bins[cycle_bins.side == side].drop_duplicates("track_name").sort_values(
            ["cluster_id", "track_id"]
        ).set_index("track_name")
        ncols = len(cycles) + 1
        fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols + 1, 11.5), squeeze=False,
                                 sharex=True, sharey=True, layout="constrained")
        clusters = ants.cluster_id.to_numpy()
        boundaries = np.flatnonzero(clusters[1:] != clusters[:-1]) + 1
        columns = [(cycle_bins[(cycle_bins.side == side) & (cycle_bins.cycle_index == row.cycle_index)],
                    f"Cycle starting {row.cycle_label}" + (" (partial)" if not row.complete_cycle else ""))
                   for row in cycles.itertuples()]
        minimum = int(profiles.min_cycles.iloc[0])
        profile_title = (f"Across-cycle mean (≥{minimum} cycles/bin)" if minimum >= 2
                         else "Observed clock profile (≥1 cycle/bin)")
        columns.append((profiles[profiles.side == side], profile_title))
        for row_index, (metric, label, cmap_name, vmax) in enumerate([
            ("mean_speed_mm_s", "Activity: mean speed (mm/s)", "magma", speed_vmax),
            ("sleep_percent", "Sleep (% of classified time)", "viridis", 100),
        ]):
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("#d7d7d7")
            for column, (table, title) in enumerate(columns):
                ax = axes[row_index, column]
                matrix = table.pivot(index="track_name", columns="zt_bin", values=metric).reindex(
                    index=ants.index, columns=range(n_bins)
                )
                im = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(float)), aspect="auto", interpolation="nearest",
                               extent=(0, 24, len(ants), 0), cmap=cmap, vmin=0, vmax=vmax)
                ax.axvline(dark_start, color="deepskyblue", ls="--", lw=0.9)
                ax.axvspan(0, dark_start, ymin=1.005, ymax=1.022, color="#f3cf56", clip_on=False)
                ax.axvspan(dark_start, 24, ymin=1.005, ymax=1.022, color="#45536d", clip_on=False)
                for boundary in boundaries:
                    ax.axhline(boundary, color="white", lw=0.6, alpha=0.8)
                ax.set_title(title, fontsize=10, pad=13)
                if column == 0:
                    ax.set_yticks(np.arange(len(ants)) + 0.5,
                                  [f"{ant.track_id} ({ant.cluster_id})" for ant in ants.itertuples()], fontsize=5.5)
                    ax.set_ylabel(f"{label}\nAnt ID (occupancy cluster)")
                ticks = np.arange(0, 25, 6)
                ax.set_xticks(ticks, [f"{hour}\n{go.format_clock_time((on + hour) * 3600)}" for hour in ticks])
                if row_index == 1:
                    ax.set_xlabel("Hours since lights-on\nClock time below")
            extend = "max" if metric == "mean_speed_mm_s" and len(valid_speed) and valid_speed.max() > vmax else "neither"
            fig.colorbar(im, ax=axes[row_index].tolist(), label=label, shrink=0.85, extend=extend)
        duration = float(cycle_bins.recording_hours.iloc[0])
        caution = (f"Only {duration:.1f} h recorded: insufficient to assess a 24-h rhythm.\n"
                   if duration < 24 else "")
        fig.suptitle(
            f"{side.capitalize()} colony — individual activity and sleep by time of day\n"
            + caution +
            f"Lights on {go.format_clock_time(on * 3600)}, off {go.format_clock_time(off * 3600)}; "
            f"{float(cycle_bins.bin_minutes.iloc[0]):g}-min bins; gray = unrecorded / insufficient data.\n"
            "Same ant order throughout; no row normalization. Shared scales; colorbar triangle marks clipping."
        )
        figures.append(fig)
    return figures
