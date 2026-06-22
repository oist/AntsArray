# %%
# Local time-of-day occupancy analysis for occupancy clusters.
try:
    get_ipython().run_line_magic("matplotlib", "qt")  # type: ignore[name-defined]
except Exception:
    pass

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = print

repo_root = Path.cwd().resolve()
for candidate in [repo_root, *repo_root.parents]:
    if (candidate / "analysis" / "grid_occupancy_utils.py").exists():
        repo_root = candidate
        break
else:
    raise FileNotFoundError("Could not find analysis/grid_occupancy_utils.py from the current working directory")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import analysis.grid_occupancy_utils as go


# %%
# Edit these settings first.
GRID_ROOT = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/grid_occupancy_histograms"
)
PER_TRACK_ROOT = GRID_ROOT.parent / "per_track"
CLUSTER_TABLE_PATH = GRID_ROOT / "track_cluster_ids.csv"
OUT_ROOT = GRID_ROOT / "time_of_day_cluster_occupancy"

FPS = 24.0
TOD_BIN_MINUTES = 60
LIGHT_ON_HOUR = 5.5
BODYPOINT = 0
FRAME_COL = "Frame"
X_COL = "TrackX"
Y_COL = "TrackY"
RECOMPUTE = False


# %%
def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def recording_start_seconds_from_metadata(tracks: pd.DataFrame) -> int:
    if tracks.empty:
        raise ValueError("Cannot infer recording start time from an empty track table")
    meta = load_json(Path(tracks["metadata_path"].iloc[0]))
    return go.parse_track_start_seconds(str(meta["track_name"]))


def load_cluster_tracks(grid_root: Path, cluster_table_path: Path, per_track_root: Path) -> pd.DataFrame:
    grid_tracks = go.load_grid_tracks(grid_root)
    cluster_ids = pd.read_csv(cluster_table_path).rename(
        columns={"TrackID": "track_id", "leiden_cluster_id": "leiden_cluster"}
    )
    tracks = grid_tracks.merge(
        cluster_ids[["track_id", "side", "cluster_id", "leiden_cluster"]],
        on=["track_id", "side"],
        how="inner",
        validate="one_to_one",
    )

    track_paths = []
    for row in tracks.itertuples(index=False):
        bucket_track = per_track_root / row.track_name
        if bucket_track.exists():
            track_paths.append(bucket_track)
            continue

        meta = load_json(Path(row.metadata_path))
        metadata_track = Path(meta["track_path"])
        if metadata_track.exists():
            track_paths.append(metadata_track)
            continue

        raise FileNotFoundError(f"Could not find per-track parquet for {row.track_name}")

    tracks = tracks.copy()
    tracks["track_path"] = track_paths
    return tracks.sort_values(["side", "cluster_id", "track_id", "track_name"], kind="mergesort").reset_index(drop=True)


def side_grid_specs(tracks: pd.DataFrame) -> dict[str, dict]:
    specs = {}
    for side, side_tracks in tracks.groupby("side", sort=True):
        row = side_tracks.iloc[0]
        meta = load_json(Path(row["metadata_path"]))
        specs[side] = {
            "x_edges_mm": np.load(row["x_edges_path"]).astype(np.float64),
            "y_edges_mm": np.load(row["y_edges_path"]).astype(np.float64),
            "mm_per_px": float(meta["mm_per_px"]),
            "x_origin_px": float(meta["input_x_origin_px"]),
            "y_origin_px": float(meta["y_origin_px"]),
            "input_x_is_side_local": bool(meta.get("input_x_is_side_local", False)),
        }
    return specs


def read_track_position(
    path: Path,
    *,
    frame_col: str = "Frame",
    x_col: str = "TrackX",
    y_col: str = "TrackY",
    bodypoint: int = 0,
) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    columns = pq.ParquetFile(path).schema.names
    missing = {frame_col, x_col, y_col}.difference(columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    read_cols = [frame_col, x_col, y_col]
    if "Bodypoint" in columns:
        table = ds.dataset(path, format="parquet").to_table(
            columns=read_cols,
            filter=pc.field("Bodypoint") == int(bodypoint),
            use_threads=True,
        )
    else:
        table = ds.dataset(path, format="parquet").to_table(columns=read_cols, use_threads=True)

    df = table.to_pandas()
    df = df.rename(columns={frame_col: "Frame", x_col: "X", y_col: "Y"})
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["Frame", "X", "Y"])
    if df.empty:
        return pd.DataFrame({"Frame": pd.Series(dtype=np.int64), "X": pd.Series(dtype=float), "Y": pd.Series(dtype=float)})

    df["Frame"] = df["Frame"].round().astype(np.int64)
    df["X"] = df["X"].astype(np.float64)
    df["Y"] = df["Y"].astype(np.float64)
    if df["Frame"].duplicated().any():
        df = df.groupby("Frame", sort=True, as_index=False).agg(X=("X", "mean"), Y=("Y", "mean"))
    return df.sort_values("Frame", kind="mergesort").reset_index(drop=True)


def bin_positions(
    position: pd.DataFrame,
    *,
    side_spec: dict,
    recording_start_clock_seconds: int,
    light_on_clock_seconds: int,
    fps: float,
    tod_bin_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_edges = side_spec["x_edges_mm"]
    y_edges = side_spec["y_edges_mm"]
    mm_per_px = float(side_spec["mm_per_px"])
    x_origin_px = 0.0 if side_spec["input_x_is_side_local"] else float(side_spec["x_origin_px"])
    y_origin_px = float(side_spec["y_origin_px"])

    frames = position["Frame"].to_numpy(np.int64)
    x_mm = (position["X"].to_numpy(np.float64) - x_origin_px) * mm_per_px
    y_mm = (position["Y"].to_numpy(np.float64) - y_origin_px) * mm_per_px

    n_tod_bins = int(round((24 * 3600) / float(tod_bin_seconds)))
    bin_frames = int(round(float(fps) * float(tod_bin_seconds)))
    relative_start_seconds = (int(recording_start_clock_seconds) - int(light_on_clock_seconds)) % (24 * 3600)
    start_frame_offset = int(round(float(relative_start_seconds) * float(fps)))
    all_tod_bin = ((frames + start_frame_offset) // bin_frames) % n_tod_bins

    x_bin = np.searchsorted(x_edges, x_mm, side="right") - 1
    y_bin = np.searchsorted(y_edges, y_mm, side="right") - 1
    x_bin[(x_bin == len(x_edges) - 1) & (x_mm == x_edges[-1])] = len(x_edges) - 2
    y_bin[(y_bin == len(y_edges) - 1) & (y_mm == y_edges[-1])] = len(y_edges) - 2

    in_grid = (
        (x_bin >= 0)
        & (x_bin < len(x_edges) - 1)
        & (y_bin >= 0)
        & (y_bin < len(y_edges) - 1)
        & np.isfinite(x_mm)
        & np.isfinite(y_mm)
    )
    return (
        all_tod_bin.astype(np.int64, copy=False),
        all_tod_bin[in_grid].astype(np.int64, copy=False),
        y_bin[in_grid].astype(np.int64, copy=False),
        x_bin[in_grid].astype(np.int64, copy=False),
    )


def compute_time_of_day_occupancy(
    tracks: pd.DataFrame,
    specs: dict[str, dict],
    *,
    fps: float,
    tod_bin_minutes: float,
    recording_start_clock_seconds: int,
    light_on_clock_seconds: int,
    bodypoint: int,
    frame_col: str,
    x_col: str,
    y_col: str,
) -> dict[str, dict]:
    tod_bin_seconds = float(tod_bin_minutes) * 60.0
    n_tod_bins = int(round((24 * 3600) / tod_bin_seconds))
    if not np.isclose(n_tod_bins * tod_bin_seconds, 24 * 3600):
        raise ValueError("TOD_BIN_MINUTES must divide 24 hours evenly")

    results = {}
    for side, side_tracks in tracks.groupby("side", sort=True):
        cluster_ids = side_tracks["cluster_id"].drop_duplicates().to_list()
        cluster_to_idx = {cluster_id: i for i, cluster_id in enumerate(cluster_ids)}
        spec = specs[side]
        n_y = len(spec["y_edges_mm"]) - 1
        n_x = len(spec["x_edges_mm"]) - 1
        counts = np.zeros((len(cluster_ids), n_tod_bins, n_y, n_x), dtype=np.uint32)
        observed = np.zeros((len(cluster_ids), n_tod_bins), dtype=np.uint64)
        out_of_grid = np.zeros((len(cluster_ids),), dtype=np.uint64)

        for i, row in enumerate(side_tracks.itertuples(index=False), start=1):
            if i == 1 or i == len(side_tracks) or i % 10 == 0:
                print(f"{side}: {i}/{len(side_tracks)} {row.track_name} cluster={row.cluster_id}")
            position = read_track_position(
                Path(row.track_path),
                frame_col=frame_col,
                x_col=x_col,
                y_col=y_col,
                bodypoint=bodypoint,
            )
            if position.empty:
                continue

            all_tod_bin, tod_bin, y_bin, x_bin = bin_positions(
                position,
                side_spec=spec,
                recording_start_clock_seconds=recording_start_clock_seconds,
                light_on_clock_seconds=light_on_clock_seconds,
                fps=fps,
                tod_bin_seconds=tod_bin_seconds,
            )
            cluster_idx = cluster_to_idx[row.cluster_id]
            np.add.at(counts[cluster_idx], (tod_bin, y_bin, x_bin), 1)
            observed[cluster_idx] += np.bincount(all_tod_bin, minlength=n_tod_bins).astype(np.uint64)
            out_of_grid[cluster_idx] += np.uint64(len(all_tod_bin) - len(tod_bin))

        denom = observed[:, :, None, None].astype(np.float32)
        occupancy = np.full(counts.shape, np.nan, dtype=np.float32)
        np.divide(counts, denom, out=occupancy, where=denom > 0)
        results[side] = {
            "counts": counts,
            "observed": observed,
            "out_of_grid": out_of_grid,
            "occupancy": occupancy,
            "cluster_ids": np.array(cluster_ids, dtype=str),
            "x_edges_mm": spec["x_edges_mm"].astype(np.float32),
            "y_edges_mm": spec["y_edges_mm"].astype(np.float32),
            "tod_edges_hours": np.arange(n_tod_bins + 1, dtype=np.float32) * float(tod_bin_minutes) / 60.0,
        }
    return results


def save_results(results: dict[str, dict], tracks: pd.DataFrame, out_root: Path, settings: dict) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(out_root / "tracks_used.csv", index=False)
    for side, result in results.items():
        np.savez(
            out_root / f"{side}_time_of_day_cluster_occupancy.npz",
            counts=result["counts"],
            observed=result["observed"],
            out_of_grid=result["out_of_grid"],
            occupancy=result["occupancy"],
            cluster_ids=result["cluster_ids"],
            x_edges_mm=result["x_edges_mm"],
            y_edges_mm=result["y_edges_mm"],
            tod_edges_hours=result["tod_edges_hours"],
        )
    (out_root / "metadata.json").write_text(json.dumps(settings, indent=2) + "\n")


def load_results(out_root: Path) -> dict[str, dict]:
    results = {}
    for path in sorted(out_root.glob("*_time_of_day_cluster_occupancy.npz")):
        side = path.name.split("_time_of_day_cluster_occupancy.npz")[0]
        with np.load(path, allow_pickle=False) as data:
            results[side] = {key: data[key] for key in data.files}
    if not results:
        raise FileNotFoundError(f"No saved time-of-day occupancy files in {out_root}")
    return results


def saved_settings_match(out_root: Path, settings: dict) -> bool:
    metadata_path = out_root / "metadata.json"
    if not metadata_path.exists():
        return False
    saved = load_json(metadata_path)
    keys = [
        "grid_root",
        "cluster_table_path",
        "fps",
        "tod_bin_minutes",
        "recording_start_clock_seconds",
        "light_on_clock_seconds",
        "time_axis",
        "bodypoint",
        "frame_col",
        "x_col",
        "y_col",
    ]
    return all(saved.get(key) == settings.get(key) for key in keys)


def plot_cluster_time_maps(
    results: dict[str, dict],
    *,
    side: str,
    cluster_id: str,
    time_bin_indices: list[int] | None = None,
    mode: str = "difference",
    vmax_percentile: float = 99.0,
    cmap: str = "viridis",
) -> None:
    result = results[side]
    cluster_ids = result["cluster_ids"].astype(str)
    matches = np.flatnonzero(cluster_ids == str(cluster_id))
    if len(matches) != 1:
        raise ValueError(f"Expected one {cluster_id=} in {side}, found {len(matches)}")
    cluster_idx = int(matches[0])
    occupancy = result["occupancy"][cluster_idx]
    tod_edges = result["tod_edges_hours"]

    if time_bin_indices is None:
        n_bins = occupancy.shape[0]
        time_bin_indices = np.linspace(0, n_bins - 1, min(8, n_bins), dtype=int).tolist()

    daily_mean = np.nanmean(occupancy, axis=0)
    maps = []
    for idx in time_bin_indices:
        values = occupancy[int(idx)]
        if mode == "difference":
            values = values - daily_mean
        elif mode == "ratio":
            values = values / np.maximum(daily_mean, 1e-12)
        elif mode != "occupancy":
            raise ValueError("mode must be 'occupancy', 'difference', or 'ratio'")
        maps.append(values)

    finite = np.concatenate([values[np.isfinite(values)].ravel() for values in maps if np.isfinite(values).any()])
    if finite.size == 0:
        raise ValueError("No finite occupancy values to plot")
    if mode == "difference":
        vmax = float(np.nanpercentile(np.abs(finite), vmax_percentile))
        vmin = -vmax
        plot_cmap = "coolwarm"
        color_label = "Occupancy - daily mean"
    elif mode == "ratio":
        vmax = float(np.nanpercentile(finite, vmax_percentile))
        vmin = 0.0
        plot_cmap = cmap
        color_label = "Occupancy / daily mean"
    else:
        vmax = float(np.nanpercentile(finite, vmax_percentile))
        vmin = 0.0
        plot_cmap = cmap
        color_label = "Fraction of observed frames"

    n_cols = min(4, len(time_bin_indices))
    n_rows = int(np.ceil(len(time_bin_indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.4 * n_rows), squeeze=False)
    last_im = None
    extent = [
        float(result["x_edges_mm"][0]),
        float(result["x_edges_mm"][-1]),
        float(result["y_edges_mm"][-1]),
        float(result["y_edges_mm"][0]),
    ]
    for ax, idx, values in zip(axes.ravel(), time_bin_indices, maps):
        last_im = ax.imshow(
            values,
            aspect="equal",
            interpolation="none",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            cmap=plot_cmap,
        )
        start_h = float(tod_edges[int(idx)])
        stop_h = float(tod_edges[int(idx) + 1])
        ax.set_title(f"ZT {start_h:04.1f}-{stop_h:04.1f} h")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    for ax in axes.ravel()[len(time_bin_indices):]:
        ax.axis("off")
    fig.suptitle(f"{side} {cluster_id} occupancy relative to light on ({mode})")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.82, label=color_label)
    plt.show()


def transform_time_occupancy(
    occupancy: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    daily_mean = np.nanmean(occupancy, axis=0)
    if mode == "difference":
        return occupancy - daily_mean[None, :, :]
    if mode == "ratio":
        return occupancy / np.maximum(daily_mean[None, :, :], 1e-12)
    if mode == "occupancy":
        return occupancy
    raise ValueError("mode must be 'occupancy', 'difference', or 'ratio'")


def coarsen_time_occupancy(
    occupancy: np.ndarray,
    observed: np.ndarray,
    tod_edges_hours: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    factor = int(factor)
    if factor <= 1:
        return occupancy, observed, tod_edges_hours

    n_clusters, n_tod, n_y, n_x = occupancy.shape
    starts = list(range(0, n_tod, factor))
    coarsened = np.full((n_clusters, len(starts), n_y, n_x), np.nan, dtype=np.float32)
    coarsened_observed = np.zeros((n_clusters, len(starts)), dtype=observed.dtype)
    coarsened_edges = [float(tod_edges_hours[0])]

    for out_idx, start in enumerate(starts):
        stop = min(start + factor, n_tod)
        weights = observed[:, start:stop].astype(np.float64)
        weighted = occupancy[:, start:stop].astype(np.float64) * weights[:, :, None, None]
        numerator = np.nansum(weighted, axis=1)
        denominator = weights.sum(axis=1)
        np.divide(
            numerator,
            denominator[:, None, None],
            out=coarsened[:, out_idx],
            where=denominator[:, None, None] > 0,
        )
        coarsened_observed[:, out_idx] = observed[:, start:stop].sum(axis=1)
        coarsened_edges.append(float(tod_edges_hours[stop]))

    return coarsened, coarsened_observed, np.asarray(coarsened_edges, dtype=np.float32)


def spatially_bin_occupancy(occupancy: np.ndarray, factor: int | tuple[int, int]) -> np.ndarray:
    if isinstance(factor, tuple):
        y_factor, x_factor = factor
    else:
        y_factor = x_factor = int(factor)
    y_factor = int(y_factor)
    x_factor = int(x_factor)
    if y_factor <= 1 and x_factor <= 1:
        return occupancy
    if y_factor <= 0 or x_factor <= 0:
        raise ValueError("Spatial bin factors must be positive")

    n_tod, n_y, n_x = occupancy.shape
    pad_y = (-n_y) % y_factor
    pad_x = (-n_x) % x_factor
    padded = np.pad(
        occupancy.astype(np.float32, copy=False),
        ((0, 0), (0, pad_y), (0, pad_x)),
        mode="constant",
        constant_values=np.nan,
    )
    reshaped = padded.reshape(n_tod, padded.shape[1] // y_factor, y_factor, padded.shape[2] // x_factor, x_factor)
    finite_count = np.isfinite(reshaped).sum(axis=(2, 4))
    binned = np.nansum(reshaped, axis=(2, 4)).astype(np.float32)
    binned[finite_count == 0] = np.nan
    return binned


def smooth_occupancy_maps(occupancy: np.ndarray, sigma_bins: float | tuple[float, float]) -> np.ndarray:
    if isinstance(sigma_bins, tuple):
        sigma_y, sigma_x = sigma_bins
    else:
        sigma_y = sigma_x = float(sigma_bins)
    sigma_y = float(sigma_y)
    sigma_x = float(sigma_x)
    if sigma_y <= 0 and sigma_x <= 0:
        return occupancy

    from scipy.ndimage import gaussian_filter

    values = occupancy.astype(np.float32, copy=False)
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0).astype(np.float32, copy=False)
    weights = finite.astype(np.float32)
    sigma = (0.0, sigma_y, sigma_x)
    smooth_values = gaussian_filter(filled, sigma=sigma, mode="nearest")
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    out = np.full_like(values, np.nan, dtype=np.float32)
    np.divide(smooth_values, smooth_weights, out=out, where=smooth_weights > 1e-6)
    return out


def prepare_occupancy_for_tile_plot(
    occupancy: np.ndarray,
    observed: np.ndarray,
    tod_edges_hours: np.ndarray,
    *,
    time_bin_factor: int = 1,
    spatial_bin_factor: int | tuple[int, int] = 1,
    smooth_sigma_bins: float | tuple[float, float] = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occupancy, observed, tod_edges_hours = coarsen_time_occupancy(
        occupancy,
        observed,
        tod_edges_hours,
        time_bin_factor,
    )
    prepared = np.stack(
        [
            smooth_occupancy_maps(
                spatially_bin_occupancy(occupancy[cluster_idx], spatial_bin_factor),
                smooth_sigma_bins,
            )
            for cluster_idx in range(occupancy.shape[0])
        ],
        axis=0,
    )
    return prepared, observed, tod_edges_hours


def plot_all_cluster_time_tiles(
    results: dict[str, dict],
    *,
    side: str,
    time_bin_indices: list[int] | None = None,
    mode: str = "difference",
    time_bin_factor: int = 1,
    spatial_bin_factor: int | tuple[int, int] = 1,
    smooth_sigma_bins: float | tuple[float, float] = 0.0,
    vmax_percentile: float = 99.0,
    cmap: str = "viridis",
) -> None:
    result = results[side]
    cluster_ids = result["cluster_ids"].astype(str)
    occupancy, _, tod_edges = prepare_occupancy_for_tile_plot(
        result["occupancy"],
        result["observed"],
        result["tod_edges_hours"],
        time_bin_factor=time_bin_factor,
        spatial_bin_factor=spatial_bin_factor,
        smooth_sigma_bins=smooth_sigma_bins,
    )

    if time_bin_indices is None:
        n_bins = occupancy.shape[1]
        time_bin_indices = np.linspace(0, n_bins - 1, min(8, n_bins), dtype=int).tolist()

    transformed = np.stack(
        [transform_time_occupancy(occupancy[i], mode=mode) for i in range(len(cluster_ids))],
        axis=0,
    )
    selected = transformed[:, time_bin_indices, :, :]
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        raise ValueError(f"No finite occupancy values to plot for {side}")

    if mode == "difference":
        vmax = float(np.nanpercentile(np.abs(finite), vmax_percentile))
        vmin = -vmax
        plot_cmap = "coolwarm"
        color_label = "Occupancy - cluster daily mean"
    elif mode == "ratio":
        vmin = 0.0
        vmax = float(np.nanpercentile(finite, vmax_percentile))
        plot_cmap = cmap
        color_label = "Occupancy / cluster daily mean"
    else:
        vmin = 0.0
        vmax = float(np.nanpercentile(finite, vmax_percentile))
        plot_cmap = cmap
        color_label = "Fraction of observed frames"

    n_rows = len(cluster_ids)
    n_cols = len(time_bin_indices)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.7 * n_cols, 2.5 * n_rows),
        squeeze=False,
    )
    extent = [
        float(result["x_edges_mm"][0]),
        float(result["x_edges_mm"][-1]),
        float(result["y_edges_mm"][-1]),
        float(result["y_edges_mm"][0]),
    ]

    last_im = None
    for row_idx, cluster_id in enumerate(cluster_ids):
        for col_idx, tod_idx in enumerate(time_bin_indices):
            ax = axes[row_idx, col_idx]
            last_im = ax.imshow(
                transformed[row_idx, int(tod_idx)],
                aspect="equal",
                interpolation="none",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
                cmap=plot_cmap,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                start_h = float(tod_edges[int(tod_idx)])
                stop_h = float(tod_edges[int(tod_idx) + 1])
                ax.set_title(f"ZT {start_h:04.1f}-{stop_h:04.1f} h", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(cluster_id, rotation=0, ha="right", va="center", labelpad=28)

    fig.suptitle(
        f"{side} cluster occupancy relative to light on ({mode}); "
        f"time x{time_bin_factor}, spatial x{spatial_bin_factor}, smooth={smooth_sigma_bins}"
    )
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8, label=color_label)
    plt.show()


def plot_cluster_time_modulation(results: dict[str, dict], *, side: str) -> pd.DataFrame:
    result = results[side]
    rows = []
    hours_since_light_on = 0.5 * (result["tod_edges_hours"][:-1] + result["tod_edges_hours"][1:])
    for cluster_idx, cluster_id in enumerate(result["cluster_ids"].astype(str)):
        occupancy = result["occupancy"][cluster_idx]
        daily_mean = np.nanmean(occupancy, axis=0)
        modulation = np.nanmean(np.abs(occupancy - daily_mean), axis=(1, 2))
        observed = result["observed"][cluster_idx]
        rows.append(
            pd.DataFrame(
                {
                    "side": side,
                    "cluster_id": cluster_id,
                    "hours_since_light_on": hours_since_light_on,
                    "mean_abs_occupancy_change": modulation,
                    "observed_frames": observed,
                }
            )
        )
    table = pd.concat(rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    for cluster_id, group in table.groupby("cluster_id", sort=False):
        ax.plot(group["hours_since_light_on"], group["mean_abs_occupancy_change"], lw=1.8, label=cluster_id)
    ax.set_xlabel("Hours since lights on at 05:30")
    ax.set_ylabel("Mean absolute occupancy change")
    ax.set_title(f"{side} cluster spatial modulation relative to light on")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.show()
    return table


# %%
# Load cluster membership saved by grid_occupancy.py.
cluster_tracks = load_cluster_tracks(GRID_ROOT, CLUSTER_TABLE_PATH, PER_TRACK_ROOT)
grid_specs = side_grid_specs(cluster_tracks)
recording_start_clock_seconds = recording_start_seconds_from_metadata(cluster_tracks)
light_on_clock_seconds = int(round(float(LIGHT_ON_HOUR) * 3600.0))

print(f"Loaded {len(cluster_tracks)} clustered tracks from {CLUSTER_TABLE_PATH}")
print(f"Recording start from metadata: {go.format_clock_time(recording_start_clock_seconds)}")
print(f"Light on reference: {go.format_clock_time(light_on_clock_seconds)}")
display(cluster_tracks.groupby(["side", "cluster_id"])["track_name"].count().rename("n_tracks"))
display(cluster_tracks[["side", "track_id", "cluster_id", "track_name", "track_path"]].head())


# %%
# Compute once, or load existing saved results.
settings = {
    "grid_root": str(GRID_ROOT),
    "per_track_root": str(PER_TRACK_ROOT),
    "cluster_table_path": str(CLUSTER_TABLE_PATH),
    "fps": FPS,
    "tod_bin_minutes": TOD_BIN_MINUTES,
    "recording_start_clock_seconds": int(recording_start_clock_seconds),
    "recording_start_clock": go.format_clock_time(recording_start_clock_seconds),
    "light_on_clock_seconds": int(light_on_clock_seconds),
    "light_on_clock": go.format_clock_time(light_on_clock_seconds),
    "time_axis": "hours_since_light_on",
    "bodypoint": BODYPOINT,
    "frame_col": FRAME_COL,
    "x_col": X_COL,
    "y_col": Y_COL,
}

expected_outputs = [
    OUT_ROOT / f"{side}_time_of_day_cluster_occupancy.npz"
    for side in cluster_tracks["side"].drop_duplicates()
]
if RECOMPUTE or not all(path.exists() for path in expected_outputs):
    should_compute = True
elif not saved_settings_match(OUT_ROOT, settings):
    print("Saved outputs use different settings; recomputing.")
    should_compute = True
else:
    should_compute = False

if should_compute:
    time_of_day_results = compute_time_of_day_occupancy(
        cluster_tracks,
        grid_specs,
        fps=FPS,
        tod_bin_minutes=TOD_BIN_MINUTES,
        recording_start_clock_seconds=recording_start_clock_seconds,
        light_on_clock_seconds=light_on_clock_seconds,
        bodypoint=BODYPOINT,
        frame_col=FRAME_COL,
        x_col=X_COL,
        y_col=Y_COL,
    )
    save_results(time_of_day_results, cluster_tracks, OUT_ROOT, settings)
else:
    time_of_day_results = load_results(OUT_ROOT)

print(f"Time-of-day occupancy results: {OUT_ROOT}")
for side, result in time_of_day_results.items():
    print(
        f"{side}: occupancy={result['occupancy'].shape}, "
        f"clusters={result['cluster_ids'].astype(str).tolist()}, "
        f"relative_light_on_bins={len(result['tod_edges_hours']) - 1}"
    )


# %%
# Plot all clusters as tiles across time of day.
PLOT_SIDES = tuple(time_of_day_results.keys())
PLOT_TIME_BIN_FACTOR = 2      # Combine neighboring computed time bins. 2 means 1 h if TOD_BIN_MINUTES is 30.
PLOT_SPATIAL_BIN_FACTOR = 10   # Combine neighboring grid cells for display. 5 means 5 mm if grid cells are 1 mm.
PLOT_SMOOTH_SIGMA_BINS = 1.0  # Gaussian sigma after spatial binning, in displayed bins. Set 0 for none.
PLOT_TIME_BIN_INDICES = None  # Indices after PLOT_TIME_BIN_FACTOR. Example: list(range(24)) for 1 h bins.
PLOT_MODE = "difference"      # "difference", "occupancy", or "ratio"

for plot_side in PLOT_SIDES:
    plot_all_cluster_time_tiles(
        time_of_day_results,
        side=plot_side,
        time_bin_indices=PLOT_TIME_BIN_INDICES,
        mode=PLOT_MODE,
        time_bin_factor=PLOT_TIME_BIN_FACTOR,
        spatial_bin_factor=PLOT_SPATIAL_BIN_FACTOR,
        smooth_sigma_bins=PLOT_SMOOTH_SIGMA_BINS,
    )


# %%
# Plot a compact time-of-day modulation summary for one side.
MODULATION_SIDE = "right"

modulation_table = plot_cluster_time_modulation(time_of_day_results, side=MODULATION_SIDE)
display(modulation_table.head())

# %%
