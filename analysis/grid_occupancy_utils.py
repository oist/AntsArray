"""Helpers for interactive grid-occupancy histogram clustering."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def side_from_track(track_name: str, track_dir: Path | None = None) -> str | None:
    stem = track_name.lower()
    dir_name = track_dir.name.lower() if track_dir is not None else ""
    if stem.endswith("_left.parquet") or stem.endswith("_left") or dir_name.endswith("_left"):
        return "left"
    if stem.endswith("_right.parquet") or stem.endswith("_right") or dir_name.endswith("_right"):
        return "right"
    return None


def track_id_from_name(track_name: str) -> int | None:
    match = re.search(r"TrackID_(\d+)", track_name)
    return int(match.group(1)) if match else None


def parse_track_start_seconds(track_name: str) -> int:
    match = re.search(r"_all_(\d{6})_", track_name)
    if match is None:
        match = re.search(r"_(\d{6})_(?:left|right)\.parquet$", track_name)
    if match is None:
        raise ValueError(f"Could not parse HHMMSS start time from {track_name!r}")

    stamp = match.group(1)
    hour = int(stamp[:2])
    minute = int(stamp[2:4])
    second = int(stamp[4:6])
    return hour * 3600 + minute * 60 + second


def format_clock_time(seconds: float) -> str:
    seconds_i = int(round(seconds)) % (24 * 3600)
    return f"{seconds_i // 3600:02d}:{(seconds_i % 3600) // 60:02d}"


def start_time_from_track_table(track_table: pd.DataFrame) -> int:
    return parse_track_start_seconds(str(track_table["track_name"].iloc[0]))


def metadata_paths(grid_root: Path) -> list[Path]:
    root = Path(grid_root)
    paths = sorted((root / "per_track").glob("*/grid_occupancy_metadata.json"))
    if not paths:
        paths = sorted(root.glob("*/grid_occupancy_metadata.json"))
    return paths


def infer_speed_root(grid_root: Path) -> Path:
    return Path(grid_root).parent / "speed_vectors"


def load_grid_tracks(grid_root: Path) -> pd.DataFrame:
    rows = []
    for metadata_path in metadata_paths(Path(grid_root)):
        with metadata_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)

        track_name = str(meta.get("track_name", metadata_path.parent.name))
        side = str(meta.get("side") or side_from_track(track_name, metadata_path.parent) or "")
        if side not in {"left", "right"}:
            continue

        frame_min = int(meta["frame_min"])
        frame_max = int(meta["frame_max"])
        n_span_frames = frame_max - frame_min + 1
        n_observed = int(meta.get("n_observed_frames", meta.get("n_detected_frames", 0)))
        occupancy_path = metadata_path.parent / "grid_occupancy_f4.npy"
        rows.append(
            {
                "track_name": track_name,
                "track_id": meta.get("track_id", track_id_from_name(track_name)),
                "side": side,
                "metadata_path": metadata_path,
                "occupancy_path": occupancy_path,
                "x_edges_path": metadata_path.parent / "grid_x_edges_mm.npy",
                "y_edges_path": metadata_path.parent / "grid_y_edges_mm.npy",
                "frame_min": frame_min,
                "frame_max": frame_max,
                "n_frames": n_span_frames,
                "n_observed_frames": n_observed,
                "present_frac": n_observed / n_span_frames if n_span_frames > 0 else np.nan,
                "n_in_grid_frames": int(meta.get("n_in_grid_frames", 0)),
                "n_out_of_grid_detected_frames": int(meta.get("n_out_of_grid_detected_frames", 0)),
                "occupancy_sum": float(meta.get("occupancy_sum", np.nan)),
                "histogram_shape_yx": tuple(meta.get("histogram_shape_yx", [])),
                "grid_size_mm": float(meta.get("grid_size_mm", np.nan)),
                "mm_per_px": float(meta.get("mm_per_px", np.nan)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise FileNotFoundError(f"No grid_occupancy_metadata.json files found under {grid_root}")
    return out.sort_values(["side", "track_id", "track_name"], na_position="last").reset_index(drop=True)


def load_speed_detection_table(speed_root: Path) -> pd.DataFrame:
    rows = []
    paths = sorted((Path(speed_root) / "per_track").glob("*/speed_metadata.json"))
    if not paths:
        paths = sorted(Path(speed_root).glob("*/speed_metadata.json"))

    for metadata_path in paths:
        with metadata_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        track_name = str(meta.get("track_name", metadata_path.parent.name))
        n_frames = int(meta["n_frames"])
        n_observed = int(meta["n_observed_frames"])
        rows.append(
            {
                "track_name": track_name,
                "speed_metadata_path": metadata_path,
                "speed_n_frames": n_frames,
                "speed_n_observed_frames": n_observed,
                "present_frac": n_observed / n_frames if n_frames else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise FileNotFoundError(f"No speed_metadata.json files found under {speed_root}")
    return out.sort_values("track_name").reset_index(drop=True)


def load_speed_tracks(speed_root: Path) -> pd.DataFrame:
    rows = []
    paths = sorted((Path(speed_root) / "per_track").glob("*/speed_metadata.json"))
    if not paths:
        paths = sorted(Path(speed_root).glob("*/speed_metadata.json"))

    for metadata_path in paths:
        with metadata_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        track_name = str(meta.get("track_name", metadata_path.parent.name))
        side = side_from_track(track_name, metadata_path.parent)
        if side is None:
            continue
        n_frames = int(meta["n_frames"])
        n_observed = int(meta["n_observed_frames"])
        rows.append(
            {
                "track_name": track_name,
                "track_id": meta.get("track_id", track_id_from_name(track_name)),
                "side": side,
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
    return out.sort_values(["side", "track_id", "track_name"]).reset_index(drop=True)


def rolling_nanmean(values: np.ndarray, window_bins: int) -> np.ndarray:
    if window_bins <= 1:
        return values.astype(np.float32, copy=True)
    return (
        pd.Series(values)
        .rolling(window=window_bins, center=True, min_periods=max(1, int(np.ceil(window_bins * 0.25))))
        .mean()
        .to_numpy(dtype=np.float32)
    )


def add_light_shading(
    ax: plt.Axes,
    start_clock_seconds: int,
    max_time_h: float,
    *,
    min_time_h: float = 0.0,
    light_off_hour: float = 18.0,
    light_on_hour: float = 6.0,
    shade_color: str = "0.88",
    shade_alpha: float = 0.45,
) -> None:
    dark_duration_h = (float(light_on_hour) - float(light_off_hour)) % 24.0
    if dark_duration_h == 0:
        dark_duration_h = 24.0

    off_clock_seconds = float(light_off_hour) * 3600.0
    first_dark_start_h = (off_clock_seconds - float(start_clock_seconds)) / 3600.0
    while first_dark_start_h > min_time_h:
        first_dark_start_h -= 24.0

    label_used = False
    dark_start_h = first_dark_start_h
    while dark_start_h <= max_time_h + 1e-9:
        dark_end_h = dark_start_h + dark_duration_h
        span_start = max(float(min_time_h), dark_start_h)
        span_end = min(float(max_time_h), dark_end_h)
        if span_end > span_start:
            ax.axvspan(
                span_start,
                span_end,
                color=shade_color,
                alpha=shade_alpha,
                lw=0,
                zorder=0,
                label="dark" if not label_used else None,
            )
            label_used = True
        dark_start_h += 24.0

    events = [
        (float(light_off_hour) * 3600.0, "lights off", "0.20", "--"),
        (float(light_on_hour) * 3600.0, "lights on", "0.65", "-."),
    ]
    for clock_seconds, label_suffix, color, linestyle in events:
        first_h = (clock_seconds - float(start_clock_seconds)) / 3600.0
        while first_h < min_time_h:
            first_h += 24.0
        for i, time_h in enumerate(np.arange(first_h, max_time_h + 1e-9, 24.0)):
            ax.axvline(
                time_h,
                color=color,
                linestyle=linestyle,
                lw=1,
                alpha=0.75,
                zorder=1,
                label=f"{format_clock_time(clock_seconds)} {label_suffix}" if i == 0 else None,
            )


def attach_detection_fraction(tracks: pd.DataFrame, speed_root: Path | None = None) -> pd.DataFrame:
    out = tracks.copy()
    if speed_root is None:
        out["present_frac_source"] = "grid_metadata_frame_span"
        return out

    speed_root = Path(speed_root)
    if not speed_root.exists():
        out["present_frac_source"] = "grid_metadata_frame_span"
        print(f"WARNING: speed metadata root does not exist; using grid metadata frame span: {speed_root}")
        return out

    try:
        speed_table = load_speed_detection_table(speed_root)
    except FileNotFoundError:
        out["present_frac_source"] = "grid_metadata_frame_span"
        print(f"WARNING: no speed metadata found; using grid metadata frame span: {speed_root}")
        return out
    out = out.merge(
        speed_table,
        on="track_name",
        how="left",
        suffixes=("", "_speed"),
        validate="one_to_one",
    )
    has_speed = out["speed_metadata_path"].notna()
    out["grid_present_frac"] = out["present_frac"]
    out.loc[has_speed, "present_frac"] = out.loc[has_speed, "present_frac_speed"]
    out["present_frac_source"] = np.where(has_speed, "speed_metadata", "grid_metadata_frame_span")
    out = out.drop(columns=["present_frac_speed"])

    missing = int((~has_speed).sum())
    if missing:
        print(f"WARNING: missing speed metadata for {missing}/{len(out)} grid tracks; using grid metadata fallback")
    return out


def select_good_tracks(
    tracks: pd.DataFrame,
    min_present_frac: float = 0.40,
    *,
    side: str | None = "both",
) -> pd.DataFrame:
    out = tracks.copy()
    if side not in (None, "both"):
        out = out[out["side"] == side]
    out = out[
        (out["present_frac"] > float(min_present_frac))
        & out["occupancy_path"].map(lambda path: Path(path).exists())
        & out["x_edges_path"].map(lambda path: Path(path).exists())
        & out["y_edges_path"].map(lambda path: Path(path).exists())
    ].copy()
    if out.empty:
        raise ValueError(f"No grid tracks passed min_present_frac={min_present_frac}")
    return out.reset_index(drop=True)


def choose_track(
    tracks: pd.DataFrame,
    *,
    row_number: int | None = None,
    track_id: int | str | None = None,
    side: str | None = "left",
) -> pd.Series:
    if row_number is not None:
        return tracks.loc[int(row_number)]

    chosen = tracks if side in (None, "both") else tracks[tracks["side"] == side]
    if track_id is not None:
        chosen = chosen[chosen["track_id"].astype(str) == str(track_id)]
    if chosen.empty:
        raise ValueError("No matching track found")
    return chosen.iloc[0]


def load_histogram(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hist = np.load(row["occupancy_path"]).astype(np.float32, copy=False)
    x_edges = np.load(row["x_edges_path"]).astype(np.float32, copy=False)
    y_edges = np.load(row["y_edges_path"]).astype(np.float32, copy=False)
    return hist, x_edges, y_edges


def display_histogram(hist: np.ndarray, mode: str) -> np.ndarray:
    if mode == "linear":
        return hist
    if mode == "sqrt":
        return np.sqrt(np.clip(hist, 0, None))
    if mode == "log1p":
        return np.log1p(np.clip(hist, 0, None))
    raise ValueError("mode must be one of: linear, sqrt, log1p")


def plot_single_histogram(
    tracks: pd.DataFrame,
    *,
    row_number: int | None = None,
    track_id: int | str | None = None,
    side: str | None = "left",
    mode: str = "sqrt",
    vmin: float | None = 0.0,
    vmax: float | None = None,
    vmax_percentile: float | None = 99.0,
    cmap: str = "viridis",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    row = choose_track(tracks, row_number=row_number, track_id=track_id, side=side)
    hist, x_edges, y_edges = load_histogram(row)
    image = display_histogram(hist, mode)
    if vmax is None and vmax_percentile is not None:
        vmax = float(np.nanpercentile(image, float(vmax_percentile)))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        image,
        origin="lower",
        aspect="equal",
        interpolation="none",
        extent=[float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    label = f"{row['side']} track {row['track_id']} row {row.name}"
    ax.set_title(f"Grid occupancy: {label}")
    ax.set_xlabel("x within colony side (mm)")
    ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=ax, label=f"occupancy ({mode})")
    fig.tight_layout()
    plt.show()
    return hist, x_edges, y_edges, row


def feature_transform_matrix(matrix: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        out = matrix.astype(np.float32, copy=True)
    elif transform == "sqrt":
        out = np.sqrt(np.clip(matrix, 0, None)).astype(np.float32, copy=False)
    elif transform == "log1p":
        out = np.log1p(np.clip(matrix, 0, None)).astype(np.float32, copy=False)
    else:
        raise ValueError("transform must be one of: none, sqrt, log1p")
    out[~np.isfinite(out)] = 0
    return out


def build_histogram_matrix(
    tracks: pd.DataFrame,
    *,
    side: str | None = "both",
    transform: str = "sqrt",
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    chosen = tracks if side in (None, "both") else tracks[tracks["side"] == side]
    if chosen.empty:
        raise ValueError("No tracks selected")

    rows = []
    x_ref = None
    y_ref = None
    shape_ref = None
    for i, (_, row) in enumerate(chosen.iterrows(), start=1):
        if i == 1 or i == len(chosen) or i % 25 == 0:
            print(f"loading histogram {i}/{len(chosen)} {row['track_name']}")
        hist, x_edges, y_edges = load_histogram(row)
        if shape_ref is None:
            shape_ref = hist.shape
            x_ref = x_edges
            y_ref = y_edges
        elif hist.shape != shape_ref or not np.array_equal(x_edges, x_ref) or not np.array_equal(y_edges, y_ref):
            raise ValueError(
                "All histograms must have the same shape and edges for clustering. "
                "If left and right colonies do not line up, run clustering separately with side='left' "
                "and side='right'. "
                f"First shape={shape_ref}, current shape={hist.shape}, track={row['track_name']}"
            )
        rows.append(hist.reshape(-1))

    matrix = np.vstack(rows).astype(np.float32, copy=False)
    features = feature_transform_matrix(matrix, transform)
    return features, chosen.reset_index(drop=True), x_ref, y_ref


def check_clustering_dependencies() -> None:
    missing = []
    hints = {
        "sklearn": "scikit-learn",
        "umap": "umap-learn",
        "igraph": "python-igraph",
        "leidenalg": "leidenalg",
    }
    for module_name, package_name in hints.items():
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(package_name)
    if missing:
        raise ImportError(
            "Missing clustering dependencies: "
            + ", ".join(missing)
            + ". Install them in the notebook environment before running clustering."
        )


def knn_graph_edges(features: np.ndarray, n_neighbors: int, metric: str) -> tuple[list[tuple[int, int]], list[float]]:
    from sklearn.neighbors import NearestNeighbors

    n_samples = int(features.shape[0])
    if n_samples < 2:
        return [], []

    k = min(max(2, int(n_neighbors) + 1), n_samples)
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    finite_dist = distances[:, 1:][np.isfinite(distances[:, 1:])]
    sigma = float(np.median(finite_dist)) if finite_dist.size else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    edge_weights: dict[tuple[int, int], float] = {}
    for i in range(n_samples):
        for j, dist in zip(indices[i, 1:], distances[i, 1:]):
            j = int(j)
            if i == j:
                continue
            a, b = sorted((int(i), j))
            weight = float(np.exp(-0.5 * (float(dist) / sigma) ** 2))
            edge_weights[(a, b)] = max(edge_weights.get((a, b), 0.0), weight)

    edges = list(edge_weights.keys())
    weights = [edge_weights[edge] for edge in edges]
    return edges, weights


def leiden_labels(
    features: np.ndarray,
    *,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    resolution: float = 1.0,
    random_state: int = 0,
) -> np.ndarray:
    import igraph as ig
    import leidenalg

    n_samples = int(features.shape[0])
    if n_samples == 0:
        return np.zeros(0, dtype=int)
    if n_samples == 1:
        return np.zeros(1, dtype=int)

    edges, weights = knn_graph_edges(features, n_neighbors=n_neighbors, metric=metric)
    graph = ig.Graph(n=n_samples, edges=edges, directed=False)
    graph.es["weight"] = weights
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=int(random_state),
    )
    return np.asarray(partition.membership, dtype=int)


def umap_embedding(
    features: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.15,
    metric: str = "euclidean",
    random_state: int = 0,
) -> np.ndarray:
    import umap

    if len(features) < 3:
        raise ValueError("Need at least 3 tracks for UMAP embedding")

    reducer = umap.UMAP(
        n_neighbors=min(max(2, int(n_neighbors)), len(features) - 1),
        min_dist=float(min_dist),
        metric=metric,
        random_state=int(random_state),
    )
    return reducer.fit_transform(features)


def run_umap_leiden(
    tracks: pd.DataFrame,
    *,
    side: str | None = "both",
    feature_transform: str = "sqrt",
    neighbor_metric: str = "euclidean",
    n_neighbors: int = 15,
    umap_min_dist: float = 0.15,
    leiden_resolution: float = 1.0,
    random_state: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    check_clustering_dependencies()
    features, chosen, _x_edges, _y_edges = build_histogram_matrix(
        tracks,
        side=side,
        transform=feature_transform,
    )
    labels = leiden_labels(
        features,
        n_neighbors=n_neighbors,
        metric=neighbor_metric,
        resolution=leiden_resolution,
        random_state=random_state,
    )
    embedding = umap_embedding(
        features,
        n_neighbors=n_neighbors,
        min_dist=umap_min_dist,
        metric=neighbor_metric,
        random_state=random_state,
    )

    out = chosen.copy()
    out["leiden_cluster"] = labels
    out["umap1"] = embedding[:, 0]
    out["umap2"] = embedding[:, 1]
    out["feature_transform"] = feature_transform
    out["neighbor_metric"] = neighbor_metric
    out["n_neighbors"] = int(n_neighbors)
    out["leiden_resolution"] = float(leiden_resolution)
    return out, features, embedding


def plot_umap_clusters(
    cluster_table: pd.DataFrame,
    *,
    color_col: str = "leiden_cluster",
    annotate_clusters: bool = True,
    cmap: str = "tab20",
    title: str | None = None,
) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(7, 6))
    values = cluster_table[color_col]
    scatter = ax.scatter(
        cluster_table["umap1"],
        cluster_table["umap2"],
        c=values.astype("category").cat.codes if values.dtype == object else values,
        s=45,
        alpha=0.85,
        cmap=cmap,
        edgecolor="none",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title or f"Grid occupancy UMAP colored by {color_col}")
    ax.grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=ax, label=color_col)

    if annotate_clusters and color_col in cluster_table:
        for cluster, group in cluster_table.groupby(color_col):
            ax.text(
                float(group["umap1"].median()),
                float(group["umap2"].median()),
                str(cluster),
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
            )

    fig.tight_layout()
    plt.show()
    return ax


def plot_cluster_mean_histograms(
    tracks: pd.DataFrame,
    cluster_table: pd.DataFrame,
    *,
    cluster_col: str = "leiden_cluster",
    mode: str = "sqrt",
    vmax_percentile: float | None = 99.0,
    cmap: str = "viridis",
    title: str | None = None,
) -> dict[int, np.ndarray]:
    clusters = sorted(cluster_table[cluster_col].dropna().unique())
    if not clusters:
        raise ValueError("No clusters to plot")

    first_row = tracks[tracks["track_name"] == cluster_table["track_name"].iloc[0]].iloc[0]
    _first_hist, x_edges, y_edges = load_histogram(first_row)
    mean_hists: dict[int, np.ndarray] = {}
    for cluster in clusters:
        names = set(cluster_table.loc[cluster_table[cluster_col] == cluster, "track_name"])
        rows = tracks[tracks["track_name"].isin(names)]
        stack = [load_histogram(row)[0] for _, row in rows.iterrows()]
        mean_hists[int(cluster)] = np.nanmean(np.stack(stack, axis=0), axis=0)

    images = [display_histogram(hist, mode) for hist in mean_hists.values()]
    vmax = None
    if vmax_percentile is not None:
        vmax = float(np.nanpercentile(np.concatenate([image.ravel() for image in images]), vmax_percentile))

    n_clusters = len(clusters)
    n_cols = min(4, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.6 * n_rows), squeeze=False)
    last_im = None
    for ax, cluster in zip(axes.ravel(), clusters):
        image = display_histogram(mean_hists[int(cluster)], mode)
        last_im = ax.imshow(
            image,
            origin="lower",
            aspect="equal",
            interpolation="none",
            extent=[float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])],
            vmin=0,
            vmax=vmax,
            cmap=cmap,
        )
        n = int((cluster_table[cluster_col] == cluster).sum())
        ax.set_title(f"cluster {cluster} n={n}")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    for ax in axes.ravel()[n_clusters:]:
        ax.axis("off")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8, label=f"mean occupancy ({mode})")
    if title is not None:
        fig.suptitle(title)
    plt.show()
    return mean_hists


def plot_cluster_example_histograms(
    tracks: pd.DataFrame,
    cluster_table: pd.DataFrame,
    *,
    n_examples: int = 6,
    cluster_col: str = "leiden_cluster",
    mode: str = "sqrt",
    vmax_percentile: float | None = 99.0,
    cmap: str = "viridis",
    random_state: int = 0,
    title: str | None = None,
) -> dict[int, pd.DataFrame]:
    clusters = sorted(cluster_table[cluster_col].dropna().unique())
    if not clusters:
        raise ValueError("No clusters to plot")
    if n_examples <= 0:
        raise ValueError("n_examples must be positive")

    rng = np.random.default_rng(int(random_state))
    selected: dict[int, pd.DataFrame] = {}
    images = []
    rows_for_plot: list[tuple[int, pd.Series, np.ndarray, np.ndarray, np.ndarray]] = []
    max_examples = 0
    for cluster in clusters:
        cluster_rows = cluster_table[cluster_table[cluster_col] == cluster].copy()
        if len(cluster_rows) > int(n_examples):
            pick = rng.choice(cluster_rows.index.to_numpy(), size=int(n_examples), replace=False)
            cluster_rows = cluster_rows.loc[np.sort(pick)]
        selected[int(cluster)] = cluster_rows
        max_examples = max(max_examples, len(cluster_rows))

        for _, cluster_row in cluster_rows.iterrows():
            track_rows = tracks[tracks["track_name"] == cluster_row["track_name"]]
            if track_rows.empty:
                raise ValueError(f"Missing track row for {cluster_row['track_name']}")
            track_row = track_rows.iloc[0]
            hist, x_edges, y_edges = load_histogram(track_row)
            image = display_histogram(hist, mode)
            images.append(image)
            rows_for_plot.append((int(cluster), track_row, image, x_edges, y_edges))

    vmax = None
    if vmax_percentile is not None and images:
        vmax = float(np.nanpercentile(np.concatenate([image.ravel() for image in images]), vmax_percentile))

    n_rows = len(clusters)
    n_cols = max_examples
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )
    last_im = None
    positions = {cluster: 0 for cluster in clusters}
    for cluster, track_row, image, x_edges, y_edges in rows_for_plot:
        row_idx = clusters.index(cluster)
        col_idx = positions[cluster]
        positions[cluster] += 1
        ax = axes[row_idx, col_idx]
        last_im = ax.imshow(
            image,
            origin="lower",
            aspect="equal",
            interpolation="none",
            extent=[float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])],
            vmin=0,
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_title(f"c{cluster} row {track_row.name} id {track_row['track_id']}", fontsize=9)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

    for row_idx, cluster in enumerate(clusters):
        for col_idx in range(positions[cluster], n_cols):
            axes[row_idx, col_idx].axis("off")

    if title is not None:
        fig.suptitle(title)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8, label=f"occupancy ({mode})")
    plt.show()
    return selected


def binned_track_speed(row: pd.Series, bin_seconds: float) -> pd.DataFrame:
    speed = np.load(row["speed_path"], mmap_mode="r")
    fps = float(row["fps"])
    frame_min = int(row["frame_min"])

    bin_frames = max(1, int(round(fps * float(bin_seconds))))
    first_bin = frame_min // bin_frames
    n_bins = int(np.ceil((frame_min + len(speed)) / bin_frames)) - first_bin
    values = np.full(n_bins, np.nan, dtype=np.float32)

    valid = np.isfinite(speed)
    if valid.any():
        valid_idx = np.flatnonzero(valid)
        local_bin_idx = ((frame_min + valid_idx) // bin_frames) - first_bin
        bin_sum = np.bincount(local_bin_idx, weights=speed[valid_idx], minlength=n_bins)
        bin_count = np.bincount(local_bin_idx, minlength=n_bins)
        keep = bin_count > 0
        values[keep] = (bin_sum[keep] / bin_count[keep]).astype(np.float32)

    return pd.DataFrame(
        {
            "time_h": ((first_bin + np.arange(n_bins)) * bin_frames) / fps / 3600.0,
            "speed_mm_s": values,
            "track_name": row["track_name"],
            "track_id": row["track_id"],
            "side": row["side"],
        }
    )


def build_quiet_period_image(
    cluster_table: pd.DataFrame,
    speed_root: Path,
    *,
    speed_threshold_mm_s: float = 0.1,
    bin_seconds: float = 60.0,
    cluster_col: str = "leiden_cluster",
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    if cluster_col not in cluster_table.columns:
        raise ValueError(f"cluster_table is missing {cluster_col!r}")

    speed_tracks = load_speed_tracks(speed_root)
    speed_cols = [
        "track_name",
        "speed_path",
        "frame_min",
        "frame_max",
        "n_frames",
        "fps",
    ]
    speed_for_merge = speed_tracks[speed_cols].rename(
        columns={
            "frame_min": "speed_vector_frame_min",
            "frame_max": "speed_vector_frame_max",
            "n_frames": "speed_vector_n_frames",
            "fps": "speed_vector_fps",
        }
    )
    tracks = cluster_table.merge(
        speed_for_merge,
        on="track_name",
        how="left",
        validate="one_to_one",
    )
    missing = tracks["speed_path"].isna() | ~tracks["speed_path"].map(
        lambda path: Path(path).exists() if pd.notna(path) else False
    )
    if missing.any():
        examples = tracks.loc[missing, "track_name"].head(10).to_list()
        raise FileNotFoundError(f"Missing speed vectors for {int(missing.sum())} clustered tracks, examples: {examples}")

    tracks["frame_min"] = tracks["speed_vector_frame_min"].astype("int64")
    tracks["frame_max"] = tracks["speed_vector_frame_max"].astype("int64")
    tracks["n_frames"] = tracks["speed_vector_n_frames"].astype("int64")
    tracks["fps"] = tracks["speed_vector_fps"].astype("float64")
    if "cluster_id" not in tracks.columns:
        tracks["cluster_id"] = tracks["side"].astype(str) + "_" + tracks[cluster_col].astype(str)
    tracks = tracks.sort_values(["side", cluster_col, "track_id", "track_name"], kind="mergesort").reset_index(drop=True)

    fps_values = tracks["fps"].dropna().unique()
    fps = float(fps_values[0])
    if len(fps_values) > 1:
        print(f"WARNING: multiple FPS values in quiet image: {fps_values}; using {fps}")

    bin_frames = max(1, int(round(fps * float(bin_seconds))))
    n_bins = int(np.ceil((tracks["frame_min"] + tracks["n_frames"]).max() / bin_frames))
    image = np.full((len(tracks), n_bins), np.nan, dtype=np.float32)
    quiet_frac = np.full(len(tracks), np.nan, dtype=np.float32)
    n_valid_frames = np.zeros(len(tracks), dtype=np.int64)
    n_quiet_frames = np.zeros(len(tracks), dtype=np.int64)

    for image_row, (_, row) in enumerate(tracks.iterrows()):
        if image_row == 0 or image_row == len(tracks) - 1 or (image_row + 1) % 25 == 0:
            print(f"quiet image: loading {image_row + 1}/{len(tracks)} {row['track_name']}")
        speed = np.load(row["speed_path"], mmap_mode="r")
        valid = np.isfinite(speed)
        if not valid.any():
            continue
        valid_idx = np.flatnonzero(valid)
        quiet = speed[valid_idx] <= float(speed_threshold_mm_s)
        bin_idx = (int(row["frame_min"]) + valid_idx) // bin_frames

        quiet_sum = np.bincount(bin_idx, weights=quiet.astype(np.float32), minlength=n_bins)
        valid_count = np.bincount(bin_idx, minlength=n_bins)
        keep = valid_count > 0
        image[image_row, keep] = (quiet_sum[keep] / valid_count[keep]).astype(np.float32)

        n_valid_frames[image_row] = int(valid_count.sum())
        n_quiet_frames[image_row] = int(quiet.sum())
        quiet_frac[image_row] = n_quiet_frames[image_row] / n_valid_frames[image_row]

    image_tracks = tracks.copy()
    image_tracks.insert(0, "image_row", np.arange(len(image_tracks)))
    image_tracks["quiet_threshold_mm_s"] = float(speed_threshold_mm_s)
    image_tracks["quiet_frac_valid"] = quiet_frac
    image_tracks["n_valid_speed_frames"] = n_valid_frames
    image_tracks["n_quiet_speed_frames"] = n_quiet_frames
    time_h = np.arange(n_bins) * bin_frames / fps / 3600.0
    return image, image_tracks, time_h


def plot_quiet_period_image(
    cluster_table: pd.DataFrame,
    speed_root: Path,
    *,
    speed_threshold_mm_s: float = 0.1,
    bin_seconds: float = 60.0,
    cluster_col: str = "leiden_cluster",
    start_clock_seconds: int | None = None,
    light_off_hour: float = 18.0,
    light_on_hour: float = 6.0,
    show_light_lines: bool = True,
    cmap: str = "Greys",
    title: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    image, image_tracks, time_h = build_quiet_period_image(
        cluster_table,
        speed_root,
        speed_threshold_mm_s=speed_threshold_mm_s,
        bin_seconds=bin_seconds,
        cluster_col=cluster_col,
    )

    fig_height = min(12.0, max(5.0, 0.055 * len(image_tracks)))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    extent = [float(time_h[0]), float(time_h[-1]), image.shape[0] - 0.5, -0.5]
    im = ax.imshow(
        image,
        aspect="auto",
        interpolation="none",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
    )

    group_key = image_tracks["cluster_id"].astype(str).to_numpy()
    boundaries = np.flatnonzero(group_key[1:] != group_key[:-1]) + 0.5
    for boundary in boundaries:
        ax.axhline(boundary, color="tab:red", lw=0.8, alpha=0.8)

    if len(image_tracks):
        starts = np.r_[0, boundaries + 0.5].astype(int)
        stops = np.r_[boundaries + 0.5, len(image_tracks)].astype(int)
        centers = (starts + stops - 1) / 2
        labels = [f"{group_key[start]} n={stop - start}" for start, stop in zip(starts, stops)]
        ax.set_yticks(centers)
        ax.set_yticklabels(labels, fontsize=8)

    if start_clock_seconds is None and not image_tracks.empty:
        start_clock_seconds = start_time_from_track_table(image_tracks)
    if show_light_lines and start_clock_seconds is not None:
        events = [
            (float(light_off_hour) * 3600.0, "0.20", "--"),
            (float(light_on_hour) * 3600.0, "0.65", "-."),
        ]
        for clock_seconds, color, linestyle in events:
            first_h = (clock_seconds - float(start_clock_seconds)) / 3600.0
            while first_h < 0:
                first_h += 24.0
            for time_h_value in np.arange(first_h, float(time_h[-1]) + 1e-9, 24.0):
                ax.axvline(time_h_value, color=color, linestyle=linestyle, lw=1, alpha=0.75)

    if start_clock_seconds is None:
        ax.set_xlabel("Elapsed time (h)")
    else:
        ax.set_xlabel(f"Elapsed time from {format_clock_time(start_clock_seconds)} (h)")
    ax.set_ylabel("Occupancy cluster")
    ax.set_title(title or f"Quiet periods, speed <= {speed_threshold_mm_s:g} mm/s")
    fig.colorbar(im, ax=ax, label="Fraction quiet within bin")
    fig.tight_layout()
    plt.show()
    return image, image_tracks, time_h


def compute_cluster_speed_timeseries(
    cluster_table: pd.DataFrame,
    speed_root: Path,
    *,
    bin_seconds: float = 10 * 60.0,
    cluster_col: str = "leiden_cluster",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cluster_col not in cluster_table.columns:
        raise ValueError(f"cluster_table is missing {cluster_col!r}")

    speed_tracks = load_speed_tracks(speed_root)
    speed_cols = [
        "track_name",
        "speed_path",
        "frame_min",
        "frame_max",
        "n_frames",
        "fps",
    ]
    speed_for_merge = speed_tracks[speed_cols].rename(
        columns={
            "frame_min": "speed_vector_frame_min",
            "frame_max": "speed_vector_frame_max",
            "n_frames": "speed_vector_n_frames",
            "fps": "speed_vector_fps",
        }
    )
    merged = cluster_table.merge(
        speed_for_merge,
        on="track_name",
        how="left",
        validate="one_to_one",
    )
    missing = merged["speed_path"].isna() | ~merged["speed_path"].map(lambda path: Path(path).exists() if pd.notna(path) else False)
    if missing.any():
        examples = merged.loc[missing, "track_name"].head(10).to_list()
        raise FileNotFoundError(f"Missing speed vectors for {int(missing.sum())} clustered tracks, examples: {examples}")

    # binned_track_speed uses the speed vector's own frame span, not the
    # occupancy metadata columns that may already exist in cluster_table.
    merged["frame_min"] = merged["speed_vector_frame_min"].astype("int64")
    merged["frame_max"] = merged["speed_vector_frame_max"].astype("int64")
    merged["n_frames"] = merged["speed_vector_n_frames"].astype("int64")
    merged["fps"] = merged["speed_vector_fps"].astype("float64")

    track_bins = []
    for i, row in merged.iterrows():
        if i == 0 or i == len(merged) - 1 or (i + 1) % 25 == 0:
            print(f"speed: binning {i + 1}/{len(merged)} {row['track_name']}")
        binned = binned_track_speed(row, bin_seconds)
        binned[cluster_col] = row[cluster_col]
        track_bins.append(binned)

    if not track_bins:
        raise ValueError("No clustered tracks to compute speed for")

    track_speed = pd.concat(track_bins, ignore_index=True)
    cluster_speed = (
        track_speed.groupby([cluster_col, "time_h"], as_index=False)
        .agg(
            mean_speed_mm_s=("speed_mm_s", "mean"),
            n_speed_tracks=("speed_mm_s", "count"),
        )
        .sort_values([cluster_col, "time_h"])
        .reset_index(drop=True)
    )
    return cluster_speed, track_speed


def smooth_cluster_speed_timeseries(
    cluster_speed: pd.DataFrame,
    *,
    smooth_seconds: float,
    bin_seconds: float,
    cluster_col: str = "leiden_cluster",
) -> pd.DataFrame:
    out = cluster_speed.copy()
    window_bins = max(1, int(round(float(smooth_seconds) / float(bin_seconds))))
    out["smoothed_speed_mm_s"] = np.nan
    for cluster, idx in out.groupby(cluster_col).groups.items():
        group = out.loc[idx].sort_values("time_h")
        smoothed = rolling_nanmean(group["mean_speed_mm_s"].to_numpy(np.float32), window_bins)
        out.loc[group.index, "smoothed_speed_mm_s"] = smoothed
    return out.sort_values([cluster_col, "time_h"]).reset_index(drop=True)


def plot_cluster_speed_timeseries(
    cluster_table: pd.DataFrame,
    speed_root: Path,
    *,
    bin_seconds: float = 10 * 60.0,
    smooth_seconds: float = 60 * 60.0,
    cluster_col: str = "leiden_cluster",
    start_clock_seconds: int | None = None,
    light_off_hour: float = 18.0,
    light_on_hour: float = 6.0,
    shade_dark: bool = True,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    cmap: str = "tab10",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_speed, track_speed = compute_cluster_speed_timeseries(
        cluster_table,
        speed_root,
        bin_seconds=bin_seconds,
        cluster_col=cluster_col,
    )
    plot_df = smooth_cluster_speed_timeseries(
        cluster_speed,
        smooth_seconds=smooth_seconds,
        bin_seconds=bin_seconds,
        cluster_col=cluster_col,
    )
    y_col = "smoothed_speed_mm_s" if smooth_seconds > 0 else "mean_speed_mm_s"

    clusters = sorted(plot_df[cluster_col].dropna().unique())
    colors = plt.get_cmap(cmap, max(len(clusters), 1))
    fig, ax = plt.subplots(figsize=(12, 5))
    if start_clock_seconds is None and not cluster_table.empty:
        start_clock_seconds = start_time_from_track_table(cluster_table)
    if shade_dark and start_clock_seconds is not None and not plot_df.empty:
        add_light_shading(
            ax,
            start_clock_seconds,
            float(plot_df["time_h"].max()),
            light_off_hour=light_off_hour,
            light_on_hour=light_on_hour,
        )

    for i, cluster in enumerate(clusters):
        group = plot_df[plot_df[cluster_col] == cluster]
        n_tracks = int(cluster_table[cluster_table[cluster_col] == cluster]["track_name"].nunique())
        ax.plot(
            group["time_h"],
            group[y_col],
            lw=1.8,
            color=colors(i),
            label=f"cluster {cluster} n={n_tracks}",
        )

    if start_clock_seconds is None:
        ax.set_xlabel("Elapsed time (h)")
    else:
        ax.set_xlabel(f"Elapsed time from {format_clock_time(start_clock_seconds)} (h)")
    ax.set_ylabel("Mean speed (mm/s)")
    ax.set_title(title or f"Speed by {cluster_col}, {smooth_seconds / 60:g} min smoothing")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()
    return plot_df, track_speed
