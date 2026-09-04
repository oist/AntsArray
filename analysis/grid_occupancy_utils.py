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
        track_path_raw = meta.get("track_path")
        rows.append(
            {
                "track_name": track_name,
                "track_id": meta.get("track_id", track_id_from_name(track_name)),
                "side": side,
                "track_path": Path(track_path_raw) if track_path_raw else None,
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
                "input_x_is_side_local": bool(meta.get("input_x_is_side_local", False)),
                "input_x_origin_px": float(meta.get("input_x_origin_px", 0.0)),
                "side_x0_px": float(meta.get("side_x0_px", meta.get("input_x_origin_px", 0.0))),
                "y_origin_px": float(meta.get("y_origin_px", 0.0)),
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


def compute_cluster_quiet_timeseries(
    quiet_image: np.ndarray,
    quiet_tracks: pd.DataFrame,
    time_h: np.ndarray,
    *,
    cluster_col: str = "leiden_cluster",
) -> pd.DataFrame:
    required = {"side", "track_name", cluster_col}
    missing = required.difference(quiet_tracks.columns)
    if missing:
        raise ValueError(f"quiet_tracks is missing columns: {sorted(missing)}")

    image = np.asarray(quiet_image, dtype=np.float32)
    times = np.asarray(time_h, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("quiet_image must be a 2D ant x time array")
    if len(times) != image.shape[1]:
        raise ValueError("time_h length must match quiet_image columns")

    tracks = quiet_tracks.reset_index(drop=True).copy()
    if "image_row" in tracks.columns:
        image_rows = tracks["image_row"].to_numpy(dtype=int)
        if len(np.unique(image_rows)) != len(image_rows):
            raise ValueError("quiet_tracks image_row values must be unique")
        if len(image_rows) and (image_rows.min() < 0 or image_rows.max() >= image.shape[0]):
            raise ValueError("quiet_tracks image_row values are outside quiet_image rows")
    else:
        if len(tracks) != image.shape[0]:
            raise ValueError("quiet_tracks rows must match quiet_image rows when image_row is absent")
        image_rows = np.arange(len(tracks), dtype=int)
    tracks["_quiet_image_row"] = image_rows

    rows = []
    group_cols = ["side", cluster_col]
    for (side, cluster), group in tracks.dropna(subset=[cluster_col]).groupby(group_cols, sort=True):
        row_idx = group["_quiet_image_row"].to_numpy(dtype=int)
        values = image[row_idx, :]
        finite = np.isfinite(values)
        n_ants_with_data = finite.sum(axis=0)
        quiet_sum = np.where(finite, values, 0.0).sum(axis=0)
        mean_quiet = np.full(image.shape[1], np.nan, dtype=np.float32)
        keep = n_ants_with_data > 0
        mean_quiet[keep] = quiet_sum[keep] / n_ants_with_data[keep]
        n_ants_total = int(group["track_name"].nunique())

        rows.extend(
            {
                "side": side,
                cluster_col: cluster,
                "cluster_id": f"{side}_{cluster}",
                "time_h": float(time_value),
                "mean_quiet_fraction": float(mean_value) if np.isfinite(mean_value) else np.nan,
                "n_ants_with_data": int(n_with_data),
                "n_ants_total": n_ants_total,
            }
            for time_value, mean_value, n_with_data in zip(times, mean_quiet, n_ants_with_data)
        )

    if not rows:
        raise ValueError("No clusters to summarize")
    return pd.DataFrame(rows).sort_values(["side", cluster_col, "time_h"], kind="mergesort").reset_index(drop=True)


def plot_cluster_quiet_timeseries(
    quiet_image: np.ndarray,
    quiet_tracks: pd.DataFrame,
    time_h: np.ndarray,
    *,
    cluster_col: str = "leiden_cluster",
    start_clock_seconds: int | None = None,
    light_off_hour: float = 18.0,
    light_on_hour: float = 6.0,
    shade_dark: bool = True,
    smooth_seconds: float = 0.0,
    bin_seconds: float | None = None,
    ylim: tuple[float, float] | None = (0.0, 1.0),
    title: str | None = None,
    cmap: str = "tab10",
) -> pd.DataFrame:
    plot_df = compute_cluster_quiet_timeseries(
        quiet_image,
        quiet_tracks,
        time_h,
        cluster_col=cluster_col,
    )
    y_col = "mean_quiet_fraction"
    if smooth_seconds > 0:
        if bin_seconds is None:
            times = np.asarray(time_h, dtype=np.float64)
            if len(times) < 2:
                raise ValueError("Need at least two time bins to infer bin_seconds for smoothing")
            bin_seconds = float(np.nanmedian(np.diff(times))) * 3600.0
        window_bins = max(1, int(round(float(smooth_seconds) / float(bin_seconds))))
        plot_df["smoothed_mean_quiet_fraction"] = np.nan
        for (_side, _cluster), idx in plot_df.groupby(["side", cluster_col]).groups.items():
            group = plot_df.loc[idx].sort_values("time_h")
            smoothed = rolling_nanmean(group["mean_quiet_fraction"].to_numpy(np.float32), window_bins)
            plot_df.loc[group.index, "smoothed_mean_quiet_fraction"] = smoothed
        y_col = "smoothed_mean_quiet_fraction"

    sides = [side for side in ("left", "right") if side in set(plot_df["side"])]
    sides.extend(sorted(side for side in plot_df["side"].dropna().unique() if side not in set(sides)))
    fig, axes = plt.subplots(
        len(sides),
        1,
        figsize=(12, max(4.0, 3.2 * len(sides))),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    max_time_h = float(np.nanmax(np.asarray(time_h, dtype=np.float64)))
    min_time_h = float(np.nanmin(np.asarray(time_h, dtype=np.float64)))
    if start_clock_seconds is None and not quiet_tracks.empty:
        try:
            start_clock_seconds = start_time_from_track_table(quiet_tracks)
        except ValueError:
            start_clock_seconds = None

    for ax, side in zip(axes.ravel(), sides):
        side_df = plot_df[plot_df["side"] == side]
        clusters = sorted(side_df[cluster_col].dropna().unique())
        colors = plt.get_cmap(cmap, max(len(clusters), 1))
        if shade_dark and start_clock_seconds is not None:
            add_light_shading(
                ax,
                start_clock_seconds,
                max_time_h,
                min_time_h=min_time_h,
                light_off_hour=light_off_hour,
                light_on_hour=light_on_hour,
            )

        for i, cluster in enumerate(clusters):
            group = side_df[side_df[cluster_col] == cluster].sort_values("time_h")
            n_ants = int(group["n_ants_total"].max())
            ax.plot(
                group["time_h"],
                group[y_col],
                lw=1.8,
                color=colors(i),
                label=f"cluster {cluster} n={n_ants}",
            )

        ax.set_ylabel("Mean quiet fraction")
        ax.set_title(f"{side} colony")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    if start_clock_seconds is None:
        axes[-1, 0].set_xlabel("Elapsed time (h)")
    else:
        axes[-1, 0].set_xlabel(f"Elapsed time from {format_clock_time(start_clock_seconds)} (h)")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    return plot_df


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


def _split_panorama_region_label(label: str) -> tuple[str | None, str]:
    """Split a semantic label into an optional side and a region type.

    A bare one-letter suffix is accepted only when it is uppercase. This is
    deliberate: treating the final lowercase ``r`` in ``water`` as ``right``
    silently assigned every unsuffixed water annotation to the right colony.
    Labels without a side are assigned from tracking geometry by
    :func:`load_panorama_regions`.
    """
    raw = str(label).strip()
    match = re.fullmatch(
        r"(?P<region_type>.+?)(?:[_\-\s]+)(?P<side>left|right|l|r)",
        raw,
        flags=re.IGNORECASE,
    )
    if match is None:
        match = re.fullmatch(r"(?P<region_type>.+?)(?P<side>left|right)", raw, flags=re.IGNORECASE)
    if match is None:
        match = re.fullmatch(r"(?P<region_type>.+?)(?P<side>[LR])", raw)

    if match is None:
        region_type = re.sub(r"[_\-\s]+$", "", raw).lower()
        if not region_type:
            raise ValueError(f"Panorama region label {label!r} has no semantic region type")
        return None, region_type

    suffix = match.group("side").lower()
    side = "left" if suffix in {"l", "left"} else "right"
    region_type = re.sub(r"[_\-\s]+$", "", match.group("region_type")).lower()
    if not region_type:
        raise ValueError(f"Panorama region label {label!r} has no semantic region type")
    return side, region_type


def load_panorama_regions(
    path: Path,
    *,
    x_split_px: float | None = None,
    validate_side_geometry: bool = True,
) -> pd.DataFrame:
    """Load panorama annotations expressed in raw tracking-pixel coordinates.

    Explicit label suffixes (``_L``, ``_R``, ``left``, or ``right``) take
    precedence. Unsuffixed labels such as ``water`` require ``x_split_px`` and
    are assigned to a colony from the region center in tracking coordinates.
    When a split is supplied, explicit labels are checked against geometry so
    a swapped annotation fails loudly.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Panorama region CSV does not exist: {path}")

    regions = pd.read_csv(path)
    required = {
        "region_id",
        "semantic_label",
        "name",
        "shape",
        "tracking_x_min_px",
        "tracking_y_min_px",
        "tracking_x_max_px",
        "tracking_y_max_px",
        "tracking_center_x_px",
        "tracking_center_y_px",
        "radius_px",
        "area_mm2",
        "mm_per_pixel",
    }
    missing = required.difference(regions.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if regions.empty:
        raise ValueError(f"{path.name} contains no regions")
    if regions["region_id"].duplicated().any() or regions["name"].duplicated().any():
        raise ValueError(f"{path.name} region_id and name values must be unique")

    regions = regions.copy()
    regions["shape"] = regions["shape"].astype(str).str.lower()
    bad_shapes = sorted(set(regions["shape"]).difference({"circle", "rectangle"}))
    if bad_shapes:
        raise ValueError(f"Unsupported panorama region shapes: {bad_shapes}")

    numeric_cols = [
        "tracking_x_min_px",
        "tracking_y_min_px",
        "tracking_x_max_px",
        "tracking_y_max_px",
        "tracking_center_x_px",
        "tracking_center_y_px",
        "radius_px",
        "area_mm2",
        "mm_per_pixel",
    ]
    for col in numeric_cols:
        regions[col] = pd.to_numeric(regions[col], errors="coerce")

    parsed = regions["semantic_label"].map(_split_panorama_region_label)
    regions["label_side"] = parsed.map(lambda value: value[0])
    regions["region_type"] = parsed.map(lambda value: value[1])

    center_x = regions["tracking_center_x_px"].copy()
    rectangle_center_x = (
        regions["tracking_x_min_px"] + regions["tracking_x_max_px"]
    ) / 2.0
    center_x = center_x.where(center_x.notna(), rectangle_center_x)
    if center_x.isna().any():
        names = regions.loc[center_x.isna(), "name"].astype(str).to_list()
        raise ValueError(f"Could not determine tracking x center for regions: {names}")

    if x_split_px is None:
        unsuffixed = regions["label_side"].isna()
        if unsuffixed.any():
            labels = regions.loc[unsuffixed, "semantic_label"].astype(str).unique().tolist()
            raise ValueError(
                "Panorama labels without an explicit colony side require x_split_px; "
                f"unsuffixed labels: {labels}"
            )
        regions["geometry_side"] = pd.NA
    else:
        split = float(x_split_px)
        regions["geometry_side"] = np.where(center_x.to_numpy(float) < split, "left", "right")
        mismatch = regions["label_side"].notna() & regions["label_side"].ne(regions["geometry_side"])
        if bool(validate_side_geometry) and mismatch.any():
            details = regions.loc[
                mismatch,
                ["name", "semantic_label", "label_side", "geometry_side"],
            ].to_dict("records")
            raise ValueError(f"Panorama region label/geometry side mismatch: {details}")

    regions["side"] = regions["label_side"].where(
        regions["label_side"].notna(),
        regions["geometry_side"],
    )
    regions["side_source"] = np.where(
        regions["label_side"].notna(),
        "semantic_label",
        "tracking_geometry",
    )

    rectangle_missing = regions["shape"].eq("rectangle") & regions[
        ["tracking_x_min_px", "tracking_y_min_px", "tracking_x_max_px", "tracking_y_max_px"]
    ].isna().any(axis=1)
    circle_missing = regions["shape"].eq("circle") & regions[
        ["tracking_center_x_px", "tracking_center_y_px", "radius_px"]
    ].isna().any(axis=1)
    invalid = rectangle_missing | circle_missing | regions["area_mm2"].isna() | regions["mm_per_pixel"].isna()
    if invalid.any():
        raise ValueError(
            f"{path.name} has incomplete geometry for regions: "
            f"{regions.loc[invalid, 'name'].astype(str).to_list()}"
        )

    return regions.sort_values(["side", "region_type", "name"], kind="mergesort").reset_index(drop=True)


def _region_geometry_in_grid_mm(region: pd.Series, track: pd.Series) -> dict[str, float]:
    required = {"mm_per_px", "input_x_is_side_local", "input_x_origin_px", "side_x0_px", "y_origin_px"}
    missing = required.difference(track.index)
    if missing:
        raise ValueError(f"Grid track metadata is missing coordinate fields: {sorted(missing)}")

    scale = float(track["mm_per_px"])
    annotation_scale = float(region["mm_per_pixel"])
    if not np.isclose(scale, annotation_scale, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Grid scale {scale:g} mm/px does not match panorama annotation scale "
            f"{annotation_scale:g} mm/px"
        )

    # Panorama annotations always use raw/global tracking pixels. A grid made
    # from already side-local x input still needs the original side origin
    # removed from annotation x coordinates.
    x_origin_px = (
        float(track["side_x0_px"])
        if bool(track["input_x_is_side_local"])
        else float(track["input_x_origin_px"])
    )
    y_origin_px = float(track["y_origin_px"])

    if region["shape"] == "rectangle":
        return {
            "x_min": (float(region["tracking_x_min_px"]) - x_origin_px) * scale,
            "y_min": (float(region["tracking_y_min_px"]) - y_origin_px) * scale,
            "x_max": (float(region["tracking_x_max_px"]) - x_origin_px) * scale,
            "y_max": (float(region["tracking_y_max_px"]) - y_origin_px) * scale,
        }
    return {
        "center_x": (float(region["tracking_center_x_px"]) - x_origin_px) * scale,
        "center_y": (float(region["tracking_center_y_px"]) - y_origin_px) * scale,
        "radius": float(region["radius_px"]) * scale,
    }


def _region_mask_for_grid(
    region: pd.Series,
    track: pd.Series,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
) -> np.ndarray:
    geometry = _region_geometry_in_grid_mm(region, track)
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    if region["shape"] == "rectangle":
        return (
            (x_grid >= geometry["x_min"])
            & (x_grid <= geometry["x_max"])
            & (y_grid >= geometry["y_min"])
            & (y_grid <= geometry["y_max"])
        )
    return (
        (x_grid - geometry["center_x"]) ** 2 + (y_grid - geometry["center_y"]) ** 2
        <= geometry["radius"] ** 2
    )


def compute_region_occupancy(
    cluster_table: pd.DataFrame,
    regions: pd.DataFrame,
    *,
    cluster_col: str = "leiden_cluster",
) -> pd.DataFrame:
    """Measure annotated-region use from normalized per-ant occupancy maps.

    The resulting fractions quantify detected time in each region. They are a
    spatial proxy for region use, not counts of temporally distinct visits.
    """
    required_tracks = {
        "track_name",
        "track_id",
        "side",
        "occupancy_path",
        "x_edges_path",
        "y_edges_path",
        cluster_col,
    }
    missing = required_tracks.difference(cluster_table.columns)
    if missing:
        raise ValueError(f"cluster_table is missing columns: {sorted(missing)}")
    if regions.empty:
        raise ValueError("regions is empty")

    rows: list[dict[str, object]] = []
    for i, (_, track) in enumerate(cluster_table.iterrows(), start=1):
        if i == 1 or i == len(cluster_table) or i % 25 == 0:
            print(f"regions: measuring {i}/{len(cluster_table)} {track['track_name']}")
        side_regions = regions[regions["side"] == track["side"]]
        if side_regions.empty:
            raise ValueError(f"No panorama regions are annotated for side={track['side']!r}")

        histogram, x_edges, y_edges = load_histogram(track)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
        if histogram.shape != (len(y_centers), len(x_centers)):
            raise ValueError(f"Histogram and grid-edge shapes disagree for {track['track_name']}")
        cell_areas = np.diff(y_edges)[:, None] * np.diff(x_edges)[None, :]
        histogram_coverage = float(np.nansum(histogram, dtype=np.float64))
        cluster_id = (
            str(track["cluster_id"])
            if "cluster_id" in track.index
            else f"{track['side']}_{track[cluster_col]}"
        )

        for _, region in side_regions.iterrows():
            mask = _region_mask_for_grid(region, track, x_centers, y_centers)
            occupancy_fraction = float(np.nansum(histogram[mask], dtype=np.float64))
            rows.append(
                {
                    "side": track["side"],
                    "track_id": track["track_id"],
                    "track_name": track["track_name"],
                    cluster_col: track[cluster_col],
                    "cluster_id": cluster_id,
                    "region_id": region["region_id"],
                    "semantic_label": region["semantic_label"],
                    "region_name": region["name"],
                    "region_type": region["region_type"],
                    "shape": region["shape"],
                    "annotated_area_mm2": float(region["area_mm2"]),
                    "grid_mask_area_mm2": float(np.sum(cell_areas[mask], dtype=np.float64)),
                    "n_grid_cells": int(mask.sum()),
                    "histogram_coverage": histogram_coverage,
                    "occupancy_fraction": occupancy_fraction,
                    "occupancy_percent": 100.0 * occupancy_fraction,
                }
            )

    if not rows:
        raise ValueError("No track/region occupancy measurements were produced")
    return pd.DataFrame(rows).sort_values(
        ["side", "cluster_id", "track_id", "region_type", "region_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_colony_use(
    region_occupancy: pd.DataFrame,
    *,
    cluster_col: str = "leiden_cluster",
) -> pd.DataFrame:
    """Return one inside/outside-colony row per ant."""
    keys = ["side", cluster_col, "cluster_id", "track_id", "track_name"]
    base = region_occupancy[keys + ["histogram_coverage"]].drop_duplicates(keys)
    colony = (
        region_occupancy[region_occupancy["region_type"] == "colony"]
        .groupby(keys, as_index=False, dropna=False)
        .agg(inside_colony_fraction=("occupancy_fraction", "sum"))
    )
    out = base.merge(colony, on=keys, how="left", validate="one_to_one")
    if out["inside_colony_fraction"].isna().any():
        sides = sorted(out.loc[out["inside_colony_fraction"].isna(), "side"].unique())
        raise ValueError(f"Missing colony annotations for sides: {sides}")
    out["inside_colony_fraction"] = np.minimum(
        out["inside_colony_fraction"].to_numpy(float),
        out["histogram_coverage"].to_numpy(float),
    )
    out["outside_colony_fraction"] = np.clip(
        out["histogram_coverage"] - out["inside_colony_fraction"],
        0.0,
        None,
    )
    out["inside_colony_percent"] = 100.0 * out["inside_colony_fraction"]
    out["outside_colony_percent"] = 100.0 * out["outside_colony_fraction"]
    return out.sort_values(["side", "cluster_id", "track_id"], kind="mergesort").reset_index(drop=True)


def summarize_cluster_colony_use(
    colony_use: pd.DataFrame,
    *,
    restricted_threshold: float = 0.90,
) -> pd.DataFrame:
    summary = (
        colony_use.groupby(["side", "cluster_id"], as_index=False)
        .agg(
            n_ants=("track_name", "nunique"),
            median_inside_colony_fraction=("inside_colony_fraction", "median"),
            mean_inside_colony_fraction=("inside_colony_fraction", "mean"),
            min_inside_colony_fraction=("inside_colony_fraction", "min"),
            max_inside_colony_fraction=("inside_colony_fraction", "max"),
            median_outside_colony_fraction=("outside_colony_fraction", "median"),
        )
        .sort_values(["side", "cluster_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    summary["putative_role"] = np.where(
        summary["median_inside_colony_fraction"] >= float(restricted_threshold),
        "colony-restricted",
        "in/out",
    )
    return summary


def select_putative_roaming_clusters(cluster_colony_use: pd.DataFrame) -> tuple[str, ...]:
    """Select every cluster classified as in/out.

    Older summary tables may not have ``putative_role``; for those, retain the
    legacy fallback of selecting the greatest outside-colony cluster per side.
    """
    required = {"side", "cluster_id", "median_outside_colony_fraction"}
    missing = required.difference(cluster_colony_use.columns)
    if missing:
        raise ValueError(f"cluster_colony_use is missing columns: {sorted(missing)}")
    if "putative_role" in cluster_colony_use.columns:
        selected_rows = cluster_colony_use[cluster_colony_use["putative_role"].eq("in/out")]
        if not selected_rows.empty:
            return tuple(
                selected_rows.sort_values(["side", "cluster_id"], kind="mergesort")["cluster_id"]
                .astype(str)
                .to_list()
            )

    selected = []
    for _, side_summary in cluster_colony_use.groupby("side", sort=True):
        row = side_summary.sort_values(
            ["median_outside_colony_fraction", "cluster_id"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        selected.append(str(row["cluster_id"]))
    return tuple(selected)


def plot_cluster_colony_use(
    colony_use: pd.DataFrame,
    *,
    restricted_threshold: float = 0.90,
) -> pd.DataFrame:
    """Plot per-ant inside-colony fractions and return cluster summaries."""
    summary = summarize_cluster_colony_use(
        colony_use,
        restricted_threshold=restricted_threshold,
    )
    sides = [side for side in ("left", "right") if side in set(colony_use["side"])]
    sides.extend(sorted(side for side in colony_use["side"].unique() if side not in set(sides)))
    fig, axes = plt.subplots(1, len(sides), figsize=(max(6.0, 5.2 * len(sides)), 5.2), squeeze=False)

    for ax, side in zip(axes.ravel(), sides):
        side_use = colony_use[colony_use["side"] == side]
        side_summary = summary[summary["side"] == side]
        clusters = side_summary["cluster_id"].to_list()
        values = [
            100.0 * side_use.loc[side_use["cluster_id"] == cluster_id, "inside_colony_fraction"].to_numpy(float)
            for cluster_id in clusters
        ]
        boxes = ax.boxplot(values, positions=np.arange(len(clusters)), widths=0.5, patch_artist=True)
        colors = plt.get_cmap("tab10", max(len(clusters), 1))
        for index, (box, cluster_values) in enumerate(zip(boxes["boxes"], values)):
            box.set_facecolor(colors(index))
            box.set_alpha(0.25)
            jitter = np.linspace(-0.15, 0.15, max(len(cluster_values), 1))[: len(cluster_values)]
            ax.scatter(
                np.full(len(cluster_values), index) + jitter,
                cluster_values,
                s=27,
                color=colors(index),
                edgecolor="white",
                linewidth=0.4,
                alpha=0.9,
                zorder=3,
            )
        labels = []
        for row in side_summary.itertuples(index=False):
            labels.append(
                f"{row.cluster_id}\n{row.putative_role}\nmedian {100 * row.median_inside_colony_fraction:.1f}%"
            )
        ax.set_xticks(np.arange(len(clusters)), labels)
        ax.axhline(
            100.0 * float(restricted_threshold),
            color="0.35",
            linestyle="--",
            lw=1,
            label=f"putative restriction threshold ({100 * restricted_threshold:g}%)",
        )
        ax.set_ylim(-2, 102)
        ax.set_ylabel("Detected time inside colony annotation (%)")
        ax.set_title(f"{side} colony: spatial-cluster use")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("Putative colony-restricted versus in/out job clusters")
    fig.tight_layout()
    plt.show()
    return summary


def plot_ant_inside_outside_colony_distribution(colony_use: pd.DataFrame) -> None:
    """Show every ant's observed inside/outside composition, sorted within side."""
    sides = [side for side in ("left", "right") if side in set(colony_use["side"].astype(str))]
    fig, axes = plt.subplots(1, len(sides), figsize=(7.2 * len(sides), 10.0), squeeze=False, sharex=True)
    for column, side in enumerate(sides):
        ax = axes[0, column]
        ants = colony_use[colony_use["side"].astype(str) == side].copy()
        observed = ants["inside_colony_fraction"] + ants["outside_colony_fraction"]
        ants["inside_percent_of_observed"] = (
            100.0 * ants["inside_colony_fraction"] / observed.replace(0, np.nan)
        )
        ants["outside_percent_of_observed"] = 100.0 - ants["inside_percent_of_observed"]
        ants = ants.sort_values(
            ["inside_percent_of_observed", "track_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        y = np.arange(len(ants))
        ax.barh(
            y,
            ants["inside_percent_of_observed"],
            height=0.84,
            color="#4c78a8",
            label="inside colony",
        )
        ax.barh(
            y,
            ants["outside_percent_of_observed"],
            left=ants["inside_percent_of_observed"],
            height=0.84,
            color="#f58518",
            label="outside colony",
        )
        ax.axvline(50, color="white", ls="--", lw=0.8, alpha=0.9)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percent of observed time")
        ax.set_yticks(y, ants["track_id"].astype(int), fontsize=5)
        ax.set_ylabel("TrackID, sorted by time inside")
        ax.invert_yaxis()
        median_inside = float(ants["inside_percent_of_observed"].median())
        ax.set_title(f"{side} colony ({len(ants)} ants)\nmedian inside = {median_inside:.1f}%")
        ax.grid(True, axis="x", alpha=0.2)
        if column == 0:
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Distribution of observed time inside versus outside the colony across ants")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.07, wspace=0.18)
    plt.show()


def plot_colony_use_vs_trip_investment(
    colony_use: pd.DataFrame,
    trip_investment: pd.DataFrame,
) -> pd.DataFrame:
    """Relate per-ant outside-colony time to trip frequency and duration."""
    from scipy.stats import pearsonr, spearmanr

    use_columns = [
        "side",
        "track_id",
        "track_name",
        "inside_colony_fraction",
        "outside_colony_fraction",
    ]
    merged = trip_investment.merge(
        colony_use[use_columns],
        on=["side", "track_id", "track_name"],
        how="inner",
        validate="one_to_one",
    )
    observed = merged["inside_colony_fraction"] + merged["outside_colony_fraction"]
    merged["outside_colony_percent"] = 100.0 * merged["outside_colony_fraction"] / observed.replace(0, np.nan)
    metric_settings = (
        (
            "completed_trips_per_day",
            "completed_trips_per_day_ci_low",
            "completed_trips_per_day_ci_high",
            "Completed trips / recording day",
            False,
        ),
        (
            "median_trip_minutes",
            "median_trip_minutes_ci_low",
            "median_trip_minutes_ci_high",
            "Median time per trip (min)",
            True,
        ),
    )
    sides = [side for side in ("left", "right") if side in set(merged["side"].astype(str))]
    fig, axes = plt.subplots(len(sides), 2, figsize=(13.5, 5.2 * len(sides)), squeeze=False)
    correlation_rows: list[dict[str, object]] = []
    for row, side in enumerate(sides):
        ants = merged[merged["side"].astype(str) == side].sort_values("track_id")
        for column, (metric, low_column, high_column, x_label, log_x) in enumerate(metric_settings):
            ax = axes[row, column]
            x = ants[metric].to_numpy(float)
            y = ants["outside_colony_percent"].to_numpy(float)
            xerr = np.vstack(
                [
                    x - ants[low_column].to_numpy(float),
                    ants[high_column].to_numpy(float) - x,
                ]
            )
            color = "tab:blue" if side == "left" else "tab:orange"
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                fmt="o",
                ms=5,
                color=color,
                ecolor="0.6",
                elinewidth=0.8,
                capsize=1.8,
                alpha=0.78,
            )
            for ant in ants.itertuples(index=False):
                ax.annotate(
                    str(int(ant.track_id)),
                    (getattr(ant, metric), ant.outside_colony_percent),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize=6,
                )
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            correlation_rows.append(
                {
                    "side": side,
                    "metric": metric,
                    "n_ants": len(ants),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                }
            )
            if log_x:
                ax.set_xscale("log")
                transformed_x = np.log10(x)
                line_x = np.geomspace(float(x.min()), float(x.max()), 200)
                slope, intercept = np.polyfit(transformed_x, y, 1)
                line_y = intercept + slope * np.log10(line_x)
            else:
                line_x = np.linspace(float(x.min()), float(x.max()), 200)
                slope, intercept = np.polyfit(x, y, 1)
                line_y = intercept + slope * line_x
            ax.plot(line_x, line_y, color=color, lw=1.5)
            ax.set_ylim(-2, 102)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Observed time outside colony (%)")
            ax.set_title(
                f"{side}: outside time versus "
                f"{'trip frequency' if column == 0 else 'trip duration'}\n"
                f"Pearson r={pearson.statistic:.2f}; Spearman ρ={spearman.statistic:.2f}"
            )
            ax.grid(True, which="both", alpha=0.2)
    fig.suptitle(
        "Does inside/outside colony use predict trip investment? Horizontal bars are 95% confidence intervals"
    )
    fig.tight_layout()
    plt.show()
    return pd.DataFrame(correlation_rows)


def plot_cluster_region_overlays(
    cluster_table: pd.DataFrame,
    regions: pd.DataFrame,
    *,
    cluster_col: str = "leiden_cluster",
    mode: str = "sqrt",
    vmax_percentile: float | None = 99.0,
    cmap: str = "viridis",
) -> dict[object, np.ndarray]:
    """Plot mean cluster occupancy with panorama-region outlines."""
    from matplotlib.patches import Circle, Patch, Rectangle

    sides = cluster_table["side"].dropna().unique()
    if len(sides) != 1:
        raise ValueError("plot_cluster_region_overlays expects one colony side at a time")
    side = str(sides[0])
    side_regions = regions[regions["side"] == side]
    if side_regions.empty:
        raise ValueError(f"No panorama regions are annotated for side={side!r}")

    clusters = sorted(cluster_table[cluster_col].dropna().unique())
    means: dict[object, np.ndarray] = {}
    display_means = []
    first_track = cluster_table.iloc[0]
    _, x_edges, y_edges = load_histogram(first_track)
    for cluster in clusters:
        maps = [
            load_histogram(row)[0]
            for _, row in cluster_table[cluster_table[cluster_col] == cluster].iterrows()
        ]
        mean_map = np.nanmean(np.stack(maps), axis=0)
        means[cluster] = mean_map
        display_means.append(display_histogram(mean_map, mode))

    vmax = None
    if vmax_percentile is not None:
        vmax = float(np.nanpercentile(np.concatenate([values.ravel() for values in display_means]), vmax_percentile))
    colors = {"colony": "cyan", "food": "tomato", "water": "deepskyblue"}
    fig, axes = plt.subplots(1, len(clusters), figsize=(5.1 * len(clusters), 5.0), squeeze=False)
    last_image = None
    for ax, cluster, image in zip(axes.ravel(), clusters, display_means):
        last_image = ax.imshow(
            image,
            origin="lower",
            aspect="equal",
            interpolation="none",
            extent=[float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])],
            vmin=0,
            vmax=vmax,
            cmap=cmap,
        )
        for _, region in side_regions.iterrows():
            geometry = _region_geometry_in_grid_mm(region, first_track)
            color = colors.get(str(region["region_type"]), "white")
            if region["shape"] == "rectangle":
                patch = Rectangle(
                    (geometry["x_min"], geometry["y_min"]),
                    geometry["x_max"] - geometry["x_min"],
                    geometry["y_max"] - geometry["y_min"],
                    fill=False,
                    edgecolor=color,
                    lw=1.6,
                )
                label_x = (geometry["x_min"] + geometry["x_max"]) / 2.0
                label_y = geometry["y_max"]
            else:
                patch = Circle(
                    (geometry["center_x"], geometry["center_y"]),
                    geometry["radius"],
                    fill=False,
                    edgecolor=color,
                    lw=1.6,
                )
                label_x = geometry["center_x"]
                label_y = geometry["center_y"] + geometry["radius"]
            ax.add_patch(patch)
            ax.text(label_x, label_y, str(region["name"]), color=color, fontsize=7, ha="center", va="bottom")
        n_ants = int((cluster_table[cluster_col] == cluster).sum())
        ax.set_title(f"{side} cluster {cluster}, n={n_ants}")
        ax.set_xlabel("x within side (mm)")
        ax.set_ylabel("y (mm)")

    legend_handles = [
        Patch(facecolor="none", edgecolor=color, label=region_type)
        for region_type, color in colors.items()
        if region_type in set(side_regions["region_type"])
    ]
    if legend_handles:
        axes[0, 0].legend(handles=legend_handles, fontsize=8, loc="upper right")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.75, label=f"mean occupancy ({mode})")
    fig.suptitle(f"{side} spatial job clusters with panorama annotations")
    plt.show()
    return means


def resource_use_table(
    region_occupancy: pd.DataFrame,
    cluster_ids: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    chosen = region_occupancy[
        region_occupancy["cluster_id"].astype(str).isin([str(value) for value in cluster_ids])
        & region_occupancy["region_type"].ne("colony")
    ].copy()
    if chosen.empty:
        raise ValueError(f"No non-colony region use found for clusters: {list(cluster_ids)}")
    keys = ["side", "cluster_id", "track_id", "track_name", "region_type", "region_name"]
    return (
        chosen.groupby(keys, as_index=False, dropna=False)
        .agg(
            occupancy_fraction=("occupancy_fraction", "sum"),
            annotated_area_mm2=("annotated_area_mm2", "sum"),
        )
        .assign(occupancy_percent=lambda frame: 100.0 * frame["occupancy_fraction"])
        .sort_values(["side", "cluster_id", "track_id", "region_type", "region_name"], kind="mergesort")
        .reset_index(drop=True)
    )


def plot_ant_resource_use_heatmaps(
    region_occupancy: pd.DataFrame,
    cluster_ids: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Plot raw annotated-resource occupancy for ants in selected clusters."""
    from matplotlib.colors import PowerNorm

    resources = resource_use_table(region_occupancy, cluster_ids)
    selected = [value for value in cluster_ids if str(value) in set(resources["cluster_id"].astype(str))]
    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(9.5, max(4.5, 0.42 * resources["track_id"].nunique() + 2.2 * len(selected))),
        squeeze=False,
    )

    for ax, cluster_id in zip(axes.ravel(), selected):
        cluster_resources = resources[resources["cluster_id"].astype(str) == str(cluster_id)]
        matrix = cluster_resources.pivot_table(
            index="track_id",
            columns="region_name",
            values="occupancy_percent",
            aggfunc="sum",
            fill_value=0.0,
        )
        matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]
        values = matrix.to_numpy(float)
        positive = values[values > 0]
        vmax = float(np.nanpercentile(positive, 99.0)) if positive.size else 1.0
        vmax = max(vmax, np.finfo(float).eps)
        image = ax.imshow(
            values,
            aspect="auto",
            interpolation="none",
            cmap="magma",
            norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax),
        )
        for row_index in range(values.shape[0]):
            for col_index in range(values.shape[1]):
                value = values[row_index, col_index]
                label = "0" if value == 0 else (f"{value:.2f}" if value >= 0.01 else "<.01")
                ax.text(col_index, row_index, label, ha="center", va="center", fontsize=7, color="white")
        ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(matrix.shape[0]), [f"Track {track_id}" for track_id in matrix.index])
        ax.set_title(f"{cluster_id}: fraction of detected time in each annotated resource")
        fig.colorbar(image, ax=ax, shrink=0.8, label="Detected time in region (%)")

    fig.suptitle("Putative ant-level resource-region specialization (occupancy proxy)", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    plt.show()
    return resources


def plot_food_water_specialization(
    region_occupancy: pd.DataFrame,
    cluster_ids: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Compare food and combined-water occupancy density for each ant."""
    resources = resource_use_table(region_occupancy, cluster_ids)
    category = (
        resources.groupby(["side", "cluster_id", "track_id", "track_name", "region_type"], as_index=False)
        .agg(
            occupancy_fraction=("occupancy_fraction", "sum"),
            annotated_area_mm2=("annotated_area_mm2", "sum"),
        )
    )
    category["occupancy_density_per_mm2"] = np.divide(
        category["occupancy_fraction"],
        category["annotated_area_mm2"],
        out=np.full(len(category), np.nan, dtype=float),
        where=category["annotated_area_mm2"].to_numpy(float) > 0,
    )
    keys = ["side", "cluster_id", "track_id", "track_name"]
    wide = category.pivot_table(
        index=keys,
        columns="region_type",
        values="occupancy_density_per_mm2",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    if "food" not in wide or "water" not in wide:
        raise ValueError("Food/water specialization plot requires both food and water annotations")

    selected = [value for value in cluster_ids if str(value) in set(wide["cluster_id"].astype(str))]
    fig, axes = plt.subplots(1, len(selected), figsize=(max(6.0, 5.4 * len(selected)), 5.0), squeeze=False)
    for ax, cluster_id in zip(axes.ravel(), selected):
        cluster = wide[wide["cluster_id"].astype(str) == str(cluster_id)]
        ax.scatter(cluster["food"], cluster["water"], s=45, color="tab:purple", alpha=0.8)
        for row in cluster.itertuples(index=False):
            ax.annotate(
                str(row.track_id),
                (row.food, row.water),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlabel("Food occupancy / annotated area (fraction mm$^{-2}$)")
        ax.set_ylabel("Water occupancy / annotated area (fraction mm$^{-2}$)")
        ax.set_title(f"{cluster_id}: labels are TrackID")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Putative food-versus-water specialization within in/out clusters")
    fig.tight_layout()
    plt.show()
    return wide.sort_values(["side", "cluster_id", "track_id"], kind="mergesort").reset_index(drop=True)


def _extract_one_track_resource_frames(task: dict[str, object]) -> pd.DataFrame:
    """Read only food/water candidates for one track and apply exact masks."""
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    track_path = Path(str(task["track_path"]))
    region_records = list(task["regions"])
    frame_col = str(task["frame_col"])
    x_col = str(task["x_col"])
    y_col = str(task["y_col"])
    bodypoint = int(task["bodypoint"])

    columns = set(pq.ParquetFile(track_path).schema.names)
    missing = {frame_col, x_col, y_col}.difference(columns)
    if missing:
        raise ValueError(f"{track_path.name} is missing required columns: {sorted(missing)}")

    bounding_filter = None
    for region in region_records:
        if region["shape"] == "rectangle":
            region_filter = (
                (ds.field(x_col) >= float(region["tracking_x_min_px"]))
                & (ds.field(x_col) <= float(region["tracking_x_max_px"]))
                & (ds.field(y_col) >= float(region["tracking_y_min_px"]))
                & (ds.field(y_col) <= float(region["tracking_y_max_px"]))
            )
        else:
            radius = float(region["radius_px"])
            center_x = float(region["tracking_center_x_px"])
            center_y = float(region["tracking_center_y_px"])
            region_filter = (
                (ds.field(x_col) >= center_x - radius)
                & (ds.field(x_col) <= center_x + radius)
                & (ds.field(y_col) >= center_y - radius)
                & (ds.field(y_col) <= center_y + radius)
            )
        bounding_filter = region_filter if bounding_filter is None else bounding_filter | region_filter

    if bounding_filter is None:
        return pd.DataFrame()
    if "Bodypoint" in columns:
        bounding_filter = (ds.field("Bodypoint") == bodypoint) & bounding_filter

    table = ds.dataset(track_path, format="parquet").to_table(
        columns=[frame_col, x_col, y_col],
        filter=bounding_filter,
        use_threads=False,
    )
    if table.num_rows == 0:
        return pd.DataFrame()

    position = table.to_pandas().rename(columns={frame_col: "frame", x_col: "x_px", y_col: "y_px"})
    for col in ("frame", "x_px", "y_px"):
        position[col] = pd.to_numeric(position[col], errors="coerce")
    position = position.dropna(subset=["frame", "x_px", "y_px"])
    if position.empty:
        return pd.DataFrame()
    position["frame"] = position["frame"].round().astype(np.int64)

    rows = []
    x = position["x_px"].to_numpy(float)
    y = position["y_px"].to_numpy(float)
    for region in region_records:
        if region["shape"] == "rectangle":
            inside = (
                (x >= float(region["tracking_x_min_px"]))
                & (x <= float(region["tracking_x_max_px"]))
                & (y >= float(region["tracking_y_min_px"]))
                & (y <= float(region["tracking_y_max_px"]))
            )
        else:
            dx = x - float(region["tracking_center_x_px"])
            dy = y - float(region["tracking_center_y_px"])
            inside = dx * dx + dy * dy <= float(region["radius_px"]) ** 2
        if not inside.any():
            continue
        hit = position.loc[inside, ["frame"]].drop_duplicates("frame").copy()
        hit["side"] = str(task["side"])
        hit["track_id"] = task["track_id"]
        hit["track_name"] = str(task["track_name"])
        hit["leiden_cluster"] = task["leiden_cluster"]
        hit["cluster_id"] = str(task["cluster_id"])
        hit["region_id"] = str(region["region_id"])
        hit["semantic_label"] = str(region["semantic_label"])
        hit["region_name"] = str(region["name"])
        hit["region_type"] = str(region["region_type"])
        rows.append(hit)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def extract_resource_presence_frames(
    cluster_table: pd.DataFrame,
    regions: pd.DataFrame,
    per_track_root: Path,
    *,
    resource_types: tuple[str, ...] = ("food", "water"),
    frame_col: str = "Frame",
    x_col: str = "TrackX",
    y_col: str = "TrackY",
    bodypoint: int = 0,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Extract exact detected frames inside annotated resource regions.

    Parquet predicate pushdown first keeps only each region's bounding box, so
    the returned table and peak memory stay small relative to the raw tracks.
    ``max_workers`` controls concurrent local reads; 4--6 is usually suitable
    for the bucket-mounted May dataset.
    """
    required_tracks = {"side", "track_id", "track_name", "leiden_cluster", "cluster_id"}
    missing = required_tracks.difference(cluster_table.columns)
    if missing:
        raise ValueError(f"cluster_table is missing columns: {sorted(missing)}")
    resource_types_normalized = tuple(str(value).lower() for value in resource_types)
    chosen_regions = regions[regions["region_type"].isin(resource_types_normalized)].copy()
    if chosen_regions.empty:
        raise ValueError(f"No regions match resource_types={resource_types_normalized}")

    per_track_root = Path(per_track_root)
    tasks: list[dict[str, object]] = []
    region_columns = [
        "region_id",
        "semantic_label",
        "name",
        "shape",
        "region_type",
        "tracking_x_min_px",
        "tracking_y_min_px",
        "tracking_x_max_px",
        "tracking_y_max_px",
        "tracking_center_x_px",
        "tracking_center_y_px",
        "radius_px",
    ]
    for track in cluster_table.itertuples(index=False):
        path = per_track_root / str(track.track_name)
        if not path.is_file():
            raise FileNotFoundError(f"Per-track parquet does not exist: {path}")
        side_regions = chosen_regions[chosen_regions["side"] == track.side]
        if side_regions.empty:
            raise ValueError(f"No resource regions are assigned to side={track.side!r}")
        tasks.append(
            {
                "track_path": str(path),
                "side": track.side,
                "track_id": track.track_id,
                "track_name": track.track_name,
                "leiden_cluster": track.leiden_cluster,
                "cluster_id": track.cluster_id,
                "regions": side_regions[region_columns].to_dict("records"),
                "frame_col": frame_col,
                "x_col": x_col,
                "y_col": y_col,
                "bodypoint": int(bodypoint),
            }
        )

    results: list[pd.DataFrame] = []
    workers = max(1, int(max_workers))
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            if index == 1 or index == len(tasks) or index % 10 == 0:
                print(f"resource frames: {index}/{len(tasks)} {task['track_name']}")
            result = _extract_one_track_resource_frames(task)
            if not result.empty:
                results.append(result)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_extract_one_track_resource_frames, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                result = future.result()
                if not result.empty:
                    results.append(result)
                if index == 1 or index == len(tasks) or index % 10 == 0:
                    print(f"resource frames: {index}/{len(tasks)} completed ({task['track_name']})")

    columns = [
        "frame",
        "side",
        "track_id",
        "track_name",
        "leiden_cluster",
        "cluster_id",
        "region_id",
        "semantic_label",
        "region_name",
        "region_type",
    ]
    if not results:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(results, ignore_index=True)[columns]
        .sort_values(["side", "track_id", "region_id", "frame"], kind="mergesort")
        .reset_index(drop=True)
    )


def build_resource_visit_bouts(
    resource_frames: pd.DataFrame,
    *,
    fps: float = 24.0,
    max_gap_seconds: float = 2.0,
    min_detected_seconds: float = 1.0,
    start_clock_seconds: float | None = None,
) -> pd.DataFrame:
    """Collapse resource-frame detections into temporally distinct visits."""
    required = {
        "frame",
        "side",
        "track_id",
        "track_name",
        "leiden_cluster",
        "cluster_id",
        "region_id",
        "region_name",
        "region_type",
    }
    missing = required.difference(resource_frames.columns)
    if missing:
        raise ValueError(f"resource_frames is missing columns: {sorted(missing)}")
    if fps <= 0 or max_gap_seconds < 0 or min_detected_seconds < 0:
        raise ValueError("fps must be positive and visit gap/duration settings must be nonnegative")

    group_cols = [
        "side",
        "track_id",
        "track_name",
        "leiden_cluster",
        "cluster_id",
        "region_id",
        "region_name",
        "region_type",
    ]
    gap_frames = max(1, int(round(float(max_gap_seconds) * float(fps))))
    min_detections = max(1, int(np.ceil(float(min_detected_seconds) * float(fps))))
    rows: list[dict[str, object]] = []
    for keys, group in resource_frames.groupby(group_cols, sort=False, dropna=False):
        frames = np.sort(group["frame"].dropna().astype(np.int64).unique())
        if not len(frames):
            continue
        split_at = np.flatnonzero(np.diff(frames) > gap_frames) + 1
        visit_number = 0
        for segment in np.split(frames, split_at):
            if len(segment) < min_detections:
                continue
            visit_number += 1
            start_frame = int(segment[0])
            end_frame = int(segment[-1])
            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "visit_number": visit_number,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "n_detected_frames": int(len(segment)),
                    "detected_seconds": float(len(segment)) / float(fps),
                    "span_seconds": float(end_frame - start_frame + 1) / float(fps),
                }
            )
            if start_clock_seconds is not None:
                absolute_seconds = float(start_clock_seconds) + start_frame / float(fps)
                row["start_elapsed_h"] = start_frame / float(fps) / 3600.0
                row["start_clock_hour"] = (absolute_seconds % 86400.0) / 3600.0
                row["recording_day"] = int(np.floor(absolute_seconds / 86400.0))
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=group_cols
            + ["visit_number", "start_frame", "end_frame", "n_detected_frames", "detected_seconds", "span_seconds"]
        )
    return pd.DataFrame(rows).sort_values(
        ["side", "track_id", "start_frame", "region_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_ant_resource_visits(
    cluster_table: pd.DataFrame,
    regions: pd.DataFrame,
    visit_bouts: pd.DataFrame,
) -> pd.DataFrame:
    """Return every ant/resource pair, including explicit zero-visit rows."""
    track_cols = ["side", "track_id", "track_name", "leiden_cluster", "cluster_id"]
    region_cols = ["side", "region_id", "semantic_label", "name", "region_type"]
    tracks = cluster_table[track_cols].drop_duplicates(track_cols)
    resources = regions[regions["region_type"].isin(["food", "water"])][region_cols].rename(
        columns={"name": "region_name"}
    )
    base = tracks.merge(resources, on="side", how="inner", validate="many_to_many")

    keys = [
        "side",
        "track_id",
        "track_name",
        "leiden_cluster",
        "cluster_id",
        "region_id",
        "region_name",
        "region_type",
    ]
    if visit_bouts.empty:
        totals = pd.DataFrame(columns=keys)
    else:
        totals = (
            visit_bouts.groupby(keys, as_index=False, dropna=False)
            .agg(
                n_visits=("visit_number", "size"),
                visit_detected_seconds=("detected_seconds", "sum"),
                visit_span_seconds=("span_seconds", "sum"),
                first_visit_frame=("start_frame", "min"),
                last_visit_frame=("end_frame", "max"),
            )
        )
    out = base.merge(totals, on=keys, how="left", validate="one_to_one")
    for col in ("n_visits", "visit_detected_seconds", "visit_span_seconds"):
        out[col] = out[col].fillna(0)
    out["n_visits"] = out["n_visits"].astype(int)
    out["visited"] = out["n_visits"] > 0
    out["visit_detected_minutes"] = out["visit_detected_seconds"] / 60.0
    out["visit_span_minutes"] = out["visit_span_seconds"] / 60.0
    return out.sort_values(
        ["side", "cluster_id", "track_id", "region_type", "region_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_activity_cluster_resource_visits(ant_resource_visits: pd.DataFrame) -> pd.DataFrame:
    """Summarize food/water participation and effort by occupancy cluster."""
    ant_type = (
        ant_resource_visits.groupby(
            ["side", "cluster_id", "leiden_cluster", "track_id", "track_name", "region_type"],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_visits=("n_visits", "sum"),
            visit_detected_minutes=("visit_detected_minutes", "sum"),
            visited=("visited", "max"),
        )
    )
    summary = (
        ant_type.groupby(["side", "cluster_id", "leiden_cluster", "region_type"], as_index=False)
        .agg(
            n_ants=("track_name", "nunique"),
            n_visitors=("visited", "sum"),
            mean_visits_per_ant=("n_visits", "mean"),
            median_visits_per_ant=("n_visits", "median"),
            mean_visit_minutes_per_ant=("visit_detected_minutes", "mean"),
            median_visit_minutes_per_ant=("visit_detected_minutes", "median"),
        )
    )
    summary["visitor_fraction"] = summary["n_visitors"] / summary["n_ants"]
    return summary.sort_values(["side", "cluster_id", "region_type"], kind="mergesort").reset_index(drop=True)


def plot_activity_cluster_resource_visits(ant_resource_visits: pd.DataFrame) -> pd.DataFrame:
    """Plot resource participation and time for every spatial activity cluster."""
    summary = summarize_activity_cluster_resource_visits(ant_resource_visits)
    sides = [side for side in ("left", "right") if side in set(summary["side"])]
    fig, axes = plt.subplots(
        len(sides),
        2,
        figsize=(12.0, max(4.2, 3.9 * len(sides))),
        squeeze=False,
    )
    colors = {"food": "tomato", "water": "deepskyblue"}
    for row_index, side in enumerate(sides):
        side_summary = summary[summary["side"] == side]
        clusters = side_summary["cluster_id"].drop_duplicates().to_list()
        x = np.arange(len(clusters), dtype=float)
        resource_types = [value for value in ("food", "water") if value in set(side_summary["region_type"])]
        width = 0.8 / max(1, len(resource_types))
        for resource_index, region_type in enumerate(resource_types):
            values = side_summary[side_summary["region_type"] == region_type].set_index("cluster_id")
            position = x - 0.4 + width / 2.0 + resource_index * width
            axes[row_index, 0].bar(
                position,
                values.reindex(clusters)["visitor_fraction"],
                width=width,
                color=colors.get(region_type),
                alpha=0.8,
                label=region_type,
            )
            axes[row_index, 1].bar(
                position,
                values.reindex(clusters)["mean_visit_minutes_per_ant"],
                width=width,
                color=colors.get(region_type),
                alpha=0.8,
                label=region_type,
            )
        axes[row_index, 0].set_ylabel("Fraction of ants with >=1 visit")
        axes[row_index, 0].set_ylim(0, 1.05)
        axes[row_index, 1].set_ylabel("Mean detected minutes per ant")
        for ax in axes[row_index]:
            ax.set_xticks(x, clusters, rotation=25, ha="right")
            ax.set_title(f"{side}: " + ("who visits" if ax is axes[row_index, 0] else "resource time"))
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("Food/water use by spatial activity cluster")
    fig.tight_layout()
    plt.show()
    return summary


def resource_time_of_day_tables(
    resource_frames: pd.DataFrame,
    cluster_table: pd.DataFrame,
    *,
    fps: float,
    start_clock_seconds: float,
    bin_minutes: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-ant and cluster-mean resource use folded onto a 24-hour day."""
    bin_seconds = float(bin_minutes) * 60.0
    n_bins_float = 86400.0 / bin_seconds
    n_bins = int(round(n_bins_float))
    if fps <= 0 or bin_seconds <= 0 or not np.isclose(n_bins, n_bins_float):
        raise ValueError("fps must be positive and bin_minutes must divide 24 hours evenly")

    # Combining two water annotations must not double-count a frame if their
    # boundaries ever touch.
    hits = resource_frames[
        ["side", "track_id", "track_name", "leiden_cluster", "cluster_id", "region_type", "frame"]
    ].drop_duplicates(["track_name", "region_type", "frame"])
    clock_seconds = (float(start_clock_seconds) + hits["frame"].to_numpy(float) / float(fps)) % 86400.0
    hits = hits.copy()
    hits["tod_bin"] = np.floor(clock_seconds / bin_seconds).astype(int)
    counts = (
        hits.groupby(
            ["side", "track_id", "track_name", "leiden_cluster", "cluster_id", "region_type", "tod_bin"],
            as_index=False,
        )
        .agg(n_detected_frames=("frame", "nunique"))
    )

    track_cols = ["side", "track_id", "track_name", "leiden_cluster", "cluster_id"]
    tracks = cluster_table[track_cols].drop_duplicates(track_cols)
    types = pd.DataFrame({"region_type": ["food", "water"]})
    base = tracks.merge(types, how="cross").merge(
        pd.DataFrame({"tod_bin": np.arange(n_bins, dtype=int)}),
        how="cross",
    )
    per_ant = base.merge(
        counts,
        on=track_cols + ["region_type", "tod_bin"],
        how="left",
        validate="one_to_one",
    )
    per_ant["n_detected_frames"] = per_ant["n_detected_frames"].fillna(0).astype(np.int64)
    per_ant["detected_seconds"] = per_ant["n_detected_frames"] / float(fps)
    per_ant["clock_hour"] = (per_ant["tod_bin"] + 0.5) * float(bin_minutes) / 60.0

    recording_frames = max(1, int(cluster_table["frame_max"].max()) - int(cluster_table["frame_min"].min()) + 1)
    recording_days = recording_frames / float(fps) / 86400.0
    per_ant["detected_seconds_per_recording_day"] = per_ant["detected_seconds"] / recording_days
    summary = (
        per_ant.groupby(
            ["side", "cluster_id", "leiden_cluster", "region_type", "tod_bin", "clock_hour"],
            as_index=False,
        )
        .agg(
            n_ants=("track_name", "nunique"),
            n_active_ants=("n_detected_frames", lambda values: int((values > 0).sum())),
            mean_seconds_per_ant_day=("detected_seconds_per_recording_day", "mean"),
            median_seconds_per_ant_day=("detected_seconds_per_recording_day", "median"),
            sem_seconds_per_ant_day=(
                "detected_seconds_per_recording_day",
                lambda values: float(pd.Series(values).sem()) if len(values) > 1 else 0.0,
            ),
        )
    )
    summary["active_ant_fraction"] = summary["n_active_ants"] / summary["n_ants"]
    return per_ant, summary


def summarize_resource_light_dark(
    resource_frames: pd.DataFrame,
    *,
    fps: float,
    start_clock_seconds: float,
    light_off_hour: float = 19.5,
    light_on_hour: float = 5.5,
) -> pd.DataFrame:
    """Summarize detected food/water time in the light versus dark phase."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    hits = resource_frames[
        ["frame", "side", "track_name", "cluster_id", "region_type"]
    ].drop_duplicates(["track_name", "region_type", "frame"])
    clock_hour = (
        float(start_clock_seconds) + hits["frame"].to_numpy(float) / float(fps)
    ) % 86400.0 / 3600.0
    hits = hits.copy()
    hits["dark"] = (clock_hour >= float(light_off_hour)) | (clock_hour < float(light_on_hour))
    per_ant = (
        hits.groupby(["side", "cluster_id", "track_name", "region_type"], as_index=False)
        .agg(n_detected_frames=("frame", "nunique"), n_dark_frames=("dark", "sum"))
    )
    per_ant["dark_fraction"] = per_ant["n_dark_frames"] / per_ant["n_detected_frames"]
    summary = (
        per_ant.groupby(["side", "cluster_id", "region_type"], as_index=False)
        .agg(
            n_visiting_ants=("track_name", "nunique"),
            n_detected_frames=("n_detected_frames", "sum"),
            n_dark_frames=("n_dark_frames", "sum"),
            median_ant_dark_fraction=("dark_fraction", "median"),
            mean_ant_dark_fraction=("dark_fraction", "mean"),
        )
    )
    summary["detected_hours"] = summary["n_detected_frames"] / float(fps) / 3600.0
    summary["pooled_dark_fraction"] = summary["n_dark_frames"] / summary["n_detected_frames"]
    return summary.sort_values(["side", "cluster_id", "region_type"], kind="mergesort").reset_index(drop=True)


def plot_resource_time_of_day(
    time_summary: pd.DataFrame,
    *,
    light_off_hour: float = 19.5,
    light_on_hour: float = 5.5,
) -> None:
    """Plot food/water use through clock time for each activity cluster."""
    sides = [side for side in ("left", "right") if side in set(time_summary["side"])]
    resource_types = [value for value in ("food", "water") if value in set(time_summary["region_type"])]
    fig, axes = plt.subplots(
        len(sides),
        len(resource_types),
        figsize=(6.4 * len(resource_types), 3.9 * len(sides)),
        sharex=True,
        squeeze=False,
    )
    for row_index, side in enumerate(sides):
        for col_index, region_type in enumerate(resource_types):
            ax = axes[row_index, col_index]
            chosen = time_summary[
                (time_summary["side"] == side) & (time_summary["region_type"] == region_type)
            ]
            ax.axvspan(float(light_off_hour), 24.0, color="0.88", alpha=0.65, lw=0)
            ax.axvspan(0.0, float(light_on_hour), color="0.88", alpha=0.65, lw=0, label="dark")
            for cluster_id, group in chosen.groupby("cluster_id", sort=True):
                group = group.sort_values("clock_hour")
                ax.plot(
                    group["clock_hour"],
                    group["mean_seconds_per_ant_day"],
                    lw=1.8,
                    label=str(cluster_id),
                )
            ax.axvline(float(light_off_hour), color="0.25", linestyle="--", lw=0.9)
            ax.axvline(float(light_on_hour), color="0.6", linestyle="-.", lw=0.9)
            ax.set_xlim(0, 24)
            ax.set_xticks(np.arange(0, 25, 4))
            ax.set_ylabel("Detected seconds / ant / day")
            ax.set_title(f"{side} {region_type}")
            ax.grid(True, alpha=0.22)
            ax.legend(fontsize=7, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("Clock hour")
    fig.suptitle("Resource use by clock time and spatial activity cluster")
    fig.tight_layout()
    plt.show()


def _resource_presence_groups(resource_frames: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "frame",
        "side",
        "track_id",
        "track_name",
        "leiden_cluster",
        "cluster_id",
        "region_id",
        "region_name",
        "region_type",
    ]
    region = resource_frames[base_cols].drop_duplicates(["track_name", "region_id", "frame"]).copy()
    region["resource_scope"] = "region"
    region["resource_key"] = region["region_name"].astype(str)

    type_level = resource_frames[
        [
            "frame",
            "side",
            "track_id",
            "track_name",
            "leiden_cluster",
            "cluster_id",
            "region_type",
        ]
    ].drop_duplicates(["track_name", "region_type", "frame"])
    type_level = type_level.copy()
    type_level["region_id"] = "type:" + type_level["region_type"].astype(str)
    type_level["region_name"] = type_level["region_type"].astype(str)
    type_level["resource_scope"] = "type"
    type_level["resource_key"] = type_level["region_type"].astype(str)
    return pd.concat([region, type_level[region.columns]], ignore_index=True)


def compute_resource_synchrony(
    resource_frames: pd.DataFrame,
    *,
    fps: float,
    bin_seconds: float = 60.0,
    recording_n_frames: int | None = None,
    n_permutations: int = 200,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare simultaneous resource use with an independent-shift null.

    Each ant's complete binary visitation series is circularly shifted by an
    independent random offset. This preserves its amount and bout structure
    while removing alignment with other ants. Positive departures indicate
    co-foraging; negative departures are consistent with temporal staggering
    or turn-taking. The null is exploratory because it also removes shared
    time-of-day preferences.
    """
    if fps <= 0 or bin_seconds <= 0 or n_permutations < 1:
        raise ValueError("fps/bin_seconds must be positive and n_permutations must be >= 1")
    bin_frames = max(1, int(round(float(fps) * float(bin_seconds))))
    grouped_frames = _resource_presence_groups(resource_frames)
    grouped_frames["time_bin"] = grouped_frames["frame"].astype(np.int64) // bin_frames
    grouped_frames = grouped_frames.drop_duplicates(
        ["side", "resource_scope", "resource_key", "track_name", "time_bin"]
    )
    if recording_n_frames is None:
        recording_n_frames = int(resource_frames["frame"].max()) + 1
    n_time_bins = max(1, int(np.ceil(int(recording_n_frames) / bin_frames)))

    concurrency = (
        grouped_frames.groupby(
            ["side", "resource_scope", "resource_key", "region_type", "time_bin"],
            as_index=False,
        )
        .agg(n_ants=("track_name", "nunique"))
    )
    concurrency["elapsed_h"] = concurrency["time_bin"] * float(bin_seconds) / 3600.0

    rng = np.random.default_rng(int(random_state))
    rows = []
    group_cols = ["side", "resource_scope", "resource_key", "region_type"]
    for keys, group in grouped_frames.groupby(group_cols, sort=True, dropna=False):
        ant_names = sorted(group["track_name"].unique())
        ant_index = {name: index for index, name in enumerate(ant_names)}
        matrix = np.zeros((len(ant_names), n_time_bins), dtype=np.bool_)
        ant_positions = group["track_name"].map(ant_index).to_numpy(int)
        time_positions = group["time_bin"].to_numpy(int)
        valid = (time_positions >= 0) & (time_positions < n_time_bins)
        matrix[ant_positions[valid], time_positions[valid]] = True

        observed_concurrency = matrix.sum(axis=0, dtype=np.int16)
        observed_pair_overlap = float(np.sum(observed_concurrency * (observed_concurrency - 1) / 2.0))
        active = observed_concurrency > 0
        observed_shared_fraction = float(np.mean(observed_concurrency[active] >= 2)) if active.any() else 0.0

        null_pair_overlap = np.empty(int(n_permutations), dtype=float)
        null_shared_fraction = np.empty(int(n_permutations), dtype=float)
        for permutation in range(int(n_permutations)):
            null_concurrency = np.zeros(n_time_bins, dtype=np.int16)
            for ant_presence in matrix:
                null_concurrency += np.roll(ant_presence, int(rng.integers(0, n_time_bins)))
            null_pair_overlap[permutation] = np.sum(
                null_concurrency * (null_concurrency - 1) / 2.0
            )
            null_active = null_concurrency > 0
            null_shared_fraction[permutation] = (
                float(np.mean(null_concurrency[null_active] >= 2)) if null_active.any() else 0.0
            )

        null_low, null_high = np.quantile(null_pair_overlap, [0.025, 0.975])
        null_mean = float(np.mean(null_pair_overlap))
        null_sd = float(np.std(null_pair_overlap, ddof=1)) if len(null_pair_overlap) > 1 else 0.0
        if len(ant_names) < 2 or (observed_pair_overlap == 0 and null_mean == 0):
            label = "insufficient overlap"
        elif observed_pair_overlap > null_high:
            label = "together"
        elif observed_pair_overlap < null_low:
            label = "staggered/turn-taking"
        else:
            label = "no clear departure"
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_visiting_ants": len(ant_names),
                "n_active_bins": int(active.sum()),
                "n_solo_bins": int(np.sum(observed_concurrency == 1)),
                "n_shared_bins": int(np.sum(observed_concurrency >= 2)),
                "max_concurrent_ants": int(observed_concurrency.max(initial=0)),
                "observed_shared_fraction": observed_shared_fraction,
                "null_mean_shared_fraction": float(np.mean(null_shared_fraction)),
                "observed_pair_overlap": observed_pair_overlap,
                "null_mean_pair_overlap": null_mean,
                "null_pair_overlap_low": float(null_low),
                "null_pair_overlap_high": float(null_high),
                "pair_overlap_z": (observed_pair_overlap - null_mean) / null_sd if null_sd > 0 else np.nan,
                "log2_observed_null_overlap": float(
                    np.log2((observed_pair_overlap + 0.5) / (null_mean + 0.5))
                ),
                "p_more_together": float((1 + np.sum(null_pair_overlap >= observed_pair_overlap)) / (len(null_pair_overlap) + 1)),
                "p_more_staggered": float((1 + np.sum(null_pair_overlap <= observed_pair_overlap)) / (len(null_pair_overlap) + 1)),
                "coordination_label": label,
                "bin_seconds": float(bin_seconds),
                "n_permutations": int(n_permutations),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(group_cols, kind="mergesort").reset_index(drop=True),
        concurrency.sort_values(group_cols + ["time_bin"], kind="mergesort").reset_index(drop=True),
    )


def plot_resource_synchrony(
    synchrony: pd.DataFrame,
    *,
    resource_scope: str = "region",
) -> None:
    """Plot observed/null simultaneous-use ratios for each annotated resource."""
    from matplotlib.patches import Patch

    chosen = synchrony[synchrony["resource_scope"] == resource_scope].copy()
    sides = [side for side in ("left", "right") if side in set(chosen["side"])]
    fig, axes = plt.subplots(1, len(sides), figsize=(max(6.0, 6.2 * len(sides)), 4.8), squeeze=False)
    colors = {
        "together": "tab:green",
        "staggered/turn-taking": "tab:orange",
        "no clear departure": "0.45",
        "insufficient overlap": "0.75",
    }
    for ax, side in zip(axes.ravel(), sides):
        side_data = chosen[chosen["side"] == side].sort_values(["region_type", "resource_key"])
        y = np.arange(len(side_data))
        values = side_data["log2_observed_null_overlap"].to_numpy(float)
        ax.barh(
            y,
            values,
            color=[colors.get(value, "0.45") for value in side_data["coordination_label"]],
            alpha=0.85,
        )
        ax.axvline(0, color="0.2", lw=1)
        if resource_scope == "region":
            tick_labels = [
                f"{row.region_type}: {row.resource_key}"
                for row in side_data.itertuples(index=False)
            ]
        else:
            tick_labels = side_data["resource_key"].astype(str).to_list()
        ax.set_yticks(y, tick_labels)
        ax.set_xlabel("log2(observed / shifted-null pair overlap)")
        ax.set_title(f"{side}:  + together, - staggered")
        ax.grid(True, axis="x", alpha=0.25)
        ax.legend(
            handles=[
                Patch(facecolor=color, label=label)
                for label, color in colors.items()
                if label in set(side_data["coordination_label"])
            ],
            fontsize=7,
            loc="best",
        )
    fig.suptitle(f"Resource-use synchrony ({resource_scope}, circular-shift null)")
    fig.tight_layout()
    plt.show()


def plot_resource_presence_heatmaps(
    resource_frames: pd.DataFrame,
    cluster_table: pd.DataFrame,
    *,
    fps: float,
    start_clock_seconds: float,
    bin_minutes: float = 5.0,
    cluster_ids: tuple[str, ...] | list[str] | None = None,
    light_off_hour: float = 19.5,
    light_on_hour: float = 5.5,
) -> pd.DataFrame:
    """Plot ant-by-time food/water presence to expose bouts and co-foraging."""
    chosen_tracks = cluster_table.copy()
    if cluster_ids is not None:
        selected = {str(value) for value in cluster_ids}
        chosen_tracks = chosen_tracks[chosen_tracks["cluster_id"].astype(str).isin(selected)]
    chosen_names = set(chosen_tracks["track_name"])
    frames = resource_frames[resource_frames["track_name"].isin(chosen_names)].copy()
    bin_seconds = float(bin_minutes) * 60.0
    bin_frames = max(1, int(round(float(fps) * bin_seconds)))
    frames["time_bin"] = frames["frame"].astype(np.int64) // bin_frames
    presence = frames.drop_duplicates(["track_name", "region_type", "time_bin"])
    presence = (
        presence.groupby(
            ["side", "cluster_id", "track_id", "track_name", "region_type", "time_bin"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "present"})
    )

    sides = [side for side in ("left", "right") if side in set(chosen_tracks["side"])]
    types = [value for value in ("food", "water") if value in set(frames["region_type"])]
    max_frame = int(chosen_tracks["frame_max"].max())
    n_bins = max(1, int(np.ceil((max_frame + 1) / bin_frames)))
    max_time_h = n_bins * bin_seconds / 3600.0
    fig, axes = plt.subplots(
        len(sides),
        len(types),
        figsize=(7.2 * len(types), max(4.2, 0.16 * len(chosen_tracks) + 2.2 * len(sides))),
        squeeze=False,
        sharex=True,
    )
    for row_index, side in enumerate(sides):
        side_tracks = chosen_tracks[chosen_tracks["side"] == side].sort_values(
            ["cluster_id", "track_id"], kind="mergesort"
        )
        name_to_row = {name: index for index, name in enumerate(side_tracks["track_name"])}
        for col_index, region_type in enumerate(types):
            ax = axes[row_index, col_index]
            matrix = np.zeros((len(side_tracks), n_bins), dtype=np.uint8)
            selected_presence = presence[
                (presence["side"] == side) & (presence["region_type"] == region_type)
            ]
            if not selected_presence.empty:
                row_position = selected_presence["track_name"].map(name_to_row).to_numpy(int)
                col_position = selected_presence["time_bin"].to_numpy(int)
                valid = (col_position >= 0) & (col_position < n_bins)
                matrix[row_position[valid], col_position[valid]] = 1
            ax.imshow(
                matrix,
                aspect="auto",
                interpolation="none",
                cmap="Greys",
                vmin=0,
                vmax=1,
                extent=[0, max_time_h, len(side_tracks), 0],
            )
            add_light_shading(
                ax,
                int(start_clock_seconds),
                max_time_h,
                light_off_hour=light_off_hour,
                light_on_hour=light_on_hour,
                shade_alpha=0.12,
            )
            labels = [f"{row.cluster_id}:{row.track_id}" for row in side_tracks.itertuples(index=False)]
            if len(labels) <= 45:
                ax.set_yticks(np.arange(len(labels)) + 0.5, labels, fontsize=6)
            else:
                ax.set_yticks([])
            ax.set_title(f"{side} {region_type} ({bin_minutes:g}-min presence)")
            ax.set_ylabel("activity cluster:TrackID")
    for ax in axes[-1]:
        ax.set_xlabel("Elapsed hours")
    fig.suptitle("Individual resource-use timing; vertical alignment indicates co-foraging")
    fig.tight_layout()
    plt.show()
    return presence


def cluster_resource_visitors(
    resource_frames: pd.DataFrame,
    visit_bouts: pd.DataFrame,
    cluster_table: pd.DataFrame,
    cluster_ids: tuple[str, ...] | list[str],
    *,
    fps: float,
    start_clock_seconds: float,
    tod_bin_hours: float = 4.0,
    n_neighbors: int = 8,
    leiden_resolution: float = 0.8,
    random_state: int = 0,
) -> dict[str, dict[str, object]]:
    """Subcluster roaming ants by resource preference and visit timing."""
    selected = {str(value) for value in cluster_ids}
    selected_tracks = cluster_table[cluster_table["cluster_id"].astype(str).isin(selected)].copy()
    if selected_tracks.empty:
        raise ValueError(f"No tracks match cluster_ids={sorted(selected)}")
    if fps <= 0 or tod_bin_hours <= 0 or not np.isclose(24.0 / tod_bin_hours, round(24.0 / tod_bin_hours)):
        raise ValueError("fps must be positive and tod_bin_hours must divide 24 hours")

    results: dict[str, dict[str, object]] = {}
    for side, side_tracks in selected_tracks.groupby("side", sort=True):
        names = set(side_tracks["track_name"])
        frames = resource_frames[resource_frames["track_name"].isin(names)].copy()
        if frames.empty:
            print(f"WARNING: no resource detections for selected {side} tracks; skipping subclustering")
            continue
        visitor_names = sorted(frames["track_name"].unique())
        if len(visitor_names) < 3:
            print(f"WARNING: only {len(visitor_names)} resource visitors on {side}; skipping subclustering")
            continue

        # Relative use of each physically distinct food/water annotation.
        region_counts = (
            frames.drop_duplicates(["track_name", "region_id", "frame"])
            .groupby(["track_name", "region_name"])["frame"]
            .nunique()
            .unstack(fill_value=0)
            .reindex(visitor_names, fill_value=0)
        )
        region_props = region_counts.div(region_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        region_props.columns = [f"region::{value}" for value in region_props.columns]

        # Time preference, folded across recording days.
        unique_type_frames = frames.drop_duplicates(["track_name", "region_type", "frame"])[
            ["track_name", "frame"]
        ].drop_duplicates()
        clock_hour = (
            float(start_clock_seconds) + unique_type_frames["frame"].to_numpy(float) / float(fps)
        ) % 86400.0 / 3600.0
        n_tod_bins = int(round(24.0 / float(tod_bin_hours)))
        unique_type_frames = unique_type_frames.copy()
        unique_type_frames["tod_bin"] = np.floor(clock_hour / float(tod_bin_hours)).astype(int)
        tod_counts = (
            unique_type_frames.groupby(["track_name", "tod_bin"])["frame"]
            .nunique()
            .unstack(fill_value=0)
            .reindex(index=visitor_names, columns=np.arange(n_tod_bins), fill_value=0)
        )
        tod_props = tod_counts.div(tod_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        tod_props.columns = [
            f"clock::{index * tod_bin_hours:g}-{(index + 1) * tod_bin_hours:g}h"
            for index in range(n_tod_bins)
        ]

        # Overall effort and bout count allow two ants with equal proportions
        # but radically different foraging effort to separate.
        total_minutes = region_counts.sum(axis=1) / float(fps) / 60.0
        side_bouts = visit_bouts[visit_bouts["track_name"].isin(visitor_names)]
        bout_counts = side_bouts.groupby("track_name").size().reindex(visitor_names, fill_value=0)
        amount = pd.DataFrame(
            {
                "amount::log1p_detected_minutes": np.log1p(total_minutes),
                "amount::log1p_visits": np.log1p(bout_counts.astype(float)),
            },
            index=visitor_names,
        )
        raw_features = pd.concat([region_props, tod_props, amount], axis=1).astype(float)
        variable = raw_features.std(axis=0, ddof=0) > 1e-12
        raw_features = raw_features.loc[:, variable]
        if raw_features.shape[1] < 2:
            print(f"WARNING: fewer than two variable visitor features on {side}; skipping subclustering")
            continue
        scaled = (raw_features - raw_features.mean(axis=0)) / raw_features.std(axis=0, ddof=0)

        check_clustering_dependencies()
        neighbor_count = min(max(2, int(n_neighbors)), len(scaled) - 1)
        labels = leiden_labels(
            scaled.to_numpy(float),
            n_neighbors=neighbor_count,
            metric="euclidean",
            resolution=float(leiden_resolution),
            random_state=int(random_state),
        )
        embedding = umap_embedding(
            scaled.to_numpy(float),
            n_neighbors=neighbor_count,
            min_dist=0.15,
            metric="euclidean",
            random_state=int(random_state),
        )

        visitor_table = side_tracks[side_tracks["track_name"].isin(visitor_names)][
            ["side", "track_id", "track_name", "leiden_cluster", "cluster_id"]
        ].drop_duplicates("track_name").set_index("track_name").reindex(visitor_names).reset_index()
        visitor_table["resource_subcluster"] = labels
        visitor_table["resource_umap1"] = embedding[:, 0]
        visitor_table["resource_umap2"] = embedding[:, 1]
        feature_table = raw_features.reset_index(names="track_name").merge(
            visitor_table,
            on="track_name",
            how="left",
            validate="one_to_one",
        )
        results[str(side)] = {
            "table": visitor_table,
            "feature_table": feature_table,
            "raw_features": raw_features,
            "scaled_features": scaled,
            "feature_names": raw_features.columns.to_list(),
        }
    return results


def plot_resource_visitor_subclusters(visitor_results: dict[str, dict[str, object]]) -> None:
    """Plot visitor UMAPs and the feature profile defining each subcluster."""
    for side, result in visitor_results.items():
        table = result["table"]
        scaled = result["scaled_features"]
        if not isinstance(table, pd.DataFrame) or not isinstance(scaled, pd.DataFrame):
            raise TypeError("visitor_results contains malformed table/features")
        ordered = table.set_index("track_name").loc[scaled.index]
        cluster_means = (
            scaled.assign(resource_subcluster=ordered["resource_subcluster"].to_numpy())
            .groupby("resource_subcluster", sort=True)
            .mean()
        )
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.0, 1.45]})
        scatter = axes[0].scatter(
            table["resource_umap1"],
            table["resource_umap2"],
            c=table["resource_subcluster"],
            cmap="tab10",
            s=52,
            alpha=0.85,
        )
        for row in table.itertuples(index=False):
            axes[0].annotate(
                str(row.track_id),
                (row.resource_umap1, row.resource_umap2),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=6,
            )
        axes[0].set_xlabel("Resource/timing UMAP 1")
        axes[0].set_ylabel("Resource/timing UMAP 2")
        axes[0].set_title(f"{side}: labels are TrackID")
        axes[0].grid(True, alpha=0.2)
        fig.colorbar(scatter, ax=axes[0], label="resource subcluster")

        limit = max(1.0, float(np.nanpercentile(np.abs(cluster_means.to_numpy(float)), 98)))
        image = axes[1].imshow(
            cluster_means.to_numpy(float),
            aspect="auto",
            interpolation="none",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axes[1].set_yticks(
            np.arange(len(cluster_means)),
            [f"subcluster {value}" for value in cluster_means.index],
        )
        axes[1].set_xticks(
            np.arange(cluster_means.shape[1]),
            cluster_means.columns,
            rotation=55,
            ha="right",
            fontsize=7,
        )
        axes[1].set_title("Mean standardized feature profile")
        fig.colorbar(image, ax=axes[1], label="feature z-score")
        fig.suptitle(f"{side} roaming ants: subclusters by resource and clock-time use")
        fig.tight_layout()
        plt.show()
