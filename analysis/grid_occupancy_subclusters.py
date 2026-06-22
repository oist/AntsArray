# %%
# Test for subclusters inside grid-occupancy clusters.
#
# This script uses the cluster assignments from grid_occupancy.py and asks whether
# tracks inside each coarse cluster still separate by:
# - specific occupied locations, using each track's normalized grid occupancy map;
# - time of day, using each track's binned mean speed profile;
# - the joint spatial + temporal feature space.

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
SPEED_ROOT = go.infer_speed_root(GRID_ROOT)
CLUSTER_TABLE_PATH = GRID_ROOT / "track_cluster_ids.csv"

MIN_PRESENT_FRAC = 0.40
LIGHT_ON_HOUR = 5.5
TEMPORAL_BIN_MINUTES = 10.0
SPATIAL_BIN_FACTOR = 5
OUT_ROOT = GRID_ROOT / f"subcluster_tests_{TEMPORAL_BIN_MINUTES:g}min"

MIN_CLUSTER_SIZE = 8
MIN_SUBCLUSTER_SIZE = 3
MAX_SUBCLUSTERS = 4
N_PERMUTATIONS = 200
RANDOM_STATE = 0
RECOMPUTE_FEATURES = False


# %%
def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def cache_settings() -> dict[str, object]:
    return {
        "grid_root": str(GRID_ROOT),
        "speed_root": str(SPEED_ROOT),
        "cluster_table_path": str(CLUSTER_TABLE_PATH),
        "min_present_frac": float(MIN_PRESENT_FRAC),
        "light_on_hour": float(LIGHT_ON_HOUR),
        "temporal_bin_minutes": float(TEMPORAL_BIN_MINUTES),
        "temporal_feature": "mean_speed_mm_s_v1",
        "spatial_bin_factor": int(SPATIAL_BIN_FACTOR),
    }


def settings_match(path: Path, settings: dict[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        saved = load_json(path)
    except Exception:
        return False
    return all(saved.get(key) == value for key, value in settings.items())


def load_clustered_tracks() -> pd.DataFrame:
    tracks = go.load_grid_tracks(GRID_ROOT)
    tracks = go.attach_detection_fraction(tracks, SPEED_ROOT)
    tracks = go.select_good_tracks(tracks, MIN_PRESENT_FRAC, side="both")

    cluster_ids = pd.read_csv(CLUSTER_TABLE_PATH).rename(
        columns={"TrackID": "track_id", "leiden_cluster_id": "leiden_cluster"}
    )
    merged = tracks.merge(
        cluster_ids[["track_id", "side", "cluster_id", "leiden_cluster"]],
        on=["track_id", "side"],
        how="inner",
        validate="one_to_one",
    )

    speed_tracks = go.load_speed_tracks(SPEED_ROOT)[
        ["track_name", "speed_path", "frame_min", "n_frames", "n_observed_frames", "fps"]
    ].rename(
        columns={
            "frame_min": "speed_frame_min",
            "n_frames": "speed_n_frames",
            "n_observed_frames": "speed_n_observed_frames",
            "fps": "speed_fps",
        }
    )
    merged = merged.merge(speed_tracks, on="track_name", how="left", validate="one_to_one")
    return merged.sort_values(["side", "cluster_id", "track_id"], kind="mergesort").reset_index(drop=True)


def coarsen_sum_2d(values: np.ndarray, factor: int) -> np.ndarray:
    factor = int(factor)
    if factor <= 1:
        return values.astype(np.float32, copy=False)
    n_y, n_x = values.shape
    pad_y = (-n_y) % factor
    pad_x = (-n_x) % factor
    padded = np.pad(values.astype(np.float32, copy=False), ((0, pad_y), (0, pad_x)), constant_values=0.0)
    return padded.reshape(padded.shape[0] // factor, factor, padded.shape[1] // factor, factor).sum(axis=(1, 3))


def probability_sqrt_features(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    row_sum = matrix.sum(axis=1, keepdims=True)
    prob = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
    return np.sqrt(np.clip(prob, 0.0, None)).astype(np.float32, copy=False)


def load_spatial_features(tracks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    maps = []
    x_edges_ref = None
    y_edges_ref = None
    for i, row in enumerate(tracks.itertuples(index=False), start=1):
        if i == 1 or i == len(tracks) or i % 25 == 0:
            print(f"spatial features: {i}/{len(tracks)} {row.track_name}")
        hist = np.load(row.occupancy_path).astype(np.float32, copy=False)
        hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
        binned = coarsen_sum_2d(hist, SPATIAL_BIN_FACTOR)
        maps.append(binned)
        if x_edges_ref is None:
            x_edges_ref = np.load(row.x_edges_path).astype(np.float32, copy=False)
            y_edges_ref = np.load(row.y_edges_path).astype(np.float32, copy=False)
    max_y = max(map_.shape[0] for map_ in maps)
    max_x = max(map_.shape[1] for map_ in maps)
    padded_maps = np.zeros((len(maps), max_y, max_x), dtype=np.float32)
    for i, map_ in enumerate(maps):
        padded_maps[i, : map_.shape[0], : map_.shape[1]] = map_
    raw = padded_maps.reshape(len(maps), -1)
    return probability_sqrt_features(raw), padded_maps, x_edges_ref, y_edges_ref


def recording_start_clock_seconds(tracks: pd.DataFrame) -> int:
    if tracks.empty:
        raise ValueError("Cannot infer recording start time from an empty table")
    return go.parse_track_start_seconds(str(tracks["track_name"].iloc[0]))


def temporal_speed_for_track(row: pd.Series, *, recording_start_seconds: int, light_on_seconds: int) -> np.ndarray:
    n_bins = int(round((24.0 * 60.0) / float(TEMPORAL_BIN_MINUTES)))
    if not np.isclose(n_bins * float(TEMPORAL_BIN_MINUTES), 24.0 * 60.0):
        raise ValueError("TEMPORAL_BIN_MINUTES must divide 24 hours evenly")
    if pd.isna(row.get("speed_path")):
        return np.full(n_bins, np.nan, dtype=np.float32)

    speed_path = Path(row["speed_path"])
    if not speed_path.exists():
        return np.full(n_bins, np.nan, dtype=np.float32)

    speed = np.load(speed_path, mmap_mode="r")
    finite_idx = np.flatnonzero(np.isfinite(speed))
    if finite_idx.size == 0:
        return np.full(n_bins, np.nan, dtype=np.float32)

    fps = float(row.get("speed_fps", 24.0))
    bin_frames = int(round(float(TEMPORAL_BIN_MINUTES) * 60.0 * fps))
    start_offset_frames = int(round(((int(recording_start_seconds) - int(light_on_seconds)) % (24 * 3600)) * fps))
    frames = finite_idx.astype(np.int64, copy=False) + int(row["speed_frame_min"])
    temporal_bin = ((frames + start_offset_frames) // bin_frames) % n_bins
    finite_speed = np.asarray(speed[finite_idx], dtype=np.float64)
    counts = np.bincount(temporal_bin, minlength=n_bins).astype(np.float64)
    sums = np.bincount(temporal_bin, weights=finite_speed, minlength=n_bins).astype(np.float64)
    mean_speed = np.full(n_bins, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=mean_speed, where=counts > 0)
    return mean_speed.astype(np.float32, copy=False)


def speed_profile_features(speed_profiles: np.ndarray) -> np.ndarray:
    values = np.asarray(speed_profiles, dtype=np.float32).copy()
    if not np.isfinite(values).any():
        return np.zeros_like(values, dtype=np.float32)
    col_fill = np.nanmedian(values, axis=0)
    global_fill = float(np.nanmedian(values))
    col_fill = np.where(np.isfinite(col_fill), col_fill, global_fill).astype(np.float32)
    missing_rows, missing_cols = np.where(~np.isfinite(values))
    if missing_rows.size:
        values[missing_rows, missing_cols] = col_fill[missing_cols]
    return values.astype(np.float32, copy=False)


def zscore_columns(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(matrix, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    active = std > 0
    scaled = np.divide(values - mean, std, out=np.zeros_like(values, dtype=np.float32), where=active)
    return scaled.astype(np.float32, copy=False), int(active.sum())


def joint_spatial_temporal_features(spatial_features: np.ndarray, temporal_features: np.ndarray) -> tuple[np.ndarray, int]:
    spatial_scaled, n_spatial_active = zscore_columns(spatial_features)
    temporal_scaled, n_temporal_active = zscore_columns(temporal_features)
    if n_spatial_active:
        spatial_scaled = spatial_scaled / math.sqrt(n_spatial_active)
    if n_temporal_active:
        temporal_scaled = temporal_scaled / math.sqrt(n_temporal_active)
    return np.concatenate([spatial_scaled, temporal_scaled], axis=1).astype(np.float32, copy=False), spatial_scaled.shape[1]


def load_temporal_features(tracks: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rec_start = recording_start_clock_seconds(tracks)
    light_on = int(round(float(LIGHT_ON_HOUR) * 3600.0))
    speed_rows = []
    for i, (_, row) in enumerate(tracks.iterrows(), start=1):
        if i == 1 or i == len(tracks) or i % 25 == 0:
            print(f"temporal speed features: {i}/{len(tracks)} {row['track_name']}")
        speed_rows.append(temporal_speed_for_track(row, recording_start_seconds=rec_start, light_on_seconds=light_on))
    speed_profiles = np.vstack(speed_rows).astype(np.float32, copy=False)
    hours = (np.arange(speed_profiles.shape[1], dtype=np.float32) + 0.5) * float(TEMPORAL_BIN_MINUTES) / 60.0
    return speed_profile_features(speed_profiles), speed_profiles, hours


def build_or_load_features(tracks: pd.DataFrame) -> dict[str, object]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    settings = cache_settings()
    metadata_path = OUT_ROOT / "feature_cache_metadata.json"
    cache_path = OUT_ROOT / "feature_cache.npz"
    tracks_path = OUT_ROOT / "tracks_for_subcluster_tests.csv"

    if not RECOMPUTE_FEATURES and cache_path.exists() and settings_match(metadata_path, settings) and tracks_path.exists():
        print(f"feature cache hit: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as data:
            return {
                "tracks": pd.read_csv(tracks_path),
                "spatial_features": data["spatial_features"],
                "spatial_maps": data["spatial_maps"],
                "temporal_features": data["temporal_features"],
                "temporal_speed": data["temporal_speed"],
                "temporal_hours": data["temporal_hours"],
                "x_edges_mm": data["x_edges_mm"],
                "y_edges_mm": data["y_edges_mm"],
            }

    spatial_features, spatial_maps, x_edges, y_edges = load_spatial_features(tracks)
    temporal_features, temporal_speed, temporal_hours = load_temporal_features(tracks)

    serializable_tracks = tracks.copy()
    for col in serializable_tracks.columns:
        if serializable_tracks[col].dtype == object:
            serializable_tracks[col] = serializable_tracks[col].map(lambda value: str(value) if isinstance(value, Path) else value)
    serializable_tracks.to_csv(tracks_path, index=False)
    np.savez_compressed(
        cache_path,
        spatial_features=spatial_features,
        spatial_maps=spatial_maps,
        temporal_features=temporal_features,
        temporal_speed=temporal_speed,
        temporal_hours=temporal_hours,
        x_edges_mm=x_edges,
        y_edges_mm=y_edges,
    )
    metadata_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"feature cache wrote: {cache_path}")
    return {
        "tracks": serializable_tracks,
        "spatial_features": spatial_features,
        "spatial_maps": spatial_maps,
        "temporal_features": temporal_features,
        "temporal_speed": temporal_speed,
        "temporal_hours": temporal_hours,
        "x_edges_mm": x_edges,
        "y_edges_mm": y_edges,
    }


def kmeans_labels(features: np.ndarray, k: int, *, random_state: int, n_init: int) -> np.ndarray | None:
    from sklearn.cluster import KMeans

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = KMeans(n_clusters=int(k), n_init=int(n_init), random_state=int(random_state)).fit_predict(features)
    if np.unique(labels).size < int(k):
        return None
    return labels.astype(int)


def best_kmeans_partition(
    features: np.ndarray,
    *,
    max_k: int,
    min_subcluster_size: int,
    random_state: int,
    n_init: int = 20,
) -> dict[str, object]:
    from sklearn.metrics import silhouette_score

    n = int(features.shape[0])
    max_k = min(int(max_k), n - 1)
    best = {
        "k": 1,
        "labels": np.zeros(n, dtype=int),
        "silhouette": np.nan,
        "counts": [n],
    }
    if n < max(3, 2 * int(min_subcluster_size)) or max_k < 2:
        return best

    for k in range(2, max_k + 1):
        labels = kmeans_labels(features, k, random_state=random_state, n_init=n_init)
        if labels is None:
            continue
        counts = np.bincount(labels)
        if counts.min() < int(min_subcluster_size):
            continue
        score = float(silhouette_score(features, labels, metric="euclidean"))
        if not np.isfinite(best["silhouette"]) or score > float(best["silhouette"]):
            best = {"k": int(k), "labels": labels, "silhouette": score, "counts": counts.tolist()}
    return best


def permute_features(
    features: np.ndarray,
    *,
    kind: str,
    rng: np.random.Generator,
    split_at: int | None = None,
) -> np.ndarray:
    out = np.asarray(features, dtype=np.float32).copy()
    if kind == "temporal":
        for i in range(out.shape[0]):
            out[i] = out[i, rng.permutation(out.shape[1])]
    elif kind == "spatial":
        for i in range(out.shape[0]):
            out[i] = out[i, rng.permutation(out.shape[1])]
    elif kind == "joint":
        if split_at is None:
            raise ValueError("split_at is required for joint feature permutation")
        split_at = int(split_at)
        for i in range(out.shape[0]):
            out[i, :split_at] = out[i, rng.permutation(split_at)]
            temporal_order = split_at + rng.permutation(out.shape[1] - split_at)
            out[i, split_at:] = out[i, temporal_order]
    else:
        raise ValueError("kind must be 'spatial', 'temporal', or 'joint'")
    return out


def null_silhouette_distribution(
    features: np.ndarray,
    *,
    kind: str,
    split_at: int | None = None,
    max_k: int,
    min_subcluster_size: int,
    n_permutations: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(random_state))
    scores = np.full(int(n_permutations), np.nan, dtype=np.float32)
    for i in range(int(n_permutations)):
        permuted = permute_features(features, kind=kind, split_at=split_at, rng=rng)
        result = best_kmeans_partition(
            permuted,
            max_k=max_k,
            min_subcluster_size=min_subcluster_size,
            random_state=int(rng.integers(0, 2**31 - 1)),
            n_init=5,
        )
        scores[i] = result["silhouette"]
    return scores


def silhouette_p_value(observed: float, null_scores: np.ndarray) -> float:
    finite = null_scores[np.isfinite(null_scores)]
    if not np.isfinite(observed) or finite.size == 0:
        return np.nan
    return float((1 + np.sum(finite >= float(observed))) / (finite.size + 1))


def pca_xy(features: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    if features.shape[0] < 2:
        return np.zeros((features.shape[0], 2), dtype=np.float32)
    return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(features)


def subcluster_label_series(labels: np.ndarray, prefix: str) -> list[str]:
    return [f"{prefix}{int(label)}" for label in labels]


def finite_sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=axis)
    std = np.nanstd(values, axis=axis, ddof=1)
    return np.divide(std, np.sqrt(counts), out=np.zeros_like(std, dtype=np.float64), where=counts > 1)


def plot_spatial_result(
    *,
    cluster_id: str,
    tracks: pd.DataFrame,
    features: np.ndarray,
    maps: np.ndarray,
    labels: np.ndarray,
    null_scores: np.ndarray,
    observed_silhouette: float,
    p_value: float,
    out_path: Path,
) -> None:
    xy = pca_xy(features)
    unique_labels = np.unique(labels)
    n_sub = len(unique_labels)
    mean_maps = [maps[labels == label].mean(axis=0) for label in unique_labels]
    if n_sub == 2:
        diff_map = mean_maps[1] - mean_maps[0]
        diff_title = f"{cluster_id} S1 - S0"
    else:
        stack = np.stack(mean_maps, axis=0)
        diff_map = stack.max(axis=0) - stack.min(axis=0)
        diff_title = f"{cluster_id} max-min"

    n_cols = max(3, n_sub + 1)
    fig = plt.figure(figsize=(4.0 * n_cols, 10.5))
    gs = fig.add_gridspec(3, n_cols, height_ratios=[1.05, 1.1, 1.0])

    ax = fig.add_subplot(gs[0, 0:2])
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=60, cmap="tab10", edgecolor="black", linewidth=0.4)
    for i, row in tracks.iterrows():
        ax.text(xy[i, 0], xy[i, 1], str(row["track_id"]), fontsize=6, alpha=0.7)
    ax.set_title(f"{cluster_id} spatial PCA/KMeans")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="spatial subcluster")

    ax = fig.add_subplot(gs[0, 2:])
    finite_null = null_scores[np.isfinite(null_scores)]
    if finite_null.size:
        ax.hist(finite_null, bins=24, color="0.75", edgecolor="white")
    ax.axvline(observed_silhouette, color="crimson", lw=2, label=f"observed={observed_silhouette:.3f}")
    ax.set_title(f"Spatial silhouette null, p={p_value:.3g}")
    ax.set_xlabel("best silhouette after per-track spatial-bin permutation")
    ax.set_ylabel("permutations")
    ax.legend()

    vmax = float(np.nanpercentile(np.concatenate([m.ravel() for m in mean_maps]), 99.0))
    for col, (label, mean_map) in enumerate(zip(unique_labels, mean_maps)):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(mean_map, origin="lower", interpolation="none", cmap="viridis", vmin=0, vmax=vmax)
        n = int(np.sum(labels == label))
        ax.set_title(f"S{int(label)} mean occupancy n={n}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[1, n_sub])
    vmax_diff = float(np.nanpercentile(np.abs(diff_map), 99.0)) if np.isfinite(diff_map).any() else 1.0
    im = ax.imshow(diff_map, origin="lower", interpolation="none", cmap="coolwarm", vmin=-vmax_diff, vmax=vmax_diff)
    ax.set_title(diff_title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for extra_col in range(n_sub + 1, n_cols):
        fig.add_subplot(gs[1, extra_col]).axis("off")

    ax = fig.add_subplot(gs[2, :])
    order = np.argsort(labels, kind="mergesort")
    image = np.asarray([maps[i].ravel() for i in order])
    image = np.sqrt(np.clip(image, 0, None))
    ax.imshow(image, aspect="auto", interpolation="none", cmap="viridis")
    ax.set_title("Tracks x spatial bins, sorted by spatial subcluster")
    ax.set_xlabel("coarse spatial bin")
    ax.set_ylabel("track")

    fig.suptitle(f"Spatial subcluster test within {cluster_id}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_temporal_result(
    *,
    cluster_id: str,
    tracks: pd.DataFrame,
    features: np.ndarray,
    speed_profiles: np.ndarray,
    hours: np.ndarray,
    labels: np.ndarray,
    null_scores: np.ndarray,
    observed_silhouette: float,
    p_value: float,
    out_path: Path,
) -> None:
    xy = pca_xy(features)
    unique_labels = np.unique(labels)
    speed_profiles = np.asarray(speed_profiles, dtype=np.float32)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.1, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=60, cmap="tab10", edgecolor="black", linewidth=0.4)
    for i, row in tracks.iterrows():
        ax.text(xy[i, 0], xy[i, 1], str(row["track_id"]), fontsize=6, alpha=0.7)
    ax.set_title(f"{cluster_id} temporal PCA/KMeans")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="temporal subcluster")

    ax = fig.add_subplot(gs[0, 1:])
    finite_null = null_scores[np.isfinite(null_scores)]
    if finite_null.size:
        ax.hist(finite_null, bins=24, color="0.75", edgecolor="white")
    ax.axvline(observed_silhouette, color="crimson", lw=2, label=f"observed={observed_silhouette:.3f}")
    ax.set_title(f"Temporal silhouette null, p={p_value:.3g}")
    ax.set_xlabel("best silhouette after per-track time-bin permutation")
    ax.set_ylabel("permutations")
    ax.legend()

    ax = fig.add_subplot(gs[1, :])
    for label in unique_labels:
        sub = speed_profiles[labels == label]
        mean = np.nanmean(sub, axis=0)
        sem = finite_sem(sub, axis=0)
        ax.plot(hours, mean, marker="o", lw=2, label=f"S{int(label)} n={len(sub)}")
        ax.fill_between(hours, mean - sem, mean + sem, alpha=0.2)
    ax.set_xlabel("hours since lights on")
    ax.set_ylabel("mean speed (mm/s)")
    ax.set_title("Time-of-day speed by temporal subcluster")
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = fig.add_subplot(gs[2, :])
    order = np.lexsort((tracks["track_id"].to_numpy(), labels))
    image = speed_profiles[order]
    vmax = float(np.nanpercentile(image, 99.0)) if np.isfinite(image).any() else None
    im = ax.imshow(
        image,
        aspect="auto",
        interpolation="none",
        cmap="magma",
        extent=[0, 24, len(order), 0],
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title("Tracks x time-of-day bins, sorted by temporal subcluster")
    ax.set_xlabel("hours since lights on")
    ax.set_ylabel("track")
    fig.colorbar(im, ax=ax, label="mean speed (mm/s)")

    fig.suptitle(f"Time-of-day speed subcluster test within {cluster_id}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_joint_result(
    *,
    cluster_id: str,
    tracks: pd.DataFrame,
    features: np.ndarray,
    maps: np.ndarray,
    speed_profiles: np.ndarray,
    hours: np.ndarray,
    labels: np.ndarray,
    null_scores: np.ndarray,
    observed_silhouette: float,
    p_value: float,
    out_path: Path,
) -> None:
    xy = pca_xy(features)
    unique_labels = np.unique(labels)
    n_sub = len(unique_labels)
    mean_maps = [maps[labels == label].mean(axis=0) for label in unique_labels]
    if n_sub == 2:
        diff_map = mean_maps[1] - mean_maps[0]
        diff_title = f"{cluster_id} S1 - S0"
    else:
        stack = np.stack(mean_maps, axis=0)
        diff_map = stack.max(axis=0) - stack.min(axis=0)
        diff_title = f"{cluster_id} spatial max-min"

    n_cols = max(3, n_sub + 1)
    fig = plt.figure(figsize=(max(16, 4.0 * n_cols), 12))
    gs = fig.add_gridspec(3, n_cols, height_ratios=[1.0, 1.05, 1.05])

    ax = fig.add_subplot(gs[0, 0:2])
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=60, cmap="tab10", edgecolor="black", linewidth=0.4)
    for i, row in tracks.iterrows():
        ax.text(xy[i, 0], xy[i, 1], str(row["track_id"]), fontsize=6, alpha=0.7)
    ax.set_title(f"{cluster_id} joint spatial + temporal PCA/KMeans")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="joint subcluster")

    ax = fig.add_subplot(gs[0, 2:])
    finite_null = null_scores[np.isfinite(null_scores)]
    if finite_null.size:
        ax.hist(finite_null, bins=24, color="0.75", edgecolor="white")
    ax.axvline(observed_silhouette, color="crimson", lw=2, label=f"observed={observed_silhouette:.3f}")
    ax.set_title(f"Joint silhouette null, p={p_value:.3g}")
    ax.set_xlabel("best silhouette after within-block feature permutation")
    ax.set_ylabel("permutations")
    ax.legend()

    ax = fig.add_subplot(gs[1, :])
    for label in unique_labels:
        sub = speed_profiles[labels == label]
        mean = np.nanmean(sub, axis=0)
        sem = finite_sem(sub, axis=0)
        ax.plot(hours, mean, marker="o", lw=2, label=f"S{int(label)} n={len(sub)}")
        ax.fill_between(hours, mean - sem, mean + sem, alpha=0.2)
    ax.set_xlabel("hours since lights on")
    ax.set_ylabel("mean speed (mm/s)")
    ax.set_title("Time-of-day speed by joint subcluster")
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.25)
    ax.legend()

    vmax = float(np.nanpercentile(np.concatenate([m.ravel() for m in mean_maps]), 99.0))
    for col, (label, mean_map) in enumerate(zip(unique_labels, mean_maps)):
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(mean_map, origin="lower", interpolation="none", cmap="viridis", vmin=0, vmax=vmax)
        n = int(np.sum(labels == label))
        ax.set_title(f"S{int(label)} mean occupancy n={n}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[2, n_sub])
    vmax_diff = float(np.nanpercentile(np.abs(diff_map), 99.0)) if np.isfinite(diff_map).any() else 1.0
    im = ax.imshow(diff_map, origin="lower", interpolation="none", cmap="coolwarm", vmin=-vmax_diff, vmax=vmax_diff)
    ax.set_title(diff_title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for extra_col in range(n_sub + 1, n_cols):
        fig.add_subplot(gs[2, extra_col]).axis("off")

    fig.suptitle(f"Joint spatial + time-of-day speed subcluster test within {cluster_id}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    feature_types = ["spatial", "temporal_speed", "joint_spatial_temporal"]
    feature_titles = {
        "spatial": "spatial occupancy",
        "temporal_speed": "speed by time of day",
        "joint_spatial_temporal": "joint spatial + speed",
    }
    fig, axes = plt.subplots(1, len(feature_types), figsize=(20, 5), sharey=True)
    for ax, feature_type in zip(axes, feature_types):
        sub = summary[summary["feature_type"] == feature_type].copy()
        sub = sub.sort_values(["side", "cluster_id"], kind="mergesort")
        x = np.arange(len(sub))
        colors = np.where(sub["p_value"] < 0.05, "tab:red", "tab:blue")
        ax.bar(x, sub["silhouette"], color=colors, alpha=0.85)
        if not sub.empty:
            ax.axhline(sub["null_p95"].median(), color="0.35", linestyle="--", lw=1, label="median null 95th pct")
        labels = [
            f"{row.cluster_id}\nk={int(row.best_k)}\np={row.p_value:.2g}"
            for row in sub.itertuples(index=False)
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(feature_titles[feature_type])
        ax.set_ylabel("best silhouette")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Within-cluster subcluster evidence; red p<0.05 vs feature-permutation null")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_subcluster_tests(features: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tracks = features["tracks"].copy()
    spatial_features = features["spatial_features"]
    spatial_maps = features["spatial_maps"]
    temporal_features = features["temporal_features"]
    temporal_speed = features["temporal_speed"]
    temporal_hours = features["temporal_hours"]

    plot_dir = OUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    assignment_rows = []
    for cluster_id, cluster_tracks in tracks.groupby("cluster_id", sort=True):
        idx = cluster_tracks.index.to_numpy()
        n_tracks = len(idx)
        if n_tracks < MIN_CLUSTER_SIZE:
            print(f"skip {cluster_id}: n={n_tracks} < MIN_CLUSTER_SIZE={MIN_CLUSTER_SIZE}")
            continue

        cluster_spatial_features = spatial_features[idx]
        cluster_temporal_features = temporal_features[idx]
        cluster_joint_features, joint_split_at = joint_spatial_temporal_features(
            cluster_spatial_features,
            cluster_temporal_features,
        )

        for feature_type, matrix, kind, split_at in [
            ("spatial", cluster_spatial_features, "spatial", None),
            ("temporal_speed", cluster_temporal_features, "temporal", None),
            ("joint_spatial_temporal", cluster_joint_features, "joint", joint_split_at),
        ]:
            max_k = min(MAX_SUBCLUSTERS, max(2, n_tracks // MIN_SUBCLUSTER_SIZE))
            result = best_kmeans_partition(
                matrix,
                max_k=max_k,
                min_subcluster_size=MIN_SUBCLUSTER_SIZE,
                random_state=RANDOM_STATE,
                n_init=30,
            )
            null_scores = null_silhouette_distribution(
                matrix,
                kind=kind,
                split_at=split_at,
                max_k=max_k,
                min_subcluster_size=MIN_SUBCLUSTER_SIZE,
                n_permutations=N_PERMUTATIONS,
                random_state=RANDOM_STATE + 101 * (len(summary_rows) + 1),
            )
            p_value = silhouette_p_value(float(result["silhouette"]), null_scores)
            finite_null = null_scores[np.isfinite(null_scores)]
            summary_rows.append(
                {
                    "side": cluster_tracks["side"].iloc[0],
                    "cluster_id": cluster_id,
                    "feature_type": feature_type,
                    "n_tracks": n_tracks,
                    "best_k": int(result["k"]),
                    "subcluster_sizes": ",".join(str(v) for v in result["counts"]),
                    "silhouette": float(result["silhouette"]) if np.isfinite(result["silhouette"]) else np.nan,
                    "null_mean": float(np.nanmean(finite_null)) if finite_null.size else np.nan,
                    "null_p95": float(np.nanpercentile(finite_null, 95)) if finite_null.size else np.nan,
                    "p_value": p_value,
                    "n_permutations": int(finite_null.size),
                }
            )

            labels = np.asarray(result["labels"], dtype=int)
            for local_i, (_, track_row) in enumerate(cluster_tracks.iterrows()):
                assignment_rows.append(
                    {
                        "side": track_row["side"],
                        "cluster_id": cluster_id,
                        "track_id": int(track_row["track_id"]),
                        "track_name": track_row["track_name"],
                        "feature_type": feature_type,
                        "subcluster": int(labels[local_i]),
                    }
                )

            print(
                f"{cluster_id} {feature_type}: n={n_tracks}, k={result['k']}, "
                f"silhouette={result['silhouette']:.3f}, p={p_value:.3g}, sizes={result['counts']}"
            )

            if feature_type == "spatial":
                plot_spatial_result(
                    cluster_id=cluster_id,
                    tracks=cluster_tracks.reset_index(drop=True),
                    features=matrix,
                    maps=spatial_maps[idx],
                    labels=labels,
                    null_scores=null_scores,
                    observed_silhouette=float(result["silhouette"]),
                    p_value=p_value,
                    out_path=plot_dir / f"{cluster_id}_spatial_subclusters.png",
                )
            elif feature_type == "temporal_speed":
                plot_temporal_result(
                    cluster_id=cluster_id,
                    tracks=cluster_tracks.reset_index(drop=True),
                    features=matrix,
                    speed_profiles=temporal_speed[idx],
                    hours=temporal_hours,
                    labels=labels,
                    null_scores=null_scores,
                    observed_silhouette=float(result["silhouette"]),
                    p_value=p_value,
                    out_path=plot_dir / f"{cluster_id}_temporal_speed_subclusters.png",
                )
            else:
                plot_joint_result(
                    cluster_id=cluster_id,
                    tracks=cluster_tracks.reset_index(drop=True),
                    features=matrix,
                    maps=spatial_maps[idx],
                    speed_profiles=temporal_speed[idx],
                    hours=temporal_hours,
                    labels=labels,
                    null_scores=null_scores,
                    observed_silhouette=float(result["silhouette"]),
                    p_value=p_value,
                    out_path=plot_dir / f"{cluster_id}_joint_spatial_temporal_subclusters.png",
                )

    summary = pd.DataFrame(summary_rows)
    assignments = pd.DataFrame(assignment_rows)
    summary.to_csv(OUT_ROOT / "subcluster_test_summary.csv", index=False)
    assignments.to_csv(OUT_ROOT / "track_subcluster_assignments.csv", index=False)
    plot_summary(summary, plot_dir / "subcluster_test_summary.png")
    return summary, assignments


# %%
def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tracks = load_clustered_tracks()
    print(f"Loaded {len(tracks)} clustered tracks")
    print(tracks.groupby(["side", "cluster_id"])["track_name"].count().rename("n_tracks"))
    features = build_or_load_features(tracks)
    summary, _assignments = run_subcluster_tests(features)
    print(f"Wrote summary: {OUT_ROOT / 'subcluster_test_summary.csv'}")
    print(f"Wrote assignments: {OUT_ROOT / 'track_subcluster_assignments.csv'}")
    print(f"Wrote plots: {OUT_ROOT / 'plots'}")
    print(summary.sort_values(["feature_type", "p_value", "cluster_id"]).to_string(index=False))


if __name__ == "__main__":
    main()
