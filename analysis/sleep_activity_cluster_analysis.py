# %%
# VS Code/Jupyter interactive script:
# spatiotemporal activity clusters, predicted sleep timing, and
# within-cluster work/sleep shifts.
try:
    get_ipython().run_line_magic("matplotlib", "qt")  # type: ignore[name-defined]
except Exception:
    pass

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = print

repo_root = Path.cwd().resolve()
for candidate in [repo_root, *repo_root.parents]:
    if (candidate / "analysis" / "sleep_analysis_utils.py").exists():
        repo_root = candidate
        break
else:
    raise FileNotFoundError("Could not find analysis utilities from the current working directory")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import analysis.colony_speed_utils as cs
import analysis.grid_occupancy_utils as go
import analysis.interaction_analysis_utils as ia
import analysis.sleep_analysis_utils as sleep_utils
from analysis.figure_saving import install_auto_savefig

importlib.reload(cs)
importlib.reload(go)
importlib.reload(ia)
importlib.reload(sleep_utils)


# %%
# Settings.
DATASET_ROOT = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02")
INTERACTION_ROOT = DATASET_ROOT / "interactions"
TRACKS_ROOT = DATASET_ROOT / "tracks"
GRID_ROOT = DATASET_ROOT / "stitched" / "grid_occupancy_histograms"
SPEED_ROOT = DATASET_ROOT / "stitched" / "speed_vectors"
SLEEP_PREDICTIONS_ROOT = DATASET_ROOT / "stitched" / "sleep_predictions"

SIDE = "left"  # Run left/right separately unless the grid edges are known to match.
CHUNKS = "all"
MAX_CHUNKS = None
FPS = 24.0
MM_PER_PX = 0.016
MIN_PRESENT_FRAC = 0.40

LIGHT_ON_HOUR = 5.5
TIME_BIN_MINUTES = 30.0

ACTIVITY_CLUSTER_COL = "activity_cluster"
SPATIAL_FEATURE_TRANSFORM = "sqrt"
SPATIAL_FEATURE_WEIGHT = 1.0
TEMPORAL_FEATURE_WEIGHT = 1.0
ACTIVITY_ACTIVE_SPEED_THRESHOLD_MM_S = 0.5
ACTIVITY_QUIET_SPEED_THRESHOLD_MM_S = 0.1
ACTIVITY_PCA_COMPONENTS = 40
ACTIVITY_CLUSTER_METHOD = "leiden"  # "leiden" or "kmeans".
ACTIVITY_N_NEIGHBORS = 15
ACTIVITY_LEIDEN_RESOLUTION = 0.9
ACTIVITY_FALLBACK_N_CLUSTERS = 6
ACTIVITY_RANDOM_STATE = 0

MAX_PAIRWISE_ANTS_PER_CLUSTER = 35
MAX_PAIRWISE_LAG_HOURS = 8.0
SLEEP_CROSS_CORR_MIN_OVERLAP_BINS = 8
SLEEP_CROSS_CORR_USE_CIRCULAR_LAGS = True
SLEEP_ACTIVITY_MAX_LAG_HOURS = 12.0
SLEEP_TIME_MODULATION_PERMUTATIONS = 500
SLEEP_TIME_MODULATION_RANDOM_STATE = ACTIVITY_RANDOM_STATE

SHIFTED_SLEEP_MIN_BEST_CORR = 0.60
SHIFTED_SLEEP_MAX_ZERO_LAG_CORR = 0.25
SHIFTED_SLEEP_MIN_CORR_GAIN = 0.30
SHIFTED_SLEEP_MIN_ABS_LAG_HOURS = 1.0
SHIFTED_SLEEP_EXAMPLE_PAIRS_PER_CLUSTER = 4

ELAPSED_SLEEP_BIN_MINUTES = TIME_BIN_MINUTES
SLEEP_REPETITION_MAX_LAG_HOURS = 72.0
SLEEP_REPETITION_MIN_LAG_HOURS = 1.0
SLEEP_REPETITION_MIN_OVERLAP_BINS = 8
SLEEP_REPETITION_TOP_LAGS_PER_CLUSTER = 8
SLEEP_REPETITION_TOP_ANTS_PER_CLUSTER = 4

INTERACTION_WAKE_PRE_SECONDS = 5 * 60.0
INTERACTION_WAKE_POST_SECONDS = 15 * 60.0
INTERACTION_WAKE_PRE_SLEEP_MIN_FRACTION = 0.50
INTERACTION_WAKE_POST_WAKE_MAX_SLEEP_FRACTION = 0.20
INTERACTION_WAKE_CONTROL_REPLICATES = 2
INTERACTION_WAKE_CONTROL_SEARCH_TRIES = 300
INTERACTION_WAKE_CONTROL_EXCLUDE_SECONDS = 60.0
INTERACTION_WAKE_MAX_EVENTS_PER_TRACK = 200
INTERACTION_WAKE_CURVE_RADIUS_SECONDS = 30 * 60.0
INTERACTION_WAKE_CURVE_BIN_SECONDS = 60.0
INTERACTION_WAKE_RANDOM_STATE = ACTIVITY_RANDOM_STATE
INTERACTION_EVENT_GAP_SECONDS = 2.0
INTERACTION_WAKE_EVENT_ROLE = "body"  # Sleeping ant as receiver/body: another ant's antenna contacts it.

OUTPUT_ROOT = DATASET_ROOT / "analysis_outputs" / "sleep_activity_cluster_analysis"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
SAVE_FIGURES = True
FIGURE_DPI = 180
install_auto_savefig(
    FIGURE_ROOT / SIDE,
    prefix=f"sleep_activity_cluster_analysis_{SIDE}",
    dpi=FIGURE_DPI,
    enabled=SAVE_FIGURES,
)
SAVE_SLEEP_BOUTS = True
SLEEP_BOUTS_PARQUET = OUTPUT_ROOT / f"{SIDE}_predicted_sleep_bouts.parquet"

SLEEP_END_TRIGGER_TRACK_ID = None  # Set to an int track_id for a specific ant; None picks the ant with most sleep bouts.
SLEEP_END_TRIGGER_TRACK_NAME = None
SLEEP_END_TRIGGER_PRE_SECONDS = 10 * 60.0
SLEEP_END_TRIGGER_POST_SECONDS = 10 * 60.0
SLEEP_END_TRIGGER_RATE_WINDOW_SECONDS = 30.0
SLEEP_END_TRIGGER_RATE_STEP_SECONDS = 1.0
SLEEP_END_TRIGGER_INTERACTION_EVENT_GAP_SECONDS = INTERACTION_EVENT_GAP_SECONDS
SLEEP_END_TRIGGER_INTERACTION_ROLE = "body"
SLEEP_END_TRIGGER_MAX_BOUTS = 150
SLEEP_END_TRIGGER_SORT_BY = "end_frame"  # "end_frame", "duration_desc", "post_rate_desc".
SLEEP_END_TRIGGER_PLOT_ORDER_BY = "duration_desc"  # "duration_desc", "duration_asc", "end_frame", "post_rate_desc".
SLEEP_END_TRIGGER_POST_RATE_SECONDS = 120.0


# %%
# Helper functions for the new analysis layer.
def time_bin_centers_hours(bin_minutes: float) -> np.ndarray:
    n_bins = int(round(24 * 60 / float(bin_minutes)))
    width_h = 24.0 / n_bins
    return (np.arange(n_bins, dtype=np.float64) + 0.5) * width_h


def frame_time_bin(
    frames: np.ndarray,
    *,
    fps: float,
    recording_start_clock_seconds: float,
    light_on_hour: float,
    bin_minutes: float,
) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float64)
    n_bins = int(round(24 * 60 / float(bin_minutes)))
    width_h = 24.0 / n_bins
    hours_since_light_on = (
        (float(recording_start_clock_seconds) + frames / float(fps)) / 3600.0
        - float(light_on_hour)
    ) % 24.0
    return np.floor(hours_since_light_on / width_h).astype(np.int64).clip(0, n_bins - 1)


def zscore_feature_block(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    if matrix.size == 0:
        return matrix.astype(np.float32, copy=True), np.zeros(matrix.shape[1], dtype=bool)
    imputed = SimpleImputer(strategy="median").fit_transform(matrix)
    finite_var = np.nanstd(imputed, axis=0) > 1e-12
    if not finite_var.any():
        return np.zeros((matrix.shape[0], 0), dtype=np.float32), finite_var
    scaled = StandardScaler().fit_transform(imputed[:, finite_var])
    return scaled.astype(np.float32, copy=False), finite_var


def activity_time_profile_matrix(
    tracks: pd.DataFrame,
    *,
    recording_start_clock_seconds: float,
    light_on_hour: float,
    bin_minutes: float,
    active_speed_threshold_mm_s: float,
    quiet_speed_threshold_mm_s: float,
    analysis_frame_start: int | None = None,
    analysis_frame_stop: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    n_bins = int(round(24 * 60 / float(bin_minutes)))
    centers_h = time_bin_centers_hours(bin_minutes)
    feature_rows = []
    long_rows = []

    for i, row in tracks.reset_index(drop=True).iterrows():
        if i == 0 or i == len(tracks) - 1 or (i + 1) % 25 == 0:
            print(f"activity profile: {i + 1}/{len(tracks)} {row['track_name']}")
        speed = np.load(row["speed_path"], mmap_mode="r")
        frame_min = int(row["speed_frame_min"])
        fps = float(row["speed_fps"])
        local_frames = np.arange(len(speed), dtype=np.int64)
        global_frames = frame_min + local_frames
        valid = np.isfinite(speed)
        if analysis_frame_start is not None:
            valid &= global_frames >= int(analysis_frame_start)
        if analysis_frame_stop is not None:
            valid &= global_frames < int(analysis_frame_stop)

        mean_speed = np.full(n_bins, np.nan, dtype=np.float32)
        active_frac = np.full(n_bins, np.nan, dtype=np.float32)
        quiet_frac = np.full(n_bins, np.nan, dtype=np.float32)
        n_valid = np.zeros(n_bins, dtype=np.int64)
        if valid.any():
            valid_frames = global_frames[valid]
            valid_speed = np.asarray(speed[valid], dtype=np.float64)
            bins = frame_time_bin(
                valid_frames,
                fps=fps,
                recording_start_clock_seconds=recording_start_clock_seconds,
                light_on_hour=light_on_hour,
                bin_minutes=bin_minutes,
            )
            count = np.bincount(bins, minlength=n_bins)
            speed_sum = np.bincount(bins, weights=valid_speed, minlength=n_bins)
            active_sum = np.bincount(
                bins,
                weights=(valid_speed > float(active_speed_threshold_mm_s)).astype(np.float64),
                minlength=n_bins,
            )
            quiet_sum = np.bincount(
                bins,
                weights=(valid_speed <= float(quiet_speed_threshold_mm_s)).astype(np.float64),
                minlength=n_bins,
            )
            keep = count > 0
            mean_speed[keep] = (speed_sum[keep] / count[keep]).astype(np.float32)
            active_frac[keep] = (active_sum[keep] / count[keep]).astype(np.float32)
            quiet_frac[keep] = (quiet_sum[keep] / count[keep]).astype(np.float32)
            n_valid = count.astype(np.int64)

        feature_rows.append(np.concatenate([mean_speed, active_frac, quiet_frac]))
        for time_bin in range(n_bins):
            long_rows.append(
                {
                    "track_name": row["track_name"],
                    "track_id": row["track_id"],
                    "side": row["side"],
                    "time_bin": int(time_bin),
                    "hours_since_light_on": float(centers_h[time_bin]),
                    "mean_speed_mm_s": float(mean_speed[time_bin]) if np.isfinite(mean_speed[time_bin]) else np.nan,
                    "active_fraction": float(active_frac[time_bin]) if np.isfinite(active_frac[time_bin]) else np.nan,
                    "quiet_fraction": float(quiet_frac[time_bin]) if np.isfinite(quiet_frac[time_bin]) else np.nan,
                    "n_valid_frames": int(n_valid[time_bin]),
                    "valid_duration_seconds": float(n_valid[time_bin] / fps),
                }
            )

    return np.vstack(feature_rows).astype(np.float32), pd.DataFrame(long_rows)


def build_spatiotemporal_activity_clusters(
    tracks: pd.DataFrame,
    *,
    recording_start_clock_seconds: float,
    analysis_frame_start: int | None,
    analysis_frame_stop: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    spatial_features, chosen, _x_edges, _y_edges = go.build_histogram_matrix(
        tracks,
        side=None,
        transform=SPATIAL_FEATURE_TRANSFORM,
    )
    temporal_features, track_activity_long = activity_time_profile_matrix(
        chosen,
        recording_start_clock_seconds=recording_start_clock_seconds,
        light_on_hour=LIGHT_ON_HOUR,
        bin_minutes=TIME_BIN_MINUTES,
        active_speed_threshold_mm_s=ACTIVITY_ACTIVE_SPEED_THRESHOLD_MM_S,
        quiet_speed_threshold_mm_s=ACTIVITY_QUIET_SPEED_THRESHOLD_MM_S,
        analysis_frame_start=analysis_frame_start,
        analysis_frame_stop=analysis_frame_stop,
    )

    spatial_z, spatial_keep = zscore_feature_block(spatial_features)
    temporal_z, temporal_keep = zscore_feature_block(temporal_features)
    if spatial_z.shape[1]:
        spatial_z = spatial_z / np.sqrt(float(spatial_z.shape[1]))
    if temporal_z.shape[1]:
        temporal_z = temporal_z / np.sqrt(float(temporal_z.shape[1]))
    joint = np.concatenate(
        [
            spatial_z * float(SPATIAL_FEATURE_WEIGHT),
            temporal_z * float(TEMPORAL_FEATURE_WEIGHT),
        ],
        axis=1,
    )
    if joint.shape[1] == 0:
        raise ValueError("No finite spatiotemporal activity features were available")

    n_components = min(int(ACTIVITY_PCA_COMPONENTS), joint.shape[0] - 1, joint.shape[1])
    graph_features = joint
    if n_components >= 2 and joint.shape[1] > n_components:
        graph_features = PCA(n_components=n_components, random_state=ACTIVITY_RANDOM_STATE).fit_transform(joint)

    used_method = ACTIVITY_CLUSTER_METHOD
    try:
        if ACTIVITY_CLUSTER_METHOD == "leiden":
            go.check_clustering_dependencies()
            labels = go.leiden_labels(
                graph_features,
                n_neighbors=ACTIVITY_N_NEIGHBORS,
                resolution=ACTIVITY_LEIDEN_RESOLUTION,
                random_state=ACTIVITY_RANDOM_STATE,
            )
            embedding = go.umap_embedding(
                graph_features,
                n_neighbors=ACTIVITY_N_NEIGHBORS,
                random_state=ACTIVITY_RANDOM_STATE,
            )
        else:
            raise ImportError("Using configured kmeans fallback")
    except Exception as exc:
        used_method = "kmeans"
        n_clusters = min(int(ACTIVITY_FALLBACK_N_CLUSTERS), max(1, len(chosen)))
        print(f"Activity clustering fallback to kmeans because Leiden/UMAP was unavailable: {exc}")
        labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=ACTIVITY_RANDOM_STATE).fit_predict(graph_features)
        n_embed = min(2, graph_features.shape[0], graph_features.shape[1])
        if n_embed >= 2:
            embedding = PCA(n_components=2, random_state=ACTIVITY_RANDOM_STATE).fit_transform(graph_features)
        else:
            embedding = np.column_stack([np.arange(len(chosen), dtype=float), np.zeros(len(chosen), dtype=float)])

    cluster_table = chosen.copy()
    cluster_table[ACTIVITY_CLUSTER_COL] = labels.astype(int)
    cluster_table["umap1"] = embedding[:, 0]
    cluster_table["umap2"] = embedding[:, 1]
    cluster_table["activity_cluster_method"] = used_method
    cluster_table["n_spatial_features_used"] = int(spatial_keep.sum())
    cluster_table["n_temporal_features_used"] = int(temporal_keep.sum())
    track_activity_long = track_activity_long.merge(
        cluster_table[["track_name", ACTIVITY_CLUSTER_COL]],
        on="track_name",
        how="left",
        validate="many_to_one",
    )
    return cluster_table, track_activity_long, chosen, joint, graph_features


def plot_activity_cluster_time_profiles(track_activity_long: pd.DataFrame) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    summary = (
        track_activity_long.groupby([ACTIVITY_CLUSTER_COL, "time_bin"], as_index=False)
        .agg(
            hours_since_light_on=("hours_since_light_on", "first"),
            mean_speed_mm_s=("mean_speed_mm_s", "mean"),
            active_fraction=("active_fraction", "mean"),
            quiet_fraction=("quiet_fraction", "mean"),
            n_tracks=("track_name", "nunique"),
        )
        .sort_values([ACTIVITY_CLUSTER_COL, "time_bin"], kind="mergesort")
    )

    clusters = sorted(summary[ACTIVITY_CLUSTER_COL].dropna().unique())
    colors = plt.get_cmap("tab20", max(len(clusters), 1))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for i, cluster in enumerate(clusters):
        group = summary[summary[ACTIVITY_CLUSTER_COL] == cluster]
        n_tracks = int(track_activity_long.loc[track_activity_long[ACTIVITY_CLUSTER_COL] == cluster, "track_name"].nunique())
        axes[0].plot(group["hours_since_light_on"], group["active_fraction"], color=colors(i), lw=1.8, label=f"c{cluster} n={n_tracks}")
        axes[1].plot(group["hours_since_light_on"], group["quiet_fraction"], color=colors(i), lw=1.8, label=f"c{cluster} n={n_tracks}")
    axes[0].set_ylabel(f"fraction speed > {ACTIVITY_ACTIVE_SPEED_THRESHOLD_MM_S:g} mm/s")
    axes[1].set_ylabel(f"fraction speed <= {ACTIVITY_QUIET_SPEED_THRESHOLD_MM_S:g} mm/s")
    axes[1].set_xlabel(f"Hours since lights on at {LIGHT_ON_HOUR:g}:00")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncols=2)
    fig.suptitle("Activity-cluster time-of-day profiles")
    fig.tight_layout()
    plt.show()
    return summary


def add_interaction_onset_aliases(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    aliases = {
        "n_interactions_total": "n_interaction_onsets",
        "n_interactions_as_antenna": "n_interaction_onsets_as_antenna",
        "n_interactions_as_body": "n_interaction_onsets_as_body",
    }
    for source, target in aliases.items():
        if source in out.columns and target not in out.columns:
            out[target] = out[source]
    return out


def load_activity_cluster_sleep_predictions(
    activity_tracks: pd.DataFrame,
    activity_cluster_table: pd.DataFrame,
    counts_by_track: dict[int, pd.DataFrame],
    *,
    analysis_frame_start: int,
    analysis_frame_stop: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sleep_prediction_tracks = sleep_utils.attach_sleep_predictions_to_tracks(
        activity_tracks,
        SLEEP_PREDICTIONS_ROOT,
        require_all=False,
    )
    clustered_keys = activity_cluster_table[["track_name", "track_id", "side"]].drop_duplicates()
    sleep_prediction_tracks = sleep_prediction_tracks.merge(
        clustered_keys,
        on=["track_name", "track_id", "side"],
        how="inner",
        validate="one_to_one",
    )
    sleep_prediction_tracks = sleep_prediction_tracks[
        sleep_prediction_tracks["has_sleep_predictions"].astype(bool)
    ].reset_index(drop=True)
    if sleep_prediction_tracks.empty:
        raise ValueError(f"No sleep prediction outputs matched activity-cluster tracks under {SLEEP_PREDICTIONS_ROOT}")

    cluster_cols = [
        col
        for col in [
            "track_name",
            "track_id",
            "side",
            ACTIVITY_CLUSTER_COL,
            "activity_cluster_id",
            "umap1",
            "umap2",
        ]
        if col in activity_cluster_table.columns
    ]
    cluster_lookup = activity_cluster_table[cluster_cols].drop_duplicates(["track_name", "track_id", "side"])
    sleep_prediction_tracks = sleep_prediction_tracks.merge(
        cluster_lookup,
        on=["track_name", "track_id", "side"],
        how="left",
        validate="one_to_one",
    )
    predicted_sleep_bouts = sleep_utils.load_predicted_sleep_bouts(
        sleep_prediction_tracks,
        counts_by_track,
        frame_start=analysis_frame_start,
        frame_stop=analysis_frame_stop,
    )
    if predicted_sleep_bouts.empty:
        return sleep_prediction_tracks, pd.DataFrame()
    predicted_sleep_bouts = add_interaction_onset_aliases(predicted_sleep_bouts)
    predicted_sleep_bouts = predicted_sleep_bouts.merge(
        cluster_lookup,
        on=["track_name", "track_id", "side"],
        how="left",
        validate="many_to_one",
    )
    return sleep_prediction_tracks, predicted_sleep_bouts.reset_index(drop=True)


def split_bout_durations_to_time_bins(
    bouts: pd.DataFrame,
    *,
    recording_start_clock_seconds: float,
    fps: float,
    bin_minutes: float,
) -> pd.DataFrame:
    if bouts.empty:
        return pd.DataFrame()
    n_bins = int(round(24 * 60 / float(bin_minutes)))
    bin_width_s = float(bin_minutes) * 60.0
    centers_h = time_bin_centers_hours(bin_minutes)
    rows = []
    for bout in bouts.itertuples(index=False):
        start_s = float(bout.bout_start_frame) / float(fps)
        end_s = float(int(bout.bout_end_frame) + 1) / float(fps)
        first_elapsed_bin = int(np.floor(start_s / bin_width_s))
        last_elapsed_bin = int(np.floor((end_s - 1e-9) / bin_width_s))
        for elapsed_bin in range(first_elapsed_bin, last_elapsed_bin + 1):
            seg_start = max(start_s, elapsed_bin * bin_width_s)
            seg_end = min(end_s, (elapsed_bin + 1) * bin_width_s)
            if seg_end <= seg_start:
                continue
            mid_frame = ((seg_start + seg_end) * 0.5) * float(fps)
            time_bin = int(
                frame_time_bin(
                    np.asarray([mid_frame]),
                    fps=fps,
                    recording_start_clock_seconds=recording_start_clock_seconds,
                    light_on_hour=LIGHT_ON_HOUR,
                    bin_minutes=bin_minutes,
                )[0]
            )
            rows.append(
                {
                    "track_name": getattr(bout, "track_name", None),
                    "track_id": getattr(bout, "track_id", np.nan),
                    "side": getattr(bout, "side", SIDE),
                    ACTIVITY_CLUSTER_COL: getattr(bout, ACTIVITY_CLUSTER_COL, np.nan),
                    "classifier_bin": getattr(bout, "classifier_bin", None),
                    "time_bin": time_bin,
                    "hours_since_light_on": float(centers_h[time_bin]),
                    "sleep_duration_seconds": float(seg_end - seg_start),
                    "n_interactions_total": getattr(bout, "n_interactions_total", np.nan),
                    "n_interaction_onsets": getattr(
                        bout,
                        "n_interaction_onsets",
                        getattr(bout, "n_interactions_total", np.nan),
                    ),
                    "mean_sleep_probability": getattr(bout, "mean_sleep_probability", np.nan),
                    "median_sleep_probability": getattr(bout, "median_sleep_probability", np.nan),
                }
            )
    return pd.DataFrame(rows)


def build_sleep_time_tables(
    sleep_bouts: pd.DataFrame,
    track_activity_long: pd.DataFrame,
    *,
    recording_start_clock_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sleep_duration_bins = split_bout_durations_to_time_bins(
        sleep_bouts,
        recording_start_clock_seconds=recording_start_clock_seconds,
        fps=FPS,
        bin_minutes=TIME_BIN_MINUTES,
    )
    sleep_track_names = set(sleep_bouts["track_name"].dropna().astype(str))
    denom = track_activity_long[track_activity_long["track_name"].astype(str).isin(sleep_track_names)].copy()

    ant_valid = (
        denom.groupby([ACTIVITY_CLUSTER_COL, "track_name", "track_id", "time_bin"], as_index=False)
        .agg(
            hours_since_light_on=("hours_since_light_on", "first"),
            valid_duration_seconds=("valid_duration_seconds", "sum"),
            active_fraction=("active_fraction", "mean"),
            quiet_fraction=("quiet_fraction", "mean"),
        )
    )
    ant_sleep = (
        sleep_duration_bins.groupby([ACTIVITY_CLUSTER_COL, "track_name", "track_id", "time_bin"], as_index=False)
        .agg(sleep_duration_seconds=("sleep_duration_seconds", "sum"))
        if not sleep_duration_bins.empty
        else pd.DataFrame(columns=[ACTIVITY_CLUSTER_COL, "track_name", "track_id", "time_bin", "sleep_duration_seconds"])
    )
    ant_time = ant_valid.merge(
        ant_sleep,
        on=[ACTIVITY_CLUSTER_COL, "track_name", "track_id", "time_bin"],
        how="left",
    )
    ant_time["sleep_duration_seconds"] = ant_time["sleep_duration_seconds"].fillna(0.0)
    ant_time["sleep_fraction_valid_time"] = ant_time["sleep_duration_seconds"] / ant_time[
        "valid_duration_seconds"
    ].replace(0, np.nan)

    cluster_time = (
        ant_time.groupby([ACTIVITY_CLUSTER_COL, "time_bin"], as_index=False)
        .agg(
            hours_since_light_on=("hours_since_light_on", "first"),
            sleep_duration_seconds=("sleep_duration_seconds", "sum"),
            valid_duration_seconds=("valid_duration_seconds", "sum"),
            mean_ant_sleep_fraction=("sleep_fraction_valid_time", "mean"),
            mean_ant_active_fraction=("active_fraction", "mean"),
            mean_ant_quiet_fraction=("quiet_fraction", "mean"),
            n_ants=("track_name", "nunique"),
        )
        .sort_values([ACTIVITY_CLUSTER_COL, "time_bin"], kind="mergesort")
    )
    cluster_time["sleep_fraction_valid_time"] = cluster_time["sleep_duration_seconds"] / cluster_time[
        "valid_duration_seconds"
    ].replace(0, np.nan)
    return cluster_time, ant_time


def elapsed_bin_centers_hours(n_bins: int, bin_minutes: float) -> np.ndarray:
    return (np.arange(int(n_bins), dtype=np.float64) + 0.5) * float(bin_minutes) / 60.0


def split_bout_durations_to_elapsed_bins(
    bouts: pd.DataFrame,
    *,
    analysis_frame_start: int,
    analysis_frame_stop: int,
    fps: float,
    bin_minutes: float,
) -> pd.DataFrame:
    if bouts.empty:
        return pd.DataFrame()
    bin_width_s = float(bin_minutes) * 60.0
    analysis_start = int(analysis_frame_start)
    analysis_stop = int(analysis_frame_stop)
    n_bins = int(np.ceil(max(0, analysis_stop - analysis_start) / float(fps) / bin_width_s))
    centers_h = elapsed_bin_centers_hours(n_bins, bin_minutes)
    rows = []
    for bout in bouts.itertuples(index=False):
        start_frame = max(int(bout.bout_start_frame), analysis_start)
        end_frame_exclusive = min(int(bout.bout_end_frame) + 1, analysis_stop)
        if end_frame_exclusive <= start_frame:
            continue
        start_s = (start_frame - analysis_start) / float(fps)
        end_s = (end_frame_exclusive - analysis_start) / float(fps)
        first_bin = int(np.floor(start_s / bin_width_s))
        last_bin = int(np.floor((end_s - 1e-9) / bin_width_s))
        for elapsed_bin in range(first_bin, last_bin + 1):
            if elapsed_bin < 0 or elapsed_bin >= n_bins:
                continue
            seg_start = max(start_s, elapsed_bin * bin_width_s)
            seg_end = min(end_s, (elapsed_bin + 1) * bin_width_s)
            if seg_end <= seg_start:
                continue
            rows.append(
                {
                    "track_name": getattr(bout, "track_name", None),
                    "track_id": getattr(bout, "track_id", np.nan),
                    "side": getattr(bout, "side", SIDE),
                    ACTIVITY_CLUSTER_COL: getattr(bout, ACTIVITY_CLUSTER_COL, np.nan),
                    "classifier_bin": getattr(bout, "classifier_bin", None),
                    "elapsed_bin": int(elapsed_bin),
                    "elapsed_hours": float(centers_h[elapsed_bin]),
                    "elapsed_days": float(centers_h[elapsed_bin] / 24.0),
                    "sleep_duration_seconds": float(seg_end - seg_start),
                    "n_interactions_total": getattr(bout, "n_interactions_total", np.nan),
                    "n_interaction_onsets": getattr(
                        bout,
                        "n_interaction_onsets",
                        getattr(bout, "n_interactions_total", np.nan),
                    ),
                    "mean_sleep_probability": getattr(bout, "mean_sleep_probability", np.nan),
                    "median_sleep_probability": getattr(bout, "median_sleep_probability", np.nan),
                }
            )
    return pd.DataFrame(rows)


def elapsed_activity_time_table(
    tracks: pd.DataFrame,
    activity_cluster_table: pd.DataFrame,
    *,
    analysis_frame_start: int,
    analysis_frame_stop: int,
    bin_minutes: float,
    active_speed_threshold_mm_s: float,
    quiet_speed_threshold_mm_s: float,
) -> pd.DataFrame:
    analysis_start = int(analysis_frame_start)
    analysis_stop = int(analysis_frame_stop)
    if analysis_stop <= analysis_start:
        return pd.DataFrame()
    n_bins = int(np.ceil((analysis_stop - analysis_start) / float(FPS) / (float(bin_minutes) * 60.0)))
    centers_h = elapsed_bin_centers_hours(n_bins, bin_minutes)
    cluster_cols = activity_cluster_table[
        ["track_name", "track_id", "side", ACTIVITY_CLUSTER_COL]
    ].drop_duplicates("track_name")
    selected = tracks.merge(cluster_cols, on=["track_name", "track_id", "side"], how="inner")
    rows = []
    for i, row in selected.reset_index(drop=True).iterrows():
        if i == 0 or i == len(selected) - 1 or (i + 1) % 25 == 0:
            print(f"elapsed activity profile: {i + 1}/{len(selected)} {row['track_name']}")
        speed = np.load(row["speed_path"], mmap_mode="r")
        frame_min = int(row["speed_frame_min"])
        fps = float(row["speed_fps"])
        global_frames = frame_min + np.arange(len(speed), dtype=np.int64)
        valid = np.isfinite(speed) & (global_frames >= analysis_start) & (global_frames < analysis_stop)
        mean_speed = np.full(n_bins, np.nan, dtype=np.float32)
        active_frac = np.full(n_bins, np.nan, dtype=np.float32)
        quiet_frac = np.full(n_bins, np.nan, dtype=np.float32)
        n_valid = np.zeros(n_bins, dtype=np.int64)
        if valid.any():
            valid_frames = global_frames[valid]
            valid_speed = np.asarray(speed[valid], dtype=np.float64)
            elapsed_s = (valid_frames - analysis_start) / fps
            bins = np.floor(elapsed_s / (float(bin_minutes) * 60.0)).astype(np.int64)
            keep_bins = (bins >= 0) & (bins < n_bins)
            bins = bins[keep_bins]
            valid_speed = valid_speed[keep_bins]
            count = np.bincount(bins, minlength=n_bins)
            speed_sum = np.bincount(bins, weights=valid_speed, minlength=n_bins)
            active_sum = np.bincount(
                bins,
                weights=(valid_speed > float(active_speed_threshold_mm_s)).astype(np.float64),
                minlength=n_bins,
            )
            quiet_sum = np.bincount(
                bins,
                weights=(valid_speed <= float(quiet_speed_threshold_mm_s)).astype(np.float64),
                minlength=n_bins,
            )
            has_data = count > 0
            mean_speed[has_data] = (speed_sum[has_data] / count[has_data]).astype(np.float32)
            active_frac[has_data] = (active_sum[has_data] / count[has_data]).astype(np.float32)
            quiet_frac[has_data] = (quiet_sum[has_data] / count[has_data]).astype(np.float32)
            n_valid = count.astype(np.int64)
        for elapsed_bin in range(n_bins):
            rows.append(
                {
                    "track_name": row["track_name"],
                    "track_id": row["track_id"],
                    "side": row["side"],
                    ACTIVITY_CLUSTER_COL: row[ACTIVITY_CLUSTER_COL],
                    "elapsed_bin": int(elapsed_bin),
                    "elapsed_hours": float(centers_h[elapsed_bin]),
                    "elapsed_days": float(centers_h[elapsed_bin] / 24.0),
                    "mean_speed_mm_s": float(mean_speed[elapsed_bin])
                    if np.isfinite(mean_speed[elapsed_bin])
                    else np.nan,
                    "active_fraction": float(active_frac[elapsed_bin])
                    if np.isfinite(active_frac[elapsed_bin])
                    else np.nan,
                    "quiet_fraction": float(quiet_frac[elapsed_bin])
                    if np.isfinite(quiet_frac[elapsed_bin])
                    else np.nan,
                    "n_valid_frames": int(n_valid[elapsed_bin]),
                    "valid_duration_seconds": float(n_valid[elapsed_bin] / fps),
                }
            )
    return pd.DataFrame(rows)


def build_elapsed_sleep_time_tables(
    sleep_bouts: pd.DataFrame,
    activity_tracks: pd.DataFrame,
    activity_cluster_table: pd.DataFrame,
    *,
    analysis_frame_start: int,
    analysis_frame_stop: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    elapsed_activity = elapsed_activity_time_table(
        activity_tracks,
        activity_cluster_table,
        analysis_frame_start=analysis_frame_start,
        analysis_frame_stop=analysis_frame_stop,
        bin_minutes=ELAPSED_SLEEP_BIN_MINUTES,
        active_speed_threshold_mm_s=ACTIVITY_ACTIVE_SPEED_THRESHOLD_MM_S,
        quiet_speed_threshold_mm_s=ACTIVITY_QUIET_SPEED_THRESHOLD_MM_S,
    )
    sleep_duration_bins = split_bout_durations_to_elapsed_bins(
        sleep_bouts,
        analysis_frame_start=analysis_frame_start,
        analysis_frame_stop=analysis_frame_stop,
        fps=FPS,
        bin_minutes=ELAPSED_SLEEP_BIN_MINUTES,
    )
    sleep_track_names = set(sleep_bouts["track_name"].dropna().astype(str))
    denom = elapsed_activity[elapsed_activity["track_name"].astype(str).isin(sleep_track_names)].copy()
    ant_valid = (
        denom.groupby([ACTIVITY_CLUSTER_COL, "track_name", "track_id", "elapsed_bin"], as_index=False)
        .agg(
            elapsed_hours=("elapsed_hours", "first"),
            elapsed_days=("elapsed_days", "first"),
            valid_duration_seconds=("valid_duration_seconds", "sum"),
            active_fraction=("active_fraction", "mean"),
            quiet_fraction=("quiet_fraction", "mean"),
        )
    )
    ant_sleep = (
        sleep_duration_bins.groupby([ACTIVITY_CLUSTER_COL, "track_name", "track_id", "elapsed_bin"], as_index=False)
        .agg(sleep_duration_seconds=("sleep_duration_seconds", "sum"))
        if not sleep_duration_bins.empty
        else pd.DataFrame(
            columns=[ACTIVITY_CLUSTER_COL, "track_name", "track_id", "elapsed_bin", "sleep_duration_seconds"]
        )
    )
    ant_time = ant_valid.merge(
        ant_sleep,
        on=[ACTIVITY_CLUSTER_COL, "track_name", "track_id", "elapsed_bin"],
        how="left",
    )
    ant_time["sleep_duration_seconds"] = ant_time["sleep_duration_seconds"].fillna(0.0)
    ant_time["sleep_fraction_valid_time"] = ant_time["sleep_duration_seconds"] / ant_time[
        "valid_duration_seconds"
    ].replace(0, np.nan)
    cluster_time = (
        ant_time.groupby([ACTIVITY_CLUSTER_COL, "elapsed_bin"], as_index=False)
        .agg(
            elapsed_hours=("elapsed_hours", "first"),
            elapsed_days=("elapsed_days", "first"),
            sleep_duration_seconds=("sleep_duration_seconds", "sum"),
            valid_duration_seconds=("valid_duration_seconds", "sum"),
            mean_ant_sleep_fraction=("sleep_fraction_valid_time", "mean"),
            mean_ant_active_fraction=("active_fraction", "mean"),
            mean_ant_quiet_fraction=("quiet_fraction", "mean"),
            n_ants=("track_name", "nunique"),
        )
        .sort_values([ACTIVITY_CLUSTER_COL, "elapsed_bin"], kind="mergesort")
    )
    cluster_time["sleep_fraction_valid_time"] = cluster_time["sleep_duration_seconds"] / cluster_time[
        "valid_duration_seconds"
    ].replace(0, np.nan)
    return cluster_time, ant_time


def plot_cluster_sleep_timing(cluster_sleep_time: pd.DataFrame, intra_cluster_interactions: pd.DataFrame | None = None) -> None:
    import matplotlib.pyplot as plt

    clusters = sorted(cluster_sleep_time[ACTIVITY_CLUSTER_COL].dropna().unique())
    if not clusters:
        print("No cluster sleep timing rows to plot")
        return
    nrows = len(clusters)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, max(3.2, 2.4 * nrows)), sharex=True, squeeze=False)
    for ax, cluster in zip(axes.ravel(), clusters):
        group = cluster_sleep_time[cluster_sleep_time[ACTIVITY_CLUSTER_COL] == cluster]
        ax.plot(
            group["hours_since_light_on"],
            group["sleep_fraction_valid_time"],
            color="tab:blue",
            lw=2.0,
            label="predicted sleep fraction",
        )
        ax.plot(
            group["hours_since_light_on"],
            group["mean_ant_active_fraction"],
            color="tab:orange",
            lw=1.4,
            alpha=0.9,
            label="mean ant active fraction",
        )
        if intra_cluster_interactions is not None and not intra_cluster_interactions.empty:
            inter = intra_cluster_interactions[intra_cluster_interactions[ACTIVITY_CLUSTER_COL] == cluster]
            if not inter.empty:
                twin = ax.twinx()
                twin.plot(
                    inter["hours_since_light_on"],
                    inter["interaction_onsets_per_pair"],
                    color="0.30",
                    lw=1.2,
                    alpha=0.8,
                    label="within-cluster interaction onsets / pair",
                )
                twin.set_ylabel("onsets / pair", color="0.30")
                twin.tick_params(axis="y", colors="0.30")
        n_ants = int(group["n_ants"].max()) if group["n_ants"].notna().any() else 0
        ax.set_title(f"activity cluster {cluster} n={n_ants} sleep/work timing")
        ax.set_ylabel("fraction")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    axes[-1, 0].set_xlabel(f"Hours since lights on at {LIGHT_ON_HOUR:g}:00")
    fig.tight_layout()
    plt.show()


def safe_pearson(x: np.ndarray, y: np.ndarray, *, min_n: int = 3) -> tuple[float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    n = int(keep.sum())
    if n < int(min_n):
        return np.nan, n
    x_keep = x[keep]
    y_keep = y[keep]
    if np.nanstd(x_keep) <= 1e-12 or np.nanstd(y_keep) <= 1e-12:
        return np.nan, n
    return float(np.corrcoef(x_keep, y_keep)[0, 1]), n


def sleep_interaction_time_correlation(
    cluster_sleep_time: pd.DataFrame,
    intra_cluster_interactions: pd.DataFrame | None,
    *,
    sleep_col: str = "sleep_fraction_valid_time",
    min_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cluster_sleep_time.empty:
        return pd.DataFrame(), pd.DataFrame()
    required_sleep = {
        ACTIVITY_CLUSTER_COL,
        "time_bin",
        "hours_since_light_on",
        sleep_col,
        "sleep_duration_seconds",
        "valid_duration_seconds",
        "mean_ant_sleep_fraction",
        "mean_ant_active_fraction",
        "mean_ant_quiet_fraction",
        "n_ants",
    }
    missing = required_sleep.difference(cluster_sleep_time.columns)
    if missing:
        raise ValueError(f"cluster_sleep_time is missing columns: {sorted(missing)}")

    sleep = cluster_sleep_time[
        [
            ACTIVITY_CLUSTER_COL,
            "time_bin",
            "hours_since_light_on",
            sleep_col,
            "sleep_duration_seconds",
            "valid_duration_seconds",
            "mean_ant_sleep_fraction",
            "mean_ant_active_fraction",
            "mean_ant_quiet_fraction",
            "n_ants",
        ]
    ].copy()
    sleep = sleep.rename(columns={sleep_col: "sleep_amount"})

    inter_cols = {
        ACTIVITY_CLUSTER_COL,
        "time_bin",
        "n_cluster_interaction_onsets",
        "n_cluster_interaction_onsets_as_antenna",
        "n_cluster_interaction_onsets_as_body",
        "interaction_onset_rate_per_ant_per_h",
        "interaction_onset_rate_as_antenna_per_ant_per_h",
        "interaction_onset_rate_as_body_per_ant_per_h",
    }
    legacy_inter_cols = {
        ACTIVITY_CLUSTER_COL,
        "time_bin",
        "n_intra_cluster_interaction_onsets",
        "n_possible_pairs",
        "interaction_onsets_per_pair",
    }
    if intra_cluster_interactions is None or intra_cluster_interactions.empty:
        inter = pd.DataFrame(columns=sorted(inter_cols))
    else:
        available = set(intra_cluster_interactions.columns)
        if inter_cols.issubset(available):
            inter = intra_cluster_interactions[list(inter_cols)].copy()
        elif legacy_inter_cols.issubset(available):
            inter = intra_cluster_interactions[list(legacy_inter_cols)].copy()
        else:
            expected = sorted(inter_cols | legacy_inter_cols)
            raise ValueError(f"interaction table is missing expected columns; need one schema from: {expected}")

    merged = sleep.merge(inter, on=[ACTIVITY_CLUSTER_COL, "time_bin"], how="left")
    bin_hours = float(TIME_BIN_MINUTES) / 60.0
    if "n_cluster_interaction_onsets" in merged.columns:
        for col in [
            "n_cluster_interaction_onsets",
            "n_cluster_interaction_onsets_as_antenna",
            "n_cluster_interaction_onsets_as_body",
        ]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(np.int64)
        denom = pd.to_numeric(merged["n_ants"], errors="coerce").replace(0, np.nan) * bin_hours
        for col in [
            "interaction_onset_rate_per_ant_per_h",
            "interaction_onset_rate_as_antenna_per_ant_per_h",
            "interaction_onset_rate_as_body_per_ant_per_h",
        ]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged["interaction_onset_rate_per_ant_per_h"] = merged["interaction_onset_rate_per_ant_per_h"].fillna(
            merged["n_cluster_interaction_onsets"] / denom
        )
        merged["interaction_onset_rate_as_antenna_per_ant_per_h"] = merged[
            "interaction_onset_rate_as_antenna_per_ant_per_h"
        ].fillna(merged["n_cluster_interaction_onsets_as_antenna"] / denom)
        merged["interaction_onset_rate_as_body_per_ant_per_h"] = merged[
            "interaction_onset_rate_as_body_per_ant_per_h"
        ].fillna(merged["n_cluster_interaction_onsets_as_body"] / denom)
    else:
        merged["n_intra_cluster_interaction_onsets"] = (
            pd.to_numeric(merged["n_intra_cluster_interaction_onsets"], errors="coerce").fillna(0).astype(np.int64)
        )
        fallback_pairs = (
            pd.to_numeric(merged["n_ants"], errors="coerce")
            * np.maximum(pd.to_numeric(merged["n_ants"], errors="coerce") - 1, 0)
            / 2.0
        )
        merged["n_possible_pairs"] = pd.to_numeric(merged["n_possible_pairs"], errors="coerce")
        merged["n_possible_pairs"] = merged.groupby(ACTIVITY_CLUSTER_COL)["n_possible_pairs"].transform(
            lambda values: values.ffill().bfill()
        )
        merged["n_possible_pairs"] = merged["n_possible_pairs"].fillna(fallback_pairs)
        merged["interaction_onsets_per_pair"] = pd.to_numeric(
            merged["interaction_onsets_per_pair"],
            errors="coerce",
        )
        merged["interaction_onsets_per_pair"] = merged["interaction_onsets_per_pair"].fillna(
            merged["n_intra_cluster_interaction_onsets"] / merged["n_possible_pairs"].replace(0, np.nan)
        )
        merged["interaction_onsets_per_pair"] = merged["interaction_onsets_per_pair"].fillna(0.0)
        merged["n_cluster_interaction_onsets"] = merged["n_intra_cluster_interaction_onsets"]
        merged["n_cluster_interaction_onsets_as_antenna"] = np.nan
        merged["n_cluster_interaction_onsets_as_body"] = np.nan
        merged["interaction_onset_rate_per_ant_per_h"] = merged["interaction_onsets_per_pair"] / bin_hours
        merged["interaction_onset_rate_as_antenna_per_ant_per_h"] = np.nan
        merged["interaction_onset_rate_as_body_per_ant_per_h"] = np.nan
    for col in [
        "interaction_onset_rate_per_ant_per_h",
        "interaction_onset_rate_as_antenna_per_ant_per_h",
        "interaction_onset_rate_as_body_per_ant_per_h",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    rows = []
    for cluster, group in merged.groupby(ACTIVITY_CLUSTER_COL, sort=True):
        group = group.sort_values("time_bin", kind="mergesort")
        sleep_amount = group["sleep_amount"].to_numpy(np.float64)
        rate_specs = {
            "all_roles": "interaction_onset_rate_per_ant_per_h",
            "receiver_body": "interaction_onset_rate_as_body_per_ant_per_h",
            "source_antenna": "interaction_onset_rate_as_antenna_per_ant_per_h",
        }
        row = {
            ACTIVITY_CLUSTER_COL: cluster,
            "mean_sleep_fraction": float(np.nanmean(sleep_amount)) if len(sleep_amount) else np.nan,
            "total_sleep_duration_seconds": float(
                pd.to_numeric(group["sleep_duration_seconds"], errors="coerce").fillna(0).sum()
            ),
            "total_cluster_interaction_onsets": int(group["n_cluster_interaction_onsets"].sum()),
            "total_cluster_interaction_onsets_as_body": int(
                pd.to_numeric(group["n_cluster_interaction_onsets_as_body"], errors="coerce").fillna(0).sum()
            ),
            "total_cluster_interaction_onsets_as_antenna": int(
                pd.to_numeric(group["n_cluster_interaction_onsets_as_antenna"], errors="coerce").fillna(0).sum()
            ),
        }
        n_values = []
        for label, col in rate_specs.items():
            interaction_rate = group[col].to_numpy(np.float64)
            pearson_r, n = safe_pearson(sleep_amount, interaction_rate, min_n=min_n)
            spearman_r, _n_rank = safe_pearson(
                pd.Series(sleep_amount).rank(method="average").to_numpy(np.float64),
                pd.Series(interaction_rate).rank(method="average").to_numpy(np.float64),
                min_n=min_n,
            )
            n_values.append(n)
            row[f"pearson_sleep_vs_{label}_interaction_rate"] = pearson_r
            row[f"spearman_sleep_vs_{label}_interaction_rate"] = spearman_r
            row[f"mean_{label}_interaction_onset_rate_per_ant_per_h"] = (
                float(np.nanmean(interaction_rate)) if len(interaction_rate) else np.nan
            )
        row["n_time_bins"] = int(max(n_values) if n_values else 0)
        row["pearson_sleep_vs_interaction_rate"] = row["pearson_sleep_vs_all_roles_interaction_rate"]
        row["spearman_sleep_vs_interaction_rate"] = row["spearman_sleep_vs_all_roles_interaction_rate"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    return (
        merged.sort_values([ACTIVITY_CLUSTER_COL, "time_bin"], kind="mergesort").reset_index(drop=True),
        summary.reset_index(drop=True),
    )


def plot_sleep_interaction_time_correlation(
    sleep_interaction_time: pd.DataFrame,
    sleep_interaction_correlation: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    if sleep_interaction_time.empty:
        print("No sleep/interaction time-bin rows to plot")
        return
    clusters = sorted(sleep_interaction_time[ACTIVITY_CLUSTER_COL].dropna().unique())
    if not clusters:
        print("No activity clusters available for sleep/interaction correlation")
        return
    ncols = min(3, len(clusters))
    nrows = int(np.ceil(len(clusters) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, 3.9 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    scatter = None
    summary = sleep_interaction_correlation.set_index(ACTIVITY_CLUSTER_COL) if not sleep_interaction_correlation.empty else pd.DataFrame()
    for ax, cluster in zip(axes.ravel(), clusters):
        group = sleep_interaction_time[sleep_interaction_time[ACTIVITY_CLUSTER_COL] == cluster].sort_values(
            "time_bin",
            kind="mergesort",
        )
        x = group["interaction_onset_rate_as_body_per_ant_per_h"].to_numpy(np.float64)
        x_source = group["interaction_onset_rate_as_antenna_per_ant_per_h"].to_numpy(np.float64)
        y = group["sleep_amount"].to_numpy(np.float64)
        colors = group["hours_since_light_on"].to_numpy(np.float64)
        keep = np.isfinite(x) & np.isfinite(y)
        scatter = ax.scatter(
            x[keep],
            y[keep],
            c=colors[keep],
            cmap="viridis",
            s=34,
            alpha=0.85,
            edgecolors="none",
            label="as body/receiver",
        )
        keep_source = np.isfinite(x_source) & np.isfinite(y)
        ax.scatter(
            x_source[keep_source],
            y[keep_source],
            facecolors="none",
            edgecolors="0.35",
            s=30,
            alpha=0.70,
            linewidths=0.8,
            label="as antenna/source",
        )
        if keep.sum() >= 2 and np.nanstd(x[keep]) > 1e-12 and np.nanstd(y[keep]) > 1e-12:
            slope, intercept = np.polyfit(x[keep], y[keep], deg=1)
            x_line = np.linspace(float(np.nanmin(x[keep])), float(np.nanmax(x[keep])), 100)
            ax.plot(x_line, slope * x_line + intercept, color="0.15", lw=1.2, alpha=0.85)
        if cluster in summary.index:
            row = summary.loc[cluster]
            ax.set_title(
                f"cluster {cluster}: body r={row['pearson_sleep_vs_receiver_body_interaction_rate']:.2f}, "
                f"all r={row['pearson_sleep_vs_all_roles_interaction_rate']:.2f}"
            )
        else:
            ax.set_title(f"cluster {cluster}")
        ax.set_xlabel("directed interaction-bout onsets / ant / h")
        ax.set_ylabel("sleep fraction")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    for ax in axes.ravel()[len(clusters):]:
        ax.set_axis_off()
    if scatter is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label=f"hours since lights on at {LIGHT_ON_HOUR:g}:00")
    fig.suptitle("Predicted sleep vs directed interaction amount")
    plt.show()


def safe_spearman(x: np.ndarray, y: np.ndarray, *, min_n: int = 3) -> tuple[float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    n = int(keep.sum())
    if n < int(min_n):
        return np.nan, n
    x_rank = pd.Series(x[keep]).rank(method="average").to_numpy(np.float64)
    y_rank = pd.Series(y[keep]).rank(method="average").to_numpy(np.float64)
    corr, _n_rank = safe_pearson(x_rank, y_rank, min_n=min_n)
    return corr, n


def _predictor_matrix(predictors: np.ndarray | pd.Series | pd.DataFrame | list | tuple) -> np.ndarray:
    if isinstance(predictors, (list, tuple)):
        arrays = [np.asarray(col, dtype=np.float64).reshape(-1) for col in predictors]
        if not arrays:
            return np.empty((0, 0), dtype=np.float64)
        return np.column_stack(arrays)
    x = np.asarray(predictors, dtype=np.float64)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def linear_r2(
    values: np.ndarray,
    predictors: np.ndarray | pd.Series | pd.DataFrame | list | tuple,
    *,
    min_n: int = 3,
) -> tuple[float, int]:
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    x = _predictor_matrix(predictors)
    if x.shape[0] != len(y):
        raise ValueError("predictors and values must have the same number of rows")
    keep = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    n = int(keep.sum())
    if n < int(min_n):
        return np.nan, n
    y_keep = y[keep]
    if np.nanstd(y_keep) <= 1e-12:
        return np.nan, n
    x_keep = x[keep]
    varying = np.nanstd(x_keep, axis=0) > 1e-12
    if not bool(varying.any()):
        return 0.0, n
    x_keep = x_keep[:, varying]
    if n < max(int(min_n), x_keep.shape[1] + 2):
        return np.nan, n
    design = np.column_stack([np.ones(n, dtype=np.float64), x_keep])
    beta, *_ = np.linalg.lstsq(design, y_keep, rcond=None)
    fitted = design @ beta
    ss_res = float(np.sum((y_keep - fitted) ** 2))
    ss_tot = float(np.sum((y_keep - float(np.mean(y_keep))) ** 2))
    if ss_tot <= 1e-12:
        return np.nan, n
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0)), n


def residualize(
    values: np.ndarray,
    predictors: np.ndarray | pd.Series | pd.DataFrame | list | tuple,
    *,
    min_n: int = 3,
) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    x = _predictor_matrix(predictors)
    if x.shape[0] != len(y):
        raise ValueError("predictors and values must have the same number of rows")
    residual = np.full(len(y), np.nan, dtype=np.float64)
    keep = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    n = int(keep.sum())
    if n < int(min_n):
        return residual
    y_keep = y[keep]
    x_keep = x[keep]
    varying = np.nanstd(x_keep, axis=0) > 1e-12
    if bool(varying.any()):
        x_keep = x_keep[:, varying]
        if n < max(int(min_n), x_keep.shape[1] + 2):
            return residual
        design = np.column_stack([np.ones(n, dtype=np.float64), x_keep])
    else:
        design = np.ones((n, 1), dtype=np.float64)
    beta, *_ = np.linalg.lstsq(design, y_keep, rcond=None)
    residual[keep] = y_keep - design @ beta
    return residual


def _add_relationship_correlations(
    row: dict,
    label: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_n: int,
) -> None:
    pearson_r, n = safe_pearson(x, y, min_n=min_n)
    spearman_r, _n_rank = safe_spearman(x, y, min_n=min_n)
    row[f"pearson_{label}"] = pearson_r
    row[f"spearman_{label}"] = spearman_r
    row[f"n_{label}"] = n


def sleep_activity_interaction_relationship_summary(
    sleep_interaction_time: pd.DataFrame,
    *,
    min_n: int = 3,
) -> pd.DataFrame:
    if sleep_interaction_time.empty:
        return pd.DataFrame()
    required = {
        ACTIVITY_CLUSTER_COL,
        "time_bin",
        "sleep_amount",
        "mean_ant_active_fraction",
        "interaction_onset_rate_per_ant_per_h",
        "interaction_onset_rate_as_body_per_ant_per_h",
        "interaction_onset_rate_as_antenna_per_ant_per_h",
    }
    missing = required.difference(sleep_interaction_time.columns)
    if missing:
        raise ValueError(f"sleep_interaction_time is missing columns: {sorted(missing)}")

    rows = []
    for cluster, group in sleep_interaction_time.groupby(ACTIVITY_CLUSTER_COL, sort=True):
        group = group.sort_values("time_bin", kind="mergesort")

        def values(col: str) -> np.ndarray:
            return pd.to_numeric(group[col], errors="coerce").to_numpy(np.float64)

        sleep = values("sleep_amount")
        active = values("mean_ant_active_fraction")
        interaction_all = values("interaction_onset_rate_per_ant_per_h")
        interaction_receiver = values("interaction_onset_rate_as_body_per_ant_per_h")
        interaction_source = values("interaction_onset_rate_as_antenna_per_ant_per_h")
        row = {
            ACTIVITY_CLUSTER_COL: cluster,
            "n_time_bins": int(len(group)),
            "mean_sleep_amount": float(np.nanmean(sleep)) if len(sleep) else np.nan,
            "mean_active_fraction": float(np.nanmean(active)) if len(active) else np.nan,
            "mean_all_interaction_onset_rate_per_ant_per_h": (
                float(np.nanmean(interaction_all)) if len(interaction_all) else np.nan
            ),
            "mean_receiver_interaction_onset_rate_per_ant_per_h": (
                float(np.nanmean(interaction_receiver)) if len(interaction_receiver) else np.nan
            ),
            "mean_source_interaction_onset_rate_per_ant_per_h": (
                float(np.nanmean(interaction_source)) if len(interaction_source) else np.nan
            ),
        }
        _add_relationship_correlations(
            row,
            "sleep_vs_active_fraction",
            sleep,
            active,
            min_n=min_n,
        )
        _add_relationship_correlations(
            row,
            "sleep_vs_all_interaction_rate",
            sleep,
            interaction_all,
            min_n=min_n,
        )
        _add_relationship_correlations(
            row,
            "sleep_vs_receiver_interaction_rate",
            sleep,
            interaction_receiver,
            min_n=min_n,
        )
        _add_relationship_correlations(
            row,
            "sleep_vs_source_interaction_rate",
            sleep,
            interaction_source,
            min_n=min_n,
        )
        _add_relationship_correlations(
            row,
            "active_fraction_vs_all_interaction_rate",
            active,
            interaction_all,
            min_n=min_n,
        )
        _add_relationship_correlations(
            row,
            "active_fraction_vs_receiver_interaction_rate",
            active,
            interaction_receiver,
            min_n=min_n,
        )
        _add_relationship_correlations(
            row,
            "active_fraction_vs_source_interaction_rate",
            active,
            interaction_source,
            min_n=min_n,
        )

        row["r2_sleep_from_active_fraction"], row["n_r2_sleep_from_active_fraction"] = linear_r2(
            sleep,
            active,
            min_n=min_n,
        )
        row["r2_sleep_from_all_interaction_rate"], row["n_r2_sleep_from_all_interaction_rate"] = linear_r2(
            sleep,
            interaction_all,
            min_n=min_n,
        )
        row["r2_sleep_from_receiver_interaction_rate"], row[
            "n_r2_sleep_from_receiver_interaction_rate"
        ] = linear_r2(
            sleep,
            interaction_receiver,
            min_n=min_n,
        )
        row["r2_sleep_from_source_interaction_rate"], row["n_r2_sleep_from_source_interaction_rate"] = linear_r2(
            sleep,
            interaction_source,
            min_n=min_n,
        )
        row["r2_sleep_from_active_and_receiver_interaction_rate"], row[
            "n_r2_sleep_from_active_and_receiver_interaction_rate"
        ] = linear_r2(
            sleep,
            [active, interaction_receiver],
            min_n=min_n,
        )
        row["r2_sleep_from_active_and_all_interaction_rate"], row[
            "n_r2_sleep_from_active_and_all_interaction_rate"
        ] = linear_r2(
            sleep,
            [active, interaction_all],
            min_n=min_n,
        )
        row["r2_receiver_interaction_rate_from_active_fraction"], row[
            "n_r2_receiver_interaction_rate_from_active_fraction"
        ] = linear_r2(
            interaction_receiver,
            active,
            min_n=min_n,
        )
        row["r2_all_interaction_rate_from_active_fraction"], row[
            "n_r2_all_interaction_rate_from_active_fraction"
        ] = linear_r2(
            interaction_all,
            active,
            min_n=min_n,
        )

        sleep_resid_active = residualize(sleep, active, min_n=min_n)
        receiver_resid_active = residualize(interaction_receiver, active, min_n=min_n)
        partial_r, partial_n = safe_pearson(sleep_resid_active, receiver_resid_active, min_n=min_n)
        row["partial_pearson_sleep_vs_receiver_interaction_rate_controlling_active_fraction"] = partial_r
        row["n_partial_sleep_vs_receiver_interaction_rate_controlling_active_fraction"] = partial_n

        sleep_resid_receiver = residualize(sleep, interaction_receiver, min_n=min_n)
        active_resid_receiver = residualize(active, interaction_receiver, min_n=min_n)
        partial_r, partial_n = safe_pearson(sleep_resid_receiver, active_resid_receiver, min_n=min_n)
        row["partial_pearson_sleep_vs_active_fraction_controlling_receiver_interaction_rate"] = partial_r
        row["n_partial_sleep_vs_active_fraction_controlling_receiver_interaction_rate"] = partial_n
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _zscore_for_plot(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.full(len(values), np.nan, dtype=np.float64)
    keep = np.isfinite(values)
    if not bool(keep.any()):
        return out
    sd = float(np.nanstd(values[keep]))
    if sd <= 1e-12:
        out[keep] = 0.0
    else:
        out[keep] = (values[keep] - float(np.nanmean(values[keep]))) / sd
    return out


def _format_plot_value(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.2f}"


def plot_sleep_activity_interaction_profiles(
    sleep_interaction_time: pd.DataFrame,
    relationship_summary: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    if sleep_interaction_time.empty:
        print("No time-bin rows to plot")
        return
    clusters = sorted(sleep_interaction_time[ACTIVITY_CLUSTER_COL].dropna().unique())
    if not clusters:
        print("No activity clusters available for sleep/activity/interaction profiles")
        return
    ncols = min(2, len(clusters))
    nrows = int(np.ceil(len(clusters) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.6 * ncols, 3.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    summary = relationship_summary.set_index(ACTIVITY_CLUSTER_COL) if not relationship_summary.empty else pd.DataFrame()
    line_specs = [
        ("sleep_amount", "sleep amount", "tab:blue", "-"),
        ("mean_ant_active_fraction", "active fraction", "tab:orange", "-"),
        ("interaction_onset_rate_as_body_per_ant_per_h", "receiver/body interactions", "tab:red", "-"),
        ("interaction_onset_rate_as_antenna_per_ant_per_h", "source/antenna interactions", "0.35", "--"),
        ("interaction_onset_rate_per_ant_per_h", "all interactions", "tab:green", ":"),
    ]
    for ax, cluster in zip(axes.ravel(), clusters):
        group = sleep_interaction_time[sleep_interaction_time[ACTIVITY_CLUSTER_COL] == cluster].sort_values(
            "time_bin",
            kind="mergesort",
        )
        hours = group["hours_since_light_on"].to_numpy(np.float64)
        for col, label, color, linestyle in line_specs:
            values = pd.to_numeric(group[col], errors="coerce").to_numpy(np.float64)
            ax.plot(
                hours,
                _zscore_for_plot(values),
                color=color,
                lw=1.5,
                ls=linestyle,
                marker="o",
                ms=3,
                alpha=0.86,
                label=label,
            )
        title = f"cluster {cluster}"
        if cluster in summary.index:
            row = summary.loc[cluster]
            title += (
                f": r(sleep,active)={_format_plot_value(row['pearson_sleep_vs_active_fraction'])}, "
                f"r(sleep,receiver)={_format_plot_value(row['pearson_sleep_vs_receiver_interaction_rate'])}, "
                "partial="
                f"{_format_plot_value(row['partial_pearson_sleep_vs_receiver_interaction_rate_controlling_active_fraction'])}"
            )
        ax.set_title(title)
        ax.axhline(0, color="0.2", lw=0.8, alpha=0.35)
        ax.set_xlabel(f"hours since lights on at {LIGHT_ON_HOUR:g}:00")
        ax.set_ylabel("within-cluster z-score")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="best")
    for ax in axes.ravel()[len(clusters):]:
        ax.set_axis_off()
    fig.suptitle("Time-of-day profiles: sleep, activity, and bout-level interaction rates")
    plt.show()


def plot_sleep_activity_interaction_pairplots(
    sleep_interaction_time: pd.DataFrame,
    relationship_summary: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    if sleep_interaction_time.empty:
        print("No time-bin rows to plot")
        return
    clusters = sorted(sleep_interaction_time[ACTIVITY_CLUSTER_COL].dropna().unique())
    if not clusters:
        print("No activity clusters available for sleep/activity/interaction pairplots")
        return
    plot_specs = [
        (
            "mean_ant_active_fraction",
            "sleep_amount",
            "active fraction",
            "sleep fraction",
            "pearson_sleep_vs_active_fraction",
        ),
        (
            "interaction_onset_rate_as_body_per_ant_per_h",
            "sleep_amount",
            "receiver/body interactions / ant / h",
            "sleep fraction",
            "pearson_sleep_vs_receiver_interaction_rate",
        ),
        (
            "interaction_onset_rate_as_antenna_per_ant_per_h",
            "sleep_amount",
            "source/antenna interactions / ant / h",
            "sleep fraction",
            "pearson_sleep_vs_source_interaction_rate",
        ),
        (
            "mean_ant_active_fraction",
            "interaction_onset_rate_as_body_per_ant_per_h",
            "active fraction",
            "receiver/body interactions / ant / h",
            "pearson_active_fraction_vs_receiver_interaction_rate",
        ),
    ]
    fig, axes = plt.subplots(
        len(clusters),
        len(plot_specs),
        figsize=(4.4 * len(plot_specs), 3.3 * len(clusters)),
        squeeze=False,
        constrained_layout=True,
    )
    summary = relationship_summary.set_index(ACTIVITY_CLUSTER_COL) if not relationship_summary.empty else pd.DataFrame()
    scatter = None
    for row_idx, cluster in enumerate(clusters):
        group = sleep_interaction_time[sleep_interaction_time[ACTIVITY_CLUSTER_COL] == cluster].sort_values(
            "time_bin",
            kind="mergesort",
        )
        colors = group["hours_since_light_on"].to_numpy(np.float64)
        for col_idx, (x_col, y_col, x_label, y_label, corr_col) in enumerate(plot_specs):
            ax = axes[row_idx, col_idx]
            x = pd.to_numeric(group[x_col], errors="coerce").to_numpy(np.float64)
            y = pd.to_numeric(group[y_col], errors="coerce").to_numpy(np.float64)
            keep = np.isfinite(x) & np.isfinite(y)
            scatter = ax.scatter(
                x[keep],
                y[keep],
                c=colors[keep],
                cmap="viridis",
                s=32,
                alpha=0.84,
                edgecolors="none",
            )
            if keep.sum() >= 2 and np.nanstd(x[keep]) > 1e-12 and np.nanstd(y[keep]) > 1e-12:
                slope, intercept = np.polyfit(x[keep], y[keep], deg=1)
                x_line = np.linspace(float(np.nanmin(x[keep])), float(np.nanmax(x[keep])), 100)
                ax.plot(x_line, slope * x_line + intercept, color="0.15", lw=1.1, alpha=0.85)
            if cluster in summary.index and corr_col in summary.columns:
                corr = _format_plot_value(float(summary.loc[cluster, corr_col]))
                ax.set_title(f"cluster {cluster}: r={corr}")
            else:
                ax.set_title(f"cluster {cluster}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.25)
    if scatter is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label=f"hours since lights on at {LIGHT_ON_HOUR:g}:00")
    fig.suptitle("Pairwise time-bin relationships across the day")
    plt.show()


def plot_sleep_activity_interaction_reducibility(relationship_summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if relationship_summary.empty:
        print("No relationship summary rows to plot")
        return
    cols = [
        ("r2_sleep_from_active_fraction", "sleep from activity"),
        ("r2_sleep_from_receiver_interaction_rate", "sleep from receiver interactions"),
        ("r2_sleep_from_active_and_receiver_interaction_rate", "sleep from activity + receiver interactions"),
        ("r2_receiver_interaction_rate_from_active_fraction", "receiver interactions from activity"),
    ]
    cols = [(col, label) for col, label in cols if col in relationship_summary.columns]
    if not cols:
        print("No R2 columns available for reducibility plot")
        return
    summary = relationship_summary.sort_values(ACTIVITY_CLUSTER_COL, kind="mergesort")
    clusters = summary[ACTIVITY_CLUSTER_COL].astype(str).to_list()
    x = np.arange(len(summary), dtype=np.float64)
    width = min(0.18, 0.82 / max(len(cols), 1))
    fig, ax = plt.subplots(figsize=(max(7.0, 1.2 * len(summary)), 4.1), constrained_layout=True)
    for idx, (col, label) in enumerate(cols):
        offset = (idx - (len(cols) - 1) / 2.0) * width
        ax.bar(x + offset, pd.to_numeric(summary[col], errors="coerce"), width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(clusters)
    ax.set_xlabel("activity cluster")
    ax.set_ylabel("linear R2 across time bins")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    ax.set_title("How much of each day profile is reducible to another signal?")
    plt.show()


def lagged_correlation(
    series_a: np.ndarray,
    series_b: np.ndarray,
    lag_bins_b_vs_a: int,
    *,
    circular: bool,
    min_overlap_bins: int,
) -> tuple[float, int]:
    """Positive lag means series_b is delayed relative to series_a."""
    a = np.asarray(series_a, dtype=np.float64)
    b = np.asarray(series_b, dtype=np.float64)
    lag = int(lag_bins_b_vs_a)
    if circular:
        return safe_pearson(a, np.roll(b, -lag), min_n=min_overlap_bins)
    if lag > 0:
        if lag >= len(a):
            return np.nan, 0
        return safe_pearson(a[:-lag], b[lag:], min_n=min_overlap_bins)
    if lag < 0:
        shift = -lag
        if shift >= len(a):
            return np.nan, 0
        return safe_pearson(a[shift:], b[:-shift], min_n=min_overlap_bins)
    return safe_pearson(a, b, min_n=min_overlap_bins)


def time_bin_eta_squared_from_arrays(time_bins: np.ndarray, values: np.ndarray) -> float:
    bins = np.asarray(time_bins)
    vals = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(vals) & pd.notna(bins)
    if keep.sum() < 3:
        return np.nan
    bins = bins[keep]
    vals = vals[keep]
    if np.nanstd(vals) <= 1e-12:
        return np.nan
    unique_bins, inverse, counts = np.unique(bins, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=vals, minlength=len(unique_bins))
    means = sums / counts
    grand_mean = float(vals.mean())
    total_ss = float(np.sum((vals - grand_mean) ** 2))
    if total_ss <= 1e-12:
        return np.nan
    between_ss = float(np.sum(counts * (means - grand_mean) ** 2))
    return between_ss / total_ss


def time_bin_eta_squared_permutation(
    data: pd.DataFrame,
    *,
    value_col: str,
    rng: np.random.Generator,
    n_permutations: int,
) -> tuple[float, float]:
    required = ["track_name", "time_bin", value_col]
    work = data[required].dropna(subset=["track_name", "time_bin", value_col]).reset_index(drop=True)
    if work.empty:
        return np.nan, np.nan
    values = work[value_col].to_numpy(np.float64)
    time_bins = work["time_bin"].to_numpy()
    observed = time_bin_eta_squared_from_arrays(time_bins, values)
    if not np.isfinite(observed) or int(n_permutations) <= 0:
        return observed, np.nan
    group_indices = [np.asarray(idx, dtype=int) for idx in work.groupby("track_name", sort=False).indices.values()]
    n_at_least_observed = 0
    for _ in range(int(n_permutations)):
        shuffled = values.copy()
        for idx in group_indices:
            if len(idx) > 1:
                shuffled[idx] = rng.permutation(shuffled[idx])
        permuted = time_bin_eta_squared_from_arrays(time_bins, shuffled)
        if np.isfinite(permuted) and permuted >= observed - 1e-15:
            n_at_least_observed += 1
    return observed, float((n_at_least_observed + 1) / (int(n_permutations) + 1))


def curve_modulation_metrics(hours: np.ndarray, values: np.ndarray, *, prefix: str) -> dict[str, float | int]:
    h = np.asarray(hours, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(h) & np.isfinite(v)
    out: dict[str, float | int] = {
        f"{prefix}_n_time_bins": int(keep.sum()),
        f"{prefix}_mean": np.nan,
        f"{prefix}_sd": np.nan,
        f"{prefix}_cv": np.nan,
        f"{prefix}_min": np.nan,
        f"{prefix}_max": np.nan,
        f"{prefix}_peak_to_trough": np.nan,
        f"{prefix}_p90_minus_p10": np.nan,
        f"{prefix}_relative_peak_to_trough": np.nan,
        f"{prefix}_mean_abs_bin_to_bin_change": np.nan,
        f"{prefix}_circular_mean_h": np.nan,
        f"{prefix}_phase_concentration": np.nan,
    }
    if not keep.any():
        return out
    h = h[keep]
    v = v[keep]
    order = np.argsort(h)
    h = h[order]
    v = v[order]
    mean = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    peak_to_trough = float(np.max(v) - np.min(v))
    p90_minus_p10 = float(np.nanpercentile(v, 90) - np.nanpercentile(v, 10))
    bin_change = np.diff(np.r_[v, v[0]]) if len(v) > 1 else np.asarray([0.0])
    circ_mean, circ_conc = circular_mean_hours(h, np.clip(v, 0.0, None))
    out.update(
        {
            f"{prefix}_mean": mean,
            f"{prefix}_sd": sd,
            f"{prefix}_cv": sd / abs(mean) if abs(mean) > 1e-12 else np.nan,
            f"{prefix}_min": float(np.min(v)),
            f"{prefix}_max": float(np.max(v)),
            f"{prefix}_peak_to_trough": peak_to_trough,
            f"{prefix}_p90_minus_p10": p90_minus_p10,
            f"{prefix}_relative_peak_to_trough": peak_to_trough / abs(mean) if abs(mean) > 1e-12 else np.nan,
            f"{prefix}_mean_abs_bin_to_bin_change": float(np.mean(np.abs(bin_change))),
            f"{prefix}_circular_mean_h": circ_mean,
            f"{prefix}_phase_concentration": circ_conc,
        }
    )
    return out


def sleep_activity_cross_correlation_for_cluster(group: pd.DataFrame) -> pd.DataFrame:
    if group.empty:
        return pd.DataFrame()
    group = group.sort_values("time_bin")
    sleep = group["sleep_fraction_valid_time"].to_numpy(np.float64)
    active = group["mean_ant_active_fraction"].to_numpy(np.float64)
    max_lag_bins = int(round(float(SLEEP_ACTIVITY_MAX_LAG_HOURS) / (float(TIME_BIN_MINUTES) / 60.0)))
    rows = []
    for lag in range(-max_lag_bins, max_lag_bins + 1):
        corr, n_overlap = lagged_correlation(
            sleep,
            active,
            lag,
            circular=True,
            min_overlap_bins=SLEEP_CROSS_CORR_MIN_OVERLAP_BINS,
        )
        rows.append(
            {
                "lag_bins_activity_vs_sleep": int(lag),
                "lag_hours_activity_vs_sleep": float(lag * TIME_BIN_MINUTES / 60.0),
                "sleep_activity_correlation": corr,
                "n_overlap_bins": n_overlap,
            }
        )
    return pd.DataFrame(rows)


def cluster_sleep_activity_modulation_summary(
    cluster_sleep_time: pd.DataFrame,
    ant_sleep_time: pd.DataFrame,
    *,
    n_permutations: int = SLEEP_TIME_MODULATION_PERMUTATIONS,
    random_state: int = SLEEP_TIME_MODULATION_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(random_state))
    summary_rows = []
    lag_rows = []
    for cluster, group in cluster_sleep_time.groupby(ACTIVITY_CLUSTER_COL, sort=True):
        group = group.sort_values("time_bin")
        hours = group["hours_since_light_on"].to_numpy(np.float64)
        sleep = group["sleep_fraction_valid_time"].to_numpy(np.float64)
        active = group["mean_ant_active_fraction"].to_numpy(np.float64)
        row: dict[str, float | int] = {
            ACTIVITY_CLUSTER_COL: cluster,
            "n_ants": int(group["n_ants"].max()) if group["n_ants"].notna().any() else 0,
        }
        row.update(curve_modulation_metrics(hours, sleep, prefix="sleep"))
        row.update(curve_modulation_metrics(hours, active, prefix="active"))
        zero_corr, zero_n = safe_pearson(sleep, active, min_n=SLEEP_CROSS_CORR_MIN_OVERLAP_BINS)
        row["sleep_activity_zero_lag_corr"] = zero_corr
        row["sleep_activity_zero_lag_overlap_bins"] = zero_n

        cluster_lags = sleep_activity_cross_correlation_for_cluster(group)
        if not cluster_lags.empty:
            cluster_lags[ACTIVITY_CLUSTER_COL] = cluster
            lag_rows.append(cluster_lags)
            valid_lags = cluster_lags.dropna(subset=["sleep_activity_correlation"])
            if not valid_lags.empty:
                best_idx = valid_lags["sleep_activity_correlation"].idxmax()
                strongest_idx = valid_lags["sleep_activity_correlation"].abs().idxmax()
                row["best_positive_sleep_activity_corr"] = float(valid_lags.loc[best_idx, "sleep_activity_correlation"])
                row["best_positive_sleep_activity_lag_h"] = float(valid_lags.loc[best_idx, "lag_hours_activity_vs_sleep"])
                row["strongest_abs_sleep_activity_corr"] = float(
                    valid_lags.loc[strongest_idx, "sleep_activity_correlation"]
                )
                row["strongest_abs_sleep_activity_lag_h"] = float(
                    valid_lags.loc[strongest_idx, "lag_hours_activity_vs_sleep"]
                )

        ant_group = ant_sleep_time[ant_sleep_time[ACTIVITY_CLUSTER_COL] == cluster]
        sleep_eta2, sleep_p = time_bin_eta_squared_permutation(
            ant_group,
            value_col="sleep_fraction_valid_time",
            rng=rng,
            n_permutations=n_permutations,
        )
        active_eta2, active_p = time_bin_eta_squared_permutation(
            ant_group,
            value_col="active_fraction",
            rng=rng,
            n_permutations=n_permutations,
        )
        row["sleep_time_bin_eta2"] = sleep_eta2
        row["sleep_time_bin_eta2_permutation_p"] = sleep_p
        row["active_time_bin_eta2"] = active_eta2
        row["active_time_bin_eta2_permutation_p"] = active_p
        row["sleep_to_active_peak_to_trough_ratio"] = (
            row["sleep_peak_to_trough"] / row["active_peak_to_trough"]
            if np.isfinite(row["active_peak_to_trough"]) and row["active_peak_to_trough"] > 1e-12
            else np.nan
        )
        row["sleep_to_active_cv_ratio"] = (
            row["sleep_cv"] / row["active_cv"]
            if np.isfinite(row["active_cv"]) and row["active_cv"] > 1e-12
            else np.nan
        )
        row["sleep_to_active_time_eta2_ratio"] = (
            row["sleep_time_bin_eta2"] / row["active_time_bin_eta2"]
            if np.isfinite(row["active_time_bin_eta2"]) and row["active_time_bin_eta2"] > 1e-12
            else np.nan
        )
        summary_rows.append(row)
    cross_corr = pd.concat(lag_rows, ignore_index=True) if lag_rows else pd.DataFrame()
    return pd.DataFrame(summary_rows), cross_corr


def plot_sleep_activity_modulation(sleep_activity_modulation: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if sleep_activity_modulation.empty:
        print("No sleep/activity modulation summary to plot")
        return
    summary = sleep_activity_modulation.sort_values(ACTIVITY_CLUSTER_COL)
    x = np.arange(len(summary))
    labels = [f"c{cluster}" for cluster in summary[ACTIVITY_CLUSTER_COL]]
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    comparisons = [
        ("peak_to_trough", "Peak-to-trough range"),
        ("cv", "CV across time bins"),
        ("time_bin_eta2", "Time-bin eta squared"),
    ]
    for ax, (suffix, ylabel) in zip(axes, comparisons):
        ax.bar(
            x - width / 2,
            summary[f"sleep_{suffix}"],
            width=width,
            label="predicted sleep",
        )
        ax.bar(x + width / 2, summary[f"active_{suffix}"], width=width, label="activity")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Time-of-day modulation: predicted sleep vs activity")
    fig.tight_layout()
    plt.show()


def plot_elapsed_cluster_sleep_timing(cluster_elapsed_sleep_time: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    clusters = sorted(cluster_elapsed_sleep_time[ACTIVITY_CLUSTER_COL].dropna().unique())
    if not clusters:
        print("No elapsed cluster sleep timing rows to plot")
        return
    nrows = len(clusters)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, max(3, 2.3 * nrows)), sharex=True, squeeze=False)
    for ax, cluster in zip(axes.ravel(), clusters):
        group = cluster_elapsed_sleep_time[cluster_elapsed_sleep_time[ACTIVITY_CLUSTER_COL] == cluster].sort_values(
            "elapsed_bin"
        )
        x = group["elapsed_hours"]
        ax.plot(
            x,
            group["sleep_fraction_valid_time"],
            color="tab:blue",
            lw=1.7,
            label="predicted sleep",
        )
        ax.plot(x, group["mean_ant_active_fraction"], color="tab:orange", lw=1.2, alpha=0.85, label="activity")
        n_ants = int(group["n_ants"].max()) if group["n_ants"].notna().any() else 0
        ax.set_title(f"activity cluster {cluster} n={n_ants}: elapsed recording time")
        ax.set_ylabel("fraction")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("elapsed recording time (h)")
    fig.tight_layout()
    plt.show()


def plot_elapsed_sleep_heatmap(ant_elapsed_sleep_time: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if ant_elapsed_sleep_time.empty:
        print("No elapsed ant sleep rows to plot")
        return
    n_bins = int(ant_elapsed_sleep_time["elapsed_bin"].max()) + 1
    peak_order = []
    for (cluster, track_name), group in ant_elapsed_sleep_time.groupby([ACTIVITY_CLUSTER_COL, "track_name"], sort=True):
        values = group.sort_values("elapsed_bin")["sleep_fraction_valid_time"].to_numpy(np.float64)
        if np.isfinite(values).any() and np.nanmax(values) > 0:
            peak_elapsed = float(group.sort_values("elapsed_bin")["elapsed_hours"].iloc[int(np.nanargmax(values))])
        else:
            peak_elapsed = np.nan
        peak_order.append(
            {
                ACTIVITY_CLUSTER_COL: cluster,
                "track_name": track_name,
                "track_id": group["track_id"].iloc[0],
                "peak_elapsed_hours": peak_elapsed,
            }
        )
    order = pd.DataFrame(peak_order).sort_values(
        [ACTIVITY_CLUSTER_COL, "peak_elapsed_hours", "track_id"],
        na_position="last",
    )
    pivot = (
        ant_elapsed_sleep_time.pivot_table(
            index="track_name",
            columns="elapsed_bin",
            values="sleep_fraction_valid_time",
            aggfunc="mean",
        )
        .reindex(index=order["track_name"], columns=np.arange(n_bins))
    )
    if pivot.empty:
        print("No elapsed sleep heatmap data")
        return
    fig, ax = plt.subplots(figsize=(13, max(4, 0.12 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(np.float64), aspect="auto", interpolation="none", vmin=0, vmax=1, cmap="Blues")
    clusters = order[ACTIVITY_CLUSTER_COL].to_numpy()
    centers = []
    labels = []
    start = 0
    for idx in range(1, len(clusters) + 1):
        if idx == len(clusters) or clusters[idx] != clusters[start]:
            if idx < len(clusters):
                ax.axhline(idx - 0.5, color="0.2", lw=0.8)
            centers.append((start + idx - 1) / 2)
            labels.append(f"c{clusters[start]} n={idx - start}")
            start = idx
    ax.set_yticks(centers)
    ax.set_yticklabels(labels)
    elapsed_hours = elapsed_bin_centers_hours(n_bins, ELAPSED_SLEEP_BIN_MINUTES)
    tick_idx = np.arange(0, n_bins, max(1, n_bins // 12))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{elapsed_hours[i]:.0f}" for i in tick_idx], rotation=45, ha="right")
    ax.set_xlabel("elapsed recording time (h)")
    ax.set_title("Predicted sleep over recording time, sorted by cluster and first peak")
    fig.colorbar(im, ax=ax, label="sleep fraction of valid observed time")
    fig.tight_layout()
    plt.show()


def sleep_repetition_autocorrelation_table(
    ant_elapsed_sleep_time: pd.DataFrame,
    *,
    value_col: str = "sleep_fraction_valid_time",
    max_lag_hours: float = SLEEP_REPETITION_MAX_LAG_HOURS,
    min_overlap_bins: int = SLEEP_REPETITION_MIN_OVERLAP_BINS,
) -> pd.DataFrame:
    if ant_elapsed_sleep_time.empty:
        return pd.DataFrame()
    n_bins = int(ant_elapsed_sleep_time["elapsed_bin"].max()) + 1
    max_lag_bins = min(
        int(round(float(max_lag_hours) / (float(ELAPSED_SLEEP_BIN_MINUTES) / 60.0))),
        max(0, n_bins - 1),
    )
    rows = []
    for (cluster, track_name), group in ant_elapsed_sleep_time.groupby([ACTIVITY_CLUSTER_COL, "track_name"], sort=True):
        series = (
            group.pivot_table(index="track_name", columns="elapsed_bin", values=value_col, aggfunc="mean")
            .reindex(columns=np.arange(n_bins))
            .iloc[0]
            .to_numpy(np.float64)
        )
        for lag in range(0, max_lag_bins + 1):
            corr, n_overlap = lagged_correlation(
                series,
                series,
                lag,
                circular=False,
                min_overlap_bins=min_overlap_bins,
            )
            rows.append(
                {
                    ACTIVITY_CLUSTER_COL: cluster,
                    "track_name": track_name,
                    "track_id": group["track_id"].iloc[0],
                    "lag_bins": int(lag),
                    "lag_hours": float(lag * ELAPSED_SLEEP_BIN_MINUTES / 60.0),
                    "sleep_autocorrelation": corr,
                    "n_overlap_bins": n_overlap,
                }
            )
    return pd.DataFrame(rows)


def summarize_sleep_repetition_autocorrelation(sleep_repetition_autocorr: pd.DataFrame) -> pd.DataFrame:
    if sleep_repetition_autocorr.empty:
        return pd.DataFrame()
    return (
        sleep_repetition_autocorr.dropna(subset=["sleep_autocorrelation"])
        .groupby([ACTIVITY_CLUSTER_COL, "lag_bins", "lag_hours"], as_index=False)
        .agg(
            median_autocorr=("sleep_autocorrelation", "median"),
            q25_autocorr=("sleep_autocorrelation", lambda x: float(np.nanpercentile(x, 25))),
            q75_autocorr=("sleep_autocorrelation", lambda x: float(np.nanpercentile(x, 75))),
            n_ants=("track_name", "nunique"),
        )
        .sort_values([ACTIVITY_CLUSTER_COL, "lag_bins"], kind="mergesort")
    )


def top_sleep_repetition_lags(
    sleep_repetition_summary: pd.DataFrame,
    *,
    min_lag_hours: float = SLEEP_REPETITION_MIN_LAG_HOURS,
    top_n: int = SLEEP_REPETITION_TOP_LAGS_PER_CLUSTER,
) -> pd.DataFrame:
    if sleep_repetition_summary.empty:
        return pd.DataFrame()
    work = sleep_repetition_summary[
        sleep_repetition_summary["lag_hours"] >= float(min_lag_hours)
    ].dropna(subset=["median_autocorr"])
    if work.empty:
        return pd.DataFrame()
    return (
        work.sort_values([ACTIVITY_CLUSTER_COL, "median_autocorr"], ascending=[True, False], kind="mergesort")
        .groupby(ACTIVITY_CLUSTER_COL, as_index=False, group_keys=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )


def top_repeating_sleep_ants(
    sleep_repetition_autocorr: pd.DataFrame,
    *,
    min_lag_hours: float = SLEEP_REPETITION_MIN_LAG_HOURS,
    top_n: int = SLEEP_REPETITION_TOP_ANTS_PER_CLUSTER,
) -> pd.DataFrame:
    if sleep_repetition_autocorr.empty:
        return pd.DataFrame()
    work = sleep_repetition_autocorr[
        sleep_repetition_autocorr["lag_hours"] >= float(min_lag_hours)
    ].dropna(subset=["sleep_autocorrelation"])
    if work.empty:
        return pd.DataFrame()
    idx = work.groupby([ACTIVITY_CLUSTER_COL, "track_name"])["sleep_autocorrelation"].idxmax()
    best = work.loc[idx].rename(
        columns={
            "lag_bins": "best_repetition_lag_bins",
            "lag_hours": "best_repetition_lag_hours",
            "sleep_autocorrelation": "best_repetition_autocorr",
            "n_overlap_bins": "best_repetition_overlap_bins",
        }
    )
    return (
        best.sort_values(
            [ACTIVITY_CLUSTER_COL, "best_repetition_autocorr"],
            ascending=[True, False],
            kind="mergesort",
        )
        .groupby(ACTIVITY_CLUSTER_COL, as_index=False, group_keys=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )


def plot_sleep_repetition_autocorrelation(sleep_repetition_summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if sleep_repetition_summary.empty:
        print("No sleep repetition autocorrelation rows to plot")
        return
    clusters = sorted(sleep_repetition_summary[ACTIVITY_CLUSTER_COL].dropna().unique())
    fig, axes = plt.subplots(len(clusters), 1, figsize=(11, max(3, 2.4 * len(clusters))), sharex=True, squeeze=False)
    for ax, cluster in zip(axes.ravel(), clusters):
        group = sleep_repetition_summary[sleep_repetition_summary[ACTIVITY_CLUSTER_COL] == cluster]
        ax.fill_between(group["lag_hours"], group["q25_autocorr"], group["q75_autocorr"], color="tab:blue", alpha=0.18)
        ax.plot(group["lag_hours"], group["median_autocorr"], color="tab:blue", lw=1.8)
        ax.axhline(0, color="0.35", lw=0.8)
        for day_h in [24, 48, 72]:
            if group["lag_hours"].max() >= day_h:
                ax.axvline(day_h, color="0.5", lw=0.8, ls=":")
        n_ants = int(group["n_ants"].max()) if group["n_ants"].notna().any() else 0
        ax.set_title(f"activity cluster {cluster}: elapsed-time sleep repetition, n={n_ants}")
        ax.set_ylabel("autocorr")
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("non-circular elapsed lag (h)")
    fig.tight_layout()
    plt.show()


def plot_repeating_sleep_ant_examples(
    ant_elapsed_sleep_time: pd.DataFrame,
    repeating_sleep_ants: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    if ant_elapsed_sleep_time.empty or repeating_sleep_ants.empty:
        print("No repeating sleep ant examples to plot")
        return
    n_bins = int(ant_elapsed_sleep_time["elapsed_bin"].max()) + 1
    elapsed_h = elapsed_bin_centers_hours(n_bins, ELAPSED_SLEEP_BIN_MINUTES)
    pivot = ant_elapsed_sleep_time.pivot_table(
        index="track_name",
        columns="elapsed_bin",
        values="sleep_fraction_valid_time",
        aggfunc="mean",
    ).reindex(columns=np.arange(n_bins))
    for cluster, examples in repeating_sleep_ants.groupby(ACTIVITY_CLUSTER_COL, sort=True):
        examples = examples.head(int(SLEEP_REPETITION_TOP_ANTS_PER_CLUSTER))
        fig, axes = plt.subplots(len(examples), 1, figsize=(12, max(3, 2.2 * len(examples))), sharex=True, squeeze=False)
        for ax, ant in zip(axes.ravel(), examples.itertuples(index=False)):
            track_name = getattr(ant, "track_name")
            if track_name not in pivot.index:
                continue
            series = pivot.loc[track_name].to_numpy(np.float64)
            lag_bins = int(getattr(ant, "best_repetition_lag_bins"))
            shifted = np.full_like(series, np.nan)
            if lag_bins > 0 and lag_bins < len(series):
                shifted[lag_bins:] = series[:-lag_bins]
            ax.plot(elapsed_h, series, color="tab:blue", lw=1.6, label="sleep")
            ax.plot(
                elapsed_h,
                shifted,
                color="tab:red",
                lw=1.2,
                ls="--",
                alpha=0.85,
                label=f"sleep shifted +{getattr(ant, 'best_repetition_lag_hours'):.1f} h",
            )
            ax.set_title(
                f"{track_name}: best repetition lag={getattr(ant, 'best_repetition_lag_hours'):.1f} h, "
                f"autocorr={getattr(ant, 'best_repetition_autocorr'):.2f}"
            )
            ax.set_ylabel("sleep frac")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="upper right")
        axes[-1, 0].set_xlabel("elapsed recording time (h)")
        fig.suptitle(f"activity cluster {cluster}: repeated predicted sleep timing examples")
        fig.tight_layout()
        plt.show()


def circular_mean_hours(hours: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    hours = np.asarray(hours, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(hours)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(hours) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return np.nan, np.nan
    angles = 2 * np.pi * (hours[keep] % 24.0) / 24.0
    w = weights[keep]
    vector = np.sum(w * np.exp(1j * angles)) / np.sum(w)
    mean_h = (np.angle(vector) % (2 * np.pi)) / (2 * np.pi) * 24.0
    concentration = np.abs(vector)
    return float(mean_h), float(concentration)


def ant_sleep_phase_table(ant_sleep_time: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cluster, track_name), group in ant_sleep_time.groupby([ACTIVITY_CLUSTER_COL, "track_name"], sort=True):
        weights = group["sleep_duration_seconds"].to_numpy(np.float64)
        hours = group["hours_since_light_on"].to_numpy(np.float64)
        mean_h, concentration = circular_mean_hours(hours, weights)
        if np.nansum(weights) > 0:
            peak_idx = int(np.nanargmax(group["sleep_fraction_valid_time"].to_numpy(np.float64)))
            peak_h = float(group["hours_since_light_on"].iloc[peak_idx])
        else:
            peak_h = np.nan
        rows.append(
            {
                ACTIVITY_CLUSTER_COL: cluster,
                "track_name": track_name,
                "track_id": group["track_id"].iloc[0],
                "total_sleep_seconds": float(np.nansum(weights)),
                "mean_sleep_phase_h": mean_h,
                "sleep_phase_concentration": concentration,
                "peak_sleep_hour": peak_h,
                "mean_sleep_fraction": float(group["sleep_fraction_valid_time"].mean()),
            }
        )
    return pd.DataFrame(rows)


def within_cluster_sleep_phase_summary(ant_phase: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, group in ant_phase.groupby(ACTIVITY_CLUSTER_COL, sort=True):
        mean_h, concentration = circular_mean_hours(
            group["mean_sleep_phase_h"].to_numpy(np.float64),
            np.ones(len(group), dtype=np.float64),
        )
        rows.append(
            {
                ACTIVITY_CLUSTER_COL: cluster,
                "n_ants": int(len(group)),
                "cluster_mean_sleep_phase_h": mean_h,
                "between_ant_phase_concentration": concentration,
                "median_total_sleep_min": float(group["total_sleep_seconds"].median() / 60.0),
                "median_sleep_fraction": float(group["mean_sleep_fraction"].median()),
            }
        )
    return pd.DataFrame(rows)


def plot_ant_sleep_heatmap(ant_sleep_time: pd.DataFrame, ant_phase: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    ordered = ant_phase.sort_values([ACTIVITY_CLUSTER_COL, "peak_sleep_hour", "track_id"], na_position="last")
    pivot = (
        ant_sleep_time.pivot_table(
            index="track_name",
            columns="time_bin",
            values="sleep_fraction_valid_time",
            aggfunc="mean",
        )
        .reindex(ordered["track_name"])
    )
    if pivot.empty:
        print("No ant sleep timing rows to plot")
        return
    fig, ax = plt.subplots(figsize=(12, max(4, 0.12 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(np.float64), aspect="auto", interpolation="none", vmin=0, vmax=1, cmap="Blues")
    boundaries = []
    labels = []
    centers = []
    start = 0
    clusters = ordered[ACTIVITY_CLUSTER_COL].to_numpy()
    for idx in range(1, len(clusters) + 1):
        if idx == len(clusters) or clusters[idx] != clusters[start]:
            boundaries.append(idx - 0.5)
            labels.append(f"c{clusters[start]} n={idx - start}")
            centers.append((start + idx - 1) / 2)
            start = idx
    for boundary in boundaries[:-1]:
        ax.axhline(boundary, color="0.2", lw=0.8)
    ax.set_yticks(centers)
    ax.set_yticklabels(labels)
    centers_h = time_bin_centers_hours(TIME_BIN_MINUTES)
    tick_idx = np.arange(0, len(centers_h), max(1, len(centers_h) // 12))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{centers_h[i]:.1f}" for i in tick_idx], rotation=45, ha="right")
    ax.set_xlabel(f"Hours since lights on at {LIGHT_ON_HOUR:g}:00")
    ax.set_title("Predicted sleep fraction by ant, sorted within activity cluster")
    fig.colorbar(im, ax=ax, label="sleep fraction of valid observed time")
    fig.tight_layout()
    plt.show()


def pairwise_time_series_cross_correlation_table(
    ant_sleep_time: pd.DataFrame,
    *,
    value_col: str = "sleep_fraction_valid_time",
    output_prefix: str = "sleep",
    max_lag_hours: float = MAX_PAIRWISE_LAG_HOURS,
    circular: bool = SLEEP_CROSS_CORR_USE_CIRCULAR_LAGS,
    min_overlap_bins: int = SLEEP_CROSS_CORR_MIN_OVERLAP_BINS,
) -> pd.DataFrame:
    n_bins = int(round(24 * 60 / float(TIME_BIN_MINUTES)))
    max_lag_bins = int(round(float(max_lag_hours) / (float(TIME_BIN_MINUTES) / 60.0)))
    lags = np.arange(-max_lag_bins, max_lag_bins + 1, dtype=int)
    rows = []
    for cluster, group in ant_sleep_time.groupby(ACTIVITY_CLUSTER_COL, sort=True):
        pivot = group.pivot_table(
            index="track_name",
            columns="time_bin",
            values=value_col,
            aggfunc="mean",
        ).reindex(columns=np.arange(n_bins))
        if len(pivot) < 2:
            continue
        if len(pivot) > int(MAX_PAIRWISE_ANTS_PER_CLUSTER):
            total_sleep = pivot.sum(axis=1, skipna=True).sort_values(ascending=False)
            pivot = pivot.loc[total_sleep.index[: int(MAX_PAIRWISE_ANTS_PER_CLUSTER)]]
        matrix = pivot.to_numpy(np.float64)
        names = pivot.index.to_list()
        finite_counts = np.isfinite(matrix).sum(axis=1)
        for i in range(len(names)):
            series_a = matrix[i]
            if finite_counts[i] < int(min_overlap_bins):
                continue
            for j in range(i + 1, len(names)):
                series_b = matrix[j]
                if finite_counts[j] < int(min_overlap_bins):
                    continue
                for lag in lags:
                    corr, n_overlap = lagged_correlation(
                        series_a,
                        series_b,
                        int(lag),
                        circular=bool(circular),
                        min_overlap_bins=int(min_overlap_bins),
                    )
                    rows.append(
                        {
                            ACTIVITY_CLUSTER_COL: cluster,
                            "track_name_a": names[i],
                            "track_name_b": names[j],
                            "lag_bins_b_vs_a": int(lag),
                            "lag_hours_b_vs_a": float(lag * TIME_BIN_MINUTES / 60.0),
                            f"{output_prefix}_cross_correlation": corr,
                            "n_overlap_bins": n_overlap,
                            "circular_lag": bool(circular),
                            "value_col": value_col,
                        }
                    )
    return pd.DataFrame(rows)


def best_cross_correlation_lags(cross_correlations: pd.DataFrame, *, output_prefix: str) -> pd.DataFrame:
    if cross_correlations.empty:
        return pd.DataFrame()
    keys = [ACTIVITY_CLUSTER_COL, "track_name_a", "track_name_b"]
    corr_col = f"{output_prefix}_cross_correlation"
    valid = cross_correlations.dropna(subset=[corr_col]).copy()
    if valid.empty:
        return pd.DataFrame()
    best_idx = valid.groupby(keys)[corr_col].idxmax()
    best = valid.loc[best_idx, keys + ["lag_bins_b_vs_a", "lag_hours_b_vs_a", corr_col, "n_overlap_bins"]]
    best = best.rename(
        columns={
            "lag_bins_b_vs_a": f"{output_prefix}_best_lag_bins_b_vs_a",
            "lag_hours_b_vs_a": f"{output_prefix}_best_lag_hours_b_vs_a",
            corr_col: f"{output_prefix}_best_correlation",
            "n_overlap_bins": f"{output_prefix}_best_overlap_bins",
        }
    )
    strongest_idx = valid.assign(abs_corr=valid[corr_col].abs()).groupby(keys)["abs_corr"].idxmax()
    strongest = valid.loc[
        strongest_idx,
        keys + ["lag_bins_b_vs_a", "lag_hours_b_vs_a", corr_col, "n_overlap_bins"],
    ].rename(
        columns={
            "lag_bins_b_vs_a": f"{output_prefix}_strongest_abs_lag_bins_b_vs_a",
            "lag_hours_b_vs_a": f"{output_prefix}_strongest_abs_lag_hours_b_vs_a",
            corr_col: f"{output_prefix}_strongest_abs_correlation",
            "n_overlap_bins": f"{output_prefix}_strongest_abs_overlap_bins",
        }
    )
    zero = valid[valid["lag_bins_b_vs_a"] == 0][keys + [corr_col, "n_overlap_bins"]].rename(
        columns={
            corr_col: f"{output_prefix}_zero_lag_correlation",
            "n_overlap_bins": f"{output_prefix}_zero_lag_overlap_bins",
        }
    )
    out = best.merge(strongest, on=keys, how="left").merge(zero, on=keys, how="left")
    return out.sort_values(keys, kind="mergesort").reset_index(drop=True)


def summarize_cross_correlations(
    cross_correlations: pd.DataFrame,
    pairwise_lags: pd.DataFrame,
    *,
    output_prefix: str,
) -> pd.DataFrame:
    if pairwise_lags.empty:
        return pd.DataFrame()
    corr_col = f"{output_prefix}_cross_correlation"
    rows = []
    for cluster, group in pairwise_lags.groupby(ACTIVITY_CLUSTER_COL, sort=True):
        zero = group[f"{output_prefix}_zero_lag_correlation"].dropna()
        best = group[f"{output_prefix}_best_correlation"].dropna()
        lags = group[f"{output_prefix}_best_lag_hours_b_vs_a"].dropna()
        rows.append(
            {
                ACTIVITY_CLUSTER_COL: cluster,
                "n_pairs": int(len(group)),
                f"{output_prefix}_median_zero_lag_corr": float(zero.median()) if len(zero) else np.nan,
                f"{output_prefix}_median_best_corr": float(best.median()) if len(best) else np.nan,
                f"{output_prefix}_median_best_lag_h": float(lags.median()) if len(lags) else np.nan,
                f"{output_prefix}_median_abs_best_lag_h": float(np.nanmedian(np.abs(lags))) if len(lags) else np.nan,
                f"{output_prefix}_frac_best_lag_within_one_bin": float((np.abs(lags) <= TIME_BIN_MINUTES / 60.0).mean())
                if len(lags)
                else np.nan,
                f"{output_prefix}_frac_positive_zero_lag_corr": float((zero > 0).mean()) if len(zero) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    if not cross_correlations.empty:
        lag_summary = (
            cross_correlations.dropna(subset=[corr_col])
            .groupby([ACTIVITY_CLUSTER_COL, "lag_bins_b_vs_a"], as_index=False)
            .agg(median_corr=(corr_col, "median"))
        )
        zero_lag = lag_summary[lag_summary["lag_bins_b_vs_a"] == 0][
            [ACTIVITY_CLUSTER_COL, "median_corr"]
        ].rename(columns={"median_corr": f"{output_prefix}_median_cross_corr_at_lag0"})
        summary = summary.merge(zero_lag, on=ACTIVITY_CLUSTER_COL, how="left")
    return summary


def plot_cross_correlations(
    cross_correlations: pd.DataFrame,
    *,
    output_prefix: str,
    title_prefix: str,
    color: str,
) -> None:
    import matplotlib.pyplot as plt

    corr_col = f"{output_prefix}_cross_correlation"
    if cross_correlations.empty:
        print(f"No pairwise {title_prefix.lower()} cross-correlations to plot")
        return
    valid = cross_correlations.dropna(subset=[corr_col]).copy()
    if valid.empty:
        print(f"Pairwise {title_prefix.lower()} cross-correlations were all undefined, likely because traces were flat")
        return
    summary = (
        valid.groupby([ACTIVITY_CLUSTER_COL, "lag_hours_b_vs_a"], as_index=False)
        .agg(
            median_corr=(corr_col, "median"),
            q25_corr=(corr_col, lambda x: float(np.nanpercentile(x, 25))),
            q75_corr=(corr_col, lambda x: float(np.nanpercentile(x, 75))),
            n_pairs=(corr_col, "size"),
        )
        .sort_values([ACTIVITY_CLUSTER_COL, "lag_hours_b_vs_a"], kind="mergesort")
    )
    clusters = sorted(summary[ACTIVITY_CLUSTER_COL].dropna().unique())
    fig, axes = plt.subplots(len(clusters), 1, figsize=(10, max(3, 2.2 * len(clusters))), sharex=True, squeeze=False)
    for ax, cluster in zip(axes.ravel(), clusters):
        group = summary[summary[ACTIVITY_CLUSTER_COL] == cluster]
        ax.fill_between(group["lag_hours_b_vs_a"], group["q25_corr"], group["q75_corr"], color=color, alpha=0.18)
        ax.plot(group["lag_hours_b_vs_a"], group["median_corr"], color=color, lw=2.0)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.axvline(0, color="0.4", lw=0.8, ls="--")
        n_pairs = int(group["n_pairs"].max()) if group["n_pairs"].notna().any() else 0
        ax.set_title(f"activity cluster {cluster}: ant-ant {title_prefix} cross-correlation, n_pairs={n_pairs}")
        ax.set_ylabel("corr")
        ax.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Lag hours, ant B vs ant A; positive means B is delayed relative to A")
    fig.tight_layout()
    plt.show()


def pairwise_sleep_cross_correlation_table(ant_sleep_time: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return pairwise_time_series_cross_correlation_table(
        ant_sleep_time,
        value_col="sleep_fraction_valid_time",
        output_prefix="sleep",
        **kwargs,
    )


def pairwise_activity_cross_correlation_table(ant_sleep_time: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return pairwise_time_series_cross_correlation_table(
        ant_sleep_time,
        value_col="active_fraction",
        output_prefix="activity",
        **kwargs,
    )


def best_sleep_cross_correlation_lags(sleep_cross_correlations: pd.DataFrame) -> pd.DataFrame:
    return best_cross_correlation_lags(sleep_cross_correlations, output_prefix="sleep")


def best_activity_cross_correlation_lags(activity_cross_correlations: pd.DataFrame) -> pd.DataFrame:
    return best_cross_correlation_lags(activity_cross_correlations, output_prefix="activity")


def summarize_sleep_cross_correlations(
    sleep_cross_correlations: pd.DataFrame,
    pairwise_sleep_lags: pd.DataFrame,
) -> pd.DataFrame:
    return summarize_cross_correlations(
        sleep_cross_correlations,
        pairwise_sleep_lags,
        output_prefix="sleep",
    )


def summarize_activity_cross_correlations(
    activity_cross_correlations: pd.DataFrame,
    pairwise_activity_lags: pd.DataFrame,
) -> pd.DataFrame:
    return summarize_cross_correlations(
        activity_cross_correlations,
        pairwise_activity_lags,
        output_prefix="activity",
    )


def plot_sleep_cross_correlations(sleep_cross_correlations: pd.DataFrame) -> None:
    plot_cross_correlations(
        sleep_cross_correlations,
        output_prefix="sleep",
        title_prefix="predicted sleep",
        color="tab:blue",
    )


def plot_activity_cross_correlations(activity_cross_correlations: pd.DataFrame) -> None:
    plot_cross_correlations(
        activity_cross_correlations,
        output_prefix="activity",
        title_prefix="activity",
        color="tab:orange",
    )


def paired_sleep_activity_synchrony_table(
    pairwise_sleep_lags: pd.DataFrame,
    pairwise_activity_lags: pd.DataFrame,
) -> pd.DataFrame:
    keys = [ACTIVITY_CLUSTER_COL, "track_name_a", "track_name_b"]
    if pairwise_sleep_lags.empty or pairwise_activity_lags.empty:
        return pd.DataFrame()
    out = pairwise_sleep_lags.merge(pairwise_activity_lags, on=keys, how="inner", validate="one_to_one")
    if out.empty:
        return out
    out["sleep_minus_activity_zero_lag_corr"] = (
        out["sleep_zero_lag_correlation"] - out["activity_zero_lag_correlation"]
    )
    out["sleep_minus_activity_best_corr"] = out["sleep_best_correlation"] - out["activity_best_correlation"]
    out["sleep_minus_activity_abs_best_lag_h"] = (
        out["sleep_best_lag_hours_b_vs_a"].abs() - out["activity_best_lag_hours_b_vs_a"].abs()
    )
    out["sleep_more_synchronous_zero_lag"] = out["sleep_zero_lag_correlation"] > out["activity_zero_lag_correlation"]
    out["sleep_more_synchronous_best_corr"] = out["sleep_best_correlation"] > out["activity_best_correlation"]
    out["sleep_best_lag_near_zero"] = out["sleep_best_lag_hours_b_vs_a"].abs() <= TIME_BIN_MINUTES / 60.0
    out["activity_best_lag_near_zero"] = out["activity_best_lag_hours_b_vs_a"].abs() <= TIME_BIN_MINUTES / 60.0
    return out


def summarize_paired_sleep_activity_synchrony(paired_synchrony: pd.DataFrame) -> pd.DataFrame:
    if paired_synchrony.empty:
        return pd.DataFrame()
    rows = []
    for cluster, group in paired_synchrony.groupby(ACTIVITY_CLUSTER_COL, sort=True):
        rows.append(
            {
                ACTIVITY_CLUSTER_COL: cluster,
                "n_pairs": int(len(group)),
                "sleep_median_zero_lag_corr": float(group["sleep_zero_lag_correlation"].median()),
                "activity_median_zero_lag_corr": float(group["activity_zero_lag_correlation"].median()),
                "sleep_minus_activity_median_zero_lag_corr": float(
                    group["sleep_minus_activity_zero_lag_corr"].median()
                ),
                "sleep_median_best_corr": float(group["sleep_best_correlation"].median()),
                "activity_median_best_corr": float(group["activity_best_correlation"].median()),
                "sleep_minus_activity_median_best_corr": float(group["sleep_minus_activity_best_corr"].median()),
                "sleep_median_abs_best_lag_h": float(np.nanmedian(group["sleep_best_lag_hours_b_vs_a"].abs())),
                "activity_median_abs_best_lag_h": float(np.nanmedian(group["activity_best_lag_hours_b_vs_a"].abs())),
                "sleep_frac_best_lag_near_zero": float(group["sleep_best_lag_near_zero"].mean()),
                "activity_frac_best_lag_near_zero": float(group["activity_best_lag_near_zero"].mean()),
                "frac_sleep_zero_lag_corr_gt_activity": float(group["sleep_more_synchronous_zero_lag"].mean()),
                "frac_sleep_best_corr_gt_activity": float(group["sleep_more_synchronous_best_corr"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_sleep_activity_synchrony_comparison(paired_synchrony: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if paired_synchrony.empty:
        print("No paired sleep/activity synchrony rows to plot")
        return
    summary = summarize_paired_sleep_activity_synchrony(paired_synchrony)
    if summary.empty:
        print("No paired sleep/activity synchrony summary to plot")
        return
    summary = summary.sort_values(ACTIVITY_CLUSTER_COL)
    x = np.arange(len(summary))
    labels = [f"c{cluster}" for cluster in summary[ACTIVITY_CLUSTER_COL]]
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    axes[0].bar(x - width / 2, summary["sleep_median_zero_lag_corr"], width=width, label="sleep")
    axes[0].bar(x + width / 2, summary["activity_median_zero_lag_corr"], width=width, label="activity")
    axes[0].set_ylabel("median zero-lag corr")

    axes[1].bar(x - width / 2, summary["sleep_median_best_corr"], width=width, label="sleep")
    axes[1].bar(x + width / 2, summary["activity_median_best_corr"], width=width, label="activity")
    axes[1].set_ylabel("median best corr")

    axes[2].bar(x - width / 2, summary["sleep_median_abs_best_lag_h"], width=width, label="sleep")
    axes[2].bar(x + width / 2, summary["activity_median_abs_best_lag_h"], width=width, label="activity")
    axes[2].set_ylabel("median abs best lag (h)")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Within-cluster ant-ant synchrony: predicted sleep vs activity")
    fig.tight_layout()
    plt.show()


def shifted_sleep_pair_candidates(
    pairwise_sleep_lags: pd.DataFrame,
    *,
    min_best_corr: float = SHIFTED_SLEEP_MIN_BEST_CORR,
    max_zero_lag_corr: float = SHIFTED_SLEEP_MAX_ZERO_LAG_CORR,
    min_corr_gain: float = SHIFTED_SLEEP_MIN_CORR_GAIN,
    min_abs_lag_hours: float = SHIFTED_SLEEP_MIN_ABS_LAG_HOURS,
) -> pd.DataFrame:
    if pairwise_sleep_lags.empty:
        return pd.DataFrame()
    required = [
        ACTIVITY_CLUSTER_COL,
        "track_name_a",
        "track_name_b",
        "sleep_zero_lag_correlation",
        "sleep_best_correlation",
        "sleep_best_lag_bins_b_vs_a",
        "sleep_best_lag_hours_b_vs_a",
    ]
    missing = [col for col in required if col not in pairwise_sleep_lags.columns]
    if missing:
        raise KeyError(f"Missing pairwise sleep lag columns: {missing}")
    work = pairwise_sleep_lags.copy()
    work["sleep_abs_best_lag_h"] = work["sleep_best_lag_hours_b_vs_a"].abs()
    work["sleep_corr_gain_from_lag"] = work["sleep_best_correlation"] - work["sleep_zero_lag_correlation"]
    work["shifted_sleep_score"] = (
        work["sleep_corr_gain_from_lag"].clip(lower=0)
        * work["sleep_abs_best_lag_h"]
        * work["sleep_best_correlation"].clip(lower=0)
    )
    keep = (
        (work["sleep_best_correlation"] >= float(min_best_corr))
        & (work["sleep_zero_lag_correlation"] <= float(max_zero_lag_corr))
        & (work["sleep_corr_gain_from_lag"] >= float(min_corr_gain))
        & (work["sleep_abs_best_lag_h"] >= float(min_abs_lag_hours))
    )
    return (
        work[keep]
        .sort_values([ACTIVITY_CLUSTER_COL, "shifted_sleep_score"], ascending=[True, False], kind="mergesort")
        .reset_index(drop=True)
    )


def summarize_shifted_sleep_candidates(shifted_sleep_pairs: pd.DataFrame) -> pd.DataFrame:
    if shifted_sleep_pairs.empty:
        return pd.DataFrame()
    return (
        shifted_sleep_pairs.groupby(ACTIVITY_CLUSTER_COL, as_index=False)
        .agg(
            n_shifted_pairs=("shifted_sleep_score", "size"),
            median_abs_best_lag_h=("sleep_abs_best_lag_h", "median"),
            median_corr_gain_from_lag=("sleep_corr_gain_from_lag", "median"),
            median_best_corr=("sleep_best_correlation", "median"),
            median_zero_lag_corr=("sleep_zero_lag_correlation", "median"),
            max_shifted_sleep_score=("shifted_sleep_score", "max"),
        )
        .sort_values(ACTIVITY_CLUSTER_COL, kind="mergesort")
    )


def plot_sleep_shift_lag_distributions(
    pairwise_sleep_lags: pd.DataFrame,
    paired_synchrony: pd.DataFrame | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if pairwise_sleep_lags.empty:
        print("No pairwise sleep lag rows to plot")
        return
    sleep = pairwise_sleep_lags.dropna(subset=["sleep_best_lag_hours_b_vs_a"]).copy()
    if sleep.empty:
        print("No finite sleep best lags to plot")
        return
    clusters = sorted(sleep[ACTIVITY_CLUSTER_COL].dropna().unique())
    fig, axes = plt.subplots(len(clusters), 2, figsize=(12, max(3, 2.8 * len(clusters))), squeeze=False)
    bin_edges = np.arange(
        -float(MAX_PAIRWISE_LAG_HOURS) - TIME_BIN_MINUTES / 120.0,
        float(MAX_PAIRWISE_LAG_HOURS) + TIME_BIN_MINUTES / 60.0,
        TIME_BIN_MINUTES / 60.0,
    )
    for row_idx, cluster in enumerate(clusters):
        group = sleep[sleep[ACTIVITY_CLUSTER_COL] == cluster]
        axes[row_idx, 0].hist(
            group["sleep_best_lag_hours_b_vs_a"],
            bins=bin_edges,
            color="tab:blue",
            alpha=0.75,
        )
        axes[row_idx, 0].axvline(0, color="0.25", lw=0.9, ls="--")
        axes[row_idx, 0].set_title(f"cluster {cluster}: sleep best-lag distribution")
        axes[row_idx, 0].set_ylabel("ant pairs")
        axes[row_idx, 0].grid(True, axis="y", alpha=0.25)

        axes[row_idx, 1].scatter(
            group["sleep_abs_best_lag_h"] if "sleep_abs_best_lag_h" in group.columns else group["sleep_best_lag_hours_b_vs_a"].abs(),
            group["sleep_best_correlation"] - group["sleep_zero_lag_correlation"],
            c=group["sleep_best_correlation"],
            cmap="viridis",
            vmin=-1,
            vmax=1,
            s=18,
            alpha=0.75,
        )
        axes[row_idx, 1].axhline(0, color="0.25", lw=0.9, ls="--")
        axes[row_idx, 1].axvline(SHIFTED_SLEEP_MIN_ABS_LAG_HOURS, color="0.5", lw=0.8, ls=":")
        axes[row_idx, 1].set_title("lag benefit vs lag size")
        axes[row_idx, 1].set_ylabel("best corr - zero-lag corr")
        axes[row_idx, 1].grid(True, alpha=0.25)

        if paired_synchrony is not None and not paired_synchrony.empty:
            paired = paired_synchrony[paired_synchrony[ACTIVITY_CLUSTER_COL] == cluster]
            if not paired.empty:
                axes[row_idx, 0].hist(
                    paired["activity_best_lag_hours_b_vs_a"],
                    bins=bin_edges,
                    color="tab:orange",
                    alpha=0.35,
                    label="activity",
                )
                axes[row_idx, 0].hist(
                    paired["sleep_best_lag_hours_b_vs_a"],
                    bins=bin_edges,
                    histtype="step",
                    color="tab:blue",
                    lw=1.4,
                    label="sleep",
                )
                axes[row_idx, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("best lag hours, B vs A")
    axes[-1, 1].set_xlabel("absolute best lag hours")
    fig.suptitle("Shifted predicted sleep candidates: lag distributions and lag benefit")
    fig.tight_layout()
    plt.show()


def plot_sleep_shift_correlation_scatter(paired_synchrony: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if paired_synchrony.empty:
        print("No paired sleep/activity synchrony rows to plot")
        return
    plot_df = paired_synchrony.dropna(
        subset=[
            "sleep_zero_lag_correlation",
            "sleep_best_correlation",
            "sleep_best_lag_hours_b_vs_a",
            "activity_zero_lag_correlation",
            "activity_best_correlation",
            "activity_best_lag_hours_b_vs_a",
        ]
    ).copy()
    if plot_df.empty:
        print("No finite paired synchrony rows to plot")
        return
    plot_df["sleep_abs_best_lag_h"] = plot_df["sleep_best_lag_hours_b_vs_a"].abs()
    plot_df["activity_abs_best_lag_h"] = plot_df["activity_best_lag_hours_b_vs_a"].abs()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sc = axes[0].scatter(
        plot_df["sleep_zero_lag_correlation"],
        plot_df["sleep_best_correlation"],
        c=plot_df["sleep_abs_best_lag_h"],
        cmap="magma",
        s=18,
        alpha=0.75,
    )
    axes[0].plot([-1, 1], [-1, 1], color="0.35", lw=0.8, ls="--")
    axes[0].axvline(SHIFTED_SLEEP_MAX_ZERO_LAG_CORR, color="0.5", lw=0.8, ls=":")
    axes[0].axhline(SHIFTED_SLEEP_MIN_BEST_CORR, color="0.5", lw=0.8, ls=":")
    axes[0].set_xlabel("sleep zero-lag corr")
    axes[0].set_ylabel("sleep best lagged corr")
    axes[0].set_title("high best corr + low zero-lag = shifted candidate")
    fig.colorbar(sc, ax=axes[0], label="sleep abs best lag (h)")

    axes[1].scatter(
        plot_df["sleep_abs_best_lag_h"],
        plot_df["sleep_best_correlation"] - plot_df["sleep_zero_lag_correlation"],
        color="tab:blue",
        s=18,
        alpha=0.65,
        label="sleep",
    )
    axes[1].scatter(
        plot_df["activity_abs_best_lag_h"],
        plot_df["activity_best_correlation"] - plot_df["activity_zero_lag_correlation"],
        color="tab:orange",
        s=18,
        alpha=0.45,
        label="activity",
    )
    axes[1].axhline(SHIFTED_SLEEP_MIN_CORR_GAIN, color="0.5", lw=0.8, ls=":")
    axes[1].axvline(SHIFTED_SLEEP_MIN_ABS_LAG_HOURS, color="0.5", lw=0.8, ls=":")
    axes[1].set_xlabel("absolute best lag (h)")
    axes[1].set_ylabel("best corr - zero-lag corr")
    axes[1].set_title("does lag help sleep more than activity?")
    axes[1].legend(fontsize=8)

    axes[2].scatter(
        plot_df["activity_best_lag_hours_b_vs_a"],
        plot_df["sleep_best_lag_hours_b_vs_a"],
        c=plot_df["sleep_best_correlation"] - plot_df["sleep_zero_lag_correlation"],
        cmap="viridis",
        s=18,
        alpha=0.75,
    )
    axes[2].axhline(0, color="0.35", lw=0.8, ls="--")
    axes[2].axvline(0, color="0.35", lw=0.8, ls="--")
    axes[2].plot(
        [-MAX_PAIRWISE_LAG_HOURS, MAX_PAIRWISE_LAG_HOURS],
        [-MAX_PAIRWISE_LAG_HOURS, MAX_PAIRWISE_LAG_HOURS],
        color="0.5",
        lw=0.8,
        ls=":",
    )
    axes[2].set_xlabel("activity best lag (h)")
    axes[2].set_ylabel("sleep best lag (h)")
    axes[2].set_title("same ant pair: sleep lag vs activity lag")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()


def plot_sleep_phase_shift_heatmaps(ant_sleep_time: pd.DataFrame, ant_phase: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if ant_sleep_time.empty or ant_phase.empty:
        print("No ant sleep timing rows to plot")
        return
    n_bins = int(round(24 * 60 / float(TIME_BIN_MINUTES)))
    centers_h = time_bin_centers_hours(TIME_BIN_MINUTES)
    clusters = sorted(ant_phase[ACTIVITY_CLUSTER_COL].dropna().unique())
    for cluster in clusters:
        ordered = ant_phase[ant_phase[ACTIVITY_CLUSTER_COL] == cluster].sort_values(
            ["peak_sleep_hour", "mean_sleep_phase_h", "track_id"],
            na_position="last",
        )
        group = ant_sleep_time[ant_sleep_time[ACTIVITY_CLUSTER_COL] == cluster]
        pivot = (
            group.pivot_table(
                index="track_name",
                columns="time_bin",
                values="sleep_fraction_valid_time",
                aggfunc="mean",
            )
            .reindex(index=ordered["track_name"], columns=np.arange(n_bins))
        )
        if pivot.empty:
            continue
        raw = pivot.to_numpy(np.float64)
        aligned = raw.copy()
        peak_bins = []
        for idx in range(aligned.shape[0]):
            row = aligned[idx]
            if np.isfinite(row).any() and np.nanmax(row) > 0:
                peak_bin = int(np.nanargmax(np.nan_to_num(row, nan=-np.inf)))
                aligned[idx] = np.roll(row, -peak_bin)
                peak_bins.append(peak_bin)
            else:
                peak_bins.append(0)
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(13, max(3.5, 0.16 * len(pivot))),
            sharey=True,
            constrained_layout=True,
        )
        im0 = axes[0].imshow(raw, aspect="auto", interpolation="none", vmin=0, vmax=1, cmap="Blues")
        axes[1].imshow(aligned, aspect="auto", interpolation="none", vmin=0, vmax=1, cmap="Blues")
        for ax, title in zip(axes, ["phase sorted raw sleep", "same ants shifted to align own peak"]):
            tick_idx = np.arange(0, n_bins, max(1, n_bins // 12))
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([f"{centers_h[i]:.1f}" for i in tick_idx], rotation=45, ha="right")
            ax.set_xlabel(f"hours since lights on at {LIGHT_ON_HOUR:g}:00")
            ax.set_title(title)
            ax.set_ylabel("ants ordered by sleep phase")
        fig.colorbar(im0, ax=axes, label="predicted sleep fraction")
        fig.suptitle(f"activity cluster {cluster}: stable shifted sleep schedule check")
        plt.show()


def plot_sleep_best_lag_matrix(pairwise_sleep_lags: pd.DataFrame, ant_phase: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if pairwise_sleep_lags.empty or ant_phase.empty:
        print("No pairwise sleep lag rows to plot")
        return
    clusters = sorted(ant_phase[ACTIVITY_CLUSTER_COL].dropna().unique())
    for cluster in clusters:
        ordered = ant_phase[ant_phase[ACTIVITY_CLUSTER_COL] == cluster].sort_values(
            ["peak_sleep_hour", "mean_sleep_phase_h", "track_id"],
            na_position="last",
        )
        names = ordered["track_name"].to_list()
        if len(names) < 2:
            continue
        index = {name: i for i, name in enumerate(names)}
        matrix = np.full((len(names), len(names)), np.nan, dtype=np.float64)
        corr_matrix = np.full_like(matrix, np.nan)
        np.fill_diagonal(matrix, 0.0)
        np.fill_diagonal(corr_matrix, 1.0)
        pairs = pairwise_sleep_lags[pairwise_sleep_lags[ACTIVITY_CLUSTER_COL] == cluster]
        for pair in pairs.itertuples(index=False):
            name_a = getattr(pair, "track_name_a")
            name_b = getattr(pair, "track_name_b")
            if name_a not in index or name_b not in index:
                continue
            i = index[name_a]
            j = index[name_b]
            lag = float(getattr(pair, "sleep_best_lag_hours_b_vs_a"))
            corr = float(getattr(pair, "sleep_best_correlation"))
            matrix[i, j] = lag
            matrix[j, i] = -lag
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
        fig, axes = plt.subplots(1, 2, figsize=(11, max(4, 0.16 * len(names))), constrained_layout=True)
        vmax = max(float(MAX_PAIRWISE_LAG_HOURS), float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0)
        im0 = axes[0].imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="none")
        im1 = axes[1].imshow(corr_matrix, cmap="viridis", vmin=-1, vmax=1, interpolation="none")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("ants ordered by sleep phase")
        axes[0].set_ylabel("ants ordered by sleep phase")
        axes[0].set_title("best sleep lag, column vs row (h)")
        axes[1].set_title("best sleep correlation")
        fig.colorbar(im0, ax=axes[0], label="lag h")
        fig.colorbar(im1, ax=axes[1], label="corr")
        fig.suptitle(f"activity cluster {cluster}: pairwise shifted sleep structure")
        plt.show()


def plot_shifted_sleep_pair_examples(
    ant_sleep_time: pd.DataFrame,
    shifted_sleep_pairs: pd.DataFrame,
    *,
    max_pairs_per_cluster: int = SHIFTED_SLEEP_EXAMPLE_PAIRS_PER_CLUSTER,
) -> None:
    import matplotlib.pyplot as plt

    if ant_sleep_time.empty or shifted_sleep_pairs.empty:
        print("No shifted sleep pair examples to plot")
        return
    n_bins = int(round(24 * 60 / float(TIME_BIN_MINUTES)))
    centers_h = time_bin_centers_hours(TIME_BIN_MINUTES)
    for cluster, candidates in shifted_sleep_pairs.groupby(ACTIVITY_CLUSTER_COL, sort=True):
        candidates = candidates.head(int(max_pairs_per_cluster))
        group = ant_sleep_time[ant_sleep_time[ACTIVITY_CLUSTER_COL] == cluster]
        pivot = group.pivot_table(
            index="track_name",
            columns="time_bin",
            values="sleep_fraction_valid_time",
            aggfunc="mean",
        ).reindex(columns=np.arange(n_bins))
        if pivot.empty:
            continue
        fig, axes = plt.subplots(len(candidates), 1, figsize=(12, max(3, 2.4 * len(candidates))), sharex=True, squeeze=False)
        for ax, pair in zip(axes.ravel(), candidates.itertuples(index=False)):
            name_a = getattr(pair, "track_name_a")
            name_b = getattr(pair, "track_name_b")
            if name_a not in pivot.index or name_b not in pivot.index:
                continue
            sleep_a = pivot.loc[name_a].to_numpy(np.float64)
            sleep_b = pivot.loc[name_b].to_numpy(np.float64)
            lag_bins = int(getattr(pair, "sleep_best_lag_bins_b_vs_a"))
            sleep_b_aligned = np.roll(sleep_b, -lag_bins)
            ax.plot(centers_h, sleep_a, color="tab:blue", lw=1.8, label=f"A {name_a}")
            ax.plot(centers_h, sleep_b, color="tab:orange", lw=1.4, alpha=0.75, label=f"B {name_b} raw")
            ax.plot(
                centers_h,
                sleep_b_aligned,
                color="tab:red",
                lw=1.4,
                alpha=0.9,
                ls="--",
                label=f"B shifted {-lag_bins} bins",
            )
            ax.set_ylabel("sleep frac")
            ax.set_title(
                "lag={lag:.2f} h, zero={zero:.2f}, best={best:.2f}, gain={gain:.2f}".format(
                    lag=float(getattr(pair, "sleep_best_lag_hours_b_vs_a")),
                    zero=float(getattr(pair, "sleep_zero_lag_correlation")),
                    best=float(getattr(pair, "sleep_best_correlation")),
                    gain=float(getattr(pair, "sleep_corr_gain_from_lag")),
                )
            )
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, ncols=3, loc="upper right")
        axes[-1, 0].set_xlabel(f"hours since lights on at {LIGHT_ON_HOUR:g}:00")
        fig.suptitle(f"activity cluster {cluster}: shifted predicted sleep pair examples")
        fig.tight_layout()
        plt.show()


def normalised_interaction_rows(interactions: pd.DataFrame) -> pd.DataFrame:
    columns = ["global_frame", "antenna_track_id", "body_track_id"]
    if interactions.empty:
        return pd.DataFrame(columns=columns)
    if "global_frame" in interactions.columns:
        work = interactions[columns].copy()
    else:
        work = interactions[["Frame", "antenna_track_id", "body_track_id"]].copy()
        work["global_frame"] = work["Frame"]
    for col in columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=columns).copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work["global_frame"] = np.rint(work["global_frame"]).astype(np.int64)
    work["antenna_track_id"] = work["antenna_track_id"].astype(int)
    work["body_track_id"] = work["body_track_id"].astype(int)
    work = work[work["antenna_track_id"] != work["body_track_id"]].copy()
    return work[columns].reset_index(drop=True)


def validate_interaction_role(role: str) -> str:
    role = str(role)
    if role not in {"body", "antenna", "both"}:
        raise ValueError("interaction role must be one of: 'body', 'antenna', 'both'")
    return role


def interaction_onset_events(
    interactions: pd.DataFrame,
    *,
    fps: float,
    event_gap_seconds: float = INTERACTION_EVENT_GAP_SECONDS,
    track_ids: set[int] | list[int] | None = None,
    analysis_frame_start: int | None = None,
    analysis_frame_stop: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return encounter onsets, not every frame-level contact detection."""
    work = normalised_interaction_rows(interactions)
    event_columns = ["global_frame", "track_id", "partner_track_id"]
    role_columns = ["global_frame", "track_id", "partner_track_id", "interaction_role"]
    if work.empty:
        return pd.DataFrame(columns=event_columns), pd.DataFrame(columns=role_columns)

    if analysis_frame_start is not None:
        work = work[work["global_frame"] >= int(analysis_frame_start)]
    if analysis_frame_stop is not None:
        work = work[work["global_frame"] < int(analysis_frame_stop)]
    track_id_set = None
    if track_ids is not None:
        track_id_set = {int(track_id) for track_id in track_ids}
        work = work[
            work["antenna_track_id"].isin(track_id_set) | work["body_track_id"].isin(track_id_set)
        ].copy()
    if work.empty:
        return pd.DataFrame(columns=event_columns), pd.DataFrame(columns=role_columns)

    antenna = work.rename(
        columns={"antenna_track_id": "track_id", "body_track_id": "partner_track_id"}
    )[["global_frame", "track_id", "partner_track_id"]].copy()
    antenna["interaction_role"] = "antenna"
    body = work.rename(
        columns={"body_track_id": "track_id", "antenna_track_id": "partner_track_id"}
    )[["global_frame", "track_id", "partner_track_id"]].copy()
    body["interaction_role"] = "body"
    participant_rows = pd.concat([antenna, body], ignore_index=True)
    if track_id_set is not None:
        participant_rows = participant_rows[participant_rows["track_id"].isin(track_id_set)].copy()
    if participant_rows.empty:
        return pd.DataFrame(columns=event_columns), pd.DataFrame(columns=role_columns)

    gap_frames = max(1, int(round(float(event_gap_seconds) * float(fps))))
    total_rows = participant_rows.drop_duplicates(["track_id", "partner_track_id", "global_frame"])
    total_rows = total_rows.sort_values(["track_id", "partner_track_id", "global_frame"], kind="mergesort")
    total_gap = total_rows.groupby(["track_id", "partner_track_id"], sort=False)["global_frame"].diff()
    total_onsets = total_rows[total_gap.isna() | (total_gap > gap_frames)][event_columns].copy()

    role_rows = participant_rows.drop_duplicates(
        ["track_id", "partner_track_id", "interaction_role", "global_frame"]
    )
    role_rows = role_rows.sort_values(
        ["track_id", "partner_track_id", "interaction_role", "global_frame"],
        kind="mergesort",
    )
    role_gap = role_rows.groupby(
        ["track_id", "partner_track_id", "interaction_role"],
        sort=False,
    )["global_frame"].diff()
    role_onsets = role_rows[role_gap.isna() | (role_gap > gap_frames)][role_columns].copy()
    return (
        total_onsets.sort_values(["track_id", "global_frame"], kind="mergesort").reset_index(drop=True),
        role_onsets.sort_values(["track_id", "global_frame"], kind="mergesort").reset_index(drop=True),
    )


def build_interaction_onset_counts_by_track(
    interactions: pd.DataFrame,
    *,
    fps: float,
    event_gap_seconds: float = INTERACTION_EVENT_GAP_SECONDS,
    analysis_frame_start: int | None = None,
    analysis_frame_stop: int | None = None,
) -> dict[int, pd.DataFrame]:
    total_onsets, role_onsets = interaction_onset_events(
        interactions,
        fps=fps,
        event_gap_seconds=event_gap_seconds,
        analysis_frame_start=analysis_frame_start,
        analysis_frame_stop=analysis_frame_stop,
    )
    if total_onsets.empty:
        return {}

    total_counts = (
        total_onsets.groupby(["track_id", "global_frame"], sort=True)
        .size()
        .rename("n_interactions_total")
        .reset_index()
    )
    if role_onsets.empty:
        counts = total_counts.copy()
        counts["n_interactions_as_antenna"] = 0
        counts["n_interactions_as_body"] = 0
    else:
        role_counts = (
            role_onsets.groupby(["track_id", "global_frame", "interaction_role"], sort=True)
            .size()
            .unstack("interaction_role", fill_value=0)
            .reset_index()
            .rename(columns={"antenna": "n_interactions_as_antenna", "body": "n_interactions_as_body"})
        )
        for col in ["n_interactions_as_antenna", "n_interactions_as_body"]:
            if col not in role_counts:
                role_counts[col] = 0
        counts = total_counts.merge(
            role_counts[["track_id", "global_frame", "n_interactions_as_antenna", "n_interactions_as_body"]],
            on=["track_id", "global_frame"],
            how="left",
        )

    for col in ["n_interactions_total", "n_interactions_as_antenna", "n_interactions_as_body"]:
        counts[col] = pd.to_numeric(counts[col], errors="coerce").fillna(0).astype(np.int64)
    return {
        int(track_id): group.sort_values("global_frame", kind="mergesort").reset_index(drop=True)
        for track_id, group in counts.groupby("track_id", sort=False)
    }


def build_intra_cluster_interaction_time_table(
    interactions: pd.DataFrame,
    activity_cluster_table: pd.DataFrame,
    *,
    recording_start_clock_seconds: float,
    fps: float = FPS,
    event_gap_seconds: float = INTERACTION_EVENT_GAP_SECONDS,
) -> pd.DataFrame:
    onsets, _role_onsets = interaction_onset_events(
        interactions,
        fps=fps,
        event_gap_seconds=event_gap_seconds,
    )
    if onsets.empty:
        return pd.DataFrame()
    lookup = activity_cluster_table.dropna(subset=["track_id"]).copy()
    lookup["track_id"] = lookup["track_id"].astype(int)
    cluster_by_track = lookup.set_index("track_id")[ACTIVITY_CLUSTER_COL].to_dict()
    work = onsets.copy()
    work["pair_track_id_a"] = np.minimum(work["track_id"], work["partner_track_id"]).astype(int)
    work["pair_track_id_b"] = np.maximum(work["track_id"], work["partner_track_id"]).astype(int)
    work = work.drop_duplicates(["pair_track_id_a", "pair_track_id_b", "global_frame"])
    work["track_activity_cluster"] = work["pair_track_id_a"].map(cluster_by_track)
    work["partner_activity_cluster"] = work["pair_track_id_b"].map(cluster_by_track)
    same = work[
        work["track_activity_cluster"].notna()
        & work["partner_activity_cluster"].notna()
        & (work["track_activity_cluster"] == work["partner_activity_cluster"])
        & (work["pair_track_id_a"] != work["pair_track_id_b"])
    ].copy()
    if same.empty:
        return pd.DataFrame()
    same[ACTIVITY_CLUSTER_COL] = same["track_activity_cluster"].astype(int)
    same["time_bin"] = frame_time_bin(
        same["global_frame"].to_numpy(np.float64),
        fps=fps,
        recording_start_clock_seconds=recording_start_clock_seconds,
        light_on_hour=LIGHT_ON_HOUR,
        bin_minutes=TIME_BIN_MINUTES,
    )
    centers_h = time_bin_centers_hours(TIME_BIN_MINUTES)
    same["hours_since_light_on"] = centers_h[same["time_bin"].to_numpy(int)]

    cluster_sizes = lookup.groupby(ACTIVITY_CLUSTER_COL)["track_id"].nunique().rename("n_ants").reset_index()
    out = (
        same.groupby([ACTIVITY_CLUSTER_COL, "time_bin"], as_index=False)
        .agg(
            hours_since_light_on=("hours_since_light_on", "first"),
            n_intra_cluster_interaction_onsets=("global_frame", "size"),
        )
        .merge(cluster_sizes, on=ACTIVITY_CLUSTER_COL, how="left")
    )
    out["n_possible_pairs"] = out["n_ants"] * np.maximum(out["n_ants"] - 1, 1) / 2.0
    out["interaction_onsets_per_pair"] = out["n_intra_cluster_interaction_onsets"] / out["n_possible_pairs"].replace(0, np.nan)
    out["interaction_event_gap_seconds"] = float(event_gap_seconds)
    return out.sort_values([ACTIVITY_CLUSTER_COL, "time_bin"], kind="mergesort").reset_index(drop=True)


def build_activity_cluster_interaction_time_table(
    interactions: pd.DataFrame,
    activity_cluster_table: pd.DataFrame,
    *,
    recording_start_clock_seconds: float,
    fps: float = FPS,
    event_gap_seconds: float = INTERACTION_EVENT_GAP_SECONDS,
) -> pd.DataFrame:
    """Count directed interaction-bout onsets for focal ants in each activity cluster.

    Unlike the within-cluster table, the partner ant can be in any activity
    cluster. A body/receiver count means another ant's antenna contacted the
    focal ant.
    """
    _total_onsets, role_onsets = interaction_onset_events(
        interactions,
        fps=fps,
        event_gap_seconds=event_gap_seconds,
    )
    if role_onsets.empty:
        return pd.DataFrame()
    lookup = activity_cluster_table.dropna(subset=["track_id"]).copy()
    lookup["track_id"] = lookup["track_id"].astype(int)
    cluster_by_track = lookup.drop_duplicates("track_id").set_index("track_id")[ACTIVITY_CLUSTER_COL].to_dict()
    work = role_onsets.copy()
    work[ACTIVITY_CLUSTER_COL] = work["track_id"].map(cluster_by_track)
    work = work[work[ACTIVITY_CLUSTER_COL].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work[ACTIVITY_CLUSTER_COL] = work[ACTIVITY_CLUSTER_COL].astype(int)
    work["time_bin"] = frame_time_bin(
        work["global_frame"].to_numpy(np.float64),
        fps=fps,
        recording_start_clock_seconds=recording_start_clock_seconds,
        light_on_hour=LIGHT_ON_HOUR,
        bin_minutes=TIME_BIN_MINUTES,
    )
    centers_h = time_bin_centers_hours(TIME_BIN_MINUTES)
    work["hours_since_light_on"] = centers_h[work["time_bin"].to_numpy(int)]

    role_counts = (
        work.groupby([ACTIVITY_CLUSTER_COL, "time_bin", "interaction_role"], sort=True)
        .size()
        .unstack("interaction_role", fill_value=0)
        .reset_index()
        .rename(columns={"antenna": "n_cluster_interaction_onsets_as_antenna", "body": "n_cluster_interaction_onsets_as_body"})
    )
    for col in ["n_cluster_interaction_onsets_as_antenna", "n_cluster_interaction_onsets_as_body"]:
        if col not in role_counts:
            role_counts[col] = 0
        role_counts[col] = pd.to_numeric(role_counts[col], errors="coerce").fillna(0).astype(np.int64)
    role_counts["n_cluster_interaction_onsets"] = (
        role_counts["n_cluster_interaction_onsets_as_antenna"]
        + role_counts["n_cluster_interaction_onsets_as_body"]
    )
    role_counts["hours_since_light_on"] = centers_h[role_counts["time_bin"].to_numpy(int)]
    cluster_sizes = lookup.groupby(ACTIVITY_CLUSTER_COL)["track_id"].nunique().rename("n_ants").reset_index()
    out = role_counts.merge(cluster_sizes, on=ACTIVITY_CLUSTER_COL, how="left")
    bin_hours = float(TIME_BIN_MINUTES) / 60.0
    denom = pd.to_numeric(out["n_ants"], errors="coerce").replace(0, np.nan) * bin_hours
    out["interaction_onset_rate_per_ant_per_h"] = out["n_cluster_interaction_onsets"] / denom
    out["interaction_onset_rate_as_antenna_per_ant_per_h"] = (
        out["n_cluster_interaction_onsets_as_antenna"] / denom
    )
    out["interaction_onset_rate_as_body_per_ant_per_h"] = (
        out["n_cluster_interaction_onsets_as_body"] / denom
    )
    out["interaction_event_gap_seconds"] = float(event_gap_seconds)
    return out.sort_values([ACTIVITY_CLUSTER_COL, "time_bin"], kind="mergesort").reset_index(drop=True)


def build_sleep_interval_index(
    sleep_bouts: pd.DataFrame,
    *,
    analysis_frame_start: int,
    analysis_frame_stop: int,
) -> dict[int, np.ndarray]:
    if sleep_bouts.empty:
        return {}
    required = {"track_id", "bout_start_frame", "bout_end_frame"}
    missing = required.difference(sleep_bouts.columns)
    if missing:
        raise KeyError(f"Sleep bouts are missing columns: {sorted(missing)}")
    work = sleep_bouts[list(required)].copy()
    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["track_id", "bout_start_frame", "bout_end_frame"]).copy()
    if work.empty:
        return {}
    work["track_id"] = work["track_id"].astype(int)
    work["start_frame"] = np.maximum(
        np.rint(work["bout_start_frame"].to_numpy(np.float64)).astype(np.int64),
        int(analysis_frame_start),
    )
    work["stop_frame"] = np.minimum(
        np.rint(work["bout_end_frame"].to_numpy(np.float64)).astype(np.int64) + 1,
        int(analysis_frame_stop),
    )
    work = work[work["stop_frame"] > work["start_frame"]].copy()
    if work.empty:
        return {}
    out: dict[int, np.ndarray] = {}
    for track_id, group in work.groupby("track_id", sort=False):
        intervals = group[["start_frame", "stop_frame"]].sort_values("start_frame").to_numpy(np.int64)
        if len(intervals) <= 1:
            out[int(track_id)] = intervals
            continue
        merged = [intervals[0].copy()]
        for start, stop in intervals[1:]:
            if int(start) <= int(merged[-1][1]):
                merged[-1][1] = max(int(merged[-1][1]), int(stop))
            else:
                merged.append(np.asarray([start, stop], dtype=np.int64))
        out[int(track_id)] = np.vstack(merged)
    return out


def sleep_overlap_frames(intervals: np.ndarray | None, start_frame: int, stop_frame: int) -> int:
    if intervals is None or len(intervals) == 0 or int(stop_frame) <= int(start_frame):
        return 0
    intervals = np.asarray(intervals, dtype=np.int64)
    keep = (intervals[:, 0] < int(stop_frame)) & (intervals[:, 1] > int(start_frame))
    if not keep.any():
        return 0
    selected = intervals[keep]
    starts = np.maximum(selected[:, 0], int(start_frame))
    stops = np.minimum(selected[:, 1], int(stop_frame))
    return int(np.maximum(0, stops - starts).sum())


def sleep_fraction_in_frame_window(
    intervals: np.ndarray | None,
    start_frame: int,
    stop_frame: int,
) -> float:
    duration = int(stop_frame) - int(start_frame)
    if duration <= 0:
        return np.nan
    return float(sleep_overlap_frames(intervals, int(start_frame), int(stop_frame)) / duration)


def interaction_event_table_from_onset_counts(
    interaction_onset_counts_by_track: dict[int, pd.DataFrame],
    activity_cluster_table: pd.DataFrame,
    *,
    analysis_frame_start: int,
    analysis_frame_stop: int,
    recording_start_clock_seconds: float,
    max_events_per_track: int | None,
    random_state: int,
    event_role: str = INTERACTION_WAKE_EVENT_ROLE,
) -> pd.DataFrame:
    event_role = validate_interaction_role(event_role)
    lookup = activity_cluster_table.dropna(subset=["track_id"]).copy()
    lookup["track_id"] = lookup["track_id"].astype(int)
    lookup = lookup.drop_duplicates("track_id").set_index("track_id")
    rng = np.random.default_rng(int(random_state))
    rows = []
    for track_id, counts in interaction_onset_counts_by_track.items():
        track_id = int(track_id)
        if track_id not in lookup.index or counts.empty:
            continue
        work = counts.copy()
        work["global_frame"] = pd.to_numeric(work["global_frame"], errors="coerce")
        work = work.dropna(subset=["global_frame"]).copy()
        work["global_frame"] = work["global_frame"].astype(np.int64)
        def count_series(col: str) -> pd.Series:
            if col in work.columns:
                return pd.to_numeric(work[col], errors="coerce").fillna(0)
            return pd.Series(0, index=work.index, dtype=np.int64)

        if event_role == "body":
            event_count = count_series("n_interactions_as_body")
        elif event_role == "antenna":
            event_count = count_series("n_interactions_as_antenna")
        else:
            event_count = count_series("n_interactions_total")
        work["n_interaction_role_onsets"] = event_count.astype(np.int64)
        work = work[event_count > 0].copy()
        work = work[
            (work["global_frame"] >= int(analysis_frame_start))
            & (work["global_frame"] < int(analysis_frame_stop))
        ].sort_values("global_frame", kind="mergesort")
        if work.empty:
            continue
        if max_events_per_track is not None and len(work) > int(max_events_per_track):
            chosen_idx = rng.choice(work.index.to_numpy(), size=int(max_events_per_track), replace=False)
            work = work.loc[np.sort(chosen_idx)].sort_values("global_frame", kind="mergesort")
        track_meta = lookup.loc[track_id]
        frames = work["global_frame"].to_numpy(np.int64)
        time_bins = frame_time_bin(
            frames,
            fps=FPS,
            recording_start_clock_seconds=recording_start_clock_seconds,
            light_on_hour=LIGHT_ON_HOUR,
            bin_minutes=TIME_BIN_MINUTES,
        )
        centers_h = time_bin_centers_hours(TIME_BIN_MINUTES)
        for row, time_bin in zip(work.itertuples(index=False), time_bins):
            rows.append(
                {
                    "track_id": track_id,
                    "track_name": track_meta["track_name"],
                    "side": track_meta["side"],
                    ACTIVITY_CLUSTER_COL: track_meta[ACTIVITY_CLUSTER_COL],
                    "event_frame": int(getattr(row, "global_frame")),
                    "time_bin": int(time_bin),
                    "hours_since_light_on": float(centers_h[int(time_bin)]),
                    "interaction_event_role": event_role,
                    "n_interaction_role_onsets": int(getattr(row, "n_interaction_role_onsets", 1)),
                    "n_interaction_onsets": int(getattr(row, "n_interactions_total", 1)),
                    "n_interaction_onsets_as_antenna": int(getattr(row, "n_interactions_as_antenna", 0)),
                    "n_interaction_onsets_as_body": int(getattr(row, "n_interactions_as_body", 0)),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["track_id", "event_frame"], kind="mergesort").reset_index(drop=True)


def time_bin_frame_ranges(
    time_bin: int,
    *,
    analysis_frame_start: int,
    analysis_frame_stop: int,
    recording_start_clock_seconds: float,
) -> list[tuple[int, int]]:
    bin_width_s = float(TIME_BIN_MINUTES) * 60.0
    day_s = 24.0 * 3600.0
    bin_start_s = int(time_bin) * bin_width_s
    bin_stop_s = (int(time_bin) + 1) * bin_width_s
    base_s = float(recording_start_clock_seconds) - float(LIGHT_ON_HOUR) * 3600.0
    rel_start_s = base_s + int(analysis_frame_start) / float(FPS)
    rel_stop_s = base_s + int(analysis_frame_stop) / float(FPS)
    first_day = int(np.floor((rel_start_s - bin_stop_s) / day_s)) - 1
    last_day = int(np.ceil((rel_stop_s - bin_start_s) / day_s)) + 1
    ranges: list[tuple[int, int]] = []
    for day in range(first_day, last_day + 1):
        start = int(np.ceil((bin_start_s + day * day_s - base_s) * float(FPS)))
        stop = int(np.ceil((bin_stop_s + day * day_s - base_s) * float(FPS)))
        start = max(start, int(analysis_frame_start))
        stop = min(stop, int(analysis_frame_stop))
        if stop > start:
            ranges.append((start, stop))
    return ranges


def sample_frame_from_ranges(
    ranges: list[tuple[int, int]],
    rng: np.random.Generator,
) -> int | None:
    if not ranges:
        return None
    lengths = np.asarray([stop - start for start, stop in ranges], dtype=np.int64)
    keep = lengths > 0
    if not keep.any():
        return None
    kept_ranges = [ranges[i] for i in np.flatnonzero(keep)]
    lengths = lengths[keep]
    idx = int(rng.choice(len(kept_ranges), p=lengths / lengths.sum()))
    start, stop = kept_ranges[idx]
    return int(rng.integers(int(start), int(stop), endpoint=False))


def has_interaction_in_frame_window(frames: np.ndarray, start_frame: int, stop_frame: int) -> bool:
    if len(frames) == 0 or int(stop_frame) <= int(start_frame):
        return False
    left = int(np.searchsorted(frames, int(start_frame), side="left"))
    right = int(np.searchsorted(frames, int(stop_frame), side="left"))
    return right > left


def trigger_sleep_curve_rows(
    centers: pd.DataFrame,
    sleep_intervals_by_track: dict[int, np.ndarray],
    *,
    condition: str,
    center_frame_col: str,
    radius_seconds: float,
    bin_seconds: float,
) -> pd.DataFrame:
    if centers.empty:
        return pd.DataFrame()
    half_bin_frames = max(1, int(round(float(bin_seconds) * float(FPS) / 2.0)))
    offsets_s = np.arange(
        -float(radius_seconds),
        float(radius_seconds) + 0.5 * float(bin_seconds),
        float(bin_seconds),
        dtype=np.float64,
    )
    rows = []
    for center in centers.itertuples(index=False):
        track_id = int(getattr(center, "track_id"))
        intervals = sleep_intervals_by_track.get(track_id)
        center_frame = int(getattr(center, center_frame_col))
        cluster = getattr(center, ACTIVITY_CLUSTER_COL)
        event_id = getattr(center, "event_id", None)
        for offset_s in offsets_s:
            offset_frames = int(round(float(offset_s) * float(FPS)))
            sample_center = center_frame + offset_frames
            start = sample_center - half_bin_frames
            stop = sample_center + half_bin_frames
            rows.append(
                {
                    "condition": condition,
                    ACTIVITY_CLUSTER_COL: cluster,
                    "event_id": event_id,
                    "track_id": track_id,
                    "relative_seconds": float(offset_s),
                    "relative_minutes": float(offset_s / 60.0),
                    "sleep_fraction": sleep_fraction_in_frame_window(intervals, start, stop),
                }
            )
    return pd.DataFrame(rows)


def summarize_trigger_sleep_curve(curve_rows: pd.DataFrame) -> pd.DataFrame:
    if curve_rows.empty:
        return pd.DataFrame()
    valid = curve_rows.dropna(subset=["sleep_fraction"]).copy()
    if valid.empty:
        return pd.DataFrame()
    return (
        valid.groupby(["condition", ACTIVITY_CLUSTER_COL, "relative_seconds", "relative_minutes"], as_index=False)
        .agg(
            mean_sleep_fraction=("sleep_fraction", "mean"),
            median_sleep_fraction=("sleep_fraction", "median"),
            q25_sleep_fraction=("sleep_fraction", lambda x: float(np.nanpercentile(x, 25))),
            q75_sleep_fraction=("sleep_fraction", lambda x: float(np.nanpercentile(x, 75))),
            n_samples=("sleep_fraction", "size"),
        )
        .sort_values(["condition", ACTIVITY_CLUSTER_COL, "relative_seconds"], kind="mergesort")
    )


def matched_interaction_wake_summary(matched_effects: pd.DataFrame) -> pd.DataFrame:
    if matched_effects.empty:
        return pd.DataFrame()

    def summarize_group(group: pd.DataFrame, cluster_label: object) -> dict[str, object]:
        diff = pd.to_numeric(group["paired_sleep_change_vs_control"], errors="coerce").dropna()
        more_wake = diff < 0
        row: dict[str, object] = {
            ACTIVITY_CLUSTER_COL: cluster_label,
            "n_matched_events": int(len(group)),
            "median_event_pre_sleep": float(group["event_pre_sleep_fraction"].median()),
            "median_event_post_sleep": float(group["event_post_sleep_fraction"].median()),
            "median_event_sleep_change": float(group["event_sleep_change"].median()),
            "median_control_pre_sleep": float(group["control_pre_sleep_fraction"].median()),
            "median_control_post_sleep": float(group["control_post_sleep_fraction"].median()),
            "median_control_sleep_change": float(group["control_sleep_change"].median()),
            "median_paired_sleep_change_vs_control": float(diff.median()) if len(diff) else np.nan,
            "fraction_interaction_more_wake_than_control": float(more_wake.mean()) if len(diff) else np.nan,
            "event_wake_fraction": float(group["event_wake_transition"].mean()),
            "control_wake_fraction": float(group["control_wake_transition"].mean()),
            "wake_fraction_difference": float(
                group["event_wake_transition"].mean() - group["control_wake_transition"].mean()
            ),
            "median_n_controls_per_event": float(group["n_controls"].median()),
        }
        try:
            from scipy.stats import binomtest

            row["one_sided_sign_p_interaction_more_wake"] = float(
                binomtest(int(more_wake.sum()), int(len(diff)), p=0.5, alternative="greater").pvalue
            ) if len(diff) else np.nan
        except Exception:
            row["one_sided_sign_p_interaction_more_wake"] = np.nan
        return row

    rows = [summarize_group(group, cluster) for cluster, group in matched_effects.groupby(ACTIVITY_CLUSTER_COL, sort=True)]
    rows.append(summarize_group(matched_effects, "all"))
    return pd.DataFrame(rows)


def interaction_triggered_wake_analysis(
    sleep_bouts: pd.DataFrame,
    interaction_onset_counts_by_track: dict[int, pd.DataFrame],
    activity_cluster_table: pd.DataFrame,
    *,
    analysis_frame_start: int,
    analysis_frame_stop: int,
    recording_start_clock_seconds: float,
    pre_seconds: float = INTERACTION_WAKE_PRE_SECONDS,
    post_seconds: float = INTERACTION_WAKE_POST_SECONDS,
    pre_sleep_min_fraction: float = INTERACTION_WAKE_PRE_SLEEP_MIN_FRACTION,
    post_wake_max_sleep_fraction: float = INTERACTION_WAKE_POST_WAKE_MAX_SLEEP_FRACTION,
    control_replicates: int = INTERACTION_WAKE_CONTROL_REPLICATES,
    control_search_tries: int = INTERACTION_WAKE_CONTROL_SEARCH_TRIES,
    control_exclude_seconds: float = INTERACTION_WAKE_CONTROL_EXCLUDE_SECONDS,
    max_events_per_track: int | None = INTERACTION_WAKE_MAX_EVENTS_PER_TRACK,
    curve_radius_seconds: float = INTERACTION_WAKE_CURVE_RADIUS_SECONDS,
    curve_bin_seconds: float = INTERACTION_WAKE_CURVE_BIN_SECONDS,
    random_state: int = INTERACTION_WAKE_RANDOM_STATE,
    event_role: str = INTERACTION_WAKE_EVENT_ROLE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_role = validate_interaction_role(event_role)
    sleep_intervals = build_sleep_interval_index(
        sleep_bouts,
        analysis_frame_start=analysis_frame_start,
        analysis_frame_stop=analysis_frame_stop,
    )
    event_candidates = interaction_event_table_from_onset_counts(
        interaction_onset_counts_by_track,
        activity_cluster_table,
        analysis_frame_start=analysis_frame_start,
        analysis_frame_stop=analysis_frame_stop,
        recording_start_clock_seconds=recording_start_clock_seconds,
        max_events_per_track=max_events_per_track,
        random_state=random_state,
        event_role=event_role,
    )
    if event_candidates.empty or not sleep_intervals:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    pre_frames = max(1, int(round(float(pre_seconds) * float(FPS))))
    post_frames = max(1, int(round(float(post_seconds) * float(FPS))))
    exclude_frames = max(1, int(round(float(control_exclude_seconds) * float(FPS))))
    rng = np.random.default_rng(int(random_state))
    interaction_frames_by_track = {}
    for track_id, counts in interaction_onset_counts_by_track.items():
        if counts.empty or "global_frame" not in counts.columns:
            continue
        work = counts.copy()
        if event_role == "body" and "n_interactions_as_body" in work.columns:
            work = work[pd.to_numeric(work["n_interactions_as_body"], errors="coerce").fillna(0) > 0]
        elif event_role == "antenna" and "n_interactions_as_antenna" in work.columns:
            work = work[pd.to_numeric(work["n_interactions_as_antenna"], errors="coerce").fillna(0) > 0]
        elif event_role == "both" and "n_interactions_total" in work.columns:
            work = work[pd.to_numeric(work["n_interactions_total"], errors="coerce").fillna(0) > 0]
        interaction_frames_by_track[int(track_id)] = work["global_frame"].dropna().astype(np.int64).sort_values().unique()

    event_rows = []
    for event in event_candidates.itertuples(index=False):
        track_id = int(getattr(event, "track_id"))
        event_frame = int(getattr(event, "event_frame"))
        if event_frame - pre_frames < int(analysis_frame_start) or event_frame + post_frames >= int(analysis_frame_stop):
            continue
        intervals = sleep_intervals.get(track_id)
        pre_sleep = sleep_fraction_in_frame_window(intervals, event_frame - pre_frames, event_frame)
        if not np.isfinite(pre_sleep) or pre_sleep < float(pre_sleep_min_fraction):
            continue
        post_sleep = sleep_fraction_in_frame_window(intervals, event_frame, event_frame + post_frames)
        event_rows.append(
            {
                "event_id": len(event_rows),
                "track_id": track_id,
                "track_name": getattr(event, "track_name"),
                "side": getattr(event, "side"),
                ACTIVITY_CLUSTER_COL: getattr(event, ACTIVITY_CLUSTER_COL),
                "event_frame": event_frame,
                "time_bin": int(getattr(event, "time_bin")),
                "hours_since_light_on": float(getattr(event, "hours_since_light_on")),
                "n_interaction_onsets": int(getattr(event, "n_interaction_onsets")),
                "n_interaction_onsets_as_antenna": int(getattr(event, "n_interaction_onsets_as_antenna")),
                "n_interaction_onsets_as_body": int(getattr(event, "n_interaction_onsets_as_body")),
                "interaction_event_role": event_role,
                "event_pre_sleep_fraction": pre_sleep,
                "event_post_sleep_fraction": post_sleep,
                "event_sleep_change": post_sleep - pre_sleep,
                "event_wake_transition": bool(
                    pre_sleep >= float(pre_sleep_min_fraction)
                    and post_sleep <= float(post_wake_max_sleep_fraction)
                ),
            }
        )
    wake_event_table = pd.DataFrame(event_rows)
    if wake_event_table.empty:
        empty = pd.DataFrame()
        return wake_event_table, empty, empty, empty, empty

    control_rows = []
    ranges_cache: dict[int, list[tuple[int, int]]] = {}
    for event in wake_event_table.itertuples(index=False):
        track_id = int(getattr(event, "track_id"))
        time_bin = int(getattr(event, "time_bin"))
        intervals = sleep_intervals.get(track_id)
        frames = interaction_frames_by_track.get(track_id, np.asarray([], dtype=np.int64))
        if time_bin not in ranges_cache:
            ranges_cache[time_bin] = time_bin_frame_ranges(
                time_bin,
                analysis_frame_start=analysis_frame_start,
                analysis_frame_stop=analysis_frame_stop,
                recording_start_clock_seconds=recording_start_clock_seconds,
            )
        n_found = 0
        n_tries = 0
        while n_found < int(control_replicates) and n_tries < int(control_search_tries):
            n_tries += 1
            control_frame = sample_frame_from_ranges(ranges_cache[time_bin], rng)
            if control_frame is None:
                break
            if control_frame - pre_frames < int(analysis_frame_start) or control_frame + post_frames >= int(analysis_frame_stop):
                continue
            if abs(int(control_frame) - int(getattr(event, "event_frame"))) <= exclude_frames:
                continue
            if has_interaction_in_frame_window(frames, control_frame - exclude_frames, control_frame + exclude_frames + 1):
                continue
            pre_sleep = sleep_fraction_in_frame_window(intervals, control_frame - pre_frames, control_frame)
            if not np.isfinite(pre_sleep) or pre_sleep < float(pre_sleep_min_fraction):
                continue
            post_sleep = sleep_fraction_in_frame_window(intervals, control_frame, control_frame + post_frames)
            control_rows.append(
                {
                    "event_id": int(getattr(event, "event_id")),
                    "control_id": n_found,
                    "track_id": track_id,
                    "track_name": getattr(event, "track_name"),
                    "side": getattr(event, "side"),
                    ACTIVITY_CLUSTER_COL: getattr(event, ACTIVITY_CLUSTER_COL),
                    "control_frame": int(control_frame),
                    "time_bin": time_bin,
                    "hours_since_light_on": float(getattr(event, "hours_since_light_on")),
                    "control_pre_sleep_fraction": pre_sleep,
                    "control_post_sleep_fraction": post_sleep,
                    "control_sleep_change": post_sleep - pre_sleep,
                    "control_wake_transition": bool(
                        pre_sleep >= float(pre_sleep_min_fraction)
                        and post_sleep <= float(post_wake_max_sleep_fraction)
                    ),
                    "control_search_tries": n_tries,
                }
            )
            n_found += 1
    wake_control_table = pd.DataFrame(control_rows)
    if wake_control_table.empty:
        empty = pd.DataFrame()
        return wake_event_table, wake_control_table, empty, empty, empty

    control_by_event = (
        wake_control_table.groupby("event_id", as_index=False)
        .agg(
            control_pre_sleep_fraction=("control_pre_sleep_fraction", "mean"),
            control_post_sleep_fraction=("control_post_sleep_fraction", "mean"),
            control_sleep_change=("control_sleep_change", "mean"),
            control_wake_transition=("control_wake_transition", "mean"),
            n_controls=("control_id", "size"),
        )
    )
    matched_effects = wake_event_table.merge(control_by_event, on="event_id", how="inner", validate="one_to_one")
    matched_effects["paired_sleep_change_vs_control"] = (
        matched_effects["event_sleep_change"] - matched_effects["control_sleep_change"]
    )
    matched_effects["paired_post_sleep_vs_control"] = (
        matched_effects["event_post_sleep_fraction"] - matched_effects["control_post_sleep_fraction"]
    )
    wake_effect_summary = matched_interaction_wake_summary(matched_effects)

    event_curve = trigger_sleep_curve_rows(
        matched_effects,
        sleep_intervals,
        condition="interaction",
        center_frame_col="event_frame",
        radius_seconds=curve_radius_seconds,
        bin_seconds=curve_bin_seconds,
    )
    matched_controls_for_curve = wake_control_table[wake_control_table["event_id"].isin(matched_effects["event_id"])]
    control_curve = trigger_sleep_curve_rows(
        matched_controls_for_curve,
        sleep_intervals,
        condition="time_matched_control",
        center_frame_col="control_frame",
        radius_seconds=curve_radius_seconds,
        bin_seconds=curve_bin_seconds,
    )
    wake_trigger_curve = summarize_trigger_sleep_curve(pd.concat([event_curve, control_curve], ignore_index=True))
    return wake_event_table, wake_control_table, matched_effects, wake_effect_summary, wake_trigger_curve


def plot_interaction_triggered_wake_curve(wake_trigger_curve: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if wake_trigger_curve.empty:
        print("No interaction-triggered wake curve rows to plot")
        return
    clusters = sorted(wake_trigger_curve[ACTIVITY_CLUSTER_COL].dropna().unique())
    fig, axes = plt.subplots(len(clusters), 1, figsize=(11, max(3, 2.5 * len(clusters))), sharex=True, squeeze=False)
    colors = {"interaction": "tab:red", "time_matched_control": "tab:blue"}
    labels = {"interaction": "interaction", "time_matched_control": "time-of-day matched control"}
    for ax, cluster in zip(axes.ravel(), clusters):
        cluster_data = wake_trigger_curve[wake_trigger_curve[ACTIVITY_CLUSTER_COL] == cluster]
        for condition, group in cluster_data.groupby("condition", sort=False):
            group = group.sort_values("relative_minutes")
            color = colors.get(str(condition), "0.25")
            ax.fill_between(
                group["relative_minutes"],
                group["q25_sleep_fraction"],
                group["q75_sleep_fraction"],
                color=color,
                alpha=0.16,
            )
            ax.plot(
                group["relative_minutes"],
                group["mean_sleep_fraction"],
                color=color,
                lw=1.8,
                label=labels.get(str(condition), str(condition)),
            )
        ax.axvline(0, color="0.35", lw=0.9, ls="--")
        ax.set_title(f"activity cluster {cluster}: sleep around interaction-trigger frames")
        ax.set_ylabel("sleep fraction")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("minutes relative to interaction/control frame")
    fig.tight_layout()
    plt.show()


def interaction_count_in_window(
    interaction_frames: np.ndarray,
    interaction_counts: np.ndarray,
    start_frame: int,
    stop_frame: int,
) -> int:
    if len(interaction_frames) == 0 or int(stop_frame) <= int(start_frame):
        return 0
    left = int(np.searchsorted(interaction_frames, int(start_frame), side="left"))
    right = int(np.searchsorted(interaction_frames, int(stop_frame), side="left"))
    if right <= left:
        return 0
    return int(np.asarray(interaction_counts[left:right], dtype=np.int64).sum())


def interaction_onset_events_for_track(
    interactions: pd.DataFrame,
    *,
    track_id: int,
    fps: float,
    event_gap_seconds: float,
    event_role: str = "both",
) -> pd.DataFrame:
    event_role = validate_interaction_role(event_role)
    onsets, role_onsets = interaction_onset_events(
        interactions,
        fps=fps,
        event_gap_seconds=event_gap_seconds,
        track_ids=[int(track_id)],
    )
    if event_role == "both":
        return onsets.sort_values("global_frame", kind="mergesort").reset_index(drop=True)
    selected = role_onsets[role_onsets["interaction_role"].astype(str) == event_role].copy()
    return selected.sort_values("global_frame", kind="mergesort").reset_index(drop=True)


def relative_time_edges_from_centers(centers: np.ndarray, step: float) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) == 0:
        return np.asarray([], dtype=np.float64)
    half_step = float(step) / 2.0
    return np.r_[centers[0] - half_step, centers + half_step]


def choose_sleep_end_trigger_track_id(
    sleep_bouts: pd.DataFrame,
    *,
    track_id: int | None = SLEEP_END_TRIGGER_TRACK_ID,
    track_name: str | None = SLEEP_END_TRIGGER_TRACK_NAME,
) -> int:
    if track_id is not None:
        return int(track_id)
    if track_name is not None:
        name = str(track_name)
        match = sleep_bouts[sleep_bouts["track_name"].astype(str) == name]
        if not match.empty:
            return int(match["track_id"].iloc[0])
        raise ValueError(f"No ant matched SLEEP_END_TRIGGER_TRACK_NAME={name!r}")

    work = sleep_bouts.copy()
    work["track_id"] = pd.to_numeric(work["track_id"], errors="coerce")
    work["bout_duration_seconds"] = pd.to_numeric(work["bout_duration_seconds"], errors="coerce")
    work = work.dropna(subset=["track_id"]).copy()
    if work.empty:
        raise ValueError("No predicted sleep bouts are available")
    summary = (
        work.groupby("track_id", as_index=False)
        .agg(
            n_sleep_bouts=("bout_start_frame", "size"),
            total_sleep_seconds=("bout_duration_seconds", "sum"),
        )
        .sort_values(["n_sleep_bouts", "total_sleep_seconds", "track_id"], ascending=[False, False, True])
    )
    return int(summary["track_id"].iloc[0])


def sleep_end_trigger_bouts_for_track(
    sleep_bouts: pd.DataFrame,
    *,
    track_id: int,
    analysis_frame_start: int,
    analysis_frame_stop: int,
    fps: float,
    max_bouts: int | None,
    sort_by: str,
) -> pd.DataFrame:
    bouts = sleep_bouts[
        pd.to_numeric(sleep_bouts["track_id"], errors="coerce") == int(track_id)
    ].copy()
    if bouts.empty:
        raise ValueError(f"No predicted sleep bouts found for track_id={track_id}")
    for col in ["bout_start_frame", "bout_end_frame", "bout_duration_seconds"]:
        bouts[col] = pd.to_numeric(bouts[col], errors="coerce")
    bouts = bouts.dropna(subset=["bout_start_frame", "bout_end_frame"]).copy()
    bouts["bout_start_frame"] = np.rint(bouts["bout_start_frame"]).astype(np.int64)
    bouts["bout_end_frame"] = np.rint(bouts["bout_end_frame"]).astype(np.int64)
    bouts["sleep_end_frame"] = bouts["bout_end_frame"] + 1
    fallback_duration = (bouts["sleep_end_frame"] - bouts["bout_start_frame"]) / float(fps)
    bouts["bout_duration_seconds"] = bouts["bout_duration_seconds"].fillna(fallback_duration)
    bouts = bouts[
        (bouts["sleep_end_frame"] > bouts["bout_start_frame"])
        & (bouts["sleep_end_frame"] > int(analysis_frame_start))
        & (bouts["sleep_end_frame"] < int(analysis_frame_stop))
    ].copy()
    if bouts.empty:
        raise ValueError(f"No selected sleep ends overlap the analysis window for track_id={track_id}")
    if sort_by == "duration_desc":
        bouts = bouts.sort_values(["bout_duration_seconds", "sleep_end_frame"], ascending=[False, True])
    else:
        bouts = bouts.sort_values("sleep_end_frame", kind="mergesort")
    if max_bouts is not None and len(bouts) > int(max_bouts):
        bouts = bouts.head(int(max_bouts)).copy()
    bouts = bouts.reset_index(drop=True)
    bouts["bout_trigger_row"] = np.arange(len(bouts), dtype=int)
    bouts["bout_duration_minutes"] = bouts["bout_duration_seconds"] / 60.0
    bouts["sleep_end_elapsed_hours"] = bouts["sleep_end_frame"] / float(fps) / 3600.0
    return bouts


def build_sleep_end_triggered_interaction_rate(
    sleep_bouts: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    track_id: int | None = SLEEP_END_TRIGGER_TRACK_ID,
    track_name: str | None = SLEEP_END_TRIGGER_TRACK_NAME,
    analysis_frame_start: int,
    analysis_frame_stop: int,
    fps: float = FPS,
    pre_seconds: float = SLEEP_END_TRIGGER_PRE_SECONDS,
    post_seconds: float = SLEEP_END_TRIGGER_POST_SECONDS,
    rate_window_seconds: float = SLEEP_END_TRIGGER_RATE_WINDOW_SECONDS,
    rate_step_seconds: float = SLEEP_END_TRIGGER_RATE_STEP_SECONDS,
    event_gap_seconds: float = SLEEP_END_TRIGGER_INTERACTION_EVENT_GAP_SECONDS,
    event_role: str = SLEEP_END_TRIGGER_INTERACTION_ROLE,
    max_bouts: int | None = SLEEP_END_TRIGGER_MAX_BOUTS,
    sort_by: str = SLEEP_END_TRIGGER_SORT_BY,
    post_rate_seconds: float = SLEEP_END_TRIGGER_POST_RATE_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray | str | int | float]]:
    event_role = validate_interaction_role(event_role)
    selected_track_id = choose_sleep_end_trigger_track_id(
        sleep_bouts,
        track_id=track_id,
        track_name=track_name,
    )
    bouts = sleep_end_trigger_bouts_for_track(
        sleep_bouts,
        track_id=selected_track_id,
        analysis_frame_start=analysis_frame_start,
        analysis_frame_stop=analysis_frame_stop,
        fps=fps,
        max_bouts=max_bouts,
        sort_by=sort_by,
    )

    onset_events = interaction_onset_events_for_track(
        interactions,
        track_id=selected_track_id,
        fps=fps,
        event_gap_seconds=event_gap_seconds,
        event_role=event_role,
    )
    onset_events = onset_events[
        (onset_events["global_frame"] >= int(analysis_frame_start))
        & (onset_events["global_frame"] < int(analysis_frame_stop))
    ].copy()
    if onset_events.empty:
        interaction_frames = np.asarray([], dtype=np.int64)
        interaction_counts = np.asarray([], dtype=np.int64)
    else:
        onset_counts = (
            onset_events.groupby("global_frame", sort=True)
            .size()
            .rename("interaction_onset_count")
            .reset_index()
        )
        interaction_frames = onset_counts["global_frame"].astype(np.int64).to_numpy()
        interaction_counts = onset_counts["interaction_onset_count"].astype(np.int64).to_numpy()

    sleep_intervals_by_track = build_sleep_interval_index(
        sleep_bouts,
        analysis_frame_start=analysis_frame_start,
        analysis_frame_stop=analysis_frame_stop,
    )
    sleep_intervals = sleep_intervals_by_track.get(selected_track_id)

    step_s = max(1e-6, float(rate_step_seconds))
    window_s = max(1e-6, float(rate_window_seconds))
    rel_centers_s = np.arange(-float(pre_seconds), float(post_seconds) + step_s * 0.5, step_s, dtype=np.float64)
    rel_edges_s = relative_time_edges_from_centers(rel_centers_s, step_s)
    n_bouts = len(bouts)
    n_times = len(rel_centers_s)
    rate = np.full((n_bouts, n_times), np.nan, dtype=np.float32)
    counts_matrix = np.zeros((n_bouts, n_times), dtype=np.float32)
    observed_seconds = np.zeros((n_bouts, n_times), dtype=np.float32)
    sleep_fraction = np.full((n_bouts, n_times), np.nan, dtype=np.float32)

    half_window_frames = int(round(window_s * float(fps) / 2.0))
    summary_rows = []
    rate_rows = []
    for bout in bouts.itertuples(index=False):
        row_idx = int(getattr(bout, "bout_trigger_row"))
        sleep_start = int(getattr(bout, "bout_start_frame"))
        sleep_end = int(getattr(bout, "sleep_end_frame"))
        duration_s = float(getattr(bout, "bout_duration_seconds"))
        for col_idx, rel_s in enumerate(rel_centers_s):
            center_frame = sleep_end + int(round(float(rel_s) * float(fps)))
            start = max(int(analysis_frame_start), center_frame - half_window_frames)
            stop = min(int(analysis_frame_stop), center_frame + half_window_frames)
            duration = float(max(0, stop - start) / float(fps))
            if duration <= 0:
                continue
            onset_count = interaction_count_in_window(interaction_frames, interaction_counts, start, stop)
            cur_rate = float(onset_count / (duration / 60.0))
            cur_sleep = sleep_fraction_in_frame_window(sleep_intervals, start, stop)
            counts_matrix[row_idx, col_idx] = onset_count
            observed_seconds[row_idx, col_idx] = duration
            rate[row_idx, col_idx] = cur_rate
            sleep_fraction[row_idx, col_idx] = cur_sleep
            rate_rows.append(
                {
                    "bout_trigger_row": row_idx,
                    "track_id": selected_track_id,
                    "track_name": getattr(bout, "track_name", None),
                    "sleep_start_frame": sleep_start,
                    "sleep_end_frame": sleep_end,
                    "bout_duration_seconds": duration_s,
                    "bout_duration_minutes": duration_s / 60.0,
                    "relative_time_seconds": float(rel_s),
                    "relative_time_min": float(rel_s / 60.0),
                    "rate_window_seconds": window_s,
                    "interaction_event_gap_seconds": float(event_gap_seconds),
                    "interaction_event_role": event_role,
                    "interaction_onset_count": onset_count,
                    "observed_duration_seconds": duration,
                    "interaction_onset_rate_per_min": cur_rate,
                    "sleep_fraction": cur_sleep,
                }
            )

        pre_mask = (rel_centers_s >= -float(post_rate_seconds)) & (rel_centers_s < 0)
        post_mask = (rel_centers_s >= 0) & (rel_centers_s <= float(post_rate_seconds))
        post_values = rate[row_idx, post_mask]
        pre_values = rate[row_idx, pre_mask]
        summary_rows.append(
            {
                "bout_trigger_row": row_idx,
                "track_id": selected_track_id,
                "track_name": getattr(bout, "track_name", None),
                "sleep_start_frame": sleep_start,
                "sleep_end_frame": sleep_end,
                "sleep_end_elapsed_hours": float(getattr(bout, "sleep_end_elapsed_hours")),
                "bout_duration_seconds": duration_s,
                "bout_duration_minutes": duration_s / 60.0,
                "interaction_event_role": event_role,
                "pre_median_interaction_onset_rate_per_min": float(np.nanmedian(pre_values)) if np.isfinite(pre_values).any() else np.nan,
                "post_median_interaction_onset_rate_per_min": float(np.nanmedian(post_values)) if np.isfinite(post_values).any() else np.nan,
                "post_peak_interaction_onset_rate_per_min": float(np.nanmax(post_values)) if np.isfinite(post_values).any() else np.nan,
                "post_minus_pre_median_onset_rate_per_min": (
                    float(np.nanmedian(post_values) - np.nanmedian(pre_values))
                    if np.isfinite(post_values).any() and np.isfinite(pre_values).any()
                    else np.nan
                ),
            }
        )

    bout_summary = pd.DataFrame(summary_rows)
    if sort_by == "post_rate_desc" and not bout_summary.empty:
        order = bout_summary.sort_values(
            ["post_peak_interaction_onset_rate_per_min", "sleep_end_frame"],
            ascending=[False, True],
        )["bout_trigger_row"].to_numpy(int)
        remap = {old: new for new, old in enumerate(order)}
        bout_summary = bout_summary.set_index("bout_trigger_row").loc[order].reset_index()
        bout_summary["bout_trigger_row"] = np.arange(len(bout_summary), dtype=int)
        rate = rate[order]
        counts_matrix = counts_matrix[order]
        observed_seconds = observed_seconds[order]
        sleep_fraction = sleep_fraction[order]
        rate_table = pd.DataFrame(rate_rows)
        rate_table = rate_table[rate_table["bout_trigger_row"].isin(order)].copy()
        rate_table["bout_trigger_row"] = rate_table["bout_trigger_row"].map(remap).astype(int)
        rate_table = rate_table.sort_values(["bout_trigger_row", "relative_time_seconds"], kind="mergesort")
    else:
        rate_table = pd.DataFrame(rate_rows)

    matrices = {
        "interaction_rate_per_min": rate,
        "interaction_onset_count": counts_matrix,
        "observed_duration_seconds": observed_seconds,
        "sleep_fraction": sleep_fraction,
        "relative_time_centers_seconds": rel_centers_s,
        "relative_time_edges_seconds": rel_edges_s,
        "relative_time_centers_min": rel_centers_s / 60.0,
        "relative_time_edges_min": rel_edges_s / 60.0,
        "rate_window_seconds": window_s,
        "rate_step_seconds": step_s,
        "interaction_measure": "event_onset",
        "interaction_event_gap_seconds": float(event_gap_seconds),
        "interaction_event_role": event_role,
        "n_interaction_onsets": int(len(onset_events)),
        "track_id": selected_track_id,
        "track_name": str(bout_summary["track_name"].dropna().iloc[0]) if bout_summary["track_name"].notna().any() else "",
        "sort_by": str(sort_by),
    }
    return bout_summary, rate_table.reset_index(drop=True), matrices


def plot_sleep_end_triggered_interaction_rate(
    bout_summary: pd.DataFrame,
    matrices: dict[str, np.ndarray | str | int | float],
    *,
    row_order_by: str = SLEEP_END_TRIGGER_PLOT_ORDER_BY,
) -> None:
    import matplotlib.pyplot as plt

    if bout_summary.empty:
        print("No sleep-end triggered bouts to plot")
        return
    rate = np.asarray(matrices["interaction_rate_per_min"], dtype=np.float64)
    sleep_fraction = np.asarray(matrices["sleep_fraction"], dtype=np.float64)
    x_edges = np.asarray(matrices["relative_time_edges_min"], dtype=np.float64)
    x_centers = np.asarray(matrices["relative_time_centers_min"], dtype=np.float64)
    if rate.ndim != 2 or rate.size == 0:
        print("Sleep-end interaction-rate matrix is empty")
        return
    plot_summary = bout_summary.reset_index(drop=True).copy()
    order_by = str(row_order_by or "none")
    if len(plot_summary) == rate.shape[0]:
        if order_by == "duration_desc":
            row_order = (
                plot_summary.sort_values(
                    ["bout_duration_seconds", "sleep_end_frame"],
                    ascending=[False, True],
                    na_position="last",
                )
                .index.to_numpy(int)
            )
        elif order_by == "duration_asc":
            row_order = (
                plot_summary.sort_values(
                    ["bout_duration_seconds", "sleep_end_frame"],
                    ascending=[True, True],
                    na_position="last",
                )
                .index.to_numpy(int)
            )
        elif order_by == "post_rate_desc" and "post_peak_interaction_onset_rate_per_min" in plot_summary:
            row_order = (
                plot_summary.sort_values(
                    ["post_peak_interaction_onset_rate_per_min", "sleep_end_frame"],
                    ascending=[False, True],
                    na_position="last",
                )
                .index.to_numpy(int)
            )
        elif order_by == "end_frame":
            row_order = plot_summary.sort_values("sleep_end_frame", na_position="last").index.to_numpy(int)
        else:
            row_order = np.arange(rate.shape[0], dtype=int)
        plot_summary = plot_summary.iloc[row_order].reset_index(drop=True)
        rate = rate[row_order]
        sleep_fraction = sleep_fraction[row_order]
    n_rows = rate.shape[0]
    with np.errstate(invalid="ignore"):
        median_rate = np.nanmedian(rate, axis=0)
        q25_rate = np.nanpercentile(rate, 25, axis=0)
        q75_rate = np.nanpercentile(rate, 75, axis=0)
        median_sleep = np.nanmedian(sleep_fraction, axis=0)

    finite_rate = rate[np.isfinite(rate)]
    rate_vmax = float(np.nanpercentile(finite_rate, 98)) if len(finite_rate) else 1.0
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, max(7, 0.055 * n_rows + 5.0)),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 4.2, 1.8]},
    )
    ax_top, ax_rate, ax_state = axes
    ax_top.fill_between(x_centers, q25_rate, q75_rate, color="tab:red", alpha=0.18)
    ax_top.plot(x_centers, median_rate, color="tab:red", lw=1.9, label="median interaction onset rate")
    ax_top.axvline(0, color="0.25", lw=0.9, ls="--")
    ax_top.set_ylabel("onsets / min")
    ax_top.grid(True, alpha=0.25)
    ax_state_top = ax_top.twinx()
    ax_state_top.plot(x_centers, median_sleep, color="0.2", lw=1.1, alpha=0.8, label="median sleep fraction")
    ax_state_top.set_ylim(0, 1)
    ax_state_top.set_ylabel("sleep fraction", color="0.2")
    ax_state_top.tick_params(axis="y", colors="0.2")

    im_rate = ax_rate.pcolormesh(
        x_edges,
        np.arange(n_rows + 1),
        np.ma.masked_invalid(rate),
        shading="flat",
        cmap="magma",
        vmin=0.0,
        vmax=max(rate_vmax, 1e-6),
    )
    im_sleep = ax_state.pcolormesh(
        x_edges,
        np.arange(n_rows + 1),
        np.ma.masked_invalid(sleep_fraction),
        shading="flat",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    tick_step = max(1, n_rows // 12)
    tick_rows = np.arange(0, n_rows, tick_step)
    for ax in [ax_rate, ax_state]:
        ax.axvline(0, color="white", lw=1.1, ls="--")
        ax.axvline(0, color="0.15", lw=0.5, ls="--")
        ax.set_ylabel("sleep bouts")
        ax.invert_yaxis()
        ax.set_yticks(tick_rows + 0.5)
        ax.set_yticklabels(
            [
                f"{int(plot_summary.iloc[int(r)]['bout_trigger_row'])} | "
                f"{plot_summary.iloc[int(r)]['bout_duration_minutes']:.1f}m"
                for r in tick_rows
            ],
            fontsize=8,
        )
    order_label = {
        "duration_desc": "longest to shortest",
        "duration_asc": "shortest to longest",
        "end_frame": "sleep-end time",
        "post_rate_desc": "post-wake interaction peak",
    }.get(order_by, "matrix order")
    ax_rate.set_title(f"interaction onset rate aligned to sleep end (rows: {order_label})")
    ax_state.set_title(f"sleep/wake classification aligned to sleep end (rows: {order_label})")
    ax_state.set_xlabel("minutes relative to sleep end")
    fig.colorbar(im_rate, ax=ax_rate, label="interaction onsets / min")
    fig.colorbar(im_sleep, ax=ax_state, label="sleep fraction")
    track_id = matrices.get("track_id", "")
    track_name = matrices.get("track_name", "")
    event_role = matrices.get("interaction_event_role", "both")
    fig.suptitle(
        f"Track {track_id}: interaction onset rate triggered on sleep end\n"
        f"{track_name}; window={float(matrices['rate_window_seconds']):g}s, "
        f"step={float(matrices['rate_step_seconds']):g}s, "
        f"event gap={float(matrices.get('interaction_event_gap_seconds', np.nan)):g}s, "
        f"role={event_role}"
    )
    fig.tight_layout()
    plt.show()


def sleep_activity_shift_summary(cluster_sleep_time: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, group in cluster_sleep_time.groupby(ACTIVITY_CLUSTER_COL, sort=True):
        group = group.sort_values("time_bin")
        sleep = group["sleep_fraction_valid_time"].fillna(0).to_numpy(np.float64)
        active = group["mean_ant_active_fraction"].fillna(0).to_numpy(np.float64)
        if len(group) == 0:
            continue
        sleep_peak_h = float(group["hours_since_light_on"].iloc[int(np.argmax(sleep))])
        active_peak_h = float(group["hours_since_light_on"].iloc[int(np.argmax(active))])
        lag_h = ((sleep_peak_h - active_peak_h + 12.0) % 24.0) - 12.0
        sleep_mean_h, sleep_conc = circular_mean_hours(group["hours_since_light_on"].to_numpy(), sleep)
        active_mean_h, active_conc = circular_mean_hours(group["hours_since_light_on"].to_numpy(), active)
        circular_mean_lag_h = ((sleep_mean_h - active_mean_h + 12.0) % 24.0) - 12.0
        rows.append(
            {
                ACTIVITY_CLUSTER_COL: cluster,
                "active_peak_h": active_peak_h,
                "sleep_peak_h": sleep_peak_h,
                "sleep_minus_active_peak_lag_h": lag_h,
                "active_circular_mean_h": active_mean_h,
                "sleep_circular_mean_h": sleep_mean_h,
                "sleep_minus_active_circular_lag_h": circular_mean_lag_h,
                "active_phase_concentration": active_conc,
                "sleep_phase_concentration": sleep_conc,
                "n_ants": int(group["n_ants"].max()),
            }
        )
    return pd.DataFrame(rows)


# %%
# Load metadata and interaction rows.
chunks = ia.resolve_chunks(
    INTERACTION_ROOT,
    TRACKS_ROOT,
    chunks=CHUNKS,
    side=SIDE,
    fps=FPS,
    max_chunks=MAX_CHUNKS,
)
chunk_summary = ia.describe_chunks(chunks)
ANALYSIS_FRAME_START = min(int(chunk.chunk_global_frame_offset) for chunk in chunks)
ANALYSIS_FRAME_STOP = max(int(chunk.chunk_global_frame_offset + chunk.chunk_frame_count) for chunk in chunks)
RECORDING_START_CLOCK_SECONDS = int(chunks[0].recording_start_clock_seconds)

grid_tracks_all = go.load_grid_tracks(GRID_ROOT)
grid_tracks_all = go.attach_detection_fraction(grid_tracks_all, SPEED_ROOT)
grid_tracks = go.select_good_tracks(grid_tracks_all, MIN_PRESENT_FRAC, side=SIDE)

speed_tracks_all = cs.load_speed_tracks(SPEED_ROOT)
speed_tracks = cs.select_tracks(speed_tracks_all, MIN_PRESENT_FRAC)
speed_cols = [
    "track_name",
    "speed_path",
    "frame_min",
    "frame_max",
    "n_frames",
    "fps",
]
activity_tracks = grid_tracks.merge(
    speed_tracks[speed_cols].rename(
        columns={
            "frame_min": "speed_frame_min",
            "frame_max": "speed_frame_max",
            "n_frames": "speed_n_frames",
            "fps": "speed_fps",
        }
    ),
    on="track_name",
    how="inner",
    validate="one_to_one",
)
activity_tracks = activity_tracks[activity_tracks["speed_path"].map(lambda path: Path(path).exists())].reset_index(drop=True)
if activity_tracks.empty:
    raise ValueError("No tracks have both grid occupancy histograms and speed vectors")

interactions_raw = ia.load_interactions_for_chunks(chunks)
interaction_onset_counts_by_track = build_interaction_onset_counts_by_track(
    interactions_raw,
    fps=FPS,
    event_gap_seconds=INTERACTION_EVENT_GAP_SECONDS,
    analysis_frame_start=ANALYSIS_FRAME_START,
    analysis_frame_stop=ANALYSIS_FRAME_STOP,
)

print(f"Chunk selection: {chunk_summary} ({SIDE})")
print(f"Frame window: {ANALYSIS_FRAME_START:,}-{ANALYSIS_FRAME_STOP:,}")
print(f"Recording start clock: {go.format_clock_time(RECORDING_START_CLOCK_SECONDS)}")
print(f"Activity tracks: {len(activity_tracks):,}")
print(f"Frame-level interaction detections loaded: {len(interactions_raw):,}")
print(
    "Interaction onsets counted: "
    f"{sum(int(counts['n_interactions_total'].sum()) for counts in interaction_onset_counts_by_track.values()):,} "
    f"(event gap {INTERACTION_EVENT_GAP_SECONDS:g}s)"
)
display(activity_tracks[["track_id", "side", "track_name", "present_frac", "n_observed_frames"]].head(20))


# %%
# Spatiotemporal activity clustering: spatial occupancy + time-of-day speed/active/quiet profiles.
activity_cluster_table, track_activity_long, activity_grid_tracks, joint_activity_features, graph_activity_features = (
    build_spatiotemporal_activity_clusters(
        activity_tracks,
        recording_start_clock_seconds=RECORDING_START_CLOCK_SECONDS,
        analysis_frame_start=ANALYSIS_FRAME_START,
        analysis_frame_stop=ANALYSIS_FRAME_STOP,
    )
)
activity_cluster_table["activity_cluster_id"] = (
    activity_cluster_table["side"].astype(str) + "_A" + activity_cluster_table[ACTIVITY_CLUSTER_COL].astype(str)
)
activity_cluster_summary = (
    activity_cluster_table.groupby(ACTIVITY_CLUSTER_COL, as_index=False)
    .agg(
        activity_cluster_id=("activity_cluster_id", "first"),
        n_tracks=("track_name", "nunique"),
        median_present_frac=("present_frac", "median"),
        median_track_id=("track_id", "median"),
    )
    .sort_values(ACTIVITY_CLUSTER_COL)
)
display(activity_cluster_summary)

go.plot_umap_clusters(
    activity_cluster_table.rename(columns={ACTIVITY_CLUSTER_COL: "leiden_cluster"}),
    color_col="leiden_cluster",
    title=f"{SIDE} spatiotemporal activity clusters",
)
activity_cluster_time_profiles = plot_activity_cluster_time_profiles(track_activity_long)


# %%
# Spatial footprint of each activity cluster.
activity_cluster_mean_hists = go.plot_cluster_mean_histograms(
    activity_grid_tracks,
    activity_cluster_table,
    cluster_col=ACTIVITY_CLUSTER_COL,
    mode="sqrt",
    title=f"{SIDE} activity cluster mean occupancy maps",
)


# %%
# Load sleep classifier predictions generated by analysis/compute_track_sleep_predictions.py.
sleep_prediction_tracks, predicted_sleep_bouts = load_activity_cluster_sleep_predictions(
    activity_tracks,
    activity_cluster_table,
    interaction_onset_counts_by_track,
    analysis_frame_start=ANALYSIS_FRAME_START,
    analysis_frame_stop=ANALYSIS_FRAME_STOP,
)
sleep_bouts = predicted_sleep_bouts
if sleep_bouts.empty:
    raise ValueError(f"No predicted sleep bouts were available under {SLEEP_PREDICTIONS_ROOT}")
if SAVE_SLEEP_BOUTS:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sleep_bouts.to_parquet(SLEEP_BOUTS_PARQUET, index=False)
    print(f"Saved predicted sleep bouts: {SLEEP_BOUTS_PARQUET}")

sleep_prediction_summary = (
    sleep_bouts.groupby([ACTIVITY_CLUSTER_COL], as_index=False)
    .agg(
        n_sleep_bouts=("bout_id", "size"),
        n_ants=("track_name", "nunique"),
        total_sleep_seconds=("bout_duration_seconds", "sum"),
        median_sleep_bout_seconds=("bout_duration_seconds", "median"),
        mean_sleep_probability=("mean_sleep_probability", "mean"),
        median_interaction_onsets=("n_interaction_onsets", "median"),
    )
    .sort_values(ACTIVITY_CLUSTER_COL, kind="mergesort")
)
display(
    sleep_prediction_tracks[
        [
            "track_id",
            "side",
            "track_name",
            ACTIVITY_CLUSTER_COL,
            "sleep_prediction_present_frac",
            "sleep_fraction_predicted_frames",
            "mean_sleep_probability",
        ]
    ].head(20)
)
display(sleep_prediction_summary)
display(sleep_bouts.head(20))


# %%
# Timing of predicted sleep by activity cluster.
cluster_sleep_time, ant_sleep_time = build_sleep_time_tables(
    sleep_bouts,
    track_activity_long,
    recording_start_clock_seconds=RECORDING_START_CLOCK_SECONDS,
)
intra_cluster_interactions = build_intra_cluster_interaction_time_table(
    interactions_raw,
    activity_cluster_table,
    recording_start_clock_seconds=RECORDING_START_CLOCK_SECONDS,
    fps=FPS,
    event_gap_seconds=INTERACTION_EVENT_GAP_SECONDS,
)
cluster_interactions = build_activity_cluster_interaction_time_table(
    interactions_raw,
    activity_cluster_table,
    recording_start_clock_seconds=RECORDING_START_CLOCK_SECONDS,
    fps=FPS,
    event_gap_seconds=INTERACTION_EVENT_GAP_SECONDS,
)
plot_cluster_sleep_timing(cluster_sleep_time, intra_cluster_interactions)
sleep_interaction_time, sleep_interaction_correlation = sleep_interaction_time_correlation(
    cluster_sleep_time,
    cluster_interactions,
)
plot_sleep_interaction_time_correlation(sleep_interaction_time, sleep_interaction_correlation)
sleep_activity_interaction_relationships = sleep_activity_interaction_relationship_summary(
    sleep_interaction_time,
)
plot_sleep_activity_interaction_profiles(sleep_interaction_time, sleep_activity_interaction_relationships)
plot_sleep_activity_interaction_pairplots(sleep_interaction_time, sleep_activity_interaction_relationships)
plot_sleep_activity_interaction_reducibility(sleep_activity_interaction_relationships)
display(cluster_sleep_time.head(30))
display(intra_cluster_interactions.head(30))
display(cluster_interactions.head(30))
display(sleep_interaction_correlation)
display(sleep_activity_interaction_relationships)
display(sleep_interaction_time.head(30))


# %%
# Event-triggered wake-up test: does a local interaction reduce sleep more than
# same-ant, same-time-of-day non-interaction controls?
(
    interaction_wake_events,
    interaction_wake_controls,
    interaction_wake_matched_effects,
    interaction_wake_summary,
    interaction_wake_curve,
) = interaction_triggered_wake_analysis(
    sleep_bouts,
    interaction_onset_counts_by_track,
    activity_cluster_table,
    analysis_frame_start=ANALYSIS_FRAME_START,
    analysis_frame_stop=ANALYSIS_FRAME_STOP,
    recording_start_clock_seconds=RECORDING_START_CLOCK_SECONDS,
)
display(interaction_wake_summary)
if not interaction_wake_matched_effects.empty:
    display(interaction_wake_matched_effects.head(30))
if not interaction_wake_events.empty:
    display(interaction_wake_events.head(30))
if not interaction_wake_controls.empty:
    display(interaction_wake_controls.head(30))
plot_interaction_triggered_wake_curve(interaction_wake_curve)


# %%
# Fine-timescale interaction rates triggered on sleep end for one ant.
selected_sleep_end_trigger_track_id = choose_sleep_end_trigger_track_id(
    sleep_bouts,
    track_id=SLEEP_END_TRIGGER_TRACK_ID,
    track_name=SLEEP_END_TRIGGER_TRACK_NAME,
)
(
    sleep_end_trigger_bout_summary,
    sleep_end_trigger_interaction_rate,
    sleep_end_trigger_matrices,
) = build_sleep_end_triggered_interaction_rate(
    sleep_bouts,
    interactions_raw,
    track_id=selected_sleep_end_trigger_track_id,
    analysis_frame_start=ANALYSIS_FRAME_START,
    analysis_frame_stop=ANALYSIS_FRAME_STOP,
    fps=FPS,
    event_gap_seconds=SLEEP_END_TRIGGER_INTERACTION_EVENT_GAP_SECONDS,
)
display(
    sleep_end_trigger_bout_summary[
        [
            "bout_trigger_row",
            "track_id",
            "sleep_end_elapsed_hours",
            "bout_duration_minutes",
            "pre_median_interaction_onset_rate_per_min",
            "post_median_interaction_onset_rate_per_min",
            "post_peak_interaction_onset_rate_per_min",
            "post_minus_pre_median_onset_rate_per_min",
        ]
    ].head(30)
)
display(sleep_end_trigger_interaction_rate.head(30))
plot_sleep_end_triggered_interaction_rate(
    sleep_end_trigger_bout_summary,
    sleep_end_trigger_matrices,
    row_order_by=SLEEP_END_TRIGGER_PLOT_ORDER_BY,
)


# %%
# Elapsed recording time analysis: no folding by time of day.
cluster_elapsed_sleep_time, ant_elapsed_sleep_time = build_elapsed_sleep_time_tables(
    sleep_bouts,
    activity_tracks,
    activity_cluster_table,
    analysis_frame_start=ANALYSIS_FRAME_START,
    analysis_frame_stop=ANALYSIS_FRAME_STOP,
)
sleep_repetition_autocorr = sleep_repetition_autocorrelation_table(ant_elapsed_sleep_time)
sleep_repetition_summary = summarize_sleep_repetition_autocorrelation(sleep_repetition_autocorr)
top_sleep_repetition_lags_table = top_sleep_repetition_lags(sleep_repetition_summary)
repeating_sleep_ants = top_repeating_sleep_ants(sleep_repetition_autocorr)

display(cluster_elapsed_sleep_time.head(30))
display(sleep_repetition_summary.head(30))
display(top_sleep_repetition_lags_table)
display(repeating_sleep_ants)

plot_elapsed_cluster_sleep_timing(cluster_elapsed_sleep_time)
plot_elapsed_sleep_heatmap(ant_elapsed_sleep_time)
plot_sleep_repetition_autocorrelation(sleep_repetition_summary)
plot_repeating_sleep_ant_examples(ant_elapsed_sleep_time, repeating_sleep_ants)


# %%
# Test whether predicted sleep is flatter over time of day than activity.
sleep_activity_modulation, sleep_activity_cross_correlations = cluster_sleep_activity_modulation_summary(
    cluster_sleep_time,
    ant_sleep_time,
)
modulation_display_cols = [
    ACTIVITY_CLUSTER_COL,
    "n_ants",
    "sleep_peak_to_trough",
    "active_peak_to_trough",
    "sleep_to_active_peak_to_trough_ratio",
    "sleep_cv",
    "active_cv",
    "sleep_to_active_cv_ratio",
    "sleep_time_bin_eta2",
    "active_time_bin_eta2",
    "sleep_to_active_time_eta2_ratio",
    "sleep_time_bin_eta2_permutation_p",
    "active_time_bin_eta2_permutation_p",
    "sleep_activity_zero_lag_corr",
    "best_positive_sleep_activity_corr",
    "best_positive_sleep_activity_lag_h",
    "strongest_abs_sleep_activity_corr",
    "strongest_abs_sleep_activity_lag_h",
]
display(sleep_activity_modulation[[col for col in modulation_display_cols if col in sleep_activity_modulation.columns]])
plot_sleep_activity_modulation(sleep_activity_modulation)


# %%
# Between-ant timing inside each activity cluster.
ant_sleep_phase = ant_sleep_phase_table(ant_sleep_time)
within_cluster_sleep_phase = within_cluster_sleep_phase_summary(ant_sleep_phase)
sleep_cross_correlations = pairwise_sleep_cross_correlation_table(ant_sleep_time)
pairwise_sleep_lags = best_sleep_cross_correlation_lags(sleep_cross_correlations)
sleep_cross_correlation_summary = summarize_sleep_cross_correlations(
    sleep_cross_correlations,
    pairwise_sleep_lags,
)
activity_cross_correlations = pairwise_activity_cross_correlation_table(ant_sleep_time)
pairwise_activity_lags = best_activity_cross_correlation_lags(activity_cross_correlations)
activity_cross_correlation_summary = summarize_activity_cross_correlations(
    activity_cross_correlations,
    pairwise_activity_lags,
)
paired_sleep_activity_synchrony = paired_sleep_activity_synchrony_table(
    pairwise_sleep_lags,
    pairwise_activity_lags,
)
paired_sleep_activity_synchrony_summary = summarize_paired_sleep_activity_synchrony(
    paired_sleep_activity_synchrony,
)
shifted_sleep_pairs = shifted_sleep_pair_candidates(pairwise_sleep_lags)
shifted_sleep_pair_summary = summarize_shifted_sleep_candidates(shifted_sleep_pairs)
sleep_work_shift_table = sleep_activity_shift_summary(cluster_sleep_time)

display(within_cluster_sleep_phase)
display(sleep_work_shift_table)
display(sleep_cross_correlation_summary)
display(activity_cross_correlation_summary)
display(paired_sleep_activity_synchrony_summary)
display(shifted_sleep_pair_summary)
if not pairwise_sleep_lags.empty:
    display(pairwise_sleep_lags.head(30))
if not paired_sleep_activity_synchrony.empty:
    display(paired_sleep_activity_synchrony.head(30))
if not shifted_sleep_pairs.empty:
    display(shifted_sleep_pairs.head(30))

plot_ant_sleep_heatmap(ant_sleep_time, ant_sleep_phase)
plot_sleep_cross_correlations(sleep_cross_correlations)
plot_activity_cross_correlations(activity_cross_correlations)
plot_sleep_activity_synchrony_comparison(paired_sleep_activity_synchrony)
plot_sleep_shift_lag_distributions(pairwise_sleep_lags, paired_sleep_activity_synchrony)
plot_sleep_shift_correlation_scatter(paired_sleep_activity_synchrony)
plot_sleep_phase_shift_heatmaps(ant_sleep_time, ant_sleep_phase)
plot_sleep_best_lag_matrix(pairwise_sleep_lags, ant_sleep_phase)
plot_shifted_sleep_pair_examples(ant_sleep_time, shifted_sleep_pairs)


# %%
# Tables to inspect interactively:
# - activity_cluster_table: ant-level spatiotemporal activity cluster assignments.
# - track_activity_long: ant x time-of-day speed/activity/quiet profiles.
# - interaction_onset_counts_by_track: per-ant encounter-onset counts; not frame-level contact detections.
# - sleep_prediction_tracks: activity-cluster tracks with sleep prediction metadata.
# - predicted_sleep_bouts / sleep_bouts: classifier-predicted sleep bouts.
# - cluster_sleep_time: cluster x time-of-day sleep/activity timing.
# - cluster_elapsed_sleep_time, ant_elapsed_sleep_time: sleep/activity over absolute recording time.
# - sleep_repetition_autocorr, sleep_repetition_summary: non-circular sleep repetition over elapsed time.
# - intra_cluster_interactions: within-cluster undirected interaction-bout onset timing.
# - cluster_interactions: directed source/receiver interaction-bout timing for focal ants in each activity cluster.
# - sleep_interaction_time, sleep_interaction_correlation: time-of-day sleep vs directed interaction amount by cluster.
# - sleep_activity_interaction_relationships: correlations, partial correlations, and R2 checks for activity, interactions, and sleep.
# - interaction_wake_events: receiver/body interaction-bout triggers where the ant was mostly asleep just before the event.
# - interaction_wake_controls: same-ant, same-time-of-day no-onset controls matched to wake events.
# - interaction_wake_matched_effects, interaction_wake_summary: event-vs-control wake-up effect estimates.
# - interaction_wake_curve: event-triggered sleep curve around interaction onsets and matched controls.
# - selected_sleep_end_trigger_track_id: ant selected for fine-timescale sleep-end triggering.
# - sleep_end_trigger_bout_summary: one row per sleep end with pre/post interaction-onset-rate summaries.
# - sleep_end_trigger_interaction_rate: continuous moving-window interaction-onset-rate signal around each sleep end.
# - sleep_end_trigger_matrices: plotting-ready interaction-onset-rate and sleep/wake matrices aligned to sleep end.
# - sleep_activity_modulation: quantifies whether sleep is flatter than activity over time of day.
# - sleep_cross_correlations: full lag-by-lag within-cluster sleep cross-correlations between ants.
# - activity_cross_correlations: same cross-correlation analysis for activity_fraction.
# - paired_sleep_activity_synchrony: same ant pairs, sleep synchrony vs activity synchrony.
# - shifted_sleep_pairs: high-best, low-zero-lag sleep pairs with nonzero best lags.
# - within_cluster_sleep_phase, pairwise_sleep_lags, sleep_work_shift_table: shift diagnostics.
print("Ready for interactive inspection.")
