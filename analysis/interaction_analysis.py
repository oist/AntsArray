# %%
# VS Code/Jupyter interactive script for directed interaction analysis.
try:
    get_ipython().run_line_magic("matplotlib", "qt")  # type: ignore[name-defined]
except Exception:
    pass

import importlib
import sys
from pathlib import Path

import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = print

repo_root = Path.cwd().resolve()
for candidate in [repo_root, *repo_root.parents]:
    if (candidate / "analysis" / "interaction_analysis_utils.py").exists():
        repo_root = candidate
        break
else:
    raise FileNotFoundError("Could not find analysis/interaction_analysis_utils.py from the current working directory")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import analysis.interaction_analysis_utils as ia
from analysis.figure_saving import install_auto_savefig

importlib.reload(ia)


# %%
# Edit these settings first. Defaults intentionally load one chunk/side for quick testing.
DATASET_ROOT = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02")
INTERACTION_ROOT = DATASET_ROOT / "interactions"
TRACKS_ROOT = DATASET_ROOT / "tracks"
GRID_ROOT = DATASET_ROOT / "stitched" / "grid_occupancy_histograms"
CLUSTER_TABLE_PATH = GRID_ROOT / "track_cluster_ids.csv"
SPEED_ROOT = DATASET_ROOT / "stitched" / "speed_vectors"
CACHE_ROOT = DATASET_ROOT / "stitched" / "analysis_cache" / "interaction_analysis"
FIGURE_ROOT = DATASET_ROOT / "analysis_outputs" / "interaction_analysis_figures"
SAVE_FIGURES = True
FIGURE_DPI = 180

# Use "all" for every chunk on this side, "000" for one chunk, or ["000", "001"].
CHUNKS = "all"
SIDE = "right"  # "left" or "right"
FPS = 24.0
MM_PER_PX = 0.016
POSITION_BODYPOINT = 0

# Set MAX_CHUNKS or MAX_INTERACTIONS_PER_CHUNK for fast smoke tests.
MAX_CHUNKS = None
MAX_INTERACTIONS_PER_CHUNK = None
SAMPLE_INTERACTIONS = False
DROP_UNCLUSTERED = True
USE_CACHE = True
FORCE_REBUILD_CACHE = False
NORMALIZE_BY_N_ANTS = True

LIGHT_ON_HOUR = 5.5
TIME_BIN_MINUTES = 30.0


# %%
# Resolve files and load interaction rows and cluster IDs.
chunks = ia.resolve_chunks(
    INTERACTION_ROOT,
    TRACKS_ROOT,
    chunks=CHUNKS,
    side=SIDE,
    fps=FPS,
    max_chunks=MAX_CHUNKS,
)
chunk_summary = ia.describe_chunks(chunks)
clusters = ia.load_cluster_table(CLUSTER_TABLE_PATH, side=SIDE)
cache_settings = {
    "cache_version": 2,
    "side": SIDE,
    "chunks": CHUNKS,
    "max_chunks": MAX_CHUNKS,
    "max_interactions_per_chunk": MAX_INTERACTIONS_PER_CHUNK,
    "sample_interactions": SAMPLE_INTERACTIONS,
    "drop_unclustered": DROP_UNCLUSTERED,
    "fps": FPS,
    "mm_per_px": MM_PER_PX,
    "position_bodypoint": POSITION_BODYPOINT,
    "light_on_hour": LIGHT_ON_HOUR,
    "time_bin_minutes": TIME_BIN_MINUTES,
    "cluster_table": ia.file_fingerprint(CLUSTER_TABLE_PATH),
}
cache_key = ia.interaction_analysis_cache_key(chunks, cache_settings)
cache_dir = CACHE_ROOT / f"{SIDE}_{cache_key}"
install_auto_savefig(
    FIGURE_ROOT / f"{SIDE}_{cache_key}",
    prefix=f"interaction_analysis_{SIDE}",
    dpi=FIGURE_DPI,
    enabled=SAVE_FIGURES,
)


def derived_cache_key(stage: str, **settings) -> str:
    return ia.interaction_analysis_cache_key(
        chunks,
        {
            **cache_settings,
            "stage": stage,
            **settings,
        },
    )


EVENT_WEIGHT_COL = "interaction_count_per_ant" if NORMALIZE_BY_N_ANTS else "interaction_count"
PAIR_VALUE_COL = "interactions_per_directed_ant_pair" if NORMALIZE_BY_N_ANTS else "n_interactions"

interactions_raw = ia.load_or_build_table(
    cache_dir / "interactions_raw.parquet",
    lambda: ia.load_interactions_for_chunks(
        chunks,
        max_interactions_per_chunk=MAX_INTERACTIONS_PER_CHUNK,
        sample=SAMPLE_INTERACTIONS,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
interactions = ia.load_or_build_table(
    cache_dir / "interactions_clustered.parquet",
    lambda: ia.attach_cluster_labels(interactions_raw, clusters, drop_unclustered=DROP_UNCLUSTERED),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
cluster_ant_counts = ia.load_or_build_table(
    cache_dir / "cluster_ant_counts.parquet",
    lambda: ia.cluster_ant_counts(clusters),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)

chunk_table = pd.DataFrame(
    [
        {
            "chunk": chunk.chunk,
            "start": ia.format_clock_time(chunk.chunk_start_clock_seconds),
            "frame_offset": chunk.chunk_global_frame_offset,
            "n_frames": chunk.chunk_frame_count,
            "interaction_file": chunk.interaction_path.name,
            "track_file": chunk.track_path.name,
        }
        for chunk in chunks
    ]
)

print(f"Chunk selection: {chunk_summary} ({SIDE})")
print(f"Interaction rows loaded: {len(interactions_raw):,}")
print(f"Interaction rows after cluster filter: {len(interactions):,}")
print(f"Clustered tracks: {len(clusters):,}")
print(f"Cluster-normalized analysis: {NORMALIZE_BY_N_ANTS} ({EVENT_WEIGHT_COL})")
print(f"Cache directory: {cache_dir}")
display(chunk_table)
display(interactions.head())
display(clusters.head())
display(cluster_ant_counts)


# %%
# Summarize directed cluster pairs: antenna/source cluster -> body/receiver cluster.
pair_cache_key = derived_cache_key(
    "cluster_pair_table",
    normalize_by_n_ants=NORMALIZE_BY_N_ANTS,
)
cluster_pair_table = ia.load_or_build_table(
    cache_dir / f"cluster_pair_table_{pair_cache_key}.parquet",
    lambda: ia.cluster_pair_counts(
        interactions,
        clusters=clusters,
        normalize_by_n_ants=NORMALIZE_BY_N_ANTS,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
display(cluster_pair_table.head(30))

cluster_pair_matrix = ia.plot_cluster_pair_matrix_from_table(
    cluster_pair_table,
    value_col=PAIR_VALUE_COL,
    title=f"{SIDE} {chunk_summary} relative directed interactions by occupancy cluster",
)
display(cluster_pair_matrix)


# %%
# Build weighted spatial event tables. Positions are loaded one chunk at a time.
# Each row is one ant/frame/cluster with interaction_count equal to the number
# of directed interactions involving that ant in that role.
antenna_events_raw = ia.load_or_build_table(
    cache_dir / "antenna_events.parquet",
    lambda: ia.role_event_positions_for_chunks(
        interactions,
        chunks,
        role="antenna",
        side=SIDE,
        mm_per_px=MM_PER_PX,
        fps=FPS,
        light_on_hour=LIGHT_ON_HOUR,
        time_bin_minutes=TIME_BIN_MINUTES,
        bodypoint=POSITION_BODYPOINT,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
body_events_raw = ia.load_or_build_table(
    cache_dir / "body_events.parquet",
    lambda: ia.role_event_positions_for_chunks(
        interactions,
        chunks,
        role="body",
        side=SIDE,
        mm_per_px=MM_PER_PX,
        fps=FPS,
        light_on_hour=LIGHT_ON_HOUR,
        time_bin_minutes=TIME_BIN_MINUTES,
        bodypoint=POSITION_BODYPOINT,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
if NORMALIZE_BY_N_ANTS:
    antenna_events = ia.load_or_build_table(
        cache_dir / "antenna_events_per_ant.parquet",
        lambda: ia.add_cluster_size_weights(antenna_events_raw, clusters),
        use_cache=USE_CACHE,
        force=FORCE_REBUILD_CACHE,
    )
    body_events = ia.load_or_build_table(
        cache_dir / "body_events_per_ant.parquet",
        lambda: ia.add_cluster_size_weights(body_events_raw, clusters),
        use_cache=USE_CACHE,
        force=FORCE_REBUILD_CACHE,
    )
else:
    antenna_events = antenna_events_raw
    body_events = body_events_raw

print(f"Antenna/source spatial events: {len(antenna_events):,}")
print(f"Body/receiver spatial events: {len(body_events):,}")
display(antenna_events.head())


# %%
# Spatial dependence by antenna/source occupancy cluster.
SPATIAL_BIN_SIZE_MM = 10.0
MAX_CLUSTERS = None
SPATIAL_NCOLS = 3
SPATIAL_NORMALIZE = False
SPATIAL_VMAX_PERCENTILE = 99.0

antenna_cluster_hists_result = ia.load_or_build_pickle(
    cache_dir / f"antenna_cluster_hists_{derived_cache_key('antenna_cluster_hists', bin_size_mm=SPATIAL_BIN_SIZE_MM, max_clusters=MAX_CLUSTERS, normalize=SPATIAL_NORMALIZE, weight_col=EVENT_WEIGHT_COL)}.pkl",
    lambda: ia.spatial_heatmaps_by_cluster(
        antenna_events,
        bin_size_mm=SPATIAL_BIN_SIZE_MM,
        max_clusters=MAX_CLUSTERS,
        normalize=SPATIAL_NORMALIZE,
        weight_col=EVENT_WEIGHT_COL,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
antenna_cluster_hists = ia.plot_spatial_heatmaps_by_cluster_result(
    antenna_cluster_hists_result,
    ncols=SPATIAL_NCOLS,
    vmax_percentile=SPATIAL_VMAX_PERCENTILE,
    title=f"{SIDE} {chunk_summary} antenna/source interaction locations by cluster",
)


# %%
# Spatial dependence by body/receiver occupancy cluster.
body_cluster_hists_result = ia.load_or_build_pickle(
    cache_dir / f"body_cluster_hists_{derived_cache_key('body_cluster_hists', bin_size_mm=SPATIAL_BIN_SIZE_MM, max_clusters=MAX_CLUSTERS, normalize=SPATIAL_NORMALIZE, weight_col=EVENT_WEIGHT_COL)}.pkl",
    lambda: ia.spatial_heatmaps_by_cluster(
        body_events,
        bin_size_mm=SPATIAL_BIN_SIZE_MM,
        max_clusters=MAX_CLUSTERS,
        normalize=SPATIAL_NORMALIZE,
        weight_col=EVENT_WEIGHT_COL,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
body_cluster_hists = ia.plot_spatial_heatmaps_by_cluster_result(
    body_cluster_hists_result,
    ncols=SPATIAL_NCOLS,
    vmax_percentile=SPATIAL_VMAX_PERCENTILE,
    title=f"{SIDE} {chunk_summary} body/receiver interaction locations by cluster",
)


# %%
# Time-of-day dependence of interaction counts by cluster.
TIME_COUNT_ROLE = "antenna"  # "antenna" or "body"
TIME_COUNT_MAX_CLUSTERS = None
TIME_COUNT_AVERAGE_OVER_DAYS = True

time_count_events = antenna_events if TIME_COUNT_ROLE == "antenna" else body_events
time_count_table = ia.load_or_build_pickle(
    cache_dir / f"time_count_table_{derived_cache_key('time_count_table', role=TIME_COUNT_ROLE, max_clusters=TIME_COUNT_MAX_CLUSTERS, average_over_days=TIME_COUNT_AVERAGE_OVER_DAYS, weight_col=EVENT_WEIGHT_COL)}.pkl",
    lambda: ia.time_counts_by_cluster(
        time_count_events,
        max_clusters=TIME_COUNT_MAX_CLUSTERS,
        average_over_days=TIME_COUNT_AVERAGE_OVER_DAYS,
        weight_col=EVENT_WEIGHT_COL,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
time_count_color_label = "directed interactions / ant" if NORMALIZE_BY_N_ANTS else "directed interactions"
if TIME_COUNT_AVERAGE_OVER_DAYS:
    time_count_color_label = (
        "mean directed interactions / ant / light-cycle day"
        if NORMALIZE_BY_N_ANTS
        else "mean directed interactions / light-cycle day"
    )
ia.plot_time_counts_table(
    time_count_table,
    color_label=time_count_color_label,
    title=(
        f"{SIDE} {chunk_summary} {TIME_COUNT_ROLE} interaction counts "
        f"by {TIME_BIN_MINUTES:g}-min time bin since light on"
    ),
)
display(time_count_table)


# %%
# Time series of directed interaction rate by cluster.
TIMESERIES_ROLE = "antenna"  # "antenna" or "body"
TIMESERIES_MAX_CLUSTERS = None
TIMESERIES_SMOOTH_BINS = 0.0
TIMESERIES_AVERAGE_OVER_DAYS = True
TIMESERIES_YLIM = None

timeseries_events = antenna_events if TIMESERIES_ROLE == "antenna" else body_events
interaction_rate_timeseries = ia.load_or_build_table(
    cache_dir / f"interaction_rate_timeseries_{derived_cache_key('interaction_rate_timeseries', role=TIMESERIES_ROLE, bin_minutes=TIME_BIN_MINUTES, smooth_bins=TIMESERIES_SMOOTH_BINS, max_clusters=TIMESERIES_MAX_CLUSTERS, average_over_days=TIMESERIES_AVERAGE_OVER_DAYS, weight_col=EVENT_WEIGHT_COL)}.parquet",
    lambda: ia.interaction_timeseries_plot_table_by_cluster(
        timeseries_events,
        bin_minutes=TIME_BIN_MINUTES,
        smooth_bins=TIMESERIES_SMOOTH_BINS,
        max_clusters=TIMESERIES_MAX_CLUSTERS,
        average_over_days=TIMESERIES_AVERAGE_OVER_DAYS,
        weight_col=EVENT_WEIGHT_COL,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
ia.plot_interaction_timeseries_table(
    interaction_rate_timeseries,
    smooth_bins=TIMESERIES_SMOOTH_BINS,
    average_over_days=TIMESERIES_AVERAGE_OVER_DAYS,
    ylim=TIMESERIES_YLIM,
    title=(
        f"{SIDE} {chunk_summary} {TIMESERIES_ROLE} interaction rate "
        f"by {TIME_BIN_MINUTES:g}-min time bin since light on"
    ),
)
display(interaction_rate_timeseries.head(20))


# %%
# Spatial dependence by cluster and time of day. Keep this small for the first check.
TILE_ROLE = "antenna"  # "antenna" or "body"
TILE_MAX_CLUSTERS = 4
TILE_MAX_TIME_BINS = None
TILE_BIN_SIZE_MM = 3.0
TILE_NORMALIZE = False
TILE_AVERAGE_OVER_DAYS = True
TILE_VMAX_PERCENTILE = 99.0

tile_events = antenna_events if TILE_ROLE == "antenna" else body_events
cluster_time_hists_result = ia.load_or_build_pickle(
    cache_dir / f"cluster_time_hists_{derived_cache_key('cluster_time_hists', role=TILE_ROLE, bin_size_mm=TILE_BIN_SIZE_MM, max_clusters=TILE_MAX_CLUSTERS, max_time_bins=TILE_MAX_TIME_BINS, average_over_days=TILE_AVERAGE_OVER_DAYS, normalize=TILE_NORMALIZE, weight_col=EVENT_WEIGHT_COL)}.pkl",
    lambda: ia.spatial_heatmaps_by_cluster_and_time(
        tile_events,
        bin_size_mm=TILE_BIN_SIZE_MM,
        max_clusters=TILE_MAX_CLUSTERS,
        max_time_bins=TILE_MAX_TIME_BINS,
        average_over_days=TILE_AVERAGE_OVER_DAYS,
        normalize=TILE_NORMALIZE,
        weight_col=EVENT_WEIGHT_COL,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
cluster_time_hists = ia.plot_spatial_heatmaps_by_cluster_and_time_result(
    cluster_time_hists_result,
    vmax_percentile=TILE_VMAX_PERCENTILE,
    title=(
        f"{SIDE} {chunk_summary} {TILE_ROLE} interaction locations "
        f"by cluster and {TIME_BIN_MINUTES:g}-min time bin since light on"
    ),
)


# %%
# Test whether immobile ants become mobile after interactions.
#
# This detects immobile bouts from speed vectors, counts directed interactions
# involving that ant during each bout, then records time to the next mobile frame.
IMMOBILE_SPEED_THRESHOLD_MM_S = 0.1
MIN_IMMOBILE_SECONDS = 30.0
IMMOBILITY_COUNT_ALL_INTERACTIONS = True

immobility_interactions = interactions_raw if IMMOBILITY_COUNT_ALL_INTERACTIONS else interactions
immobility_cache_key = ia.interaction_analysis_cache_key(
    chunks,
    {
        **cache_settings,
        "stage": "immobility_bouts",
        "speed_root": str(SPEED_ROOT),
        "speed_threshold_mm_s": IMMOBILE_SPEED_THRESHOLD_MM_S,
        "min_immobile_seconds": MIN_IMMOBILE_SECONDS,
        "count_all_interactions": IMMOBILITY_COUNT_ALL_INTERACTIONS,
    },
)
immobility_bouts = ia.load_or_build_table(
    cache_dir / f"immobility_bouts_{immobility_cache_key}.parquet",
    lambda: ia.immobility_interaction_analysis_for_chunks(
        speed_root=SPEED_ROOT,
        interactions=immobility_interactions,
        clusters=clusters,
        chunks=chunks,
        speed_threshold_mm_s=IMMOBILE_SPEED_THRESHOLD_MM_S,
        min_immobile_seconds=MIN_IMMOBILE_SECONDS,
        fps=FPS,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)

print(f"Immobile speed threshold: <= {IMMOBILE_SPEED_THRESHOLD_MM_S} mm/s")
print(f"Minimum immobile bout: {MIN_IMMOBILE_SECONDS} s")
print(f"Interaction rows counted: {len(immobility_interactions):,}")
print(f"Immobile bouts: {len(immobility_bouts):,}")
if not immobility_bouts.empty:
    display(
        immobility_bouts[
            [
                "track_id",
                "cluster_id",
                "bout_start_frame",
                "bout_end_frame",
                "immobile_duration_seconds",
                "mobility_observed",
                "time_to_mobility_seconds",
                "n_interactions_total",
                "n_interactions_as_antenna",
                "n_interactions_as_body",
                "time_from_first_interaction_to_mobility_seconds",
            ]
        ].head(30)
    )
    display(
        immobility_bouts.groupby("cluster_id", dropna=False)
        .agg(
            n_bouts=("track_id", "size"),
            n_ants=("track_id", "nunique"),
            median_duration_s=("immobile_duration_seconds", "median"),
            median_interactions=("n_interactions_total", "median"),
            frac_with_interactions=("has_interaction", "mean"),
        )
        .reset_index()
    )


# %%
# Multiple regression: what predicts waking from an immobile bout?
#
# Each immobile bout is split into time bins. The final bin of a completed bout
# is labeled woke=1; earlier bins are woke=0. Predictors are elapsed immobility
# time and cumulative interactions involving that ant so far.
WAKE_BIN_SECONDS = 60.0
WAKE_INCLUDE_CENSORED_BOUTS = False
WAKE_MAX_BINS_PER_BOUT = None
WAKE_MODEL_COMPARISON_METRIC = "aic"

wake_cache_key = ia.interaction_analysis_cache_key(
    chunks,
    {
        **cache_settings,
        "stage": "wake_regression_rows",
        "immobility_cache_key": immobility_cache_key,
        "wake_bin_seconds": WAKE_BIN_SECONDS,
        "wake_include_censored": WAKE_INCLUDE_CENSORED_BOUTS,
        "wake_max_bins_per_bout": WAKE_MAX_BINS_PER_BOUT,
        "count_all_interactions": IMMOBILITY_COUNT_ALL_INTERACTIONS,
    },
)
wake_prediction_rows = ia.load_or_build_table(
    cache_dir / f"wake_prediction_rows_{wake_cache_key}.parquet",
    lambda: ia.immobility_wake_prediction_table(
        immobility_bouts,
        immobility_interactions,
        fps=FPS,
        bin_seconds=WAKE_BIN_SECONDS,
        include_censored=WAKE_INCLUDE_CENSORED_BOUTS,
        max_bins_per_bout=WAKE_MAX_BINS_PER_BOUT,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)

print(f"Wake regression rows: {len(wake_prediction_rows):,}")
display(wake_prediction_rows.head(30))

if not wake_prediction_rows.empty:
    wake_model_table, wake_coef_table = ia.fit_wake_logistic_regressions(wake_prediction_rows)
    display(wake_model_table)
    display(wake_coef_table)
    valid_wake_models = wake_model_table.dropna(subset=[WAKE_MODEL_COMPARISON_METRIC])
    if valid_wake_models.empty:
        print("No finite wake model comparison metric; try a smaller WAKE_BIN_SECONDS or more bouts.")
    else:
        wake_model_plot = ia.plot_wake_regression_model_comparison(
            wake_model_table,
            metric=WAKE_MODEL_COMPARISON_METRIC,
        )
        best_wake_model = valid_wake_models.iloc[0]
        print("Best wake model:", best_wake_model["model"])
        print("Standardized score:", best_wake_model["formula_standardized"])
        if pd.notna(best_wake_model["second_to_first_weight"]):
            print(
                "For z-scored predictors, interaction/time weight ratio:",
                f"{best_wake_model['second_to_first_weight']:.3g}",
            )


# %%
# Correlate interaction count during immobility with immobility length or time to mobility.
CORRELATION_X = "n_interactions_total"
CORRELATION_Y = "time_to_mobility_seconds"  # Or "immobile_duration_seconds"
CORRELATION_COMPLETE_ONLY = True
CORRELATION_COLOR_BY = "cluster_id"
CORRELATION_LOG_X = False
CORRELATION_LOG_Y = False

immobility_correlation_points, immobility_correlation_stats = ia.plot_immobility_interaction_correlation(
    immobility_bouts,
    x_col=CORRELATION_X,
    y_col=CORRELATION_Y,
    complete_only=CORRELATION_COMPLETE_ONLY,
    color_col=CORRELATION_COLOR_BY,
    log_x=CORRELATION_LOG_X,
    log_y=CORRELATION_LOG_Y,
    title=f"{SIDE} {chunk_summary}: interactions during immobility vs {CORRELATION_Y}",
)
display(immobility_correlation_stats)
display(immobility_correlation_points.head(30))


# %%
# Change CHUNKS/SIDE above and rerun from the load cell to inspect another selection.
print(
    "Current chunk selection:",
    chunk_summary,
    SIDE,
    "start",
    ia.format_clock_time(chunks[0].chunk_start_clock_seconds),
)
