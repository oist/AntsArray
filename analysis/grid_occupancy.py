# %%
# Canonical VS Code/Jupyter workflow for ant job analysis.
#
# This script intentionally covers only the analyses that are currently ready
# for routine use:
# 1. cluster ants by their spatial occupancy; and
# 2. compare speed through time across those spatial clusters; and
# 3. summarize trip investment and clock-time use by putative roaming ants.
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
    if (candidate / "analysis" / "grid_occupancy_utils.py").exists():
        repo_root = candidate
        break
else:
    raise FileNotFoundError("Could not find analysis/grid_occupancy_utils.py from the current working directory")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import analysis.grid_occupancy_utils as go
from analysis.figure_saving import install_auto_savefig

importlib.reload(go)


# %%
# Edit this dataset path first. The two inputs are sibling preprocessing
# outputs under ``stitched``.
DATASET_ROOT = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02")
GRID_OUTPUT_NAME = "grid_occupancy_histograms_0p5mm_inferred_bounds"

GRID_ROOT = DATASET_ROOT / "stitched" / GRID_OUTPUT_NAME
SPEED_ROOT = DATASET_ROOT / "stitched" / "speed_vectors"
MIN_PRESENT_FRAC = 0.40
LIGHT_OFF_HOUR = 19.5
LIGHT_ON_HOUR = 5.5
SAVE_FIGURES = True
FIGURE_DPI = 180
FIGURE_ROOT = GRID_ROOT / "panorama_region_analysis" / "figures"

install_auto_savefig(
    FIGURE_ROOT,
    prefix="grid_occupancy",
    dpi=FIGURE_DPI,
    enabled=SAVE_FIGURES,
)


# %%
# Load grid histogram metadata and keep only well-observed ants.
track_table = go.load_grid_tracks(GRID_ROOT)
track_table = go.attach_detection_fraction(track_table, SPEED_ROOT)
experiment_start_clock_seconds = go.start_time_from_track_table(track_table)
good_tracks = go.select_good_tracks(track_table, MIN_PRESENT_FRAC, side="both")

print(f"Loaded {len(track_table)} grid histograms from {GRID_ROOT}")
print(f"Detection metadata: {SPEED_ROOT}")
print(f"Selected {len(good_tracks)} tracks with present_frac > {MIN_PRESENT_FRAC}")
print(f"Experiment start clock: {go.format_clock_time(experiment_start_clock_seconds)}")
display(good_tracks.groupby(["side", "present_frac_source"])["track_name"].count().rename("n_tracks"))
display(good_tracks.head())


# %%
# Inspect rows. Use these row numbers for the single-ant spatial plot below.
display(
    good_tracks[
        [
            "side",
            "track_id",
            "track_name",
            "present_frac",
            "n_observed_frames",
            "n_frames",
            "occupancy_sum",
            "n_out_of_grid_detected_frames",
        ]
    ].head(60)
)


# %%
# Sanity-check one ant's spatial occupancy before clustering.
SINGLE_TRACK_ROW = 10        # Row from good_tracks. Set to None to use SINGLE_TRACK_ID.
SINGLE_TRACK_ID = None
SINGLE_TRACK_SIDE = "left"  # Used only when SINGLE_TRACK_ROW is None.
SINGLE_HIST_MODE = "sqrt"   # "linear", "sqrt", or "log1p"
SINGLE_HIST_VMAX_PERCENTILE = 99.0

single_hist, single_x_edges, single_y_edges, single_row = go.plot_single_histogram(
    good_tracks,
    row_number=SINGLE_TRACK_ROW,
    track_id=SINGLE_TRACK_ID,
    side=SINGLE_TRACK_SIDE,
    mode=SINGLE_HIST_MODE,
    vmax_percentile=SINGLE_HIST_VMAX_PERCENTILE,
)
display(single_row)


# %%
# Cluster spatial occupancy separately for the left and right colonies.
CLUSTER_SIDES = ("left", "right")
FEATURE_TRANSFORM = "sqrt"        # "none", "sqrt", or "log1p"
NEIGHBOR_METRIC = "euclidean"
N_NEIGHBORS = 10
UMAP_MIN_DIST = 0.1
LEIDEN_RESOLUTION = 1
RANDOM_STATE = 0

cluster_results = {}
for cluster_side in CLUSTER_SIDES:
    cluster_table, histogram_features, umap_xy = go.run_umap_leiden(
        good_tracks,
        side=cluster_side,
        feature_transform=FEATURE_TRANSFORM,
        neighbor_metric=NEIGHBOR_METRIC,
        n_neighbors=N_NEIGHBORS,
        umap_min_dist=UMAP_MIN_DIST,
        leiden_resolution=LEIDEN_RESOLUTION,
        random_state=RANDOM_STATE,
    )
    cluster_results[cluster_side] = {
        "cluster_table": cluster_table,
        "histogram_features": histogram_features,
        "umap_xy": umap_xy,
    }
    print(f"{cluster_side}: clustered {len(cluster_table)} tracks")
    display(cluster_table.groupby("leiden_cluster")["track_name"].count().rename("n_tracks"))
    display(cluster_table.head())


# %%
# Save the stable hand-off from spatial clustering to downstream analyses.
CLUSTER_ID_TABLE_PATH = GRID_ROOT / "track_cluster_ids.csv"

cluster_id_table = pd.concat(
    [
        result["cluster_table"][["track_id", "track_name", "side", "leiden_cluster"]].assign(
            cluster_id=lambda df, cluster_side=cluster_side: (
                cluster_side + "_" + df["leiden_cluster"].astype(str)
            )
        )
        for cluster_side, result in cluster_results.items()
    ],
    ignore_index=True,
).rename(columns={"track_id": "TrackID", "leiden_cluster": "leiden_cluster_id"})

cluster_id_table = cluster_id_table[
    ["TrackID", "track_name", "side", "cluster_id", "leiden_cluster_id"]
].sort_values(
    ["side", "TrackID", "track_name"]
)
CLUSTER_ID_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
cluster_id_table.to_csv(CLUSTER_ID_TABLE_PATH, index=False)

print(f"Saved {len(cluster_id_table)} cluster assignments to {CLUSTER_ID_TABLE_PATH}")
display(cluster_id_table.head(20))


# %%
# Inspect cluster separation in UMAP space.
for cluster_side, result in cluster_results.items():
    go.plot_umap_clusters(
        result["cluster_table"],
        color_col="leiden_cluster",
        title=f"{cluster_side} colony grid occupancy UMAP",
    )


# %%
# Interpret each job cluster from its mean spatial occupancy.
CLUSTER_MEAN_MODE = "sqrt"
CLUSTER_MEAN_VMAX_PERCENTILE = 99.0

cluster_mean_histograms = {}
for cluster_side, result in cluster_results.items():
    cluster_mean_histograms[cluster_side] = go.plot_cluster_mean_histograms(
        good_tracks,
        result["cluster_table"],
        mode=CLUSTER_MEAN_MODE,
        vmax_percentile=CLUSTER_MEAN_VMAX_PERCENTILE,
        title=f"{cluster_side} colony cluster mean occupancy",
    )


# %%
# Check that individual ants resemble their job cluster's mean occupancy.
N_EXAMPLES_PER_CLUSTER = 6
CLUSTER_EXAMPLE_MODE = "sqrt"
CLUSTER_EXAMPLE_VMAX_PERCENTILE = 99.0
CLUSTER_EXAMPLE_RANDOM_STATE = 0

cluster_example_tracks = {}
for cluster_side, result in cluster_results.items():
    cluster_example_tracks[cluster_side] = go.plot_cluster_example_histograms(
        good_tracks,
        result["cluster_table"],
        n_examples=N_EXAMPLES_PER_CLUSTER,
        mode=CLUSTER_EXAMPLE_MODE,
        vmax_percentile=CLUSTER_EXAMPLE_VMAX_PERCENTILE,
        random_state=CLUSTER_EXAMPLE_RANDOM_STATE,
        title=f"{cluster_side} colony example occupancy histograms",
    )


# %%
# Compare speed through the recording by spatial job cluster. Absolute start
# time and light/dark shading retain the time-of-day context.
CLUSTER_SPEED_BIN_SECONDS = 10 * 60.0
CLUSTER_SPEED_SMOOTH_SECONDS = 10 * 60.0
CLUSTER_SPEED_YLIM = None

cluster_speed_timeseries = {}
cluster_speed_track_bins = {}
for cluster_side, result in cluster_results.items():
    speed_df, track_speed_df = go.plot_cluster_speed_timeseries(
        result["cluster_table"],
        SPEED_ROOT,
        bin_seconds=CLUSTER_SPEED_BIN_SECONDS,
        smooth_seconds=CLUSTER_SPEED_SMOOTH_SECONDS,
        start_clock_seconds=experiment_start_clock_seconds,
        light_off_hour=LIGHT_OFF_HOUR,
        light_on_hour=LIGHT_ON_HOUR,
        ylim=CLUSTER_SPEED_YLIM,
        title=f"{cluster_side} colony speed by occupancy cluster",
    )
    cluster_speed_timeseries[cluster_side] = speed_df
    cluster_speed_track_bins[cluster_side] = track_speed_df
    display(speed_df.head())


# %%
# Use the panorama annotations to test whether the spatial clusters separate
# colony-restricted ants from ants that move in and out. This first pass uses
# normalized occupancy histograms; the later cells add temporally distinct
# visits from exact tracking frames.
PANORAMA_REGIONS_PATH = DATASET_ROOT / "panorama_regions.csv"
COLONY_RESTRICTED_MEDIAN_THRESHOLD = 0.90
REGION_ANALYSIS_OUTPUT_ROOT = GRID_ROOT / "panorama_region_analysis"
SAVE_REGION_TABLES = True

# The right grid begins at the global left/right split. Passing this split is
# essential for unsuffixed annotations such as the current ``water`` labels.
GRID_X_SPLIT_PX = float(track_table.loc[track_table["side"] == "right", "side_x0_px"].median())
panorama_regions = go.load_panorama_regions(
    PANORAMA_REGIONS_PATH,
    x_split_px=GRID_X_SPLIT_PX,
)
clustered_tracks = pd.concat(
    [
        result["cluster_table"].assign(
            cluster_id=cluster_side + "_" + result["cluster_table"]["leiden_cluster"].astype(str)
        )
        for cluster_side, result in cluster_results.items()
    ],
    ignore_index=True,
)
region_occupancy = go.compute_region_occupancy(clustered_tracks, panorama_regions)
colony_use_by_ant = go.summarize_colony_use(region_occupancy)
cluster_colony_use = go.summarize_cluster_colony_use(
    colony_use_by_ant,
    restricted_threshold=COLONY_RESTRICTED_MEDIAN_THRESHOLD,
)

if SAVE_REGION_TABLES:
    REGION_ANALYSIS_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    region_occupancy.to_csv(REGION_ANALYSIS_OUTPUT_ROOT / "ant_region_occupancy.csv", index=False)
    colony_use_by_ant.to_csv(REGION_ANALYSIS_OUTPUT_ROOT / "ant_colony_use.csv", index=False)
    cluster_colony_use.to_csv(REGION_ANALYSIS_OUTPUT_ROOT / "cluster_colony_use.csv", index=False)
    print(f"Saved panorama-region tables to {REGION_ANALYSIS_OUTPUT_ROOT}")

display(
    panorama_regions[
        ["region_id", "side", "side_source", "region_type", "name", "shape", "area_mm2"]
    ]
)
display(cluster_colony_use)


# Select candidate roaming ants from the main spatial clusters. Set explicit
# cluster IDs here only when automatic selection needs a manual override.
REGION_DETAIL_CLUSTER_IDS = None  # Example: ("left_1", "right_1")

if REGION_DETAIL_CLUSTER_IDS is None:
    putative_in_out_cluster_ids = go.select_putative_roaming_clusters(cluster_colony_use)
else:
    putative_in_out_cluster_ids = tuple(REGION_DETAIL_CLUSTER_IDS)

print(f"Putative in/out clusters: {putative_in_out_cluster_ids}")


# %%
# Temporal follow-up: extract exact food/water detections from the raw tracks.
# This is the slower step, so cache the compact frame table. It is recomputed
# automatically after panorama_regions.csv changes; set the override to True
# after changing extraction parameters.
PER_TRACK_ROOT = DATASET_ROOT / "stitched" / "per_track"
RESOURCE_PRESENCE_CACHE = REGION_ANALYSIS_OUTPUT_ROOT / "resource_presence_frames.parquet"
RECOMPUTE_RESOURCE_PRESENCE = False
RESOURCE_READ_WORKERS = 6
RESOURCE_BODYPOINT = 0
FPS = 24.0

cache_is_current = (
    RESOURCE_PRESENCE_CACHE.is_file()
    and RESOURCE_PRESENCE_CACHE.stat().st_mtime >= PANORAMA_REGIONS_PATH.stat().st_mtime
)
if RECOMPUTE_RESOURCE_PRESENCE or not cache_is_current:
    resource_presence_frames = go.extract_resource_presence_frames(
        clustered_tracks,
        panorama_regions,
        PER_TRACK_ROOT,
        bodypoint=RESOURCE_BODYPOINT,
        max_workers=RESOURCE_READ_WORKERS,
    )
    RESOURCE_PRESENCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    resource_presence_frames.to_parquet(RESOURCE_PRESENCE_CACHE, index=False)
    print(f"Saved {len(resource_presence_frames):,} resource-frame detections to {RESOURCE_PRESENCE_CACHE}")
else:
    resource_presence_frames = pd.read_parquet(RESOURCE_PRESENCE_CACHE)
    print(f"Loaded {len(resource_presence_frames):,} cached resource-frame detections")

# Cluster membership can change without invalidating the raw region/frame
# cache, so always refresh it from this notebook run.
resource_presence_frames = resource_presence_frames.drop(
    columns=["leiden_cluster", "cluster_id"],
    errors="ignore",
).merge(
    clustered_tracks[["track_name", "leiden_cluster", "cluster_id"]],
    on="track_name",
    how="inner",
    validate="many_to_one",
)


# %%
# OPTIONAL TRIP PHENOTYPING START
#
# This experiment is intentionally isolated at the end of the workflow. To
# remove it, delete from OPTIONAL TRIP PHENOTYPING START through END and delete
# analysis/trip_phenotyping_utils.py; no earlier analysis depends on it.
RUN_OPTIONAL_TRIP_PHENOTYPING = True

if RUN_OPTIONAL_TRIP_PHENOTYPING:
    import analysis.trip_phenotyping_utils as trip_go

    importlib.reload(trip_go)

    # Candidate foragers are all ants from the original putative in/out spatial
    # clusters, including ants that never touched an annotated food/water area.
    trip_candidate_tracks = clustered_tracks[
        clustered_tracks["cluster_id"].isin(putative_in_out_cluster_ids)
    ].copy()

    TRIP_OUTPUT_ROOT = REGION_ANALYSIS_OUTPUT_ROOT / "optional_trip_phenotyping"
    RECOMPUTE_TRIPS = False
    TRIP_READ_WORKERS = 4
    TRIP_POSITION_BIN_SECONDS = 1.0
    TRIP_MAX_STATE_GAP_SECONDS = 30.0
    TRIP_MIN_STATE_RUN_SECONDS = 5.0
    TRIP_MIN_COLONY_ANCHOR_SECONDS = 5.0
    TRIP_MIN_DURATION_SECONDS = 30.0
    TRIP_MIN_OBSERVED_COVERAGE = 0.20
    TRIP_MAX_PATH_GAP_SECONDS = 3.0

    completed_trips, completed_trip_positions, trip_extraction_diagnostics = (
        trip_go.load_or_extract_completed_trips(
            trip_candidate_tracks,
            panorama_regions,
            PER_TRACK_ROOT,
            TRIP_OUTPUT_ROOT,
            fps=FPS,
            start_clock_seconds=experiment_start_clock_seconds,
            bodypoint=RESOURCE_BODYPOINT,
            position_bin_seconds=TRIP_POSITION_BIN_SECONDS,
            max_state_gap_seconds=TRIP_MAX_STATE_GAP_SECONDS,
            min_state_run_seconds=TRIP_MIN_STATE_RUN_SECONDS,
            min_colony_anchor_seconds=TRIP_MIN_COLONY_ANCHOR_SECONDS,
            min_trip_seconds=TRIP_MIN_DURATION_SECONDS,
            min_trip_coverage=TRIP_MIN_OBSERVED_COVERAGE,
            max_path_gap_seconds=TRIP_MAX_PATH_GAP_SECONDS,
            max_workers=TRIP_READ_WORKERS,
            recompute=RECOMPUTE_TRIPS,
        )
    )

    # The focused analysis treats foraging effort as continuous. Require a few
    # completed trips so per-ant duration estimates and clock-time profiles are
    # interpretable, but do not impose another clustering layer.
    TRIP_MIN_TRIPS_FOR_SUMMARY = 3
    trip_candidate_summary = trip_go.summarize_trip_candidates(
        trip_candidate_tracks,
        completed_trips,
        completed_trip_positions,
        fps=FPS,
    )
    trip_summary = trip_candidate_summary[
        trip_candidate_summary["n_completed_trips"] >= TRIP_MIN_TRIPS_FOR_SUMMARY
    ].copy()
    TRIP_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    completed_trips.to_csv(TRIP_OUTPUT_ROOT / "completed_trips.csv", index=False)
    trip_extraction_diagnostics.to_csv(
        TRIP_OUTPUT_ROOT / "trip_extraction_diagnostics.csv", index=False
    )
    trip_candidate_summary.to_csv(
        TRIP_OUTPUT_ROOT / "trip_candidate_summary.csv", index=False
    )
    trip_summary.to_csv(TRIP_OUTPUT_ROOT / "trip_summary.csv", index=False)

    # Final focused output: per-ant trip investment with two-dimensional
    # uncertainty, followed by separate trip-time and resource-time heatmaps.
    # Every heatmap row is normalized within ant so it sums to 100%.
    FORAGING_SUMMARY_ROOT = TRIP_OUTPUT_ROOT / "foraging_summary"
    TRIP_INVESTMENT_BOOTSTRAPS = 2_000
    TIME_OF_DAY_BIN_MINUTES = 30.0
    trip_investment_confidence = trip_go.compute_trip_investment_confidence(
        trip_summary,
        completed_trips,
        fps=FPS,
        n_bootstrap=TRIP_INVESTMENT_BOOTSTRAPS,
        random_state=RANDOM_STATE,
    )
    trip_time_of_day_percent, resource_time_of_day_percent = trip_go.compute_ant_time_of_day_percent(
        trip_summary,
        completed_trip_positions,
        resource_presence_frames,
        fps=FPS,
        start_clock_seconds=experiment_start_clock_seconds,
        bin_minutes=TIME_OF_DAY_BIN_MINUTES,
    )
    FORAGING_SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    trip_investment_confidence.to_csv(
        FORAGING_SUMMARY_ROOT / "trip_investment_confidence.csv", index=False
    )
    trip_time_of_day_percent.to_csv(
        FORAGING_SUMMARY_ROOT / "trip_time_of_day_percent.csv", index=False
    )
    resource_time_of_day_percent.to_csv(
        FORAGING_SUMMARY_ROOT / "resource_time_of_day_percent.csv", index=False
    )
    install_auto_savefig(
        FORAGING_SUMMARY_ROOT / "figures",
        prefix="foraging_summary",
        dpi=FIGURE_DPI,
        enabled=SAVE_FIGURES,
    )
    trip_go.plot_trip_investment_confidence(trip_investment_confidence)
    trip_go.plot_ant_time_of_day_heatmap(
        trip_time_of_day_percent,
        source="completed_trip",
        light_off_hour=LIGHT_OFF_HOUR,
        light_on_hour=LIGHT_ON_HOUR,
    )
    trip_go.plot_ant_time_of_day_heatmap(
        resource_time_of_day_percent,
        source="resource",
        light_off_hour=LIGHT_OFF_HOUR,
        light_on_hour=LIGHT_ON_HOUR,
    )
    go.plot_ant_inside_outside_colony_distribution(colony_use_by_ant)
    colony_use_trip_correlations = go.plot_colony_use_vs_trip_investment(
        colony_use_by_ant,
        trip_investment_confidence,
    )
    colony_use_trip_correlations.to_csv(
        FORAGING_SUMMARY_ROOT / "colony_use_trip_correlations.csv",
        index=False,
    )

    print(f"Saved optional trip analysis to {TRIP_OUTPUT_ROOT}")
    print(
        f"Candidates={len(trip_candidate_tracks)}, completed trips={len(completed_trips):,}, "
        f"ants with completed trips={completed_trips['track_name'].nunique()}, "
        f"ants with >= {TRIP_MIN_TRIPS_FOR_SUMMARY} trips={len(trip_summary)}"
    )
    display(trip_extraction_diagnostics)
    display(trip_summary)
    display(trip_investment_confidence)
    display(trip_time_of_day_percent)
    display(resource_time_of_day_percent)
    display(colony_use_trip_correlations)

# OPTIONAL TRIP PHENOTYPING END
