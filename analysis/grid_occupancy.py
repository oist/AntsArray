# %%
# Canonical VS Code/Jupyter workflow for ant job analysis.
#
# This script intentionally covers only the analyses that are currently ready
# for routine use:
# 1. cluster ants by their spatial occupancy; and
# 2. compare speed and cached sleep through time across those clusters; and
# 3. compare post-return activity and sleeping-recipient responses; and
# 4. summarize trip investment and clock-time use by putative roaming ants.
try:
    get_ipython().run_line_magic("matplotlib", "qt")  # type: ignore[name-defined]
except Exception:
    pass

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Cluster spatial occupancy for a block or combined recording.")
parser.add_argument("dataset", nargs="?", type=Path, help="Block, date folder with --continuous, or exact combined folder")
parser.add_argument("--continuous", action="store_true", default=os.environ.get("ANTS_CONTINUOUS") == "1",
                    help="Select continous_stitched/continuous_stitched under the data folder")
parser.add_argument("--occupancy-only", action="store_true", default=os.environ.get("ANTS_OCCUPANCY_ONLY") == "1",
                    help="Save spatial clustering figures and stop before the speed/sleep/trip sections")
parser.add_argument("--headless", action="store_true", help="Use the noninteractive Agg backend")
parser.add_argument("--figure-root", type=Path, default=os.environ.get("ANTS_FIGURE_ROOT"), help="Figure output folder")
parser.add_argument("--grid-workers", type=int, default=2, help="Parallel arena-cache builders (default: 2)")
parser.add_argument("--grid-size-mm", type=float, default=os.environ.get("ANTS_GRID_SIZE_MM"),
                    help="Single-block arena bin width in mm; otherwise read grid_occupancy_settings.json or use the 0.25 mm default")
# Notebook kernels have their own command-line flags; retain the editable cell workflow.
args = parser.parse_args() if __name__ == "__main__" and "get_ipython" not in globals() else parser.parse_args([])
if args.headless:
    import matplotlib
    matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except Exception:
    display = print

repo_root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
for candidate in [repo_root, *repo_root.parents]:
    if (candidate / "analysis" / "grid_occupancy_utils.py").exists():
        repo_root = candidate
        break
else:
    raise FileNotFoundError("Could not find analysis/grid_occupancy_utils.py from the current working directory")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import analysis.grid_occupancy_utils as go
import analysis.sleep_motion_analysis_utils as sma
import analysis.return_sleep_utils as rs
import analysis.arena_grid_utils as arena_go
from analysis.figure_saving import install_auto_savefig

importlib.reload(go)
importlib.reload(sma)
importlib.reload(rs)


# %%
# Accept a block, a direct combined recording, or a date folder with --continuous.
DATASET_ROOT = go.resolve_analysis_dataset(
    args.dataset or Path(os.environ.get("ANTS_DATASET_ROOT", "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260810/block02-w000-031")),
    continuous=args.continuous,
)
STITCHED_ROOT = go.resolve_stitched_root(DATASET_ROOT)
ANNOTATION_ROOT = DATASET_ROOT.parent if DATASET_ROOT.name == "stitched" else DATASET_ROOT
GRID_OUTPUT_NAME = os.environ.get("ANTS_GRID_OUTPUT_NAME") or None  # None selects the pipeline cache.

SOURCE_GRID_ROOT = go.resolve_grid_root(DATASET_ROOT, GRID_OUTPUT_NAME)
# Combined recordings already contain compatible, combined occupancy grids.
# Reuse those arrays directly. Preserve the existing single-block arena workflow.
USE_ANNOTATED_ARENA_BOUNDS = not go.is_combined_recording(DATASET_ROOT)
PANORAMA_REGIONS_PATH = go.panorama_regions_path(ANNOTATION_ROOT)
if not USE_ANNOTATED_ARENA_BOUNDS:
    print(f"Reusing existing occupancy histograms: {SOURCE_GRID_ROOT} (no track rereading)", flush=True)
GRID_ROOT = (
    arena_go.prepare_arena_grid_cache(DATASET_ROOT, SOURCE_GRID_ROOT, PANORAMA_REGIONS_PATH,
                                     max_workers=args.grid_workers, grid_size_mm=args.grid_size_mm)
    if USE_ANNOTATED_ARENA_BOUNDS else SOURCE_GRID_ROOT
)
if not USE_ANNOTATED_ARENA_BOUNDS and args.grid_size_mm is not None:
    requested_size = arena_go.configured_grid_size_mm(DATASET_ROOT, args.grid_size_mm)
    if not go.load_grid_tracks(GRID_ROOT).grid_size_mm.eq(requested_size).all():
        raise ValueError("Combined recordings reuse their existing grids. Rebuild source-block grids at the requested spacing before combining them.")
SPEED_ROOT = STITCHED_ROOT / "speed_vectors"
MIN_PRESENT_FRAC = 0.40
# May 15's 10-hour right-ant-8 export repeats the full recording's pose rows
# (verified across all overlapping frames). Preserve its files but do not
# count it as another ant or let its different filename clock affect analysis.
EXCLUDED_TRACK_NAMES = {
    ("20260515", "block02"): {"TrackID_0008_all_142044_right.parquet"},
}.get((DATASET_ROOT.parent.name, DATASET_ROOT.name), set())
LIGHT_OFF_HOUR = 19.5
LIGHT_ON_HOUR = 5.5
SAVE_FIGURES = True
FIGURE_DPI = 180
# All workflow figures, including the later trip summaries, use this folder.
FIGURE_ROOT = args.figure_root or DATASET_ROOT / "analysis_outputs"
install_auto_savefig(
    FIGURE_ROOT,
    prefix="grid_occupancy",
    dpi=FIGURE_DPI,
    enabled=SAVE_FIGURES,
)


# %%
# Load grid histogram metadata and keep only well-observed ants.
track_table = go.load_grid_tracks(GRID_ROOT)
excluded_track_table = track_table[track_table["track_name"].isin(EXCLUDED_TRACK_NAMES)].copy()
if not excluded_track_table.empty:
    print("Excluded known duplicate exports (source files preserved):")
    display(excluded_track_table[["side", "track_id", "track_name", "frame_min", "frame_max"]])
track_table = track_table[~track_table["track_name"].isin(EXCLUDED_TRACK_NAMES)].copy()
if track_table.duplicated(["side", "track_id"]).any():
    raise ValueError("Duplicate ant identities remain; resolve track versions before clustering")
track_table = go.attach_detection_fraction(track_table, SPEED_ROOT)
RECORDING_CONTEXT = go.recording_context(DATASET_ROOT, track_table)
RECORDING_LABEL = (f"{RECORDING_CONTEXT['recording_date']}: {' + '.join(RECORDING_CONTEXT['blocks'])}"
                   if RECORDING_CONTEXT["blocks"] else f"{DATASET_ROOT.parent.name}/{DATASET_ROOT.name}")
experiment_start_clock_seconds = RECORDING_CONTEXT["start_clock_seconds"]
recording_start_frame = RECORDING_CONTEXT["frame_min"]
recording_stop_frame = RECORDING_CONTEXT["frame_stop"]
good_tracks = go.select_good_tracks(track_table, MIN_PRESENT_FRAC, side="both")

print(f"Loaded {len(track_table)} grid histograms from {GRID_ROOT}")
if RECORDING_CONTEXT["blocks"]:
    print(f"Combined blocks: {', '.join(RECORDING_CONTEXT['blocks'])}")
print(f"Detection metadata: {SPEED_ROOT}")
print(f"Selected {len(good_tracks)} tracks with present_frac > {MIN_PRESENT_FRAC}")
print(f"Experiment start clock: {go.format_clock_time(experiment_start_clock_seconds)}")
print(f"Recording frame window: [{recording_start_frame}, {recording_stop_frame})")
print(f"Occupancy grid spacing (mm): {sorted(track_table['grid_size_mm'].unique())}")
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
        title=f"{RECORDING_LABEL} — {cluster_side} colony cluster mean occupancy",
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
if args.occupancy_only:
    print(f"Spatial occupancy analysis complete. Figures: {FIGURE_ROOT}")
    plt.close("all")
    raise SystemExit(0)

# Compare speed through the recording by spatial job cluster. The x axis uses
# recording clock time (HH:MM), with light/dark shading across midnight.
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
# Return/sleep figures 1 and 2: cached body/antenna sleep labels, averaged
# equally over ants within each spatial cluster. Unknown frames are excluded;
# the coverage panel shows
# how many ants contribute to each time bin.
SLEEP_LABEL_ROOT = STITCHED_ROOT / "sleep_motion_labels"
CLUSTER_SLEEP_BIN_SECONDS = 10 * 60.0
CLUSTER_SLEEP_MIN_CLASSIFIED_FRACTION = 0.50

sleep_label_tracks = None
try:
    sleep_label_tracks = sma.load_sleep_label_tracks(SLEEP_LABEL_ROOT, cluster_id_table)
except FileNotFoundError as error:
    print(f"MISSING INPUT for sleep figures 1 and 2 ({SLEEP_LABEL_ROOT}):\n{error}")
    print("Generate the missing caches with analysis/compute_sleep_motion_labels.py, then rerun this cell.")

if sleep_label_tracks is not None:
    cluster_sleep_summary, cluster_sleep_ant_bins = sma.cluster_sleep_timeseries(
        sleep_label_tracks,
        bin_seconds=CLUSTER_SLEEP_BIN_SECONDS,
        min_classified_fraction=CLUSTER_SLEEP_MIN_CLASSIFIED_FRACTION,
        recording_start_frame=recording_start_frame,
        recording_stop_frame=recording_stop_frame,
    )
    sleep_output_root = GRID_ROOT / "sleep_motion_analysis"
    sleep_output_root.mkdir(parents=True, exist_ok=True)
    cluster_sleep_summary.to_csv(sleep_output_root / "cluster_sleep_timeseries.csv", index=False)
    cluster_sleep_ant_bins.to_parquet(sleep_output_root / "ant_sleep_time_bins.parquet", index=False)
    cluster_sleep_figures = {}
    ant_sleep_figures = {}
    for cluster_side in CLUSTER_SIDES:
        cluster_sleep_figures[cluster_side] = sma.plot_cluster_sleep_timeseries(
            cluster_sleep_summary[cluster_sleep_summary.side == cluster_side],
            start_clock_seconds=experiment_start_clock_seconds,
            light_off_hour=LIGHT_OFF_HOUR, light_on_hour=LIGHT_ON_HOUR,
        )
        cluster_sleep_figures[cluster_side].suptitle(f"{cluster_side.capitalize()} colony — sleep by occupancy cluster")
        ant_sleep_figures[cluster_side] = sma.plot_ant_sleep_heatmap(
            cluster_sleep_ant_bins[cluster_sleep_ant_bins.side == cluster_side],
            start_clock_seconds=experiment_start_clock_seconds,
        )
        ant_sleep_figures[cluster_side].suptitle(f"{cluster_side.capitalize()} colony — individual sleep through time")
    plt.show()
    display(cluster_sleep_summary.head())


# %%
# Individual activity/sleep versus time of day: each light–dark cycle beside
# its observed clock profile. This uses ALL spatially clustered ants,
# not just candidate foragers, with a fixed cluster/ID row order throughout.
# It describes time-of-day organization, not proof of an endogenous rhythm
# or a colony-versus-isolation difference (no isolation data are loaded here).
importlib.reload(sma)
clock_speed_tracks = go.load_speed_tracks(SPEED_ROOT)
clock_fps = float(clock_speed_tracks["fps"].median())
clock_recording_start_frame = recording_start_frame
clock_recording_stop_frame = recording_stop_frame
clock_cycle_offset_seconds = LIGHT_ON_HOUR * 3600 - experiment_start_clock_seconds
clock_n_complete_cycles = max(
    0,
    math.floor((clock_recording_stop_frame / clock_fps - clock_cycle_offset_seconds) / 86400)
    - math.ceil((clock_recording_start_frame / clock_fps - clock_cycle_offset_seconds) / 86400),
)
CLOCK_MATRIX_BIN_MINUTES = 30.0
CLOCK_MATRIX_MIN_BIN_COVERAGE = 0.50
# Prefer complete lights-on-to-lights-on days, requiring two valid cycles per
# clock bin when available. Short recordings (e.g. July 23) remain explicitly
# partial observations rather than being presented as repeated-cycle estimates.
CLOCK_MATRIX_MIN_CYCLES = min(2, max(1, clock_n_complete_cycles))
CLOCK_MATRIX_INCLUDE_PARTIAL_CYCLES = clock_n_complete_cycles == 0
CLOCK_MATRIX_SPEED_VMAX = None  # Shared across colonies; None uses pooled 99th percentile.
CLOCK_MATRIX_WORKERS = 4
CLOCK_MATRIX_OUTPUT_ROOT = GRID_ROOT / "activity_sleep_clock"

print(f"Clock matrices: {clock_n_complete_cycles} complete light-dark cycles; "
      f"minimum {CLOCK_MATRIX_MIN_CYCLES} valid cycles per clock bin; "
      f"include partial cycles={CLOCK_MATRIX_INCLUDE_PARTIAL_CYCLES}")
clock_sleep_tracks = sleep_label_tracks
if clock_sleep_tracks is None:
    try:
        # If the earlier strict cluster loader found missing ants, use the
        # available labels here and explicitly mark absent ants as unknown.
        clock_sleep_tracks = sma.load_sleep_label_tracks(SLEEP_LABEL_ROOT)
    except FileNotFoundError as error:
        print(f"MISSING SLEEP INPUT for clock matrices: {error}")
        print("Activity can still be plotted; missing sleep is gray, never zero.")

activity_sleep_cycle_bins, activity_sleep_clock_profiles, activity_sleep_clock_audit = (
    sma.compute_activity_sleep_clock_profiles(
        cluster_id_table,
        clock_speed_tracks,
        clock_sleep_tracks,
        fps=clock_fps,
        recording_stop_frame=clock_recording_stop_frame,
        recording_start_frame=clock_recording_start_frame,
        start_clock_seconds=experiment_start_clock_seconds,
        light_on_hour=LIGHT_ON_HOUR,
        light_off_hour=LIGHT_OFF_HOUR,
        bin_minutes=CLOCK_MATRIX_BIN_MINUTES,
        min_bin_coverage=CLOCK_MATRIX_MIN_BIN_COVERAGE,
        min_cycles=CLOCK_MATRIX_MIN_CYCLES,
        include_partial_cycles=CLOCK_MATRIX_INCLUDE_PARTIAL_CYCLES,
        recording_date=RECORDING_CONTEXT["recording_date"],
        max_workers=CLOCK_MATRIX_WORKERS,
    )
)
CLOCK_MATRIX_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
activity_sleep_cycle_bins.to_parquet(CLOCK_MATRIX_OUTPUT_ROOT / "ant_cycle_time_bins.parquet", index=False)
activity_sleep_clock_profiles.to_csv(CLOCK_MATRIX_OUTPUT_ROOT / "ant_time_of_day_profiles.csv", index=False)
activity_sleep_clock_audit.to_csv(CLOCK_MATRIX_OUTPUT_ROOT / "input_coverage_audit.csv", index=False)
activity_sleep_clock_figures = sma.plot_activity_sleep_clock_matrices(
    activity_sleep_cycle_bins, activity_sleep_clock_profiles, speed_vmax=CLOCK_MATRIX_SPEED_VMAX,
)
plt.show()
display(activity_sleep_clock_audit.groupby(["side", "speed_status", "sleep_status"]).size().rename("n_ants"))
display(activity_sleep_clock_profiles.groupby("side")[["n_speed_cycles", "n_sleep_cycles"]].describe())


# %%
# Sleep/wake posture distribution, using the cached motion labels rather than
# the older exploratory posture classifier. Pool the current clustered ants by
# colony; each ant has equal density weight within sleep and wake separately.
RUN_SLEEP_POSTURE = True
POSTURE_FRAMES_PER_ANT_STATE = 1000
POSTURE_DENSITY_BINS = 160
POSTURE_EXTENT_MM = None
POSTURE_RANDOM_STATE = 0
RECOMPUTE_SLEEP_POSTURE = False
SLEEP_POSTURE_ROOT = GRID_ROOT / "sleep_motion_analysis" / "posture"

if not RUN_SLEEP_POSTURE:
    print("Sleep/wake posture plots disabled")
elif sleep_label_tracks is None:
    print(f"MISSING INPUT: sleep/wake posture plots require complete cached labels at {SLEEP_LABEL_ROOT}")
else:
    try:
        sleep_posture_points, sleep_posture_sampling = sma.load_sleep_posture_points(
            sleep_label_tracks, DATASET_ROOT, SOURCE_GRID_ROOT / "sleep_motion_analysis" / "posture" / "cache",
            max_frames_per_ant_state=POSTURE_FRAMES_PER_ANT_STATE,
            random_state=POSTURE_RANDOM_STATE, force=RECOMPUTE_SLEEP_POSTURE,
        )
    except FileNotFoundError as error:
        print(f"MISSING INPUT: sleep/wake posture plots were not generated:\n{error}")
    else:
        sleep_posture_figures, sleep_posture_summary, sleep_posture_medians = sma.plot_sleep_posture_distributions(
            sleep_posture_points, bins=POSTURE_DENSITY_BINS, extent_mm=POSTURE_EXTENT_MM,
        )
        SLEEP_POSTURE_ROOT.mkdir(parents=True, exist_ok=True)
        sleep_posture_points.to_parquet(SLEEP_POSTURE_ROOT / "aligned_posture_points.parquet", index=False)
        sleep_posture_sampling.to_csv(SLEEP_POSTURE_ROOT / "sampling_coverage.csv", index=False)
        sleep_posture_summary.to_csv(SLEEP_POSTURE_ROOT / "state_summary.csv", index=False)
        sleep_posture_medians.to_csv(SLEEP_POSTURE_ROOT / "median_bodypoints.csv", index=False)
        posture_settings = {
            "max_frames_per_ant_state": POSTURE_FRAMES_PER_ANT_STATE, "random_state": POSTURE_RANDOM_STATE,
            "bins": POSTURE_DENSITY_BINS, "extent_mm": POSTURE_EXTENT_MM,
            "weighting": "equal_ant_within_colony_state", "unknown_labels": "excluded",
            "alignment": "bodypoint 0 at origin; 0 -> 1 upward; calibrated mm, no size normalization",
            "sleep_classifier_parameters": json.loads(sleep_label_tracks.classifier_parameters.iloc[0]),
            "label_sources": [rs.fingerprint(Path(p)) for p in sleep_label_tracks.metadata_path],
            "clusters": sma.normalize_clusters(cluster_id_table).to_dict("records"),
        }
        (SLEEP_POSTURE_ROOT / "settings.json").write_text(json.dumps(posture_settings, indent=2) + "\n")
        plt.show()
        display(sleep_posture_summary)
        print(f"Sleep/wake posture tables and per-ant caches: {SLEEP_POSTURE_ROOT}")


# %%
# Use the panorama annotations to test whether the spatial clusters separate
# colony-restricted ants from ants that move in and out. This first pass uses
# normalized occupancy histograms; the later cells add temporally distinct
# visits from exact tracking frames.
PANORAMA_REGIONS_PATH = go.panorama_regions_path(ANNOTATION_ROOT)
print(f"Panorama regions: {PANORAMA_REGIONS_PATH}")
COLONY_RESTRICTED_MEDIAN_THRESHOLD = 0.90
REGION_ANALYSIS_OUTPUT_ROOT = GRID_ROOT / "panorama_region_analysis"
SAVE_REGION_TABLES = True

# Infer sides from arena geometry (colony regions when arenas are absent).
# Set an explicit tracking-pixel divider only for a different arena layout.
REGION_X_SPLIT_PX = None
panorama_regions = go.load_panorama_regions(
    PANORAMA_REGIONS_PATH,
    x_split_px=REGION_X_SPLIT_PX,
)
print("Region sides:")
display(panorama_regions[["name", "region_type", "side", "side_source", "side_split_x_px"]])
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
# Return/sleep figure 4: motion and new pair contacts around the first observed
# colony-entry crossing. Pool all excursion types and use the current clusters.
# The raw interaction rule is skeleton distance <=0.1 mm; the 2-second gap only
# merges frame-level hits into bouts so sustained contact has a single onset.
RUN_RETURN_SLEEP_ANALYSIS = True
RETURN_SLEEP_SETTINGS = rs.ReturnSettings()
RETURN_SLEEP_INTERACTION_ROOT = rs.resolve_interaction_root(DATASET_ROOT)
RETURN_SLEEP_CACHE_ROOT = STITCHED_ROOT / "analysis_cache" / "return_sleep"
RETURN_SLEEP_OUTPUT_ROOT = GRID_ROOT / "sleep_motion_analysis" / "return_response"
RECOMPUTE_RETURN_SLEEP = False

return_sleep_ready = False
return_sleep_missing = []
for input_name, input_path in (
    ("finished track chunks", DATASET_ROOT / "tracks"),
    ("finished per-ant tracks", STITCHED_ROOT / "per_track"),
    ("body/antenna motion cache", STITCHED_ROOT / "sleep_motion" / "per_track"),
):
    if not input_path.is_dir():
        return_sleep_missing.append(f"{input_name}: {input_path}")
for input_name, input_path in (
    ("completed interaction transfer", RETURN_SLEEP_INTERACTION_ROOT / "transfer_complete.ok"),
    ("interaction run manifest", RETURN_SLEEP_INTERACTION_ROOT / "run_manifest.json"),
    ("colony annotations", PANORAMA_REGIONS_PATH),
):
    if not input_path.is_file():
        return_sleep_missing.append(f"{input_name}: {input_path}")
if not any((SLEEP_LABEL_ROOT / "per_track").glob("*/sleep_motion_label_metadata.json")):
    return_sleep_missing.append(f"sleep labels: {SLEEP_LABEL_ROOT / 'per_track'}")

if not RUN_RETURN_SLEEP_ANALYSIS:
    print("Return/sleep response plots disabled")
elif return_sleep_missing:
    print("MISSING INPUTS: return/sleep figures 4 and 7 were not generated:")
    for missing_input in return_sleep_missing:
        print(f"  - {missing_input}")
    print("Generate missing motion/sleep caches or finish publishing the 0.1 mm skeleton-interaction fanout, then rerun these cells.")
else:
    RETURN_SLEEP_SETTINGS.validate()
    return_sleep_all_tracks = sma.load_sleep_label_tracks(SLEEP_LABEL_ROOT)
    return_sleep_tracks = sma.load_sleep_label_tracks(SLEEP_LABEL_ROOT, cluster_id_table)
    if not return_sleep_all_tracks.fps.eq(RETURN_SLEEP_SETTINGS.fps).all():
        raise ValueError("Return/sleep FPS differs from cached labels")
    if go.start_time_from_track_table(return_sleep_tracks) != experiment_start_clock_seconds:
        raise ValueError("Sleep-label and grid recording clocks differ")
    return_sleep_chunks, return_sleep_run = rs.load_published_interaction_chunks(
        RETURN_SLEEP_INTERACTION_ROOT, DATASET_ROOT / "tracks", fps=RETURN_SLEEP_SETTINGS.fps,
        start_clock_seconds=experiment_start_clock_seconds,
    )
    return_sleep_contexts = rs.load_contexts(
        return_sleep_all_tracks, DATASET_ROOT, panorama_regions, RETURN_SLEEP_SETTINGS,
        RETURN_SLEEP_CACHE_ROOT, force=RECOMPUTE_RETURN_SLEEP,
    )
    colony_returns = rs.extract_returns(return_sleep_contexts, cluster_id_table, RETURN_SLEEP_SETTINGS)
    return_contact_bouts, return_interaction_coverage = rs.load_contact_bouts(
        return_sleep_chunks, RETURN_SLEEP_SETTINGS, RETURN_SLEEP_CACHE_ROOT, force=RECOMPUTE_RETURN_SLEEP,
    )
    rs.attach_contact_context(return_sleep_contexts, return_contact_bouts, return_interaction_coverage, RETURN_SLEEP_SETTINGS)
    return_activity_rows, return_sleep_latency = rs.return_activity_curves(
        colony_returns, return_sleep_contexts, RETURN_SLEEP_SETTINGS,
    )
    # Former plot 16: one row of three response panels per colony.
    return_activity_figures, return_activity_summaries = {}, []
    for cluster_side in CLUSTER_SIDES:
        return_activity_figures[cluster_side], side_summary = rs.plot_return_curves(
            return_activity_rows, side=cluster_side, random_state=RETURN_SLEEP_SETTINGS.random_state,
        )
        return_activity_summaries.append(side_summary)
    return_activity_summary = pd.concat(return_activity_summaries, ignore_index=True)
    plt.show()

    RETURN_SLEEP_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return_sleep_manifest = {
        "settings": rs.asdict(RETURN_SLEEP_SETTINGS),
        "interaction_run_id": return_sleep_run["run_id"],
        "interaction_parameters": return_sleep_run["parameters"],
        "clusters": sma.normalize_clusters(cluster_id_table).to_dict("records"),
        "regions": rs.fingerprint(PANORAMA_REGIONS_PATH),
        "labels": [rs.fingerprint(Path(p)) for p in return_sleep_all_tracks.metadata_path],
        "sleep_classifier_parameters": json.loads(return_sleep_tracks.classifier_parameters.iloc[0]),
    }
    (RETURN_SLEEP_OUTPUT_ROOT / "settings.json").write_text(json.dumps(return_sleep_manifest, indent=2) + "\n")
    colony_returns.to_csv(RETURN_SLEEP_OUTPUT_ROOT / "returns.csv", index=False)
    return_activity_rows.to_parquet(RETURN_SLEEP_OUTPUT_ROOT / "return_activity_rows.parquet", index=False)
    return_activity_summary.to_csv(RETURN_SLEEP_OUTPUT_ROOT / "return_activity_summary.csv", index=False)
    print(f"Return-aligned activity: {len(colony_returns)} returns; {len(return_sleep_chunks)} interaction chunks")
    return_sleep_ready = True


# %%
# Return/sleep figure 7: recipients asleep for 10 s before a new contact with
# an ant that returned in the previous 5 min. Compare same-ant matched quiet
# no-contact times and other contacts. This descriptive response includes later
# contacts; the standalone probe retains the separate censored waking analysis.
RECIPIENT_RESPONSE_XLIM_SECONDS = (-30, 30)
if return_sleep_ready:
    sleeping_return_contacts = rs.eligible_sleeping_contacts(
        return_sleep_tracks, return_sleep_contexts, return_contact_bouts, colony_returns, RETURN_SLEEP_SETTINGS,
    )
    sleeping_recipient_triggers, sleeping_match_diagnostics = rs.match_sleeping_controls(
        sleeping_return_contacts, return_sleep_contexts, RETURN_SLEEP_SETTINGS, resource_only=False,
    )
    sleeping_recipient_rows = rs.trigger_state_curves(
        sleeping_recipient_triggers, return_sleep_tracks, return_sleep_contexts, RETURN_SLEEP_SETTINGS,
    )
    # Former plot 17: three response panels per colony, zoomed around contact.
    # Full follow-up and contributing-ant counts remain in the saved tables.
    sleeping_recipient_figures, sleeping_recipient_summaries = {}, []
    for cluster_side in CLUSTER_SIDES:
        sleeping_recipient_figures[cluster_side], side_summary = rs.plot_recipient_curves(
            sleeping_recipient_rows, side=cluster_side, xlim=RECIPIENT_RESPONSE_XLIM_SECONDS,
            random_state=RETURN_SLEEP_SETTINGS.random_state,
        )
        sleeping_recipient_summaries.append(side_summary)
    sleeping_recipient_summary = pd.concat(sleeping_recipient_summaries, ignore_index=True)
    plt.show()
    sleeping_recipient_triggers.to_csv(RETURN_SLEEP_OUTPUT_ROOT / "matched_triggers.csv", index=False)
    sleeping_match_diagnostics.to_csv(RETURN_SLEEP_OUTPUT_ROOT / "matching_diagnostics.csv", index=False)
    sleeping_recipient_rows.to_parquet(RETURN_SLEEP_OUTPUT_ROOT / "recipient_curve_rows.parquet", index=False)
    sleeping_recipient_summary.to_csv(RETURN_SLEEP_OUTPUT_ROOT / "recipient_curve_summary.csv", index=False)
    display(sleeping_recipient_triggers.groupby(["side", "condition"]).size().rename("n_matched_triggers"))
    print(f"Return/sleep response tables: {RETURN_SLEEP_OUTPUT_ROOT}")


# %%
# Temporal follow-up: extract exact food/water detections from the raw tracks.
# This is the slower step, so cache the compact frame table. It is recomputed
# automatically after panorama_regions.csv changes; set the override to True
# after changing extraction parameters.
PER_TRACK_ROOT = STITCHED_ROOT / "per_track"
RESOURCE_PRESENCE_CACHE = REGION_ANALYSIS_OUTPUT_ROOT / "resource_presence_frames.parquet"
RECOMPUTE_RESOURCE_PRESENCE = False
RESOURCE_READ_WORKERS = 6
RESOURCE_BODYPOINT = 0
FPS = RECORDING_CONTEXT.get("fps", 24.0)

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
        FIGURE_ROOT,
        prefix="foraging_summary",
        dpi=FIGURE_DPI,
        enabled=SAVE_FIGURES,
    )
    colony_use_trip_correlation_tables = []
    for cluster_side in CLUSTER_SIDES:
        trip_go.plot_trip_investment_confidence(trip_investment_confidence, side=cluster_side)
        trip_go.plot_ant_time_of_day_heatmap(
            trip_time_of_day_percent, side=cluster_side,
            source="completed_trip",
            light_off_hour=LIGHT_OFF_HOUR,
            light_on_hour=LIGHT_ON_HOUR,
        )
        trip_go.plot_ant_time_of_day_heatmap(
            resource_time_of_day_percent, side=cluster_side,
            source="resource",
            light_off_hour=LIGHT_OFF_HOUR,
            light_on_hour=LIGHT_ON_HOUR,
        )
        go.plot_ant_inside_outside_colony_distribution(colony_use_by_ant, side=cluster_side)
        colony_use_trip_correlation_tables.append(go.plot_colony_use_vs_trip_investment(
            colony_use_by_ant, trip_investment_confidence, side=cluster_side,
        ))
    colony_use_trip_correlations = pd.concat(colony_use_trip_correlation_tables, ignore_index=True)
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


# %%
# Do the same candidate foragers invest more outside time / trips every day?
# Use ALL candidate ants, retaining zero-trip days; do not select active ants
# separately each day. Days run from LIGHT_ON_HOUR to the next LIGHT_ON_HOUR
# (05:30–05:30 here), containing one full light phase and one full dark phase.
# Only complete cycles with sufficient coverage in both phases enter comparisons.
# Changing the light schedule automatically invalidates daily caches. The first
# run reads raw positions for daily exposure; per-ant caches make reruns quick.
if RUN_OPTIONAL_TRIP_PHENOTYPING:
    importlib.reload(trip_go)

    TRIP_REPEATABILITY_MIN_COVERAGE = 0.70
    TRIP_REPEATABILITY_BOOTSTRAPS = 2_000
    RECOMPUTE_DAILY_FORAGING = False

    daily_foraging_investment = trip_go.load_daily_foraging_investment(
        trip_candidate_tracks,
        completed_trips,
        panorama_regions,
        PER_TRACK_ROOT,
        FORAGING_SUMMARY_ROOT,
        fps=FPS,
        start_clock_seconds=experiment_start_clock_seconds,
        light_on_hour=LIGHT_ON_HOUR,
        light_off_hour=LIGHT_OFF_HOUR,
        min_coverage=TRIP_REPEATABILITY_MIN_COVERAGE,
        bodypoint=RESOURCE_BODYPOINT,
        max_workers=TRIP_READ_WORKERS,
        recompute=RECOMPUTE_DAILY_FORAGING,
        recording_date=RECORDING_CONTEXT["recording_date"],
        recording_start_frame=recording_start_frame,
        recording_stop_frame=recording_stop_frame,
    )
    daily_foraging_repeatability = trip_go.compute_daily_foraging_repeatability(
        daily_foraging_investment,
        n_bootstrap=TRIP_REPEATABILITY_BOOTSTRAPS,
        random_state=RANDOM_STATE,
    )
    daily_foraging_investment.to_csv(
        FORAGING_SUMMARY_ROOT / "daily_foraging_investment.csv", index=False
    )
    daily_foraging_repeatability.to_csv(
        FORAGING_SUMMARY_ROOT / "daily_foraging_repeatability.csv", index=False
    )
    for cluster_side in CLUSTER_SIDES:
        trip_go.plot_daily_foraging_investment(daily_foraging_investment, side=cluster_side)
        trip_go.plot_daily_foraging_repeatability(
            daily_foraging_investment, daily_foraging_repeatability, side=cluster_side,
        )
    display(daily_foraging_investment.groupby(["side", "day_label"])["eligible"].agg(["sum", "size"]))
    display(daily_foraging_repeatability)

# OPTIONAL TRIP PHENOTYPING END
