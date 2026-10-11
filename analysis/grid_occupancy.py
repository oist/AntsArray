# %%
"""Prepare spatial reference labels and QC plots for eigenposture_interactive.

This is preprocessing for the post hoc spatial comparison. Behavioral states,
ant classes, sleep and trip analyses live in eigenposture_interactive.py.
Supports single blocks and previously combined recordings.
"""
try:
    get_ipython().run_line_magic("matplotlib", "qt")  # type: ignore[name-defined]
except Exception:
    pass

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Cluster spatial occupancy for a block or combined recording.")
parser.add_argument("dataset", nargs="?", type=Path, help="Block, date folder with --continuous, or exact combined folder")
parser.add_argument("--continuous", action="store_true", default=os.environ.get("ANTS_CONTINUOUS") == "1",
                    help="Select continous_stitched/continuous_stitched under the data folder")
parser.add_argument("--occupancy-only", action="store_true", default=os.environ.get("ANTS_OCCUPANCY_ONLY") == "1",
                    help="Compatibility flag: this script now only prepares spatial reference labels and QC figures")
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
import analysis.arena_grid_utils as arena_go
from analysis.figure_saving import install_auto_savefig

importlib.reload(go)


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
print(f"Selected panorama annotations: {PANORAMA_REGIONS_PATH}", flush=True)
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
# Spatial-reference QC figures use this folder; downstream figures use the report folder.
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
recording_frame_count = recording_stop_frame - recording_start_frame
combination_manifest_path = STITCHED_ROOT / "block_combination.json"
if combination_manifest_path.is_file():
    combination_manifest = json.loads(combination_manifest_path.read_text())
    recording_frame_count = sum(block["num_frames"] for block in combination_manifest["blocks"])
track_table = go.use_recording_detection_fraction(track_table, recording_frame_count)
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
print(f"Spatial reference complete: {CLUSTER_ID_TABLE_PATH}; QC figures: {FIGURE_ROOT}")
if args.headless:
    plt.close("all")
