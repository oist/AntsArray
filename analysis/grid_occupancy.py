# %%
# VS Code/Jupyter interactive script for grid-occupancy histograms.
try:
    get_ipython().run_line_magic("matplotlib", "qt")  # type: ignore[name-defined]
except Exception:
    pass

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

importlib.reload(go)


# %%
# Edit these settings first.
GRID_ROOT = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02/stitched/grid_occupancy_histograms"
)
SPEED_ROOT = go.infer_speed_root(GRID_ROOT)
MIN_PRESENT_FRAC = 0.40
LIGHT_OFF_HOUR = 19.5
LIGHT_ON_HOUR = 5.5


# %%
# Load grid histogram metadata and keep only good ants.
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
# Inspect rows. Use row numbers below for single-ant plotting.
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
# Plot one ant's 2D occupancy histogram.
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
# Run UMAP + Leiden clustering separately for left and right colonies.
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
# Save a simple TrackID-to-cluster table.
CLUSTER_ID_TABLE_PATH = GRID_ROOT / "track_cluster_ids.csv"

cluster_id_table = pd.concat(
    [
        result["cluster_table"][["track_id", "side", "leiden_cluster"]].assign(
            cluster_id=lambda df, cluster_side=cluster_side: (
                cluster_side + "_" + df["leiden_cluster"].astype(str)
            )
        )
        for cluster_side, result in cluster_results.items()
    ],
    ignore_index=True,
).rename(columns={"track_id": "TrackID", "leiden_cluster": "leiden_cluster_id"})

cluster_id_table = cluster_id_table[["TrackID", "side", "cluster_id", "leiden_cluster_id"]].sort_values(
    ["side", "TrackID"]
)
CLUSTER_ID_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
cluster_id_table.to_csv(CLUSTER_ID_TABLE_PATH, index=False)

print(f"Saved {len(cluster_id_table)} cluster assignments to {CLUSTER_ID_TABLE_PATH}")
display(cluster_id_table.head(20))


# %%
# Plot clustered responses in UMAP space, separately by colony side.
for cluster_side, result in cluster_results.items():
    go.plot_umap_clusters(
        result["cluster_table"],
        color_col="leiden_cluster",
        title=f"{cluster_side} colony grid occupancy UMAP",
    )


# %%
# Plot mean 2D occupancy histogram for each Leiden cluster.
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
# Plot example ant histograms from each Leiden cluster.
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
# Plot mean speed over time for each Leiden cluster.
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
# Plot quiet periods for all clustered ants, sorted by occupancy cluster.
QUIET_SPEED_THRESHOLD_MM_S = 0.1
QUIET_BIN_SECONDS = 60.0
QUIET_CMAP = "Greys"

quiet_cluster_table = pd.concat(
    [
        result["cluster_table"].assign(
            cluster_id=lambda df, cluster_side=cluster_side: (
                cluster_side + "_" + df["leiden_cluster"].astype(str)
            )
        )
        for cluster_side, result in cluster_results.items()
    ],
    ignore_index=True,
)

quiet_image, quiet_image_tracks, quiet_time_h = go.plot_quiet_period_image(
    quiet_cluster_table,
    SPEED_ROOT,
    speed_threshold_mm_s=QUIET_SPEED_THRESHOLD_MM_S,
    bin_seconds=QUIET_BIN_SECONDS,
    start_clock_seconds=experiment_start_clock_seconds,
    light_off_hour=LIGHT_OFF_HOUR,
    light_on_hour=LIGHT_ON_HOUR,
    cmap=QUIET_CMAP,
    title=f"Quiet periods by occupancy cluster, speed <= {QUIET_SPEED_THRESHOLD_MM_S:g} mm/s",
)
display(
    quiet_image_tracks[
        [
            "image_row",
            "side",
            "track_id",
            "cluster_id",
            "leiden_cluster",
            "quiet_frac_valid",
            "n_valid_speed_frames",
        ]
    ].head(30)
)


# %%
# Plot a single ant's speed over time.
SINGLE_SPEED_TRACK_ID = 58
SINGLE_SPEED_SIDE = "right"
SINGLE_SPEED_BIN_SECONDS = 30
SINGLE_SPEED_SMOOTH_SECONDS = 300
SINGLE_SPEED_YLIM = None

speed_tracks = go.load_speed_tracks(SPEED_ROOT)
single_speed_matches = speed_tracks[
    (speed_tracks["track_id"] == SINGLE_SPEED_TRACK_ID)
    & (speed_tracks["side"] == SINGLE_SPEED_SIDE)
]
if single_speed_matches.empty:
    raise ValueError(f"No speed vector found for TrackID {SINGLE_SPEED_TRACK_ID} on {SINGLE_SPEED_SIDE}")

single_speed_row = single_speed_matches.iloc[0]
single_speed_timeseries = go.binned_track_speed(single_speed_row, SINGLE_SPEED_BIN_SECONDS)
single_speed_window_bins = max(1, int(round(SINGLE_SPEED_SMOOTH_SECONDS / SINGLE_SPEED_BIN_SECONDS)))
single_speed_timeseries["smoothed_speed_mm_s"] = go.rolling_nanmean(
    single_speed_timeseries["speed_mm_s"].to_numpy(),
    single_speed_window_bins,
)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(
    single_speed_timeseries["time_h"],
    single_speed_timeseries["smoothed_speed_mm_s"],
    lw=1.3,
)
ax.set_xlabel("Elapsed time (h)")
ax.set_ylabel("Speed (mm/s)")
ax.set_title(
    f"TrackID {SINGLE_SPEED_TRACK_ID} {SINGLE_SPEED_SIDE} speed, "
    f"{SINGLE_SPEED_SMOOTH_SECONDS / 60:g} min smoothing"
)
if SINGLE_SPEED_YLIM is not None:
    ax.set_ylim(*SINGLE_SPEED_YLIM)
ax.grid(True, alpha=0.25)
plt.show()

display(single_speed_row)
display(single_speed_timeseries.head())

# %%
