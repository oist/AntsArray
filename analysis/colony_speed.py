# %%
# VS Code/Jupyter interactive script for colony speed vectors.
try:
    get_ipython().run_line_magic("matplotlib", "qt")  # type: ignore[name-defined]
except Exception:
    pass

import importlib
import sys
from pathlib import Path

try:
    from IPython.display import display
except Exception:
    display = print

repo_root = Path.cwd().resolve()
for candidate in [repo_root, *repo_root.parents]:
    if (candidate / "analysis" / "colony_speed_utils.py").exists():
        repo_root = candidate
        break
else:
    raise FileNotFoundError("Could not find analysis/colony_speed_utils.py from the current working directory")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import analysis.colony_speed_utils as cs

importlib.reload(cs)


# %%
# Edit these settings first.
SPEED_ROOT = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260702/block01/stitched/speed_vectors"
)
PRESENCE_ROOT = cs.infer_presence_root(SPEED_ROOT)
SLEEP_ROOT = cs.infer_sleep_prediction_root(SPEED_ROOT)
CONDUCTOR_PATH = cs.infer_conductor_path(SPEED_ROOT)
MIN_PRESENT_FRAC = 0.40
BIN_SECONDS = 30.0

COLONY_SMOOTH_SECONDS =  10
INDIVIDUAL_SMOOTH_SECONDS = 10
IMAGE_SMOOTH_SECONDS = 0

PRE_PULSE_SECONDS = 120.0
POST_PULSE_SECONDS = 800.0
PULSE_RESPONSE_SIDE = "both"  # "left", "right", or "both"
PULSE_RESPONSE_SMOOTH_SECONDS = 0.0
PULSE_RESPONSE_YLIM = None
PULSE_RESPONSE_VMIN = 0.0
PULSE_RESPONSE_VMAX = None
PULSE_RESPONSE_VMAX_PERCENTILE = 99.0
PULSE_RESPONSE_CMAP = "viridis"
PULSE_RESPONSE_SORT_BY_STIMULUS_STRENGTH = True
PULSE_RESPONSE_STIMULUS_STRENGTH_COL = "auto"  # "auto" prefers duty, duty_pct, then sensor magnitudes.
PULSE_RESPONSE_STIMULUS_SORT_ASCENDING = True

SLEEP_SPLIT_PULSE_RESPONSE_SIDE = PULSE_RESPONSE_SIDE
SLEEP_SPLIT_PULSE_RESPONSE_SMOOTH_SECONDS = PULSE_RESPONSE_SMOOTH_SECONDS
SLEEP_SPLIT_PULSE_RESPONSE_YLIM = PULSE_RESPONSE_YLIM
SLEEP_SPLIT_PULSE_RESPONSE_VMIN = PULSE_RESPONSE_VMIN
SLEEP_SPLIT_PULSE_RESPONSE_VMAX = PULSE_RESPONSE_VMAX
SLEEP_SPLIT_PULSE_RESPONSE_VMAX_PERCENTILE = PULSE_RESPONSE_VMAX_PERCENTILE
SLEEP_SPLIT_PULSE_RESPONSE_CMAP = PULSE_RESPONSE_CMAP
SLEEP_SPLIT_PULSE_RESPONSE_SORT_BY_STIMULUS_STRENGTH = PULSE_RESPONSE_SORT_BY_STIMULUS_STRENGTH
SLEEP_SPLIT_PULSE_RESPONSE_STIMULUS_STRENGTH_COL = PULSE_RESPONSE_STIMULUS_STRENGTH_COL
SLEEP_SPLIT_PULSE_RESPONSE_STIMULUS_SORT_ASCENDING = PULSE_RESPONSE_STIMULUS_SORT_ASCENDING

LIGHT_OFF_HOUR = 18.0
LIGHT_ON_HOUR = 5.5

# %%
# Load speed-vector metadata and apply the metadata-based presence threshold.
track_table = cs.load_speed_tracks(SPEED_ROOT)
experiment_start_clock_seconds = cs.start_time_from_track_table(track_table)
tracks = cs.select_tracks(track_table, MIN_PRESENT_FRAC)
tracks = cs.attach_presence_to_tracks(tracks, PRESENCE_ROOT, require_all=False)
tracks = cs.attach_sleep_predictions_to_tracks(tracks, SLEEP_ROOT, require_all=False)

print(f"Loaded {len(track_table)} speed metadata files from {SPEED_ROOT}")
print("Presence filter: n_observed_frames / n_frames from each speed_metadata.json")
print(f"Selected {len(tracks)} tracks with present_frac > {MIN_PRESENT_FRAC}")
print(f"Loaded colony presence vectors from {PRESENCE_ROOT}")
print(f"Loaded sleep predictions from {SLEEP_ROOT}")
print(f"Conductor log: {CONDUCTOR_PATH}")
print(f"Experiment start clock: {cs.format_clock_time(experiment_start_clock_seconds)}")
display(tracks.groupby("side")["track_name"].count().rename("n_tracks"))
display(tracks.head())


# %%
# Compute colony-average speed time series.
colony_speed_timeseries, colony_speed_arrays = cs.compute_colony_speed_timeseries(
    tracks,
    bin_seconds=BIN_SECONDS,
    start_clock_seconds=experiment_start_clock_seconds,
)
display(colony_speed_timeseries.head())


# %%
# Plot smoothed colony-average speed.
COLONY_SPEED_YLIM = None

smoothed_colony_speed_timeseries = cs.plot_colony_speed(
    colony_speed_timeseries,
    start_clock_seconds=experiment_start_clock_seconds,
    smooth_seconds=COLONY_SMOOTH_SECONDS,
    bin_seconds=BIN_SECONDS,
    light_off_hour=LIGHT_OFF_HOUR,
    light_on_hour=LIGHT_ON_HOUR,
    ylim=COLONY_SPEED_YLIM,
)
display(smoothed_colony_speed_timeseries.head())


# %%
# Pulse-triggered colony speed around the last recording's CSV_PULSE camFrameStart values.
csv_pulses = cs.load_last_session_csv_pulses(CONDUCTOR_PATH)
display(csv_pulses.head())

pulse_speed_response = cs.plot_pulse_triggered_colony_speed(
    tracks,
    csv_pulses,
    pre_seconds=PRE_PULSE_SECONDS,
    post_seconds=POST_PULSE_SECONDS,
    side=PULSE_RESPONSE_SIDE,
    frame_col="camFrameStart",
    smooth_seconds=PULSE_RESPONSE_SMOOTH_SECONDS,
    vmin=PULSE_RESPONSE_VMIN,
    vmax=PULSE_RESPONSE_VMAX,
    vmax_percentile=PULSE_RESPONSE_VMAX_PERCENTILE,
    cmap=PULSE_RESPONSE_CMAP,
    ylim=PULSE_RESPONSE_YLIM,
    sort_by_stimulus_strength=PULSE_RESPONSE_SORT_BY_STIMULUS_STRENGTH,
    stimulus_strength_col=PULSE_RESPONSE_STIMULUS_STRENGTH_COL,
    stimulus_sort_ascending=PULSE_RESPONSE_STIMULUS_SORT_ASCENDING,
)
display(pulse_speed_response["average"].head())
display(
    pulse_speed_response["pulse_table"][
        [col for col in [
            "response_row",
            "trial",
            "stimulus_strength",
            "stimulus_strength_col",
            "duty",
            "dur_s",
            "camFrameStart",
            "camFrameEnd",
            "valid_fraction",
            "mean_speed_mm_s",
        ] if col in pulse_speed_response["pulse_table"].columns]
    ]
)


# %%
# Pulse-triggered speed split by sleep state at each CSV_PULSE camFrameStart.
sleep_split_pulse_speed_response = cs.plot_sleep_split_pulse_triggered_colony_speed(
    tracks,
    csv_pulses,
    pre_seconds=PRE_PULSE_SECONDS,
    post_seconds=POST_PULSE_SECONDS,
    side=SLEEP_SPLIT_PULSE_RESPONSE_SIDE,
    frame_col="camFrameStart",
    smooth_seconds=SLEEP_SPLIT_PULSE_RESPONSE_SMOOTH_SECONDS,
    vmin=SLEEP_SPLIT_PULSE_RESPONSE_VMIN,
    vmax=SLEEP_SPLIT_PULSE_RESPONSE_VMAX,
    vmax_percentile=SLEEP_SPLIT_PULSE_RESPONSE_VMAX_PERCENTILE,
    cmap=SLEEP_SPLIT_PULSE_RESPONSE_CMAP,
    ylim=SLEEP_SPLIT_PULSE_RESPONSE_YLIM,
    sort_by_stimulus_strength=SLEEP_SPLIT_PULSE_RESPONSE_SORT_BY_STIMULUS_STRENGTH,
    stimulus_strength_col=SLEEP_SPLIT_PULSE_RESPONSE_STIMULUS_STRENGTH_COL,
    stimulus_sort_ascending=SLEEP_SPLIT_PULSE_RESPONSE_STIMULUS_SORT_ASCENDING,
)
display(sleep_split_pulse_speed_response["average_by_state"]["sleeping"].head())
display(sleep_split_pulse_speed_response["average_by_state"]["not_sleeping"].head())
display(
    sleep_split_pulse_speed_response["pulse_table"][
        [col for col in [
            "response_row",
            "trial",
            "stimulus_strength",
            "stimulus_strength_col",
            "duty",
            "camFrameStart",
            "n_sleeping_tracks_at_pulse",
            "n_not_sleeping_tracks_at_pulse",
            "n_unknown_sleep_tracks_at_pulse",
            "sleeping_mean_speed_mm_s",
            "not_sleeping_mean_speed_mm_s",
        ] if col in sleep_split_pulse_speed_response["pulse_table"].columns]
    ]
)


# %%
# Inspect track rows. Use these row numbers in INDIVIDUAL_TRACK_ROWS below.
display(
    tracks[
        [
            "side",
            "track_id",
            "track_name",
            "present_frac",
            "inside_colony_frac_valid",
            "has_colony_presence",
            "has_sleep_predictions",
            "sleep_fraction_predicted_frames",
            "mean_sleep_probability",
            "n_observed_frames",
            "n_frames",
        ]
    ].head(40)
)


# %%
# Plot individual ant speeds, smoothed only.
INDIVIDUAL_SIDE = "left"       # "left", "right", or "both"
INDIVIDUAL_TRACK_ROWS = None   # Example: [0, 3, 10]
INDIVIDUAL_TRACK_IDS = None    # Example: [12, 18]
INDIVIDUAL_MAX_TRACKS = 6
INDIVIDUAL_YLIM = None

individual_speed_timeseries = cs.plot_individual_speeds(
    tracks,
    side=INDIVIDUAL_SIDE,
    row_numbers=INDIVIDUAL_TRACK_ROWS,
    track_ids=INDIVIDUAL_TRACK_IDS,
    max_tracks=INDIVIDUAL_MAX_TRACKS,
    bin_seconds=BIN_SECONDS,
    smooth_seconds=INDIVIDUAL_SMOOTH_SECONDS,
    ylim=INDIVIDUAL_YLIM,
)
display(individual_speed_timeseries.head())


# %%
# Plot speed and colony in/out for one ant.
SINGLE_ANT_ROW = 0        # Row from the track table above. Set to None to use SINGLE_ANT_TRACK_ID.
SINGLE_ANT_TRACK_ID = None
SINGLE_ANT_SIDE = "left"  # Used only when SINGLE_ANT_ROW is None.
SINGLE_ANT_SPEED_YLIM = None

single_ant_speed_presence = cs.plot_speed_and_presence_for_ant(
    tracks,
    row_number=SINGLE_ANT_ROW,
    track_id=SINGLE_ANT_TRACK_ID,
    side=SINGLE_ANT_SIDE,
    bin_seconds=BIN_SECONDS,
    speed_smooth_seconds=INDIVIDUAL_SMOOTH_SECONDS,
    speed_ylim=SINGLE_ANT_SPEED_YLIM,
)
display(single_ant_speed_presence.head())


# %%
# Plot all selected ant speeds as an image, smoothed only.
IMAGE_SIDE = "both"  # "left", "right", or "both"
IMAGE_VMIN = 0.0
IMAGE_VMAX = None
IMAGE_VMAX_PERCENTILE = 99.0
IMAGE_CMAP = "viridis"

speed_image_matrix, speed_image_tracks, speed_image_time_h = cs.plot_speed_image(
    tracks,
    side=IMAGE_SIDE,
    bin_seconds=BIN_SECONDS,
    smooth_seconds=IMAGE_SMOOTH_SECONDS,
    vmin=IMAGE_VMIN,
    vmax=IMAGE_VMAX,
    vmax_percentile=IMAGE_VMAX_PERCENTILE,
    cmap=IMAGE_CMAP,
)
display(speed_image_tracks[["track_row", "side", "track_id", "track_name"]].head())


# %%
# Plot all selected ant speeds ordered by fraction of valid frames spent in colony.
ORDERED_IMAGE_SIDE = "both"
ORDERED_IMAGE_COLONY_FRAC_ASCENDING = False
ORDERED_IMAGE_VMIN = 0.0
ORDERED_IMAGE_VMAX = None
ORDERED_IMAGE_VMAX_PERCENTILE = 99.0
ORDERED_IMAGE_CMAP = "viridis"

ordered_speed_image_matrix, ordered_speed_image_tracks, ordered_speed_image_time_h = cs.plot_speed_image(
    tracks,
    side=ORDERED_IMAGE_SIDE,
    bin_seconds=BIN_SECONDS,
    smooth_seconds=IMAGE_SMOOTH_SECONDS,
    order_by_colony_frac=True,
    colony_frac_ascending=ORDERED_IMAGE_COLONY_FRAC_ASCENDING,
    vmin=ORDERED_IMAGE_VMIN,
    vmax=ORDERED_IMAGE_VMAX,
    vmax_percentile=ORDERED_IMAGE_VMAX_PERCENTILE,
    cmap=ORDERED_IMAGE_CMAP,
)
display(
    ordered_speed_image_tracks[
        ["track_row", "side", "track_id", "track_name", "inside_colony_frac_valid"]
    ].head()
)

# %%
