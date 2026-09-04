# %%
# Exploratory VS Code/Jupyter script for sleep and interaction tuning.
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
    if (candidate / "analysis" / "sleep_analysis_utils.py").exists():
        repo_root = candidate
        break
else:
    raise FileNotFoundError("Could not find analysis/sleep_analysis_utils.py from the current working directory")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import analysis.colony_speed_utils as cs
import analysis.interaction_analysis_utils as ia
import analysis.sleep_analysis_utils as sleep_utils
from analysis.figure_saving import install_auto_savefig

importlib.reload(cs)
importlib.reload(ia)
importlib.reload(sleep_utils)


# %%
# Edit these settings first.
DATASET_ROOT = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20260723/block02")
INTERACTION_ROOT = DATASET_ROOT / "interactions"
TRACKS_ROOT = DATASET_ROOT / "tracks"
PER_TRACK_ROOT = DATASET_ROOT / "stitched" / "per_track"
SPEED_ROOT = DATASET_ROOT / "stitched" / "speed_vectors"
SLEEP_PREDICTIONS_ROOT = DATASET_ROOT / "stitched" / "sleep_predictions"
FIGURE_ROOT = Path("/home/sam-reiter/bucket/ReiterU/sam/analysis_outputs") / "sleep_analysis_figures"
BODYPOINT_SLEEP_OUTPUT_ROOT = FIGURE_ROOT / "bodypoint_speed_sleep_rates"
SAVE_FIGURES = True
FIGURE_DPI = 180

SIDE = "left"  # "left" or "right"
CHUNKS = "all"  # Use "all" for the full side, or ["000", "001"].
MAX_CHUNKS = None
FPS = 24.0
MIN_PRESENT_FRAC = 0.40

SINGLE_ANT_ROW = None       # Row from speed_tracks; overrides track id when not None.
SINGLE_ANT_TRACK_ID = 22  # Set to an integer TrackID, or None for first selected track.
SINGLE_ANT_SIDE = 'left'

BIN_SECONDS = 30.0
SPEED_SMOOTH_SECONDS = 5 * 60
SPEED_YLIM = None
INTERACTION_YLIM = None
INTERACTION_BOUT_GAP_SECONDS = 2.0

MM_PER_PX = 0.016
BODYPOINT_SLEEP_BIN_SECONDS = 5 * 60.0
BODYPOINT_SLEEP_SMOOTH_SECONDS = 30 * 60.0
BODYPOINT_SLEEP_SPEED_THRESHOLD_MM_S = 1.0
BODYPOINT_SLEEP_SPEED_SMOOTH_SECONDS = 2.0
BODYPOINT_SLEEP_MIN_LOW_BODYPOINT_FRACTION = 0.80
BODYPOINT_SLEEP_MIN_VALID_BODYPOINT_FRACTION = 0.60
BODYPOINT_SLEEP_ALLOWED_MOVEMENT_SECONDS = 3.0
BODYPOINT_SLEEP_MIN_SECONDS = 60.0
BODYPOINT_SLEEP_BIN_SLEEP_THRESHOLD = 0.5
BODYPOINT_SLEEP_MIN_BIN_CLASSIFIED_FRACTION = 0.25
BODYPOINT_SLEEP_MAX_TRACKS_PER_SIDE = None  # Set to 1 or 2 for a quick dry run.

RUN_CLASSIFIER_SINGLE_ANT_COMPARISON = False


# %%
# Resolve chunk files and load speed-vector metadata.
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

speed_tracks_all = cs.load_speed_tracks(SPEED_ROOT)
speed_tracks = cs.select_tracks(speed_tracks_all, MIN_PRESENT_FRAC)
speed_tracks_side = speed_tracks[speed_tracks["side"] == SINGLE_ANT_SIDE].reset_index(drop=False)
if speed_tracks_side.empty:
    raise ValueError(f"No speed tracks passed MIN_PRESENT_FRAC={MIN_PRESENT_FRAC} for side={SINGLE_ANT_SIDE!r}")

if SINGLE_ANT_TRACK_ID is None and SINGLE_ANT_ROW is None:
    SINGLE_ANT_TRACK_ID = int(speed_tracks_side.iloc[0]["track_id"])
single_ant_figure_label = (
    f"track{SINGLE_ANT_TRACK_ID}"
    if SINGLE_ANT_ROW is None
    else (
        f"row{SINGLE_ANT_ROW}"
        if SINGLE_ANT_TRACK_ID is None
        else f"row{SINGLE_ANT_ROW}_track{SINGLE_ANT_TRACK_ID}"
    )
)
interactions_raw = pd.DataFrame()
interaction_bout_counts_by_track = {}
if RUN_CLASSIFIER_SINGLE_ANT_COMPARISON:
    install_auto_savefig(
        FIGURE_ROOT / f"{SINGLE_ANT_SIDE}_{single_ant_figure_label}",
        prefix=f"sleep_analysis_{SINGLE_ANT_SIDE}_{single_ant_figure_label}",
        dpi=FIGURE_DPI,
        enabled=SAVE_FIGURES,
    )
    interactions_raw = ia.load_interactions_for_chunks(chunks)
    interaction_bout_counts_by_track = ia.interaction_bout_counts_by_track(
        interactions_raw,
        chunk_global_frame_offset=0,
        fps=FPS,
        event_gap_seconds=INTERACTION_BOUT_GAP_SECONDS,
    )
interaction_counts_by_track = interaction_bout_counts_by_track  # Backward-compatible alias for interactive use.

chunk_table = pd.DataFrame(
    [
        {
            "chunk": chunk.chunk,
            "side": chunk.side,
            "start": ia.format_clock_time(chunk.chunk_start_clock_seconds),
            "frame_offset": chunk.chunk_global_frame_offset,
            "n_frames": chunk.chunk_frame_count,
            "interaction_file": chunk.interaction_path.name,
        }
        for chunk in chunks
    ]
)

print(f"Chunk selection: {chunk_summary} ({SIDE})")
print(f"Loaded speed tracks: {len(speed_tracks):,}/{len(speed_tracks_all):,} passing present_frac > {MIN_PRESENT_FRAC:g}")
if RUN_CLASSIFIER_SINGLE_ANT_COMPARISON:
    print(f"Frame-level interaction rows loaded: {len(interactions_raw):,}")
    print(
        "Interaction bouts counted: "
        f"{sum(int(counts['n_interactions_total'].sum()) for counts in interaction_bout_counts_by_track.values()):,} "
        f"(gap {INTERACTION_BOUT_GAP_SECONDS:g}s)"
    )
print(f"Selected ant: row={SINGLE_ANT_ROW}, track_id={SINGLE_ANT_TRACK_ID}, side={SINGLE_ANT_SIDE}")
display(chunk_table)
display(
    speed_tracks_side[
        ["index", "side", "track_id", "track_name", "present_frac", "n_observed_frames", "n_frames"]
    ].head(20)
)


# %%
# Bodypoint-speed sleep definition:
# sleep = a >=1 minute run where most valid bodypoints stay below threshold,
# while short movement gaps are bridged.
install_auto_savefig(
    BODYPOINT_SLEEP_OUTPUT_ROOT / "figures",
    prefix="bodypoint_speed_sleep_rates",
    dpi=FIGURE_DPI,
    enabled=SAVE_FIGURES,
)
bodypoint_sleep_result = sleep_utils.plot_bodypoint_sleep_percent_timeseries(
    speed_tracks,
    per_track_root=PER_TRACK_ROOT,
    bin_seconds=BODYPOINT_SLEEP_BIN_SECONDS,
    smooth_seconds=BODYPOINT_SLEEP_SMOOTH_SECONDS,
    side="both",
    sleep_threshold=BODYPOINT_SLEEP_BIN_SLEEP_THRESHOLD,
    min_bin_classified_fraction=BODYPOINT_SLEEP_MIN_BIN_CLASSIFIED_FRACTION,
    start_clock_seconds=cs.start_time_from_track_table(speed_tracks),
    mm_per_px=MM_PER_PX,
    speed_threshold_mm_s=BODYPOINT_SLEEP_SPEED_THRESHOLD_MM_S,
    min_low_bodypoint_fraction=BODYPOINT_SLEEP_MIN_LOW_BODYPOINT_FRACTION,
    min_valid_bodypoint_fraction=BODYPOINT_SLEEP_MIN_VALID_BODYPOINT_FRACTION,
    min_sleep_seconds=BODYPOINT_SLEEP_MIN_SECONDS,
    allowed_movement_seconds=BODYPOINT_SLEEP_ALLOWED_MOVEMENT_SECONDS,
    speed_smooth_seconds=BODYPOINT_SLEEP_SPEED_SMOOTH_SECONDS,
    frame_start=ANALYSIS_FRAME_START,
    frame_stop=ANALYSIS_FRAME_STOP,
    max_tracks_per_side=BODYPOINT_SLEEP_MAX_TRACKS_PER_SIDE,
)
bodypoint_sleep_timeseries = bodypoint_sleep_result["timeseries"]
bodypoint_sleep_track_summary = bodypoint_sleep_result["track_summary"]
bodypoint_sleep_bouts = bodypoint_sleep_result["sleep_bouts"]

BODYPOINT_SLEEP_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
bodypoint_sleep_timeseries.to_csv(BODYPOINT_SLEEP_OUTPUT_ROOT / "bodypoint_sleep_timeseries.csv", index=False)
bodypoint_sleep_track_summary.to_csv(BODYPOINT_SLEEP_OUTPUT_ROOT / "bodypoint_sleep_track_summary.csv", index=False)
bodypoint_sleep_bouts.to_csv(BODYPOINT_SLEEP_OUTPUT_ROOT / "bodypoint_sleep_bouts.csv", index=False)

print(f"Saved bodypoint sleep outputs to {BODYPOINT_SLEEP_OUTPUT_ROOT}")
display(bodypoint_sleep_timeseries.head(20))
display(
    bodypoint_sleep_track_summary.groupby("side", as_index=False).agg(
        n_tracks=("track_name", "nunique"),
        median_sleep_fraction=("sleep_fraction_classified_frames", "median"),
        median_classifiable_fraction=("classifiable_fraction_frames", "median"),
        total_sleep_bouts=("n_sleep_bouts", "sum"),
    )
)


# %%
# Optional classifier comparison generated by analysis/compute_track_sleep_predictions.py.
if RUN_CLASSIFIER_SINGLE_ANT_COMPARISON:
    sleep_tracks = sleep_utils.attach_sleep_predictions_to_tracks(
        speed_tracks,
        SLEEP_PREDICTIONS_ROOT,
        require_all=False,
    )
    sleep_tracks = sleep_tracks[sleep_tracks["has_sleep_predictions"]].reset_index(drop=True)
    sleep_tracks_side = sleep_tracks[sleep_tracks["side"] == SINGLE_ANT_SIDE].reset_index(drop=True)
    if sleep_tracks_side.empty:
        raise ValueError(f"No sleep predictions available for side={SINGLE_ANT_SIDE!r}")

    print(f"Loaded sleep predictions for {len(sleep_tracks):,}/{len(speed_tracks):,} selected tracks")
    display(
        sleep_tracks_side[
            [
                "side",
                "track_id",
                "track_name",
                "sleep_prediction_present_frac",
                "sleep_fraction_predicted_frames",
                "mean_sleep_probability",
            ]
        ].head(20)
    )

    single_ant_sleep_interactions = sleep_utils.plot_single_ant_sleep_predictions_interactions(
        sleep_tracks,
        speed_tracks,
        interactions_raw,
        row_number=SINGLE_ANT_ROW,
        track_id=SINGLE_ANT_TRACK_ID,
        side=SINGLE_ANT_SIDE,
        bin_seconds=BIN_SECONDS,
        speed_smooth_seconds=SPEED_SMOOTH_SECONDS,
        analysis_frame_start=ANALYSIS_FRAME_START,
        analysis_frame_stop=ANALYSIS_FRAME_STOP,
        speed_ylim=SPEED_YLIM,
        interaction_ylim=INTERACTION_YLIM,
        counts_by_track=interaction_bout_counts_by_track,
    )
    display(single_ant_sleep_interactions.head(20))


# %%
# Bout-level quantification for the plotted ant, using predicted sleep bouts.
if RUN_CLASSIFIER_SINGLE_ANT_COMPARISON:
    single_ant_sleep_bouts = single_ant_sleep_interactions.attrs.get("sleep_bouts", pd.DataFrame()).copy()
    if single_ant_sleep_bouts.empty:
        single_ant_sleep_summary = pd.DataFrame()
    else:
        single_ant_sleep_bouts["has_interaction"] = single_ant_sleep_bouts["n_interactions_total"] > 0
        single_ant_sleep_summary = pd.DataFrame(
            [
                {
                    "state": "predicted_sleep",
                    "n_bouts": len(single_ant_sleep_bouts),
                    "total_bout_duration_s": float(single_ant_sleep_bouts["bout_duration_seconds"].sum()),
                    "median_bout_duration_s": float(single_ant_sleep_bouts["bout_duration_seconds"].median()),
                    "fraction_bouts_with_interaction": float(single_ant_sleep_bouts["has_interaction"].mean()),
                    "mean_interaction_bouts_per_bout": float(single_ant_sleep_bouts["n_interactions_total"].mean()),
                    "median_interaction_bouts_per_bout": float(single_ant_sleep_bouts["n_interactions_total"].median()),
                    "mean_sleep_probability": float(single_ant_sleep_bouts["mean_sleep_probability"].mean()),
                }
            ]
        )
    display(single_ant_sleep_bouts.head(20))
    display(single_ant_sleep_summary)


# %%
# All predicted sleep bouts for the selected side, ready for downstream inspection.
if RUN_CLASSIFIER_SINGLE_ANT_COMPARISON:
    predicted_sleep_bouts = sleep_utils.load_predicted_sleep_bouts(
        sleep_tracks_side,
        interaction_bout_counts_by_track,
        frame_start=ANALYSIS_FRAME_START,
        frame_stop=ANALYSIS_FRAME_STOP,
    )
    if predicted_sleep_bouts.empty:
        predicted_sleep_summary = pd.DataFrame()
    else:
        predicted_sleep_bouts["has_interaction"] = predicted_sleep_bouts["n_interactions_total"] > 0
        predicted_sleep_summary = (
            predicted_sleep_bouts.groupby(["side"], as_index=False)
            .agg(
                n_tracks=("track_name", "nunique"),
                n_sleep_bouts=("bout_id", "size"),
                total_sleep_seconds=("bout_duration_seconds", "sum"),
                median_sleep_bout_seconds=("bout_duration_seconds", "median"),
                fraction_bouts_with_interaction=("has_interaction", "mean"),
                mean_sleep_probability=("mean_sleep_probability", "mean"),
            )
        )
    display(predicted_sleep_summary)
    display(predicted_sleep_bouts.head(30))


# %%
# Coarse correlations for the plotted ant.
if RUN_CLASSIFIER_SINGLE_ANT_COMPARISON:
    same_ant_sleep_interaction_corr = single_ant_sleep_interactions[
        ["sleep_fraction", "speed_mm_s", "smoothed_speed_mm_s", "n_interactions_total"]
    ].corr(numeric_only=True)
    display(same_ant_sleep_interaction_corr)

# %%
