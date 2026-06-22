# %%
# VS Code/Jupyter interactive script for sleep and interaction exploration.
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
    raise FileNotFoundError("Could not find analysis/sleep_analysis_utils.py from the current working directory")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import analysis.colony_speed_utils as cs
import analysis.interaction_analysis_utils as ia
import analysis.sleep_analysis_utils as sleep_utils

importlib.reload(cs)
importlib.reload(ia)
importlib.reload(sleep_utils)


# %%
# Edit these settings first. Defaults load one chunk/side for fast plotting.
DATASET_ROOT = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20260515/block02")
INTERACTION_ROOT = DATASET_ROOT / "interactions"
TRACKS_ROOT = DATASET_ROOT / "tracks"
SPEED_ROOT = DATASET_ROOT / "stitched" / "speed_vectors"
PER_TRACK_ROOT = DATASET_ROOT / "stitched" / "per_track"

SIDE = "left"  # "left" or "right"
CHUNKS = "all"  # Use "all" for the full side, or ["000", "001"].
MAX_CHUNKS = None
FPS = 24.0
MM_PER_PX = 0.016
MIN_PRESENT_FRAC = 0.40

SINGLE_ANT_ROW = None       # Row from speed_tracks; overrides track id when not None.
SINGLE_ANT_TRACK_ID = 22  # Set to an integer TrackID, or None for first selected track.
SINGLE_ANT_SIDE = 'left'

BIN_SECONDS = 30.0
SPEED_SMOOTH_SECONDS = 5 * 60
QUIESCENCE_SPEED_THRESHOLD_MM_S = 0.1
MIN_QUIESCENT_BOUT_SECONDS = 10.0
SPEED_YLIM = None
INTERACTION_YLIM = None

POSTURE_LOW_INTERACTION_MAX = 0
POSTURE_HIGH_INTERACTION_MIN = None  # None uses POSTURE_HIGH_INTERACTION_QUANTILE.
POSTURE_HIGH_INTERACTION_QUANTILE = 0.75
POSTURE_INCLUDE_NON_QUIESCENT = True
POSTURE_MAX_FRAMES_PER_GROUP = 20000
POSTURE_RANDOM_STATE = 0
POSTURE_JOINT_MAX_POINTS_PER_GROUP = 5000
POSTURE_FEATURE_COLUMNS = [
    "angle_in_left_deg",
    "angle_out_left_deg",
    "angle_in_right_deg",
    "angle_out_right_deg",
    "pose_width_mm",
    "pose_height_mm",
    "pose_area_mm2",
]
POSTURE_JOINT_FEATURE_COLUMNS = [
    *POSTURE_FEATURE_COLUMNS,
    "bp1_bp4_dist_mm",
    "bp1_bp7_dist_mm",
    "bp4_bp5_dist_mm",
    "bp5_bp6_dist_mm",
    "bp7_bp8_dist_mm",
    "bp8_bp9_dist_mm",
    "bp4_bp7_dist_mm",
    "bp5_bp8_dist_mm",
]
POSTURE_EXTRA_DISTANCE_FEATURE_COLUMNS = [
    "bp1_bp5_dist_mm",
    "bp1_bp6_dist_mm",
    "bp1_bp8_dist_mm",
    "bp1_bp9_dist_mm",
    "bp4_bp6_dist_mm",
    "bp4_bp8_dist_mm",
    "bp4_bp9_dist_mm",
    "bp5_bp7_dist_mm",
    "bp5_bp9_dist_mm",
    "bp6_bp7_dist_mm",
    "bp6_bp8_dist_mm",
    "bp6_bp9_dist_mm",
    "bp7_bp9_dist_mm",
]
POSTURE_ENGINEERED_FEATURE_COLUMNS = [
    "pose_aspect_ratio",
    "pose_extent_mm",
    "pose_compactness",
    "angle_in_mean_deg",
    "angle_out_mean_deg",
    "angle_all_mean_deg",
    "angle_in_abs_diff_deg",
    "angle_out_abs_diff_deg",
    "angle_left_sum_deg",
    "angle_right_sum_deg",
    "left_side_chain_mm",
    "right_side_chain_mm",
    "side_chain_mean_mm",
    "side_chain_abs_diff_mm",
    "side_chain_ratio",
    "left_segment_ratio",
    "right_segment_ratio",
    "mid_span_ratio",
    "tip_span_mm",
    "tip_span_to_pose_width",
    "head_to_left_tip_ratio",
    "head_to_right_tip_ratio",
]
POSTURE_NORMALIZED_DISTANCE_FEATURE_COLUMNS = [
    f"{feature}_per_pose_width"
    for feature in [
        *POSTURE_JOINT_FEATURE_COLUMNS[7:],
        *POSTURE_EXTRA_DISTANCE_FEATURE_COLUMNS,
    ]
]
POSTURE_CLASSIFIER_FEATURE_COLUMNS = [
    *POSTURE_JOINT_FEATURE_COLUMNS,
    *POSTURE_EXTRA_DISTANCE_FEATURE_COLUMNS,
    *POSTURE_ENGINEERED_FEATURE_COLUMNS,
    *POSTURE_NORMALIZED_DISTANCE_FEATURE_COLUMNS,
]

CLASSIFIER_FEATURE_COLUMNS = POSTURE_CLASSIFIER_FEATURE_COLUMNS
CLASSIFIER_SPLIT_BY = "time_bout"  # "time_bout", "random_bout", or "random_frame".
CLASSIFIER_TEST_SIZE = 0.5
CLASSIFIER_RANDOM_STATE = POSTURE_RANDOM_STATE
CLASSIFIER_MAX_WEIGHT_FEATURES = 12

LOW_INTERACTION_CLUSTER_SPLIT = "test"  # Use held-out bouts/ants by default.
LOW_INTERACTION_CLUSTER_SCORE_THRESHOLD = 0.5
LOW_INTERACTION_CLUSTER_MIN_FRAME_FRACTION = 0.5
LOW_INTERACTION_CLUSTER_CORRELATION_X = "n_interactions_total"
LOW_INTERACTION_CLUSTER_CORRELATION_Y = "bout_duration_seconds"

CROSS_ANT_TRACK_IDS = None  # None uses the first CROSS_ANT_MAX_TRACKS selected tracks on SINGLE_ANT_SIDE.
CROSS_ANT_MAX_TRACKS = 20
CROSS_ANT_TRAIN_TRACK_IDS = None  # Set explicit track IDs here, or leave None for a random ant split.
CROSS_ANT_TEST_TRACK_IDS = None
CROSS_ANT_MAX_FRAMES_PER_GROUP = 5000


# %%
# Resolve chunk files, load speed-vector metadata, and load directed interactions.
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

interactions_raw = ia.load_interactions_for_chunks(chunks)
interaction_counts_by_track = ia.interaction_frame_counts_by_track(
    interactions_raw,
    chunk_global_frame_offset=0,
)

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
print(f"Interaction rows loaded: {len(interactions_raw):,}")
print(f"Selected ant: row={SINGLE_ANT_ROW}, track_id={SINGLE_ANT_TRACK_ID}, side={SINGLE_ANT_SIDE}")
display(chunk_table)
display(
    speed_tracks_side[
        ["index", "side", "track_id", "track_name", "present_frac", "n_observed_frames", "n_frames"]
    ].head(20)
)


# %%
# Single-ant plot: 5-minute-smoothed speed plus directed social interactions.
single_ant_speed_interactions = sleep_utils.plot_single_ant_speed_interactions(
    speed_tracks,
    interactions_raw,
    row_number=SINGLE_ANT_ROW,
    track_id=SINGLE_ANT_TRACK_ID,
    side=SINGLE_ANT_SIDE,
    bin_seconds=BIN_SECONDS,
    speed_smooth_seconds=SPEED_SMOOTH_SECONDS,
    quiescence_speed_threshold_mm_s=QUIESCENCE_SPEED_THRESHOLD_MM_S,
    min_quiescent_seconds=MIN_QUIESCENT_BOUT_SECONDS,
    analysis_frame_start=ANALYSIS_FRAME_START,
    analysis_frame_stop=ANALYSIS_FRAME_STOP,
    speed_ylim=SPEED_YLIM,
    interaction_ylim=INTERACTION_YLIM,
    counts_by_track=interaction_counts_by_track,
)
display(single_ant_speed_interactions.head(20))


# %%
# Quick bout-level quantification for the plotted ant.
single_ant_track_row = sleep_utils.selected_speed_track(
    speed_tracks,
    row_number=SINGLE_ANT_ROW,
    track_id=SINGLE_ANT_TRACK_ID,
    side=SINGLE_ANT_SIDE,
)
single_ant_quiescent_bouts = single_ant_speed_interactions.attrs.get("quiescent_bouts", pd.DataFrame()).copy()
single_ant_non_quiescent_bouts = sleep_utils.non_quiescent_threshold_bouts_for_track_row(
    single_ant_track_row,
    interaction_counts_by_track,
    speed_smooth_seconds=SPEED_SMOOTH_SECONDS,
    quiescence_speed_threshold_mm_s=QUIESCENCE_SPEED_THRESHOLD_MM_S,
    min_non_quiescent_seconds=MIN_QUIESCENT_BOUT_SECONDS,
    frame_start=ANALYSIS_FRAME_START,
    frame_stop=ANALYSIS_FRAME_STOP,
)
if not single_ant_quiescent_bouts.empty:
    single_ant_quiescent_bouts["state"] = "quiescent"
if not single_ant_non_quiescent_bouts.empty:
    single_ant_non_quiescent_bouts["state"] = "not_quiescent"
single_ant_bouts = pd.concat(
    [df for df in [single_ant_quiescent_bouts, single_ant_non_quiescent_bouts] if not df.empty],
    ignore_index=True,
) if (not single_ant_quiescent_bouts.empty or not single_ant_non_quiescent_bouts.empty) else pd.DataFrame()

if single_ant_bouts.empty:
    single_ant_quiescence_summary = pd.DataFrame()
else:
    single_ant_bouts["has_interaction"] = single_ant_bouts["n_interactions_total"] > 0
    single_ant_quiescence_summary = (
        single_ant_bouts.groupby("state", sort=False)
        .agg(
            n_bouts=("bout_id", "size"),
            total_bout_duration_s=("bout_duration_seconds", "sum"),
            median_bout_duration_s=("bout_duration_seconds", "median"),
            fraction_bouts_with_interaction=("has_interaction", "mean"),
            mean_interactions_per_bout=("n_interactions_total", "mean"),
            median_interactions_per_bout=("n_interactions_total", "median"),
        )
        .reset_index()
    )
display(single_ant_bouts.head(20))
display(single_ant_quiescence_summary)


# %%
# Compare posture distributions in quiescent periods with little/no vs more interactions.
posture_samples, posture_bout_summary = sleep_utils.quiescent_posture_samples_by_interaction(
    single_ant_speed_interactions,
    speed_tracks,
    row_number=SINGLE_ANT_ROW,
    track_id=SINGLE_ANT_TRACK_ID,
    side=SINGLE_ANT_SIDE,
    counts_by_track=interaction_counts_by_track,
    per_track_root=PER_TRACK_ROOT,
    fps=FPS,
    mm_per_px=MM_PER_PX,
    speed_smooth_seconds=SPEED_SMOOTH_SECONDS,
    quiescence_speed_threshold_mm_s=QUIESCENCE_SPEED_THRESHOLD_MM_S,
    min_quiescent_seconds=MIN_QUIESCENT_BOUT_SECONDS,
    analysis_frame_start=ANALYSIS_FRAME_START,
    analysis_frame_stop=ANALYSIS_FRAME_STOP,
    low_interaction_max=POSTURE_LOW_INTERACTION_MAX,
    high_interaction_min=POSTURE_HIGH_INTERACTION_MIN,
    high_interaction_quantile=POSTURE_HIGH_INTERACTION_QUANTILE,
    include_non_quiescent=POSTURE_INCLUDE_NON_QUIESCENT,
    max_frames_per_group=POSTURE_MAX_FRAMES_PER_GROUP,
    random_state=POSTURE_RANDOM_STATE,
)
display(posture_bout_summary)

posture_distribution_summary = sleep_utils.plot_quiescent_posture_distributions(
    posture_samples,
    feature_cols=POSTURE_FEATURE_COLUMNS,
)
display(posture_distribution_summary)


# %%
# Joint posture analysis for quiescent little/no-interaction vs more-interaction frames.
posture_joint_embedding, posture_joint_metrics, posture_joint_loadings = (
    sleep_utils.plot_multivariate_quiescent_posture_analysis(
        posture_samples,
        feature_cols=POSTURE_JOINT_FEATURE_COLUMNS,
        groups=("little_or_no_interaction", "more_interaction"),
        random_state=POSTURE_RANDOM_STATE,
        max_points_per_group=POSTURE_JOINT_MAX_POINTS_PER_GROUP,
    )
)
display(posture_joint_metrics)
display(posture_joint_loadings)


# %%
# Classifier reliability for the selected ant: train on one subset of quiescent bouts, test on held-out bouts.
same_ant_classifier_predictions, same_ant_classifier_metrics, same_ant_classifier_weights, same_ant_classifier_model = (
    sleep_utils.plot_quiescent_posture_classifier(
        posture_samples,
        feature_cols=CLASSIFIER_FEATURE_COLUMNS,
        groups=("little_or_no_interaction", "more_interaction"),
        split_by=CLASSIFIER_SPLIT_BY,
        test_size=CLASSIFIER_TEST_SIZE,
        random_state=CLASSIFIER_RANDOM_STATE,
        max_weight_features=CLASSIFIER_MAX_WEIGHT_FEATURES,
    )
)
display(same_ant_classifier_metrics)
display(same_ant_classifier_weights)


# %%
# Treat classifier-low quiescent bouts as their own state and ask how interactions relate to that state.
same_ant_low_quiescent_bins, same_ant_low_quiescent_summary, same_ant_low_quiescent_bouts, same_ant_low_quiescent_corr = (
    sleep_utils.plot_low_interaction_quiescent_cluster_analysis(
        same_ant_classifier_predictions,
        groups=("little_or_no_interaction", "more_interaction"),
        split=LOW_INTERACTION_CLUSTER_SPLIT,
        bin_seconds=BIN_SECONDS,
        low_score_threshold=LOW_INTERACTION_CLUSTER_SCORE_THRESHOLD,
        min_predicted_low_fraction=LOW_INTERACTION_CLUSTER_MIN_FRAME_FRACTION,
        correlation_x_col=LOW_INTERACTION_CLUSTER_CORRELATION_X,
        correlation_y_col=LOW_INTERACTION_CLUSTER_CORRELATION_Y,
        title=f"Selected ant {SINGLE_ANT_SIDE} TrackID {SINGLE_ANT_TRACK_ID}: low-interaction quiescent classifier state",
    )
)
display(same_ant_low_quiescent_summary)
display(same_ant_low_quiescent_corr.to_frame().T)
display(same_ant_low_quiescent_bouts.head(30))


# %%
# Build posture samples from several ants for cross-ant train/test.
if CROSS_ANT_TRACK_IDS is None:
    explicit_cross_ant_ids = []
    if CROSS_ANT_TRAIN_TRACK_IDS is not None:
        explicit_cross_ant_ids.extend(list(CROSS_ANT_TRAIN_TRACK_IDS))
    if CROSS_ANT_TEST_TRACK_IDS is not None:
        explicit_cross_ant_ids.extend(list(CROSS_ANT_TEST_TRACK_IDS))
    if explicit_cross_ant_ids:
        cross_ant_track_ids = list(dict.fromkeys(int(track_id) for track_id in explicit_cross_ant_ids))
    else:
        cross_ant_track_ids = (
            speed_tracks_side["track_id"]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .head(int(CROSS_ANT_MAX_TRACKS))
            .to_list()
        )
else:
    cross_ant_track_ids = list(CROSS_ANT_TRACK_IDS)

if len(cross_ant_track_ids) < 2:
    raise ValueError("Need at least two CROSS_ANT_TRACK_IDS for a cross-ant classifier split")

cross_ant_posture_samples, cross_ant_bout_summary, cross_ant_speed_interactions = (
    sleep_utils.quiescent_posture_samples_for_tracks(
        speed_tracks,
        interactions_raw,
        track_ids=cross_ant_track_ids,
        side=SINGLE_ANT_SIDE,
        per_track_root=PER_TRACK_ROOT,
        fps=FPS,
        mm_per_px=MM_PER_PX,
        bin_seconds=BIN_SECONDS,
        speed_smooth_seconds=SPEED_SMOOTH_SECONDS,
        quiescence_speed_threshold_mm_s=QUIESCENCE_SPEED_THRESHOLD_MM_S,
        min_quiescent_seconds=MIN_QUIESCENT_BOUT_SECONDS,
        analysis_frame_start=ANALYSIS_FRAME_START,
        analysis_frame_stop=ANALYSIS_FRAME_STOP,
        low_interaction_max=POSTURE_LOW_INTERACTION_MAX,
        high_interaction_min=POSTURE_HIGH_INTERACTION_MIN,
        high_interaction_quantile=POSTURE_HIGH_INTERACTION_QUANTILE,
        include_non_quiescent=False,
        max_frames_per_group=CROSS_ANT_MAX_FRAMES_PER_GROUP,
        random_state=CLASSIFIER_RANDOM_STATE,
        counts_by_track=interaction_counts_by_track,
    )
)
display(cross_ant_bout_summary)


# %%
# Cross-ant classifier: train on some ants and test on different ants.
available_cross_ant_track_ids = sorted(cross_ant_posture_samples["track_id"].dropna().astype(int).unique().tolist())
if CROSS_ANT_TRAIN_TRACK_IDS is None and CROSS_ANT_TEST_TRACK_IDS is None:
    rng = np.random.default_rng(CLASSIFIER_RANDOM_STATE)
    shuffled_track_ids = np.asarray(available_cross_ant_track_ids, dtype=int)
    rng.shuffle(shuffled_track_ids)
    split_at = max(1, len(shuffled_track_ids) // 2)
    train_track_ids = sorted(shuffled_track_ids[:split_at].astype(int).tolist())
    test_track_ids = sorted(shuffled_track_ids[split_at:].astype(int).tolist())
elif CROSS_ANT_TRAIN_TRACK_IDS is None:
    test_track_ids = [int(track_id) for track_id in CROSS_ANT_TEST_TRACK_IDS]
    train_track_ids = [track_id for track_id in available_cross_ant_track_ids if track_id not in set(test_track_ids)]
elif CROSS_ANT_TEST_TRACK_IDS is None:
    train_track_ids = [int(track_id) for track_id in CROSS_ANT_TRAIN_TRACK_IDS]
    test_track_ids = [track_id for track_id in available_cross_ant_track_ids if track_id not in set(train_track_ids)]
else:
    train_track_ids = [int(track_id) for track_id in CROSS_ANT_TRAIN_TRACK_IDS]
    test_track_ids = [int(track_id) for track_id in CROSS_ANT_TEST_TRACK_IDS]

if not train_track_ids or not test_track_ids:
    raise ValueError("Cross-ant classifier needs at least one train ant and one test ant")

train_track_id_set = {str(track_id) for track_id in train_track_ids}
test_track_id_set = {str(track_id) for track_id in test_track_ids}
cross_ant_train_mask = cross_ant_posture_samples["track_id"].astype(str).isin(train_track_id_set)
cross_ant_test_mask = cross_ant_posture_samples["track_id"].astype(str).isin(test_track_id_set)

print(f"Cross-ant train track IDs: {train_track_ids}")
print(f"Cross-ant test track IDs: {test_track_ids}")

cross_ant_classifier_predictions, cross_ant_classifier_metrics, cross_ant_classifier_weights, cross_ant_classifier_model = (
    sleep_utils.plot_quiescent_posture_classifier(
        cross_ant_posture_samples,
        feature_cols=CLASSIFIER_FEATURE_COLUMNS,
        groups=("little_or_no_interaction", "more_interaction"),
        train_mask=cross_ant_train_mask,
        test_mask=cross_ant_test_mask,
        random_state=CLASSIFIER_RANDOM_STATE,
        split_label="cross_ant",
        max_weight_features=CLASSIFIER_MAX_WEIGHT_FEATURES,
    )
)
display(cross_ant_classifier_metrics)
display(cross_ant_classifier_weights)


# %%
# Same low-interaction-quiescent state analysis on held-out ants from the cross-ant classifier.
cross_ant_low_quiescent_bins, cross_ant_low_quiescent_summary, cross_ant_low_quiescent_bouts, cross_ant_low_quiescent_corr = (
    sleep_utils.plot_low_interaction_quiescent_cluster_analysis(
        cross_ant_classifier_predictions,
        groups=("little_or_no_interaction", "more_interaction"),
        split=LOW_INTERACTION_CLUSTER_SPLIT,
        bin_seconds=BIN_SECONDS,
        low_score_threshold=LOW_INTERACTION_CLUSTER_SCORE_THRESHOLD,
        min_predicted_low_fraction=LOW_INTERACTION_CLUSTER_MIN_FRAME_FRACTION,
        correlation_x_col=LOW_INTERACTION_CLUSTER_CORRELATION_X,
        correlation_y_col=LOW_INTERACTION_CLUSTER_CORRELATION_Y,
        title=f"Cross-ant held-out ants: low-interaction quiescent classifier state",
    )
)
display(cross_ant_low_quiescent_summary)
display(cross_ant_low_quiescent_corr.to_frame().T)
display(cross_ant_low_quiescent_bouts.head(30))

# %%
