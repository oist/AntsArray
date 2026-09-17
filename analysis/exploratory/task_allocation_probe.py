# %%
# Exploratory VS Code/Jupyter script for probing local task-allocation rules.
#
# This first pass uses speed-derived activation as the behavioral transition,
# occupancy clusters as provisional task/spatial classes, and directed
# antenna/body contacts as social-history events.
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
    if (candidate / "analysis" / "task_allocation_probe_utils.py").exists():
        repo_root = candidate
        break
else:
    raise FileNotFoundError("Could not find analysis/task_allocation_probe_utils.py from the current working directory")

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import analysis.interaction_analysis_utils as ia
import analysis.task_allocation_probe_utils as tap
from analysis.figure_saving import install_auto_savefig

importlib.reload(ia)
importlib.reload(tap)


# %%
# Editable settings.
DATASET_ROOT = Path("/home/sam-reiter/bucket/ReiterU/Ants/basler/20260723/block02")
INTERACTION_ROOT = DATASET_ROOT / "interactions"
TRACKS_ROOT = DATASET_ROOT / "tracks"
SPEED_ROOT = DATASET_ROOT / "stitched" / "speed_vectors"
CLUSTER_TABLE_PATH = tap.resolve_cluster_table_path(DATASET_ROOT)

SIDES_TO_ANALYZE = ("left", "right")
CHUNKS = "all"
MAX_CHUNKS = 4  # Set to None for the full 0723 block after the workflow is tuned.
FPS = 24.0
MM_PER_PX = 0.016
MIN_PRESENT_FRAC = 0.40

STATE_BIN_SECONDS = 60.0
QUIET_SPEED_THRESHOLD_MM_S = 0.1
ACTIVE_SPEED_THRESHOLD_MM_S = 0.5
QUIET_FRACTION_THRESHOLD = 0.80
ACTIVE_FRACTION_THRESHOLD = 0.50
MIN_VALID_FRACTION_PER_BIN = 0.20
LIGHT_ON_HOUR = 5.5

COLLAPSE_CONTACTS_TO_BOUTS = True
INTERACTION_EVENT_GAP_SECONDS = 2.0
KEEP_FIRST_OBSERVED_CONTACT_AS_ONSET = False
HISTORY_WINDOWS_SECONDS = (30.0, 120.0, 600.0)
HISTORY_LAG_WINDOWS_SECONDS = ((0.0, 30.0), (30.0, 120.0), (120.0, 600.0))
HISTORY_LEAKY_TAU_SECONDS = (60.0, 300.0)
PARTNER_CLUSTER_WINDOW_SECONDS = 600.0
MAX_PARTNER_CLUSTER_FEATURES = 10

LOAD_POSITION_CONTEXT = True
POSITION_BODYPOINT = 0
POSITION_FRAME_STEP = 24  # 24 at 24 fps samples positions at 1 Hz.
LOCAL_NEIGHBOR_RADIUS_MM = 15.0

MAX_INTERACTIONS_PER_CHUNK = None
SAMPLE_INTERACTIONS = False
DROP_UNCLUSTERED_INTERACTIONS = True
USE_CACHE = True
FORCE_REBUILD_CACHE = False

EVENT_TRIGGER_PRE_SECONDS = 10 * 60.0
EVENT_TRIGGER_POST_SECONDS = 20 * 60.0
EVENT_TRIGGER_CONDITION_COL = "interaction_role"
EVENT_TRIGGER_REQUIRE_QUIET = True
EVENT_TRIGGER_MAX_EVENTS = 5000
EVENT_TRIGGER_CONTROL_REPLICATES = 2
EVENT_TRIGGER_CONTROL_EXCLUDE_SECONDS = 0.0
EVENT_TRIGGER_CONTROL_MATCH_LEVELS = ("same_ant", "same_side_cluster", "same_side")
EVENT_TRIGGER_SEPARATE_CONTROL_MATCH_LEVELS = True
CONTROL_EXCLUDE_ANY_CONTACT_BIN = True
EVENT_TRIGGER_POOL_INTERACTION_ROLES = True
FINE_EVENT_STATE_BIN_SECONDS = 5.0
FINE_EVENT_TRIGGER_PRE_SECONDS = 2 * 60.0
FINE_EVENT_TRIGGER_POST_SECONDS = 5 * 60.0
FINE_EVENT_TRIGGER_CONTROL_EXCLUDE_SECONDS = 0.0

MODEL_COMPARISON_METRIC = "aic"
OUTPUT_ROOT = DATASET_ROOT / "analysis_outputs" / "task_allocation_probe"
CACHE_ROOT = DATASET_ROOT / "stitched" / "analysis_cache" / "task_allocation_probe"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
SAVE_FIGURES = True
FIGURE_DPI = 180


# %%
# Resolve chunks, clusters, cache, and figure output.
chunks = []
for side in SIDES_TO_ANALYZE:
    chunks.extend(
        ia.resolve_chunks(
            INTERACTION_ROOT,
            TRACKS_ROOT,
            chunks=CHUNKS,
            side=side,
            fps=FPS,
            max_chunks=MAX_CHUNKS,
        )
    )

clusters = tap.load_task_proxy_clusters(CLUSTER_TABLE_PATH, sides=SIDES_TO_ANALYZE)
cache_settings = {
    "cache_version": 1,
    "dataset_root": str(DATASET_ROOT),
    "cluster_table": ia.file_fingerprint(CLUSTER_TABLE_PATH),
    "sides": list(SIDES_TO_ANALYZE),
    "chunks": CHUNKS,
    "max_chunks": MAX_CHUNKS,
    "fps": FPS,
    "mm_per_px": MM_PER_PX,
    "min_present_frac": MIN_PRESENT_FRAC,
    "state_bin_seconds": STATE_BIN_SECONDS,
    "quiet_speed_threshold_mm_s": QUIET_SPEED_THRESHOLD_MM_S,
    "active_speed_threshold_mm_s": ACTIVE_SPEED_THRESHOLD_MM_S,
    "quiet_fraction_threshold": QUIET_FRACTION_THRESHOLD,
    "active_fraction_threshold": ACTIVE_FRACTION_THRESHOLD,
    "min_valid_fraction_per_bin": MIN_VALID_FRACTION_PER_BIN,
    "collapse_contacts_to_bouts": COLLAPSE_CONTACTS_TO_BOUTS,
    "interaction_event_gap_seconds": INTERACTION_EVENT_GAP_SECONDS,
    "keep_first_observed_contact_as_onset": KEEP_FIRST_OBSERVED_CONTACT_AS_ONSET,
    "history_windows_seconds": HISTORY_WINDOWS_SECONDS,
    "history_lag_windows_seconds": HISTORY_LAG_WINDOWS_SECONDS,
    "history_leaky_tau_seconds": HISTORY_LEAKY_TAU_SECONDS,
    "partner_cluster_window_seconds": PARTNER_CLUSTER_WINDOW_SECONDS,
    "max_partner_cluster_features": MAX_PARTNER_CLUSTER_FEATURES,
    "load_position_context": LOAD_POSITION_CONTEXT,
    "position_bodypoint": POSITION_BODYPOINT,
    "position_frame_step": POSITION_FRAME_STEP,
    "local_neighbor_radius_mm": LOCAL_NEIGHBOR_RADIUS_MM,
    "max_interactions_per_chunk": MAX_INTERACTIONS_PER_CHUNK,
    "sample_interactions": SAMPLE_INTERACTIONS,
    "drop_unclustered_interactions": DROP_UNCLUSTERED_INTERACTIONS,
    "event_trigger_pre_seconds": EVENT_TRIGGER_PRE_SECONDS,
    "event_trigger_post_seconds": EVENT_TRIGGER_POST_SECONDS,
    "event_trigger_condition_col": EVENT_TRIGGER_CONDITION_COL,
    "event_trigger_require_quiet": EVENT_TRIGGER_REQUIRE_QUIET,
    "event_trigger_max_events": EVENT_TRIGGER_MAX_EVENTS,
    "event_trigger_control_replicates": EVENT_TRIGGER_CONTROL_REPLICATES,
    "event_trigger_control_exclude_seconds": EVENT_TRIGGER_CONTROL_EXCLUDE_SECONDS,
    "event_trigger_control_match_levels": EVENT_TRIGGER_CONTROL_MATCH_LEVELS,
    "event_trigger_separate_control_match_levels": EVENT_TRIGGER_SEPARATE_CONTROL_MATCH_LEVELS,
    "control_exclude_any_contact_bin": CONTROL_EXCLUDE_ANY_CONTACT_BIN,
    "event_trigger_pool_interaction_roles": EVENT_TRIGGER_POOL_INTERACTION_ROLES,
    "fine_event_state_bin_seconds": FINE_EVENT_STATE_BIN_SECONDS,
    "fine_event_trigger_pre_seconds": FINE_EVENT_TRIGGER_PRE_SECONDS,
    "fine_event_trigger_post_seconds": FINE_EVENT_TRIGGER_POST_SECONDS,
    "fine_event_trigger_control_exclude_seconds": FINE_EVENT_TRIGGER_CONTROL_EXCLUDE_SECONDS,
}
cache_key = ia.interaction_analysis_cache_key(chunks, cache_settings)
cache_dir = CACHE_ROOT / cache_key
install_auto_savefig(
    FIGURE_ROOT / cache_key,
    prefix="task_allocation_probe",
    dpi=FIGURE_DPI,
    enabled=SAVE_FIGURES,
)

chunk_table = pd.DataFrame(
    [
        {
            "side": chunk.side,
            "chunk": chunk.chunk,
            "chunk_start": ia.format_clock_time(chunk.chunk_start_clock_seconds),
            "frame_offset": chunk.chunk_global_frame_offset,
            "n_frames": chunk.chunk_frame_count,
            "interaction_file": chunk.interaction_path.name,
            "track_file": chunk.track_path.name,
        }
        for chunk in chunks
    ]
)

print(f"Dataset: {DATASET_ROOT}")
print(f"Cluster table: {CLUSTER_TABLE_PATH}")
print(f"Cache directory: {cache_dir}")
print(f"Selected chunks: {len(chunks)} across {', '.join(SIDES_TO_ANALYZE)}")
display(chunk_table)
display(clusters.groupby(["side", "cluster_id"])["TrackID"].nunique().rename("n_ants").reset_index())


# %%
# Build speed-derived ant state bins.
speed_tracks = tap.load_speed_tracks_with_clusters(
    SPEED_ROOT,
    clusters,
    min_present_frac=MIN_PRESENT_FRAC,
)
state_bins_raw = ia.load_or_build_table(
    cache_dir / "state_bins_raw.parquet",
    lambda: tap.build_state_bins_from_speed(
        speed_tracks,
        chunks,
        bin_seconds=STATE_BIN_SECONDS,
        fps=FPS,
        quiet_speed_threshold_mm_s=QUIET_SPEED_THRESHOLD_MM_S,
        active_speed_threshold_mm_s=ACTIVE_SPEED_THRESHOLD_MM_S,
        quiet_fraction_threshold=QUIET_FRACTION_THRESHOLD,
        active_fraction_threshold=ACTIVE_FRACTION_THRESHOLD,
        min_valid_fraction=MIN_VALID_FRACTION_PER_BIN,
        light_on_hour=LIGHT_ON_HOUR,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
state_bins = tap.add_next_state_columns(state_bins_raw, bin_seconds=STATE_BIN_SECONDS)

print(f"Speed tracks: {len(speed_tracks):,}")
print(f"State bins: {len(state_bins):,}")
display(
    state_bins.groupby(["side", "cluster_id", "state"])["track_id"]
    .count()
    .rename("n_ant_time_bins")
    .reset_index()
    .head(40)
)
display(state_bins.head())


# %%
# Add coarse spatial and local-availability context from chunk track positions.
if LOAD_POSITION_CONTEXT:
    position_bins = ia.load_or_build_table(
        cache_dir / "position_context_bins.parquet",
        lambda: tap.build_position_context_bins(
            chunks,
            bin_seconds=STATE_BIN_SECONDS,
            fps=FPS,
            mm_per_px=MM_PER_PX,
            bodypoint=POSITION_BODYPOINT,
            frame_step=POSITION_FRAME_STEP,
            local_radius_mm=LOCAL_NEIGHBOR_RADIUS_MM,
        ),
        use_cache=USE_CACHE,
        force=FORCE_REBUILD_CACHE,
    )
    state_bins = state_bins.merge(
        position_bins,
        on=["side", "track_id", "bin_index"],
        how="left",
        validate="one_to_one",
    )
else:
    position_bins = pd.DataFrame()

print(f"Position-context bins: {len(position_bins):,}")
display(position_bins.head())


# %%
# Load directed interactions, collapse frame-level contacts to onsets, and
# attach recent social-history features to each ant-time bin.
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
    lambda: tap.attach_side_aware_cluster_labels(
        interactions_raw,
        clusters,
        drop_unclustered=DROP_UNCLUSTERED_INTERACTIONS,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
focal_events = ia.load_or_build_table(
    cache_dir / "focal_interaction_onsets.parquet",
    lambda: tap.directed_interaction_onsets(
        interactions,
        fps=FPS,
        event_gap_seconds=INTERACTION_EVENT_GAP_SECONDS,
        collapse_contacts=COLLAPSE_CONTACTS_TO_BOUTS,
        keep_first_observed_contact=KEEP_FIRST_OBSERVED_CONTACT_AS_ONSET,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
focal_contact_frames = ia.load_or_build_table(
    cache_dir / "focal_contact_frames.parquet",
    lambda: tap.directed_interaction_onsets(
        interactions,
        fps=FPS,
        event_gap_seconds=INTERACTION_EVENT_GAP_SECONDS,
        collapse_contacts=False,
        keep_first_observed_contact=True,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
state_history = ia.load_or_build_table(
    cache_dir / "state_bins_with_interaction_history.parquet",
    lambda: tap.add_interaction_history_features(
        state_bins,
        focal_events,
        fps=FPS,
        cumulative_windows_seconds=HISTORY_WINDOWS_SECONDS,
        lag_windows_seconds=HISTORY_LAG_WINDOWS_SECONDS,
        leaky_tau_seconds=HISTORY_LEAKY_TAU_SECONDS,
        partner_cluster_window_seconds=PARTNER_CLUSTER_WINDOW_SECONDS,
        max_partner_cluster_features=MAX_PARTNER_CLUSTER_FEATURES,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
transition_design = tap.make_quiet_to_active_design(state_history)

print(f"Raw directed interaction rows: {len(interactions_raw):,}")
print(f"Clustered directed interaction rows: {len(interactions):,}")
print(f"Focal interaction onsets: {len(focal_events):,}")
print(f"Focal interaction contact-frame rows: {len(focal_contact_frames):,}")
print(f"Quiet-to-active design rows: {len(transition_design):,}")
display(
    transition_design.groupby(["side", "cluster_id"])
    .agg(
        n_quiet_bins=("activate_next_bin", "size"),
        n_activations=("activate_next_bin", "sum"),
        activation_probability=("activate_next_bin", "mean"),
    )
    .reset_index()
)
display(transition_design.head())


# %%
# Plot colony-side and cluster-level behavioral state over time.
cluster_active_timeseries = tap.plot_cluster_state_timeseries(
    state_history,
    value_col="active_fraction",
    smooth_bins=2.0,
    title="Speed-derived active fraction by occupancy cluster",
)
cluster_quiet_timeseries = tap.plot_cluster_state_timeseries(
    state_history,
    value_col="quiet_fraction",
    smooth_bins=2.0,
    title="Speed-derived quiet fraction by occupancy cluster",
)
display(cluster_active_timeseries.head(30))


# %%
# Event-triggered activation around new directed interaction onsets, compared
# with matched quiet no-interaction control times.
trigger_table, event_curve_rows, event_curve_summary = ia.load_or_build_pickle(
    cache_dir / "interaction_control_triggered_state_curve.pkl",
    lambda: tap.build_interaction_control_triggered_state_curve(
        state_history,
        focal_events,
        fps=FPS,
        bin_seconds=STATE_BIN_SECONDS,
        pre_seconds=EVENT_TRIGGER_PRE_SECONDS,
        post_seconds=EVENT_TRIGGER_POST_SECONDS,
        require_quiet_at_trigger=EVENT_TRIGGER_REQUIRE_QUIET,
        control_replicates=EVENT_TRIGGER_CONTROL_REPLICATES,
        control_exclude_seconds=EVENT_TRIGGER_CONTROL_EXCLUDE_SECONDS,
        control_exclusion_events=focal_contact_frames if CONTROL_EXCLUDE_ANY_CONTACT_BIN else focal_events,
        control_match_levels=EVENT_TRIGGER_CONTROL_MATCH_LEVELS,
        separate_control_match_levels=EVENT_TRIGGER_SEPARATE_CONTROL_MATCH_LEVELS,
        pool_interaction_roles=EVENT_TRIGGER_POOL_INTERACTION_ROLES,
        max_events=EVENT_TRIGGER_MAX_EVENTS,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
tap.plot_event_triggered_state_curve(
    event_curve_summary,
    y_col="active_fraction",
    title="New-interaction vs no-interaction control: activation probability",
)
tap.plot_event_triggered_state_curve(
    event_curve_summary,
    y_col="mean_speed_mm_s",
    title="New-interaction vs no-interaction control: mean speed",
)
display(
    trigger_table.groupby(["trigger_condition", "control_match_level"])["trigger_row_id"]
    .nunique()
    .rename("n_triggers")
    .reset_index()
)
display(event_curve_summary.head(30))


# %%
# Higher-resolution event-triggered curves. These use the same quiet-at-trigger
# rule and interaction onsets, but summarize speed in much smaller bins.
fine_bin_label = f"{FINE_EVENT_STATE_BIN_SECONDS:g}s".replace(".", "p")
fine_state_bins_raw = ia.load_or_build_table(
    cache_dir / f"state_bins_raw_{fine_bin_label}.parquet",
    lambda: tap.build_state_bins_from_speed(
        speed_tracks,
        chunks,
        bin_seconds=FINE_EVENT_STATE_BIN_SECONDS,
        fps=FPS,
        quiet_speed_threshold_mm_s=QUIET_SPEED_THRESHOLD_MM_S,
        active_speed_threshold_mm_s=ACTIVE_SPEED_THRESHOLD_MM_S,
        quiet_fraction_threshold=QUIET_FRACTION_THRESHOLD,
        active_fraction_threshold=ACTIVE_FRACTION_THRESHOLD,
        min_valid_fraction=MIN_VALID_FRACTION_PER_BIN,
        light_on_hour=LIGHT_ON_HOUR,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
fine_state_bins = tap.add_next_state_columns(fine_state_bins_raw, bin_seconds=FINE_EVENT_STATE_BIN_SECONDS)
fine_trigger_table, fine_event_curve_rows, fine_event_curve_summary = ia.load_or_build_pickle(
    cache_dir / f"interaction_control_triggered_state_curve_{fine_bin_label}.pkl",
    lambda: tap.build_interaction_control_triggered_state_curve(
        fine_state_bins,
        focal_events,
        fps=FPS,
        bin_seconds=FINE_EVENT_STATE_BIN_SECONDS,
        pre_seconds=FINE_EVENT_TRIGGER_PRE_SECONDS,
        post_seconds=FINE_EVENT_TRIGGER_POST_SECONDS,
        require_quiet_at_trigger=EVENT_TRIGGER_REQUIRE_QUIET,
        control_replicates=EVENT_TRIGGER_CONTROL_REPLICATES,
        control_exclude_seconds=FINE_EVENT_TRIGGER_CONTROL_EXCLUDE_SECONDS,
        control_exclusion_events=focal_contact_frames if CONTROL_EXCLUDE_ANY_CONTACT_BIN else focal_events,
        control_match_levels=EVENT_TRIGGER_CONTROL_MATCH_LEVELS,
        separate_control_match_levels=EVENT_TRIGGER_SEPARATE_CONTROL_MATCH_LEVELS,
        pool_interaction_roles=EVENT_TRIGGER_POOL_INTERACTION_ROLES,
        max_events=EVENT_TRIGGER_MAX_EVENTS,
    ),
    use_cache=USE_CACHE,
    force=FORCE_REBUILD_CACHE,
)
tap.plot_event_triggered_state_curve(
    fine_event_curve_summary,
    y_col="mean_speed_mm_s",
    title=f"{fine_bin_label} new-interaction vs no-interaction control: mean speed",
    marker=None,
)
tap.plot_event_triggered_state_curve(
    fine_event_curve_summary,
    y_col="active_fraction",
    title=f"{fine_bin_label} new-interaction vs no-interaction control: activation probability",
    marker=None,
)
display(
    fine_trigger_table.groupby(["trigger_condition", "control_match_level"])["trigger_row_id"]
    .nunique()
    .rename("n_triggers")
    .reset_index()
)
display(fine_event_curve_summary.head(30))


# %%
# Nonparametric transition hazards: how does recent contact history change the
# probability that a quiet bin becomes active in the next bin?
CONTACT_COUNT_COL = "n_interactions_lag_0s_30s"
RECENT_WINDOW_COL = "n_interactions_prev_120s"

contact_hazard_by_side = tap.plot_transition_hazard_by_contact_count(
    transition_design,
    predictor_col=CONTACT_COUNT_COL,
    group_col="side",
    max_count_bin=5,
    title=f"Quiet-to-active hazard by contacts in previous 30 s",
)
contact_hazard_by_cluster = tap.plot_transition_hazard_by_contact_count(
    transition_design,
    predictor_col=RECENT_WINDOW_COL,
    group_col="cluster_id",
    max_count_bin=5,
    title=f"Quiet-to-active hazard by contacts in previous 120 s",
)
time_since_contact_hazard = tap.plot_transition_hazard_by_time_since_contact(
    transition_design,
    group_col="side",
    title="Quiet-to-active hazard by recency of last interaction",
)
display(contact_hazard_by_side)
display(time_since_contact_hazard)


# %%
# Model hierarchy: does actual interaction history add predictive information
# beyond own recent state, time of day, spatial context, and available workers?
own_state_features = [
    "mean_speed_mm_s",
    "quiet_fraction",
    "active_fraction",
    "quiet_run_minutes",
    "hours_since_light_on",
    "time_since_light_on_sin",
    "time_since_light_on_cos",
]
space_context_features = [
    "x_mm",
    "y_mm",
    "mean_n_visible_ants",
    "mean_local_neighbor_count",
]
contact_kernel_features = [
    "n_interactions_lag_0s_30s",
    "n_interactions_lag_30s_120s",
    "n_interactions_lag_120s_600s",
    "time_since_last_interaction_capped_seconds",
    "leaky_interactions_tau_60s",
    "leaky_interactions_tau_300s",
]
contact_role_features = [
    "n_interactions_as_antenna_lag_0s_30s",
    "n_interactions_as_body_lag_0s_30s",
    "n_interactions_as_antenna_lag_30s_120s",
    "n_interactions_as_body_lag_30s_120s",
    "n_interactions_as_antenna_lag_120s_600s",
    "n_interactions_as_body_lag_120s_600s",
    "n_unique_partners_prev_600s",
    "n_partner_clusters_prev_600s",
]
partner_cluster_features = [
    col
    for col in transition_design.columns
    if col.startswith("n_partner_cluster_") and col.endswith("_prev_600s")
]

model_specs = [
    {
        "name": "own_state",
        "predictors": own_state_features,
        "categorical": [],
    },
    {
        "name": "own_state_plus_cluster",
        "predictors": [*own_state_features, "cluster_id", "side"],
        "categorical": ["cluster_id", "side"],
    },
    {
        "name": "own_state_plus_space",
        "predictors": [*own_state_features, "cluster_id", "side", *space_context_features],
        "categorical": ["cluster_id", "side"],
    },
    {
        "name": "plus_contact_kernel",
        "predictors": [
            *own_state_features,
            "cluster_id",
            "side",
            *space_context_features,
            *contact_kernel_features,
        ],
        "categorical": ["cluster_id", "side"],
    },
    {
        "name": "plus_contact_role_and_diversity",
        "predictors": [
            *own_state_features,
            "cluster_id",
            "side",
            *space_context_features,
            *contact_kernel_features,
            *contact_role_features,
        ],
        "categorical": ["cluster_id", "side"],
    },
    {
        "name": "plus_partner_cluster_proxy",
        "predictors": [
            *own_state_features,
            "cluster_id",
            "side",
            *space_context_features,
            *contact_kernel_features,
            *contact_role_features,
            *partner_cluster_features,
        ],
        "categorical": ["cluster_id", "side"],
    },
]

transition_model_table, transition_coef_table = tap.fit_transition_model_hierarchy(
    transition_design,
    model_specs,
    outcome_col="activate_next_bin",
)
tap.plot_transition_model_comparison(
    transition_model_table,
    metric=MODEL_COMPARISON_METRIC,
    title="Quiet-to-active transition model hierarchy",
)

display(transition_model_table)
if not transition_model_table.dropna(subset=[MODEL_COMPARISON_METRIC]).empty:
    best_model_name = transition_model_table.dropna(subset=[MODEL_COMPARISON_METRIC]).iloc[0]["model"]
    display(tap.top_coefficients(transition_coef_table, model=best_model_name, n=30))
display(transition_coef_table.head(60))


# %%
# Useful columns to inspect when deciding which local-history model to refine.
display(
    transition_design[
        [
            "side",
            "track_id",
            "cluster_id",
            "elapsed_time_h",
            "state",
            "next_state",
            "activate_next_bin",
            "mean_speed_mm_s",
            "quiet_run_minutes",
            "n_interactions_lag_0s_30s",
            "n_interactions_lag_30s_120s",
            "n_interactions_lag_120s_600s",
            "n_unique_partners_prev_600s",
            "n_partner_clusters_prev_600s",
            "time_since_last_interaction_capped_seconds",
        ]
    ].head(60)
)

print("Set MAX_CHUNKS=None for a full-block run once these probes look sensible.")
