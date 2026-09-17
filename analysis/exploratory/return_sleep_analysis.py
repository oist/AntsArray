# %%
# Standalone VS Code/Jupyter analysis of 0723 block02 returns and sleeping partners.
# Execute cells in order, or run with MPLBACKEND=Agg to export all figures.
import importlib
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

try:
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        get_ipython().run_line_magic("matplotlib", "qt")  # type: ignore[name-defined]
except Exception:
    pass
try:
    from IPython.display import display
except ImportError:
    display = print

repo_root = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd().resolve()
for candidate in [repo_root, *repo_root.parents]:
    if (candidate / "analysis" / "return_sleep_utils.py").is_file():
        repo_root = candidate
        break
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from analysis import grid_occupancy_utils as go
from analysis import sleep_motion_analysis_utils as sma
from analysis import return_sleep_utils as rs

for module in (sma, rs):
    importlib.reload(module)


# %%
# Settings. These govern event selection/matching; the sleep classifier itself
# is read from the cached metadata and is never silently retuned here.
DATASET_ROOT = Path(os.environ.get("ANTS_DATASET_ROOT", "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260723/block02"))
GRID_ROOT = go.resolve_grid_root(DATASET_ROOT, os.environ.get("ANTS_GRID_OUTPUT_NAME") or None)
if not os.environ.get("ANTS_GRID_OUTPUT_NAME"):
    arena_grid_root = DATASET_ROOT / "stitched" / "grid_occupancy_histograms_arena"
    if (arena_grid_root / "track_cluster_ids.csv").is_file():
        GRID_ROOT = arena_grid_root
CLUSTER_TABLE_PATH = GRID_ROOT / "track_cluster_ids.csv"
SLEEP_LABEL_ROOT = DATASET_ROOT / "stitched" / "sleep_motion_labels"
CACHE_ROOT = DATASET_ROOT / "stitched" / "analysis_cache" / "return_sleep"
OUTPUT_ROOT = DATASET_ROOT / "analysis_outputs" / "return_sleep"
INTERACTION_ROOT = rs.resolve_interaction_root(DATASET_ROOT)
PANORAMA_REGIONS_PATH = go.panorama_regions_path(DATASET_ROOT)
FORCE_REBUILD = False
SLEEP_BIN_SECONDS = 600.0
SLEEP_MIN_CLASSIFIED_FRACTION = 0.50
LIGHT_OFF_HOUR = 19.5
LIGHT_ON_HOUR = 5.5
RESOURCE_VISITS_ONLY = False  # True restricts contact triggers to resource-visit returns.
RUN_SENSITIVITY_COMPARISONS = False
SETTINGS = rs.ReturnSettings(
    min_outside_seconds=30.0,
    min_colony_anchor_seconds=5.0,
    recent_return_seconds=300.0,
    contact_gap_seconds=2.0,
    contact_min_detection_frames=1,  # No additional duration filter on the reviewed distance hits.
    prior_sleep_seconds=10.0,
    wake_sustain_seconds=2.0,
    match_clock_seconds=1800.0,
    match_distance_mm=5.0,
    match_density_difference=2.0,
    match_body_speed_mm_s=0.2,
    match_antenna_speed_mm_s=0.3,
)
SETTINGS.validate()


# %%
# Load existing cluster IDs and sleep labels. All tracked ants contribute to
# density; only ants with saved cluster IDs enter the hypothesis comparisons.
clusters = pd.read_csv(CLUSTER_TABLE_PATH)
all_sleep_tracks = sma.load_sleep_label_tracks(SLEEP_LABEL_ROOT)
sleep_tracks = sma.load_sleep_label_tracks(SLEEP_LABEL_ROOT, clusters)
if not all_sleep_tracks.fps.eq(SETTINGS.fps).all():
    raise ValueError("Analysis FPS differs from cached labels")
start_clock_seconds = go.start_time_from_track_table(sleep_tracks)
grid_tracks = go.load_grid_tracks(GRID_ROOT)
x_split_px = None  # Infer sides from colony-region positions; optional explicit override.
regions = go.load_panorama_regions(PANORAMA_REGIONS_PATH, x_split_px=x_split_px)
chunks, interaction_run = rs.load_published_interaction_chunks(
    INTERACTION_ROOT, DATASET_ROOT / "tracks", fps=SETTINGS.fps,
    start_clock_seconds=start_clock_seconds,
)
interaction_parameters = interaction_run["parameters"]
settings_manifest = {
    "analysis_version": 2, "settings": rs.asdict(SETTINGS),
    "interaction_distance_mm": 0.1, "interaction_root": str(INTERACTION_ROOT),
    "interaction_parameters": interaction_parameters, "interaction_run_id": interaction_run["run_id"],
    "sleep_bin_seconds": SLEEP_BIN_SECONDS, "sleep_min_classified_fraction": SLEEP_MIN_CLASSIFIED_FRACTION,
    "resource_visits_only": RESOURCE_VISITS_ONLY,
    "run_sensitivity_comparisons": RUN_SENSITIVITY_COMPARISONS,
    "clusters": rs.fingerprint(CLUSTER_TABLE_PATH),
    "regions": rs.fingerprint(PANORAMA_REGIONS_PATH),
    "labels": [rs.fingerprint(Path(p)) for p in all_sleep_tracks.metadata_path],
    "contacts": [rs.fingerprint(c.interaction_path) for c in chunks],
    "tracks": [rs.fingerprint(DATASET_ROOT / "stitched" / "per_track" / name) for name in all_sleep_tracks.track_name],
    "sleep_classifier_parameters": json.loads(sleep_tracks.classifier_parameters.iloc[0]),
    "source_code": [rs.fingerprint(repo_root / "analysis" / filename) for filename in (
        "return_sleep_utils.py", "sleep_motion_analysis_utils.py", "return_sleep_report.py",
        "exploratory/return_sleep_analysis.py",
    )],
}
run_key = rs.cache_key(settings_manifest)
run_root = OUTPUT_ROOT / run_key
figure_root = run_root / "figures"
figure_root.mkdir(parents=True, exist_ok=True)
(run_root / "settings.json").write_text(json.dumps(settings_manifest, indent=2) + "\n")
figure_paths = []


def export_figure(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(figure_root / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    figure_paths.append(figure_root / f"{name}.png")
    if os.environ.get("MPLBACKEND", "").lower() == "agg":
        plt.close(fig)
    else:
        plt.show(block=False)


def export_table(table, name):
    table.to_parquet(run_root / f"{name}.parquet", index=False)
    if len(table) < 100_000:
        table.to_csv(run_root / f"{name}.csv", index=False)


print(f"{len(sleep_tracks)} clustered ants; {len(all_sleep_tracks)} ants for local density; {len(chunks)} interaction chunks")
print(f"Recording starts {go.format_clock_time(start_clock_seconds)}; output: {run_root}")
display(sleep_tracks.groupby(["side", "cluster_id"]).size().rename("n_ants"))


# %%
# Mean of per-ant sleep fractions through the recording, plus individual rows.
cluster_sleep, ant_sleep_bins = sma.cluster_sleep_timeseries(
    sleep_tracks, bin_seconds=SLEEP_BIN_SECONDS,
    min_classified_fraction=SLEEP_MIN_CLASSIFIED_FRACTION,
)
export_table(cluster_sleep, "cluster_sleep_timeseries")
export_table(ant_sleep_bins, "ant_sleep_time_bins")
export_figure(sma.plot_cluster_sleep_timeseries(
    cluster_sleep, start_clock_seconds=start_clock_seconds,
    light_off_hour=LIGHT_OFF_HOUR, light_on_hour=LIGHT_ON_HOUR), "01_cluster_sleep")
export_figure(sma.plot_ant_sleep_heatmap(ant_sleep_bins, start_clock_seconds=start_clock_seconds), "02_ant_sleep")


# %%
# Corrected position context comes from current panorama annotations. Existing
# colony_presence_vectors use legacy boxes and are not inputs to this analysis.
contexts = rs.load_contexts(all_sleep_tracks, DATASET_ROOT, regions, SETTINGS, CACHE_ROOT, force=FORCE_REBUILD)
returns = rs.extract_returns(contexts, clusters, SETTINGS)
export_table(returns, "returns")
print(f"Observed returns: {len(returns)}")
display(returns.groupby(["side", "resource_visit"]).size().rename("n_returns"))
fig, return_examples = rs.plot_return_examples(returns, contexts, regions, SETTINGS)
export_figure(fig, "03_return_geometry")
export_table(return_examples, "return_examples")


# %%
# Undirected skeleton-distance contacts continue across chunk borders.
# A new onset requires >2 s since the pair's previous detection. Left-censored
# contacts at an observation boundary remain available for control exclusion.
contact_bouts, interaction_coverage = rs.load_contact_bouts(chunks, SETTINGS, CACHE_ROOT, force=FORCE_REBUILD)
contact_audit = contact_bouts.groupby("side").agg(
    pair_bouts=("start_frame", "size"), new_onsets=("is_new_onset", "sum"),
    single_frame_bouts=("n_detection_frames", lambda values: int(values.eq(1).sum())),
    fewer_than_three_frames=("n_detection_frames", lambda values: int(values.lt(3).sum())),
).reset_index()
export_table(contact_audit, "contact_audit")
display(contact_audit)
focal_contacts = rs.attach_contact_context(contexts, contact_bouts, interaction_coverage, SETTINGS)
return_curves, return_sleep_latency = rs.return_activity_curves(returns, contexts, SETTINGS)
return_effects = rs.return_early_late_effects(return_curves, random_state=SETTINGS.random_state)
export_table(return_curves, "return_activity_rows")
export_table(return_sleep_latency, "return_sleep_latency")
export_table(return_effects, "return_early_minus_late_effects")
fig, return_curve_summary = rs.plot_return_curves(return_curves, random_state=SETTINGS.random_state)
export_figure(fig, "04_return_activity")
export_table(return_curve_summary, "return_activity_summary")
export_figure(rs.plot_survival(return_sleep_latency, event_column="sleep_observed", group_column="trip_type",
    title="First sustained sleep after return; censored at exit or lost labels", ylabel="Cumulative probability of sleep",
    horizon=600), "05_return_to_sleep")
display(return_effects)


# %%
# Select recipients asleep for the 10 s BEFORE a new contact. Returning partners
# must still be inside the colony within 5 min of a qualifying observed return.
sleeping_contacts = rs.eligible_sleeping_contacts(sleep_tracks, contexts, contact_bouts, returns, SETTINGS)
triggers, matching_diagnostics = rs.match_sleeping_controls(
    sleeping_contacts, contexts, SETTINGS, resource_only=RESOURCE_VISITS_ONLY,
)
export_table(sleeping_contacts, "eligible_sleeping_contacts")
export_table(triggers, "matched_triggers")
export_table(matching_diagnostics, "matching_diagnostics")
return_contacts = rs.return_contact_summary(returns, focal_contacts, sleeping_contacts, SETTINGS)
export_table(return_contacts, "return_contact_summary")
display(sleeping_contacts.groupby(["side", "condition"]).size().rename("n_eligible_contacts"))
display(triggers.groupby(["side", "condition"]).size().rename("n_matched_triggers"))
fig, matching_balance = rs.plot_matching_diagnostics(triggers, matching_diagnostics)
export_figure(fig, "06_matching_balance")
export_table(matching_balance, "matching_balance")


# %%
# Label and motion traces show the full observed response. The separate wake
# analysis censors both contacts and controls at the next new recipient contact,
# an unknown label, or exit/loss of colony-position coverage.
recipient_curves = rs.trigger_state_curves(triggers, sleep_tracks, contexts, SETTINGS)
export_table(recipient_curves, "recipient_curve_rows")
fig, recipient_summary = rs.plot_recipient_curves(recipient_curves, random_state=SETTINGS.random_state)
export_figure(fig, "07_sleeping_recipient_response")
export_table(recipient_summary, "recipient_curve_summary")
wake_outcomes = rs.waking_outcomes(triggers, sleep_tracks, contexts, focal_contacts, SETTINGS)
wake_effects = rs.paired_wake_effects(wake_outcomes, random_state=SETTINGS.random_state)
export_table(wake_outcomes, "wake_outcomes")
export_table(wake_effects, "paired_wake_effects")
export_figure(rs.plot_survival(wake_outcomes, event_column="wake_observed", group_column="condition",
    title="Wake after a new contact; censored at subsequent contact, exit, or lost labels",
    ylabel="Cumulative probability of sustained wake", horizon=SETTINGS.wake_followup_seconds), "08_wake_followup")
export_figure(rs.plot_paired_wake_effects(wake_effects), "09_paired_wake_effects")
display(wake_effects)
display(wake_outcomes.groupby(["side", "condition", "censor_reason"]).size().rename("n_events"))


# %%
# Directional sensitivity applies only to legacy directed inputs, not skeleton distances.
sensitivity_counts = []
sensitivity_effect_tables = []
if RUN_SENSITIVITY_COMPARISONS and sleeping_contacts.recipient_body_contact.notna().any():
    for name, resource_only, body_only in (
        ("returner_antenna_to_sleeping_ant", False, True),
    ):
        selected_contacts = sleeping_contacts
        if body_only:
            selected_contacts = selected_contacts[selected_contacts.recipient_body_contact]
        selected_triggers, selected_diagnostics = rs.match_sleeping_controls(
            selected_contacts, contexts, SETTINGS, resource_only=resource_only,
        )
        selected_outcomes = rs.waking_outcomes(selected_triggers, sleep_tracks, contexts, focal_contacts, SETTINGS)
        selected_effects = rs.paired_wake_effects(selected_outcomes, random_state=SETTINGS.random_state)
        selected_curves = rs.trigger_state_curves(selected_triggers, sleep_tracks, contexts, SETTINGS)
        export_table(selected_triggers, name + "_triggers")
        export_table(selected_diagnostics, name + "_matching")
        export_table(selected_outcomes, name + "_wake_outcomes")
        export_table(selected_effects, name + "_paired_effects")
        fig, selected_summary = rs.plot_recipient_curves(selected_curves, random_state=SETTINGS.random_state)
        fig.suptitle(name.replace("_", " ").capitalize() + ": sleeping-recipient response")
        export_figure(fig, "10_" + name)
        export_table(selected_summary, name + "_response_summary")
        sensitivity_effect_tables.append(selected_effects.assign(analysis=name))
        for side in ("left", "right"):
            group = selected_triggers[(selected_triggers.side == side) & (selected_triggers.condition == "Recent return contact")]
            sensitivity_counts.append({"analysis": name, "side": side, "matched_pairs": len(group),
                                       "recipient_ants": group.track_id.nunique()})
sensitivity_counts = pd.DataFrame(sensitivity_counts)
sensitivity_effects = pd.concat(sensitivity_effect_tables, ignore_index=True) if sensitivity_effect_tables else pd.DataFrame()
export_table(sensitivity_counts, "sensitivity_counts")
export_table(sensitivity_effects, "sensitivity_wake_effects")
display(sensitivity_counts)


# %%
# Portable figure index and methods/data-quality notes alongside the tables.
from analysis.return_sleep_report import write_report

write_report(run_root, figure_paths, settings_manifest, sleep_tracks, returns, return_effects,
             sleeping_contacts, triggers, matching_diagnostics, wake_outcomes, wake_effects,
             start_clock_seconds=start_clock_seconds, sensitivity_counts=sensitivity_counts,
             sensitivity_effects=sensitivity_effects)
print(f"Figures and event tables: {run_root}")
print(f"Open figure index: {run_root / 'index.html'}")
