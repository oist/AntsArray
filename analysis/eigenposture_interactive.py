# %% 1. Settings and inputs
"""Right colony: two-second behavior states -> ant classes -> occupancy analyses.

Run the numbered cells in VS Code/Jupyter. All classification steps are below;
repository utilities only handle established cache formats and downstream plots.
Start from the measured fine_behavior_20261010 packet (not raw videos).
"""
from pathlib import Path
import os
import sys
import json
import warnings

repo = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
import matplotlib
if os.environ.get("MPLBACKEND", "").lower() != "agg":
    try:
        get_ipython().run_line_magic("matplotlib", "qt")
    except (NameError, ImportError):
        pass
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from umap import UMAP
from analysis import grid_occupancy_utils as go
from analysis import sleep_motion_analysis_utils as sma
from analysis import return_sleep_utils as rs

BLOCK = Path(os.environ.get("ANTS_DATASET_ROOT", "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260724/block01"))
SOURCE = BLOCK / "analysis_outputs/fine_behavior_20261010"
OUTPUT = Path(os.environ.get("ANTS_ROLE_OUTPUT", str(BLOCK / "analysis_outputs/right_behavior_classes_20261010")))
SEED = 7241011
MIN_WINDOWS_PER_HOUR, MIN_HOURS = 20, 24
BOOTSTRAPS = 50
LIGHT_ON, LIGHT_OFF = 5.5, 19.5
threadpool_limits(limits=4)
OUTPUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def show(name):
    """Save every new figure from this cell, then display it without blocking."""
    for number in plt.get_fignums():
        figure = plt.figure(number)
        if not getattr(figure, "_right_saved", False):
            figure.savefig(OUTPUT / f"{name}_{number:02d}.png", dpi=160, bbox_inches="tight")
            figure._right_saved = True
    plt.show(block=False)


inventory = pd.read_csv(SOURCE / "inputs/all_ant_coverage.csv")
with np.load(SOURCE / "short_models/states.npz") as archive:
    measured = dict(archive)
right_ids = np.flatnonzero(inventory.side.eq("right"))
ants = inventory.iloc[right_ids].reset_index(drop=True)
rows = np.flatnonzero(np.isin(measured["ant"], right_ids))
ant_index = np.searchsorted(right_ids, measured["ant"][rows])
hour = measured["hour"][rows].astype(int)
print(f"{len(ants)} right-colony identities; {len(rows):,} measured windows", flush=True)

# %% 2. Inspect the features and balance the three feature families
# Measurement choices: random 60-s segment/ant/hour over all 48 hours, 12 Hz;
# 2-s windows every 0.25 s. Gaps/camera switches invalidate a window. Consecutive
# windows overlap: window count is NOT a count of independent observations.
# PC1-4 use the posture basis fitted across the full 48-h recording. Four
# head/gaster angle coordinates retain articulation. Body velocities are unsigned
# forward/lateral maxima; both antennal velocities are head-relative means.
# Contacts are merged-bout context (<=2-s gaps), not frame-exact physical contact.
# The selected minimal representation excludes wavelets/PC rates: the longer
# wavelet support lost fast, fragmented tracks. This keeps the approved 8 states.
names = ["PC1", "PC2", "PC3", "PC4", "head_cos", "head_sin", "gaster_cos", "gaster_sin",
         "forward_peak", "lateral_peak", "antenna_A_mean", "antenna_B_mean", "turn_mean",
         "mean_contact_partners", "contact_bout_fraction", "new_contact_bouts"]
raw = measured["raw"].astype(float)
transformed = raw.copy()
transformed[:, 8:13] = np.log1p(transformed[:, 8:13] / .1)
transformed[:, [13, 15]] = np.log1p(transformed[:, [13, 15]])
dictionary = measured["dictionary"]
# Each FAMILY has unit total variance on training ants; no extra PCA or spatial
# coordinate enters the distance. Keep all 16 coordinates, including quiet motion.
x = np.empty_like(transformed, dtype=np.float32)
for columns in (slice(0, 8), slice(8, 13), slice(13, 16)):
    training = transformed[dictionary, columns]
    center = training.mean(axis=0)
    scale = max(np.sqrt(training.var(axis=0).sum()), 1e-6)
    x[:, columns] = (transformed[:, columns] - center) / scale
# Test: rebuilding the normalization must reproduce the validated feature space.
np.testing.assert_allclose(x, measured["x"], atol=2e-6)
assert np.isfinite(x).all() and (raw[:, 8:] >= 0).all()

# %% 3. Assign two-second states; UMAP is only a visualization
# The shared dictionary was fitted without spatial/sleep labels, on training
# ants only. 30-neighbor fuzzy graph, Leiden resolution .5, full convergence.
# Eight is a working vocabulary, not proof of eight discrete natural behaviors.
# In particular S4 was unstable under removal of training ants; inspect it cautiously.
dictionary_state = measured["dictionary_state"].copy()
n_states = int(dictionary_state.max() + 1)
distance, neighbor = NearestNeighbors(n_neighbors=5, n_jobs=4).fit(x[dictionary]).kneighbors(x[rows])
scores = np.zeros((len(rows), n_states))
for j in range(5):
    np.add.at(scores, (np.arange(len(rows)), dictionary_state[neighbor[:, j]]), 1 / np.maximum(distance[:, j], 1e-6))
state = scores.argmax(axis=1)
confidence = scores.max(axis=1) / scores.sum(axis=1)
# Exact graph membership takes precedence for dictionary nodes.
common, local_row, dictionary_row = np.intersect1d(rows, dictionary, return_indices=True)
state[local_row], confidence[local_row] = dictionary_state[dictionary_row], 1
np.testing.assert_array_equal(state, measured["state"][rows])
state_names = [f"S{s}" + ("*" if s == 4 else "") for s in range(n_states)]
colors = plt.get_cmap("tab20")(np.arange(n_states) % 20)
rng = np.random.default_rng(SEED)
sample = np.sort(rng.choice(len(rows), min(6000, len(rows)), replace=False))
xy = UMAP(n_neighbors=30, min_dist=.05, random_state=SEED, n_jobs=1).fit_transform(x[rows[sample]])
fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
for s in range(n_states):
    selected = state[sample] == s
    axes[0].scatter(*xy[selected].T, s=3, color=colors[s], label=state_names[s], rasterized=True)
axes[0].legend(markerscale=3, ncol=2, fontsize=8)
axes[0].set(title="1. Two-second behavior states (UMAP view)", xlabel="UMAP 1", ylabel="UMAP 2")
profiles = np.array([np.median(x[rows][state == s], axis=0) for s in range(n_states)])
im = axes[1].imshow(profiles, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
axes[1].set(yticks=np.arange(n_states), yticklabels=state_names, xticks=np.arange(16),
            xticklabels=names, title="State medians in the balanced feature space")
axes[1].tick_params(axis="x", rotation=90)
fig.colorbar(im, ax=axes[1], label="Centered / family scale")
show("01_states")
pd.DataFrame(profiles, index=state_names, columns=names).to_csv(OUTPUT / "state_profiles.csv")

# %% 4. From states to ants: average observed-hour proportions, not day totals
# Equal hour weights prevent a well-tracked hour dominating the ant's profile.
# 20 windows/hour and >=24 qualifying hours are coverage rules, not 24 h of
# continuous tracking: only one minute per hour was sampled. Missing stays NaN.
def hourly_proportions(ant, hours, labels, n_ants, k, minimum=20):
    counts = np.zeros((n_ants, 48, k), dtype=int)
    np.add.at(counts, (ant, hours, labels), 1)
    total = counts.sum(axis=2, keepdims=True)
    return np.divide(counts, total, out=np.full(counts.shape, np.nan), where=total >= minimum)


hourly = hourly_proportions(ant_index, hour, state, len(ants), n_states, MIN_WINDOWS_PER_HOUR)
valid_hours = np.isfinite(hourly).all(axis=2).sum(axis=1)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    proportions = np.nanmean(hourly, axis=1)
eligible = valid_hours >= MIN_HOURS
ids = np.flatnonzero(eligible)
# Square root gives Hellinger geometry for compositions; keep ALL state axes.
role_x = np.sqrt(proportions[ids])
np.testing.assert_allclose(proportions[ids].sum(axis=1), 1)
assert len(ids) >= 10, "Too few ants with adequate hourly coverage"

# %% 5. Justify the number of ant classes without choosing it spatially
# Tied covariance limits parameters for this small ant cohort. Choose K=1..4
# using BIC AND resampling/temporal stability, >=5 ants per class. Within 2 BIC
# prefer smaller K. K=1 is the fallback; two classes are not imposed.
def fit_roles(values, k, seed=SEED):
    model = GaussianMixture(k, covariance_type="tied", reg_covar=.001,
                            n_init=10, max_iter=500, random_state=seed).fit(values)
    assert model.converged_, "Role model did not converge"
    return model


fits, results = {}, []
for k in range(1, 5):
    fit = fits[k] = fit_roles(role_x, k)
    labels = fit.predict(role_x)
    bootstrap, temporal = [], []
    rng = np.random.default_rng(SEED)
    if k > 1:
        for repeat in range(BOOTSTRAPS):
            draws = rng.integers(len(ids), size=len(ids))
            predicted = fit_roles(role_x[draws], k, SEED + repeat).predict(role_x)
            bootstrap.append(adjusted_rand_score(labels, predicted))
        for width in (1, 2, 4):
            halves = []
            for half in (0, 1):
                mask = (np.arange(48) // width) % 2 == half
                values = np.sqrt(np.nanmean(hourly[ids][:, mask], axis=1))
                halves.append(fit_roles(values, k).predict(values))
            temporal.append(adjusted_rand_score(*halves))
    results.append(dict(k=k, bic=fit.bic(role_x), minimum_ants=int(np.bincount(labels, minlength=k).min()),
                        bootstrap_median=np.median(bootstrap) if bootstrap else np.nan,
                        temporal_median=np.median(temporal) if temporal else np.nan))
k_table = pd.DataFrame(results).set_index("k")
k_table["qualifies"] = ((k_table.bic < k_table.loc[1, "bic"]) & (k_table.minimum_ants >= 5)
                        & (k_table.bootstrap_median >= .8) & (k_table.temporal_median >= .6))
passing = k_table[k_table.qualifies]
k_roles = 1 if passing.empty else int(passing[passing.bic <= passing.bic.min() + 2].index.min())
ants["valid_hours"], ants["eligible"] = valid_hours, eligible
ants["ant_class"] = -1  # -1 means unclassified, never a biological class.
ants.loc[eligible, "ant_class"] = fits[k_roles].predict(role_x)
print(k_table.to_string(), "\nSelected classes:", ants[eligible].ant_class.value_counts().sort_index().to_dict(), flush=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
axes[0].plot(k_table.index, k_table.bic - k_table.loc[1, "bic"], "o-")
axes[0].scatter([k_roles], [k_table.loc[k_roles, "bic"] - k_table.loc[1, "bic"]], s=120, facecolors="none", edgecolors="red")
axes[0].set(xlabel="K ant classes", ylabel="BIC minus K=1", xticks=k_table.index, title=f"2. Selected K={k_roles}")
k_table[["bootstrap_median", "temporal_median"]].plot(ax=axes[1], marker="o")
axes[1].axhline(.8, color="C0", ls=":"); axes[1].axhline(.6, color="C1", ls=":")
axes[1].set(ylabel="Adjusted Rand index", ylim=(0, 1.05), title="Ant resampling and alternating hours")
show("02_class_selection")

# %% 6. Class composition and state changes within sampled segments
ordered = ants[eligible].sort_values(["ant_class", "track_id"]).index.to_numpy()
fig, axes = plt.subplots(1, 2, figsize=(14, 9), layout="constrained")
im = axes[0].imshow(proportions[ordered], aspect="auto", vmin=0,
                    vmax=np.ceil(proportions[ordered].max() * 10) / 10, cmap="viridis")
axes[0].set(xticks=np.arange(n_states), xticklabels=state_names, yticks=np.arange(len(ordered)),
            yticklabels=[f"C{ants.loc[a, 'ant_class']} / {ants.loc[a, 'track_id']:03d}" for a in ordered],
            title="3. Ant classes from state proportions")
axes[0].tick_params(axis="y", labelsize=7)
fig.colorbar(im, ax=axes[0], label="Mean observed-hour fraction")
# Show the hour with the most observed eligible ants; gaps remain gray. Each
# ant's randomly sampled minute starts at a DIFFERENT time within that hour.
example_hour = int(np.isfinite(hourly[ids]).all(axis=2).sum(axis=0).argmax())
ethogram = np.full((len(ants), 240), np.nan)
chosen = hour == example_hour
ethogram[ant_index[chosen], measured["sample"][rows][chosen] // 3] = state[chosen]
cmap = ListedColormap(colors); cmap.set_bad(".85")
im = axes[1].imshow(ethogram[ordered], aspect="auto", interpolation="nearest", cmap=cmap,
                   norm=BoundaryNorm(np.arange(n_states + 1) - .5, n_states), extent=(0, 60, len(ordered), 0))
axes[1].set(xlabel="Seconds since each ant's sampled segment", yticks=[],
            title=f"States within sampled minute, hour {example_hour}\nGray = unobserved; rows match left panel")
fig.colorbar(im, ax=axes[1], ticks=np.arange(n_states), label="State").set_ticklabels(state_names)
for boundary in np.flatnonzero(np.diff(ants.loc[ordered, "ant_class"])) + 1:
    axes[0].axhline(boundary - .5, color="white", lw=1)
    axes[1].axhline(boundary, color="white", lw=1)
show("03_ant_classes")
ants.to_csv(OUTPUT / "ant_classes.csv", index=False)
k_table.to_csv(OUTPUT / "class_k_selection.csv")
pd.DataFrame(proportions, index=ants.ant, columns=state_names).to_csv(OUTPUT / "ant_state_proportions.csv")
np.savez_compressed(OUTPUT / "hourly_state_proportions.npz", ants=ants.ant.to_numpy(str), proportions=hourly)
pd.DataFrame(dict(ant=ants.ant.to_numpy()[ant_index], frame=measured["frame"][rows], hour=hour,
                  state=state, confidence=confidence)).to_csv(OUTPUT / "window_states.csv.gz", index=False)

# %% 7. Hand the fixed behavioral classes to the grid-occupancy utilities
# This is the FIRST read of spatial information. Require an exact identity AND
# filename match for every eligible ant; do not silently drop or relabel ants.
# The utilities call the numerical class column 'leiden_cluster' for historical
# reasons. Here it is our GMM ANT CLASS, not a spatial or short-window cluster.
stitched = BLOCK / "stitched"
grid_root = stitched / "grid_occupancy_histograms_arena"
grid = go.load_grid_tracks(grid_root)
cluster_table = ants[eligible][["side", "track_id", "track_name", "ant_class"]].merge(
    grid, on=["side", "track_id", "track_name"], how="left", validate="one_to_one", indicator=True)
assert cluster_table._merge.eq("both").all(), "Missing grid cache for a classified ant"
cluster_table = cluster_table.drop(columns="_merge").assign(leiden_cluster=lambda d: d.ant_class,
    cluster_id=lambda d: "right_behavior_" + d.ant_class.astype(str))
cluster_ids = cluster_table[["side", "track_id", "track_name", "cluster_id", "leiden_cluster"]]
cluster_ids.to_csv(OUTPUT / "behavior_class_ids.csv", index=False)
# Never overwrite the spatial track_cluster_ids.csv. Reference labels are only
# a post hoc agreement check and cannot alter states, K, or eligibility.
reference = pd.read_csv(SOURCE / "inputs/spatial_reference.csv")
comparison = ants[eligible].merge(reference[["side", "track_id", "spatial_cluster"]],
                                  on=["side", "track_id"], how="left", validate="one_to_one")
observed = comparison.dropna(subset="spatial_cluster")
spatial_ari = adjusted_rand_score(observed.ant_class, observed.spatial_cluster)
print(f"Post hoc spatial agreement: ARI={spatial_ari:.3f}, {len(observed)}/{len(ids)} ants", flush=True)
comparison.to_csv(OUTPUT / "posthoc_spatial_comparison.csv", index=False)
go.plot_cluster_mean_histograms(grid, cluster_table, title="4. Occupancy emerging from behavioral ant classes")
go.plot_cluster_example_histograms(grid, cluster_table, n_examples=4, title="5. Individual occupancy, grouped by behavioral class")
regions = go.load_panorama_regions(go.panorama_regions_path(BLOCK), x_split_px=None)
region_use = go.compute_region_occupancy(cluster_table, regions)
colony_use = go.summarize_colony_use(region_use)
go.plot_cluster_colony_use(colony_use)
plt.gcf().suptitle("6. Nest occupancy by behavioral class (post hoc)")
colony_use.to_csv(OUTPUT / "ant_colony_use.csv", index=False)
show("04_occupancy")

# %% 8. Full-recording speed and cached sleep by behavioral class
# Downstream plots use the full recording, as in grid_occupancy.py. Behavioral
# classes were fitted only on sampled windows within its central 48 hours.
# Equal ant weights; unknown sleep/velocity is missing, never zero. Motion-defined
# sleep is an interpretation check, not an independent physiological validation.
context = go.recording_context(BLOCK, grid)
clock = context["start_clock_seconds"]
start, stop = context["frame_min"], context["frame_stop"]
speed, ant_speed = go.plot_cluster_speed_timeseries(cluster_table, stitched / "speed_vectors",
    bin_seconds=600, smooth_seconds=600, start_clock_seconds=clock,
    light_off_hour=LIGHT_OFF, light_on_hour=LIGHT_ON, title="7. Speed by behavioral ant class")
sleep_tracks = sma.load_sleep_label_tracks(stitched / "sleep_motion_labels", cluster_ids)
sleep, ant_sleep = sma.cluster_sleep_timeseries(sleep_tracks, bin_seconds=600,
    min_classified_fraction=.5, recording_start_frame=start, recording_stop_frame=stop)
sma.plot_cluster_sleep_timeseries(sleep, start_clock_seconds=clock).suptitle("8. Sleep by behavioral ant class")
sma.plot_ant_sleep_heatmap(ant_sleep, start_clock_seconds=clock).suptitle("9. Individual sleep, ordered by behavioral class")
speed.to_csv(OUTPUT / "class_speed.csv", index=False)
sleep.to_csv(OUTPUT / "class_sleep.csv", index=False)
ant_sleep.to_parquet(OUTPUT / "ant_sleep_bins.parquet", index=False)
show("05_speed_sleep")

# %% 9. Individual activity/sleep versus time of day
# Use complete 05:30-to-05:30 cycles only when available, not partially observed
# days disguised as repeat cycles. This dataset has limited independent days.
fps = float(sleep_tracks.fps.iloc[0])
offset = LIGHT_ON * 3600 - clock
complete_cycles = max(0, int(np.floor((stop / fps - offset) / 86400) - np.ceil((start / fps - offset) / 86400)))
cycle_bins, clock_profiles, clock_audit = sma.compute_activity_sleep_clock_profiles(
    cluster_ids, go.load_speed_tracks(stitched / "speed_vectors"), sleep_tracks, fps=fps,
    recording_start_frame=start, recording_stop_frame=stop, start_clock_seconds=clock,
    light_on_hour=LIGHT_ON, light_off_hour=LIGHT_OFF, bin_minutes=30, min_bin_coverage=.5,
    min_cycles=min(2, max(1, complete_cycles)), include_partial_cycles=complete_cycles == 0,
    recording_date=context["recording_date"], max_workers=4)
sma.plot_activity_sleep_clock_matrices(cycle_bins, clock_profiles)
clock_profiles.to_csv(OUTPUT / "ant_clock_profiles.csv", index=False)
clock_audit.to_csv(OUTPUT / "clock_coverage.csv", index=False)
show("06_clock_profiles")

# %% 10. Returns and sleeping-recipient responses, retaining the same classes
# Same definitions as grid_occupancy: >=30 s outside, >=5 s colony anchors,
# observed entry crossing within 5 frames; new skeleton contacts <=0.1 mm.
# Retain ALL right ants as social context, including ants without a role. Their
# returns can expose classified recipients, without calling those ants a class.
settings = rs.ReturnSettings()
all_sleep = sma.load_sleep_label_tracks(stitched / "sleep_motion_labels")
all_sleep = all_sleep[all_sleep.side.eq("right")].copy()
cache = stitched / "analysis_cache/return_sleep"
chunks, interaction_run = rs.load_published_interaction_chunks(rs.resolve_interaction_root(BLOCK),
    BLOCK / "tracks", fps=settings.fps, start_clock_seconds=clock)
contexts = rs.load_contexts(all_sleep, BLOCK, regions, settings, cache, force=False)
bouts, coverage = rs.load_contact_bouts(chunks, settings, cache, force=False)
rs.attach_contact_context(contexts, bouts, coverage, settings)
all_classes = all_sleep[["side", "track_id"]].merge(cluster_ids[["side", "track_id", "cluster_id"]],
    on=["side", "track_id"], how="left", validate="one_to_one").fillna({"cluster_id": "unclassified_context"})
all_returns = rs.extract_returns(contexts, all_classes, settings)
returns = all_returns[all_returns.cluster_id.isin(cluster_ids.cluster_id)].copy()
return_curves, latency = rs.return_activity_curves(returns, contexts, settings)
contacts = rs.eligible_sleeping_contacts(sleep_tracks, contexts, bouts, all_returns, settings)
triggers, matching = rs.match_sleeping_controls(contacts, contexts, settings)
recipient_curves = rs.trigger_state_curves(triggers, sleep_tracks, contexts, settings)
# The canonical plots pool classes; explicitly subset here so class membership
# survives all the way to the return and recipient comparisons.
for label in sorted(cluster_ids.cluster_id.unique()):
    fig, summary = rs.plot_return_curves(return_curves[return_curves.cluster_id.eq(label)], side="right")
    fig.suptitle(f"Returns: {label} (equal-ant mean; 95% ant bootstrap)")
    summary.to_csv(OUTPUT / f"{label}_return_summary.csv", index=False)
    fig, summary = rs.plot_recipient_curves(recipient_curves[recipient_curves.cluster_id.eq(label)], side="right", xlim=(-30, 30))
    fig.suptitle(f"Sleeping recipients: {label} (descriptive matched comparison)")
    summary.to_csv(OUTPUT / f"{label}_recipient_summary.csv", index=False)
returns.to_csv(OUTPUT / "returns.csv", index=False)
triggers.to_csv(OUTPUT / "matched_triggers.csv", index=False)
matching.to_csv(OUTPUT / "matching_diagnostics.csv", index=False)
show("07_return_response")
(OUTPUT / "summary.json").write_text(json.dumps(dict(source=str(SOURCE), dictionary="validated shared eight-state model",
    n_right_ants=len(ants), n_eligible=len(ids), n_states=n_states, n_classes=k_roles,
    class_sizes=ants[eligible].ant_class.value_counts().sort_index().to_dict(), spatial_ari=spatial_ari,
    minimum_windows_per_hour=MIN_WINDOWS_PER_HOUR, minimum_observed_hours=MIN_HOURS,
    bootstrap_repeats=BOOTSTRAPS, seed=SEED, complete_clock_cycles=complete_cycles), indent=2) + "\n")
print(f"Finished. Figures and tables: {OUTPUT}", flush=True)
