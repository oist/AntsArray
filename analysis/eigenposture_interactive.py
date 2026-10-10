# %% 1. Settings — run cells in VS Code/Spyder, or: python -i analysis/eigenposture_interactive.py
"""Inspect the July 24 posture/velocity analysis, one calculation at a time.

Requires numpy, pandas, matplotlib, scipy and scikit-learn. No repo imports.
Change settings and rerun cells in order. All arrays remain available.

Tracking, body alignment and clip QC have already been done. One observation per
minute summarizes a sampled 2.5-second clip, NOT a full minute. Five-minute bins
summarize these observations. No explicit posture-PC derivatives are used.
The landmark PCA uses both recorded days. Coordinate means are recovered from
all 12 archived modes (up to cache rounding) and projected onto this new basis.
Each clustering row is one ant's mean posture and signed velocity across bins.
Four posture PCs plus forward/lateral velocity give exactly six features.
Clustering uses all six balanced features, with no second PCA/projection.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

DATA = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260724/block01/analysis_outputs/eigenposture_20261009"
)
SIDE = "right"  # 'left' or 'right'; fits are independent by colony
BIN_MINUTES = 5  # try 1, 15, 30, 60, 120, 240 (must divide 1440)
POSTURE_PCS = 4  # 1–12; all coordinate modes are available
SCALING = "family"  # 'family' or 'standard'
COVARIANCE = "tied"  # shared full covariance in balanced feature space
BOOTSTRAPS = 500  # use 30 for a quick preview
SEED = 7241010
COLORS = np.array(["#237b9c", "#e18132", "#8867ab", "#a34d52"])

plt.ion()  # figures stay interactive; no files are overwritten
threadpool_limits(limits=1)  # avoid slow BLAS oversubscription on these tiny fits
assert SIDE in ("left", "right") and 1440 % BIN_MINUTES == 0
assert 1 <= POSTURE_PCS <= 12 and SCALING in ("family", "standard")

# %% 2. Load coordinate means/velocities; inspect observation coverage
coverage = pd.read_csv(DATA / "all_ant_coverage.csv")
# The old four-mode cache is insufficient for changing the landmark PCA basis.
# All 12 modes are an invertible coordinate representation, not a truncation.
with np.load(DATA / "full_rank" / "eigenposture_measurements.npz") as cache:
    np.testing.assert_array_equal(cache["ants"], coverage.ant)
    cached = cache["minute"]
    all_velocity = cached[:, :, :2]  # signed forward/lateral means; no RMS
    old_posture_scores = cached[:, :, 4:16]
with np.load(DATA / "full_rank" / "landmark_pca.npz") as old_basis:
    coordinate_means = old_posture_scores @ old_basis["components"] + old_basis["mean"]
del cached, old_posture_scores
# Eligibility no longer depends on the discarded PC-rate channels.
for day in (1, 2):
    coverage[f"day{day}_eligible"] = (coverage[f"day{day}_posture_hours"] >= 12) & (
        coverage[f"day{day}_velocity_hours"] >= 12
    )
print(coverage.groupby("side")[["day1_eligible", "day2_eligible"]].sum())

# %% 3. Landmark PCA over the FULL 48 hours, with equal colony and ant weights
# Six antennal landmarks × (forward, lateral), relative to the head, normalized
# by a fixed per-ant body-axis length. Use exact saved moments of valid windows,
# not covariance of minute means (which would discard within-clip variation).
with np.load(DATA / "ant_coordinate_moments.npz") as moments:
    np.testing.assert_array_equal(moments["ants"], coverage.ant)
    day_indices = [list(moments["versions"]).index(day) for day in ("day1", "day2")]
    daily_first = moments["means"][:, day_indices]
    daily_second = moments["seconds"][:, day_indices]
hours = coverage[["day1_posture_hours", "day2_posture_hours"]].to_numpy(float)
hours[~np.isfinite(daily_first).all(axis=2)] = 0
# Day moments already weight observed hours/minutes/windows equally. Weight each
# day by its observed hours, so one recorded hour has the same weight on either day.
weights = hours / np.maximum(hours.sum(axis=1, keepdims=True), 1)
first = np.sum(np.nan_to_num(daily_first) * weights[:, :, None], axis=1)
second = np.sum(np.nan_to_num(daily_second) * weights[:, :, None, None], axis=1)
pca_eligible = hours.sum(axis=1) >= 24  # at least half of the full recording
colony_indices = [
    np.flatnonzero(coverage.side.eq(side) & pca_eligible) for side in ("left", "right")
]
coordinate_center = np.mean([first[i].mean(axis=0) for i in colony_indices], axis=0)
coordinate_second = np.mean([second[i].mean(axis=0) for i in colony_indices], axis=0)
coordinate_covariance = coordinate_second - np.outer(
    coordinate_center, coordinate_center
)
coordinate_values, coordinate_vectors = np.linalg.eigh(
    (coordinate_covariance + coordinate_covariance.T) / 2
)
order = np.argsort(coordinate_values)[::-1]
coordinate_values = np.maximum(coordinate_values[order], 0)
coordinate_modes = coordinate_vectors[:, order].T
coordinate_modes *= np.sign(
    coordinate_modes[np.arange(12), np.abs(coordinate_modes).argmax(axis=1)]
)[:, None]
posture_variance = coordinate_values / coordinate_values.sum()
all_posture = (coordinate_means - coordinate_center) @ coordinate_modes[:POSTURE_PCS].T
print("Full-recording PCA ants per colony:", [len(i) for i in colony_indices])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
axes[0].plot(np.arange(1, 13), posture_variance.cumsum(), "o-")
axes[0].set(
    xlabel="Number of landmark PCs",
    ylabel="Cumulative variance explained",
    ylim=(0, 1.02),
)
im = axes[1].imshow(
    coordinate_modes[:POSTURE_PCS], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1
)
axes[1].set(
    xlabel="Landmark coordinate (A1 forward/lateral … B3 forward/lateral)",
    ylabel="Posture PC",
)
axes[1].set_yticks(np.arange(POSTURE_PCS), np.arange(1, POSTURE_PCS + 1))
fig.colorbar(im, ax=axes[1], label="Loading")
print(
    f"First {POSTURE_PCS} posture PCs explain {posture_variance[:POSTURE_PCS].sum():.1%}"
)

# %% 4. Select one colony and channels; inspect an individual ant's time series
ant_indices = np.flatnonzero(coverage.side.eq(SIDE) & coverage.day1_eligible)
ants = coverage.iloc[ant_indices].reset_index(drop=True)
minute = np.concatenate([all_velocity[ant_indices], all_posture[ant_indices]], axis=2)
names = np.array(
    ["forward_velocity", "lateral_velocity"]
    + [f"posture_pc{i+1}" for i in range(POSTURE_PCS)]
)
families = np.array(["velocity"] * 2 + ["posture"] * POSTURE_PCS)
print(pd.DataFrame({"channel": names, "family": families}))
ANT = 0  # row in 'ants'; change to inspect another identity
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, layout="constrained")
axes[0].plot(np.arange(1440) / 60, minute[ANT, :1440, 0], lw=0.6)
axes[0].set(ylabel="Forward velocity (mm/s)", title=ants.ant.iloc[ANT])
axes[1].plot(np.arange(1440) / 60, minute[ANT, :1440, 2], lw=0.6)
axes[1].set(xlabel="Hours from July 24, 10:00 JST", ylabel="Posture PC1")


# %% 5. Minute clips → bin means → each ant's SIX means across bins
# One row per ant: no SD, RMS, rates, or concatenated time-bin features.
# These means do not retain temporal order or excursion variability. Signed
# lateral motion in opposite directions can cancel. Larger bins use the same
# sampled clips; they do not add observations or preserve more dynamics.
# A small helper makes exactly the same calculation reusable for time splits/day2.
def make_profiles(data, mask=None):
    data = data.copy()
    if mask is not None:
        data[:, ~mask] = np.nan
    bins = data.reshape(len(data), 1440 // BIN_MINUTES, BIN_MINUTES, len(names))
    minimum_clips = max(
        1, int(np.ceil(BIN_MINUTES / 12 * (1 if mask is None else mask.mean())))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # unobserved bins stay NaN
        binned = np.nanmean(bins, axis=2)
        binned[np.isfinite(bins).sum(axis=2) < minimum_clips] = np.nan
        raw = np.nanmean(binned, axis=1)
    transformed = raw.copy()
    # Keep the existing signed, invertible velocity transform; no new channels.
    transformed[:, :2] = np.arcsinh(raw[:, :2] / 0.1)
    return binned, raw, transformed


binned, profiles_raw, profiles = make_profiles(minute[:, :1440])
profile_names = [f"mean: {name}" for name in names]
profile_families = families.copy()
profile_table = pd.DataFrame(profiles_raw, index=ants.ant, columns=profile_names)
print("minute:", minute.shape, "binned:", binned.shape, "profiles:", profiles.shape)
print(profile_table.head())


# %% 6. Balance velocity/posture families; keep EVERY retained feature dimension
# Every intermediate is returned so you can inspect it in the variable explorer.
def balance_profiles(x):
    keep = np.isfinite(x).sum(axis=0) >= max(4, int(0.5 * len(x)))
    fill = np.nanmedian(x[:, keep], axis=0)
    imputed = np.where(np.isfinite(x[:, keep]), x[:, keep], fill)
    center, std = imputed.mean(axis=0), imputed.std(axis=0)
    scale = np.maximum(std, 1e-6) if SCALING == "standard" else np.ones(len(std))
    if SCALING == "family":
        for family in np.unique(profile_families):
            columns = profile_families[keep] == family
            scale[columns] = max(np.sqrt(np.sum(std[columns] ** 2)), 1e-6)
    balanced = (imputed - center) / scale
    return dict(
        keep=keep,
        fill=fill,
        imputed=imputed,
        center=center,
        scale=scale,
        balanced=balanced,
    )


def apply_balance(x, state):
    x = x[:, state["keep"]]
    x = np.where(np.isfinite(x), x, state["fill"])
    return (x - state["center"]) / state["scale"]


state = balance_profiles(profiles)
balanced = state["balanced"]
feature_names = np.array(profile_names)[state["keep"]]
balanced_table = pd.DataFrame(balanced, index=ants.ant, columns=feature_names)
# Ordering is for display only; no fitted axis is used for ordering or clustering.
ant_order = np.argsort(profiles_raw[:, 0])
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
im = axes[0].imshow(balanced[ant_order], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].set(
    xlabel="Profile feature",
    ylabel="Ant (ordered by mean forward velocity)",
    title="All balanced features",
)
fig.colorbar(im, ax=axes[0])
axes[1].barh(np.arange(len(feature_names)), balanced.std(axis=0))
axes[1].set_yticks(np.arange(len(feature_names)), feature_names, fontsize=7)
axes[1].set(
    xlabel="SD after family balancing",
    title=f"Clustering input: {balanced.shape[1]} dimensions",
)
print(
    "Balanced family variance:",
    {
        family: float(
            balanced[:, profile_families[state["keep"]] == family].var(axis=0).sum()
        )
        for family in np.unique(families)
    },
)


# %% 7. Fit K=1–4 mixtures without looking at spatial labels
# K-means initializations and mixture fitting both use all balanced dimensions.
def fit_mixture(z, k, seed=SEED, n_init=20):
    model = GaussianMixture(
        k,
        covariance_type=COVARIANCE,
        reg_covar=0.001,
        n_init=n_init,
        max_iter=500,
        random_state=seed,
    ).fit(z)
    if not model.converged_:
        raise RuntimeError("Mixture did not converge")
    return model


mixtures = {k: fit_mixture(balanced, k) for k in range(1, 5)}
k_table = pd.DataFrame(
    [
        dict(
            k=k,
            bic=m.bic(balanced),
            smallest_group=np.bincount(m.predict(balanced), minlength=k).min(),
        )
        for k, m in mixtures.items()
    ]
).set_index("k")
print(k_table)

# %% 8. Refit on resampled ants and disjoint time samples; then choose K
# Every bootstrap refits imputation, family scaling and the full-space mixture.
bootstrap_ari, temporal_ari = {}, {}
for k in (2, 3, 4):
    labels = mixtures[k].predict(balanced)
    rng = np.random.default_rng(SEED + 60000)
    bootstrap_ari[k] = []
    for repeat in range(BOOTSTRAPS if k == 2 else min(BOOTSTRAPS, 100)):
        sample = rng.integers(len(ants), size=len(ants))
        boot_state = balance_profiles(profiles[sample])
        boot_model = fit_mixture(boot_state["balanced"], k, SEED + 60000 + repeat)
        prediction = boot_model.predict(apply_balance(profiles, boot_state))
        bootstrap_ari[k].append(adjusted_rand_score(labels, prediction))
    temporal_ari[k] = []
    for block_minutes in (1, 5, 15, 30, 60, 120):
        halves = []
        for half in (0, 1):
            mask = (np.arange(1440) // block_minutes) % 2 == half
            _, _, split_profiles = make_profiles(minute[:, :1440], mask)
            split_state = balance_profiles(split_profiles)
            halves.append(
                fit_mixture(split_state["balanced"], k).predict(split_state["balanced"])
            )
        temporal_ari[k].append(adjusted_rand_score(*halves))
    k_table.loc[k, ["bootstrap_median", "bootstrap_p10", "temporal_median"]] = [
        np.median(bootstrap_ari[k]),
        np.quantile(bootstrap_ari[k], 0.1),
        np.median(temporal_ari[k]),
    ]
    print("K", k, k_table.loc[k].to_dict())
eligible = k_table[
    (k_table.bic < k_table.loc[1, "bic"])
    & (k_table.smallest_group >= 5)
    & (k_table.bootstrap_median >= 0.8)
    & (k_table.temporal_median >= 0.6)
]
selected_k = (
    1
    if eligible.empty
    else int(eligible[eligible.bic <= eligible.bic.min() + 2].index.min())
)
model = mixtures[selected_k]
labels = model.predict(balanced)
# Name groups from low to high mean forward velocity; this does not affect fitting.
remap = np.argsort(
    np.argsort([np.nanmedian(profiles_raw[labels == g, 0]) for g in range(selected_k)])
)
groups = remap[labels]
print("Selected K:", selected_k, "| Group sizes:", np.bincount(groups))
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
axes[0].plot(k_table.index, k_table.bic - k_table.bic.min(), "o-")
too_small = k_table[k_table.smallest_group < 5]
axes[0].scatter(
    too_small.index,
    too_small.bic - k_table.bic.min(),
    marker="x",
    color="red",
    s=70,
    label="Group smaller than 5",
)
axes[0].set(
    xlabel="K",
    ylabel="BIC − minimum",
    xticks=[1, 2, 3, 4],
    title=f"Selected K={selected_k}",
)
axes[0].legend(fontsize=8)
axes[1].boxplot(
    [bootstrap_ari[k] for k in (2, 3, 4)], tick_labels=["2", "3", "4"], showfliers=False
)
axes[1].set(xlabel="K", ylabel="Ant-bootstrap ARI", ylim=(-0.1, 1.05))
# Show two physical feature coordinates, purely for display. The fitted model
# above uses every balanced column, not this 2D view and not a learned projection.
plot_columns = [
    list(feature_names).index(name)
    for name in ("mean: forward_velocity", "mean: posture_pc1")
]
axes[2].scatter(
    balanced[:, plot_columns[0]], balanced[:, plot_columns[1]], c=COLORS[groups]
)
axes[2].set(
    xlabel="Balanced mean forward velocity",
    ylabel="Balanced posture PC1 mean",
    title=f"{SIDE}: K={selected_k} (two-feature view)",
)

# %% 9. Apply day1 balancing/clustering to day2 in the same full-recording basis
_, day2_raw, day2_profiles = make_profiles(minute[:, 1440:])
ok = ants.day2_eligible.to_numpy(bool)
day2_balanced = apply_balance(day2_profiles[ok], state)
day2_groups = remap[model.predict(day2_balanced)]
if selected_k > 1:
    print(f"Day-to-day retention: {(groups[ok] == day2_groups).sum()}/{ok.sum()}")
else:
    print("K=1: no supported separation; single-group retention is trivial.")
fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
for ax, column in zip(axes, plot_columns):
    ax.scatter(balanced[ok, column], day2_balanced[:, column], c=COLORS[groups[ok]])
    ax.axline((0, 0), slope=1, color=".7", linestyle="--")
    ax.set(
        xlabel="Day1 balanced feature",
        ylabel="Day2 balanced feature",
        title=feature_names[column],
    )
# The landmark PCA includes BOTH days. This is a repeatability check in a shared
# basis, not held-out validation of the complete pipeline. Day1 fitted only the
# balancing and mixture; day2 does not change those two fitted stages.

# %% 10. Only now load the spatial classes and compare ant identities
reference = pd.read_csv(DATA / "spatial_reference.csv")
assignments = ants[["ant", "side", "track_id", "track_name"]].copy()
assignments["activity_group"] = groups
assignments = assignments.merge(
    reference[["side", "track_id", "spatial_cluster"]],
    on=["side", "track_id"],
    validate="one_to_one",
)
contingency = pd.crosstab(assignments.activity_group, assignments.spatial_cluster)
rows, columns = linear_sum_assignment(-contingency.to_numpy())
matched = contingency.to_numpy()[rows, columns].sum()
print(contingency)
if selected_k > 1:
    print(
        f"Spatial agreement after matching names: {matched}/{len(ants)}; ARI =",
        adjusted_rand_score(assignments.spatial_cluster, assignments.activity_group),
    )
else:
    print("K=1: activity features did not recover the two spatial classes.")

# %% 11. Plot where each fitted group actually spends time (equal weight per ant)
occupancy = DATA / "reproduction" / "occupancy" / "per_track"
fig, axes = plt.subplots(
    1, 2 + selected_k, figsize=(3 * (2 + selected_k), 5), layout="constrained"
)
mean_maps, map_titles = [], []
for column, prefix in [("spatial_cluster", "Spatial"), ("activity_group", "Activity")]:
    for group, subset in assignments.groupby(column):
        histograms = []
        for ant in subset.itertuples():
            folder = occupancy / Path(ant.track_name).stem
            histogram = np.load(folder / "grid_occupancy_f4.npy").astype(float)
            histograms.append(histogram / histogram.sum())
            x_edges = np.load(folder / "grid_x_edges_mm.npy")
            y_edges = np.load(folder / "grid_y_edges_mm.npy")
        mean_maps.append(np.mean(histograms, axis=0))
        map_titles.append(f"{prefix} {group} · n={len(subset)}")
vmax = max(np.quantile(m, 0.995) for m in mean_maps)
for ax, data, title in zip(axes, mean_maps, map_titles):
    im = ax.imshow(
        data,
        origin="upper",
        extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
        cmap="magma",
        norm=PowerNorm(0.5, vmin=0, vmax=vmax),
    )
    ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
fig.colorbar(im, ax=axes, label="Mean fraction of observations / bin", shrink=0.7)
plt.show()
# Inspect: coordinate_modes, minute, binned, profile_table, state, balanced_table,
# k_table, bootstrap_ari, temporal_ari, assignments, day2_groups, mean_maps.
# For parameter exploration, change BIN_MINUTES/SCALING/COVARIANCE at the top
# and rerun the cells. Keep spatial agreement out of parameter selection.
