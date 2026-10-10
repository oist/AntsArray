# %% 1. Settings — run cells in VS Code/Spyder, or: python -i analysis/eigenposture_interactive.py
"""Inspect the selected July 24 analysis, one calculation at a time.

Requires numpy, pandas, matplotlib, scipy and scikit-learn. No repo imports.
Change SIDE/BIN_MINUTES/etc. and rerun from cell 4. All arrays remain available.

Input boundary: tracking, body alignment and clip QC have already been done.
One measurement/minute summarizes a sampled 2.5-second clip, NOT a full minute.
The landmark PCA is rebuilt below from exact saved coordinate moments; per-clip
PC means/rates and velocities are loaded from the corresponding measurement cache.
See eigenposture_features.py for the upstream coordinate reconstruction and QC.
This runs the chosen analysis, not the previous 152-configuration search.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

DATA = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260724/block01/analysis_outputs/eigenposture_20261009"
)
SIDE = "right"  # 'left' or 'right'; fits are independent by colony
BIN_MINUTES = 5  # try 1, 15, 30, 60, 120, 240 (must divide 1440)
POSTURE_PCS = 4  # 1–4; four modes are available in this cache
SCALING = "family"  # 'family' or 'standard'
COVARIANCE = "tied"  # 'tied' = shared variance; try 'full' for separate
BOOTSTRAPS = 500  # use 30 for a quick preview; 500 reproduces the report
SEED = 7241010
COLORS = np.array(["#237b9c", "#e18132", "#8867ab", "#a34d52"])

plt.ion()  # figures stay interactive; no files are overwritten
threadpool_limits(limits=1)  # avoid slow BLAS oversubscription on these tiny fits
assert SIDE in ("left", "right") and 1440 % BIN_MINUTES == 0
assert 1 <= POSTURE_PCS <= 4 and SCALING in ("family", "standard")

# %% 2. Load measurements and inspect which ants have enough observations
coverage = pd.read_csv(DATA / "all_ant_coverage.csv")
with np.load(DATA / "eigenposture_measurements.npz") as cache:
    all_minutes = cache["minute"]  # [114 ants, 2880 minutes, 12 channels]
    all_names, all_families = cache["feature_names"], cache["families"]
    np.testing.assert_array_equal(cache["ants"], coverage.ant)
print(coverage.groupby("side")[["day1_eligible", "day2_eligible"]].sum())
print(pd.DataFrame({"channel": all_names, "family": all_families}))

# %% 3. Rebuild the raw-landmark PCA (this is NOT the later ant-profile PCA)
# Coordinates: six antennal landmarks × (forward, lateral), relative to the head,
# normalized by each ant's fixed day1 body-axis length. No straightness feature.
# Saved moments already weight observed hours/minutes/windows equally within ants.
with np.load(DATA / "ant_coordinate_moments.npz") as moments:
    np.testing.assert_array_equal(moments["ants"], coverage.ant)
    day1_version = list(moments["versions"]).index("day1")
    first, second = (
        moments["means"][:, day1_version],
        moments["seconds"][:, day1_version],
    )
colony_indices = [
    np.flatnonzero(coverage.side.eq(s) & coverage.day1_eligible)
    for s in ("left", "right")
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
with np.load(DATA / "landmark_pca.npz") as saved:
    np.testing.assert_allclose(coordinate_modes, saved["components"], atol=1e-9)
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
channels = np.r_[np.arange(4), 4 + np.arange(POSTURE_PCS), 8 + np.arange(POSTURE_PCS)]
minute = all_minutes[ant_indices][:, :, channels]
names, families = all_names[channels], all_families[channels]
# Cache construction: PC mean = (mean coordinates - center) @ mode.T;
# PC rate = sqrt(mean((12 * difference of adjacent smoothed PC samples)**2)).
# Velocity is projected along the anterior/lateral axes before clip mean/RMS.
ANT = 0  # row in 'ants'; change to inspect another identity
fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, layout="constrained")
axes[0].plot(np.arange(1440) / 60, minute[ANT, :1440, 2], lw=0.6)
axes[0].set(ylabel="Forward RMS (mm/s)", title=ants.ant.iloc[ANT])
axes[1].plot(np.arange(1440) / 60, minute[ANT, :1440, 4], lw=0.6)
axes[1].set(xlabel="Hours from July 24, 10:00 JST", ylabel="Posture PC1")


# %% 5. Minute clips → bin means → each ant's mean and SD across bins
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
        raw = np.concatenate(
            [np.nanmean(binned, axis=1), np.nanstd(binned, axis=1)], axis=1
        )
    transformed = raw.copy()
    for j, name in enumerate(np.tile(names, 2)):
        if name in ("forward_mean", "lateral_mean"):
            transformed[:, j] = (
                np.arcsinh(raw[:, j] / 0.1)
                if j < len(names)
                else np.log1p(np.maximum(raw[:, j], 0) / 0.1)
            )
        elif name in ("forward_rms", "lateral_rms"):
            transformed[:, j] = np.log1p(np.maximum(raw[:, j], 0) / 0.1)
        elif name.endswith("_rate"):
            transformed[:, j] = np.log1p(np.maximum(raw[:, j], 0))
    return binned, raw, transformed


binned, profiles_raw, profiles = make_profiles(minute[:, :1440])
profile_names = [f"{stat}: {name}" for stat in ("mean", "sd") for name in names]
profile_families = np.tile(families, 2)
profile_table = pd.DataFrame(profiles_raw, index=ants.ant, columns=profile_names)
print("minute:", minute.shape, "binned:", binned.shape, "profiles:", profiles.shape)
print(profile_table.head())


# %% 6. Balance feature families, then learn ONE axis describing differences among ants
# Every intermediate is returned so you can inspect it in the variable explorer.
def fit_axis(x):
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
    pca = PCA(1, svd_solver="full").fit(balanced)
    score_scale = np.sqrt(pca.explained_variance_[0])
    scores = pca.transform(balanced) / score_scale
    return dict(
        keep=keep,
        fill=fill,
        imputed=imputed,
        center=center,
        scale=scale,
        balanced=balanced,
        pca=pca,
        score_scale=score_scale,
        scores=scores,
    )


def project_profiles(x, state):
    x = x[:, state["keep"]]
    x = np.where(np.isfinite(x), x, state["fill"])
    return (
        state["pca"].transform((x - state["center"]) / state["scale"])
        / state["score_scale"]
    )


state = fit_axis(profiles)
balanced, profile_pca, scores = state["balanced"], state["pca"], state["scores"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
im = axes[0].imshow(
    balanced[np.argsort(scores[:, 0])], aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1
)
axes[0].set(
    xlabel="Profile feature",
    ylabel="Ant (ordered by profile PC1)",
    title="Balanced profiles",
)
fig.colorbar(im, ax=axes[0])
axes[1].barh(np.arange(state["keep"].sum()), profile_pca.components_[0])
axes[1].set_yticks(
    np.arange(state["keep"].sum()), np.array(profile_names)[state["keep"]], fontsize=7
)
axes[1].set(
    xlabel="Loading on ant-profile PC1",
    title=f"{profile_pca.explained_variance_ratio_[0]:.1%} of profile variance",
)


# %% 7. Fit K=1–4 mixtures without looking at spatial labels
# Both random and sorted-score initializations are tried, as in the full analysis.
def fit_mixture(z, k, seed=SEED, n_init=5):
    initial_means = [None]
    if k > 1:
        initial_means += [
            np.stack(
                [z[g].mean(axis=0) for g in np.array_split(np.argsort(z[:, 0]), k)]
            )
        ]
    fits = [
        GaussianMixture(
            k,
            covariance_type=COVARIANCE,
            reg_covar=0.001,
            n_init=n_init if means is None else 1,
            means_init=means,
            max_iter=500,
            random_state=seed,
        ).fit(z)
        for means in initial_means
    ]
    return max(
        (fit for fit in fits if fit.converged_), key=lambda fit: fit.lower_bound_
    )


mixtures = {k: fit_mixture(scores, k, n_init=5 if k == 2 else 20) for k in range(1, 5)}
k_table = pd.DataFrame(
    [
        dict(
            k=k,
            bic=m.bic(scores),
            smallest_group=np.bincount(m.predict(scores), minlength=k).min(),
        )
        for k, m in mixtures.items()
    ]
).set_index("k")
print(k_table)

# %% 8. Refit on resampled ants and disjoint time samples; then choose K
# Every bootstrap refits imputation, family scaling, profile PCA AND mixture.
bootstrap_ari, temporal_ari = {}, {}
for k in (2, 3, 4):
    labels = mixtures[k].predict(scores)
    rng = np.random.default_rng(SEED + 60000)
    bootstrap_ari[k] = []
    for repeat in range(BOOTSTRAPS if k == 2 else min(BOOTSTRAPS, 100)):
        sample = rng.integers(len(ants), size=len(ants))
        boot_state = fit_axis(profiles[sample])
        boot_model = fit_mixture(boot_state["scores"], k, SEED + 60000 + repeat)
        prediction = boot_model.predict(project_profiles(profiles, boot_state))
        bootstrap_ari[k].append(adjusted_rand_score(labels, prediction))
    temporal_ari[k] = []
    for block_minutes in (1, 5, 15, 30, 60, 120):
        halves = []
        for half in (0, 1):
            mask = (np.arange(1440) // block_minutes) % 2 == half
            _, _, split_profiles = make_profiles(minute[:, :1440], mask)
            split_state = fit_axis(split_profiles)
            halves.append(
                fit_mixture(split_state["scores"], k).predict(split_state["scores"])
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
labels = model.predict(scores)
# Name groups from low to high forward RMS velocity; this does not affect fitting.
remap = np.argsort(
    np.argsort([np.nanmedian(minute[labels == g, :1440, 2]) for g in range(selected_k)])
)
groups = remap[labels]
print("Selected K:", selected_k, "| Group sizes:", np.bincount(groups))
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
axes[0].plot(k_table.index, k_table.bic - k_table.bic.min(), "o-")
axes[0].set(xlabel="K", ylabel="BIC − minimum", xticks=[1, 2, 3, 4])
axes[1].boxplot(
    [bootstrap_ari[k] for k in (2, 3, 4)], tick_labels=["2", "3", "4"], showfliers=False
)
axes[1].set(xlabel="K", ylabel="Ant-bootstrap ARI", ylim=(-0.1, 1.05))
grid = np.linspace(scores.min() - 0.3, scores.max() + 0.3, 500)
axes[2].hist(scores[:, 0], bins=12, density=True, color=".8")
axes[2].plot(grid, np.exp(model.score_samples(grid[:, None])), color="black")
axes[2].scatter(scores[:, 0], np.zeros(len(ants)), c=COLORS[groups], s=20)
axes[2].set(xlabel="Ant-profile PC1", ylabel="Density", title=f"{SIDE}: K={selected_k}")

# %% 9. Apply the frozen day1 transform/model to day2
_, day2_raw, day2_profiles = make_profiles(minute[:, 1440:])
ok = ants.day2_eligible.to_numpy(bool)
day2_scores = project_profiles(day2_profiles[ok], state)
day2_groups = remap[model.predict(day2_scores)]
print(f"Next-day retention: {(groups[ok] == day2_groups).sum()}/{ok.sum()}")
fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
ax.scatter(scores[ok, 0], day2_scores[:, 0], c=COLORS[groups[ok]])
ax.axline((0, 0), slope=1, color=".7", linestyle="--")
ax.set(
    xlabel="Day1 profile score",
    ylabel="Day2 score (frozen model)",
    title="Each point is one ant",
)
# Day2 was examined previously; this is a repeatability check, not a fresh holdout.

# %% 10. Only now load the spatial classes and compare ant identities
reference = pd.read_csv(DATA / "spatial_reference.csv")
assignments = ants[["ant", "side", "track_id", "track_name"]].copy()
assignments["activity_group"], assignments["score"] = groups, scores[:, 0]
assignments = assignments.merge(
    reference[["side", "track_id", "spatial_cluster"]],
    on=["side", "track_id"],
    validate="one_to_one",
)
contingency = pd.crosstab(assignments.activity_group, assignments.spatial_cluster)
rows, columns = linear_sum_assignment(-contingency.to_numpy())
matched = contingency.to_numpy()[rows, columns].sum()
print(contingency)
print(
    f"Spatial agreement after matching names: {matched}/{len(ants)}; ARI =",
    adjusted_rand_score(assignments.spatial_cluster, assignments.activity_group),
)
# K=1 is not recovery of two classes, even though its majority-match fraction is nonzero.

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
# Inspect: coordinate_modes, minute, binned, profile_table, state, scores,
# k_table, bootstrap_ari, temporal_ari, assignments, day2_groups, mean_maps.
# For parameter exploration, change BIN_MINUTES/SCALING/COVARIANCE at the top
# and rerun cells 4 onward. Keep spatial agreement out of parameter selection.
