# %% 1. Settings — run these cells in order in VS Code/Spyder
"""Five-minute behavior states, their time courses, and ant task distributions.

Seven features per bin: max unsigned forward/lateral sample velocity + four mean
posture PCs + new interaction bouts. First cluster BINS pooled across all ants;
then compare ANT groups from state proportions alone versus proportions plus
adjacent-bin transitions. Identity and position never enter either clustering.
UMAP is a visualization only. States are not verified biological tasks.
One sampled 2.5-second clip/minute is available, not continuous five-minute video.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, PowerNorm
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from umap import UMAP

DATA = Path(
    "/home/sam-reiter/bucket/ReiterU/Ants/basler/20260724/block01/analysis_outputs/eigenposture_20261009"
)
VELOCITY_DATA = DATA.parent / "task_states_20261010" / "unsigned_velocity.npz"
INTERACTION_DATA = DATA.parent / "task_states_interactions_20261010" / "interaction_counts.npz"
BIN_MINUTES = 5
MIN_CLIPS = 3  # jointly observed clips per bin; missing bins remain unassigned
POSTURE_PCS = 4
TASK_KS = range(2, 11)  # discretization resolutions, chosen without spatial labels
ANT_KS = range(1, 5)  # includes one ant group
MIN_ANT_HOURS = 12  # observed five-minute bins across the full 48 hours
BOOTSTRAPS = 200  # ant resamples per candidate K; try 20 for a preview
SEED = 7241010
UMAP_POINTS = 10000  # random pooled bins for a quick static visualization

plt.ion()
threadpool_limits(limits=1)
assert BIN_MINUTES == 5 and POSTURE_PCS == 4

# %% 2. Load all ants: coordinates and unsigned velocity measured BEFORE averaging
coverage = pd.read_csv(DATA / "all_ant_coverage.csv")
with np.load(DATA / "full_rank" / "eigenposture_measurements.npz") as cache:
    np.testing.assert_array_equal(cache["ants"], coverage.ant)
    old_scores = cache["minute"][:, :, 4:16]
with np.load(DATA / "full_rank" / "landmark_pca.npz") as old_basis:
    coordinate_means = old_scores @ old_basis["components"] + old_basis["mean"]
with np.load(VELOCITY_DATA) as cache:
    np.testing.assert_array_equal(cache["ants"], coverage.ant)
    unsigned_velocity = cache["unsigned_max"]
assert np.nanmin(unsigned_velocity) >= 0
# This new cache applies abs to the original short-window velocities, then max.
# abs(signed clip mean) would lose reversals within a clip and is NOT used.
del old_scores

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


# %% 4. Make a seven-dimensional observation for EACH ant × five-minute bin
names = ["Forward peak (mm/s)", "Lateral peak (mm/s)"] + [
    f"Posture PC{i}" for i in range(1, 5)
]
minute = np.concatenate([unsigned_velocity, all_posture], axis=2)


def bin_features(data):
    """Common clip support for all six features; invalid bins stay entirely NaN."""
    data = data.copy()
    data[~np.isfinite(data).all(axis=2)] = np.nan
    blocks = data.reshape(len(data), -1, BIN_MINUTES, 6)
    counts = np.isfinite(blocks).all(axis=3).sum(axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmean(blocks, axis=2)
        result[:, :, :2] = np.nanmax(blocks[:, :, :, :2], axis=2)
    result[counts < MIN_CLIPS] = np.nan
    return result, counts


binned, clip_counts = bin_features(minute)
with np.load(INTERACTION_DATA) as cache:
    np.testing.assert_array_equal(cache["ants"], coverage.ant)
    assert cache["start_frame"] == 41520 and cache["bin_frames"] == 7200
    interaction_counts = cache["counts"]
assert interaction_counts.shape == binned.shape[:2]
assert np.nanmin(interaction_counts) >= 0
binned = np.concatenate([binned, interaction_counts[:, :, None]], axis=2)
names.append("Interaction onsets / 5 min")
valid = np.isfinite(binned).all(axis=2)
ants = coverage.copy()
ants["observed_bin_hours"] = valid.sum(axis=1) * BIN_MINUTES / 60
ants["eligible"] = ants.observed_bin_hours >= MIN_ANT_HOURS
print(
    ants.groupby("side").agg(identities=("ant", "size"), eligible=("eligible", "sum"))
)
print("All five-minute vectors:", binned.shape, "| valid:", valid.sum())

# %% 5. Pool bins; balance velocity, posture and interaction families
raw = binned[valid]  # no ant-level summary, identity, time, or space in this matrix
transformed = raw.copy()
transformed[:, :2] = np.log1p(transformed[:, :2] / 0.1)
transformed[:, 6] = np.log1p(transformed[:, 6])
feature_center = transformed.mean(axis=0)
feature_scale = np.ones(7)
for columns in (slice(0, 2), slice(2, 6), slice(6, 7)):
    feature_scale[columns] = max(
        np.sqrt(transformed[:, columns].var(axis=0).sum()), 1e-6
    )
task_input = (transformed - feature_center) / feature_scale

# %% 6. Choose a task-state resolution in the full seven-dimensional space
# K-means discretizes behavior; silhouette chooses separation among K=2..10.
# This does not prove a number of biological tasks or test continuous vs discrete.
sample = np.random.default_rng(SEED).choice(
    len(raw), min(3000, len(raw)), replace=False
)
task_models, task_rows = {}, []
for k in TASK_KS:
    candidate = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(task_input)
    task_models[k] = candidate
    task_rows.append(
        dict(
            k=k,
            silhouette=silhouette_score(task_input[sample], candidate.labels_[sample]),
            inertia=candidate.inertia_,
            smallest_bin_fraction=np.bincount(candidate.labels_).min() / len(raw),
        )
    )
task_k_table = pd.DataFrame(task_rows).set_index("k")
task_k = int(task_k_table.silhouette.idxmax())
task_model = task_models[task_k]
# Rename states from lower to higher mean unsigned forward peak, for display only.
state_speed = np.array([raw[task_model.labels_ == k, 0].mean() for k in range(task_k)])
state_remap = np.argsort(np.argsort(state_speed))
tasks = np.full(valid.shape, -1, dtype=int)
tasks[valid] = state_remap[task_model.labels_]
task_means = np.stack([raw[tasks[valid] == k].mean(axis=0) for k in range(task_k)])
task_summary = pd.DataFrame(
    task_means, columns=names, index=pd.Index(range(task_k), name="task")
)
task_summary["bins"] = np.bincount(tasks[valid], minlength=task_k)
print(task_k_table)
print("Selected task-state K:", task_k)
if task_k in (min(TASK_KS), max(TASK_KS)):
    print("Task K is at the tested boundary; it is a descriptive resolution.")
print(task_summary)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
axes[0].plot(task_k_table.index, task_k_table.silhouette, "o-")
axes[0].axvline(task_k, color=".5", ls="--")
axes[0].set(
    xlabel="Task-state K",
    ylabel="Silhouette (seven dimensions)",
    title="Choose a behavioral resolution",
)
centroids = np.stack(
    [task_input[tasks[valid] == k].mean(axis=0) for k in range(task_k)]
)
im = axes[1].imshow(centroids, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
axes[1].set_xticks(range(7), ["Forward", "Lateral", "PC1", "PC2", "PC3", "PC4", "Contacts"], rotation=25)
axes[1].set_yticks(range(task_k), [f"T{k}" for k in range(task_k)])
axes[1].set(title="What distinguishes the task states?", xlabel="Balanced feature")
fig.colorbar(im, ax=axes[1], label="Mean balanced value")

# %% 7. Static UMAP of the fitted states; it never determines cluster assignments
state_colors = plt.get_cmap("tab10")(np.arange(task_k))
state_cmap = ListedColormap(state_colors)
state_cmap.set_bad("#dddddd")
state_norm = BoundaryNorm(np.arange(task_k + 1) - 0.5, task_k)
umap_sample = np.random.default_rng(SEED).choice(len(raw), min(UMAP_POINTS, len(raw)), replace=False)
umap_coordinates = UMAP(n_neighbors=20, min_dist=0.05, metric="euclidean",
                        n_epochs=300, random_state=SEED, n_jobs=1).fit_transform(task_input[umap_sample])
umap_labels = tasks[valid][umap_sample]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
axes[0].scatter(*umap_coordinates.T, c=umap_labels, cmap=state_cmap,
                norm=state_norm, s=3, alpha=0.65, rasterized=True)
for state in range(task_k):
    axes[0].scatter([], [], color=state_colors[state], label=f"T{state}")
axes[0].legend(markerscale=0.8)
axes[0].set_title(f"States fitted in 7D (K={task_k})")
for ax, column, label in zip(axes[1:], (6, 0), ("Interaction onsets / 5 min", "Forward peak (mm/s)")):
    values = raw[umap_sample, column]
    im = ax.scatter(*umap_coordinates.T, c=np.log1p(values), cmap="viridis", s=3,
                    alpha=0.65, rasterized=True)
    fig.colorbar(im, ax=ax, label=f"log(1 + {label})")
    ax.set_title(label)
for ax in axes:
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", xticks=[], yticks=[])
fig.suptitle(f"Random sample of {len(umap_sample):,} bins; visualization only")
bin_rows, bin_columns = np.where(valid)
umap_table = pd.DataFrame(dict(ant=ants.ant.to_numpy()[bin_rows[umap_sample]],
                               bin=bin_columns[umap_sample], state=umap_labels,
                               umap1=umap_coordinates[:, 0], umap2=umap_coordinates[:, 1]))

# %% 8. Each ant's task assignments across the complete 48 hours
clock_ticks = np.arange(0, 49, 12)
clock_labels = [
    "Jul24 10:00",
    "Jul24 22:00",
    "Jul25 10:00",
    "Jul25 22:00",
    "Jul26 10:00",
]


def plot_timelines(orderings, title, show_groups=False, groups=None):
    groups = ant_groups if groups is None and show_groups else groups
    fig, axes = plt.subplots(2, 1, figsize=(15, 15), sharex=True, layout="constrained")
    for ax, side in zip(axes, ("left", "right")):
        rows = orderings[side]
        im = ax.imshow(
            np.ma.masked_less(tasks[rows], 0),
            aspect="auto",
            interpolation="nearest",
            extent=[0, 48, len(rows) - 0.5, -0.5],
            cmap=state_cmap,
            norm=state_norm,
        )
        row_labels = (
            [
                (
                    f"{ants.ant.iloc[i]} G{groups[i]}"
                    if groups[i] >= 0
                    else f"{ants.ant.iloc[i]} (low coverage)"
                )
                for i in rows
            ]
            if show_groups
            else ants.ant.iloc[rows]
        )
        ax.set_yticks(np.arange(len(rows)), row_labels, fontsize=6)
        if show_groups:
            for boundary in np.flatnonzero(np.diff(groups[rows])):
                ax.axhline(boundary + 0.5, color="black", lw=1)
        ax.axvline(24, color="black", ls="--", lw=0.8)
        ax.set(title=f"{side}: {title}", ylabel="Ant; gray = insufficient observations")
    axes[-1].set_xticks(clock_ticks, clock_labels)
    axes[-1].set_xlabel("Recording time (JST); one column = five minutes")
    bar = fig.colorbar(im, ax=axes, ticks=np.arange(task_k), shrink=0.6, pad=0.01)
    bar.ax.set_yticklabels([f"T{k}" for k in range(task_k)])
    return fig


by_identity = {side: np.flatnonzero(ants.side.eq(side)) for side in ("left", "right")}
plot_timelines(by_identity, "task states, identity order")
# A tidy table retains every bin, including -1 for missing observations.
clock = pd.date_range(
    "2026-07-24 10:00", periods=tasks.shape[1], freq="5min", tz="Asia/Tokyo"
)
bin_table = pd.DataFrame(
    dict(
        ant=np.repeat(ants.ant, tasks.shape[1]),
        time=np.tile(clock, len(ants)),
        task=tasks.ravel(),
        valid_clips=clip_counts.ravel(),
    )
)
for column, name in enumerate(names):
    bin_table[name] = binned[:, :, column].ravel()

# %% 9. State proportions and transitions between adjacent observed bins


def task_proportions(labels):
    counts = np.stack([(labels == k).sum(axis=1) for k in range(task_k)], axis=1)
    total = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, total, out=np.full(counts.shape, np.nan), where=total > 0)


proportions = task_proportions(tasks)
proportion_table = pd.DataFrame(
    proportions, index=ants.ant, columns=[f"T{k}" for k in range(task_k)]
)
def transition_counts(labels):
    """Counts of ordered state pairs, never spanning a missing bin or split gap."""
    result = np.zeros((len(labels), task_k, task_k), dtype=int)
    for ant, row in enumerate(labels):
        valid_pair = (row[:-1] >= 0) & (row[1:] >= 0)
        np.add.at(result[ant], (row[:-1][valid_pair], row[1:][valid_pair]), 1)
    return result


def ant_features(labels, with_transitions=False, transition_scale=1.0):
    # Square roots retain all proportion/transition coordinates; no further PCA.
    fractions = np.sqrt(task_proportions(labels))
    if not with_transitions:
        return fractions
    pairs = transition_counts(labels).reshape(len(labels), -1)
    total = pairs.sum(axis=1, keepdims=True)
    joint = np.divide(pairs, total, out=np.full(pairs.shape, np.nan), where=total > 0)
    return np.column_stack([fractions, np.sqrt(joint) * transition_scale])


pairs = transition_counts(tasks)
next_counts = pairs.sum(axis=2)
stay_probabilities = np.divide(
    np.diagonal(pairs, axis1=1, axis2=2), next_counts,
    out=np.full(next_counts.shape, np.nan), where=next_counts > 0,
)
transition_table = pd.DataFrame(pairs.reshape(len(ants), -1), index=ants.ant,
                               columns=[f"T{i}->T{j}" for i in range(task_k) for j in range(task_k)])
print(proportion_table.head())


# %% 10. Compare groups from proportions alone vs proportions + state transitions
# K=1 is allowed. Stability checks condition on the already fitted state dictionary.
def fit_ant_model(x, k, seed=SEED):
    model = GaussianMixture(k, covariance_type="tied", reg_covar=0.001,
                            n_init=10, max_iter=500, random_state=seed).fit(x)
    if not model.converged_:
        raise RuntimeError("Ant mixture did not converge")
    return model


def cluster_ants(with_transitions):
    groups = np.full(len(ants), -1, dtype=int)
    selected_models, tables, scales = {}, {}, {}
    for side in ("left", "right"):
        indices = np.flatnonzero(ants.side.eq(side) & ants.eligible)
        labels_by_ant = tasks[indices]
        base = ant_features(labels_by_ant)
        scale = 1.0
        if with_transitions:
            joint = ant_features(labels_by_ant, True)[:, task_k:]
            # Equal total between-ant variance for proportions and transitions.
            scale = np.sqrt(base.var(axis=0).sum() / max(joint.var(axis=0).sum(), 1e-12))
        scales[side] = scale
        x = ant_features(labels_by_ant, with_transitions, scale)
        assert np.isfinite(x).all()
        models = {k: fit_ant_model(x, k) for k in ANT_KS}
        table = pd.DataFrame([
            dict(k=k, bic=m.bic(x), smallest_group=np.bincount(m.predict(x), minlength=k).min())
            for k, m in models.items()
        ]).set_index("k")
        for k in list(ANT_KS)[1:]:
            labels = models[k].predict(x)
            rng = np.random.default_rng(SEED)
            bootstrap, temporal = [], []
            for repeat in range(BOOTSTRAPS):
                sample = rng.integers(len(x), size=len(x))
                m = fit_ant_model(x[sample], k, SEED + repeat)
                bootstrap.append(adjusted_rand_score(labels, m.predict(x)))
            for block_minutes in (30, 60, 120):
                halves = []
                for half in (0, 1):
                    mask = (np.arange(tasks.shape[1]) * BIN_MINUTES // block_minutes) % 2 == half
                    half_labels = labels_by_ant.copy()
                    half_labels[:, ~mask] = -1  # keep time gaps: NEVER concatenate halves
                    half_x = ant_features(half_labels, with_transitions, scale)
                    assert np.isfinite(half_x).all()
                    halves.append(fit_ant_model(half_x, k).predict(half_x))
                temporal.append(adjusted_rand_score(*halves))
            table.loc[k, ["bootstrap_median", "bootstrap_p10", "temporal_median"]] = [
                np.median(bootstrap), np.quantile(bootstrap, 0.1), np.median(temporal)]
        admissible = table[(table.bic < table.loc[1, "bic"]) & (table.smallest_group >= 5)
                           & (table.bootstrap_median >= 0.8) & (table.temporal_median >= 0.6)]
        table["admissible"] = table.index.isin(admissible.index)
        k = 1 if admissible.empty else int(admissible[admissible.bic <= admissible.bic.min() + 2].index.min())
        model = models[k]
        labels = model.predict(x)
        expected_speed = proportions[indices] @ task_means[:, 0]
        remap = np.argsort(np.argsort([expected_speed[labels == g].mean() for g in range(k)]))
        groups[indices] = remap[labels]
        selected_models[side], tables[side] = model, table
        print(f"{side}, transitions={with_transitions}: K={k}, n={len(indices)}, sizes={np.bincount(groups[indices])}", flush=True)
        print(table, flush=True)
    return groups, selected_models, tables, scales


baseline_groups, baseline_models, baseline_k_tables, _ = cluster_ants(False)
ant_groups, ant_models, ant_k_tables, transition_scales = cluster_ants(True)
ants["proportion_group"] = baseline_groups
ants["ant_group"] = ant_groups
comparison_rows = []
fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
for row, (method, models, tables) in enumerate([
    ("Proportions", baseline_models, baseline_k_tables),
    ("Proportions + transitions", ant_models, ant_k_tables),
]):
    for ax, side in zip(axes[row], ("left", "right")):
        table = tables[side]
        selected_k = models[side].n_components
        ax.plot(table.index, table.bic - table.bic.min(), "o-")
        rejected = table[(table.index > 1) & ~table.admissible]
        ax.scatter(rejected.index, rejected.bic - table.bic.min(), marker="x", s=70,
                   color="red", label="Fails BIC/size/stability rule")
        ax.axvline(selected_k, ls="--", color=".5")
        ax.set(xlabel="Ant-group K", ylabel="BIC − minimum", xticks=list(ANT_KS),
               title=f"{side}: {method}; selected K={selected_k}")
        ax.legend(fontsize=7)
        comparison_rows.append(dict(side=side, method=method, k=selected_k,
                                    **table.loc[selected_k].drop("admissible").to_dict()))
comparison_table = pd.DataFrame(comparison_rows)

# Show persistence conditional on the current state, alongside the group overlap.
fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
for row, side in enumerate(("left", "right")):
    indices = np.flatnonzero(ants.side.eq(side) & ants.eligible)
    cross = pd.crosstab(baseline_groups[indices], ant_groups[indices])
    axes[row, 0].imshow(cross, cmap="Blues", vmin=0)
    for (i, j), value in np.ndenumerate(cross.to_numpy()):
        axes[row, 0].text(j, i, str(value), ha="center", va="center",
                          color="white" if value > cross.to_numpy().max() / 2 else "black")
    axes[row, 0].set(xlabel="Group: proportions + transitions", ylabel="Group: proportions only",
                     xticks=range(cross.shape[1]), yticks=range(cross.shape[0]), title=f"{side}: same ants")
    for ax, values, ylabel in zip(axes[row, 1:], (proportions, stay_probabilities),
                                  ("Fraction of observed bins", "P(stay in next 5-minute bin)")):
        for g in range(baseline_models[side].n_components):
            group = indices[baseline_groups[indices] == g]
            means = np.nanmean(values[group], axis=0)
            line, = ax.plot(range(task_k), means, "o-", label=f"G{g}, n={len(group)}")
            for state in range(task_k):
                ax.scatter(np.full(len(group), state) + (g - 1) * .04,
                           values[group, state], s=10, alpha=.2, color=line.get_color())
        ax.set(xticks=range(task_k), xticklabels=[f"T{s}" for s in range(task_k)],
               ylabel=ylabel, ylim=(0, 1), title=f"{side}: proportion groups")
        ax.legend(fontsize=7)

# %% 11. Compare task distributions and timelines after grouping ants
ordered = {}
for side, rows in by_identity.items():
    # Low-coverage ants remain visible at the bottom, with proportion_group=-1.
    order = np.lexsort(
        (
            np.nan_to_num(proportions[rows] @ task_means[:, 0]),
            np.where(baseline_groups[rows] < 0, 99, baseline_groups[rows]),
        )
    )
    ordered[side] = rows[order]
plot_timelines(ordered, "ordered by ant proportion group", show_groups=True, groups=baseline_groups)
fig, axes = plt.subplots(2, 1, figsize=(15, 8), layout="constrained")
for ax, side in zip(axes, ("left", "right")):
    rows = ordered[side]
    bottom = np.zeros(len(rows))
    for task in range(task_k):
        height = proportions[rows, task]
        ax.bar(
            np.arange(len(rows)),
            height,
            bottom=bottom,
            color=state_colors[task],
            label=f"T{task}",
        )
        bottom += height
    ax.set_xticks(
        np.arange(len(rows)),
        [
            (
                f"{ants.ant.iloc[i]}\nG{baseline_groups[i]}"
                if baseline_groups[i] >= 0
                else f"{ants.ant.iloc[i]}\nlow coverage"
            )
            for i in rows
        ],
        rotation=90,
        fontsize=6,
    )
    ax.set(
        ylabel="Fraction of observed bins",
        ylim=(0, 1),
        title=f"{side}: each ant's task distribution",
    )
axes[0].legend(ncols=task_k, fontsize=8)

# %% 12. Only NOW compare with spatial classes; they did not choose either clustering
reference = pd.read_csv(DATA / "spatial_reference.csv")
assignments = ants.merge(
    reference[["side", "track_id", "spatial_cluster"]],
    on=["side", "track_id"],
    how="left",  # keep identities that lack a spatial reference
    validate="one_to_one",
)
assert len(assignments) == len(ants)
assert assignments.loc[assignments.eligible, "spatial_cluster"].notna().all()
spatial_results = []
for side in ("left", "right"):
    subset = assignments[assignments.side.eq(side) & assignments.eligible]
    for method, column, models in [("Proportions", "proportion_group", baseline_models),
                                    ("Proportions + transitions", "ant_group", ant_models)]:
        contingency = pd.crosstab(subset[column], subset.spatial_cluster)
        ari = adjusted_rand_score(subset.spatial_cluster, subset[column])
        matched = None
        if models[side].n_components == 2:
            rows, columns = linear_sum_assignment(-contingency.to_numpy())
            matched = int(contingency.to_numpy()[rows, columns].sum())
        print(side, method, "spatial comparison:\n", contingency)
        print(f"ARI={ari:.3f}; K={models[side].n_components}")
        spatial_results.append(dict(side=side, method=method, matched=matched,
                                    n=len(subset), ari=ari))

# %% 13. Where do the ant groups spend time? Equal weight per ant, same cohort
occupancy = DATA / "reproduction" / "occupancy" / "per_track"
for side in ("left", "right"):
    subset = assignments[assignments.side.eq(side) & assignments.eligible]
    maps, titles = [], []
    for column, prefix in [("spatial_cluster", "Spatial"), ("proportion_group", "Proportions")]:
        for group, group_ants in subset.groupby(column):
            histograms = []
            for ant in group_ants.itertuples():
                folder = occupancy / Path(ant.track_name).stem
                histogram = np.load(folder / "grid_occupancy_f4.npy").astype(float)
                histograms.append(histogram / histogram.sum())
                x_edges = np.load(folder / "grid_x_edges_mm.npy")
                y_edges = np.load(folder / "grid_y_edges_mm.npy")
            maps.append(np.mean(histograms, axis=0))
            titles.append(f"{side} {prefix} {group} · n={len(group_ants)}")
    fig, axes = plt.subplots(
        1, len(maps), figsize=(3 * len(maps), 5), layout="constrained"
    )
    vmax = max(np.quantile(m, 0.995) for m in maps)
    for ax, data, title in zip(axes, maps, titles):
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
# Inspect: coordinate_modes, binned, clip_counts, task_input, task_k_table,
# task_summary, umap_table, tasks, bin_table, proportion_table, transition_table,
# baseline_k_tables, ant_k_tables, comparison_table, assignments.
