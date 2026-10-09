"""Report the frozen PC/velocity parameter audit and post-selection checks."""

from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import PowerNorm
import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.eigenposture_parameters import Config, aggregate, load_source

COLORS = ["#237b9c", "#e18132"]
SIDES = ["left", "right"]


def report(source, output, occupancy):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    cfg = Config(
        **json.loads((output / "SELECTED_FROZEN.json").read_text())["configuration"]
    )
    q, minute, names, families = load_source(source)
    a = pd.read_csv(output / "assignments_evaluated.csv")
    ev = pd.read_csv(output / "evaluation.csv").set_index("side")
    ks = pd.read_csv(output / "k_selection.csv")
    val = pd.read_csv(output / "validation.csv")
    v = val[val.config.eq(cfg.key)].set_index("side")
    baseline = pd.read_csv(output / "baseline_diagnosis.csv")
    reps = pd.read_csv(output / "baseline_replicates.csv")
    control = pd.read_csv(output / "controls.csv")
    post = pd.read_csv(output / "controls_posthoc.csv")
    screen = pd.read_csv(output / "screen.csv")
    models = joblib.load(output / "models.joblib")
    x = aggregate(minute[:, :1440], cfg)
    day2 = aggregate(minute[:, 1440:], cfg)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 170,
        }
    )
    captions = []
    pdf = PdfPages(output / "parameter_audit_figures.pdf")

    def save(fig, key, caption):
        fig.savefig(figures / f"{key}.png", bbox_inches="tight")
        fig.savefig(figures / f"{key}.pdf", bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        captions.append((key, caption))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for r, side in enumerate(SIDES):
        ax = axs[r, 0]
        data = []
        for covariance, fixed in [
            ("full", False),
            ("full", True),
            ("tied", False),
            ("tied", True),
        ]:
            data.append(
                reps[
                    reps.side.eq(side)
                    & reps.covariance.eq(covariance)
                    & reps.fixed_representation.eq(fixed)
                ].ari
            )
        ax.boxplot(
            data,
            showfliers=False,
            whis=(10, 90),
            tick_labels=[
                "Separate\nrefit PCs",
                "Separate\nfixed PCs",
                "Shared\nrefit PCs",
                "Shared\nfixed PCs",
            ],
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Bootstrap adjusted Rand index")
        ax.set_title(
            f"{side.capitalize()}: original hourly profiles; change only mixture"
        )
        ax = axs[r, 1]
        for cov, color, label in [
            ("full", "#a34d52", "Original profiles, separate variance (24)"),
            ("tied", "#5579aa", "Original profiles, shared variance (24)"),
        ]:
            ss = screen[
                screen.side.eq(side)
                & screen.summary.eq("quartiles")
                & screen.scaling.eq("standard")
                & screen.dimensions.eq(1)
                & screen.covariance.eq(cov)
            ].sort_values("bin_minutes")
            ax.plot(
                ss.bin_minutes,
                ss.bootstrap_median,
                "o--",
                color=color,
                label=label,
                alpha=0.85,
            )
        cc = control[
            control.side.eq(side) & control.families.eq("velocity+posture+dynamics")
        ].sort_values("bin_minutes")
        ax.plot(
            cc.bin_minutes,
            cc.bootstrap_median,
            "o-",
            color="#238366",
            label="New profiles, shared variance (200)",
        )
        ax.fill_between(
            cc.bin_minutes,
            cc.bootstrap_p10,
            cc.bootstrap_median,
            alpha=0.14,
            color="#238366",
        )
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 15, 30, 60, 120, 240], [1, 5, 15, 30, 60, 120, 240])
        ax.set_ylim(0.25, 1.05)
        ax.set_xlabel("Bin width (minutes)")
        ax.set_ylabel("Median bootstrap ARI")
        ax.set_title(f"{side.capitalize()}: bin size is a secondary effect")
        if r == 0:
            ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "1  Instability is mainly in the partition, not the posture PCs", fontsize=17
    )
    save(
        fig,
        "01_instability_and_binning",
        "Left: 300 ant bootstraps of the exact old hourly representation. Holding profile scaling/PCA fixed barely helps the separate-variance mixture. A shared variance stabilizes it. “PCs” here means the ant-profile PCA; the raw-landmark basis is held fixed throughout this parameter audit. Boxes show quartiles, whiskers the 10th–90th percentiles. Right: controlled bin-size changes, with replicate counts in parentheses. The green band spans the 10th percentile to the median. A value of one means identical partitions, up to group names.",
    )

    fig, axs = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for r, side in enumerate(SIDES):
        sub = a[a.side.eq(side)]
        mod = models[side]
        t = mod["transform"]
        m = mod["mixture"]
        ix = mod["indices"]
        z = t.transform(x[ix])[:, 0]
        grid = np.linspace(z.min() - 0.4, z.max() + 0.4, 500)
        ax = axs[r, 0]
        ax.hist(z, bins=12, density=True, color="#dddddd", edgecolor="white")
        ax.plot(grid, np.exp(m.score_samples(grid[:, None])), color="black", lw=1.5)
        for c in [0, 1]:
            g = mod["remap"][c]
            ax.fill_between(
                grid,
                m.weights_[c]
                * norm.pdf(grid, m.means_[c, 0], np.sqrt(m.covariances_[0, 0])),
                color=COLORS[g],
                alpha=0.35,
            )
        ax.scatter(z, np.zeros(len(z)) - 0.015, c=[COLORS[g] for g in sub.group], s=22)
        ax.set_xlabel("Ant-profile PC1 (training SD units)")
        ax.set_ylabel("Density")
        ax.set_title(f"{side.capitalize()}: {np.bincount(sub.group).tolist()} ants")
        ax = axs[r, 1]
        tab = ks[ks.side.eq(side)]
        ax.plot(tab.k, tab.bic - tab.bic.min(), "o-", color="#334e68")
        ax.set_xticks([1, 2, 3, 4])
        small = tab[tab.smallest_group < 5]
        ax.scatter(
            small.k,
            small.bic - tab.bic.min(),
            marker="x",
            s=100,
            c="#b33",
            label="Group smaller than 5",
        )
        ax.set_xlabel("Number of groups, K")
        ax.set_ylabel("BIC − minimum (lower is better)")
        gain = float(tab[tab.k.eq(1)].bic.iloc[0] - tab[tab.k.eq(2)].bic.iloc[0])
        ax.set_title(f"K=2 selected; BIC gain over K=1: {gain:.2f}")
        if len(small):
            ax.legend(fontsize=8)
        ax = axs[r, 2]
        ct = pd.crosstab(sub.group, sub.spatial_group).reindex(
            index=[0, 1], columns=[0, 1], fill_value=0
        )
        ax.imshow(ct, cmap="Blues", vmin=0, vmax=ct.to_numpy().max())
        for i in [0, 1]:
            for j in [0, 1]:
                ax.text(
                    j,
                    i,
                    str(ct.iloc[i, j]),
                    ha="center",
                    va="center",
                    fontsize=21,
                    color=(
                        "white" if ct.iloc[i, j] > ct.to_numpy().max() / 2 else "black"
                    ),
                )
        ax.set_xticks([0, 1], ["S0", "S1"])
        ax.set_yticks([0, 1], ["A0", "A1"])
        ax.set_xlabel("Spatial reference")
        ax.set_ylabel("Activity/posture group")
        ax.set_title(
            f'{int(ev.loc[side,"spatial_correct"])}/{len(sub)} spatial agreement ({ev.loc[side,"spatial_accuracy"]:.1%})'
        )
    fig.suptitle(
        "2  A stable two-group fit, with incomplete spatial agreement", fontsize=17
    )
    save(
        fig,
        "02_groups_k_and_space",
        "The selected setting uses five-minute means of twelve PC/velocity channels, then each channel’s mean and standard deviation across bins, three equally weighted feature families, one ant-profile PC, and a shared-variance Gaussian mixture. K=1–4 is checked afterward using BIC, minimum group size five, median bootstrap ARI ≥0.8 and median temporal-split ARI ≥0.6; prefer the smaller admissible K within two BIC units. Both select K=2. The left-colony BIC gain of 2.82 is modest, so assignment stability is stronger evidence than the claim of two discrete populations. Spatial labels are used only after model and assignment files are hashed.",
    )

    fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(minute[:, :1440], axis=1)
    loading = pd.read_csv(output / "profile_loadings.csv")
    feature_short = [
        "Forward mean",
        "Lateral mean",
        "Forward RMS",
        "Lateral RMS",
        "Posture PC1",
        "Posture PC2",
        "Posture PC3",
        "Posture PC4",
        "PC1 rate",
        "PC2 rate",
        "PC3 rate",
        "PC4 rate",
    ]
    for r, side in enumerate(SIDES):
        sub = a[a.side.eq(side)]
        ix = models[side]["indices"]
        colors = [COLORS[g] for g in sub.group]
        ax = axs[r, 0]
        ax.scatter(
            med[ix, 4], med[ix, 2], c=colors, s=45, edgecolors="white", linewidths=0.5
        )
        ax.set_yscale("log")
        ax.set_xlabel("Median posture PC1 (body-axis lengths)")
        ax.set_ylabel("Median forward RMS velocity (mm/s)")
        ax.set_title(f"{side.capitalize()}: learned posture and velocity covary")
        ax = axs[r, 1]
        raw = med[ix]
        scale = np.nanstd(raw, axis=0)
        center = np.nanmean(raw, axis=0)
        for g in [0, 1]:
            yy = (np.nanmedian(raw[sub.group.to_numpy() == g], axis=0) - center) / scale
            ax.plot(yy, np.arange(12), "o-", color=COLORS[g], label=f"A{g}")
        ax.set_yticks(np.arange(12), feature_short)
        ax.invert_yaxis()
        ax.axvline(0, color=".7", lw=0.7)
        ax.set_xlabel("Group median relative to cohort (SD)")
        ax.set_title("Physical channels, before profile PCA")
        ax.legend(fontsize=8)
        ax = axs[r, 2]
        ll = loading[loading.side.eq(side)].copy()
        ll["magnitude"] = ll.coefficient.abs()
        ll = ll.nlargest(10, "magnitude").sort_values("magnitude")
        ax.barh(
            np.arange(len(ll)),
            ll.coefficient,
            color=[
                (
                    "#446c9c"
                    if f == "velocity"
                    else "#5a997e" if f == "posture" else "#a589b5"
                )
                for f in ll.family
            ],
        )
        labels = [
            s.replace("posture_", "")
            .replace("_mean", "")
            .replace("_rate", " rate")
            .replace(":", ": ")
            for s in ll.feature
        ]
        ax.set_yticks(np.arange(len(ll)), labels)
        ax.set_xlabel("Ant-profile PC1 loading")
        ax.set_title("Largest contributions to the learned axis")
    fig.suptitle(
        "3  The separation combines locomotion and learned antennal posture",
        fontsize=17,
    )
    save(
        fig,
        "03_physical_features_and_loadings",
        "A0/A1 are ordered by median forward RMS velocity, after fitting. The posture inputs remain raw-coordinate PCs; neither straightness nor spatial occupancy enters this model. Group summaries use one median per ant as the replicate. Feature profiles are standardized for display only. Mean and SD in the loadings refer to the across-bin profile; PC rate is the RMS derivative within the sampled 2.5-second clips. Removing landmark PCs 3–4 leaves every selected assignment unchanged in both colonies. Ant-profile PC1 captures 79.6%/78.6% of the balanced feature variance.",
    )

    fig, axs = plt.subplots(2, 4, figsize=(12, 10), layout="constrained")
    mapdata = {}
    sources = []
    for r, side in enumerate(SIDES):
        sub = a[a.side.eq(side)]
        maps = []
        titles = []
        for column, label in [("spatial_group", "Spatial"), ("group", "Activity")]:
            for group in [0, 1]:
                stack = []
                for row in sub[sub[column].eq(group)].itertuples():
                    path = occupancy / "per_track" / Path(row.track_name).stem
                    h = np.load(path / "grid_occupancy_f4.npy").astype(float)
                    h /= h.sum()
                    stack.append(h)
                    xe = np.load(path / "grid_x_edges_mm.npy")
                    ye = np.load(path / "grid_y_edges_mm.npy")
                    sources.append(dict(ant=row.ant, path=str(path)))
                maps.append(np.mean(stack, axis=0))
                titles.append(f"{label} {group} · n={len(stack)}")
                mapdata[f"{side}_{label.lower()}_{group}"] = maps[-1]
        vmax = max(np.quantile(m, 0.995) for m in maps)
        for col, (data, title) in enumerate(zip(maps, titles)):
            ax = axs[r, col]
            im = ax.imshow(
                data,
                origin="upper",
                extent=[xe[0], xe[-1], ye[-1], ye[0]],
                cmap="magma",
                norm=PowerNorm(0.5, vmin=0, vmax=vmax),
            )
            ax.set_title(title)
            ax.set_xlabel("Arena x (mm)")
            if col == 0:
                ax.set_ylabel(f"{side.capitalize()}\nArena y (mm)")
        cb = fig.colorbar(im, ax=axs[r, :], fraction=0.025, pad=0.02)
        cb.set_label("Mean fraction of observations / bin")
    fig.suptitle(
        "4  Spatial segregation emerges after activity/posture clustering", fontsize=17
    )
    np.savez_compressed(output / "mean_occupancy.npz", **mapdata)
    pd.DataFrame(sources).drop_duplicates().to_csv(
        output / "occupancy_sources.csv", index=False
    )
    save(
        fig,
        "04_emergent_spatial_occupancy",
        "Each ant’s occupancy histogram is normalized to one before averaging, giving every ant equal weight. The same color scale applies across the four maps within each colony; square-root color scaling and 99.5th-percentile clipping expose lower-density structure. Maps come from the existing occupancy reference and are a post-selection comparison, not a new time-held-out spatial test. No coordinates or occupancy labels were used to select parameters.",
    )

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for r, side in enumerate(SIDES):
        mod = models[side]
        ix = mod["indices"]
        sub = a[a.side.eq(side)]
        ok = q.iloc[ix].day2_eligible.to_numpy(bool)
        z1 = mod["transform"].transform(x[ix])[:, 0]
        z2 = mod["transform"].transform(day2[ix[ok]])[:, 0]
        ax = axs[r, 0]
        for jj, j in enumerate(np.flatnonzero(ok)):
            ax.plot(
                [1, 2],
                [z1[j], z2[jj]],
                color=COLORS[sub.group.iloc[j]],
                alpha=0.45,
                lw=0.8,
            )
        ax.scatter(
            np.ones(ok.sum()), z1[ok], c=[COLORS[g] for g in sub.group[ok]], s=22
        )
        ax.scatter(
            np.ones(ok.sum()) * 2,
            z2,
            c=[COLORS[int(g)] for g in sub.day2_group[ok]],
            s=22,
        )
        ax.set_xticks([1, 2], ["First 24 h", "Next 24 h"])
        ax.set_xlim(0.8, 2.2)
        ax.set_ylabel("Frozen ant-profile PC1 score")
        ax.set_title(
            f'{side.capitalize()}: {int(ev.loc[side,"day2_retained"])}/{int(ev.loc[side,"day2_n"])} retain group next day'
        )
        ax = axs[r, 1]
        order = np.argsort(z1)
        co = np.load(output / f"{side}_coassignment.npy")
        im = ax.imshow(
            co[np.ix_(order, order)],
            vmin=0,
            vmax=1,
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_xlabel("Ants ordered by phenotype score")
        ax.set_ylabel("Same ant ordering")
        ax.set_title("500 bootstraps: probability of sharing a group")
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(
        "5  Assignments repeat across ants, time samples, and days", fontsize=17
    )
    save(
        fig,
        "05_repeatability",
        "Bootstrap samples draw ants with replacement and refit feature scaling, ant-profile PCA, and the mixture; the four-mode landmark basis stays fixed. Final median bootstrap ARI is 1.00 in both colonies, with 10th percentiles 0.90/0.91. Every leave-one-ant-out fit reproduces all full-data assignments. Frozen next-day retention is 39/41 left and 43/44 right; separately refitting day2 gives ARI 0.81/0.82 against day1. Day2 was already examined during the preceding analysis and is not a fresh prospective validation dataset. One right ant fails the day2 coverage rule.",
    )

    fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    spatial = pd.read_csv(output / "screen_posthoc_spatial.csv")
    merged = screen.merge(spatial, on=["config", "side"], validate="one_to_one")
    joint = control.merge(
        post, on=["config", "side", "families"], validate="one_to_one"
    )
    flabel = {
        "velocity": "Velocity",
        "posture": "Posture PCs",
        "dynamics": "PC dynamics",
        "posture+dynamics": "Posture + dynamics",
        "velocity+posture": "Velocity + posture",
        "velocity+dynamics": "Velocity + dynamics",
        "velocity+posture+dynamics": "All three",
    }
    for r, side in enumerate(SIDES):
        ax = axs[r, 0]
        for cov, color in [("full", "#a86e77"), ("tied", "#3a8c83")]:
            ss = merged[merged.side.eq(side) & merged.covariance.eq(cov)]
            xx = (ss.bootstrap_median + ss.minute_ari + ss.block30_ari) / 3
            ax.scatter(
                xx,
                ss.spatial_accuracy,
                c=color,
                alpha=0.55,
                s=25,
                label="Separate variance" if cov == "full" else "Shared variance",
            )
        ss = merged[merged.side.eq(side) & merged.config.eq(cfg.key)].iloc[0]
        ax.scatter(
            [(ss.bootstrap_median + ss.minute_ari + ss.block30_ari) / 3],
            [ss.spatial_accuracy],
            marker="*",
            s=220,
            c="#e3a326",
            edgecolors="black",
            label="Selected without space",
        )
        ax.set_xlabel("Screen repeatability (mean of three ARIs)")
        ax.set_ylabel("Post-selection spatial agreement")
        ax.set_title(f"{side.capitalize()}: Repeatability and spatial agreement")
        ax.set_ylim(0.5, 1.02)
        if r == 0:
            ax.legend(fontsize=8)
        ax = axs[r, 1]
        ss = joint[joint.side.eq(side) & joint.bin_minutes.eq(5)].copy()
        ss["label"] = ss.families.map(flabel)
        yy = np.arange(len(ss))
        ax.scatter(
            ss.bootstrap_median, yy, c="#3a8c83", s=45, label="Bootstrap median ARI"
        )
        ax.scatter(
            ss.spatial_accuracy,
            yy,
            c="#b77b42",
            marker="s",
            s=35,
            label="Spatial agreement",
        )
        for i, row in enumerate(ss.itertuples()):
            ax.plot(
                [row.bootstrap_p10, row.bootstrap_median], [i, i], c="#3a8c83", lw=2
            )
            if row.bic_gain <= 0:
                ax.text(0.02, i, "K=1 favored", fontsize=7, va="center", color="#a33")
        ax.set_yticks(yy, ss.label)
        ax.set_xlim(0, 1.04)
        ax.set_xlabel("ARI / fraction agreeing")
        ax.set_title("Feature ablations (post-selection diagnostic)")
        if r == 0:
            ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "6  Parameter exploration reveals a stability–agreement tradeoff", fontsize=17
    )
    save(
        fig,
        "06_parameter_tradeoffs",
        "All 152 screened configurations are shown after freezing the selected configuration; none is selected or re-ranked by spatial agreement. Screen repeatability averages bootstrap median, even/odd-minute ARI, and alternating-half-hour ARI, each within one colony. Final selection used the conservative worst-colony 10th-percentile metrics from the larger validation stage instead. Ablations hold the selected representation and mixture settings fixed, refit without individual feature families, and use 200 bootstraps. Spatial agreement and ARI are different quantities. PC dynamics alone does not favor K=2 by BIC in either colony.",
    )
    pdf.close()

    rows = []
    for side in SIDES:
        k = ks[ks.side.eq(side) & ks.k.eq(2)].iloc[0]
        rows.append(
            dict(
                Colony=side.capitalize(),
                Ants=len(a[a.side.eq(side)]),
                Groups=k.group_sizes,
                Bootstrap=f"{k.bootstrap_median:.2f} ({k.bootstrap_p10:.2f})",
                Time_splits=f'{v.loc[side,"temporal_median"]:.2f} ({v.loc[side,"temporal_p10"]:.2f})',
                Spatial=f'{int(ev.loc[side,"spatial_correct"])}/{int(ev.loc[side,"n"])}',
                Next_day=f'{int(ev.loc[side,"day2_retained"])}/{int(ev.loc[side,"day2_n"])}',
            )
        )
    results = pd.DataFrame(rows)
    body = """<p class="lead">Hourly binning is not the main source of instability. The old separate-variance mixture can move its boundary when a few ants are resampled. A shared-variance mixture gives a substantially more repeatable activity/posture partition, but it does not exactly reproduce the spatial groups.</p>
<p>The raw-landmark basis is stable: four posture PCs explain 93.2% of coordinate variance; the maximum angle between independently learned even/odd-minute bases was 0.64°. In a new 300-bootstrap audit of the exact old hourly features, fixing the ant-profile PCA hardly changes the instability (left median ARI 0.72 → 0.72; right 0.50 → 0.46). Changing only the mixture to a shared variance raises the medians to 1.00 in both colonies. The old fitted component variances differed about fivefold on the left and tenfold on the right; that extra flexibility lets the boundary shift among intermediate ants.</p>
<p><b>Selected setting:</b> five-minute bin means → each channel’s across-bin mean and SD → one scale per feature family (velocity, posture, dynamics), giving equal total variance to each family → one ant-profile PC → shared-variance Gaussian mixture. The twelve channels are signed forward/lateral velocity, forward/lateral RMS velocity, four learned posture-PC means, and four within-clip RMS PC derivatives. Antennal straightness is not an input. Bins average sampled clips, not continuous recordings: there is one 2.5-second clip per minute.</p>"""
    body += results.to_html(index=False, border=0)
    body += """<p>Bootstrap and time-split columns show median ARI, with the 10th percentile in parentheses. ARI=1 means identical partitions, apart from group names. Final ant bootstraps: 500; random paired half-hour time splits: 30. All 86 leave-one-ant-out fits preserve every assignment.</p>
<p><b>Is five minutes essential?</b> No. With the selected weighting, summary, and mixture, bin widths 1, 5, 15, and 30 minutes produce exactly the same full-data labels in both colonies. Widths 60–240 minutes preserve every left label and change one right label. Hourly bins can smooth brief bouts; the controlled results show that they were not the main cause of the earlier unstable partition. Five minutes was selected by worst-colony lower-tail resampling performance, not spatial agreement. Its precise optimality should not be overinterpreted.</p>
<p><b>How close is the spatial picture?</b> The repeatable split agrees with 39/41 left and 39/45 right spatial labels (78/86 total). The previous unequal-variance K=2 candidate agreed with 39/41 and 41/45 (80/86), so improved stability costs two right-colony spatial matches. It is a robust behavioral partition with substantial spatial correspondence, not proof that the two spatial classes are identical to two intrinsic behavioral types. The left-colony BIC improvement over K=1 is only 2.82; the right improvement is 9.57. Assignment repeatability is stronger than the evidence for two discrete populations, particularly on the left.</p>
<p><b>Further checks:</b> retaining only the two largest landmark PCs, applying a model trained on the other colony, increasing minimum coverage to 18/20/22 hours, and varying covariance regularization from 0.0001 to 0.1 with tighter convergence all preserve the compared labels. PC dynamics alone does not favor two groups by BIC. These are checks of the frozen choice; none was used to select a different spatially preferred model.</p>
<p><b>Selection and limits:</b> 152 prespecified combinations of bin width, distribution versus concatenated-time summaries, scaling, profile dimension, and mixture covariance were screened with 24 bootstraps. Fifteen configurations (the top eight plus seven original-profile bin controls) were validated with 200 bootstraps, leave-one-ant-out refits, and 30 disjoint temporal splits. Candidates required positive K=2 versus K=1 BIC gain, groups ≥5, bootstrap median ARI ≥0.8, and time-split median ≥0.6 in both colonies. We ranked the average of the worst-colony bootstrap and time-split 10th-percentile ARIs. Spatial labels and day2 were excluded from ranking and entered only after model hashes were frozen.</p>
<p>This is exploratory tuning on a previously examined dataset, with only two colonies and 41/45 eligible ants. Reusing these ants for ranking and robustness reporting is optimistic; the final 500 resampling draws are new draws, not new biological data. Day2 was also previously examined. The landmark basis is held fixed during this audit; its independent stability was assessed in the preceding analysis. Neither repeated sampling of individual ants nor cross-colony transfer substitutes for replication across additional colonies. Camera-specific pose bias is not newly audited here. The day1 eligibility rule requires ≥12 observed hours for all three feature families; all 114 tracked identities remain in the coverage audit. Each colony has 57 tracked identities. Missing ordered-bin values are imputed from training-ants’ column medians only; no missingness indicators are fit inputs.</p>"""
    for key, caption in captions:
        body += f'<figure><a href="figures/{key}.pdf"><img src="figures/{key}.png" alt="{html.escape(key)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'
    files = [
        "parameter_audit_figures.pdf",
        "screen.csv",
        "screen_ranked.csv",
        "validation.csv",
        "validation_ranked.csv",
        "validation_replicates.csv",
        "k_selection.csv",
        "assignments_evaluated.csv",
        "baseline_diagnosis.csv",
        "baseline_mixture_parameters.csv",
        "controls.csv",
        "controls_posthoc.csv",
        "sensitivity.csv",
        "physical_features.csv",
        "screen_posthoc_spatial.csv",
        "screen_protocol.json",
        "validation_protocol.json",
        "MODELS_FROZEN.json",
    ]
    body += (
        "<p>Download: " + " · ".join(f'<a href="{f}">{f}</a>' for f in files) + "</p>"
    )
    style = "body{font:16px/1.6 system-ui,sans-serif;color:#203040;max-width:1200px;margin:40px auto;padding:0 24px;background:#fbfcfd}h1{line-height:1.2;font-size:34px}.lead{font-size:21px}figure{margin:45px 0}img{width:100%;background:white}figcaption{font-size:14px;color:#465566}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}a{color:#176087}"
    (output / "index.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Posture PCs and velocity: parameter audit</title><style>{style}</style><body><h1>Posture PCs and velocity: why the split was unstable</h1><p>20260724 · block01 · parameter audit 2026-10-09</p>{body}</body></html>'
    )
    lines = [
        "# Posture PCs and velocity: parameter audit",
        "",
        "Hourly binning is not the main problem. Separate-variance Gaussian mixtures shift their boundary when ants are resampled. Shared-variance mixtures stabilize the partition.",
        "",
        f"Selected without spatial labels: `{cfg.key}`. Inputs remain learned raw-coordinate posture PCs, their derivatives, and body-axis velocities. No straightness descriptor.",
        "",
        results.to_string(index=False),
        "",
        "Bootstrap and temporal metrics are median ARI (10th percentile). See index.html for methods, limitations and six annotated figures.",
        "",
        "The new split agrees with 78/86 spatial labels versus 80/86 for the earlier, less stable candidate. The tradeoff is explicit: greater assignment stability, slightly poorer right-colony spatial agreement.",
        "",
        "Five-minute bins are not essential: 1–30 minutes produce identical full-data labels with the selected method. Hourly and longer bins change only one right label. Two landmark PCs suffice to preserve all assignments; cross-colony model transfer preserves all labels.",
        "",
        "BIC gain for K=2 over K=1: left 2.82 (modest), right 9.57. This exploratory analysis does not establish discrete intrinsic behavioral types or generalization to new colonies.",
        "",
        "Screen:152 configurations×24 bootstraps. Validation:15×200 bootstraps plus30 random temporal splits. Final model:500 new bootstrap draws. Day2 had already been examined previously and is not a prospective holdout.",
        "",
        "Reproduction instructions: analysis/eigenposture_parameters.md in the repository, archived under reproduction/code/analysis/.",
    ]
    for key, caption in captions:
        lines += ["", f"![{key}](figures/{key}.png)", "", caption]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    (output / "REPORT_COMPLETE.json").write_text(
        json.dumps(
            dict(
                figures=len(captions),
                configuration=cfg.key,
                source=str(source),
                occupancy=str(occupancy),
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--occupancy", type=Path, required=True)
    a = p.parse_args()
    report(a.source, a.output, a.occupancy)


if __name__ == "__main__":
    main()
