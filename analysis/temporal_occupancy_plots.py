"""Continuity, novelty and single-ant views of time-resolved spatial profiles."""
from __future__ import annotations

import base64
import gzip
import html
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex
import numpy as np
import pandas as pd

COLORS=["#4477AA", "#EE6677", "#228833", "#AA3377"]


def color(label):
    return COLORS[int(str(label).split("_")[-1])%len(COLORS)] if pd.notna(label) else "#aaa"


def save(fig, path):
    fig.savefig(path,dpi=150,bbox_inches="tight")
    plt.close(fig)


def dates(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x",labelrotation=25,labelsize=8)


def clock_shading(ax, manifest, start, stop):
    for day in pd.date_range(pd.Timestamp(start).normalize()-pd.Timedelta(days=1),pd.Timestamp(stop).normalize()):
        ax.axvspan(day+pd.Timedelta(hours=manifest["light_off_hour"]),day+pd.Timedelta(days=1,hours=manifest["light_on_hour"]),color=".92",zorder=-10)


def calendar_series(ax, part, column, *, hours=4, **kwargs):
    if part.empty: return
    p=part.drop_duplicates("bin").set_index("bin").sort_index()
    index=np.arange(p.index.min(),p.index.max()+1)
    x=pd.to_datetime((index+.5)*hours*3600,unit="s")
    ax.plot(x,p[column].reindex(index),**kwargs)


def reference_background(ax, ref):
    for label,p in ref.groupby("original_cluster"):
        ax.scatter(p.umap_x,p.umap_y,s=25,c=color(label),alpha=.22,marker="s",label=label)
    ax.set(xlabel="0723 UMAP 1",ylabel="0723 UMAP 2")


def limits(ref, table):
    x=pd.concat([ref.umap_x,table.umap_x]).dropna(); y=pd.concat([ref.umap_y,table.umap_y]).dropna()
    return (x.min()-.07*max(x.max()-x.min(),1),x.max()+.07*max(x.max()-x.min(),1)),(y.min()-.07*max(y.max()-y.min(),1),y.max()+.07*max(y.max()-y.min(),1))


def whole_projection(tables, output):
    ref=tables["reference"]; whole=tables["whole_projected"]; old=tables["whole_assignments"]
    for i,side in enumerate(("left","right")):
        fig,axes=plt.subplots(1,3,figsize=(16,5.5),squeeze=False)
        r=ref[ref.side.eq(side)]
        lim=limits(r,tables["profiles_4h"].query("side == @side and reference_cluster.notna()"))
        for j,b in enumerate((1,2,3)):
            ax=axes[0,j];reference_background(ax,r)
            if b==1:
                p=r; labels=p.original_cluster; novel=np.zeros(len(p),bool)
            else:
                p=whole[whole.block_index.eq(b)&whole.ant.isin(r.ant)].merge(old[["profile_key","prediction","in_reference_range"]],on="profile_key",validate="one_to_one")
                labels=p.prediction;novel=~p.in_reference_range
            ax.scatter(p.umap_x,p.umap_y,c=[color(v) for v in labels],s=42,edgecolors=np.where(novel,"black","none"),linewidths=1.1)
            ax.set(xlim=lim[0],ylim=lim[1],title=f"{side}: "+["0723 block02 reference","0724 recording","0729 window (Jul 30–31)"][j])
        fig.suptitle(f"{side.capitalize()} colony\n" + "Whole-recording maps transformed into the unchanged 0723 UMAP",fontsize=15)
        fig.text(.5,.01,"Faint squares: original reference. Black outlines: beyond original whole-block distance range. UMAP distances are visual; novelty is measured in the full grid space.",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.96));save(fig,output/f"01_{side}_whole_recording_umap_projection.png")


def day_projections(tables, output):
    table=tables["profiles_4h"].query("reference_cluster.notna() and prediction.notna()").copy()
    table["day"]=table.center.dt.strftime("%Y-%m-%d")
    days=sorted(table.day.unique())
    for side in ("left","right"):
        p=table[table.side.eq(side)];r=tables["reference"][tables["reference"].side.eq(side)]
        lim=limits(r,p)
        fig,axes=plt.subplots(2,3,figsize=(16,10),squeeze=False)
        for ax,day in zip(axes.flat,days):
            reference_background(ax,r);d=p[p.day.eq(day)]
            good=d[d.reliable];other=d[~d.reliable]
            ax.scatter(other.umap_x,other.umap_y,c=[color(v) for v in other.prediction],marker="x",alpha=.35,s=18)
            ax.scatter(good.umap_x,good.umap_y,c=[color(v) for v in good.prediction],s=22+25*good.coverage,
                       edgecolors=np.where(good.novelty_ratio.gt(1),"black","none"),linewidths=.8,alpha=.75)
            ax.set(xlim=lim[0],ylim=lim[1],title=f"{day}: {d.bin.nunique()} windows, {d.ant.nunique()} ants")
        for ax in list(axes.flat)[len(days):]:ax.set_visible(False)
        fig.suptitle(f"{side}: four-hour profiles in the fixed reference UMAP",fontsize=15)
        fig.text(.5,.01,"Colors: fixed-reference KNN labels. Black outlines: beyond duration-matched reference range. ×: partial window or <40% detected. Each point is one ant in one window.",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.96));save(fig,output/f"02_{side}_daily_four_hour_umap.png")


def novelty(tables, manifest, output):
    p=tables["profiles_4h"].query("reference_cluster.notna() and prediction.notna()")
    for j,side in enumerate(("left","right")):
        fig,axes=plt.subplots(2,1,figsize=(12,9),sharex=True,squeeze=False)
        d=p[p.side.eq(side)]
        for mode,rows,style in [("All observed",d,"-"),("Well observed full windows",d[d.reliable],"--")]:
            grouped=rows.groupby("bin")
            stats=grouped.agg(center=("center","first"),median=("novelty_ratio","median"),n=("ant","size"))
            stats["matched_fraction"]=grouped.novelty_ratio.apply(lambda x:100*x.gt(1).mean())
            stats["original_fraction"]=grouped.whole_distance_ratio.apply(lambda x:100*x.gt(1).mean())
            stats=stats.reset_index()
            calendar_series(axes[0,0],stats,"matched_fraction",color=COLORS[j],ls=style,label=mode+", matched duration")
            if mode=="All observed":
                calendar_series(axes[0,0],stats,"original_fraction",color=".6",lw=1,label="Original whole-block cutoff")
            calendar_series(axes[1,0],stats,"median",color=COLORS[j],ls=style,label=mode)
        axes[0,0].set(title=side,ylabel="Profiles beyond reference range (%)",ylim=(-2,102))
        axes[1,0].set(ylabel="Median distance / 4 h reference P95");axes[1,0].axhline(1,color="black",ls=":")
        for ax in axes[:,0]:
            clock_shading(ax,manifest,p.start.min(),p.stop.max());dates(ax);ax.legend(fontsize=7)
        fig.suptitle(f"{side.capitalize()} colony\n" + "Does the later colony leave the reference's range of spatial behavior?",fontsize=15)
        fig.text(.5,.005,"Duration-matched reference range uses complete 4 h windows inside 0723 block02, excluding each query ant's own baseline map. Gaps remain blank.",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.035,1,.96));save(fig,output/f"03_{side}_novelty_over_calendar_time.png")


def recording_cohort_behavior(tables):
    """Join each ant's immutable reference label to its observed block means."""
    reference = tables["reference"][["side", "ant", "original_cluster"]].rename(
        columns={"original_cluster": "baseline_cluster"})
    if reference.duplicated(["side", "ant"]).any():
        raise ValueError("Expected one original cluster per colony and ant")
    whole = tables["whole_projected"].drop(columns=["baseline_cluster"], errors="ignore")
    if whole.duplicated(["side", "ant", "block_index"]).any():
        raise ValueError("Expected one observed behavior profile per ant and recording period")
    return whole.merge(reference, on=["side", "ant"], how="inner", validate="many_to_one").sort_values(
        ["side", "ant", "start"])


def novelty_activity(tables, output):
    """Figure 7: follow original groups in directly observed behavior units."""
    output = Path(output)
    data = recording_cohort_behavior(tables)
    data.to_csv(output / "07_original_cluster_behavior_by_period.csv", index=False)
    periods = tables["whole_projected"].drop_duplicates("block_index").sort_values("start")
    blocks = periods.block_index.to_list()
    reference = tables["reference"]
    reference_block = int(reference.block_index.iloc[0])
    reference_label = reference.block_label.iloc[0]
    speed_max = max(0.1, float(data.mean_speed_mm_s.max()) * 1.08)
    labels = [f"{pd.Timestamp(row.start):%b %d %H:%M}–\n{pd.Timestamp(row.stop):%b %d %H:%M}"
              for row in periods.itertuples()]
    for side in ("left", "right"):
        part = data[data.side.eq(side)]
        ref = reference[reference.side.eq(side)]
        clusters = sorted(ref.original_cluster.unique())
        fig = plt.figure(figsize=(5 * len(blocks), 10), layout="constrained")
        grid = fig.add_gridspec(2, 2 * len(blocks), height_ratios=[1.25, 1])
        for column, row in enumerate(periods.itertuples()):
            ax = fig.add_subplot(grid[0, 2 * column:2 * column + 2])
            observed = part[part.block_index.eq(row.block_index)]
            for cluster in clusters:
                group = observed[observed.baseline_cluster.eq(cluster)].dropna(
                    subset=["colony_percent", "mean_speed_mm_s"])
                for good, subset in group.groupby(group.coverage.ge(.4)):
                    ax.scatter(subset.colony_percent, subset.mean_speed_mm_s, s=40,
                               facecolors=color(cluster) if good else "none",
                               edgecolors=color(cluster), alpha=.75, linewidths=1)
            role = "Reference labels" if row.block_index == reference_block else (
                "Earlier observation" if row.block_index < reference_block else "Later observation")
            n = observed.dropna(subset=["colony_percent", "mean_speed_mm_s"]).ant.nunique()
            ax.set(title=f"{labels[column]}\n{role} · {n} ants", xlim=(-2, 102), ylim=(0, speed_max),
                   xlabel="Time inside colony (%)", ylabel="Mean movement speed (mm/s)")
            ax.grid(alpha=.18)
        for metric, ylabel, columns in [
            ("colony_percent", "Time inside colony (%)", slice(0, len(blocks))),
            ("mean_speed_mm_s", "Mean movement speed (mm/s)", slice(len(blocks), 2 * len(blocks))),
        ]:
            ax = fig.add_subplot(grid[1, columns])
            for cluster_index, cluster in enumerate(clusters):
                group = part[part.baseline_cluster.eq(cluster)]
                wide = group.pivot(index="ant", columns="block_index", values=metric).reindex(columns=blocks)
                ax.plot(range(len(blocks)), wide.to_numpy().T, color=color(cluster), alpha=.14, lw=.7)
                ax.plot(range(len(blocks)), wide.median(), "o-", color=color(cluster), lw=2.5,
                        label=f"Original cluster {str(cluster).rsplit('_', 1)[-1]}")
                for x, count in enumerate(wide.count()):
                    if count:
                        ax.annotate(f"n={count}", (x, wide.median().iloc[x]), xytext=(5, 6 if cluster_index % 2 == 0 else -12),
                                    textcoords="offset points", fontsize=8, color=color(cluster))
            ax.set(xticks=range(len(blocks)), xticklabels=labels, ylabel=ylabel,
                   title="Same ants through recording periods; bold = original-group median")
            ax.set_ylim((-2, 105) if metric == "colony_percent" else (0, speed_max))
            ax.grid(alpha=.18)
            ax.legend(fontsize=9)
        fig.suptitle(f"{side.capitalize()} colony — follow the original clusters through every recording\n"
                     f"Colors stay fixed to {reference_label}; one dot = one ant's observed period mean", fontsize=16)
        fig.supxlabel("Hollow dots: <40% detected. Thin lines follow individual ants; missing values break lines. "
                       "Periods have unequal duration and clock coverage.\n"
                       "Only ants with original reference labels are shown. Counts give observed ants in each group; "
                       "later cluster assignments do not change their colors.", fontsize=10)
        save(fig, output / f"07_{side}_original_clusters_across_recordings.png")


def geometry_and_partitions(tables, manifest, output):
    steps=tables["steps_4h"];parts=tables["partition_continuity"]
    for j,side in enumerate(("left","right")):
        fig,axes=plt.subplots(3,1,figsize=(12,12),sharex=True,squeeze=False)
        s=steps[steps.side.eq(side)&steps.reference_ant]
        for reliable,style,label in [(False,"-","All observed"),(True,"--","Both windows well observed")]:
            d=s[s.reliable] if reliable else s
            stats=d.groupby("bin").agg(center=("center","first"),median=("feature_step","median"),n=("ant","size")).reset_index()
            calendar_series(axes[0,0],stats,"median",color=COLORS[j],ls=style,label=label)
        local=tables["local_clusters"].query("side == @side").groupby("bin").size().rename("n_clusters").reset_index()
        calendar_series(axes[1,0],local,"n_clusters",color=COLORS[j],marker="o",ms=3)
        for mode,d in parts[parts.side.eq(side)].groupby("mode"):
            calendar_series(axes[2,0],d,"adjusted_rand",color=COLORS[j] if mode=="all observed" else "black",ls="-" if mode=="all observed" else "--",label=mode)
        axes[0,0].set(title=side,ylabel="Median step in original √ grid features")
        axes[1,0].set(ylabel="Independent Leiden clusters / window")
        axes[2,0].set(ylabel="Partition continuity (adjusted Rand)",ylim=(-.1,1.05))
        for ax in axes[:,0]:dates(ax);ax.legend(fontsize=8) if ax.get_legend_handles_labels()[0] else None
        fig.suptitle(f"{side.capitalize()} colony\n" + "Continuous feature movement and independently fitted four-hour clusters",fontsize=15)
        fig.text(.5,.005,"Steps and partition comparisons only join adjacent observed windows with ≤5 min recording gap. ARI=1 means identical grouping; numeric cluster IDs are never matched across fits.",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.035,1,.96));save(fig,output/f"04_{side}_feature_steps_and_partition_continuity.png")


def ant_heatmaps(tables, output):
    p=tables["profiles_4h"]
    for side in ("left","right"):
        ref=tables["reference"].query("side == @side").sort_values(["original_cluster","ant"])
        d=p[p.ant.isin(ref.ant)].copy();d["cluster_code"]=d.prediction.map(lambda s:float(str(s).split("_")[-1]) if pd.notna(s) else np.nan)
        bins=np.arange(d.bin.min(),d.bin.max()+1)
        fig,axes=plt.subplots(1,3,figsize=(19,12),sharey=True)
        for ax,(metric,title,cmap,lo,hi) in zip(axes,[("cluster_code","Fixed reference assignment",ListedColormap(COLORS[:2]),-.5,1.5),("novelty_ratio","Distance / matched reference P95","magma",0,2),("coverage","Detected fraction of recorded time","viridis",0,1)]):
            wide=d.pivot(index="ant",columns="bin",values=metric).reindex(index=ref.ant,columns=bins)
            im=ax.imshow(wide,aspect="auto",interpolation="none",cmap=cmap,vmin=lo,vmax=hi)
            tick=np.flatnonzero(bins%6==0);ax.set_xticks(tick,pd.to_datetime(bins[tick]*4*3600,unit="s").strftime("%b %d"),rotation=70,fontsize=8)
            ax.set_yticks(range(len(ref)),ref.ant,fontsize=7);ax.set_title(title);fig.colorbar(im,ax=ax,orientation="horizontal",pad=.12,fraction=.025)
        fig.suptitle(f"{side}: every reference ant across four-hour windows",fontsize=15)
        fig.text(.5,.005,"White cells have no observed map. Calendar spacing preserves recording gaps. Detection coverage is relative to recorded time; partial recording windows remain included.",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.035,1,.96));save(fig,output/f"05_{side}_all_ant_assignments_novelty_coverage.png")


def model_overlay(ax,tables,ant,metric):
    choices=tables["change_models"]
    if choices.empty:return
    best=choices[(choices.ant==ant)&(choices.metric==metric)&choices.delta_bic.eq(0)]
    for row in best.itertuples():
        fits=tables["model_fits"].query("ant == @ant and metric == @metric and segment == @row.segment and model == @row.model")
        calendar_series(ax,fits,"fitted",hours=1,color="black",ls="--",lw=1,label=row.model+f" (ΔBIC {row.winning_margin:.1f})")


def ant_detail(ant,tables,manifest,output):
    four=tables["profiles_4h"].query("ant == @ant").sort_values("bin")
    one=tables["profiles_1h"].query("ant == @ant").sort_values("bin")
    side=ant.split(":")[0];ref=tables["reference"].query("side == @side")
    fig=plt.figure(figsize=(15,17));grid=fig.add_gridspec(7,3)
    axmap=fig.add_subplot(grid[:2,0]);reference_background(axmap,ref)
    valid=four[four.prediction.notna()].copy();valid["group"]=((valid.bin.diff()!=1)|(valid.first_observed-valid.last_observed.shift()).dt.total_seconds().gt(300)).cumsum()
    for _,part in valid.groupby("group"):
        axmap.plot(part.umap_x,part.umap_y,color=".6",lw=1,zorder=0)
    points=axmap.scatter(valid.umap_x,valid.umap_y,c=mdates.date2num(valid.center),cmap="viridis",s=40,edgecolors=np.where(valid.novelty_ratio.gt(1),"black","none"))
    axmap.set_title("Four-hour UMAP path\nBlack outline: beyond reference range",fontsize=9)
    cb=fig.colorbar(points,ax=axmap,orientation="horizontal",pad=.14);cb.ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"));cb.ax.tick_params(labelsize=7,rotation=30)
    axes=[fig.add_subplot(grid[i,1:] if i<2 else grid[i,:]) for i in range(7)]
    calendar_series(axes[0],one,"reference_axis",hours=1,color=COLORS[0],lw=1,label="1 h continuous feature score")
    calendar_series(axes[0],four,"reference_axis",color=COLORS[1],marker="o",ms=3,label="4 h score")
    model_overlay(axes[0],tables,ant,"reference_axis")
    axes[0].axhline(0,color=".5",ls=":");axes[0].set_ylabel("Reference axis\n← cluster 0 | cluster 1 →")
    calendar_series(axes[1],four,"cluster1_vote",color=COLORS[1],marker="s",ms=3,label="KNN cluster 1 vote")
    axes[1].set(ylim=(-.05,1.05),ylabel="KNN vote fraction")
    calendar_series(axes[2],four,"whole_distance_ratio",color=".65",label="Whole-block reference cutoff")
    calendar_series(axes[2],four,"novelty_ratio",color=COLORS[0],marker="o",ms=3,label="4 h reference cutoff")
    axes[2].axhline(1,color="black",ls=":");axes[2].set_ylabel("Novelty distance / P95")
    steps=tables["steps_4h"].query("ant == @ant")
    calendar_series(axes[3],steps,"feature_step",color=COLORS[2],marker="o",ms=3,label="Adjacent 4 h feature distance")
    axes[3].set_ylabel("Size of feature step")
    calendar_series(axes[4],one,"colony_percent",hours=1,color=COLORS[0],label="Inside colony (%)")
    model_overlay(axes[4],tables,ant,"colony_percent");axes[4].set(ylabel="Inside colony (%)",ylim=(-3,103))
    calendar_series(axes[5],one,"mean_speed_mm_s",hours=1,color=COLORS[2],label="Mean locomotor speed")
    axes[5].set_ylabel("Speed (mm/s)")
    calendar_series(axes[6],one,"coverage",hours=1,color=COLORS[0],label="Detected / recorded time")
    calendar_series(axes[6],one,"recording_fraction",hours=1,color=".6",label="Recorded fraction of hour")
    axes[6].axhline(.4,color="black",ls=":");axes[6].set(ylabel="Observation coverage",ylim=(-.03,1.05))
    for ax in axes:
        clock_shading(ax,manifest,one.start.min(),one.stop.max());dates(ax);ax.legend(fontsize=7,loc="best")
        ax.set_xlim(one.start.min()-pd.Timedelta(hours=2),one.stop.max()+pd.Timedelta(hours=2))
    baseline=ref[ref.ant.eq(ant)].original_cluster.iloc[0]
    fig.suptitle(f"{ant} — original {baseline}: drift, jumps, rhythms and observation",fontsize=16)
    fig.text(.5,.003,"Gray background: lights off. Gaps remain blank. Step/drift fits include clock rhythm and lights-on effects; BIC is descriptive with autocorrelated data.\nLow-coverage points remain visible, but fitted models use ≥40% detection and ≥95% recorded hours. UMAP movement alone is not evidence of a discrete behavioral transition.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.975));save(fig,output/"individual_ants"/(ant.replace(":","_")+".png"))


def filmstrip(ant,tables,maps,edges,output):
    p=tables["profiles_4h"].query("ant == @ant and occupancy_sum > 0").sort_values("bin")
    n=len(p);cols=8;rows=int(np.ceil(n/cols));side=ant.split(":")[0];x,y=edges[side]
    images=[np.sqrt(maps[key]) for key in p.profile_key]
    vmax=max(float(im.max()) for im in images)
    fig,axes=plt.subplots(rows,cols,figsize=(18,rows*3.6),squeeze=False)
    for ax,row,im in zip(axes.flat,p.itertuples(),images):
        ax.imshow(im,origin="lower",extent=[x[0],x[-1],y[0],y[-1]],cmap="magma",vmin=0,vmax=vmax)
        ax.set_title(row.start.strftime("%b %d %H:%M")+f"\n{row.prediction}; vote {row.vote:.0%}\ncoverage {row.coverage:.0%}; novelty {row.novelty_ratio:.2f}",fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[n:]:ax.set_visible(False)
    fig.suptitle(f"{ant}: observed four-hour occupancy maps in chronological order",fontsize=15)
    fig.text(.5,.005,"One shared √ occupancy color scale. Columns are consecutive available windows; inspect timestamps for recording gaps. Partial windows are retained.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.96));save(fig,output/"filmstrips"/(ant.replace(":","_")+".png"))


def novelty_examples(tables,maps,edges,output):
    old=tables["whole_assignments"]
    selected=old[(old.block_index==3)&~old.switch&~old.in_reference_range&old.coverage.ge(.4)&old.baseline_coverage.ge(.4)]
    picked=selected.sort_values("distance_ratio",ascending=False).groupby("side",sort=True).head(2)
    if picked.empty:return
    for side, side_picked in picked.groupby("side", sort=True):
        fig,axes=plt.subplots(len(side_picked),3,figsize=(13,4.1*len(side_picked)),squeeze=False)
        whole=tables["whole_projected"]
        for i,row in enumerate(side_picked.itertuples()):
            first=whole[whole.ant.eq(row.ant)&whole.block_index.eq(1)].iloc[0]
            images=[np.sqrt(maps["whole|"+first.profile_key]),np.sqrt(maps["whole|"+row.profile_key])]
            vmax=max(float(v.max()) for v in images);delta=images[1]-images[0]
            x,y=edges[row.side]
            titles=[f"{row.ant}: 0723 reference\n{row.reference_cluster}; colony {first.colony_percent:.1f}%",
                    f"Final window: {row.prediction}\ncolony {row.colony_percent:.1f}%; distance / P95 {row.distance_ratio:.2f}",
                    "Change in √ occupancy\nBlue: less use; red: more use"]
            for j,im in enumerate(images+[delta]):
                limit=float(np.abs(delta).max()) if j==2 else vmax
                obj=axes[i,j].imshow(im,origin="lower",extent=[x[0],x[-1],y[0],y[-1]],cmap="coolwarm" if j==2 else "magma",vmin=-limit if j==2 else 0,vmax=limit)
                axes[i,j].set(title=titles[j],xlabel="Arena x (mm)",ylabel="Arena y (mm)")
                fig.colorbar(obj,ax=axes[i,j],fraction=.03,pad=.02)
        fig.suptitle(f"{side.capitalize()} colony\n" + "Novel occupancy patterns among ants retaining their reference label",fontsize=15)
        fig.text(.5,.005,"Examples selected from final whole-recording profiles outside the original reference range, with ≥40% detection at both endpoints. Exact maps and common row scales.",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.03,1,.97));save(fig,output/f"06_{side}_novel_occupancy_with_stable_labels.png")


def save_overview_figures(tables,maps,edges,manifest,output):
    """Redraw colony summaries from saved tables without rerendering every ant."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    novelty_activity(tables,output)
    whole_projection(tables,output);day_projections(tables,output);novelty(tables,manifest,output)
    geometry_and_partitions(tables,manifest,output);ant_heatmaps(tables,output)
    novelty_examples(tables,maps,edges,output)


def save_figures(tables,maps,edges,manifest,output):
    output=Path(output)
    save_overview_figures(tables,maps,edges,manifest,output)
    (output/"individual_ants").mkdir(exist_ok=True);(output/"filmstrips").mkdir(exist_ok=True)
    for i,ant in enumerate(tables["reference"].ant,1):
        ant_detail(ant,tables,manifest,output)
        if i%10==0:print("INDIVIDUAL_FIGURES",i,flush=True)
    # Every previous whole-recording switch plus every newly supported 4 h switch.
    picked=set(tables["whole_assignments"].loc[tables["whole_assignments"].switch,"ant"])
    picked.update(tables["profiles_4h"].loc[tables["profiles_4h"].supported_switch,"ant"])
    for ant in sorted(picked):filmstrip(ant,tables,maps,edges,output)


def records(table):
    return json.loads(table.to_json(orient="records",date_format="iso",double_precision=7))


def write_explorer(tables,maps,edges,manifest,output):
    from plotly.offline import get_plotlyjs
    keep4=["profile_key","bin","ant","side","start","stop","center","first_observed","last_observed","coverage","recording_fraction","prediction","reference_cluster","vote","cluster1_vote","reference_axis","novelty_ratio","whole_distance_ratio","umap_x","umap_y","local_cluster","reliable","supported_switch","colony_percent","mean_speed_mm_s"]
    keep1=[c for c in keep4 if c not in ["local_cluster","umap_x","umap_y"]]+["mean_trip_minutes","trip_rate"]
    data=dict(reference=records(tables["reference"]),four=records(tables["profiles_4h"][keep4]),one=records(tables["profiles_1h"][keep1]),
              windows=records(tables["windows_4h"]),models=records(tables["change_models"]),steps=records(tables["steps_4h"]),partition=records(tables["partition_continuity"]),
              maps={key:np.round(value,8).tolist() for key,value in maps.items() if key.startswith("4|")},
              edges={side:[v.tolist() for v in pair] for side,pair in edges.items()},colors=COLORS,
              magma=[[float(t),to_hex(plt.get_cmap("magma")(t))] for t in np.linspace(0,1,20)])
    compressed=base64.b64encode(gzip.compress(json.dumps(data,separators=(",",":")).encode())).decode()
    template=Path(__file__).with_name("temporal_occupancy_dashboard.html").read_text()
    (output/"interactive.html").write_text(template.replace("__PLOTLY__",get_plotlyjs()).replace("__DATA__",compressed))


def write_report(tables,manifest,output):
    output=Path(output);four=tables["profiles_4h"];ref=four[four.reference_cluster.notna()&four.prediction.notna()]
    lines=["# Four-hour occupancy dynamics", "", "The fixed space and saved labels come from 0723/block02. Reference UMAP coordinates were reproduced exactly (tolerance 1e-5); all later maps use that fitted UMAP's transform. KNN labels use the original full-dimensional square-root grids. Independent Leiden clustering is also fitted in every four-hour window with the original settings (10 neighbors, resolution 1, seed 0). Its numeric IDs are local, and are not treated as switches.", "",
           "Calendar-aligned windows are 00–04, 04–08, 08–12, 12–16, 16–20, and 20–24 local recording time. Near-continuous source blocks contribute to the same window. Real gaps and partial windows remain explicit. Exact half-hour count caches reproduce all 456 original histograms; one-hour supplementary profiles reuse these counts.", "",
           "## Beyond the reference", "", "| Colony | Original whole-map P95 distance | Four-hour matched P95 | Reference calibration profiles |", "|---|---:|---:|---:|"]
    for side,c in tables["calibration_4h"].groupby("side"):
        lines.append(f"| {side} | {c.whole_block_cutoff.iloc[0]:.3f} | {c.cutoff.iloc[0]:.3f} | {len(c)} |")
    lines += ["", "The earlier 32/86 count used the whole-recording cutoff and whole-recording maps. Four-hour maps naturally have greater sampling and time-of-day variation. Both the original cutoff and a duration-matched cutoff are shown. The latter is calibrated on complete four-hour windows inside the reference, excluding each ant's own baseline neighbor. A separate >=40% detection calibration is retained in tables. Novelty is measured in full grid space, not by apparent UMAP separation.", "",
              "| Observed calendar date | Colony | Reference-ant windows assigned | Beyond 4 h range | Well-observed windows | Beyond range among well-observed |", "|---|---|---:|---:|---:|---:|"]
    for (day,side),p in ref.groupby([ref.center.dt.strftime("%Y-%m-%d"),"side"]):
        good=p[p.reliable]
        lines.append(f"| {day} | {side} | {len(p)} | {p.novelty_ratio.gt(1).sum()} | {len(good)} | {good.novelty_ratio.gt(1).sum()} |")
    lines += ["", "Rows count ant-window observations, not independent ants. 'Well observed' means >=40% detection and >=95% of the window recorded; all other maps remain in the main tables and interactive views.", "",
              "The strongest whole-map novelty examples include near-stationary ants: left:004, left:010 and right:057 retain cluster 1 but concentrate occupancy into a few locations. Their final mean speeds are 0.0038, 0.0432 and 0.0063 mm/s, versus 1.42, 1.04 and 0.98 mm/s in the reference. The saved sleep classifier marks 98%, 94% and 72% of their observed time respectively; this does not establish normal sleep or explain the immobility. Thus novelty includes reduced movement as well as spatial redistribution, and cannot be equated with new task allocation. Figure 07 follows the original July 23 block02 groups through every recording period using colony presence (%) and movement speed (mm/s). Colors remain fixed to original labels; each point is one ant’s observed block mean, and thin lines connect the same ant across periods. Hollow points mark <40% detection. Periods differ in duration and clock coverage; missing values remain missing.", "",
              "## The two earlier candidates", "",
              "left:036 shows a persistent transition during the evening of July 24. Colony presence falls from nearly 100% at 16:00–18:00 to 53% at 20:00 and 1% at 21:00; its fixed label changes to cluster 1. The continuous feature score also changes over those few hours. A step describes the longer record better than a linear drift after accounting for daily rhythm, but detection is only about 25% at 18:00–19:00. The observations support a rapid change over hours, not an instantaneous jump.", "",
              "right:052 also changes during the first few observed hours of the later recording, on July 30. The initial 21:00 bin contains only 11 minutes of recording and remains cluster 0 with 100% colony presence. From 22:00, its label becomes cluster 1 while colony presence falls through 53%, 36%, and 19% in successive hours. The beginning is partly observed; the preceding multi-day gap leaves its earlier history unknown. Neither a label crossing nor a good descriptive step fit establishes a discrete biological task state.", ""]
    for ant in ("left:036","right:052"):
        p=ref[ref.ant.eq(ant)].sort_values("bin")
        lines += [f"### {ant}", "", "| Window begins | Assigned cluster | Vote | Colony % | Detected fraction | Recorded fraction | Distance / 4 h P95 |", "|---|---|---:|---:|---:|---:|---:|"]
        for r in p.itertuples():
            lines.append(f"| {r.start:%b %d %H:%M} | {r.prediction} | {r.vote:.0%} | {r.colony_percent:.1f} | {r.coverage:.0%} | {r.recording_fraction:.0%} | {r.novelty_ratio:.2f} |")
        one=tables["profiles_1h"]
        start,stop=("2026-07-24 16:00","2026-07-25 00:00") if ant=="left:036" else ("2026-07-30 21:00","2026-07-31 02:00")
        detail=one.loc[one.ant.eq(ant)&one.start.ge(pd.Timestamp(start))&one.start.lt(pd.Timestamp(stop))]
        lines += ["", "Hourly detail around the observed transition:", "", "| Hour begins | Fixed label | Continuous axis | Colony % | Detected fraction | Recorded minutes |", "|---|---|---:|---:|---:|---:|"]
        for r in detail.itertuples():
            lines.append(f"| {r.start:%b %d %H:%M} | {r.prediction} | {r.reference_axis:.3f} | {r.colony_percent:.1f} | {r.coverage:.0%} | {60*r.recording_fraction:.1f} |")
        lines += [""]
        fits=tables["change_models"].query("ant == @ant and delta_bic == 0")
        for r in fits.itertuples():
            lines.append(f"- {r.metric}, {r.start:%b %d %H:%M}–{r.stop:%b %d %H:%M}: {r.model}, BIC advantage {r.winning_margin:.1f}; change {r.change:.3g}; candidate breakpoint {r.break_time} (neighboring observations {r.break_gap_hours:g} h apart).")
    lines += ["", "## Reading continuity and jumps", "",
              "A hard KNN label can change when a smoothly moving profile crosses its decision boundary. Read the continuous reference axis (relative distance to the two reference prototypes), the original-feature step size, colony occupancy, and activity together. Independent partition continuity uses adjusted Rand index, so relabeling a cluster cannot create a false jump. UMAP can distort trajectory shapes and distances and cannot by itself demonstrate discrete states.", "",
              "Hourly diagnostic fits compare circadian-only behavior, gradual linear drift, and a single step. All include a daily sine/cosine and lights-on indicator; the searched breakpoint incurs an extra BIC parameter. Fits use >=40% detected and >=95% recorded hours in contiguous observed runs, with at least 12 observations. BIC remains descriptive because hourly samples are autocorrelated. A transition inside a recording gap is unobserved and cannot be classified as gradual or sudden. Breakpoint gaps, fit residuals, all alternatives and coverage are saved.", "",
              "Open interactive.html for a four-hour window selector, fixed UMAP, individual trajectories and neighboring occupancy maps. There is one detailed PNG for every reference ant in individual_ants/. Filmstrips cover every previous whole-recording switch and every supported four-hour switch. Optional trip caches are incomplete for some ants; NA does not mean zero trips."]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    figures="".join(f'<h2>{html.escape(p.stem.replace("_"," "))}</h2><a href="{p.name}"><img loading="lazy" src="{p.name}"></a>' for p in sorted(output.glob("*.png")))
    ants=" · ".join(f'<a href="individual_ants/{a.replace(":","_")}.png">{a}</a>' for a in tables["reference"].ant)
    films=" · ".join(f'<a href="filmstrips/{p.name}">{p.stem}</a>' for p in sorted((output/"filmstrips").glob("*.png")))
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Four-hour ant occupancy dynamics</title><style>body{font:16px system-ui;margin:30px;line-height:1.5}img{width:100%;max-width:1300px}pre{white-space:pre-wrap;max-width:1200px}</style><h1>Four-hour occupancy: continuity and jumps</h1><p><a href="interactive.html">Open interactive UMAP and ant explorer</a> · <a href="report.md">Methods and findings</a> · <a href="profiles_4h.csv">Every four-hour profile</a> · <a href="change_models.csv">Hourly model comparisons</a></p><details><summary>Detailed figures for all reference ants</summary>'+ants+'</details><details><summary>Occupancy filmstrips</summary>'+films+'</details>'+figures+'<pre>'+html.escape("\n".join(lines))+'</pre>')
