"""Audited, removable exploration of 20260624/block02 task proxies.

Run with --dataset PATH. Outputs live below that dataset's analysis_outputs.
No inherited colony/resource labels are used. Positions and speed are reported
in panorama pixels because the physical calibration must be checked per block.
The annotation audit and follow-up clock windows are specific to this recording;
they must be revisited before using this exploratory script on another block.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


def metadata(dataset):
    rows = []
    for path in sorted((dataset / "stitched/speed_vectors").rglob("speed_metadata.json")):
        record = json.loads(path.read_text())
        record["side"] = "left" if "_left" in record["track_name"] else "right"
        record["speed_path"] = str(path.parent / "speed_mm_s.npy")
        rows.append(record)
    table = pd.DataFrame(rows)
    n_frames = int(table.frame_max.max()) + 1
    table["recording_coverage"] = table.n_observed_frames / n_frames
    return table, n_frames


def parse_pulses(dataset):
    rows = []
    fields = "iso_time trial duty dur_s interval_s frame_start frame_end fs_hz samples gyro_rms gyro_peak acc_rms acc_peak temperature".split()
    for path in dataset.glob("sess_*.txt"):
        for line in path.read_text().splitlines():
            if "CSV_PULSE," not in line:
                continue
            values = line.split("CSV_PULSE,", 1)[1].split(",")
            if values[0] == "iso_time":
                continue
            record = dict(zip(fields, values))
            for field in fields[1:]:
                record[field] = float(record[field])
            rows.append(record)
    return pd.DataFrame(rows)


def read_ant(record, dataset, n_frames, fps):
    width = int(5 * fps)
    n_bins = int(np.ceil(n_frames / width))
    dense = np.full(n_bins * width, np.nan, dtype=np.float32)
    raw = np.load(record.speed_path, mmap_mode="r")
    dense[int(record.frame_min):int(record.frame_min) + len(raw)] = raw / record.mm_per_px
    blocks = dense.reshape(-1, width)
    counts = np.isfinite(blocks).sum(axis=1)
    means = np.divide(np.nansum(blocks, axis=1), counts, out=np.full(n_bins, np.nan), where=counts > 0)
    means[counts < width * .25] = np.nan
    coverage = counts / width

    path = dataset / "stitched/per_track" / record.track_name
    scanner = ds.dataset(path).scanner(columns=["Frame", "TrackX", "TrackY"], filter=ds.field("Bodypoint") == 0)
    parts = []
    for batch in scanner.to_batches():
        frames = batch.column(0).to_numpy()
        keep = frames % int(fps) == 0
        if keep.any():
            parts.append(np.column_stack([frames[keep], batch.column(1).to_numpy()[keep], batch.column(2).to_numpy()[keep]]))
    values = np.concatenate(parts)
    positions = pd.DataFrame(values, columns=["frame", "x_px", "y_px"]).dropna().drop_duplicates("frame")
    positions["track_name"] = record.track_name
    positions["track_id"] = record.track_id
    positions["side"] = record.side
    return record.track_name, means, coverage, positions


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label_clock(ax, start, duration):
    ticks = np.arange(np.ceil(start / 3600), np.floor(start / 3600 + duration) + 1, 2)
    ax.set_xticks(ticks - start / 3600, [f"{int(t)%24:02d}:00" for t in ticks])
    ax.set_xlabel("Clock time (recording crosses midnight)")


def bootstrap_mean_ci(values, rng, count=2000):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return [np.nan, np.nan]
    return np.quantile(values[rng.integers(len(values), size=(count, len(values)))].mean(axis=1), [.025, .975]).tolist()


def nanmean(values, axis=None):
    """Leave unobserved ant/windows missing without noisy empty-slice warnings."""
    values = np.asarray(values)
    counts = np.isfinite(values).sum(axis=axis)
    return np.divide(np.nansum(values, axis=axis), counts,
                     out=np.full(np.shape(counts), np.nan), where=counts > 0)


def follow_up_checks(output, chosen, speed, coverage, positions, pulses, t, start, rng):
    """Check stimulus adaptation and the shared morning activity increase."""
    import statsmodels.formula.api as smf

    results = {}
    responses = pd.read_csv(output / "pulse_responses.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, side in enumerate(["left", "right"]):
        r = responses[(responses.side == side) & responses.adequate & responses.sham_adequate].copy()
        r["trial_z"] = (r.trial - 15.5) / 10
        r["duty_z"] = (r.duty - .6) / .2
        r["baseline_z"] = r.baseline / 30
        fit = smf.ols("post ~ baseline_z + I(baseline_z**2) + duty_z + trial_z + pre_coverage + post_coverage + C(track_id)", r).fit(cov_type="cluster", cov_kwds={"groups": r.trial})
        ci = fit.conf_int().loc["trial_z"].tolist()
        # Fit nuisance effects separately to independent halves, then compare
        # per-ant residual responses; ant identity is not used in these fits.
        r["half"] = np.where(r.trial <= 15, 0, 1)
        for half, half_data in r.groupby("half"):
            nuisance = smf.ols("post ~ baseline_z + I(baseline_z**2) + duty_z + trial_z + pre_coverage + post_coverage", half_data).fit()
            r.loc[half_data.index, "adjusted_response"] = nuisance.resid
        repeat = r.groupby(["track_id", "half"]).adjusted_response.agg(["mean", "count"]).unstack()
        repeat_ok = (repeat["count"] >= 8).all(axis=1)
        repeat_rho = spearmanr(repeat["mean"].loc[repeat_ok, 0], repeat["mean"].loc[repeat_ok, 1]).statistic
        r.to_csv(output / f"{side}_adjusted_pulse_responses.csv", index=False)
        r["phase"] = np.where(r.trial <= 5, "first", np.where(r.trial >= 26, "last", "middle"))
        counts = r[r.phase != "middle"].groupby(["track_id", "phase"]).size().unstack(fill_value=0)
        matched = counts.index[(counts >= 3).all(axis=1)]
        matched_r = r[r.track_id.isin(matched)]
        trial = matched_r.groupby("trial")[["delta", "sham_delta", "baseline", "post"]].mean()
        axes[0, col].plot(trial.index, trial.delta, "o-", label="Real pulse")
        axes[0, col].plot(trial.index, trial.sham_delta, "o-", color=".6", label="Midpoint timing control")
        axes[0, col].axhline(0, color="k", lw=.5)
        axes[0, col].set_title(f"{side}: same {len(matched)} ants across first/last trials")
        axes[0, col].set_xlabel("Pulse number (ten minutes apart)")
        axes[0, col].set_ylabel("Mean post minus pre speed (px/s)")
        axes[0, col].legend(fontsize=8)
        means = matched_r[matched_r.phase != "middle"].groupby(["track_id", "phase"]).delta.mean().unstack()
        axes[1, col].scatter(means["first"], means["last"], s=25)
        low, high = np.nanmin(means.to_numpy()), np.nanmax(means.to_numpy())
        axes[1, col].plot([low, high], [low, high], "--", color=".6")
        axes[1, col].set_xlabel("Mean response to first five pulses (px/s)")
        axes[1, col].set_ylabel("Mean response to last five pulses (px/s)")
        axes[1, col].set_title(f"Adjusted decline per ten trials: {fit.params.trial_z:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]")
        results.setdefault("adaptation", {})[side] = {
            "matched_ants": int(len(matched)), "first5_delta": float(means['first'].mean()),
            "last5_delta": float(means['last'].mean()), "trial_slope_per10": float(fit.params.trial_z),
            "trial_clustered_95ci": ci, "real_and_control_adequate_ant_trials": len(r),
            "adjusted_response_repeatability_rho": float(repeat_rho),
            "adjusted_response_repeatability_n": int(repeat_ok.sum()),
            "interpretation": "Habituation-like decline; elapsed time and trial order remain confounded."
        }
    fig.suptitle("Response wanes with repeated stimulation, including in the same ants")
    fig.tight_layout(); save(fig, output, "06_response_adaptation")

    clock = (t + start) / 3600
    # These descriptive windows bracket the visually detected morning step.
    habitual_window = (clock >= 25.75) & (clock < 28.75)  # 01:45–04:45
    pre_window = (clock >= 29.15) & (clock < 29.48)       # 05:09–05:29
    post_window = (clock >= 29.75) & (clock < 30.08)     # 05:45–06:05
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    morning_rows = []
    for col, side in enumerate(["left", "right"]):
        idx = np.flatnonzero(chosen.side == side)
        s = speed[idx]
        habitual = nanmean(s[:, habitual_window], axis=1)
        before = nanmean(s[:, pre_window], axis=1)
        after = nanmean(s[:, post_window], axis=1)
        valid = (np.isfinite(s[:, pre_window]).mean(axis=1) >= .7) & (np.isfinite(s[:, post_window]).mean(axis=1) >= .7) & np.isfinite(habitual)
        rise = after[valid] - before[valid]
        baseline = habitual[valid]
        cutoff = np.median(baseline)
        low_group = baseline <= cutoff
        positive = np.maximum(rise, 0)
        low_share = positive[low_group].sum() / positive.sum()
        rho = spearmanr(baseline, rise).statistic
        mean = nanmean(s, axis=0)
        morning = (clock >= 28.5) & (clock <= 32.5)
        axes[0,col].plot(clock[morning] % 24, mean[morning], lw=.5)
        axes[0,col].set_xlim(4.5, 8.5)
        axes[0,col].set_xticks([4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5],
                               ["04:30", "05:00", "05:30", "06:00", "06:30", "07:00", "07:30", "08:00", "08:30"], rotation=30)
        axes[0,col].axvspan(5.15,5.48,color="C0",alpha=.15,label="Before")
        axes[0,col].axvspan(5.75,6.08,color="C1",alpha=.15,label="After")
        axes[0,col].set_xlabel("Clock time");axes[0,col].set_ylabel("Colony mean speed (px/s)")
        axes[0,col].set_title(f"{side}: matched-ant speed rises {after[valid].mean()/before[valid].mean():.2f}×")
        axes[0,col].legend(fontsize=8)
        axes[1,col].scatter(baseline,rise,c=np.where(low_group,0,1),cmap="coolwarm",s=30)
        for label, color in [("Earlier less-active half", plt.get_cmap("coolwarm")(0.0)),
                             ("Earlier more-active half", plt.get_cmap("coolwarm")(1.0))]:
            axes[1,col].scatter([], [], color=color, label=label)
        axes[1,col].legend(fontsize=7)
        for ant,x,y in zip(chosen.iloc[idx[valid]].track_id,baseline,rise):
            axes[1,col].annotate(str(ant),(x,y),fontsize=6)
        axes[1,col].axhline(0,color=".5",lw=.6)
        axes[1,col].set_xlabel("Earlier habitual speed, 01:45–04:45 (px/s)")
        axes[1,col].set_ylabel("Morning increase in speed (px/s)")
        axes[1,col].set_title(f"Less-active half supplies {100*low_share:.0f}% of positive increase; rho={rho:.2f}")
        p = positions[positions.side == side]
        independent = []
        selected_names = set(chosen.iloc[idx[valid]].track_name)
        for name, ant in p.groupby("track_name"):
            if name not in selected_names:
                continue
            ant = ant.sort_values("frame")
            steps = np.hypot(ant.x_px.diff(), ant.y_px.diff())
            usable = (ant.frame.diff() == 24) & (steps < 312.5)
            hours = ant.frame / 24 / 3600 + start / 3600
            pre = steps[usable & (hours >= 29.15) & (hours < 29.48)].mean()
            post = steps[usable & (hours >= 29.75) & (hours < 30.08)].mean()
            independent.append([pre,post])
        independent=np.array(independent)
        for ant,h,b,a in zip(chosen.iloc[idx[valid]].itertuples(),baseline,before[valid],after[valid]):
            morning_rows.append({"side":side,"track_id":ant.track_id,"track_name":ant.track_name,"habitual_speed":h,"pre_speed":b,"post_speed":a,"delta":a-b,"lower_activity_half":h<=cutoff})
        results.setdefault("morning_rise",{})[side]={
            "matched_ants":int(valid.sum()),"pre_speed":float(before[valid].mean()),"post_speed":float(after[valid].mean()),
            "fold_change":float(after[valid].mean()/before[valid].mean()),"fraction_ants_increasing":float((rise>0).mean()),
            "prior_activity_vs_increase_rho":float(rho),"less_active_half_increment_share":float(low_share),
            "more_active_half_pre_movement_share":float(before[valid][~low_group].sum()/before[valid].sum()),
            "more_active_half_post_movement_share":float(after[valid][~low_group].sum()/after[valid].sum()),
            "more_active_half_fold_change":float(after[valid][~low_group].mean()/before[valid][~low_group].mean()),
            "less_active_half_fold_change":float(after[valid][low_group].mean()/before[valid][low_group].mean()),
            "mean_increase_ant_bootstrap_95ci":bootstrap_mean_ci(rise,rng),
            "independent_1hz_displacement_pre":float(nanmean(independent[:,0])),
            "independent_1hz_displacement_post":float(nanmean(independent[:,1])),
            "pre_speed_coverage":float(coverage[idx[valid]][:,pre_window].mean()),"post_speed_coverage":float(coverage[idx[valid]][:,post_window].mean()),
            "interpretation":"Common external trigger plausible; light/feeding logs needed. Low movement is not evidence of no work."
        }
    fig.suptitle("The morning increase is supplied disproportionately by ants already moving more")
    fig.tight_layout();save(fig,output,"07_morning_allocation")
    pd.DataFrame(morning_rows).to_csv(output / "morning_allocation.csv",index=False)

    # Exclude the common morning event; ask if synchrony still survives.
    checks=[]
    for period, low, high in [("quiet_overnight",25.75,28.75),("late_morning",30.25,32.75)]:
        mask=(clock>=low)&(clock<high)
        for side in ["left","right"]:
            idx=np.flatnonzero(chosen.side==side);s=speed[idx][:,mask]
            valid=np.isfinite(s);keep=valid.mean(axis=1)>=.7;s=s[keep];valid=valid[keep]
            for smooth_minutes in [15,30,60]:
                width=smooth_minutes*12
                trend=uniform_filter1d(np.where(valid,s,0),size=width,axis=1,mode='nearest')/np.maximum(uniform_filter1d(valid.astype(float),size=width,axis=1,mode='nearest'),.01)
                residual=np.where(valid,s-trend,np.nan);residual/=np.nanstd(residual,axis=1,keepdims=True)
                residual=np.where(np.isfinite(residual),residual,0)
                observed=float(np.var(residual.mean(axis=0)));null=[]
                for _ in range(300):
                    shuffled=np.empty_like(residual)
                    for low_bin in range(0,residual.shape[1],720):
                        size=min(720,residual.shape[1]-low_bin)
                        cols=(np.arange(size)[None,:]+rng.integers(size,size=(len(residual),1)))%size
                        shuffled[:,low_bin:low_bin+size]=np.take_along_axis(residual[:,low_bin:low_bin+size],cols,axis=1)
                    null.append(np.var(shuffled.mean(axis=0)))
                checks.append({"period":period,"side":side,"trend_minutes":smooth_minutes,"n_ants":len(residual),"variance_ratio":observed/np.mean(null),"permutation_p":(1+np.sum(np.array(null)>=observed))/301})
    check_table = pd.DataFrame(checks)
    check_table.to_csv(output / "synchrony_sensitivity.csv",index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for col, side in enumerate(["left", "right"]):
        for period, label in [("quiet_overnight", "01:45–04:45"), ("late_morning", "06:15–08:45")]:
            points = check_table[(check_table.side == side) & (check_table.period == period)]
            axes[col].plot(points.trend_minutes, points.variance_ratio, "o-", label=label)
        axes[col].axhline(1, color=".5", linestyle="--")
        axes[col].set_xlabel("Window used to remove slow trends (minutes)")
        axes[col].set_xticks([15,30,60])
        axes[col].set_title(f"{side}: after excluding the common morning increase")
        axes[col].legend(fontsize=8)
    axes[0].set_ylabel("Collective variance / independently shifted controls")
    fig.suptitle("Apparent synchrony is modest and depends on timescale and recording period")
    fig.tight_layout(); save(fig, output, "05_residual_synchrony")
    results["synchrony_sensitivity"]=checks
    (output / "follow_up_results.json").write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)


def analyze(dataset, output):
    output.mkdir(parents=True, exist_ok=True)
    table, n_frames = metadata(dataset)
    fps = float(table.fps.median())
    seconds = n_frames / fps
    chosen = table[table.recording_coverage >= .4].sort_values(["side", "track_id"]).reset_index(drop=True)
    match = re.search(r"_all_(\d{6})_", chosen.track_name.iloc[0])
    clock = match[1]
    start = int(clock[:2]) * 3600 + int(clock[2:4]) * 60 + int(clock[4:])
    pulses = parse_pulses(dataset)
    table.to_csv(output / "track_audit.csv", index=False)
    pulses.to_csv(output / "pulses.csv", index=False)
    audit = {"hours": seconds/3600, "total_tracks": len(table), "selected": chosen.groupby("side").size().to_dict(), "n_pulses": len(pulses), "start_clock": clock, "calibration": "Unverified; speed converted back to panorama px/s", "colony_annotations": "Inherited vectors all zero inside; not used", "days_comparison_possible": seconds >= 2*86400}
    print(json.dumps(audit, indent=2), flush=True)
    cache = output / "sampled_tracks.npz"
    position_cache = output / "positions_1hz.parquet"
    fingerprint = {"version": 1, "n_frames": n_frames, "tracks": [{"name": row.track_name, "speed_mtime": Path(row.speed_path).stat().st_mtime_ns, "track_mtime": (dataset / "stitched/per_track" / row.track_name).stat().st_mtime_ns} for row in chosen.itertuples()]}
    settings = output / "cache_metadata.json"
    if cache.exists() and position_cache.exists() and settings.exists() and json.loads(settings.read_text()) == fingerprint:
        with np.load(cache) as arrays:
            speed, coverage = arrays["speed"], arrays["coverage"]
        positions = pd.read_parquet(position_cache)
    else:
        results = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks = [pool.submit(read_ant, row, dataset, n_frames, fps) for row in chosen.itertuples()]
            for index, future in enumerate(as_completed(tasks), 1):
                name, s, c, p = future.result()
                results[name] = (s, c, p)
                if index % 10 == 0 or index == len(tasks):
                    print(f"Read {index}/{len(tasks)} ants", flush=True)
        speed = np.stack([results[name][0] for name in chosen.track_name])
        coverage = np.stack([results[name][1] for name in chosen.track_name])
        positions = pd.concat([results[name][2] for name in chosen.track_name], ignore_index=True)
        np.savez_compressed(cache, speed=speed, coverage=coverage)
        positions.to_parquet(position_cache, index=False)
        settings.write_text(json.dumps(fingerprint, indent=2))
    rng = np.random.default_rng(20260908)
    t = (np.arange(speed.shape[1]) + .5) * 5
    duration = seconds / 3600
    sides = ["left", "right"]
    results = {"audit": audit}
    chosen["mean_speed_px_s"] = np.nanmean(speed, axis=1)
    epoch_edges = np.array([0, 2, 4, 6, 8, 10, 12, duration]) * 3600
    epoch_edges = np.unique(epoch_edges[epoch_edges <= seconds])
    epoch_rows = []
    for i, ant in chosen.iterrows():
        for epoch, (low, high) in enumerate(zip(epoch_edges[:-1], epoch_edges[1:])):
            mask = (t >= low) & (t < high)
            epoch_rows.append({"side": ant.side, "track_id": ant.track_id, "track_name": ant.track_name, "epoch": epoch, "start_h": low/3600, "end_h": high/3600, "mean_speed_px_s": nanmean(speed[i, mask]), "valid_fraction": np.isfinite(speed[i, mask]).mean(), "mean_coverage": coverage[i, mask].mean()})
    epochs = pd.DataFrame(epoch_rows)
    epochs.to_csv(output / "ant_epochs.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for col, side in enumerate(sides):
        idx = np.flatnonzero(chosen.side == side)
        s = speed[idx]
        order = np.argsort(np.nanmean(s, axis=1))
        image = axes[0,col].imshow(s[order], aspect="auto", extent=[0,duration,len(idx),0], vmin=0, vmax=np.nanpercentile(s,95), cmap="magma")
        axes[0,col].set_title(f"{side}: {len(idx)} ants; sorted by overall movement")
        axes[0,col].set_ylabel("Ant rank")
        fig.colorbar(image, ax=axes[0,col], label="Mean speed (panorama px/s)")
        mean = np.nanmean(s, axis=0)
        axes[1,col].plot(t/3600, mean, lw=.7)
        axes[1,col].set_ylabel("Colony mean speed (px/s)")
        axes[2,col].plot(t/3600, np.nanmean(coverage[idx],axis=0), lw=.6)
        axes[2,col].set_ylim(0,1)
        axes[2,col].set_ylabel("Speed coverage fraction")
        for ax in axes[:,col]:
            if not pulses.empty:
                for frame in pulses.frame_start:
                    ax.axvline(frame/fps/3600, color="cyan", alpha=.3, lw=.4)
            label_clock(ax,start,duration)
    fig.suptitle("One overnight record: separate individual activity, colony activity, and visibility")
    fig.tight_layout()
    save(fig,output,"01_activity_and_coverage")

    # Spatial profiles and independent early/late location fidelity.
    fig, axes = plt.subplots(2,2,figsize=(12,10))
    spatial_rows = []
    for col, side in enumerate(sides):
        p = positions[positions.side == side].copy()
        xlim = np.quantile(p.x_px,[.001,.999]); ylim=np.quantile(p.y_px,[.001,.999])
        bins=[np.linspace(*xlim,35),np.linspace(*ylim,45)]
        h,xe,ye=np.histogram2d(p.x_px,p.y_px,bins=bins)
        axes[0,col].imshow(np.log1p(h.T),extent=[*xlim,*ylim[::-1]],aspect="equal",cmap="magma")
        axes[0,col].set_title(f"{side}: observed occupancy (log scale)")
        axes[0,col].set_xlabel("Panorama X (px)");axes[0,col].set_ylabel("Panorama Y (px)")
        early=[];late=[];names=[]
        for name,ant in p.groupby("track_name",sort=True):
            halves=[]
            for keep in [ant.frame < n_frames/2,ant.frame >= n_frames/2]:
                q=ant[keep];hist=np.histogram2d(q.x_px,q.y_px,bins=bins)[0].ravel()+1e-12
                halves.append(hist/hist.sum())
            early.append(halves[0]);late.append(halves[1]);names.append(name)
        distances=np.array([[jensenshannon(a,b,base=2) for b in late] for a in early])
        within=np.diag(distances);others=distances[~np.eye(len(names),dtype=bool)]
        null=np.array([distances[np.arange(len(names)),rng.permutation(len(names))].mean() for _ in range(2000)])
        axes[1,col].hist(others,bins=35,density=True,color=".8",label="Different ants (early vs late)")
        axes[1,col].hist(within,bins=15,density=True,alpha=.7,label="Same ant (early vs late)")
        axes[1,col].set_xlabel("Spatial-profile distance (0 = identical, 1 = disjoint)")
        axes[1,col].set_ylabel("Density");axes[1,col].legend(fontsize=8)
        results.setdefault("spatial_fidelity",{})[side]={"same_ant_distance":float(within.mean()),"different_ant_distance":float(others.mean()),"identity_permutation_p":float((1+(null<=within.mean()).sum())/2001)}
        spatial_rows.extend({"side":side,"track_name":name,"early_late_distance":float(d)} for name,d in zip(names,within))
    fig.suptitle("Spatial habits: are ants in the same places early and late?")
    fig.tight_layout();save(fig,output,"02_spatial_fidelity")
    pd.DataFrame(spatial_rows).to_csv(output / "spatial_fidelity.csv", index=False)

    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for col,side in enumerate(sides):
        idx=np.flatnonzero(chosen.side==side);s=speed[idx]
        # Matched well-observed ants and equal-duration halves avoid survivor artifacts.
        a=np.nanmean(s[:,t<seconds/2],axis=1);b=np.nanmean(s[:,t>=seconds/2],axis=1)
        valid=(np.isfinite(s[:,t<seconds/2]).mean(axis=1)>=.7)&(np.isfinite(s[:,t>=seconds/2]).mean(axis=1)>=.7)
        rho=spearmanr(a[valid],b[valid]).statistic
        k=int(np.ceil(valid.sum()*.2));top_a=np.argsort(a[valid])[-k:];top_b=np.argsort(b[valid])[-k:]
        overlap=len(set(top_a)&set(top_b))/k
        for ant,x,y in zip(chosen.iloc[idx[valid]].track_id,a[valid],b[valid]):
            axes[0,col].scatter(x,y,s=20);axes[0,col].annotate(str(ant),(x,y),fontsize=6)
        axes[0,col].set_xlabel("Early-half mean speed (px/s)");axes[0,col].set_ylabel("Late-half mean speed (px/s)")
        axes[0,col].set_title(f"{side}: movement rank persistence; rho={rho:.2f}")
        parts=[]
        for epoch in range(len(epoch_edges)-1):
            mask=(t>=epoch_edges[epoch])&(t<epoch_edges[epoch+1]);v=nanmean(s[:,mask],axis=1)
            parts.append(v)
        matrix=np.array(parts).T
        rank=pd.DataFrame(matrix).rank(pct=True).to_numpy()
        order=np.argsort(matrix[:,0])
        axes[1,col].imshow(rank[order],aspect="auto",vmin=0,vmax=1,cmap="viridis")
        axes[1,col].set_xticks(range(len(parts)),[f"{epoch_edges[j]/3600:g}–{epoch_edges[j+1]/3600:g}" for j in range(len(parts))],rotation=30)
        axes[1,col].set_xlabel("Hours since recording start");axes[1,col].set_ylabel("Ants ordered by first epoch")
        results.setdefault("movement_persistence",{})[side]={"n_matched_ants":int(valid.sum()),"spearman_rho":float(rho),"top20_overlap":float(overlap),"top20_share_early":float(np.sort(a[valid])[-k:].sum()/a[valid].sum()),"top20_share_late":float(np.sort(b[valid])[-k:].sum()/b[valid].sum()),"mean_early":float(a[valid].mean()),"mean_late":float(b[valid].mean())}
    fig.suptitle("Who supplies movement, and does the ranking change? Color = within-epoch percentile")
    fig.tight_layout();save(fig,output,"03_movement_allocation")

    # Responses exclude the five-second pulse to reduce camera/motor artifacts.
    if not pulses.empty:
        pulse_rows=[];curves=[]
        lags=np.arange(-60,181,5)
        for i,ant in chosen.iterrows():
            for pulse in pulses.itertuples():
                onset=pulse.frame_start/fps
                bins_idx=np.floor((onset+lags)/5).astype(int)
                curve=speed[i,bins_idx]
                cov=coverage[i,bins_idx]
                baseline=nanmean(curve[(lags>=-60)&(lags<-10)])
                post=nanmean(curve[(lags>=10)&(lags<60)])
                adequate=np.isfinite(curve[(lags>=-60)&(lags<-10)]).mean()>=.8 and np.isfinite(curve[(lags>=10)&(lags<60)]).mean()>=.8
                # A point halfway to the next real pulse is a local timing control.
                sham_bins=np.floor((onset+300+lags)/5).astype(int)
                sham=speed[i,sham_bins]
                sham_base=nanmean(sham[(lags>=-60)&(lags<-10)])
                sham_post=nanmean(sham[(lags>=10)&(lags<60)])
                sham_adequate=np.isfinite(sham[(lags>=-60)&(lags<-10)]).mean()>=.8 and np.isfinite(sham[(lags>=10)&(lags<60)]).mean()>=.8
                pulse_rows.append({"side":ant.side,"track_id":ant.track_id,"trial":int(pulse.trial),"duty":pulse.duty,"baseline":float(baseline),"post":float(post),"delta":float(post-baseline),"adequate":adequate,"sham_adequate":sham_adequate,"sham_delta":float(sham_post-sham_base),"pre_coverage":float(cov[(lags>=-60)&(lags<-10)].mean()),"post_coverage":float(cov[(lags>=10)&(lags<60)].mean())})
                if adequate:
                    curves.append((ant.side,ant.track_id,int(pulse.trial),curve-baseline))
        responses=pd.DataFrame(pulse_rows);responses.to_csv(output / "pulse_responses.csv",index=False)
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        for col,side in enumerate(sides):
            r=responses[(responses.side==side)&responses.adequate&responses.sham_adequate].copy()
            c=np.stack([q[3] for q in curves if q[0]==side])
            axes[0,col].plot(lags,np.nanmean(c,axis=0),color="C0")
            axes[0,col].axvspan(0,5,color=".7",alpha=.5)
            axes[0,col].axhline(0,color="black",lw=.5)
            axes[0,col].set_xlabel("Seconds relative to logged pulse frame")
            axes[0,col].set_ylabel("Speed change from baseline (px/s)")
            trial=r.groupby('trial')[['delta','sham_delta','baseline','post']].mean()
            corrected=trial.delta-trial.sham_delta
            axes[0,col].set_title(f"{side}: response minus midpoint control = {corrected.mean():.2f} px/s")
            r["half"]=np.where(r.trial<=15,"first","second")
            means=r.groupby(['track_id','half']).delta.agg(['mean','count']).unstack('half')
            eligible=(means['count']>=8).all(axis=1)
            a=means['mean'].loc[eligible,'first'];b=means['mean'].loc[eligible,'second']
            rho=spearmanr(a,b).statistic
            axes[1,col].scatter(a,b,s=23)
            for ant,x,y in zip(a.index,a,b):axes[1,col].annotate(str(ant),(x,y),fontsize=6)
            axes[1,col].set_xlabel("Mean response, trials 1–15 (px/s)");axes[1,col].set_ylabel("Mean response, trials 16–30 (px/s)")
            axes[1,col].set_title(f"Individual response persistence: rho={rho:.2f}, n={len(a)}")
            results.setdefault("stimulus",{})[side]={"adequate_ant_trials":len(r),"n_ants":int(r.track_id.nunique()),"mean_delta_px_s":float(trial.delta.mean()),"mean_sham_delta_px_s":float(trial.sham_delta.mean()),"corrected_delta_px_s":float(corrected.mean()),"trial_bootstrap_95ci":bootstrap_mean_ci(corrected,rng),"baseline_px_s":float(trial.baseline.mean()),"post_px_s":float(trial.post.mean()),"response_reliability_rho":float(rho),"response_reliability_n":int(len(a)),"delta_coverage":float((r.post_coverage-r.pre_coverage).mean()),"trial_response_vs_order_rho":float(spearmanr(trial.index,trial.delta).statistic),"duty_vs_response_rho":float(spearmanr(pulses.duty,trial.delta).statistic)}
        fig.suptitle("Stimulation response and repeatability: 10–60 s after vs 60–10 s before")
        fig.tight_layout();save(fig,output,"04_stimulus_responses")

    # Nonstimulus synchrony, with independent shifts inside one-hour blocks.
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for col,side in enumerate(sides):
        idx=np.flatnonzero(chosen.side==side)
        after=max(pulses.frame_end)/fps+600 if not pulses.empty else 0
        mask=t>=after;s=speed[idx][:,mask]
        valid=np.isfinite(s)
        filled=np.where(valid,s,0)
        trend=uniform_filter1d(filled,size=720,axis=1,mode='nearest')/np.maximum(uniform_filter1d(valid.astype(float),size=720,axis=1,mode='nearest'),.01)
        resid=np.where(valid,s-trend,np.nan)
        resid/=np.nanstd(resid,axis=1,keepdims=True)
        resid=np.where(np.isfinite(resid),resid,0)
        # Fixed denominator and zeros for missing residuals; shift masks with data.
        observed=float(np.var(resid.mean(axis=0)))
        null=[]
        for rep in range(400):
            shuffled=np.empty_like(resid)
            for low in range(0,resid.shape[1],720):
                width=min(720,resid.shape[1]-low)
                shifts=rng.integers(width,size=(len(idx),1))
                cols=(np.arange(width)[None,:]+shifts)%width
                shuffled[:,low:low+width]=np.take_along_axis(resid[:,low:low+width],cols,axis=1)
            null.append(float(np.var(shuffled.mean(axis=0))))
        axes[col].hist(null,bins=25,color='.7');axes[col].axvline(observed,color='C1',lw=2)
        axes[col].set_title(f"{side}: residual collective variance = {observed/np.mean(null):.2f} × null")
        axes[col].set_xlabel("Variance of mean standardized movement residual")
        axes[col].set_ylabel("Shifted controls")
        results.setdefault('nonstimulus_synchrony',{})[side]={"observed_variance":observed,"null_mean":float(np.mean(null)),"variance_ratio":observed/np.mean(null),"null_95ci":np.quantile(null,[.025,.975]).tolist(),"permutation_p":float((1+np.sum(np.array(null)>=observed))/401),"hours_analyzed":float(mask.sum()*5/3600)}
    fig.suptitle("Coordination after stimulation: remove one-hour trend, shuffle each ant within hours")
    fig.tight_layout();save(fig,output,"05_residual_synchrony")
    chosen.to_csv(output / "selected_ants.csv",index=False)
    (output / "results.json").write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)
    if not pulses.empty:
        follow_up_checks(output,chosen,speed,coverage,positions,pulses,t,start,rng)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    analyze(args.dataset,args.output or args.dataset / "analysis_outputs/collective_task_probe_20260908")
