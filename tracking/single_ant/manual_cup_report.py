"""Review a manual-cup pilot, including natural and deliberately hidden tag gaps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from tracking.single_ant.manual_cup_tracking import Settings, track_cup, write_json

COLORS = dict(aruco_sleap="#168462", sleap_bridge="#e58a22", aruco_only="#777777")


def intervals(mask):
    changes = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def steps(track):
    xy = track[["TrackX", "TrackY"]].to_numpy()
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return d[np.isfinite(d)]


def trajectory(ax, track, cup):
    center = np.array(cup["center_mm"])
    xy = track[["TrackX", "TrackY"]].to_numpy() - center
    ok = np.isfinite(xy).all(axis=1)
    if len(xy) > 1:
        good = ok[:-1] & ok[1:]
        segs = np.stack([xy[:-1][good], xy[1:][good]], axis=1)
        colors = [COLORS.get(s, "#999999") for s in track.PositionSource.iloc[1:][good]]
        ax.add_collection(LineCollection(segs, colors=colors, linewidths=.65, alpha=.7))
    for source, color in COLORS.items():
        use = track.PositionSource.eq(source).to_numpy() & ok
        ax.scatter(*xy[use].T, s=1.5, color=color, label=source.replace("_", " "), rasterized=True)
    ax.add_patch(Circle((0, 0), cup["radius_mm"], fill=False, color="#333333", lw=1))
    r = cup["radius_mm"] * 1.06
    ax.set(xlim=(-r, r), ylim=(r, -r), aspect="equal", xlabel="x from cup center (mm)", ylabel="y (mm)")


def xy_time(ax, track, cup, fps, bounds=None):
    t = (track.Frame - track.Frame.min()) / fps
    for coord, center, color in [("X", cup["center_mm"][0], "#2b6cb0"),
                                 ("Y", cup["center_mm"][1], "#b73f65")]:
        ax.plot(t, track["Track" + coord] - center, color=color, lw=.6, label=coord.lower())
        ax.scatter(t, track["Aruco" + coord] - center, color=color, s=2, alpha=.4)
    if bounds is not None:
        ax.set_xlim(bounds)
    else:
        ax.set_xlim(0, float(t.iloc[-1]))
    stopped = track.PositionSource.eq("stopped_multi_ant")
    if stopped.any():
        cutoff = float(t.loc[stopped].iloc[0])
        ax.axvline(cutoff, color="#9e3029", ls="--", lw=1)
        ax.axvspan(cutoff, float(t.iloc[-1]), color="#777777", alpha=.12)
        ax.text(cutoff, .98, f" STOP {cutoff:.1f}s", transform=ax.get_xaxis_transform(),
                color="#9e3029", fontsize=8, va="top")
    ax.set(xlabel="seconds from pilot start", ylabel="position from center (mm)")
    ax.legend(fontsize=7, ncol=2, loc="upper right")


def bridge_zoom(ax, track, cup, fps):
    gaps = intervals(track.PositionSource.eq("sleap_bridge"))
    if not gaps:
        ax.text(.5, .5, "No natural SLEAP bridge", ha="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    # Prefer a gap bracketed by tag matches, then a long gap.
    bracketed = [(lo, hi) for lo, hi in gaps if lo > 0 and hi < len(track)
                 and track.PositionSource.iloc[lo - 1] == "aruco_sleap"
                 and track.PositionSource.iloc[hi] == "aruco_sleap"]
    lo, hi = max(bracketed or gaps, key=lambda pair: pair[1] - pair[0])
    middle = (lo + hi) // 2 if hi - lo > 240 else lo
    a, b = max(0, middle - 48), min(len(track), max(middle + 192, min(hi + 48, middle + 240)))
    window = track.iloc[a:b]
    t = (window.Frame - track.Frame.min()) / fps
    for coord, center, color in [("X", cup["center_mm"][0], "#2b6cb0"),
                                 ("Y", cup["center_mm"][1], "#b73f65")]:
        ax.plot(t, window["Track" + coord] - center, color=color, lw=1, label=coord.lower())
        ax.scatter(t, window["Aruco" + coord] - center, edgecolors="black", linewidths=.4,
                   color=color, s=18, zorder=3)
    ax.axvspan(lo / fps, hi / fps, color=COLORS["sleap_bridge"], alpha=.18)
    ax.set(xlim=(t.iloc[0], t.iloc[-1]), xlabel="seconds from pilot start", ylabel="position (mm)",
           title=f"Natural tag gap: {(hi - lo) / fps:.2f} s (dots = actual tags)")


def hidden_tag_validation(run, cups, a, poses, counts, frames, settings):
    """Fixed cup identity, hide contiguous tag windows, compare to withheld reads.

    These are consistency checks against the detector, not independent manual
    ground truth. Counts are retained only for multiplicity vetoes; no hidden
    tag XY or tag presence is consulted by the missing-tag matching path.
    """
    start, stop, fps = run["start_frame"], run["stop_frame"], run["fps"]
    records, details = [], []
    for cup in cups:
        baseline = frames.loc[frames.CupID.eq(cup["cup_id"])].reset_index(drop=True)
        if baseline.empty:
            continue
        n = len(baseline)
        mask = np.zeros(n, bool)
        windows = []
        for seconds in [1, 5, 20]:
            length = int(round(seconds * fps))
            chosen = 0
            for lo in np.linspace(24, max(24, n - length - 24), 60, dtype=int):
                hi = lo + length
                if hi >= n or mask[max(0, lo - 24):min(n, hi + 24)].any():
                    continue
                if baseline.PositionSource.iloc[lo:hi].eq("aruco_sleap").mean() < .8:
                    continue
                mask[lo:hi] = True
                windows.append((seconds, lo, hi))
                chosen += 1
                if chosen == 3:
                    break
        hidden = track_cup(cup, int(baseline.ArucoID.iloc[0]), a, poses, counts,
                           start, stop, settings, mask_tags=mask)
        for seconds, lo, hi in windows:
            base, test = baseline.iloc[lo:hi], hidden.iloc[lo:hi]
            has_tag = base.ArucoX.notna().to_numpy()
            valid = has_tag & test.TrackX.notna().to_numpy()
            error = np.linalg.norm(test[["TrackX", "TrackY"]].to_numpy()
                                   - base[["ArucoX", "ArucoY"]].to_numpy(), axis=1)
            # Compare the selected posture anchor too: a changed camera view is
            # not automatically an identity error.
            anchor_error = np.linalg.norm(test[["SleapAnchorX", "SleapAnchorY"]].to_numpy()
                                          - base[["SleapAnchorX", "SleapAnchorY"]].to_numpy(), axis=1)
            values = error[valid]
            records.append(dict(Cup=cup["name"], CupID=cup["cup_id"], HiddenSeconds=seconds,
                                StartFrame=int(base.Frame.iloc[0]), WithheldTagFrames=int(has_tag.sum()),
                                RecoveredFrames=int(valid.sum()), RecoveryFraction=float(valid.sum() / max(1, has_tag.sum())),
                                MedianTagErrorMM=float(np.median(values)) if len(values) else None,
                                P95TagErrorMM=float(np.quantile(values, .95)) if len(values) else None,
                                MaxTagErrorMM=float(values.max()) if len(values) else None))
            details.append(pd.DataFrame(dict(CupID=cup["cup_id"], Frame=base.Frame.to_numpy(),
                                             HiddenSeconds=seconds, TagErrorMM=error,
                                             BaselineAnchorErrorMM=anchor_error,
                                             WithheldTag=has_tag, Recovered=valid)))
    return pd.DataFrame(records), pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def report(output, panorama=None):
    output = Path(output)
    run = json.loads((output / "run.json").read_text())
    complete = json.loads((output / "PILOT_COMPLETE.json").read_text())
    q = pd.read_csv(output / "cup_qualification.csv")
    cups = run["cups"]
    plotdir = output / "plots"
    plotdir.mkdir(exist_ok=True)
    fps, n = run["fps"], run["stop_frame"] - run["start_frame"]
    summary = dict(complete, pilot_seconds=n / fps, qualification_scope=run["qualification_scope"])
    frames = pd.read_parquet(output / "pilot_frames_mm.parquet") if complete["tracked"] else pd.DataFrame()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    with PdfPages(output / "manual_cup_pilot_report.pdf") as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(16, 10), layout="constrained", gridspec_kw={"width_ratios": [1, 1.5]})
        ax = axes[0]
        colors = np.where(q.Eligible, "#168462", "#b85b55")
        if "FlaggedMultiAnt" in q:
            colors = np.where(q.Eligible & q.FlaggedMultiAnt, "#d69331", colors)
        ax.barh(q.CupID, q.SingleIDFraction * 100, color=colors)
        ax.axvline(60, color="black", ls="--", lw=.8)
        ax.set(yticks=q.CupID, yticklabels=q.Cup, xlabel="all pilot frames with one resident-ID instance, ignoring unsupported extra IDs (%)",
               ylim=(len(q) + 1, 0), xlim=(0, 105), title="Strict >60% ID rule + second-ant evidence veto")
        for row in q.itertuples(index=False):
            extra = f"ID {row.ArucoID}" if row.ArucoID >= 0 else "no tag"
            if row.MultipleArucoFrames:
                extra += f" | multi-tag {row.MultipleArucoFrames}f"
            if row.MultipleStrongSleapFrames:
                extra += f" | multi-pose {row.MultipleStrongSleapFrames}f"
            if getattr(row, "FlaggedMultiAnt", False):
                extra += f" | STOP {row.StopFrame / fps:.1f}s"
            ax.text(2, row.CupID, extra, color="black", va="center", fontsize=6)
        ax = axes[1]
        if panorama is not None:
            meta = json.loads((output / "panorama_metadata.json").read_text())
            b = meta["raw_tracking_bounds_px"]
            scale = meta["mm_per_pixel"]
            ax.imshow(plt.imread(panorama), cmap="gray", extent=[b["x_min"] * scale, b["x_max"] * scale,
                      b["y_max"] * scale, b["y_min"] * scale], alpha=.65)
        selected = {s["cup_id"] for s in complete["tracked"]}
        for cup in cups:
            row = q.loc[q.CupID.eq(cup["cup_id"])].iloc[0]
            color = "#d69331" if getattr(row, "FlaggedMultiAnt", False) and row.Eligible else ("#168462" if row.Eligible else "#b85b55")
            ax.add_patch(Circle(cup["center_mm"], cup["radius_mm"], fill=False, color=color, lw=1.2))
            x, y = cup["center_mm"]
            text = f"{cup['name']}\nID {row.ArucoID}" + ("\nPILOT" if cup["cup_id"] in selected else "")
            ax.text(x, y, text, ha="center", va="center", fontsize=6,
                    bbox=dict(fc="white", ec="none", alpha=.7, pad=1))
        if panorama is None:
            ax.autoscale_view()
            ax.invert_yaxis()
        ax.set(aspect="equal", xlabel="raw calibrated x (mm)", ylabel="raw calibrated y (mm)",
               title=f"{len(cups)} manual cups; green = eligible; orange = stop event\n{len(selected)} tracked pilots marked")
        fig.suptitle(f"Oct 26 block 03 — {n / fps / 60:g}-minute pilot, {int(q.Eligible.sum())}/{len(q)} cups pass\n"
                     "Qualification covers this window only. Red cups are excluded; orange cups stop at their event.", fontsize=14)
        fig.savefig(plotdir / "cup_qualification.png", dpi=160)
        pdf.savefig(fig)
        plt.close(fig)
        if not len(frames):
            write_json(output / "review_summary.json", summary)
            return
        selected_cups = [c for c in cups if c["cup_id"] in set(frames.CupID)]
        fig, axes = plt.subplots(len(selected_cups), 3, figsize=(17, 4.1 * len(selected_cups)), layout="constrained", squeeze=False)
        performance = []
        for row, cup in enumerate(selected_cups):
            t = frames.loc[frames.CupID.eq(cup["cup_id"])].reset_index(drop=True)
            trajectory(axes[row, 0], t, cup)
            pose_fraction = t.PoseKey.ge(0).mean()
            active = ~t.PositionSource.eq("stopped_multi_ant")
            active_n = int(active.sum())
            active_pose = t.loc[active, "PoseKey"].ge(0).mean() if active_n else 0.
            qualifier = " before stop" if not active.all() else ""
            axes[row, 0].set_title(f"{cup['name']} | ID {int(t.ArucoID.iloc[0])} | observed pose {active_pose:.1%}{qualifier}")
            if row == 0:
                axes[row, 0].legend(fontsize=6, loc="lower left")
            xy_time(axes[row, 1], t, cup, fps)
            sources = t.PositionSource.value_counts()
            axes[row, 1].set_title(f"tag+pose {sources.get('aruco_sleap', 0)/max(1,active_n):.1%}; "
                                  f"SLEAP bridge {sources.get('sleap_bridge', 0)/max(1,active_n):.1%}; "
                                  f"missing {t.loc[active, 'TrackX'].isna().mean():.1%}{qualifier}")
            bridge_zoom(axes[row, 2], t, cup, fps)
            delta = steps(t)
            match = t.MatchDistanceMM.dropna().to_numpy()
            bridge_lengths = np.array([hi - lo for lo, hi in intervals(t.PositionSource.eq("sleap_bridge"))]) / fps
            performance.append(dict(Cup=cup["name"], CupID=cup["cup_id"], ArucoID=int(t.ArucoID.iloc[0]),
                TagAndPoseFraction=float(t.PositionSource.eq("aruco_sleap").mean()),
                BridgeFraction=float(t.PositionSource.eq("sleap_bridge").mean()),
                PositionFraction=float(t.TrackX.notna().mean()), PoseFraction=float(pose_fraction),
                ActiveFrames=int(active.sum()), StoppedFrames=int((~active).sum()),
                ActivePositionFraction=float(t.loc[active, "TrackX"].notna().mean()) if active.any() else 0.,
                ActivePoseFraction=float(t.loc[active, "PoseKey"].ge(0).mean()) if active.any() else 0.,
                ActiveFullPoseFraction=float(t.loc[active, "NNodes"].eq(10).mean()) if active.any() else 0.,
                MissingFrames=int(t.TrackX.isna().sum()), OutsideCupFrames=int((np.linalg.norm(t[["TrackX", "TrackY"]]
                     .to_numpy() - cup["center_mm"], axis=1) > cup["radius_mm"] + 1e-8).sum()),
                MedianMatchMM=float(np.median(match)) if len(match) else None,
                P95MatchMM=float(np.quantile(match, .95)) if len(match) else None,
                MaxStepMM=float(delta.max()) if len(delta) else None,
                P99StepMM=float(np.quantile(delta, .99)) if len(delta) else None,
                LongestBridgeSeconds=float(bridge_lengths.max()) if len(bridge_lengths) else 0.))
        fig.suptitle("Pilot trajectories and natural tag gaps — green = tag+pose; orange = observed SLEAP bridge", fontsize=14)
        fig.savefig(plotdir / "selected_cup_tracking.png", dpi=150)
        pdf.savefig(fig)
        plt.close(fig)
        pd.DataFrame(performance).to_csv(output / "tracking_performance.csv", index=False)
        exported = pd.read_parquet(output / "pilot_tracks.parquet", columns=["CupID", "PoseKey", "Frame", "Bodypoint"])
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
        coverage, node_counts = [], []
        for cup in selected_cups:
            t = frames.loc[frames.CupID.eq(cup["cup_id"]) & frames.PositionSource.ne("stopped_multi_ant")]
            denominator = max(1, len(t))
            b = exported.loc[exported.CupID.eq(cup["cup_id"]) & exported.PoseKey.ge(0)]
            coverage.append(b.groupby("Bodypoint").Frame.nunique().reindex(range(10), fill_value=0).to_numpy() / denominator)
            node_counts.append(t.NNodes.value_counts().reindex(range(11), fill_value=0).to_numpy() / denominator)
        coverage, node_counts = np.asarray(coverage), np.asarray(node_counts)
        image = axes[0].imshow(coverage * 100, vmin=0, vmax=100, cmap="YlGn", aspect="auto")
        axes[0].set(xticks=range(10), xlabel="SLEAP bodypoint", yticks=range(len(selected_cups)),
                    yticklabels=[c["name"] for c in selected_cups], title="Observed node coverage before stop (%)")
        fig.colorbar(image, ax=axes[0], label="active frames with observed bodypoint (%)")
        for i in range(len(selected_cups)):
            for j in range(10):
                axes[0].text(j, i, f"{coverage[i,j]*100:.0f}", ha="center", va="center", fontsize=7)
        bottom = np.zeros(len(selected_cups))
        for k in [0, 4, 5, 6, 7, 8, 9, 10]:
            axes[1].bar(range(len(selected_cups)), node_counts[:, k] * 100, bottom=bottom,
                         label=f"{k} nodes" if k else "no pose", color=plt.cm.viridis(k / 10))
            bottom += node_counts[:, k] * 100
        axes[1].set(xticks=range(len(selected_cups)), xticklabels=[c["name"] for c in selected_cups],
                    ylabel="active frames (%)", ylim=(0, 100), title="Completeness of selected whole skeletons")
        axes[1].legend(fontsize=7, loc="lower right")
        fig.suptitle("Posture quality — observed nodes only; no interpolation", fontsize=14)
        fig.savefig(plotdir / "posture_completeness.png", dpi=160)
        pdf.savefig(fig)
        plt.close(fig)
        a = pd.read_parquet(output / "aruco_mm.parquet")
        poses = pd.read_parquet(output / "poses_mm.parquet")
        counts = pd.read_parquet(output / "frame_counts.parquet")
        if (output / "hidden_tag_validation.csv").exists() and (output / "hidden_tag_frames.parquet").exists():
            print("Reusing hidden-tag checks from this immutable pilot directory", flush=True)
            holdouts = pd.read_csv(output / "hidden_tag_validation.csv")
            details = pd.read_parquet(output / "hidden_tag_frames.parquet")
        else:
            print("Testing deliberately hidden 1 s, 5 s, and 20 s tag windows", flush=True)
            holdouts, details = hidden_tag_validation(run, selected_cups, a, poses, counts, frames, Settings(**run["settings"]))
            holdouts.to_csv(output / "hidden_tag_validation.csv", index=False)
            details.to_parquet(output / "hidden_tag_frames.parquet", index=False)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
        for cup in selected_cups:
            t = frames.loc[frames.CupID.eq(cup["cup_id"])].reset_index(drop=True)
            d = steps(t)
            if len(d):
                axes[0, 0].plot(np.sort(d), np.arange(1, len(d) + 1) / len(d), label=cup["name"])
            match = np.sort(t.MatchDistanceMM.dropna().to_numpy())
            if len(match):
                axes[0, 1].plot(match, np.arange(1, len(match) + 1) / len(match), label=cup["name"])
        axes[0, 0].set(xlabel="consecutive-frame displacement (mm)", ylabel="cumulative fraction", xscale="symlog",
                       title="Jumps remain visible for review; zero outside-cup positions required")
        axes[0, 1].axvline(run["settings"]["aruco_sleap_distance_mm"], color="black", ls="--", lw=.8)
        axes[0, 1].set(xlabel="actual tag to selected SLEAP anchor distance (mm)", ylabel="cumulative fraction", title="Nearest-neighbor association")
        axes[0, 0].legend()
        if len(details):
            data, labels = [], []
            for seconds in sorted(details.HiddenSeconds.unique()):
                v = details.loc[details.HiddenSeconds.eq(seconds) & details.Recovered, "TagErrorMM"].dropna()
                data.append(v.to_numpy())
                labels.append(f"{seconds:g}s\nn={len(v)}")
            axes[1, 0].boxplot(data, tick_labels=labels, showfliers=False)
            axes[1, 0].set(ylabel="position vs withheld tag (mm)", title="Artificial gaps: detector consistency, not manual ground truth")
            piv = holdouts.groupby(["Cup", "HiddenSeconds"], sort=False).agg(Recovered=("RecoveredFrames", "sum"), Total=("WithheldTagFrames", "sum"))
            piv["Recovery"] = piv.Recovered / piv.Total
            matrix = piv.Recovery.unstack().reindex([c["name"] for c in selected_cups])
            axes[1, 1].imshow(matrix, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
            axes[1, 1].set(xticks=range(len(matrix.columns)), xticklabels=[f"{s:g}s hidden" for s in matrix.columns],
                           yticks=range(len(matrix)), yticklabels=matrix.index, title="Withheld-tag frames recovered using posture only")
            for i in range(len(matrix)):
                for j in range(len(matrix.columns)):
                    axes[1, 1].text(j, i, f"{matrix.iloc[i, j]:.1%}", ha="center", va="center")
        fig.suptitle("Tracking checks and deliberately hidden-tag validation", fontsize=15)
        fig.savefig(plotdir / "gap_and_jump_validation.png", dpi=160)
        pdf.savefig(fig)
        plt.close(fig)
    summary["performance"] = performance
    summary["hidden_tag_windows"] = len(holdouts)
    summary["hidden_tag_recovery_fraction"] = float(holdouts.RecoveredFrames.sum() / holdouts.WithheldTagFrames.sum()) if len(holdouts) else None
    summary["outside_cup_frames"] = sum(x["OutsideCupFrames"] for x in performance)
    write_json(output / "review_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--panorama", type=Path)
    args = p.parse_args()
    report(args.output, args.panorama)
