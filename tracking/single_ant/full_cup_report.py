"""Colony motion calculations and full-duration diagnostics for manual cups."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from analysis.compute_track_speed_vector import speed_vector
from tracking.single_ant.full_cup_tracking import MM_PER_PX, root
from tracking.single_ant.manual_cup_tracking import write_json

SPEED = dict(max_interp_gap_frames=5, smooth_sigma_frames=2., max_speed_mm_s=20.)
COLORS = ["#ffffff", "#168462", "#e58a22", "#777777"]


def binned(values, width, operation="mean"):
    values = np.asarray(values, dtype=float)
    padded = np.pad(values, (0, (-len(values)) % width), constant_values=np.nan)
    matrix = padded.reshape(-1, width)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return getattr(np, "nan" + operation)(matrix, axis=1)


def gap_lengths(valid, fps):
    mask = ~np.asarray(valid, bool)
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return (np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)) / fps


def finite_stat(values, operation, *args):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    return float(getattr(np, operation)(v, *args)) if len(v) else None


def shade_unrecorded(ax, m):
    previous = 0
    for segment in m.get("recording_segments", []):
        start = segment["frame_offset"]
        if start > previous:
            ax.axvspan(previous / m["fps"] / 3600, start / m["fps"] / 3600,
                       color="#888888", alpha=.15)
        previous = start + segment["num_frames"]


def diagnose_cup(m, index):
    cup = m["cups"][index]
    cid = cup["cup_id"]
    tid = cup.get("track_id", cid)
    q = pd.read_csv(root(m) / "cup_identities.csv")
    identity = q.loc[q.get("TrackID", q.CupID).eq(tid)].iloc[0]
    if not identity.Eligible:
        return
    folder = root(m) / "diagnostics" / cup["name"]
    if (folder / "COMPLETE.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    track = root(m) / "per_track" / f"TrackID_{tid:04d}_all_single.parquet"
    d = pd.read_parquet(track)
    if d.duplicated(["Frame", "Bodypoint"]).any():
        raise AssertionError("Duplicated frame/bodypoint in stitched resident track")
    if len(d) and (not d.TrackID.eq(tid).all() or not d.CupID.eq(cid).all() or not d.ArucoID.eq(identity.ArucoID).all()):
        raise AssertionError("Stitched track changed cup or dedicated tag")
    n, fps = m["n_frames"], m["fps"]
    anchor = d.loc[d.Bodypoint.eq(0)].copy()
    frames = anchor.Frame.to_numpy(np.int64)
    if len(frames) and (frames.min() < 0 or frames.max() >= n):
        raise AssertionError("Track escaped the full recording timeline")
    x, y, aruco_x, aruco_y, sleap_x, sleap_y = [np.full(n, np.nan, np.float32) for _ in range(6)]
    for dest, col in zip([x, y, aruco_x, aruco_y, sleap_x, sleap_y],
                         ["TrackX", "TrackY", "ArucoX", "ArucoY", "SleapAnchorX", "SleapAnchorY"]):
        dest[frames] = anchor[col].to_numpy(np.float32)
    observed = np.isfinite(x) & np.isfinite(y)
    tagged = np.isfinite(aruco_x) & np.isfinite(aruco_y)
    posed = np.isfinite(sleap_x) & np.isfinite(sleap_y)
    source = np.zeros(n, np.uint8)
    source[tagged & posed] = 1
    source[observed & ~tagged & posed] = 2
    source[tagged & ~posed] = 3
    node_count = np.bincount(d.Frame.to_numpy(np.int64), minlength=n).astype(np.uint8)
    x_mm, y_mm = x * MM_PER_PX, y * MM_PER_PX
    radius = np.hypot(x_mm - cup["center_mm"][0], y_mm - cup["center_mm"][1])
    outside = int(np.count_nonzero(radius > cup["radius_mm"] + 1e-3))  # float32 plot-coordinate rounding
    if outside:
        raise AssertionError(f"{cup['name']}: {outside} positions outside manual cup")
    vector_dir = root(m) / "analysis" / "speed_vectors" / "per_track" / track.stem
    vector_dir.mkdir(parents=True, exist_ok=True)
    if m.get("preserve_block_speeds"):
        speed = np.load(vector_dir / "speed_mm_s.npy")
        unclipped = np.load(vector_dir / "speed_unclipped_mm_s.npy")
        if speed.shape != (n,) or unclipped.shape != (n,):
            raise ValueError("Stitched speed vector does not cover the combined timeline")
    else:
        speed = speed_vector(x, y, fps=fps, mm_per_px=MM_PER_PX, **SPEED)
        unclipped = speed_vector(x, y, fps=fps, mm_per_px=MM_PER_PX,
                                 **dict(SPEED, max_speed_mm_s=None))
        np.save(vector_dir / "speed_mm_s.npy", speed)
        np.save(vector_dir / "speed_unclipped_mm_s.npy", unclipped)
        write_json(vector_dir / "speed_metadata.json", dict(track_path=str(track), track_name=track.name,
                   track_id=tid, aruco_id=int(identity.ArucoID), speed_path=str(vector_dir / "speed_mm_s.npy"),
                   speed_units="mm/s", fps=fps, mm_per_px=MM_PER_PX, frame_min=0, frame_max=n-1,
                   n_frames=n, n_observed_frames=int(observed.sum()), n_valid_speed_frames=int(np.isfinite(speed).sum()),
                   x_col="TrackX", y_col="TrackY", bodypoint_filter=0, dtype=str(speed.dtype), **SPEED))
    distance = np.hypot(aruco_x - sleap_x, aruco_y - sleap_y) * MM_PER_PX
    step = np.hypot(np.diff(x_mm), np.diff(y_mm))
    recorded = m.get("recorded_frames", n)
    summary = dict(Cup=cup["name"], CupID=cid, TrackID=tid, ArucoID=int(identity.ArucoID), Frames=n,
        RecordedFrames=recorded, RecordedPositionFraction=float(observed.sum()/recorded),
        ObservedFrames=int(observed.sum()), PositionFraction=float(observed.mean()),
        PoseFraction=float(posed.mean()), Full10NodePoseFraction=float(((node_count == 10) & posed).mean()),
        TagAndPoseFraction=float((source == 1).mean()), SleapBridgeFraction=float((source == 2).mean()),
        ArucoOnlyFraction=float((source == 3).mean()), MissingFraction=float((~observed).mean()),
        ValidSpeedFraction=float(np.isfinite(speed).mean()), MedianSpeedMMS=finite_stat(speed, "median"),
        P95SpeedMMS=finite_stat(speed, "quantile", .95), MaxUnclippedSpeedMMS=finite_stat(unclipped, "max"),
        SpeedAbove20Frames=int(np.count_nonzero(unclipped > SPEED["max_speed_mm_s"])),
        MedianTagAnchorDistanceMM=finite_stat(distance, "median"), P95TagAnchorDistanceMM=finite_stat(distance, "quantile", .95),
        MaxConsecutiveStepMM=finite_stat(step, "max"), OutsideCupFrames=outside, PermanentStop=False)
    write_json(folder / "metrics.json", summary)
    width = max(1, int(round(fps)))
    time_h = np.arange(len(binned(x_mm, width))) * width / fps / 3600
    duration = n / fps / 3600
    speed_mean, speed_max = binned(speed, width), binned(speed, width, "max")
    unclipped_max = binned(unclipped, width, "max")
    minute = max(1, int(round(60 * fps)))
    minute_h = np.arange(len(binned(source, minute))) * minute / fps / 3600
    minute_coverage = binned(observed, minute)
    np.savez_compressed(folder / "binned_diagnostics.npz", time_hours=time_h,
        x_mm=binned(x_mm, width), y_mm=binned(y_mm, width), speed_mean_mm_s=speed_mean,
        speed_max_mm_s=speed_max, speed_unclipped_max_mm_s=unclipped_max,
        minute_hours=minute_h, position_coverage=minute_coverage,
        minute_speed_mm_s=binned(speed, minute),
        full_pose_coverage=binned((node_count == 10) & posed, minute),
        tag_pose_fraction=binned(source == 1, minute), bridge_fraction=binned(source == 2, minute))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    xy = np.c_[x_mm - cup["center_mm"][0], y_mm - cup["center_mm"][1]]
    keep = observed[:-1] & observed[1:]
    segments = np.stack([xy[:-1][keep], xy[1:][keep]], axis=1)
    lc = LineCollection(segments, cmap="viridis", norm=Normalize(0, duration), linewidths=.5, rasterized=True)
    lc.set_array(np.flatnonzero(keep) / fps / 3600)
    axes[0, 0].add_collection(lc)
    axes[0, 0].add_patch(Circle((0, 0), cup["radius_mm"], fill=False, color="black", lw=1))
    r = cup["radius_mm"] * 1.05
    axes[0, 0].set(xlim=(-r, r), ylim=(r, -r), aspect="equal", xlabel="x from cup center (mm)",
                   ylabel="y from cup center (mm)", title="Trajectory inside the manual cup")
    fig.colorbar(lc, ax=axes[0, 0], label="hours from trigger", shrink=.8)
    for values, center, color, label in [(x_mm, cup["center_mm"][0], "#2b6cb0", "x"),
                                          (y_mm, cup["center_mm"][1], "#b73f65", "y")]:
        axes[0, 1].fill_between(time_h, binned(values-center, width, "min"), binned(values-center, width, "max"), color=color, alpha=.15)
        axes[0, 1].plot(time_h, binned(values-center, width), color=color, lw=.6, label=label)
    axes[0, 1].set(ylabel="position from cup center (mm)", title="Position over time: 1-second mean and range")
    axes[0, 1].legend()
    axes[1, 0].fill_between(time_h, 0, speed_max, color="#409185", alpha=.25, label="1-second maximum")
    axes[1, 0].plot(time_h, speed_mean, color="#176b58", lw=.6, label="1-second mean")
    axes[1, 0].set(ylabel="speed (mm/s)", ylim=(0, SPEED["max_speed_mm_s"]), title="Standard colony speed calculation")
    axes[1, 0].legend()
    axes[1, 1].plot(time_h, unclipped_max, color="#925e43", lw=.5)
    axes[1, 1].axhline(20, color="#b62e2e", ls="--", lw=.8, label="standard 20 mm/s exclusion")
    axes[1, 1].set(yscale="symlog", ylabel="unclipped speed (mm/s)", title="Unclipped speed: jumps remain visible")
    axes[1, 1].legend()
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.set(xlim=(0, duration), xlabel="hours from common hardware trigger")
        shade_unrecorded(ax, m)
    fig.suptitle(f"{cup['name']} — dedicated ArUco ID {int(identity.ArucoID)} — {duration:.2f} hours\n"
                 f"recorded-frame coverage: position {summary['RecordedPositionFraction']:.1%}; posture {posed.sum()/recorded:.1%}; full 10-node posture {((node_count == 10) & posed).sum()/recorded:.1%}; no permanent stops", fontsize=13)
    fig.savefig(folder / "position_speed.png", dpi=160)
    fig.savefig(folder / "position_speed.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")
    fractions = [binned(source == k, minute) for k in [1, 2, 3, 0]]
    axes[0, 0].stackplot(minute_h, fractions, labels=["tag + posture", "SLEAP bridge", "tag only", "missing"],
                         colors=[COLORS[1], COLORS[2], COLORS[3], "#dddddd"])
    axes[0, 0].set(xlim=(0, duration), ylim=(0, 1), ylabel="fraction of each minute", xlabel="hours from trigger", title="Tracking source and coverage over the entire recording")
    axes[0, 0].legend(fontsize=8, ncol=2, loc="lower left")
    axes[0, 1].plot(minute_h, binned(node_count, minute), label="mean observed bodypoints", color="#316b9e")
    axes[0, 1].set(xlim=(0, duration), ylim=(0, 10), ylabel="bodypoints", xlabel="hours from trigger", title="Posture completeness before interpolation")
    twin = axes[0, 1].twinx()
    twin.plot(minute_h, binned((node_count == 10) & posed, minute) * 100, color="#28885b", label="complete posture (%)")
    twin.set(ylim=(0, 100), ylabel="frames with all 10 bodypoints (%)")
    axes[0, 1].legend(loc="lower left"); twin.legend(loc="upper right")
    for ax in axes[0]:
        shade_unrecorded(ax, m)
    for valid, label, color in [(observed, "missing position", "#686868"), (tagged, "missing tag", "#e58a22")]:
        gaps = gap_lengths(valid, fps)
        if len(gaps):
            edges = np.geomspace(1/fps, max(2/fps, float(gaps.max()) * 1.01), 35)
            axes[1, 0].hist(gaps, bins=edges, histtype="step", label=label, color=color)
    axes[1, 0].set(xscale="log", yscale="log", xlabel="gap length (seconds)", ylabel="number of gaps", title="Detection gaps are retained on the true timeline")
    axes[1, 0].legend()
    values = distance[np.isfinite(distance)]
    if len(values):
        axes[1, 1].hist(values, bins=80, color="#458780")
    axes[1, 1].axvline(m["tracking"]["aruco_sleap_max_distance"] * MM_PER_PX, color="black", ls="--")
    axes[1, 1].set(xlabel="tag to assigned posture anchor (mm)", ylabel="frames", title="Actual native colony matching distances")
    fig.suptitle(f"{cup['name']} — ID {int(identity.ArucoID)} — coverage and tracking diagnostics", fontsize=14)
    fig.savefig(folder / "quality.png", dpi=160)
    fig.savefig(folder / "quality.pdf")
    plt.close(fig)
    write_json(folder / "COMPLETE.json", dict(cup=cid, track=str(track), frames=n, permanent_stops=0))
    print(json.dumps(summary), flush=True)


def aggregate_report(m):
    output = root(m)
    q = pd.read_csv(output / "cup_identities.csv")
    summaries, series = [], []
    for cup in m["cups"]:
        if not q.loc[q.get("TrackID", q.CupID).eq(cup.get("track_id", cup["cup_id"])), "Eligible"].iloc[0]:
            continue
        folder = output / "diagnostics" / cup["name"]
        if not (folder / "COMPLETE.json").exists():
            raise ValueError(f"Diagnostics incomplete: {folder}")
        summaries.append(json.loads((folder / "metrics.json").read_text()))
        series.append(np.load(folder / "binned_diagnostics.npz"))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "tracking_summary.csv", index=False)
    if summary.empty:
        raise ValueError("No cups produced full tracks")
    figures = output / "plots"
    figures.mkdir(exist_ok=True)
    duration = m["n_frames"] / m["fps"] / 3600
    fig, axes = plt.subplots(1, 2, figsize=(17, 10), layout="constrained")
    panorama = output / "panorama.png"
    if panorama.exists():
        meta = json.loads((output / "panorama_metadata.json").read_text())
        b, scale = meta["raw_tracking_bounds_px"], meta["mm_per_pixel"]
        axes[0].imshow(plt.imread(panorama), cmap="gray", alpha=.6,
                       extent=[b["x_min"]*scale, b["x_max"]*scale, b["y_max"]*scale, b["y_min"]*scale])
    for cup in m.get("physical_cups", m["cups"]):
        cid = cup["cup_id"]
        entries = summary.loc[summary.CupID.eq(cid)]
        found = len(entries) > 0
        frac = float(entries.RecordedPositionFraction.sum()) if found else 0.
        color = plt.cm.RdYlGn(frac) if found else "#999999"
        axes[0].add_patch(Circle(cup["center_mm"], cup["radius_mm"], fill=False, color=color, lw=1.5))
        tags = "/".join(str(int(v)) for v in entries.ArucoID)
        text = f"{cup['name']}\nID {tags}\n{frac:.0%}" if found else f"{cup['name']}\nunassigned"
        axes[0].text(*cup["center_mm"], text, ha="center", va="center", fontsize=6,
                     bbox=dict(fc="white", ec="none", alpha=.75, pad=1))
    if not panorama.exists():
        axes[0].autoscale_view(); axes[0].invert_yaxis()
    axes[0].set(aspect="equal", xlabel="calibrated x (mm)", ylabel="calibrated y (mm)", title="Manual cups and full-duration position coverage")
    y = np.arange(len(summary))
    axes[1].barh(y-.18, summary.RecordedPositionFraction*100, height=.36, color="#287e63", label="position")
    axes[1].barh(y+.18, summary.Full10NodePoseFraction*m["n_frames"]/m.get("recorded_frames",m["n_frames"])*100,
                 height=.36, color="#547dba", label="all 10 posture nodes")
    axes[1].set(yticks=y, yticklabels=summary.Cup, xlabel="all recording frames (%)", xlim=(0, 105), title="Position and posture coverage")
    axes[1].invert_yaxis(); axes[1].legend(loc="lower right")
    title = m.get("title", "Single-ant tracking")
    fig.suptitle(f"{title} — {duration:.2f} hours — {len(summary)} identified ant records\n"
                 "Native colony tracking inside manual cups; extra IDs never permanently censor tracks", fontsize=14)
    fig.savefig(figures / "cup_coverage.png", dpi=170)
    summary_pdf = output / "summary_report.pdf"
    full_pdf = output / "full_tracking_diagnostics.pdf"
    with PdfPages(summary_pdf) as pdf, PdfPages(full_pdf) as full:
        pdf.savefig(fig); full.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(17, max(9, len(summary)*.28)), layout="constrained")
        # All speed samples contribute to these one-minute means. Detailed
        # per-cup plots retain one-second means and maxima, including outliers.
        speed_matrix = np.stack([s["minute_speed_mm_s"] for s in series])
        coverage_matrix = np.stack([s["position_coverage"] for s in series])
        image = axes[0].imshow(speed_matrix, aspect="auto", origin="upper", cmap="magma", vmin=0,
                              vmax=20, extent=[0, duration, len(summary)-.5, -.5], interpolation="nearest")
        axes[0].set(yticks=y, yticklabels=summary.Cup, xlabel="hours from trigger", title="Standard colony speed: one-minute mean (mm/s)")
        fig.colorbar(image, ax=axes[0], label="mm/s")
        image = axes[1].imshow(coverage_matrix*100, aspect="auto", origin="upper", cmap="YlGn", vmin=0,
                              vmax=100, extent=[0, duration, len(summary)-.5, -.5], interpolation="nearest")
        axes[1].set(yticks=y, yticklabels=summary.Cup, xlabel="hours from trigger", title="Observed position coverage in each minute")
        fig.colorbar(image, ax=axes[1], label="observed frames (%)")
        fig.suptitle("Full recording motion and tracking coverage", fontsize=15)
        fig.savefig(figures / "speed_and_coverage_over_time.png", dpi=170)
        pdf.savefig(fig); full.savefig(fig); plt.close(fig)
        for cup in summary.Cup:
            for kind in ["position_speed", "quality"]:
                fig = plt.figure(figsize=(16, 10), layout="constrained")
                ax = fig.add_subplot(111)
                ax.imshow(plt.imread(output / "diagnostics" / cup / f"{kind}.png"))
                ax.set_axis_off()
                full.savefig(fig, dpi=170)
                plt.close(fig)
    for s in series:
        s.close()
    write_json(output / "FULL_RUN_COMPLETE.json", dict(protocol=m["protocol"], frames=m["n_frames"], fps=m["fps"],
               hours=duration, ant_records=len(summary), assigned_cups=int(summary.CupID.nunique()),
               labeled_cups=len(m.get("physical_cups", m["cups"])),
               outside_cup_positions=int(summary.OutsideCupFrames.sum()), permanent_stops=0,
               standard_speed_settings=SPEED, summary_csv=str(output / "tracking_summary.csv"),
               diagnostic_report=str(full_pdf), summary_report=str(summary_pdf)))


def relocate_publication_metadata(m, destination):
    """Keep published caches usable after the scratch run is cleaned up."""
    output, destination = root(m), Path(destination)
    from tracking.colony.combine_blocks import fingerprint
    locations = dict(m.get("publication_relocations", {}))
    locations[str(output)] = str(destination)
    locations = sorted(locations.items(), key=lambda pair: len(pair[0]), reverse=True)
    def relocate(value):
        if isinstance(value, dict):
            result = {key: relocate(item) for key, item in value.items()}
            if (set(value) == {"path", "size", "mtime_ns"}
                    and result["path"] != value["path"]):
                # Published JSON metadata can differ in size and mtime after
                # paths are rewritten; pin its actual published counterpart.
                return fingerprint(result["path"])
            return result
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str):
            for source, target in locations:
                if value == source or value.startswith(source + "/"):
                    return target + value[len(source):]
        return value
    paths = list(output.glob("*.json"))
    for name in ["analysis", "diagnostics", "combination_checks"]:
        paths.extend((output / name).rglob("*.json"))
    for path in paths:
        target = destination / path.relative_to(output)
        if target.is_file():
            original = json.loads(path.read_text())
            published = relocate(original)
            if published != original:
                write_json(target, published)


def publish(m, destination):
    output, destination = root(m), Path(destination)
    if not (output / "FULL_RUN_COMPLETE.json").exists():
        raise ValueError("Cannot publish an unfinished run")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ["per_track", "analysis", "diagnostics", "plots", "combination_checks"]:
        if (output / name).is_dir():
            subprocess.run(["rsync", "-a", "--partial", str(output / name) + "/", str(destination / name) + "/"], check=True)
    for path in output.iterdir():
        if path.is_file() and path.suffix in {".json", ".csv", ".tsv", ".pdf", ".png"}:
            shutil.copy2(path, destination / path.name)
    relocate_publication_metadata(m, destination)
    write_json(destination / "PUBLICATION_COMPLETE.json", dict(source=str(output), published=str(destination),
               frames=m["n_frames"], permanent_stops=0,
               metadata_relocations=m.get("publication_relocations", {})))
