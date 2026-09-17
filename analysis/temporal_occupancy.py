"""Four-hour occupancy clustering, fixed-reference UMAP, and change dynamics.

Consumes exact half-hour count caches made by temporal_occupancy_cache.py.
Independent Leiden labels are local to each window; continuity is measured
using feature distances and label-invariant coassignment, never label numbers.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

from analysis import grid_occupancy_utils as go
from analysis import long_timescale_cluster_switching as cs
from analysis import temporal_occupancy_cache as cache


def calendar_windows(manifest, hours):
    """Calendar-aligned windows, union of actual source intervals, no gap fill."""
    intervals = {}
    width = int(hours*3600*10**9)
    for index, window in enumerate(manifest["windows"]):
        start = pd.Timestamp(window["start"]).value
        stop = pd.Timestamp(window["stop"]).value + round(10**9/manifest["sources"][index]["info"]["fps"])
        for b in range(start//width, (stop-1)//width+1):
            intervals.setdefault(b, []).append((max(start, b*width), min(stop, (b+1)*width), index))
    rows = []
    for b, spans in sorted(intervals.items()):
        spans.sort()
        total = 0
        last = None
        gaps = []
        for start, stop, _ in spans:
            if last is not None:
                if start < last-100:
                    raise ValueError("Overlapping source recordings would double-count positions")
                gaps.append((start-last)/10**9)
            total += stop-start
            last = stop
        rows.append(dict(bin=b, hours=hours, start=pd.Timestamp(b*width), stop=pd.Timestamp((b+1)*width),
                         center=pd.Timestamp(b*width+width//2), recorded_hours=total/3.6e12,
                         recording_fraction=total/width, first_observed=pd.Timestamp(spans[0][0]),
                         last_observed=pd.Timestamp(spans[-1][1]), max_recording_gap_seconds=max(gaps, default=0),
                         source_indices=json.dumps(sorted({s[2] for s in spans}))))
    return pd.DataFrame(rows)


def aggregate_atoms(tasks, root, hours):
    histograms, n_detected, provenance = {}, {}, []
    edges = {}
    factor = int(hours*2)
    if factor != hours*2 or factor < 1:
        raise ValueError("Window width must be a positive multiple of half an hour")
    for task in tasks:
        ant = task["ant"]
        side = ant.split(":")[0]
        for entry in task["entries"]:
            path = root/"atoms"/str(entry["block_index"])/(ant.replace(":", "_")+".npz")
            meta = json.loads(path.with_suffix(".json").read_text())
            if not meta["original_histogram_exact"] or meta["signature"]["entry"] != entry:
                raise ValueError(f"Unverified or mismatched atom cache: {path}")
            provenance.append(cache.stamp(path))
            with np.load(path) as data:
                if side in edges:
                    assert np.array_equal(edges[side][0], data["x_edges"]) and np.array_equal(edges[side][1], data["y_edges"])
                else:
                    edges[side] = (data["x_edges"].copy(), data["y_edges"].copy())
                bins = data["calendar_bin"]//factor
                for b in np.unique(bins):
                    key = (int(b), ant)
                    counts = data["counts"][bins == b].sum(axis=0, dtype=np.int64)
                    detected = int(data["detected"][bins == b].sum())
                    if key in histograms:
                        histograms[key] += counts
                        n_detected[key] += detected
                    else:
                        histograms[key] = counts
                        n_detected[key] = detected
    maps, rows = {}, []
    for (b, ant), counts in sorted(histograms.items()):
        n = n_detected[b, ant]
        key = f"{hours:g}|{b}|{ant}"
        maps[key] = (counts/max(n, 1)).astype(np.float32)
        rows.append(dict(profile_key=key, bin=b, hours=hours, ant=ant, side=ant.split(":")[0],
                         n_detected=n, n_in_arena=int(counts.sum()), occupancy_sum=float(counts.sum()/n) if n else 0))
    return pd.DataFrame(rows), maps, edges, provenance


def temporal_behavior(bins, hours):
    """Reuse exact exposure-weighted existing behavior and optional trip tables."""
    data = bins.copy()
    data["bin"] = data.timestamp.dt.as_unit("ns").astype("int64")//int(hours*3600*1e9)
    data["source_block"] = data.bin.astype(str)
    out = cs.block_metrics(data).rename(columns={"source_block": "bin"})
    out["bin"] = out.bin.astype(int)
    return out


def reference_models(prior, output):
    import umap
    import joblib
    refs = pd.read_parquet(prior/"reference.parquet")
    if set(refs.reference_index) != {1}:
        raise ValueError("This analysis expects the requested July 23 block02 reference (index 1)")
    models = {}
    with np.load(prior/"occupancy_maps.npz") as saved:
        for side, r in refs.groupby("side", sort=True):
            r = r.reset_index(drop=True)
            raw = np.stack([saved[key].ravel() for key in r.profile_key])
            features = go.feature_transform_matrix(raw, "sqrt")
            model = umap.UMAP(n_neighbors=10, min_dist=.1, metric="euclidean", random_state=0)
            xy = model.fit_transform(features)
            np.testing.assert_allclose(xy, r[["umap_x", "umap_y"]].to_numpy(), rtol=0, atol=1e-5,
                                       err_msg="Recreated reference UMAP differs from saved 0723 space")
            joblib.dump(model, output/f"reference_umap_{side}.joblib")
            classes = sorted(r.original_cluster.unique())
            if len(classes) != 2:
                raise ValueError("The continuous 0-to-1 reference axis currently expects two reference clusters per colony")
            prototypes = np.stack([np.sqrt(raw[r.original_cluster.eq(label)].mean(axis=0)) for label in classes])
            models[side] = dict(model=model, reference=r, features=features,
                                labels=r.original_cluster.to_numpy(str), classes=classes, prototypes=prototypes)
    return models, refs


def assign_reference(table, maps, models, manifest, hours, *, project=True):
    rows, calibration = [], []
    reference_window = manifest["windows"][1]
    for side, part in table.groupby("side", sort=True):
        cfg = models[side]
        ref, features = cfg["reference"], cfg["features"]
        valid = part[part.occupancy_sum.gt(0)].copy().reset_index(drop=True)
        query = go.feature_transform_matrix(np.stack([maps[key].ravel() for key in valid.profile_key]), "sqrt")
        distances = cdist(query, features)
        votes = {k: cs.neighbor_vote(distances, cfg["labels"], k) for k in (3, 5, 10)}
        own = {ant: i for i, ant in enumerate(ref.ant)}
        excluded = cs.neighbor_vote(distances, cfg["labels"], 5, [own.get(ant, -1) for ant in valid.ant])
        mean_loo_distance = np.array([v["mean_distance"] for v in excluded])
        within_reference = (valid.start.ge(pd.Timestamp(reference_window["start"])) &
                            valid.stop.le(pd.Timestamp(reference_window["stop"])) & valid.ant.isin(own) & valid.recording_fraction.ge(.95))
        if not within_reference.any():
            raise ValueError(f"No complete {hours} h windows available to calibrate reference variability")
        cutoff = float(np.quantile(mean_loo_distance[within_reference], .95))
        good = within_reference & valid.coverage.ge(.4)
        good_cutoff = float(np.quantile(mean_loo_distance[good], .95)) if good.any() else np.nan
        original_cutoff = float(ref.distance_cutoff.iloc[0])
        for i in np.flatnonzero(within_reference):
            calibration.append(dict(hours=hours, side=side, ant=valid.iloc[i].ant, bin=int(valid.iloc[i].bin),
                                    mean_distance=mean_loo_distance[i], coverage=valid.iloc[i].coverage,
                                    cutoff=cutoff, well_observed_cutoff=good_cutoff, whole_block_cutoff=original_cutoff,
                                    prediction=excluded[i]["prediction"], original_cluster=ref.iloc[own[valid.iloc[i].ant]].original_cluster))
        xy = cfg["model"].transform(query) if project else np.full((len(query), 2), np.nan)
        proto_dist = cdist(query, cfg["prototypes"])
        for i, row in enumerate(valid.itertuples(index=False)):
            r = row._asdict()
            v = votes[5][i]
            label = ref.iloc[own[row.ant]].original_cluster if row.ant in own else None
            agreement = len({votes[k][i]["prediction"] for k in votes}) == 1
            ratio = v["mean_distance"]/cutoff
            supported = v["vote"] >= .8 and agreement and excluded[i]["prediction"] == v["prediction"] and ratio <= 1
            weights_cluster1 = float(v["weights"][cfg["labels"][v["indices"]] == cfg["classes"][1]].sum())
            r.update(reference_cluster=label, prediction=v["prediction"], vote=v["vote"], k_agreement=agreement,
                     mean_distance=v["mean_distance"], novelty_ratio=ratio, whole_distance_ratio=v["mean_distance"]/original_cutoff,
                     robust_distance_ratio=v["mean_distance"]/good_cutoff, own_excluded_prediction=excluded[i]["prediction"],
                     cluster1_vote=weights_cluster1,
                     reference_axis=float((proto_dist[i, 0]-proto_dist[i, 1])/max(proto_dist[i].sum(), 1e-12)),
                     umap_x=float(xy[i, 0]), umap_y=float(xy[i, 1]), supported=supported,
                     switch=label is not None and v["prediction"] != label,
                     reliable=row.coverage >= .4 and row.recording_fraction >= .95,
                     calibration_window=bool(within_reference.iloc[i]), assignment_status="assigned")
            r["supported_switch"] = bool(r["switch"] and supported and r["reliable"])
            rows.append(r)
        for row in part[part.occupancy_sum.le(0)].to_dict("records"):
            label = ref.iloc[own[row["ant"]]].original_cluster if row["ant"] in own else None
            rows.append(dict(**row, reference_cluster=label, prediction=None, assignment_status="no in-arena observations",
                             switch=False, supported_switch=False, supported=False, reliable=False, calibration_window=False))
    return pd.DataFrame(rows), pd.DataFrame(calibration)


def local_clustering(table, maps, models):
    """Fit every four-hour window independently; labels have no temporal ID."""
    out = table.copy()
    out["local_cluster"] = pd.Series(index=out.index, dtype="object")
    prototypes = []
    for (b, side), part in out[out.occupancy_sum.gt(0)].groupby(["bin", "side"]):
        matrix = np.stack([maps[key].ravel() for key in part.profile_key])
        features = go.feature_transform_matrix(matrix, "sqrt")
        labels = go.leiden_labels(features, n_neighbors=10, metric="euclidean", resolution=1., random_state=0)
        out.loc[part.index, "local_cluster"] = [f"{side}:{b}:{label}" for label in labels]
        for label in sorted(set(labels)):
            selected = part.iloc[np.flatnonzero(labels == label)]
            prototype = np.sqrt(matrix[labels == label].mean(axis=0))
            d = cdist(prototype[None, :], models[side]["prototypes"])[0]
            prototypes.append(dict(bin=b, side=side, local_cluster=f"{side}:{b}:{label}", n_ants=len(selected),
                                   n_reliable=int(selected.reliable.sum()), n_reference=int(selected.reference_cluster.notna().sum()),
                                   nearest_reference=models[side]["classes"][int(d.argmin())],
                                   reference_distance=float(d.min()), median_colony_percent=float(selected.colony_percent.median()),
                                   median_novelty_ratio=float(selected.novelty_ratio.median())))
    return out, pd.DataFrame(prototypes)


def continuity(table, maps):
    """Adjacent observations only; actual gaps break lines, regardless of IDs."""
    steps, partitions = [], []
    for side, colony in table.groupby("side"):
        groups = {b: p.set_index("ant") for b, p in colony.groupby("bin")}
        for b, current in groups.items():
            if b-1 not in groups:
                continue
            previous = groups[b-1]
            gap = (current.first_observed.iloc[0]-previous.last_observed.iloc[0]).total_seconds()
            if gap > 300:
                continue
            common = previous.index.intersection(current.index)
            common = [ant for ant in common if previous.loc[ant].occupancy_sum > 0 and current.loc[ant].occupancy_sum > 0]
            for ant in common:
                a, z = previous.loc[ant], current.loc[ant]
                distance = float(np.linalg.norm(np.sqrt(maps[a.profile_key])-np.sqrt(maps[z.profile_key])))
                steps.append(dict(ant=ant, side=side, bin=b, previous_bin=b-1, hours=z.hours, center=z.center,
                                  feature_step=distance, reliable=bool(a.reliable and z.reliable),
                                  coverage_min=min(a.coverage, z.coverage),
                                  reference_ant=pd.notna(z.reference_cluster),
                                  reference_axis_change=z.reference_axis-a.reference_axis,
                                  colony_change=z.colony_percent-a.colony_percent,
                                  fixed_label_changed=a.prediction != z.prediction,
                                  original_switch=bool(z.switch), recording_gap_seconds=gap))
            if "local_cluster" in colony:
                for mode in ("all observed", "both endpoints well observed"):
                    identities = common if mode == "all observed" else [ant for ant in common if previous.loc[ant].reliable and current.loc[ant].reliable]
                    if len(identities) >= 2:
                        partitions.append(dict(side=side, bin=b, previous_bin=b-1, center=current.center.iloc[0], mode=mode,
                                               n_ants=len(identities), adjusted_rand=adjusted_rand_score(previous.loc[identities].local_cluster, current.loc[identities].local_cluster),
                                               previous_clusters=previous.loc[identities].local_cluster.nunique(), current_clusters=current.loc[identities].local_cluster.nunique()))
    return pd.DataFrame(steps), pd.DataFrame(partitions)


def fit_change_models(time_hours, values, weights, light_on=5.5, light_off=19.5):
    """Descriptive constant/circadian, drift and step alternatives; no p-values.

    BIC counts the searched step location as an additional parameter. Hourly
    autocorrelation remains; these rankings do not establish abrupt biology.
    """
    t, y, w = np.asarray(time_hours), np.asarray(values), np.asarray(weights)
    if len(y) < 12:
        return []
    clock = t % 24
    base = np.column_stack([np.ones(len(t)), np.sin(2*np.pi*clock/24), np.cos(2*np.pi*clock/24), ((clock >= light_on) & (clock < light_off)).astype(float)])
    def fit(x, parameters):
        if np.linalg.matrix_rank(x) < x.shape[1]:
            return None
        beta = np.linalg.lstsq(x*np.sqrt(w[:, None]), y*np.sqrt(w), rcond=None)[0]
        prediction = x @ beta
        rss = float(np.sum(w*(y-prediction)**2))
        return dict(bic=len(y)*np.log(max(rss/w.sum(), 1e-12))+parameters*np.log(len(y)),
                    weighted_rmse=np.sqrt(rss/w.sum()), fitted=prediction, beta=beta)
    results = []
    constant = fit(base, 4)
    if constant is None:
        return []
    results.append(dict(model="circadian only", **constant, change=0., break_hour=np.nan, break_gap_hours=np.nan))
    trend = fit(np.column_stack([base, (t-t[0])/(t[-1]-t[0])]), 5)
    if trend:
        results.append(dict(model="gradual drift", **trend, change=float(trend["beta"][-1]), break_hour=np.nan, break_gap_hours=np.nan))
    best = None
    for split in range(3, len(t)-3):
        step = fit(np.column_stack([base, (np.arange(len(t)) >= split).astype(float)]), 6)
        if step and (best is None or step["bic"] < best["bic"]):
            best = dict(model="single step", **step, change=float(step["beta"][-1]), break_hour=float((t[split]+t[split-1])/2),
                        break_gap_hours=float(t[split]-t[split-1]))
    if best:
        results.append(best)
    return results


def change_models(hourly, manifest):
    rows, fitted = [], []
    for ant, part in hourly[hourly.reference_cluster.notna()].groupby("ant"):
        part = part.sort_values("bin").copy()
        # Gaps in recording, not missing ant detections, define observed runs.
        breaks = part.bin.diff().ne(1) | (part.first_observed-part.last_observed.shift()).dt.total_seconds().gt(300)
        part["segment"] = breaks.cumsum()
        for segment, all_rows in part.groupby("segment"):
            valid = all_rows[all_rows.reliable & all_rows.reference_axis.notna()]
            if len(valid) < 12:
                continue
            t = valid.center.dt.as_unit("ns").astype("int64").to_numpy()/3.6e12
            for metric in ("reference_axis", "colony_percent"):
                selection = valid[metric].notna()
                results = fit_change_models(t[selection], valid.loc[selection, metric], valid.loc[selection, "coverage"],
                                            manifest["light_on_hour"], manifest["light_off_hour"])
                if not results:
                    continue
                best = min(r["bic"] for r in results)
                ordered = sorted(r["bic"] for r in results)
                for result in results:
                    rows.append(dict(ant=ant, side=ant.split(":")[0], segment=int(segment), metric=metric,
                                     start=valid.start.min(), stop=valid.stop.max(), n_hours=int(selection.sum()),
                                     model=result["model"], bic=result["bic"], delta_bic=result["bic"]-best,
                                     winning_margin=ordered[1]-ordered[0], weighted_rmse=result["weighted_rmse"],
                                     change=result["change"], break_time=pd.Timestamp(result["break_hour"]*3.6e12) if np.isfinite(result["break_hour"]) else pd.NaT,
                                     break_gap_hours=result["break_gap_hours"]))
                    for b, prediction in zip(valid.loc[selection, "bin"], result["fitted"]):
                        fitted.append(dict(ant=ant, segment=int(segment), metric=metric, model=result["model"], bin=b, fitted=float(prediction)))
    return pd.DataFrame(rows), pd.DataFrame(fitted)


def run(source, output, tasks_path, *, tables_only=False):
    from analysis import temporal_occupancy_plots as plots
    source, output = Path(source), Path(output)
    (output/"COMPLETE.json").unlink(missing_ok=True)
    manifest = json.loads((source/"run_manifest.json").read_text())
    rates = {s["info"]["fps"] for s in manifest["sources"]}
    if len(rates) != 1:
        raise ValueError("Pooling detected-frame counts requires the same fps across sources")
    fps = float(next(iter(rates)))
    prior = source/"cluster_switching"
    models, refs = reference_models(prior, output)
    tasks = json.loads(Path(tasks_path).read_text())
    behavior = pd.read_parquet(source/"task_bins.parquet")
    tables, all_maps, stamps = {"reference": refs}, {}, []
    for hours in (4, 1):
        table, maps, edges, atoms = aggregate_atoms(tasks, output, hours)
        windows = calendar_windows(manifest, hours)
        table = table.merge(windows, on=["bin", "hours"], validate="many_to_one")
        table["coverage"] = table.n_detected / (table.recorded_hours*3600*fps)
        if table.coverage.gt(1.000001).any():
            raise ValueError("Position counts exceed recorded time")
        table = table.merge(temporal_behavior(behavior, hours), on=["bin", "ant"], how="left", validate="one_to_one", indicator=True)
        if table._merge.ne("both").any():
            raise ValueError("Temporal profile keys lack behavior-bin entries; check timestamp units and calendar alignment")
        table = table.drop(columns="_merge")
        table, calibration = assign_reference(table, maps, models, manifest, hours, project=hours == 4)
        if hours == 4:
            table, local = local_clustering(table, maps, models)
            tables["local_clusters"] = local
        steps, partitions = continuity(table, maps)
        tables[f"profiles_{hours}h"], tables[f"windows_{hours}h"] = table, windows
        tables[f"calibration_{hours}h"], tables[f"steps_{hours}h"] = calibration, steps
        if hours == 4:
            tables["partition_continuity"] = partitions
        all_maps.update(maps)
        stamps += atoms
        print("TEMPORAL_TABLES", hours, len(table), flush=True)
    tables["change_models"], tables["model_fits"] = change_models(tables["profiles_1h"], manifest)
    tables["whole_assignments"] = pd.read_parquet(prior/"assignments.parquet")
    # Whole-recording profiles projected by the identical fitted UMAP transform.
    whole = pd.read_parquet(prior/"profiles.parquet")
    with np.load(prior/"occupancy_maps.npz") as saved:
        all_maps.update({"whole|"+key:saved[key].copy() for key in saved.files})
        for side, part in whole[whole.occupancy_sum.gt(0)].groupby("side"):
            xy = models[side]["model"].transform(go.feature_transform_matrix(np.stack([saved[k].ravel() for k in part.profile_key]), "sqrt"))
            whole.loc[part.index, ["umap_x", "umap_y"]] = xy
    tables["whole_projected"] = whole
    for name, table in tables.items():
        table.to_parquet(output/(name+".parquet"), index=False)
        table.to_csv(output/(name+".csv"), index=False)
    np.savez_compressed(output/"maps_4h.npz", **{k:v for k,v in all_maps.items() if k.startswith("4|")})
    np.savez_compressed(output/"grid_edges.npz", **{side+"_"+axis:values for side,pair in edges.items() for axis,values in zip("xy",pair)})
    if tables_only:
        result=dict(profiles_4h=len(tables["profiles_4h"]),profiles_1h=len(tables["profiles_1h"]),
                    next_stage="temporal_occupancy_render.py: ant shards followed by --finish")
        (output/"tables_complete.json").write_text(json.dumps(result,indent=2)+"\n")
        print("TEMPORAL_TABLES_COMPLETE",json.dumps(result),flush=True)
        return
    plots.save_figures(tables, all_maps, edges, manifest, output)
    plots.write_explorer(tables, all_maps, edges, manifest, output)
    plots.write_report(tables, manifest, output)
    provenance = dict(created=datetime.now().isoformat(), source=str(source), reference_window=manifest["windows"][1],
                      atom_sources=stamps, input_sources=[cache.stamp(source/name) for name in ("run_manifest.json", "task_bins.parquet")],
                      reference_sources=[cache.stamp(prior/name) for name in ("reference.parquet", "occupancy_maps.npz", "assignments.parquet")],
                      projection="UMAP fit to original 0723 block02 labeled maps only; original reference coordinates reproduced to 1e-5; transform of later maps",
                      windows="calendar aligned 4h; half-hour exact counts; one-hour supplementary trajectories; observed block contributions pooled at near-continuous handoffs",
                      labels="KNN reference labels fixed; independent Leiden labels local to each window; ARI compares partitions without label matching",
                      novelty="both original whole-block cutoff and duration-matched reference-window leave-own-ant-out 95th percentile; alternative >=40% reference coverage cutoff retained",
                      model_comparison="descriptive weighted circadian-only, linear drift and searched single-step fits on hourly profiles with >=40% detection and >=95% recorded window; autocorrelation not accounted for by BIC",
                      code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(plots.__file__),Path(cache.__file__),Path(go.__file__),Path(cs.__file__))},
                      software={name:importlib.metadata.version(name) for name in ("numpy","pandas","scipy","umap-learn","scikit-learn","leidenalg","plotly")})
    (output/"run_manifest.json").write_text(json.dumps(provenance,indent=2)+"\n")
    result=dict(profiles_4h=len(tables["profiles_4h"]),profiles_1h=len(tables["profiles_1h"]),
                figures=len(list(output.glob("*.png"))), individual_figures=len(list((output/"individual_ants").glob("*.png"))))
    (output/"COMPLETE.json").write_text(json.dumps(result,indent=2)+"\n")
    print("TEMPORAL_ANALYSIS_COMPLETE",json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-folder",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--tasks",type=Path,required=True)
    p.add_argument("--tables-only",action="store_true",help="Save numerical results, then render ants with independent jobs")
    args=p.parse_args()
    run(args.analysis_folder,args.output,args.tasks,tables_only=args.tables_only)


if __name__ == "__main__":
    main()
