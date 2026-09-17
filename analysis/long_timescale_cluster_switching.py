"""Map later cached occupancy into a fixed first-block clustering with KNN.

Saved cluster IDs are authoritative. Later blocks never train the classifier or
the reference UMAP. All cached ants are assigned and audited; switching requires
the same colony-side/tag identity to have an original reference label.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from analysis import grid_occupancy_utils as go
from analysis import long_timescale_utils as lt

K_VALUES = (3, 5, 10)


def neighbor_vote(distances, labels, k=5, excluded=None):
    """Inverse-distance vote, deterministic ties, exact matches get all weight.

    excluded holds one reference row per query (-1 means none). It supports
    leave-one-out calibration and excluding the query ant's own baseline map.
    """
    distances = np.asarray(distances, dtype=float).copy()
    labels = np.asarray(labels, dtype=str)
    if excluded is not None:
        for row, col in enumerate(excluded):
            if col >= 0:
                distances[row, col] = np.inf
    results = []
    for row in distances:
        order = np.argsort(row, kind="stable")
        indices = order[np.isfinite(row[order])][:k]
        if not len(indices):
            raise ValueError("No eligible reference neighbors")
        d = row[indices]
        exact = d <= 1e-12
        weights = exact.astype(float) if exact.any() else 1 / np.maximum(d, 1e-12)
        weights /= weights.sum()
        totals = {label: float(weights[labels[indices] == label].sum()) for label in sorted(set(labels))}
        ranking = sorted(totals, key=lambda label: (-totals[label], label))
        results.append(dict(prediction=ranking[0], vote=totals[ranking[0]],
                            margin=totals[ranking[0]] - (totals[ranking[1]] if len(ranking) > 1 else 0),
                            mean_distance=float(d.mean()), nearest_distance=float(d[0]),
                            indices=indices, weights=weights, distances=d))
    return results


def block_metrics(bins):
    rows = []
    metrics = {"colony_percent": "position_coverage", "food_percent": "resource_coverage",
               "water_percent": "resource_coverage", "mean_speed_mm_s": "speed_coverage",
               "sleep_percent": "sleep_coverage"}
    for (block, ant), part in bins.groupby(["source_block", "ant"]):
        row = dict(source_block=block, ant=ant)
        for metric, coverage in metrics.items():
            w = part.n_expected_frames * part[coverage]
            good = part[metric].notna() & w.gt(0)
            row[metric] = float(np.average(part.loc[good, metric], weights=w[good])) if good.any() else np.nan
        available = bool(part.trip_available.any())
        count = float(part.trip_count.sum()) if available else np.nan
        hours = float(part.trip_observed_hours.sum()) if available else np.nan
        row.update(trip_count=count, trip_observed_hours=hours,
                   trip_rate=count / hours if hours > 0 else np.nan,
                   mean_trip_minutes=float(part.trip_duration_sum_minutes.sum()) / count if count > 0 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def matched_behavior(bins, profiles, reference_indices):
    """Supplementary behavior check on shared clock bins, not new grid fitting.

    Average repeated cycles within each block using observed-frame weights,
    then give each shared half-hour clock slot equal weight in every block.
    Matching is per ant and metric and retains slots with positive observations
    in all compared blocks. No coverage threshold is imposed.
    """
    data = bins.merge(profiles[["source_block", "ant", "block_index"]],
                      on=["source_block", "ant"], validate="many_to_one")
    widths = bins.bin_minutes.dropna().unique()
    if len(widths) != 1:
        raise ValueError("Clock matching requires one common bin width")
    rows = []
    for reference_index in reference_indices:
        ref_ants = set(profiles.loc[profiles.block_index.eq(reference_index) & profiles.original_selected, "ant"])
        blocks = sorted(profiles.loc[profiles.block_index.ge(reference_index), "block_index"].unique())
        part = data[data.block_index.isin(blocks) & data.ant.isin(ref_ants)]
        for metric, coverage in [("colony_percent", "position_coverage"), ("mean_speed_mm_s", "speed_coverage"), ("sleep_percent", "sleep_coverage")]:
            d = part.copy()
            d["weight"] = d.n_expected_frames * d[coverage]
            d = d[d[metric].notna() & d.weight.gt(0)].copy()
            d["weighted_value"] = d[metric] * d.weight
            sums = d.groupby(["ant", "block_index", "clock_hour"])[["weighted_value", "weight"]].sum()
            means = (sums.weighted_value / sums.weight).rename("value").reset_index()
            for ant, a in means.groupby("ant"):
                wide = a.pivot(index="clock_hour", columns="block_index", values="value").reindex(columns=blocks).dropna()
                if wide.empty:
                    continue
                for block in blocks:
                    rows.append(dict(reference_index=reference_index, ant=ant, side=ant.split(":")[0],
                                     block_index=block, metric=metric, value=float(wide[block].mean()),
                                     shared_clock_bins=len(wide), shared_clock_hours=len(wide)*float(widths[0])/60,
                                     shared_clock_centers=json.dumps(wide.index.tolist())))
    return pd.DataFrame(rows)


def load_profiles(source, manifest, resolve):
    """Validate old labels, fingerprints and arena geometry before comparing."""
    stamps = []
    for stamp in manifest["grid_sources"]:
        stat = resolve(stamp["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (stamp["size"], stamp["mtime_ns"]):
            raise ValueError(f"Grid changed since longitudinal analysis: {stamp['path']}")
    inventory = pd.read_parquet(source / "inventory.parquet")
    rows, maps, edges, geometry = [], {}, {}, {}
    for index, window in enumerate(manifest["windows"]):
        block = window["block"]
        root = resolve(block) / "stitched/grid_occupancy_histograms_arena"
        labels_path = root / "track_cluster_ids.csv"
        labels = pd.read_csv(labels_path).rename(columns={"TrackID": "track_id"})
        if labels.duplicated(["side", "track_id"]).any():
            raise ValueError(f"Duplicate original labels: {labels_path}")
        old = inventory[inventory.source_block.eq(block)].dropna(subset=["cluster_id"])
        expected = old.set_index(["side", "track_id"]).cluster_id.sort_index()
        actual = labels.set_index(["side", "track_id"]).cluster_id.sort_index()
        pd.testing.assert_series_equal(expected, actual, check_names=False)
        stamps.append(lt.rs.fingerprint(labels_path))
        tracks = go.load_grid_tracks(root).merge(labels[["side", "track_id", "cluster_id"]],
                                                on=["side", "track_id"], how="left", validate="one_to_one")
        inv = inventory[inventory.source_block.eq(block)]
        if set(zip(tracks.side, tracks.track_id)) != set(zip(inv.side, inv.track_id)):
            raise ValueError(f"Inventory and grid identities differ: {block}")
        duration = (pd.Timestamp(window["stop"]) - pd.Timestamp(window["start"])).total_seconds()
        for _, track in tracks.iterrows():
            hist, x, y = go.load_histogram(track)
            meta = json.loads(track.metadata_path.read_text())
            side = track.side
            # Edges alone would miss a changed image-to-arena origin.
            geo = {key: meta[key] for key in ("mm_per_px", "grid_size_mm", "arena_bounds_px",
                                             "input_x_is_side_local", "input_x_origin_px", "y_origin_px",
                                             "bodypoint_filter", "x_col", "y_col", "normalization")}
            if side in edges:
                if not (np.array_equal(x, edges[side][0]) and np.array_equal(y, edges[side][1])) or geo != geometry[side]:
                    raise ValueError(f"Spatial coordinate systems differ: {block}, {side}")
            else:
                edges[side], geometry[side] = (x, y), geo
            if not np.isfinite(hist).all() or (hist < 0).any() or hist.sum() > 1.0001:
                raise ValueError(f"Invalid occupancy: {track.occupancy_path}")
            ant = f"{side}:{int(track.track_id):03d}"
            key = f"{index}|{ant}"
            maps[key] = hist
            detected = int(meta["n_detected_frames"])
            fps = manifest["sources"][index]["info"]["fps"]
            row = dict(profile_key=key, block_index=index, source_block=block,
                       block_label=Path(block).parent.name + "/" + Path(block).name,
                       start=window["start"], stop=window["stop"], side=side, ant=ant,
                       track_id=int(track.track_id), track_name=track.track_name,
                       original_cluster=track.cluster_id, n_detected_frames=detected,
                       detected_hours=detected / fps / 3600,
                       coverage=detected / (duration * fps + 1), occupancy_sum=float(hist.sum()),
                       original_selected=pd.notna(track.cluster_id))
            rows.append(row)
            for name in ("metadata_path", "occupancy_path", "x_edges_path", "y_edges_path"):
                stamps.append(lt.rs.fingerprint(track[name]))
    profiles = pd.DataFrame(rows).merge(block_metrics(pd.read_parquet(source / "task_bins.parquet")),
                                       on=["source_block", "ant"], how="left", validate="one_to_one")
    return profiles, maps, edges, geometry, stamps


def classify_reference(profiles, maps, reference_index):
    assignments, references, neighbor_rows, calibrations = [], [], [], []
    for side in ("left", "right"):
        ref = profiles[profiles.block_index.eq(reference_index) & profiles.side.eq(side) & profiles.original_selected].copy().reset_index(drop=True)
        if len(ref) <= max(K_VALUES):
            raise ValueError(f"Need at least {max(K_VALUES)+1} labeled reference ants per side")
        if ref.occupancy_sum.le(0).any():
            raise ValueError("A labeled reference ant has no in-arena occupancy")
        # EXACT original grid_occupancy feature transform, without renormalizing.
        features = go.feature_transform_matrix(np.stack([maps[key].ravel() for key in ref.profile_key]), "sqrt")
        labels = ref.original_cluster.to_numpy(str)
        distance = cdist(features, features, metric="euclidean")
        loo = {k: neighbor_vote(distance, labels, k, np.arange(len(ref))) for k in K_VALUES}
        cutoff = float(np.quantile([r["mean_distance"] for r in loo[5]], .95))
        # Only the reference features fit this display. Labels are never refitted.
        xy = go.umap_embedding(features, n_neighbors=10, min_dist=.1, metric="euclidean", random_state=0)
        ref["reference_index"] = reference_index
        ref["umap_x"], ref["umap_y"] = xy.T
        ref["loo_prediction"] = [r["prediction"] for r in loo[5]]
        ref["loo_vote"] = [r["vote"] for r in loo[5]]
        ref["loo_correct"] = ref.loo_prediction.eq(ref.original_cluster)
        ref["loo_k_agreement"] = [len({loo[k][i]["prediction"] for k in K_VALUES}) == 1 for i in range(len(ref))]
        ref["loo_mean_distance"] = [r["mean_distance"] for r in loo[5]]
        ref["distance_cutoff"] = cutoff
        references.append(ref)
        for k in K_VALUES:
            for cluster in sorted(set(labels)):
                subset = labels == cluster
                pred = np.array([r["prediction"] for r in loo[k]])
                calibrations.append(dict(reference_index=reference_index, side=side, k=k,
                                         cluster=cluster, n_reference=int(subset.sum()),
                                         loo_accuracy=float((pred[subset] == labels[subset]).mean()), distance_cutoff=cutoff))
        own = {ant: i for i, ant in enumerate(ref.ant)}
        future = profiles[profiles.block_index.gt(reference_index) & profiles.side.eq(side)].copy()
        if future.empty:
            continue
        query = go.feature_transform_matrix(np.stack([maps[key].ravel() for key in future.profile_key]), "sqrt")
        d = cdist(query, features, metric="euclidean")
        predictions = {k: neighbor_vote(d, labels, k) for k in K_VALUES}
        without_own = neighbor_vote(d, labels, 5, [own.get(ant, -1) for ant in future.ant])
        for i, (_, row) in enumerate(future.iterrows()):
            result = predictions[5][i]
            reference_row = ref.iloc[own[row.ant]] if row.ant in own else None
            valid = row.occupancy_sum > 0
            pred = result["prediction"] if valid else None
            base_label = reference_row.original_cluster if reference_row is not None else None
            switch = bool(valid and base_label is not None and pred != base_label)
            k_agreement = len({predictions[k][i]["prediction"] for k in K_VALUES}) == 1
            in_domain = valid and result["mean_distance"] <= cutoff
            baseline_reliable = reference_row is not None and reference_row.loo_correct and reference_row.loo_k_agreement and reference_row.loo_vote >= .8
            supported = bool(valid and result["vote"] >= .8 and k_agreement and in_domain
                             and without_own[i]["prediction"] == pred and baseline_reliable)
            observed = bool(reference_row is not None and min(row.coverage, reference_row.coverage) >= .4
                            and min(row.detected_hours, reference_row.detected_hours) >= 1)
            projected = result["weights"] @ xy[result["indices"]] if valid else [np.nan, np.nan]
            assignments.append(dict(**row.to_dict(), reference_index=reference_index, reference_cluster=base_label,
                                    prediction=pred, vote=result["vote"] if valid else np.nan,
                                    margin=result["margin"] if valid else np.nan,
                                    k3=predictions[3][i]["prediction"] if valid else None,
                                    k10=predictions[10][i]["prediction"] if valid else None,
                                    without_own_prediction=without_own[i]["prediction"] if valid else None,
                                    k_agreement=k_agreement, mean_distance=result["mean_distance"] if valid else np.nan,
                                    distance_ratio=result["mean_distance"] / cutoff if valid else np.nan,
                                    in_reference_range=in_domain, baseline_reliable=bool(baseline_reliable),
                                    switch=switch, supported=supported, well_observed=observed,
                                    supported_switch=switch and supported,
                                    example_switch=switch and supported and observed,
                                    umap_x=float(projected[0]), umap_y=float(projected[1]),
                                    baseline_coverage=reference_row.coverage if reference_row is not None else np.nan,
                                    baseline_colony_percent=reference_row.colony_percent if reference_row is not None else np.nan,
                                    assignment_status="no in-arena occupancy" if not valid else "assigned"))
            if valid:
                for rank, (j, dist, weight) in enumerate(zip(result["indices"], result["distances"], result["weights"]), 1):
                    neighbor_rows.append(dict(reference_index=reference_index, profile_key=row.profile_key,
                                              ant=row.ant, block_index=row.block_index, neighbor_ant=ref.iloc[j].ant,
                                              neighbor_cluster=labels[j], distance=float(dist), weight=float(weight), rank=rank))
    return (pd.DataFrame(assignments), pd.concat(references, ignore_index=True),
            pd.DataFrame(neighbor_rows), pd.DataFrame(calibrations))


def transition_summary(assignments):
    rows = []
    for keys, part in assignments.groupby(["reference_index", "side", "block_index"]):
        paired = part[part.reference_cluster.notna() & part.prediction.notna()]
        rows.append(dict(zip(["reference_index", "side", "block_index"], keys),
                         all_future_profiles=len(part), same_ant_assigned=len(paired),
                         raw_switches=int(paired.switch.sum()), supported_switches=int(paired.supported_switch.sum()),
                         well_observed_supported_switches=int(paired.example_switch.sum()),
                         supported_assignments=int(paired.supported.sum()),
                         outside_reference_range=int((~paired.in_reference_range).sum())))
    return pd.DataFrame(rows)


def run(source, output, reference_indices=(1,)):
    from analysis import long_timescale_cluster_plots as plots
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "COMPLETE.json").unlink(missing_ok=True)
    manifest = json.loads((source / "run_manifest.json").read_text())
    mappings = [] if Path("/flash/ReiterU").exists() else ["/flash/ReiterU=/home/sam-reiter/flash", "/bucket/ReiterU=/home/sam-reiter/bucket/ReiterU", "/home/s/samuel-reiter=/home/sam-reiter/saionHome"]
    profiles, maps, edges, geometry, stamps = load_profiles(source, manifest, lt.path_resolver(mappings))
    outputs = [classify_reference(profiles, maps, index) for index in reference_indices]
    tables = dict(zip(["assignments", "reference", "neighbors", "calibration"],
                      [pd.concat([out[i] for out in outputs], ignore_index=True) for i in range(4)]))
    tables["profiles"] = profiles
    tables["clock_matched_behavior"] = matched_behavior(pd.read_parquet(source / "task_bins.parquet"), profiles, reference_indices)
    tables["transitions"] = transition_summary(tables["assignments"])
    tables["examples"] = tables["assignments"].query("example_switch").sort_values(
        ["reference_index", "side", "vote", "distance_ratio"], ascending=[True, True, False, True])
    for name, table in tables.items():
        table.to_parquet(output / (name + ".parquet"), index=False)
        table.to_csv(output / (name + ".csv"), index=False)
    np.savez_compressed(output / "occupancy_maps.npz", **maps)
    np.savez_compressed(output / "grid_edges.npz", **{side + "_" + axis: values for side, pair in edges.items() for axis, values in zip("xy", pair)})
    print("CLUSTER_TABLES_COMPLETE", tables["transitions"].to_json(orient="records"), flush=True)
    plots.save_figures(tables, maps, edges, manifest, output)
    plots.write_explorer(tables, maps, edges, manifest, output)
    provenance = dict(created=datetime.now().isoformat(), source=str(source), reference_indices=list(reference_indices),
                      source_fingerprints=[lt.rs.fingerprint(source / name) for name in ("run_manifest.json", "inventory.parquet", "task_bins.parquet")],
                      grid_sources=stamps, geometry=geometry, k=5, sensitivity_k=list(K_VALUES),
                      feature_transform="sqrt of original bin_count/n_detected_frames; no further normalization",
                      metric="euclidean", vote_weights="inverse distance; exact matches only if present",
                      reference_umap=dict(n_neighbors=10, min_dist=.1, random_state=0, fit="reference only"),
                      projection="inverse-distance KNN barycenter of reference UMAP; not an out-of-sample UMAP transform",
                      supported="same labeled ant; original label agrees with LOO k3/5/10, LOO vote>=0.8; future vote>=0.8, k3/5/10 agree, leave-own-ant-out agrees, mean 5-neighbor distance<=reference LOO95th percentile",
                      example_extra="both endpoint detection coverage>=0.4 and detected time>=1h; audit tables retain all profiles",
                      code_sha256={str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (Path(__file__), Path(plots.__file__), Path(go.__file__))})
    (output / "run_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plots.write_report(tables, manifest, output)
    result = dict(profiles=len(profiles), assignments=len(tables["assignments"]),
                  figures=len(list(output.glob("*.png"))), example_rows=len(tables["examples"]),
                  report=str(output / "index.html"))
    (output / "COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
    print("CLUSTER_SWITCHING_COMPLETE", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-indices", type=int, nargs="+", default=[1],
                        help="Chronological window indices; each is a separate fixed reference")
    args = parser.parse_args()
    if min(args.reference_indices) < 0 or len(set(args.reference_indices)) != len(args.reference_indices):
        parser.error("Reference indices must be unique and nonnegative")
    import matplotlib
    matplotlib.use("Agg")
    run(args.analysis_folder, args.output, args.reference_indices)


if __name__ == "__main__":
    main()
