"""Joint occupancy clustering across recording dates in full grid features.

UMAP, time, ant identity and behavior summaries are never clustering inputs.
One Slurm task per colony and candidate K can run independently. Selection uses
separation, ant-bootstrap stability and representation across multiple ants.
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
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples


K_VALUES = tuple(range(2, 9))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def occupancy_features(maps, sigma_bins=0):
    """Hellinger features including observed probability outside arena bounds.

Missing tracking is not an occupancy category. Call only for nonempty maps.
Optional smoothing is used only for a spatial-resolution sensitivity check.
"""
    values = np.asarray(maps, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected finite nonnegative occupancy maps")
    mass = values.sum(axis=(1, 2))
    if (mass <= 0).any() or (mass > 1 + 1e-5).any():
        raise ValueError("Occupancy maps must have mass in (0, 1]")
    if sigma_bins:
        values = gaussian_filter(values, sigma=(0, sigma_bins, sigma_bins), mode="reflect")
    probabilities = np.column_stack([values.reshape(len(values), -1), np.maximum(0, 1 - mass)])
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.sqrt(probabilities).astype(np.float64)


def align_labels(reference, alternative, weights, k):
    """Hungarian alignment uses in-bag observations only during bootstrap."""
    table = np.zeros((k, k))
    np.add.at(table, (reference, alternative), weights)
    left, right = linear_sum_assignment(-table)
    mapping = np.empty(k, dtype=int)
    mapping[right] = left
    return mapping[alternative]


def choose_k(scores, stability_min=.8, silhouette_tolerance=.02):
    """Prefer the simplest well-supported partition near the best separation."""
    supported = scores.loc[(scores.bootstrap_ari_median >= stability_min) &
                           (scores.min_distinct_ants >= 3) & (scores.min_observed_hours >= 1)]
    warning = ""
    if supported.empty:
        supported = scores
        warning = "No candidate meets all stability/representation criteria; partition is exploratory."
    best = supported.weighted_silhouette.max()
    choices = supported.loc[supported.weighted_silhouette >= best - silhouette_tolerance]
    return int(choices.k.min()), warning


def load_colony(source, side):
    all_rows = pd.read_parquet(source / "profiles_4h.parquet")
    rows = all_rows.loc[all_rows.side.eq(side) & all_rows.occupancy_sum.gt(0)].copy()
    rows = rows.sort_values(["ant", "bin"]).reset_index(drop=True)
    with np.load(source / "maps_4h.npz") as maps:
        raw = np.stack([maps[key] for key in rows.profile_key])
    weights = rows.coverage.to_numpy() * rows.recorded_hours.to_numpy()
    if not (weights > 0).all():
        raise ValueError("Nonempty maps require positive observed exposure")
    return rows, raw, weights


def fit_candidate(source, output, side, k, bootstraps):
    output.mkdir(parents=True, exist_ok=True)
    signatures = {name: digest(source / name) for name in ("profiles_4h.parquet", "maps_4h.npz")}
    rows, raw, weights = load_colony(source, side)
    features = occupancy_features(raw)
    # These are the complete grid dimensions plus an outside-arena probability.
    assert features.shape[1] == np.prod(raw.shape[1:]) + 1
    model = KMeans(n_clusters=k, n_init=30, max_iter=300, random_state=0).fit(features, sample_weight=weights)
    labels = model.labels_
    samples = silhouette_samples(features, labels, metric="euclidean")
    rng = np.random.default_rng(7140)
    ants, inverse = np.unique(rows.ant.to_numpy(), return_inverse=True)
    all_counts = np.zeros((len(rows), k), dtype=np.int32)
    oob_counts = np.zeros_like(all_counts)
    oob_trials = np.zeros(len(rows), dtype=np.int32)
    ari, bootstrap_records = [], []
    for replicate in range(bootstraps):
        multiplicity = np.bincount(rng.integers(0, len(ants), size=len(ants)), minlength=len(ants))
        bag_weights = weights * multiplicity[inverse]
        replica = KMeans(n_clusters=k, n_init=5, max_iter=300, random_state=replicate + 100).fit(features, sample_weight=bag_weights)
        alternate = align_labels(labels, replica.predict(features), bag_weights, k)
        out_of_bag = multiplicity[inverse] == 0
        np.add.at(all_counts, (np.arange(len(rows)), alternate), 1)
        np.add.at(oob_counts, (np.flatnonzero(out_of_bag), alternate[out_of_bag]), 1)
        oob_trials += out_of_bag
        score = adjusted_rand_score(labels, alternate)
        ari.append(score)
        bootstrap_records.append(dict(replicate=replicate,adjusted_rand=score,
                                      out_of_bag_ants=int((multiplicity == 0).sum()),
                                      out_of_bag_ari=adjusted_rand_score(labels[out_of_bag], alternate[out_of_bag])))
        if (replicate + 1) % 8 == 0:
            print("BOOTSTRAP",side,k,replicate+1,flush=True)
    clusters = []
    for label in range(k):
        use = labels == label
        clusters.append(dict(label=label,profiles=int(use.sum()),ants=int(rows.loc[use,"ant"].nunique()),
                             observed_hours=float(weights[use].sum())))
    result = dict(side=side,k=k,profiles=len(rows),ants=len(ants),dimensions=features.shape[1],
                  inertia=float(model.inertia_),weighted_silhouette=float(np.average(samples,weights=weights)),
                  silhouette=float(samples.mean()),negative_silhouette_fraction=float((samples < 0).mean()),
                  bootstrap_ari_median=float(np.median(ari)),bootstrap_ari_p10=float(np.quantile(ari,.1)),
                  min_distinct_ants=min(c["ants"] for c in clusters),min_observed_hours=min(c["observed_hours"] for c in clusters),
                  bootstraps=bootstraps,clusters=clusters,input_sha256=signatures,script_sha256=digest(__file__))
    stem = output / f"{side}_k{k}"
    np.savez_compressed(stem.with_suffix(".npz"),keys=rows.profile_key.to_numpy(str),labels=labels,
                        centroids=model.cluster_centers_,silhouette=samples,all_counts=all_counts,
                        oob_counts=oob_counts,oob_trials=oob_trials,weights=weights)
    pd.DataFrame(bootstrap_records).to_csv(stem.with_name(stem.name+"_bootstrap.csv"),index=False)
    assert signatures == {name:digest(source/name) for name in signatures}
    stem.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
    print("JOINT_CANDIDATE_COMPLETE",json.dumps(result),flush=True)


def assemble(source, output):
    from scipy.spatial.distance import cdist
    from analysis import temporal_occupancy_joint_plots as plots
    output.mkdir(parents=True,exist_ok=True)
    source_rows = pd.read_parquet(source/"profiles_4h.parquet")
    assignments, summaries, choices, sensitivity, prototypes = [], [], [], [], {}
    signature = {name:digest(source/name) for name in ("profiles_4h.parquet","maps_4h.npz")}
    for side in ("left","right"):
        scores=[]
        for k in K_VALUES:
            metadata=json.loads((output/"candidates"/f"{side}_k{k}.json").read_text())
            assert metadata["input_sha256"]==signature
            assert metadata["script_sha256"]==digest(__file__)
            scores.append({key:value for key,value in metadata.items() if key not in ("clusters","input_sha256","script_sha256")})
        scores=pd.DataFrame(scores)
        k,warning=choose_k(scores)
        scores["selected"]=scores.k.eq(k);choices.extend(scores.to_dict("records"))
        rows,raw,weights=load_colony(source,side)
        features=occupancy_features(raw)
        with np.load(output/"candidates"/f"{side}_k{k}.npz") as saved:
            assert np.array_equal(saved["keys"],rows.profile_key.to_numpy(str))
            labels=saved["labels"].copy();centroids=saved["centroids"].copy()
            counts=saved["all_counts"].copy();oob=saved["oob_counts"].copy();trials=saved["oob_trials"].copy()
            silhouettes=saved["silhouette"].copy()
        # Stable display order only: colony occupancy sorts labels after fitting.
        order=sorted(range(k),key=lambda label:(-float(rows.loc[labels==label,"colony_percent"].median()),label))
        mapping={old:new for new,old in enumerate(order)}
        distance=cdist(features,centroids)
        nearest=np.argsort(distance,axis=1)[:,:2]
        for i,row in enumerate(rows.itertuples(index=False)):
            chosen=int(labels[i]);idx=int(mapping[chosen]);d1,d2=distance[i,nearest[i]]
            oob_agreement=float(oob[i,chosen]/trials[i]) if trials[i] else np.nan
            assignment=row._asdict()
            assignment.update(joint_cluster=f"{side}_J{idx}",joint_code=idx,old_prediction=row.prediction,
                              centroid_distance=float(d1),centroid_margin=float((d2-d1)/max(d2,1e-12)),
                              silhouette=float(silhouettes[i]),bootstrap_agreement=float(counts[i,chosen]/counts[i].sum()),
                              oob_agreement=oob_agreement,oob_trials=int(trials[i]),
                              joint_supported=bool(row.reliable and trials[i]>=5 and oob_agreement>=.8 and (d2-d1)/max(d2,1e-12)>=.1))
            assignments.append(assignment)
        for original in order:
            label=f"{side}_J{mapping[original]}";use=labels==original
            p=rows.loc[use]
            prototype=np.average(raw[use],axis=0,weights=weights[use])
            prototypes[label]=prototype
            summaries.append(dict(side=side,joint_cluster=label,profiles=len(p),ants=p.ant.nunique(),
                                  observed_hours=float(weights[use].sum()),median_colony_percent=float(p.colony_percent.median()),
                                  median_speed_mm_s=float(p.mean_speed_mm_s.median()),median_sleep_percent=float(p.sleep_percent.median()),
                                  median_trip_rate=float(p.trip_rate.median()),median_trip_minutes=float(p.mean_trip_minutes.median()),
                                  median_coverage=float(p.coverage.median()),selection_warning=warning))
        for name,x,w in [("1 mm spatial smoothing",occupancy_features(raw,1),weights),
                         ("2 mm spatial smoothing",occupancy_features(raw,2),weights),
                         ("equal profile weights",features,np.ones(len(weights)))]:
            alternate=KMeans(n_clusters=k,n_init=30,random_state=0).fit_predict(x,sample_weight=w)
            sensitivity.append(dict(side=side,comparison=name,k=k,adjusted_rand=adjusted_rand_score(labels,alternate)))
        np.savez_compressed(output/f"{side}_model.npz",centroids=centroids[order],labels=np.array([f"{side}_J{i}" for i in range(k)]),
                            grid_shape=np.array(raw.shape[1:]),feature_transform="sqrt probabilities including outside arena; no UMAP")
    classified=pd.DataFrame(assignments)
    extra=[c for c in classified if c not in source_rows]
    result=source_rows.merge(classified[["profile_key"]+extra],on="profile_key",how="left",validate="one_to_one")
    result["joint_status"]=np.where(result.joint_cluster.notna(),"assigned","no in-arena observations")
    tables=dict(assignments=result,cluster_summary=pd.DataFrame(summaries),model_selection=pd.DataFrame(choices),sensitivity=pd.DataFrame(sensitivity))
    for name,table in tables.items():
        table.to_parquet(output/f"{name}.parquet",index=False);table.to_csv(output/f"{name}.csv",index=False)
    np.savez_compressed(output/"cluster_maps.npz",**prototypes)
    plots.render(source,output,tables,prototypes)
    assert len(result)==len(source_rows) and result.profile_key.nunique()==len(result)
    assert result.joint_cluster.notna().sum()==source_rows.occupancy_sum.gt(0).sum()
    assert signature=={name:digest(source/name) for name in signature}
    provenance=dict(created=datetime.now().isoformat(),source=str(source),input_sha256=signature,
                    method="Exposure-weighted KMeans jointly across all four-hour maps, independently per colony",
                    features="Complete square-root occupancy probability vector plus outside-arena mass; no PCA, UMAP, time, identity or behavioral covariates",
                    weights="Observed tracking hours: coverage times recorded duration; every nonempty map gets positive weight",
                    selection="K=2..8; median ant-bootstrap ARI>=0.8, >=3 distinct ants and >=1 observed hour per cluster; smallest K within 0.02 of best exposure-weighted mean silhouette",
                    bootstrap="24 resamples of whole ant identities; all dates for each sampled ant stay together. Alignment uses in-bag observations. Out-of-bag agreement is stability, not a posterior probability",
                    interpretation="Joint retrospective clustering uses later data to define groups; it is distinct from prospective mapping to the July 23 baseline. Cluster labels are descriptive, not proof of discrete tasks",
                    code_sha256={Path(__file__).name:digest(__file__),Path(plots.__file__).name:digest(plots.__file__)},
                    software={name:importlib.metadata.version(name) for name in ("numpy","pandas","scipy","scikit-learn","matplotlib")},
                    profiles=len(result),assigned=int(result.joint_cluster.notna().sum()),
                    chosen_k=tables["model_selection"].loc[tables["model_selection"].selected,["side","k"]].to_dict("records"))
    (output/"run_manifest.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print("JOINT_ANALYSIS_COMPLETE",json.dumps(provenance),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temporal-folder",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-index",type=int)
    group.add_argument("--finish",action="store_true")
    p.add_argument("--bootstraps",type=int,default=24)
    a=p.parse_args()
    if a.finish:assemble(a.temporal_folder,a.output)
    else:
        side=("left","right")[a.task_index//len(K_VALUES)]
        k=K_VALUES[a.task_index%len(K_VALUES)]
        fit_candidate(a.temporal_folder,a.output/"candidates",side,k,a.bootstraps)


if __name__=="__main__":main()
