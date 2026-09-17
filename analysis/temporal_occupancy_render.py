"""Render cached temporal analysis in independent ant shards, then finalize.

This stage never refits clusters or reads tracking. Submit --shard 0..N-1 as a
Slurm array, followed by --finish with an afterok dependency on that array.
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
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis import temporal_occupancy as temporal
from analysis import temporal_occupancy_cache as cache
from analysis import temporal_occupancy_plots as plots


def load(source,output):
    tables={p.stem:pd.read_parquet(p) for p in output.glob("*.parquet")}
    manifest=json.loads((source/"run_manifest.json").read_text())
    with np.load(output/"maps_4h.npz") as data: maps={key:data[key].copy() for key in data.files}
    with np.load(output/"grid_edges.npz") as data: edges={side:(data[side+"_x"],data[side+"_y"]) for side in ("left","right")}
    return tables,maps,edges,manifest


def render_shard(source,output,shard,shards):
    tables,maps,edges,manifest=load(source,output)
    ants=list(tables["reference"].ant)
    selected=ants[shard::shards]
    film=set(tables["whole_assignments"].loc[tables["whole_assignments"].switch,"ant"])
    film.update(tables["profiles_4h"].loc[tables["profiles_4h"].supported_switch,"ant"])
    for ant in selected:
        plots.ant_detail(ant,tables,manifest,output)
        if ant in film:plots.filmstrip(ant,tables,maps,edges,output)
        print("ANT_RENDERED",ant,flush=True)
    (output/f"render_shard_{shard}.json").write_text(json.dumps(dict(shard=shard,shards=shards,ants=selected,complete=True),indent=2)+"\n")


def finish(source,output,shards):
    from PIL import Image
    import joblib
    tables,maps,edges,manifest=load(source,output)
    ants=set(tables["reference"].ant);rendered=[]
    for shard in range(shards):
        result=json.loads((output/f"render_shard_{shard}.json").read_text())
        assert result["complete"] and result["shards"]==shards
        rendered+=result["ants"]
    assert len(rendered)==len(ants) and set(rendered)==ants
    for ant in ants:
        with Image.open(output/"individual_ants"/(ant.replace(":","_")+".png")) as im:im.verify()
    for side,ref in tables["reference"].groupby("side"):
        model=joblib.load(output/f"reference_umap_{side}.joblib")
        np.testing.assert_allclose(model.embedding_,ref[["umap_x","umap_y"]],rtol=0,atol=1e-5)
    four,one=tables["profiles_4h"],tables["profiles_1h"]
    assert len(four)==2964 and len(one)==10602
    assert not four.start.dt.strftime("%Y-%m-%d").eq("2026-07-28").any()
    windows=tables["windows_4h"]
    handoff=windows.loc[windows.start.eq(pd.Timestamp("2026-07-24 08:00:00"))].iloc[0]
    assert handoff.recording_fraction>.99 and json.loads(handoff.source_indices)==[1,2]
    # This check specifically protects the behavior join against us/ns mismatch.
    assert one.colony_percent.notna().sum()==9633
    assert four.coverage.le(1.000001).all() and one.coverage.le(1.000001).all()
    assert four.loc[four.occupancy_sum.le(0),"prediction"].isna().all()
    atom_metadata=list((output/"atoms").glob("*/*.json"));assert len(atom_metadata)==456
    source_count=0
    for path in atom_metadata:
        m=json.loads(path.read_text());assert m["original_histogram_exact"]
        for stamp in m["signature"]["inputs"]:
            assert cache.stamp(stamp["path"])==stamp,stamp["path"]
            source_count+=1
    prior=source/"cluster_switching"
    with np.load(prior/"occupancy_maps.npz") as saved:
        maps.update({"whole|"+key:saved[key].copy() for key in saved.files})
    plots.save_overview_figures(tables,maps,edges,manifest,output)
    plots.write_explorer(tables,maps,edges,manifest,output)
    plots.write_report(tables,manifest,output)
    for p in output.glob("*.png"):
        with Image.open(p) as im:im.verify()
    for p in (output/"filmstrips").glob("*.png"):
        with Image.open(p) as im:im.verify()
    code=[Path(temporal.__file__),Path(cache.__file__),Path(plots.__file__),Path(__file__),Path(plots.__file__).with_name("temporal_occupancy_dashboard.html"),Path(temporal.go.__file__),Path(temporal.cs.__file__)]
    provenance=dict(created=datetime.now().isoformat(),source=str(source),reference_window=manifest["windows"][1],
                    input_sources=[cache.stamp(source/name) for name in ("run_manifest.json","task_bins.parquet")],
                    reference_sources=[cache.stamp(prior/name) for name in ("reference.parquet","occupancy_maps.npz","assignments.parquet")],
                    atom_sources=[cache.stamp(p) for p in sorted((output/"atoms").glob("*/*.npz"))],
                    reference=dict(n_neighbors=10,min_dist=.1,metric="euclidean",random_state=0,fit="original labeled 0723/block02 maps only",coordinates_reproduced=True),
                    projection="UMAP.transform with fixed fitted reference; no later samples in fit",
                    classifier=dict(k=5,weights="inverse distance",sensitivity_k=[3,5,10],feature_transform="sqrt of original detected-frame-normalized histogram"),
                    independent_clustering=dict(algorithm="Leiden",n_neighbors=10,resolution=1.,seed=0,cohort="all nonzero observed maps; local labels never treated as temporal IDs"),
                    window_policy="calendar-aligned four-hour and one-hour windows; exact cached half-hour counts; partial windows and gaps preserved; near-continuous source contributions pooled",
                    novelty="whole-block threshold plus duration-matched reference-window leave-own-ant-out 95th percentile; alternative >=40% detection calibration retained",
                    model_comparison="weighted circadian-only, gradual drift and single step; minimum12 hours, >=40% detection and >=95% recorded; searched breakpoint counts as a parameter; descriptive BIC with unmodeled autocorrelation",
                    code_sha256={str(p.relative_to(Path(__file__).parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
                    software={name:importlib.metadata.version(name) for name in ("numpy","pandas","scipy","umap-learn","scikit-learn","leidenalg","plotly")})
    (output/"run_manifest.json").write_text(json.dumps(provenance,indent=2)+"\n")
    (output/"source_run_manifest.json").write_bytes((source/"run_manifest.json").read_bytes())
    check=dict(passed=True,source_fingerprints_verified=source_count,atom_maps_verified=456,profiles_4h=len(four),profiles_1h=len(one),
               exact_reference_coordinates=True,behavior_timestamp_units_verified=True,recording_handoff_preserved=True,real_gaps_preserved=True,
               all_reference_ant_figures_verified=len(ants),unit_tests=5)
    (output/"data_check.json").write_text(json.dumps(check,indent=2)+"\n")
    result=dict(profiles_4h=len(four),profiles_1h=len(one),windows_4h=len(tables["windows_4h"]),
                overview_figures=len(list(output.glob("*.png"))),individual_figures=len(ants),filmstrips=len(list((output/"filmstrips").glob("*.png"))))
    (output/"COMPLETE.json").write_text(json.dumps(result,indent=2)+"\n")
    print("TEMPORAL_RENDERING_COMPLETE",json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-folder",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--shard",type=int)
    group.add_argument("--finish",action="store_true")
    p.add_argument("--shards",type=int,default=8)
    args=p.parse_args()
    if args.shards<1 or args.shard is not None and not 0<=args.shard<args.shards:p.error("Invalid shard")
    if args.finish:finish(args.analysis_folder,args.output,args.shards)
    else:render_shard(args.analysis_folder,args.output,args.shard,args.shards)


if __name__=="__main__":main()
